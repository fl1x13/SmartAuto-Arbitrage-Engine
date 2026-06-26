"""Periodic scraping via APScheduler: `python -m scraper.scheduler`.

Every ``cfg.scrape_interval_minutes`` the freshest listing pages are
scraped (sorted newest-first on auto.ru), so newly posted cars land in
the DB within one interval. Once ``cfg.retrain_min_new_ads`` new ads
accumulate, the model is retrained in a worker thread so fair prices
reflect the current market.
"""

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import cfg

logger = logging.getLogger(__name__)

_new_ads_since_retrain = 0


async def scheduled_scrape_job() -> None:
    """Scheduler task: scrape fresh pages, retrain model when enough is new."""
    global _new_ads_since_retrain
    from scraper.runner import run_full_pipeline

    logger.info("Scheduled scrape started")
    inserted = await run_full_pipeline()
    _new_ads_since_retrain += inserted
    logger.info(
        "Scheduled scrape finished: %d new records (%d since last retrain)",
        inserted,
        _new_ads_since_retrain,
    )

    # Flag sold/removed listings so they drop off the deal surfaces. Runs off
    # the event loop (it makes blocking page fetches) and never blocks the
    # scrape cycle on failure.
    try:
        from scripts.prune_sold import prune_sold

        flagged = await asyncio.to_thread(prune_sold)
        logger.info("Sold-listing prune: %d newly flagged", flagged)
    except Exception as e:  # noqa: BLE001 — a prune failure must not stop scraping
        logger.error("Sold-listing prune failed: %s: %s", type(e).__name__, e)

    if _new_ads_since_retrain >= cfg.retrain_min_new_ads:
        await _retrain_model()
        _new_ads_since_retrain = 0


async def _retrain_model() -> None:
    """Retrain the CatBoost model off the event loop."""
    from model.train import run_training_pipeline

    logger.info("Retraining model (threshold of %d new ads reached)",
                cfg.retrain_min_new_ads)
    metrics = await asyncio.to_thread(run_training_pipeline)
    logger.info("Model retrained: MAPE=%.2f%%", metrics["mape"])


def start_scheduler(interval_minutes: float | None = None) -> AsyncIOScheduler:
    """Create and start the scraping scheduler.

    Args:
        interval_minutes: Scrape interval; defaults to
            cfg.scrape_interval_minutes.

    Returns:
        The started AsyncIOScheduler instance.
    """
    interval_minutes = interval_minutes or cfg.scrape_interval_minutes
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_scrape_job, "interval", minutes=interval_minutes)
    scheduler.start()
    logger.info("Scheduler started: scraping every %s minutes", interval_minutes)
    return scheduler


async def _main() -> None:
    start_scheduler()
    # Run one scrape immediately so the DB is fresh on startup
    await scheduled_scrape_job()
    await asyncio.Event().wait()  # keep the loop alive


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
