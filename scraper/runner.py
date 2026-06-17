"""Scraper entry point: `python -m scraper.runner`."""

import argparse
import asyncio
import logging

from config import cfg
from scraper.parser import run_scraper
from scraper.storage import save_ads

logger = logging.getLogger(__name__)


async def run_full_pipeline(max_pages: int | None = None) -> int:
    """Scrape listings and persist them to the database.

    Args:
        max_pages: Page count to scrape; defaults to cfg.scraper_max_pages.

    Returns:
        Number of newly inserted records.
    """
    ads = await run_scraper(max_pages)
    inserted = save_ads(ads)
    logger.info("Pipeline finished: %d new records persisted", inserted)
    return inserted


def main() -> None:
    """CLI wrapper around the scraping pipeline."""
    parser = argparse.ArgumentParser(description="Scrape car listings into the DB")
    parser.add_argument(
        "--pages",
        type=int,
        default=cfg.scraper_max_pages,
        help=f"Number of listing pages to scrape (default: {cfg.scraper_max_pages})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run_full_pipeline(args.pages))


if __name__ == "__main__":
    main()
