"""Flag sold/removed listings so they stop surfacing as deals.

Sold cars keep their auto.ru page (for price history) but drop the price and
show a "продан" banner. The DB has no reliable "last seen" signal — an ad
re-seen at a steady price keeps a stale timestamp — so staleness alone cannot
tell sold from simply-old. Instead this verifies the highest-scoring ads by
fetching their pages and marks the sold ones, which the bot, dashboard, Mini
App and the precision table all then exclude (``WHERE sold = 0``).

Bounded by ``cfg.liveness_check_top_n``: only ads that could be recommended are
checked, so the network cost stays small. Run manually or from the scheduler:

    python -m scripts.prune_sold
"""

import logging
import sys

import pandas as pd

from config import cfg
from model.predict import enrich_with_predictions, load_model
from processing.preprocessor import DataPreprocessor
from scraper.autoru_ad import find_sold_ad_ids
from scraper.storage import get_engine, get_price_dynamics, mark_ads_sold

logger = logging.getLogger(__name__)


def prune_sold(top_n: int | None = None, engine=None) -> int:
    """Verify the top-scoring live ads and flag the sold ones.

    Args:
        top_n: How many highest-scoring ads to check; cfg.liveness_check_top_n
            when None.
        engine: Optional SQLAlchemy engine; creates one from config if not given.

    Returns:
        Number of listings newly flagged sold.
    """
    if engine is None:
        engine = get_engine()  # ensures the sold column exists (runs migrations)
    top_n = top_n or cfg.liveness_check_top_n

    df = pd.read_sql("SELECT * FROM raw_ads WHERE COALESCE(sold, 0) = 0", engine)
    if df.empty:
        return 0
    df = DataPreprocessor().fit_transform(df)
    df = enrich_with_predictions(
        df, load_model(), price_dynamics=get_price_dynamics(engine)
    )

    candidates = df.sort_values("score", ascending=False).head(top_n)
    pairs = list(zip(candidates["ad_id"].astype(int), candidates["url"]))
    sold = find_sold_ad_ids(pairs)
    flagged = mark_ads_sold(sold, engine)
    logger.info(
        "Liveness check: %d of top %d verified sold, %d newly flagged",
        len(sold),
        len(pairs),
        flagged,
    )
    return flagged


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flagged = prune_sold()
    print(f"Flagged {flagged} sold listings out of the top deals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
