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
from scraper.autoru_ad import inspect_listings
from scraper.storage import (
    get_engine,
    get_price_dynamics,
    mark_ads_sold,
    update_autoru_badges,
)

logger = logging.getLogger(__name__)


def prune_sold(top_n: int | None = None, engine=None) -> int:
    """Verify the top-scoring live ads: flag the sold, re-rate the rest.

    Fetches each top candidate's page once and reads both its liveness and
    auto.ru's own price rating from it — so the same fetch that drops sold
    listings also refreshes the independent valuation that vetoes our model's
    fake discounts, at no extra request cost.

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
    states = inspect_listings(pairs)

    sold = {ad_id for ad_id, st in states.items() if st.sold}
    badges = {
        ad_id: (st.badge, st.badge_pct)
        for ad_id, st in states.items()
        if not st.sold and st.badge is not None
    }
    flagged = mark_ads_sold(sold, engine)
    rerated = update_autoru_badges(badges, engine)
    logger.info(
        "Liveness check: %d of top %d sold (%d flagged); %d re-rated by auto.ru",
        len(sold),
        len(pairs),
        flagged,
        rerated,
    )
    return flagged


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flagged = prune_sold()
    print(f"Flagged {flagged} sold listings out of the top deals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
