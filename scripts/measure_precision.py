"""Measure precision@20 for the top-deal ranking.

Loads the live listings, runs them through the exact same pipeline the bot
and dashboard use (DataPreprocessor -> enrich_with_predictions), takes the
top 20 by arbitrage score (excluding low-confidence price estimates), and —
if a hand-labelled ``labels.csv`` exists in the project root — reports how
many of those top deals were genuinely good.

precision@20 = good / labelled  (target: 75%+)

Usage:
    python -m scripts.measure_precision
"""

import sys

import pandas as pd
from sqlalchemy import create_engine

from config import BASE_DIR, cfg
from model.predict import enrich_with_predictions, load_model
from processing.preprocessor import DataPreprocessor
from scraper.storage import get_price_dynamics

TOP_N = 20
LABELS_PATH = BASE_DIR / "labels.csv"
GOOD_LABELS = {"good"}
BAD_LABELS = {"trash", "scam"}
TARGET_PRECISION = 0.75

# Columns shown in the top-N table (in display order).
DISPLAY_COLS = [
    "ad_id",
    "brand",
    "model",
    "year",
    "mileage",
    "price",
    "discount_pct",
    "deal_grade",
]


def load_top_deals(n: int = TOP_N) -> pd.DataFrame:
    """Return the top-``n`` listings by score, excluding low-confidence ones.

    Mirrors ``app.data_loader.load_enriched_data`` so the ranking measured
    here is the same one the bot and dashboard surface to users.
    """
    engine = create_engine(cfg.db_url)
    df = pd.read_sql("SELECT * FROM raw_ads", engine)
    df = DataPreprocessor().fit_transform(df)
    df = enrich_with_predictions(
        df, load_model(), price_dynamics=get_price_dynamics(engine)
    )
    df = df[df["confidence"] != "low"]
    return df.sort_values("score", ascending=False).head(n).reset_index(drop=True)


def report_precision(top: pd.DataFrame) -> None:
    """Join the top deals against labels.csv and print precision@20."""
    if not LABELS_PATH.exists():
        print(
            f"\n⚠️  {LABELS_PATH.name} not found in project root — skipping "
            "precision@20.\n"
            "    Label the ad_ids below as good/trash/scam in labels.csv "
            "and re-run."
        )
        return

    labels = pd.read_csv(LABELS_PATH, dtype={"ad_id": "Int64"})
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    labels = labels[labels["label"].isin(GOOD_LABELS | BAD_LABELS)]

    labelled = top.merge(labels[["ad_id", "label"]], on="ad_id", how="inner")
    n_labelled = len(labelled)
    if n_labelled == 0:
        print(
            f"\n⚠️  None of the top {len(top)} ad_ids are labelled in "
            f"{LABELS_PATH.name} yet — nothing to score."
        )
        return

    n_good = int(labelled["label"].isin(GOOD_LABELS).sum())
    n_trash = int((labelled["label"] == "trash").sum())
    n_scam = int((labelled["label"] == "scam").sum())
    precision = n_good / n_labelled

    print("\n=== precision@%d ===" % len(top))
    print(f"labelled:  {n_labelled} / {len(top)}")
    print(f"good:      {n_good}")
    print(f"trash:     {n_trash}")
    print(f"scam:      {n_scam}")
    status = "✅ on target" if precision >= TARGET_PRECISION else "❌ below target"
    print(
        f"precision: {precision:.0%}  "
        f"(target {TARGET_PRECISION:.0%}) — {status}"
    )


def write_label_template(top: pd.DataFrame) -> None:
    """Seed labels.csv with the current top-N ad_ids and empty labels.

    The ranking shifts as the market refreshes and as the scoring evolves, so
    the label set has to be re-seeded; existing labels for ad_ids that resurface
    still match on join, so labelling effort accumulates rather than resets.
    """
    template = pd.DataFrame({"ad_id": top["ad_id"], "label": ""})
    template.to_csv(LABELS_PATH, index=False)
    print(
        f"Wrote {len(template)} ad_ids to {LABELS_PATH.name}. "
        "Fill the label column with good / trash / scam and re-run."
    )


def main() -> int:
    top = load_top_deals()
    if top.empty:
        print("No medium/high-confidence listings found — is the DB populated?")
        return 1

    if "--init-labels" in sys.argv[1:]:
        write_label_template(top)
        return 0

    report_precision(top)

    print(f"\n=== Top {len(top)} deals by score ===")
    table = top[DISPLAY_COLS].copy()
    table["price"] = table["price"].map("{:,.0f}".format)
    table["mileage"] = table["mileage"].map("{:,.0f}".format)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
