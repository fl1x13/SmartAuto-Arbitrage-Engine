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

from config import BASE_DIR
from model.predict import enrich_with_predictions, load_model
from processing.preprocessor import DataPreprocessor
from scraper.storage import get_engine, get_price_dynamics

TOP_N = 20
LABELS_PATH = BASE_DIR / "labels.csv"
GOOD_LABELS = {"good"}
# Non-good but still "checked, not a deal" — counts against precision.
# norm/market = market-priced (the 🔥 grade oversold it); trash = worthless or
# fake discount; scam = fraud/hidden problem.
NONGOOD_LABELS = {"trash", "scam", "norm", "market"}
RECOGNIZED_LABELS = GOOD_LABELS | NONGOOD_LABELS
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
    engine = get_engine()  # ensures the sold column exists (runs migrations)
    df = pd.read_sql("SELECT * FROM raw_ads WHERE COALESCE(sold, 0) = 0", engine)
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

    # sep=None lets the csv sniffer accept both comma and the semicolon that
    # Excel writes under a Russian locale; utf-8-sig strips the BOM Excel
    # prepends (which would otherwise rename the first column to "﻿ad_id").
    labels = pd.read_csv(
        LABELS_PATH,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        encoding_errors="replace",  # Excel can corrupt the unread discount cells
        dtype={"ad_id": "Int64"},
    )
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    unknown = sorted(
        set(labels.loc[labels["label"] != "", "label"]) - RECOGNIZED_LABELS
        - {"nan"}
    )
    labels = labels[labels["label"].isin(RECOGNIZED_LABELS)]

    labelled = top.merge(labels[["ad_id", "label"]], on="ad_id", how="inner")
    n_labelled = len(labelled)
    if n_labelled == 0:
        print(
            f"\n⚠️  None of the top {len(top)} ad_ids are labelled in "
            f"{LABELS_PATH.name} yet — nothing to score."
        )
        return

    n_good = int(labelled["label"].isin(GOOD_LABELS).sum())
    precision = n_good / n_labelled

    print("\n=== precision@%d ===" % len(top))
    print(f"labelled:  {n_labelled} / {len(top)}")
    for label, n in labelled["label"].value_counts().items():
        print(f"  {label:<8} {n}")
    if unknown:
        print(f"(ignored unrecognised labels: {', '.join(unknown)})")
    status = "✅ on target" if precision >= TARGET_PRECISION else "❌ below target"
    print(
        f"precision@{len(top)} = good / labelled = "
        f"{n_good}/{n_labelled} = {precision:.0%}  "
        f"(target {TARGET_PRECISION:.0%}) — {status}"
    )


# Context columns written into the label template so each row can be judged
# (and its ad opened) straight from the spreadsheet. Only ad_id and the empty
# label column are read back by report_precision; the rest is for the human.
TEMPLATE_COLS = [
    "ad_id",
    "brand",
    "model",
    "year",
    "mileage",
    "price",
    "discount_pct",
    "url",
]


def _existing_labels() -> pd.DataFrame:
    """Read prior labels.csv as ad_id+label, tolerant of Excel's quirks.

    Returns an empty frame when there is nothing usable to carry over.
    """
    if not LABELS_PATH.exists():
        return pd.DataFrame(columns=["ad_id", "label"])
    prev = pd.read_csv(
        LABELS_PATH,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        encoding_errors="replace",  # Excel can corrupt the unread discount cells
        dtype={"ad_id": "Int64"},
    )
    if "ad_id" not in prev.columns or "label" not in prev.columns:
        return pd.DataFrame(columns=["ad_id", "label"])
    prev = prev.dropna(subset=["ad_id"])
    prev["ad_id"] = prev["ad_id"].astype("int64")
    return prev[["ad_id", "label"]]


def write_label_template(top: pd.DataFrame) -> None:
    """Seed labels.csv with the current top-N deals, preserving prior labels.

    Each row carries brand/model/year/price and the auto.ru link, so the deals
    can be judged and opened directly from the spreadsheet — no digging through
    the bot. The ranking shifts as the market refreshes and as the scoring
    evolves, so the set is re-seeded on each run; labels already given for an
    ad_id that is still in the top are carried over, so labelling effort
    accumulates rather than resetting — only the new entrants need judging.
    """
    cols = [c for c in TEMPLATE_COLS if c in top.columns]
    template = top[cols].copy()
    prior = _existing_labels()
    template = template.merge(prior, on="ad_id", how="left")
    template["label"] = template["label"].fillna("")  # fill with good/trash/scam
    template.to_csv(LABELS_PATH, index=False)

    carried = int((template["label"].astype(str).str.strip() != "").sum())
    print(
        f"Wrote {len(template)} deals to {LABELS_PATH.name} "
        f"(columns: {', '.join(cols)}, label).\n"
        f"Carried over {carried} existing labels; "
        f"{len(template) - carried} new rows need a label.\n"
        "Open it, judge the blank rows via their url with good / trash / "
        "scam / norm, then re-run without --init-labels."
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
