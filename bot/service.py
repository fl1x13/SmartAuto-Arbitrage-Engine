"""Data access layer for the Telegram bot.

Reuses the dashboard pipeline (preprocess → predict → score) over the same
DB, with a small TTL cache so chat commands don't re-run inference on every
tap. All functions return plain pandas rows ready for bot.formatting.
"""

import logging
import time
from datetime import datetime

import pandas as pd

from bot import feedback
from config import cfg
from model.predict import (
    enrich_with_predictions,
    evaluate_listing,
    explain_prediction,
    load_model,
)
from processing.preprocessor import DataPreprocessor
from scraper.autoru import AD_URL_RE
from scraper.storage import get_engine, get_price_dynamics

logger = logging.getLogger(__name__)

# Human-readable names for the price-factor breakdown in the deal report.
_RU_FEATURE = {
    "brand": "Марка",
    "model": "Модель",
    "generation": "Поколение",
    "modification": "Модификация",
    "body_type": "Тип кузова",
    "transmission": "Коробка передач",
    "region": "Регион",
    "drive": "Привод",
    "fuel_type": "Тип топлива",
    "brand_segment": "Сегмент марки",
    "year": "Год выпуска",
    "car_age": "Возраст",
    "mileage": "Пробег",
    "mileage_per_year": "Пробег в год",
    "engine_volume": "Объём двигателя",
    "horse_power": "Мощность",
    "power_density": "Л.с. на литр",
    "owners_count": "Число владельцев",
    "owners_per_year": "Владельцев в год",
}

_cache: dict = {"df": None, "loaded_at": 0.0}


def load_market(force: bool = False) -> pd.DataFrame:
    """Load the enriched market frame (cached for cfg.bot_cache_ttl_seconds).

    Args:
        force: Bypass the cache (used by the deal watcher after scrapes).

    Returns:
        Enriched DataFrame with prediction and deal-analysis columns.
    """
    age = time.monotonic() - _cache["loaded_at"]
    if not force and _cache["df"] is not None and age < cfg.bot_cache_ttl_seconds:
        return _cache["df"]

    engine = get_engine()  # ensures the sold column exists (runs migrations)
    df = pd.read_sql("SELECT * FROM raw_ads WHERE COALESCE(sold, 0) = 0", engine)
    df = DataPreprocessor().fit_transform(df)
    df = enrich_with_predictions(
        df,
        load_model(),
        price_dynamics=get_price_dynamics(engine),
        corrections=feedback.prediction_corrections(),
    )
    _cache["df"] = df
    _cache["loaded_at"] = time.monotonic()
    logger.info("Bot market cache refreshed: %d rows", len(df))
    return df


# Smart category filters for the deals tab — meaningful groups, not raw brands.
_GERMAN = {
    "bmw", "mercedes", "audi", "volkswagen", "porsche", "opel", "mini",
    "smart", "maybach", "alpina",
}
_JAPANESE = {
    "toyota", "honda", "nissan", "mazda", "mitsubishi", "subaru", "lexus",
    "infiniti", "suzuki", "daihatsu", "datsun", "acura",
}
_SPORTY_BODY = {"купе", "кабриолет", "родстер", "тарга", "купе-хардтоп"}


def _category_mask(df: pd.DataFrame, category: str) -> "pd.Series":
    """Boolean mask for a smart category (German/Chinese/electric/sporty…)."""
    if category == "german":
        return df["brand"].isin(_GERMAN)
    if category == "japanese":
        return df["brand"].isin(_JAPANESE)
    if category == "chinese":
        return df["brand_segment"] == "china"
    if category == "premium":
        return df["brand_segment"].isin(["premium", "luxury"])
    if category == "suv":
        return df["body_type"].str.startswith("внедорожник")
    if category == "sporty":
        return df["body_type"].isin(_SPORTY_BODY)
    if category == "electric":
        return df["fuel_type"] == "электро"
    return pd.Series(True, index=df.index)


def top_deals(
    n: int | None = None,
    max_price: int | None = None,
    brand: str | None = None,
    model: str | None = None,
    min_price: int | None = None,
    offset: int = 0,
    category: str | None = None,
) -> pd.DataFrame:
    """Best non-suspicious deals, sorted by score.

    Args:
        n: Number of deals; cfg.bot_top_n when None.
        max_price: Optional price ceiling in rubles.
        brand: Optional brand filter (lowercase).
        model: Optional model filter (lowercase).
        min_price: Optional price floor in rubles (strict budget band).
        offset: Skip this many top deals — lets the «Другие объявления» button
            page through the ranking instead of always showing the same cars.

    Returns:
        A page of the score-ranked frame.
    """
    df = load_market()
    # Exclude low-confidence rows: a "fair price" from <6 comparable listings
    # is unreliable and tends to invent 50%+ discounts on rare/old cars, which
    # would otherwise dominate the ranking (see the Hyundai Matrix case).
    mask = ~df["is_suspicious"] & (df["confidence"] != "low")
    if min_price:
        mask &= df["price"] >= min_price
    if max_price:
        mask &= df["price"] <= max_price
    if brand:
        mask &= df["brand"] == brand
    if model:
        mask &= df["model"] == model
    if category and category != "all":
        mask &= _category_mask(df, category)
    ranked = df[mask].sort_values("score", ascending=False)
    n = n or cfg.bot_top_n
    return ranked.iloc[offset : offset + n]


def pick_cars(
    budget_from: int,
    budget_to: int,
    brand: str | None,
    model: str | None = None,
    n: int | None = None,
    offset: int = 0,
) -> pd.DataFrame:
    """Car-picker results: best deals within a budget, paged.

    Args:
        budget_from: Lower price bound, rubles.
        budget_to: Upper price bound, rubles.
        brand: Brand (lowercase) or None for any.
        model: Model (lowercase) or None for any.
        n: Page size; cfg.bot_top_n when None.
        offset: Skip this many results — lets the Mini App "show more" button
            load further pages instead of always the same top cars.

    Returns:
        A page of the score-ranked frame.
    """
    df = load_market()
    mask = (
        df["price"].between(budget_from, budget_to)
        & ~df["is_suspicious"]
        & (df["confidence"] != "low")
    )
    if brand:
        mask &= df["brand"] == brand
    if model:
        mask &= df["model"] == model
    ranked = df[mask].sort_values("score", ascending=False)
    n = n or cfg.bot_top_n
    return ranked.iloc[offset : offset + n]


def popular_brands(limit: int = 8) -> list[str]:
    """Most-listed brands (lowercase), for keyboard buttons."""
    return load_market()["brand"].value_counts().head(limit).index.tolist()


def all_brands() -> list[str]:
    """Every brand in the market (lowercase), alphabetical — for the picker.

    Returns:
        All distinct brand slugs so the Mini App can search the full catalog,
        not just the most-listed few.
    """
    return sorted(load_market()["brand"].dropna().unique().tolist())


def models_for_brand(brand: str, limit: int | None = None) -> list[str]:
    """Models (lowercase) of a brand for the picker.

    Args:
        brand: Brand slug (lowercase).
        limit: Max models, ordered by listing count; when None, returns every
            model of the brand, alphabetical (so the picker covers the catalog).

    Returns:
        Model names — popularity-ranked when limited, else alphabetical.
    """
    models = load_market()
    series = models[models["brand"] == brand]["model"]
    if limit is None:
        return sorted(series.dropna().unique().tolist())
    return series.value_counts().head(limit).index.tolist()


def evaluate_url(url: str) -> pd.Series | None:
    """Evaluate one auto.ru listing by URL (DB row or live fetch).

    Args:
        url: Ad URL pasted into the chat.

    Returns:
        Evaluated row (enriched fields present), or None when the ad can't
        be fetched/parsed.
    """
    url_match = AD_URL_RE.search(url)
    if url_match is None:
        return None
    df = load_market()
    in_db = df[df["ad_id"] == int(url_match["ad_id"])]
    if not in_db.empty:
        return in_db.iloc[0]

    from scraper.autoru_ad import fetch_ad_by_url

    ad = fetch_ad_by_url(url)
    if ad is None:
        return None
    row_df = DataPreprocessor().engineer_features(pd.DataFrame([ad.model_dump()]))
    return evaluate_listing(row_df, df, load_model())


def new_hot_deals(
    since: datetime, max_price: int | None = None, brand: str | None = None
) -> pd.DataFrame:
    """Fresh 🔥 deals scraped after `since` (for subscription pushes).

    Args:
        since: Only ads first seen after this UTC timestamp.
        max_price: Optional subscriber price ceiling.
        brand: Optional subscriber brand filter (lowercase).

    Returns:
        Matching rows sorted by score (may be empty).
    """
    # The watcher refreshes the cache once per tick (load_market(force=True))
    # before fanning out per subscriber, so the cached frame is fresh here.
    df = load_market()
    mask = (df["deal_grade"] == "🔥 Горячая") & ~df["is_suspicious"]
    mask &= pd.to_datetime(df["scraped_at"]) > since
    if max_price:
        mask &= df["price"] <= max_price
    if brand:
        mask &= df["brand"] == brand
    return df[mask].sort_values("score", ascending=False)


def _title(row: pd.Series) -> str:
    """Card title: full modification when known, else 'Brand Model'."""
    if row.get("modification"):
        return str(row["modification"])
    return f"{str(row['brand']).title()} {str(row['model']).title()}"


def _price_factors(
    row: pd.Series, model, limit: int = 6
) -> tuple[list[dict], int, "pd.Series"]:
    """Top SHAP price factors as signed percentage effects on the fair price.

    Returns:
        (factors, base_price, top) — factors is a list of {"label", "pct"}
        ordered by impact (the dashboard's technical view); base_price is the
        model's average market price before per-feature adjustments; top is the
        trimmed contributions Series (feature → %), used to build the
        consumer-friendly explanation.
    """
    contributions, base_value, _ = explain_prediction(pd.DataFrame([row]), model)
    contributions = contributions[contributions.abs() >= 1.0]
    top = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    ).head(limit)
    factors = [
        {"label": _RU_FEATURE.get(f, f), "pct": round(float(v), 1)}
        for f, v in top.items()
    ]
    return factors, int(base_value), top


def _consumer_factor(feature: str, pct: float, row: pd.Series) -> str | None:
    """One trait written like an appraiser's note — natural, with the number.

    The report groups these into strengths (``pct``≥0) and caveats (``pct``<0),
    and the section header sets the per-model frame, so each line can stay short
    and human (no repetitive "чем у таких же"). ``pct`` is the SHAP price effect.
    Returns None for non-distinguishing features (brand/model/segment).
    """
    up = pct >= 0
    cap = lambda s: str(s).capitalize()  # noqa: E731

    if feature == "mileage":
        km = int(row["mileage"]) // 1000
        return f"{'Небольшой' if up else 'Большой'} пробег — {km} тыс. км"
    if feature == "horse_power":
        hp = int(row["horse_power"])
        return f"{'Мощный' if up else 'Скромный'} мотор — {hp} л.с."
    if feature in ("year", "car_age"):
        yr = int(row["year"])
        return f"Свежий год — {yr}" if up else f"Возраст — {yr} год"
    if feature == "engine_volume" and row.get("engine_volume"):
        size = "Крупный" if up else "Небольшой"
        return f"{size} двигатель — {row['engine_volume']} л"
    if feature == "owners_count":
        n = int(row["owners_count"])
        if up:
            return "Один владелец" if n == 1 else f"Мало владельцев — {n}"
        return f"Несколько владельцев — {n}"
    if feature == "drive" and row.get("drive"):
        suffix = "ценная версия" if up else "попроще"
        return f"{cap(row['drive'])} привод — {suffix}"
    if feature == "body_type" and row.get("body_type"):
        return f"{'Востребованный' if up else 'Простой'} кузов — {row['body_type']}"
    if feature == "fuel_type" and row.get("fuel_type"):
        return (
            f"{cap(row['fuel_type'])} — в плюс к цене" if up
            else f"Топливо — {row['fuel_type']}"
        )
    if feature in ("generation", "modification"):
        return "Удачные поколение и комплектация" if up \
            else "Поколение и комплектация попроще"
    if feature == "region" and row.get("region"):
        return (
            f"Регион {row['region']} — здесь такие дороже" if up
            else f"Регион {row['region']} — здесь дешевле"
        )
    return None


def _verdict(row: pd.Series) -> str:
    """The arbitrage thesis, grounded in the real market (the comparables)."""
    disc = float(row.get("discount_pct", 0.0))
    if disc >= 20 and row.get("confidence") == "high":
        return (
            "Главное: такие же машины на рынке стоят заметно дороже (они ниже), "
            "а эту отдают ниже рыночной цены. Это и есть недооценка — хорошая "
            "цель для перепродажи."
        )
    if disc >= 10:
        return (
            "Похожие в продаже стоят дороже, а тут цена ниже рыночной — "
            "машину недооценили, стоит брать."
        )
    if disc > 0:
        return "Чуть дешевле, чем стоят такие же на рынке."
    if disc > -15:
        return "Просят немного дороже, чем стоят такие же на рынке."
    return (
        "Такие же машины на рынке стоят дешевле (они ниже), а тут просят выше "
        "рыночной цены. Это переплата — лучше выбрать другой вариант или "
        "торговаться."
    )


def _comparables(df: pd.DataFrame, row: pd.Series, limit: int = 4) -> list[dict]:
    """Similar same-model listings that prove the verdict, direction-aware.

    For an underpriced car we show the same model on sale for MORE ("looks the
    same but costs more"); for an overpriced one we show it on sale for LESS,
    i.e. closer to the real market price. Prefers a ±2-year window, relaxing to
    the whole model when thin. ``diff_pct`` is signed (>0 dearer, <0 cheaper).
    """
    overpriced = float(row.get("discount_pct", 0.0)) <= 0
    same = df[
        (df["brand"] == row["brand"])
        & (df["model"] == row["model"])
        & (df["ad_id"] != row.get("ad_id", -1))
    ]
    same = same[same["price"] < row["price"]] if overpriced \
        else same[same["price"] > row["price"]]
    near = same[(same["year"] - row["year"]).abs() <= 2]
    # Closest comparables first: cheaper ones from the top down, dearer ones up.
    pool = (near if len(near) >= 2 else same).sort_values(
        "price", ascending=not overpriced
    ).head(limit)
    return [
        {
            "title": _title(r),
            "year": int(r["year"]),
            "mileage": int(r["mileage"]),
            "price": int(r["price"]),
            "url": r["url"],
            "diff_pct": round((r["price"] - row["price"]) / row["price"] * 100, 1),
        }
        for _, r in pool.iterrows()
    ]


def build_report(row: pd.Series, df: pd.DataFrame, model=None) -> dict:
    """Assemble the 'why this is a good deal' report for one listing.

    Args:
        row: Enriched listing row (predicted_price, discount_pct, confidence…).
        df: Enriched market frame, for the comparable listings.
        model: Trained model; loaded from disk if not given.

    Returns:
        JSON-serialisable report dict consumed by the Mini App detail screen.
    """
    model = model or load_model()
    factors, base_price, top = _price_factors(row, model)
    # Split into strengths (raise the fair price — what a buyer pays more for)
    # and caveats (lower it — already in the price). Dedupe: year/car_age map
    # to the same sentence.
    strengths: list[str] = []
    caveats: list[str] = []
    for f, v in top.items():
        line = _consumer_factor(f, float(v), row)
        if not line:
            continue
        bucket = strengths if float(v) >= 0 else caveats
        if line not in strengths and line not in caveats:
            bucket.append(line)
    overpriced = float(row.get("discount_pct", 0.0)) <= 0
    comp_title = (
        "Похожие в продаже — дешевле" if overpriced
        else "Похожие в продаже — дороже"
    )
    comp_sub = (
        "Такие же модели стоят меньше — это и есть рыночная цена:" if overpriced
        else "Такие же модели на рынке сейчас, но просят больше:"
    )
    return {
        "title": _title(row),
        "ad_id": int(row["ad_id"]) if pd.notna(row.get("ad_id")) else None,
        "url": row["url"],
        "year": int(row["year"]),
        "mileage": int(row["mileage"]),
        "price": int(row["price"]),
        "predicted_price": int(row["predicted_price"]),
        "discount_pct": round(float(row.get("discount_pct", 0.0)), 1),
        "grade": row.get("deal_grade", ""),
        "confidence": row.get("confidence", "medium"),
        "sample_count": int(row.get("sample_count", 0)),
        "base_price": base_price,
        "factors": factors,        # technical % weights (dashboard)
        "strengths": strengths,    # traits buyers pay more for (Mini App)
        "caveats": caveats,        # minuses already priced in (Mini App)
        "verdict": _verdict(row),
        "comparables": _comparables(df, row),
        "comp_title": comp_title,
        "comp_sub": comp_sub,
    }


def deal_report(ad_id: int) -> dict | None:
    """Full deal report for an ad already in the market frame, or None."""
    df = load_market()
    rows = df[df["ad_id"] == ad_id]
    if rows.empty:
        return None
    return build_report(rows.iloc[0], df)


def reliability_for_ad(ad_id: int) -> dict | None:
    """AI reliability verdict for an ad in the market frame, or None if absent."""
    from bot.reliability import reliability_report

    df = load_market()
    rows = df[df["ad_id"] == ad_id]
    if rows.empty:
        return None
    return reliability_report(rows.iloc[0])


def record_prediction_feedback(ad_id: int, verdict: str, chat_id: int = 0) -> dict:
    """Save a user's verdict on a deal call, snapshotting the model's claim.

    Looks the ad up in the market frame to attach the segment (brand/model) and
    the prediction being judged, then returns the refreshed tally for the ad.
    The next ``load_market`` refresh picks up the new calibration.
    """
    df = load_market()
    rows = df[df["ad_id"] == ad_id]
    snap = {}
    if not rows.empty:
        row = rows.iloc[0]
        snap = {
            "brand": str(row.get("brand", "")),
            "model": str(row.get("model", "")),
            "discount_pct": float(row.get("discount_pct", 0.0)),
            "predicted_price": int(row.get("predicted_price", 0)),
            "price": int(row.get("price", 0)),
        }
    feedback.record_feedback(ad_id, verdict, chat_id=chat_id, **snap)
    return feedback.feedback_for_ad(ad_id, chat_id)


def feedback_for_ad(ad_id: int, chat_id: int = 0) -> dict:
    """Current vote tally for an ad plus this user's own verdict."""
    return feedback.feedback_for_ad(ad_id, chat_id)
