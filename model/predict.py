"""Model inference, arbitrage scoring, anomaly detection, and deal grading."""

import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from config import cfg

logger = logging.getLogger(__name__)

_model_cache: CatBoostRegressor | None = None
_meta_cache: dict | None = None


def _model_meta() -> dict:
    """Load (and cache) the saved model's metadata."""
    global _meta_cache
    if _meta_cache is None:
        if cfg.model_meta_path.exists():
            with open(cfg.model_meta_path) as f:
                _meta_cache = json.load(f)
        else:
            _meta_cache = {}  # artifacts predate the log-target pipeline
    return _meta_cache


def _target_is_log() -> bool:
    """Whether the saved model predicts log(price) instead of rubles."""
    return _model_meta().get("target_transform") == "log"


def predict_price(
    df: pd.DataFrame, model: CatBoostRegressor | None = None
) -> np.ndarray:
    """Predict fair market prices in rubles.

    Wraps ``model.predict``, undoes the log-target transform when the
    artifact was trained on log(price), and applies the saved per-mileage
    calibration (the raw model overprices high-mileage cars) — always call
    this instead of ``model.predict`` to get ruble values.

    Args:
        df: DataFrame containing the model's feature columns.
        model: Trained model; loaded from disk if not provided.

    Returns:
        Array of predicted prices in rubles.
    """
    if model is None:
        model = load_model()
    preds = model.predict(df[cfg.cat_features + cfg.num_features])
    if _target_is_log():
        preds = np.exp(preds)
    calibration = _model_meta().get("mileage_calibration")
    if calibration:
        from model.train import apply_mileage_calibration

        segment = (
            df["brand_segment"].values if "brand_segment" in df.columns else None
        )
        preds = apply_mileage_calibration(
            preds, df["mileage"].values, calibration, segment
        )
    return preds


def load_model() -> CatBoostRegressor:
    """Load the trained CatBoost model from disk (cached in memory).

    Returns:
        Loaded CatBoostRegressor instance.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {cfg.model_path}. Run `python -m model.train` first."
        )
    model = CatBoostRegressor()
    model.load_model(str(cfg.model_path))
    _model_cache = model
    logger.info("Model loaded from %s", cfg.model_path)
    return model


def mileage_penalty_share(df: pd.DataFrame) -> pd.Series:
    """How far past the comfortable-mileage threshold a listing is.

    0 below ``cfg.mileage_penalty_start``; +1 for each
    ``cfg.mileage_penalty_scale`` km above it. Used as ``score -= w4 * share``:
    a 450k-km car may be honestly cheap, but it is much harder to resell,
    so it must not crowd genuinely good deals out of the top.

    Args:
        df: DataFrame with a mileage column.

    Returns:
        Float Series aligned with df.index.
    """
    return (
        (df["mileage"] - cfg.mileage_penalty_start).clip(lower=0)
        / cfg.mileage_penalty_scale
    )


def delivery_surcharge(df: pd.DataFrame) -> pd.Series:
    """Delivery cost to central Russia per listing, by region.

    A car in a far-east hub (``cfg.import_regions``) is cheaper at source but
    the buyer pays ``cfg.import_delivery_surcharge`` to bring it across the
    country — and these are the Korea/Japan import lanes, where the sticker is
    often only the price "to Vladivostok". Adding this to the price before the
    discount is computed compares the *landed* cost against the model's
    nationwide fair value, so an import no longer looks cheaper than a domestic
    car just for omitting the freight.

    Args:
        df: DataFrame with a region column (0 surcharge when absent).

    Returns:
        Integer Series aligned with df.index.
    """
    if "region" not in df.columns:
        return pd.Series(0, index=df.index)
    return df["region"].isin(cfg.import_regions).astype(int) * (
        cfg.import_delivery_surcharge
    )


def autoru_badge_overclaim(df: pd.DataFrame, our_discount: pd.Series) -> pd.Series:
    """How much more discount we claim than auto.ru's own price rating allows.

    auto.ru anchors each listing's fair price with a badge — overwhelmingly
    "Справедливая цена" (``autoru_discount_pct`` 0), occasionally "Ниже оценки
    на X%" (negative). It is an independent authority: when auto.ru calls a
    price fair but our model claims a 29% discount, the excess is almost always
    our model overvaluing the car (the Prado/old-car failure mode), not a real
    deal. The *overclaim* — our discount beyond what auto.ru grants — is
    returned so the score can penalise it. Agreement, or us being the more
    conservative of the two, costs nothing; a listing with no badge → 0.

    Args:
        df: DataFrame with an autoru_discount_pct column (0 when absent).
        our_discount: Our own discount share (landed) aligned with df.index.

    Returns:
        Float Series in [0, …] aligned with df.index.
    """
    if "autoru_discount_pct" not in df.columns:
        return pd.Series(0.0, index=df.index)
    autoru_share = -pd.to_numeric(df["autoru_discount_pct"], errors="coerce") / 100
    overclaim = (our_discount - autoru_share).clip(lower=0)
    return overclaim.fillna(0.0)  # NaN = no badge → no adjustment


def segment_discount_haircut(df: pd.DataFrame) -> pd.Series:
    """Discount noise allowance per listing, by market segment.

    Premium and luxury cars have far wider price dispersion than mass-market
    ones, so a 30% gap below the model's estimate is mostly noise there — and
    a cheap premium car usually hides a costly reason (accident, grey import,
    looming repairs). Shaving ``cfg.segment_discount_noise`` off the discount
    before it earns score or a hot grade keeps premium out of the top deals
    unless it is dramatically — not merely moderately — underpriced.

    Args:
        df: DataFrame with a brand_segment column (0 allowance when absent).

    Returns:
        Float Series aligned with df.index.
    """
    if "brand_segment" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["brand_segment"].map(cfg.segment_discount_noise).fillna(0.0)


def age_discount_haircut(df: pd.DataFrame) -> pd.Series:
    """Discount noise allowance per listing, by car age.

    The price model overvalues old cars, so their discounts are largely model
    error rather than real deals (hand-labelling showed the misses clustered
    in cars older than ``cfg.age_discount_start``). Shave
    ``cfg.age_discount_per_year`` of discount per year past that age, capped at
    ``cfg.age_discount_cap``, before it earns score or a hot grade.

    Args:
        df: DataFrame with a car_age column (0 allowance when absent).

    Returns:
        Float Series aligned with df.index.
    """
    if "car_age" not in df.columns:
        return pd.Series(0.0, index=df.index)
    over = (df["car_age"] - cfg.age_discount_start).clip(lower=0)
    return (over * cfg.age_discount_per_year).clip(upper=cfg.age_discount_cap)


def total_discount_haircut(df: pd.DataFrame) -> pd.Series:
    """Combined discount haircut: segment noise plus the old-car allowance."""
    return segment_discount_haircut(df) + age_discount_haircut(df)


def market_liquidity(df: pd.DataFrame) -> pd.Series:
    """Per-listing liquidity coefficient within ``cfg.liquidity_range``.

    Data-driven: the percentile rank of brand+model market depth (how many
    such cars are on sale right now). Deep segments (Granta, Camry, Polo)
    turn over fast and resell easily; a one-of-a-kind exotic may take months
    to find a buyer. Manual brand overrides from ``cfg.liquidity_map`` take
    precedence where defined.

    Args:
        df: DataFrame with brand and model columns.

    Returns:
        Float Series aligned with df.index.
    """
    lo, hi = cfg.liquidity_range
    depth = df.groupby(["brand", "model"])["model"].transform("size")
    liquidity = lo + (hi - lo) * depth.rank(pct=True)
    override = df["brand"].str.lower().map(cfg.liquidity_map)
    return override.fillna(liquidity)


def calculate_arbitrage_score(p_pred: float, p_act: float, brand: str) -> float:
    """Compute the arbitrage attractiveness score for a listing.

    Scalar reference form of the formula:
    score = w1 * (P_pred - P_act) / P_pred + w2 * L

    The vectorized pipeline (:func:`enrich_with_predictions`, :func:`rescore`)
    uses the data-driven :func:`market_liquidity` for L; this scalar helper
    falls back to the static brand map.

    Args:
        p_pred: Predicted fair market price.
        p_act: Actual listing price.
        brand: Car brand (used to look up liquidity coefficient).

    Returns:
        Arbitrage score. Higher = more attractive deal.
    """
    if p_pred <= 0:
        return 0.0
    liquidity = cfg.liquidity_map.get(brand.lower(), cfg.default_liquidity)
    price_discount = (p_pred - p_act) / p_pred
    return cfg.w1 * price_discount + cfg.w2 * liquidity


# Thresholds for the suspicion rules (см. flag_suspicious_listings)
SUSPICIOUS_DISCOUNT_PCT = 35.0  # too cheap vs market estimate
EXTREME_DISCOUNT_PCT = 60.0  # too cheap even for a shaky estimate
MIN_KM_PER_YEAR = 3_000  # below this an odometer rollback is likely
MIN_AGE_FOR_MILEAGE_CHECK = 5  # young cars legitimately have low mileage


def flag_suspicious_listings(df: pd.DataFrame) -> pd.Series:
    """Flag too-good-to-be-true listings with a human-readable reason.

    A listing is suspicious when its price is implausibly far below the
    model's market estimate (scammers and cars with hidden problems —
    salvage, liens, flood damage — are priced to sell instantly) or when
    its mileage is implausibly low for its age (odometer-fraud pattern).
    The moderate-discount rule only fires when the estimate is backed by
    enough similar listings (confidence != "low"), so rare brands with
    shaky price estimates are not flagged just for being rare.

    Args:
        df: Enriched DataFrame with discount_pct, confidence, car_age and
            mileage_per_year columns.

    Returns:
        String Series of reasons aligned with df.index ("" = not suspicious).
    """
    reasons = pd.Series("", index=df.index, name="suspicious_reason")
    discount = df["discount_pct"]

    extreme = discount >= EXTREME_DISCOUNT_PCT
    big = (
        (discount >= SUSPICIOUS_DISCOUNT_PCT)
        & ~extreme
        & (df["confidence"] != "low")
    )
    low_mileage = (df["car_age"] >= MIN_AGE_FOR_MILEAGE_CHECK) & (
        df["mileage_per_year"] < MIN_KM_PER_YEAR
    )

    pct = discount.round(0).astype(int).astype(str)
    reasons[extreme] = (
        "Цена на " + pct[extreme] + "% ниже рыночной оценки — так дёшево "
        "обычно продают авто со скрытыми проблемами или мошенники"
    )
    reasons[big] = (
        "Цена на " + pct[big] + "% ниже рыночной оценки по "
        + df.loc[big, "sample_count"].astype(str)
        + " похожим объявлениям — слишком хорошо, чтобы быть правдой"
    )
    km_year = (df["mileage_per_year"] / 1000).round(1).astype(str)
    age = df["car_age"].astype(int).astype(str)
    mileage_only = low_mileage & (reasons == "")
    reasons[mileage_only] = (
        "Пробег ~" + km_year[mileage_only] + " тыс. км/год при возрасте "
        + age[mileage_only] + " лет — возможна скрутка одометра"
    )

    logger.info("Suspicion rules: %d listings flagged", int((reasons != "").sum()))
    return reasons


def _confidence_from_sample_count(counts: pd.Series) -> pd.Series:
    """Map per-group sample counts to low/medium/high confidence labels."""
    return pd.cut(
        counts,
        bins=[0, 5, 20, float("inf")],
        labels=["low", "medium", "high"],
    ).astype(str)


def _grade_deals(
    df: pd.DataFrame, hot_threshold: float = 15.0, good_threshold: float = 5.0
) -> pd.Series:
    """Combine discount, confidence, and anomaly flag into a human-readable grade.

    Grades are based on discount_pct rather than score: the liquidity term
    of the score is a constant offset per brand, so a score threshold would
    mark even overpriced cars of liquid brands as good deals.
    """
    # Grade on the discount net of the segment noise allowance, so a premium
    # car merely 30% below market is graded "good", not "hot" — the same
    # haircut the score applies, kept consistent with what the user is shown.
    eff_discount = df["discount_pct"] - total_discount_haircut(df) * 100
    conditions = [
        df["is_suspicious"],
        (eff_discount >= hot_threshold) & (df["confidence"] != "low"),
        eff_discount >= good_threshold,
    ]
    choices = ["⚠️ Подозрительная", "🔥 Горячая", "👍 Хорошая"]
    return pd.Series(
        np.select(conditions, choices, default="— Рыночная"),
        index=df.index,
        name="deal_grade",
    )


def rescore(
    df: pd.DataFrame,
    w1: float | None = None,
    w2: float | None = None,
    w3: float | None = None,
    w4: float | None = None,
    hot_threshold: float = 15.0,
    good_threshold: float = 5.0,
) -> pd.DataFrame:
    """Recompute score and deal_grade on an already-enriched DataFrame.

    Lets the dashboard expose the scoring formula as live controls without
    re-running model inference or anomaly detection.

    Args:
        df: Enriched DataFrame (output of :func:`enrich_with_predictions`).
        w1: Weight of the price discount term; ``cfg.w1`` when None.
        w2: Weight of the brand liquidity term; ``cfg.w2`` when None.
        w3: Weight of the price-drop term; ``cfg.w3`` when None.
        w4: Weight of the high-mileage penalty; ``cfg.w4`` when None.
        hot_threshold: Min discount (%) for the "🔥 Горячая" grade.
        good_threshold: Min discount (%) for the "👍 Хорошая" grade.

    Returns:
        Copy of df with score and deal_grade recomputed.
    """
    w1 = cfg.w1 if w1 is None else w1
    w2 = cfg.w2 if w2 is None else w2
    w3 = cfg.w3 if w3 is None else w3
    w4 = cfg.w4 if w4 is None else w4

    df = df.copy()
    liquidity = market_liquidity(df)
    pred = df["predicted_price"].where(df["predicted_price"] > 0)
    df["delivery_surcharge"] = delivery_surcharge(df)
    df["landed_price"] = df["price"] + df["delivery_surcharge"]
    discount = ((pred - df["landed_price"]) / pred).fillna(0.0)
    drop_share = (
        df["price_drop_pct"].clip(lower=0) / 100
        if "price_drop_pct" in df.columns
        else 0.0
    )
    eff_discount = (discount - total_discount_haircut(df)).clip(lower=0)
    df["score"] = (
        w1 * eff_discount.clip(upper=cfg.discount_reward_cap)
        + w2 * liquidity
        + w3 * drop_share
        - w4 * mileage_penalty_share(df)
        - cfg.w5 * df["is_suspicious"].astype(float)
        - cfg.w6 * autoru_badge_overclaim(df, discount)
    )
    df.loc[df["predicted_price"] <= 0, "score"] = 0.0
    df["discount_pct"] = (discount * 100).round(1)
    df["deal_grade"] = _grade_deals(df, hot_threshold, good_threshold)
    return df


def evaluate_listing(
    row_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model: CatBoostRegressor | None = None,
) -> pd.Series:
    """Evaluate one listing against the enriched market frame.

    Unlike :func:`enrich_with_predictions`, confidence is taken from the
    market corpus (how many same brand+model ads it holds), not from the
    single row itself — a lone row would always look low-confidence.

    Args:
        row_df: Single-row DataFrame with engineered feature columns and price.
        market_df: Enriched market DataFrame (output of the dashboard loader).
        model: Trained model; loaded from disk if not provided.

    Returns:
        The row as a Series with predicted_price, discount_pct, sample_count,
        confidence, suspicious_reason, is_suspicious, deal_grade appended.
    """
    df = row_df.copy()
    df["predicted_price"] = predict_price(df, model).astype(int)
    pred = df["predicted_price"].where(df["predicted_price"] > 0)
    df["delivery_surcharge"] = delivery_surcharge(df)
    df["landed_price"] = df["price"] + df["delivery_surcharge"]
    df["discount_pct"] = (
        ((pred - df["landed_price"]) / pred).fillna(0.0) * 100
    ).round(1)

    row = df.iloc[0]
    same_model = (market_df["brand"] == row["brand"]) & (
        market_df["model"] == row["model"]
    )
    df["sample_count"] = int(same_model.sum())
    df["confidence"] = _confidence_from_sample_count(df["sample_count"])

    df["suspicious_reason"] = flag_suspicious_listings(df)
    df["is_suspicious"] = df["suspicious_reason"] != ""
    df["deal_grade"] = _grade_deals(df)
    return df.iloc[0]


def explain_prediction(
    row_df: pd.DataFrame, model: CatBoostRegressor | None = None
) -> tuple[pd.Series, float, str]:
    """Explain a single price prediction via CatBoost-native SHAP values.

    For log-target models, SHAP values live in log space, so they are
    converted to multiplicative percentage effects on the price
    ((exp(v) − 1) · 100) and the base value to rubles.

    Args:
        row_df: Single-row DataFrame with the model's feature columns.
        model: Trained model; loaded from disk if not provided.

    Returns:
        Tuple of (per-feature contributions, base value in rubles — the
        model's average prediction over the training data, units of the
        contributions: "%" for log-target models, "₽" otherwise).
    """
    if model is None:
        model = load_model()

    feature_cols = cfg.cat_features + cfg.num_features
    cat_idx = [feature_cols.index(f) for f in cfg.cat_features]
    pool = Pool(row_df[feature_cols], cat_features=cat_idx)
    shap_matrix = model.get_feature_importance(type="ShapValues", data=pool)
    contributions = pd.Series(shap_matrix[0][:-1], index=feature_cols)
    base_value = float(shap_matrix[0][-1])
    if _target_is_log():
        return (np.exp(contributions) - 1) * 100, float(np.exp(base_value)), "%"
    return contributions, base_value, "₽"


def _apply_corrections(
    df: pd.DataFrame, corrections: dict[tuple[str, str], float] | None
) -> "pd.Series":
    """Scale ``predicted_price`` by per-(brand, model) feedback factors.

    No-op (returns the column unchanged) when there are no corrections.
    """
    if not corrections:
        return df["predicted_price"]
    factors = [
        corrections.get((str(b).lower(), str(m).lower()), 1.0)
        for b, m in zip(df["brand"], df["model"])
    ]
    scaled = df["predicted_price"] * pd.Series(factors, index=df.index)
    return scaled.round().astype(int)


def enrich_with_predictions(
    df: pd.DataFrame,
    model: CatBoostRegressor | None = None,
    price_dynamics: pd.DataFrame | None = None,
    corrections: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Add prediction and deal-analysis columns to the DataFrame.

    Appended columns: predicted_price, score, discount_pct, is_suspicious,
    suspicious_reason, sample_count, confidence, deal_grade, price_drop_pct,
    n_price_changes.

    Args:
        df: DataFrame with feature columns matching the trained model.
        model: Trained model; loaded from disk if not provided.
        price_dynamics: Optional per-ad price-drop stats from
            :func:`scraper.storage.get_price_dynamics`. When present, a
            price-drop bonus (w3 * drop share) is added to the score —
            a seller cutting the price is motivated to sell fast.
        corrections: Optional per-(brand, model) fair-price multipliers learned
            from user feedback (see :func:`bot.feedback.prediction_corrections`).
            Applied to ``predicted_price`` before discount/score so a segment
            users keep flagging as over-rated is demoted everywhere.

    Returns:
        DataFrame with analysis columns appended.
    """
    if model is None:
        model = load_model()

    df = df.copy()
    df["predicted_price"] = predict_price(df, model).astype(int)
    df["predicted_price"] = _apply_corrections(df, corrections)

    pred = df["predicted_price"].where(df["predicted_price"] > 0)
    # Discount is measured against the landed price (sticker + delivery to
    # central Russia), so a far-east import is not flattered by freight it
    # omits from the sticker.
    df["delivery_surcharge"] = delivery_surcharge(df)
    df["landed_price"] = df["price"] + df["delivery_surcharge"]
    discount = ((pred - df["landed_price"]) / pred).fillna(0.0)
    df["discount_pct"] = (discount * 100).round(1)
    df["sample_count"] = df.groupby(["brand", "model"])["price"].transform("size")
    df["confidence"] = _confidence_from_sample_count(df["sample_count"])
    # Suspicion is computed before the score so it can demote the listing:
    # scams and hidden-problem cars are flagged precisely because their price
    # gap is too good to be true, and that gap must not also reward them.
    df["suspicious_reason"] = flag_suspicious_listings(df)
    df["is_suspicious"] = df["suspicious_reason"] != ""

    # Vectorized arbitrage score:
    # w1·min(eff_discount, cap) + w2·liquidity − w4·mileage_penalty
    # − w5·suspicious (+ w3·drop below). The reward is capped so a 90% gap
    # (almost always a scam) cannot outrank an honest 25% deal, and a
    # per-segment haircut keeps premium out of the top unless it is
    # dramatically underpriced.
    liquidity = market_liquidity(df)
    eff_discount = (discount - total_discount_haircut(df)).clip(lower=0)
    df["score"] = (
        cfg.w1 * eff_discount.clip(upper=cfg.discount_reward_cap)
        + cfg.w2 * liquidity
        - cfg.w4 * mileage_penalty_share(df)
        - cfg.w5 * df["is_suspicious"].astype(float)
        - cfg.w6 * autoru_badge_overclaim(df, discount)
    )
    df.loc[df["predicted_price"] <= 0, "score"] = 0.0

    if price_dynamics is not None and not price_dynamics.empty:
        df = df.merge(
            price_dynamics[["ad_id", "price_drop_pct", "n_price_changes"]],
            on="ad_id",
            how="left",
        )
        df["price_drop_pct"] = df["price_drop_pct"].fillna(0.0)
        df["n_price_changes"] = df["n_price_changes"].fillna(0).astype(int)
        df["score"] += cfg.w3 * (df["price_drop_pct"].clip(lower=0) / 100)
    else:
        df["price_drop_pct"] = 0.0
        df["n_price_changes"] = 0

    df["deal_grade"] = _grade_deals(df)
    return df
