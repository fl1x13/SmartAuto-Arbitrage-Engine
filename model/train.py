"""CatBoost training pipeline with metric logging and artifact persistence."""

import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from config import cfg
from model.metrics import compute_regression_metrics
from processing.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


# Mileage buckets for prediction calibration (upper bounds, km). Extended to
# 500k/600k because the worst overpricing is on 300k+ km cars.
MILEAGE_BUCKETS = [
    50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 500_000, 600_000
]
# Clamp range for the correction factor. The floor is well below 1.0 so worn-
# out cars (whose fair price the raw model badly overstates) can be marked
# down hard; the ceiling stays near 1.0 to avoid inventing discounts.
_CAL_FLOOR, _CAL_CEIL = 0.5, 1.1
_MIN_BUCKET_SAMPLES = 20


def _bucket_factors(
    mileage: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> list[float]:
    """Median actual/predicted ratio per mileage bucket, clamped.

    Buckets with too few cars keep 1.0 (no correction).
    """
    bucket_idx = np.searchsorted(MILEAGE_BUCKETS, mileage)
    factors = []
    for b in range(len(MILEAGE_BUCKETS) + 1):
        in_bucket = bucket_idx == b
        if in_bucket.sum() >= _MIN_BUCKET_SAMPLES:
            ratio = float(np.median(actual[in_bucket] / predicted[in_bucket]))
            factors.append(round(min(max(ratio, _CAL_FLOOR), _CAL_CEIL), 4))
        else:
            factors.append(1.0)
    return factors


def _mileage_calibration(
    mileage: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    segment: np.ndarray,
) -> dict:
    """Segment-aware per-mileage-bucket correction factors.

    The model systematically overprices high-mileage cars (sparse data in
    that range), which inflates their apparent discount and floods the top
    deals with worn-out cars. Depreciation with mileage differs sharply by
    market segment — a 300k-km budget Lada loses far more of its value than a
    300k-km premium SUV — so factors are computed per (segment × bucket) with
    a global per-bucket fallback for thin segments.

    Returns:
        Dict with the bucket edges, a global factor list, and a per-segment
        factor map. Consumed by :func:`apply_mileage_calibration`.
    """
    return {
        "buckets": MILEAGE_BUCKETS,
        "global": _bucket_factors(mileage, actual, predicted),
        "by_segment": {
            seg: _bucket_factors(
                mileage[segment == seg],
                actual[segment == seg],
                predicted[segment == seg],
            )
            for seg in np.unique(segment)
            if (segment == seg).sum() >= _MIN_BUCKET_SAMPLES
        },
    }


def apply_mileage_calibration(
    preds: np.ndarray,
    mileage: np.ndarray,
    calibration,
    segment: np.ndarray | None = None,
) -> np.ndarray:
    """Apply per-bucket correction factors to ruble predictions.

    Accepts either the segment-aware dict produced by
    :func:`_mileage_calibration` or a legacy flat factor list (older
    artifacts). When ``segment`` is given, the matching per-segment factors
    are used per row, falling back to the global bucket factor.
    """
    if not calibration:
        return preds
    if isinstance(calibration, list):  # legacy artifact
        bucket_idx = np.searchsorted(MILEAGE_BUCKETS, mileage)
        return preds * np.asarray(calibration)[bucket_idx]

    buckets = calibration.get("buckets", MILEAGE_BUCKETS)
    global_factors = np.asarray(calibration.get("global", [1.0]))
    bucket_idx = np.searchsorted(buckets, mileage)
    factors = global_factors[bucket_idx]
    by_segment = calibration.get("by_segment", {})
    if segment is not None and by_segment:
        segment = np.asarray(segment)
        for seg, seg_factors in by_segment.items():
            mask = segment == seg
            if mask.any():
                factors = factors.copy()
                factors[mask] = np.asarray(seg_factors)[bucket_idx[mask]]
    return preds * factors


def train_model(df: pd.DataFrame) -> tuple[CatBoostRegressor, dict]:
    """Train a CatBoostRegressor on the preprocessed DataFrame.

    The model is trained on log(price): listing prices span three orders
    of magnitude (35K–50M ₽), so an RMSE loss in rubles is dominated by
    expensive cars and yields poor relative (MAPE) accuracy. Metrics are
    reported in ruble space; predictions must be exponentiated back
    (handled by :func:`model.predict.predict_price` via model_meta.json).

    Args:
        df: Cleaned DataFrame with all required features and a 'price' column.

    Returns:
        Tuple of (trained model, metrics dict with rmse and mape).
    """
    feature_cols = cfg.cat_features + cfg.num_features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X, y = df[feature_cols], df["price"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_idx = [feature_cols.index(f) for f in cfg.cat_features]
    train_pool = Pool(X_train, np.log(y_train), cat_features=cat_idx)
    val_pool = Pool(X_val, np.log(y_val), cat_features=cat_idx)

    # lr/depth/l2 picked by 5-fold CV grid search on real data (see PROGRESS.md)
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=100,
        random_seed=42,
        verbose=200,
    )
    model.fit(train_pool, eval_set=val_pool)

    preds = np.exp(model.predict(X_val))
    seg_val = X_val["brand_segment"].values
    calibration = _mileage_calibration(
        X_val["mileage"].values, y_val, preds, seg_val
    )
    preds = apply_mileage_calibration(
        preds, X_val["mileage"].values, calibration, seg_val
    )
    # Metrics are reported after calibration on the same validation split,
    # so they are mildly optimistic about the calibration step itself.
    metrics = compute_regression_metrics(y_val, preds)
    metrics["train_size"] = len(X_train)
    metrics["val_size"] = len(X_val)

    logger.info(
        "Training complete. MAPE=%.2f%%, RMSE=%.0f, train_size=%d",
        metrics["mape"],
        metrics["rmse"],
        metrics["train_size"],
    )

    # Persist artifacts
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(cfg.model_path))
    with open(cfg.feature_names_path, "w") as f:
        json.dump(feature_cols, f)
    with open(cfg.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(cfg.model_meta_path, "w") as f:
        json.dump(
            {"target_transform": "log", "mileage_calibration": calibration}, f
        )

    logger.info("Model saved to %s", cfg.model_path)
    return model, metrics


def run_training_pipeline() -> dict:
    """Load data from DB, preprocess, train, and return metrics.

    Returns:
        Metrics dict from training.
    """
    engine = create_engine(cfg.db_url)
    df = pd.read_sql("SELECT * FROM raw_ads", engine)
    logger.info("Loaded %d rows from database", len(df))

    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)

    _, metrics = train_model(df)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = run_training_pipeline()
    print(f"MAPE: {metrics['mape']:.2f}%  RMSE: {metrics['rmse']:,.0f}")
