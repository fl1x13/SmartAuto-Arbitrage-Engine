"""Regression metrics: computation and logging helpers."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE and MAPE for a regression prediction.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dict with keys "rmse" and "mape" (MAPE in percent).
    """
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"rmse": rmse, "mape": mape}


def log_metrics(metrics: dict, prefix: str = "Validation") -> None:
    """Log MAPE/RMSE at INFO level in a consistent format.

    Args:
        metrics: Dict containing "mape" and "rmse".
        prefix: Context label for the log line.
    """
    logger.info(
        "%s metrics: MAPE=%.2f%%, RMSE=%.0f",
        prefix,
        metrics["mape"],
        metrics["rmse"],
    )


def mape_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    """Compute MAPE separately for each group label (e.g. per brand).

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        groups: Group label for each sample, same length as y_true.

    Returns:
        Mapping of group label to MAPE percentage, sorted by MAPE descending.
    """
    result: dict[str, float] = {}
    for label in np.unique(groups):
        mask = groups == label
        result[str(label)] = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
