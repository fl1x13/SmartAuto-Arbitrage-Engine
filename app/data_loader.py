"""Cached data loading functions for the Streamlit dashboard."""

import json
import logging

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from config import cfg
from model.predict import enrich_with_predictions, load_model
from processing.preprocessor import DataPreprocessor
from scraper.storage import get_price_dynamics

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300, show_spinner="Loading market data...")
def load_enriched_data() -> pd.DataFrame:
    """Load listings from DB, preprocess, and enrich with model predictions.

    Returns:
        DataFrame with all feature columns plus prediction/deal-analysis
        columns (predicted_price, score, deal_grade, price_drop_pct, ...).
    """
    engine = create_engine(cfg.db_url)
    df = pd.read_sql("SELECT * FROM raw_ads", engine)
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    model = load_model()
    df = enrich_with_predictions(df, model, price_dynamics=get_price_dynamics(engine))
    logger.info("Dashboard data loaded: %d rows", len(df))
    return df


@st.cache_data(ttl=300)
def load_price_history(ad_id: int) -> pd.DataFrame:
    """Load all recorded price points for one listing.

    Args:
        ad_id: Listing identifier.

    Returns:
        DataFrame with columns price, recorded_at (sorted by time).
    """
    engine = create_engine(cfg.db_url)
    return pd.read_sql(
        "SELECT price, recorded_at FROM price_history "
        "WHERE ad_id = :ad_id ORDER BY recorded_at",
        engine,
        params={"ad_id": ad_id},
    )


@st.cache_data(ttl=600, show_spinner="Скачиваю объявление с auto.ru...")
def fetch_ad_for_evaluation(url: str) -> dict | None:
    """Download one auto.ru ad by URL for the evaluate-by-link feature.

    Args:
        url: Full ad URL.

    Returns:
        Parsed listing as a dict (CarAdSchema.model_dump()), or None when
        the page could not be fetched/parsed. Cached so re-runs of the
        Streamlit script don't re-hit auto.ru.
    """
    from scraper.autoru_ad import fetch_ad_by_url

    ad = fetch_ad_by_url(url)
    return ad.model_dump() if ad is not None else None


@st.cache_data(ttl=86400, show_spinner=False)
def load_reliability_report(
    ad_id: int,
    brand: str,
    model: str,
    generation: str,
    modification: str,
    year: int,
    mileage: int,
    engine_volume: float,
    horse_power: int,
    fuel_type: str,
    transmission: str,
    drive: str,
    price: int,
    url: str = "",
) -> dict:
    """AI reliability verdict for one listing (Gemini + live web search).

    Reuses ``bot.reliability.reliability_report`` so the dashboard and the
    Telegram Mini App tell the identical story. The spec is passed as plain
    scalars (not the DataFrame row) so the verdict is cacheable across
    Streamlit reruns; ``ad_id`` only participates in the cache key.

    Args:
        ad_id: Listing identifier (cache key).
        brand, model, generation, modification, year, mileage, engine_volume,
        horse_power, fuel_type, transmission, drive, price: Listing spec.

    Returns:
        The verdict dict (score, verdict, summary, weak_points, at_mileage,
        checklist, title), or ``{"error": "no_key"}`` when no LLM key is
        configured, or ``{"error": "failed"}`` on any provider error.
    """
    from bot.reliability import NoApiKey, reliability_report

    row = pd.Series(
        {
            "brand": brand,
            "model": model,
            "generation": generation,
            "modification": modification,
            "year": year,
            "mileage": mileage,
            "engine_volume": engine_volume,
            "horse_power": horse_power,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "drive": drive,
            "price": price,
            "url": url,
        }
    )
    try:
        return reliability_report(row)
    except NoApiKey:
        return {"error": "no_key"}
    except Exception as e:  # noqa: BLE001 — surface a friendly message, never crash
        logger.error("Reliability report failed: %s: %s", type(e).__name__, e)
        return {"error": "failed"}


@st.cache_data(ttl=3600)
def load_model_metrics() -> dict:
    """Load the last training metrics from disk.

    Returns:
        Dict with keys: mape, rmse, train_size, val_size (or empty dict).
    """
    if cfg.metrics_path.exists():
        with open(cfg.metrics_path) as f:
            return json.load(f)
    return {}
