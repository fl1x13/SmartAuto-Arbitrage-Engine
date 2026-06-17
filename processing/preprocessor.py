"""Data cleaning and feature engineering pipeline."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config import cfg
from scraper.schemas import extract_generation

logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now().year


class DataPreprocessor:
    """Clean and enrich a raw car listings DataFrame.

    Stages:
        1. Drop duplicates by ad_id.
        2. Remove outliers via per-brand+model quantile filtering.
        3. Engineer derived features: car_age, mileage_per_year, price_segment.
    """

    def __init__(
        self,
        lower_q: float = cfg.outlier_lower_q,
        upper_q: float = cfg.outlier_upper_q,
        mileage_lower_q: float = cfg.mileage_lower_q,
        mileage_upper_q: float = cfg.mileage_upper_q,
    ) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.mileage_lower_q = mileage_lower_q
        self.mileage_upper_q = mileage_upper_q

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing pipeline.

        Args:
            df: Raw DataFrame loaded from the database.

        Returns:
            Cleaned DataFrame with engineered features.
        """
        df = self._drop_duplicates(df)
        df = self._filter_outliers(df)
        df = self.engineer_features(df)
        logger.info("Preprocessing complete. Output shape: %s", df.shape)
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["ad_id"])
        removed = before - len(df)
        if removed:
            logger.info("Dropped %d duplicate ad_ids", removed)
        return df

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove extreme price/mileage rows per brand+model group."""
        before = len(df)

        def _quantile_filter(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < 10:
                return group
            low_p = group["price"].quantile(self.lower_q)
            high_p = group["price"].quantile(self.upper_q)
            low_m = group["mileage"].quantile(self.mileage_lower_q)
            high_m = group["mileage"].quantile(self.mileage_upper_q)
            return group[
                group["price"].between(low_p, high_p)
                & group["mileage"].between(low_m, high_m)
            ]

        df = (
            df.groupby(["brand", "model"], group_keys=False)[df.columns.tolist()]
            .apply(_quantile_filter)
            .reset_index(drop=True)
        )
        logger.info("Outlier removal: dropped %d rows", before - len(df))
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive model features from raw listing columns.

        Works on any frame with the raw columns — the training corpus or a
        single hand-built row from the valuation page — so every consumer
        shares one definition of the engineered features.

        Args:
            df: DataFrame with raw listing columns.

        Returns:
            Copy of df with engineered feature columns appended.
        """
        df = df.copy()
        df["car_age"] = CURRENT_YEAR - df["year"]
        df["mileage_per_year"] = df["mileage"] / (df["car_age"] + 1)
        # hp per litre separates turbo/tuned trims from base ones sharing the
        # same displacement; NaN for electric cars (CatBoost handles NaN).
        df["power_density"] = df["horse_power"] / df["engine_volume"].where(
            df["engine_volume"] > 0, np.nan
        )
        df["owners_per_year"] = df["owners_count"] / (df["car_age"] + 1)
        df["brand_segment"] = (
            df["brand"]
            .str.lower()
            .map(cfg.brand_segment_map)
            .fillna(cfg.default_brand_segment)
        )
        # Scraped after the corresponding columns were added; older rows
        # (and the demo source) carry an empty string.
        for col in ("fuel_type", "modification", "generation"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("")
        # Derive the generation key from the title for any row that has a
        # modification but no stored generation (existing DB rows, demo data),
        # so training benefits immediately without waiting for a re-scrape.
        needs_gen = (df["generation"] == "") & (df["modification"] != "")
        if needs_gen.any():
            df.loc[needs_gen, "generation"] = (
                df.loc[needs_gen, "modification"].map(extract_generation)
            )
        if "price" in df.columns:
            df["price_segment"] = pd.cut(
                df["price"],
                bins=[0, 800_000, 2_500_000, float("inf")],
                labels=["budget", "mid", "premium"],
            ).astype(str)
        return df
