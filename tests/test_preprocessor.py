"""Tests for the DataPreprocessor pipeline."""

import pandas as pd
import pytest

from processing.preprocessor import CURRENT_YEAR, DataPreprocessor


@pytest.fixture()
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor()


class TestDropDuplicates:
    def test_removes_duplicate_ad_ids(self, preprocessor, sample_df):
        doubled = pd.concat([sample_df, sample_df], ignore_index=True)
        result = preprocessor.fit_transform(doubled)
        assert result["ad_id"].is_unique

    def test_keeps_unique_rows(self, preprocessor, sample_df):
        result = preprocessor._drop_duplicates(sample_df)
        assert len(result) == len(sample_df)


class TestFeatureEngineering:
    def test_no_negative_car_age(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df)
        assert (result["car_age"] >= 0).all()

    def test_mileage_per_year_formula(self, preprocessor, sample_df):
        result = preprocessor.engineer_features(sample_df)
        expected = result["mileage"] / (result["car_age"] + 1)
        pd.testing.assert_series_equal(
            result["mileage_per_year"], expected, check_names=False
        )

    def test_car_age_matches_year(self, preprocessor, sample_df):
        result = preprocessor.engineer_features(sample_df)
        assert (result["car_age"] == CURRENT_YEAR - result["year"]).all()

    def test_price_segment_labels(self, preprocessor, sample_df):
        result = preprocessor.engineer_features(sample_df)
        assert set(result["price_segment"].unique()) <= {"budget", "mid", "premium"}

    def test_power_density_formula(self, preprocessor, sample_df):
        result = preprocessor.engineer_features(sample_df)
        with_engine = result[result["engine_volume"] > 0]
        expected = with_engine["horse_power"] / with_engine["engine_volume"]
        pd.testing.assert_series_equal(
            with_engine["power_density"], expected, check_names=False
        )

    def test_power_density_nan_for_electric(self, preprocessor, sample_df):
        df = sample_df.copy()
        df.loc[df.index[0], "engine_volume"] = 0.0
        result = preprocessor.engineer_features(df)
        assert pd.isna(result.loc[result.index[0], "power_density"])

    def test_brand_segment_mapping(self, preprocessor, sample_df):
        df = sample_df.copy()
        df.loc[df.index[0], "brand"] = "bentley"
        df.loc[df.index[1], "brand"] = "nofamebrand"
        result = preprocessor.engineer_features(df)
        assert result.loc[result.index[0], "brand_segment"] == "luxury"
        assert result.loc[result.index[1], "brand_segment"] == "other"

    def test_fuel_and_modification_default_to_empty(self, preprocessor, sample_df):
        result = preprocessor.engineer_features(sample_df)
        assert (result["fuel_type"] == "").all()
        assert (result["modification"] == "").all()


class TestOutlierFiltering:
    def test_small_groups_untouched(self, preprocessor):
        """Groups under 10 rows must not be filtered (too few samples)."""
        df = pd.DataFrame(
            {
                "ad_id": range(5),
                "brand": ["lada"] * 5,
                "model": ["vesta"] * 5,
                "price": [100, 200, 300, 400, 10_000_000],
                "mileage": [10, 20, 30, 40, 50],
            }
        )
        result = preprocessor._filter_outliers(df)
        assert len(result) == 5

    def test_extreme_price_dropped_in_large_group(self, preprocessor):
        prices = [1_000_000] * 19 + [50_000_000]
        df = pd.DataFrame(
            {
                "ad_id": range(20),
                "brand": ["bmw"] * 20,
                "model": ["x5"] * 20,
                "price": prices,
                "mileage": [50_000] * 20,
            }
        )
        result = preprocessor._filter_outliers(df)
        assert 50_000_000 not in result["price"].values


class TestImplausiblePriceFilter:
    def test_drops_price_below_floor(self, preprocessor):
        """A rare model (group < 10) skips quantile filtering, so an absurd
        price must be caught by the absolute floor instead."""
        df = pd.DataFrame(
            {
                "ad_id": [1, 2, 3],
                "brand": ["jaecoo"] * 3,
                "model": ["j8"] * 3,
                "price": [3_750, 2_800_000, 2_900_000],
                "mileage": [34_000, 20_000, 25_000],
            }
        )
        result = preprocessor._filter_implausible(df)
        assert 3_750 not in result["price"].values
        assert len(result) == 2

    def test_keeps_cheap_but_plausible_car(self, preprocessor):
        df = pd.DataFrame(
            {
                "ad_id": [1, 2],
                "brand": ["vaz"] * 2,
                "model": ["2107"] * 2,
                "price": [45_000, 60_000],
                "mileage": [120_000, 90_000],
            }
        )
        result = preprocessor._filter_implausible(df)
        assert len(result) == 2
