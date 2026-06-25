"""Tests for arbitrage scoring and model inference."""

import pandas as pd
import pytest

from config import cfg
from model.predict import (
    calculate_arbitrage_score,
    flag_suspicious_listings,
    load_model,
)


class TestArbitrageScore:
    def test_underpriced_car_gets_positive_score(self):
        score = calculate_arbitrage_score(p_pred=1_000_000, p_act=700_000, brand="lada")
        assert score > 0

    def test_overpriced_car_gets_lower_score(self):
        underpriced = calculate_arbitrage_score(1_000_000, 700_000, "lada")
        overpriced = calculate_arbitrage_score(1_000_000, 1_300_000, "lada")
        assert underpriced > overpriced

    def test_liquid_brand_scores_higher_than_unknown(self):
        toyota = calculate_arbitrage_score(1_000_000, 800_000, "toyota")
        unknown = calculate_arbitrage_score(1_000_000, 800_000, "nofamebrand")
        assert toyota > unknown

    def test_brand_lookup_is_case_insensitive(self):
        lower = calculate_arbitrage_score(1_000_000, 800_000, "toyota")
        upper = calculate_arbitrage_score(1_000_000, 800_000, "TOYOTA")
        assert lower == upper

    def test_zero_predicted_price_returns_zero(self):
        assert calculate_arbitrage_score(0, 500_000, "kia") == 0.0

    def test_formula_matches_config_weights(self):
        p_pred, p_act = 1_000_000, 750_000
        expected = cfg.w1 * (p_pred - p_act) / p_pred + cfg.w2 * 1.2
        score = calculate_arbitrage_score(p_pred, p_act, "kia")
        assert score == pytest.approx(expected)


class TestEnrichment:
    @pytest.fixture()
    def enriched(self, sample_df):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet")
        from model.predict import enrich_with_predictions
        from processing.preprocessor import DataPreprocessor

        df = DataPreprocessor().engineer_features(sample_df)
        return enrich_with_predictions(df)

    def test_all_analysis_columns_present(self, enriched):
        expected = {
            "predicted_price", "score", "discount_pct", "is_suspicious",
            "suspicious_reason", "sample_count", "confidence", "deal_grade",
        }
        assert expected <= set(enriched.columns)

    def test_vectorized_score_matches_formula(self, enriched):
        from model.predict import market_liquidity, mileage_penalty_share

        row = enriched.iloc[0]
        liquidity = market_liquidity(enriched).iloc[0]
        discount = (row["predicted_price"] - row["price"]) / row["predicted_price"]
        expected = (
            cfg.w1 * min(discount, cfg.discount_reward_cap)
            + cfg.w2 * liquidity
            - cfg.w4 * mileage_penalty_share(enriched).iloc[0]
            - cfg.w5 * float(row["is_suspicious"])
        )
        assert row["score"] == pytest.approx(expected)

    def test_confidence_labels_valid(self, enriched):
        assert set(enriched["confidence"].unique()) <= {"low", "medium", "high"}

    def test_deal_grades_valid(self, enriched):
        valid = {"⚠️ Подозрительная", "🔥 Горячая", "👍 Хорошая", "— Рыночная"}
        assert set(enriched["deal_grade"].unique()) <= valid

    def test_suspicious_listings_never_graded_hot(self, enriched):
        suspicious = enriched[enriched["is_suspicious"]]
        assert (suspicious["deal_grade"] == "⚠️ Подозрительная").all()

    def test_every_suspicious_listing_has_a_reason(self, enriched):
        flagged = enriched[enriched["is_suspicious"]]
        assert (flagged["suspicious_reason"] != "").all()
        clean = enriched[~enriched["is_suspicious"]]
        assert (clean["suspicious_reason"] == "").all()


class TestMarketLiquidity:
    def _market(self) -> pd.DataFrame:
        rows = [{"brand": "vaz", "model": "granta"}] * 30
        rows += [{"brand": "ferrari", "model": "roma"}]
        rows += [{"brand": "toyota", "model": "camry"}] * 5
        return pd.DataFrame(rows)

    def test_deep_segment_more_liquid_than_exotic(self):
        from model.predict import market_liquidity

        liq = market_liquidity(self._market())
        granta = liq[0]
        ferrari = liq[30]
        assert granta > ferrari

    def test_values_within_configured_range(self):
        from model.predict import market_liquidity

        lo, hi = cfg.liquidity_range
        liq = market_liquidity(self._market())
        # manual overrides (e.g. toyota = 1.2) may sit at the range edge
        assert liq.between(min(lo, *cfg.liquidity_map.values()),
                           max(hi, *cfg.liquidity_map.values())).all()

    def test_manual_brand_override_wins(self):
        from model.predict import market_liquidity

        liq = market_liquidity(self._market())
        assert (liq[35:] == cfg.liquidity_map["toyota"]).all()


class TestSuspiciousPenalty:
    def test_suspicious_twin_scores_lower(self, sample_df):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet")
        from model.predict import enrich_with_predictions, rescore
        from processing.preprocessor import DataPreprocessor

        df = DataPreprocessor().engineer_features(sample_df)
        enriched = enrich_with_predictions(df)
        twin = enriched.iloc[[0, 0]].copy().reset_index(drop=True)
        twin.loc[0, "is_suspicious"] = False
        twin.loc[1, "is_suspicious"] = True
        rescored = rescore(twin)
        assert rescored.loc[1, "score"] < rescored.loc[0, "score"]
        assert rescored.loc[0, "score"] - rescored.loc[1, "score"] == pytest.approx(
            cfg.w5
        )


class TestDiscountRewardCap:
    def test_reward_saturates_above_cap(self, sample_df):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet")
        from model.predict import enrich_with_predictions, rescore
        from processing.preprocessor import DataPreprocessor

        df = DataPreprocessor().engineer_features(sample_df)
        enriched = enrich_with_predictions(df)
        twin = enriched.iloc[[0, 0]].copy().reset_index(drop=True)
        # Same car, same fair price; one priced for a 30% gap, one for 80%.
        twin["is_suspicious"] = False
        twin["mileage"] = 50_000  # equal mileage penalty for both
        pred = int(twin.loc[0, "predicted_price"])
        twin.loc[0, "price"] = int(pred * 0.50)  # 50% discount (above the cap)
        twin.loc[1, "price"] = int(pred * 0.20)  # 80% discount (further above)
        rescored = rescore(twin)
        # Beyond the cap the extra discount earns nothing, so scores match.
        assert rescored.loc[0, "score"] == pytest.approx(rescored.loc[1, "score"])


class TestMileagePenalty:
    def test_low_mileage_has_no_penalty(self):
        from model.predict import mileage_penalty_share

        df = pd.DataFrame({"mileage": [30_000, 149_999]})
        assert (mileage_penalty_share(df) == 0).all()

    def test_penalty_grows_with_mileage(self):
        from model.predict import mileage_penalty_share

        df = pd.DataFrame({"mileage": [300_000, 450_000]})
        share = mileage_penalty_share(df)
        assert share.iloc[0] == pytest.approx(1.0)
        assert share.iloc[1] == pytest.approx(2.0)

    def test_high_mileage_twin_scores_lower(self, sample_df):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet")
        from model.predict import enrich_with_predictions
        from processing.preprocessor import DataPreprocessor

        df = DataPreprocessor().engineer_features(sample_df)
        enriched = enrich_with_predictions(df)
        twin = enriched.iloc[[0, 0]].copy().reset_index(drop=True)
        twin.loc[1, "mileage"] = twin.loc[0, "mileage"] + 400_000
        from model.predict import rescore

        rescored = rescore(twin)
        assert rescored.loc[1, "score"] < rescored.loc[0, "score"]


class TestMileageCalibration:
    def test_apply_uses_bucket_factors_legacy_list(self):
        import numpy as np

        from model.train import MILEAGE_BUCKETS, apply_mileage_calibration

        # Legacy flat-list artifacts must still apply: only the top bucket
        # (above the last edge) is corrected here.
        factors = [1.0] * len(MILEAGE_BUCKETS) + [0.8]
        preds = np.array([1_000_000.0, 1_000_000.0])
        out = apply_mileage_calibration(
            preds, np.array([100_000, 700_000]), factors
        )
        assert out[0] == pytest.approx(1_000_000)
        assert out[1] == pytest.approx(800_000)

    def test_apply_uses_segment_factors(self):
        import numpy as np

        from model.train import MILEAGE_BUCKETS, apply_mileage_calibration

        n = len(MILEAGE_BUCKETS) + 1
        calibration = {
            "buckets": MILEAGE_BUCKETS,
            "global": [1.0] * n,
            "by_segment": {"budget": [0.6] * n},
        }
        preds = np.array([1_000_000.0, 1_000_000.0])
        out = apply_mileage_calibration(
            preds,
            np.array([100_000, 100_000]),
            calibration,
            segment=np.array(["budget", "premium"]),
        )
        assert out[0] == pytest.approx(600_000)  # budget → segment factor
        assert out[1] == pytest.approx(1_000_000)  # premium → global fallback

    def test_calibration_clamped_and_defaulted(self):
        import numpy as np

        from model.train import MILEAGE_BUCKETS, _mileage_calibration

        rng = np.random.default_rng(0)
        mileage = rng.uniform(0, 500_000, 400)
        predicted = np.full(400, 1_000_000.0)
        actual = predicted * 0.4  # extreme ratio must clamp to the 0.5 floor
        segment = np.full(400, "mass")
        calibration = _mileage_calibration(mileage, actual, predicted, segment)
        assert calibration["buckets"] == MILEAGE_BUCKETS
        assert len(calibration["global"]) == len(MILEAGE_BUCKETS) + 1
        assert all(0.5 <= f <= 1.1 for f in calibration["global"])
        assert all(0.5 <= f <= 1.1 for f in calibration["by_segment"]["mass"])


class TestSuspicionRules:
    def _make_df(self, **overrides) -> "pd.DataFrame":
        base = {
            "discount_pct": 10.0,
            "confidence": "high",
            "sample_count": 50,
            "car_age": 3,
            "mileage_per_year": 15_000.0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_market_priced_car_is_clean(self):
        reasons = flag_suspicious_listings(self._make_df())
        assert reasons.iloc[0] == ""

    def test_big_discount_with_reliable_estimate_is_flagged(self):
        reasons = flag_suspicious_listings(self._make_df(discount_pct=40.0))
        assert "ниже рыночной оценки" in reasons.iloc[0]

    def test_big_discount_with_low_confidence_is_not_flagged(self):
        # Rare brands (e.g. 13 Bentley ads) get shaky price estimates;
        # a moderate discount there must not mark the whole brand suspicious.
        reasons = flag_suspicious_listings(
            self._make_df(discount_pct=40.0, confidence="low", sample_count=3)
        )
        assert reasons.iloc[0] == ""

    def test_extreme_discount_is_flagged_even_with_low_confidence(self):
        reasons = flag_suspicious_listings(
            self._make_df(discount_pct=70.0, confidence="low", sample_count=3)
        )
        assert "70%" in reasons.iloc[0]

    def test_implausibly_low_mileage_is_flagged(self):
        reasons = flag_suspicious_listings(
            self._make_df(car_age=10, mileage_per_year=1_500.0)
        )
        assert "скрутка" in reasons.iloc[0]

    def test_new_car_with_low_mileage_is_clean(self):
        reasons = flag_suspicious_listings(
            self._make_df(car_age=1, mileage_per_year=1_500.0)
        )
        assert reasons.iloc[0] == ""


class TestEvaluateListing:
    @pytest.fixture()
    def market(self, sample_df):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet")
        from model.predict import enrich_with_predictions
        from processing.preprocessor import DataPreprocessor

        df = DataPreprocessor().engineer_features(sample_df)
        return enrich_with_predictions(df)

    def test_appends_evaluation_fields(self, market):
        from model.predict import evaluate_listing

        row = evaluate_listing(market.iloc[[0]].copy(), market)
        for field in (
            "predicted_price", "discount_pct", "sample_count",
            "confidence", "deal_grade", "suspicious_reason",
        ):
            assert field in row.index

    def test_confidence_counts_market_not_row(self, market):
        from model.predict import evaluate_listing

        row = evaluate_listing(market.iloc[[0]].copy(), market)
        same = (market["brand"] == row["brand"]) & (market["model"] == row["model"])
        assert row["sample_count"] == int(same.sum())


class TestLoadModel:
    def test_load_model_succeeds_when_artifact_exists(self):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet — run `python -m model.train`")
        model = load_model()
        assert model.is_fitted()

    def test_loaded_model_predicts_plausible_prices(self, sample_df):
        if not cfg.model_path.exists():
            pytest.skip("Model artifact not trained yet")
        from model.predict import predict_price
        from processing.preprocessor import DataPreprocessor

        df = DataPreprocessor().engineer_features(sample_df)
        preds = predict_price(df, load_model())
        assert (preds > 10_000).all()  # ruble space, not log space
