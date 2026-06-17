"""Tests for the Telegram bot service layer and message formatting."""

import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from bot import formatting, service


@pytest.fixture()
def market(monkeypatch) -> pd.DataFrame:
    """Synthetic enriched frame injected into the bot's market cache."""
    now = datetime(2026, 6, 12, 12, 0, 0)
    df = pd.DataFrame(
        [
            # brand, price, score, grade, suspicious, scraped_at
            ("toyota", 1_500_000, 0.50, "🔥 Горячая", False, now),
            ("toyota", 900_000, 0.40, "👍 Хорошая", False, now - timedelta(days=2)),
            ("bmw", 3_000_000, 0.45, "🔥 Горячая", False, now - timedelta(days=2)),
            ("vaz", 300_000, 0.60, "🔥 Горячая", True, now),
        ],
        columns=[
            "brand", "price", "score", "deal_grade", "is_suspicious", "scraped_at",
        ],
    )
    df["ad_id"] = range(1, len(df) + 1)
    # Surfaced deals require a trustworthy (non-low) confidence estimate.
    df["confidence"] = "high"
    monkeypatch.setitem(service._cache, "df", df)
    monkeypatch.setitem(service._cache, "loaded_at", time.monotonic())
    return df


class TestService:
    def test_top_deals_excludes_suspicious(self, market):
        deals = service.top_deals(n=10)
        assert not deals["is_suspicious"].any()

    def test_top_deals_strict_budget_band(self, market):
        # "2–4 млн" must not leak cheaper cars: both bounds are enforced.
        deals = service.top_deals(n=10, min_price=2_000_000, max_price=4_000_000)
        assert deals["price"].between(2_000_000, 4_000_000).all()
        assert (deals["price"] >= 2_000_000).all()

    def test_top_deals_excludes_low_confidence(self, market):
        # A high-score deal whose estimate rests on too few comps must not
        # surface (the Hyundai-Matrix fake-discount case).
        market.loc[market["brand"] == "toyota", "confidence"] = "low"
        deals = service.top_deals(n=10)
        assert (deals["confidence"] != "low").all()
        assert "toyota" not in deals["brand"].values

    def test_top_deals_respects_price_and_brand(self, market):
        deals = service.top_deals(n=10, max_price=1_000_000, brand="toyota")
        assert (deals["price"] <= 1_000_000).all()
        assert (deals["brand"] == "toyota").all()

    def test_pick_cars_budget_window(self, market):
        deals = service.pick_cars(1_000_000, 2_000_000, None)
        assert deals["price"].between(1_000_000, 2_000_000).all()

    def test_new_hot_deals_only_fresh_hot(self, market):
        since = datetime(2026, 6, 12, 0, 0, 0)
        deals = service.new_hot_deals(since)
        # only the fresh, non-suspicious 🔥 toyota qualifies
        assert list(deals["brand"]) == ["toyota"]

    def test_popular_brands_sorted_by_count(self, market):
        assert service.popular_brands(2) == ["toyota", "bmw"]


class TestFormatting:
    @pytest.fixture()
    def row(self) -> pd.Series:
        return pd.Series(
            {
                "brand": "toyota",
                "model": "camry",
                "year": 2019,
                "modification": "Toyota Camry 70 Рестайлинг",
                "price": 1_850_000,
                "predicted_price": 2_100_000,
                "discount_pct": 11.9,
                "mileage": 95_000,
                "engine_volume": 2.5,
                "horse_power": 181,
                "fuel_type": "бензин",
                "transmission": "автомат",
                "region": "Москва",
                "confidence": "high",
                "deal_grade": "👍 Хорошая",
                "is_suspicious": False,
                "suspicious_reason": "",
                "url": "https://auto.ru/cars/used/sale/toyota/camry/1-a/",
            }
        )

    def test_caption_contains_key_facts(self, row):
        caption = formatting.deal_caption(row)
        assert "Toyota Camry 70 Рестайлинг, 2019" in caption
        assert "1 850 000 ₽" in caption
        assert "+11.9%" in caption
        assert "95 000 км" in caption
        assert "Москва" in caption

    def test_caption_fits_telegram_limit(self, row):
        caption = formatting.deal_caption(row, header="🔔 Новая горячая сделка")
        assert len(caption) <= 1024

    def test_suspicious_reason_shown(self, row):
        row["is_suspicious"] = True
        row["suspicious_reason"] = "Цена на 70% ниже рыночной оценки"
        assert "70% ниже" in formatting.deal_caption(row)

    def test_html_escaped(self, row):
        row["modification"] = "Camry <script>"
        assert "<script>" not in formatting.deal_caption(row)


class TestDispatcher:
    def test_dispatcher_builds(self):
        from bot.main import build_dispatcher

        dp = build_dispatcher()
        assert dp.message.handlers  # все хендлеры зарегистрированы
