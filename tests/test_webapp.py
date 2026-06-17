"""Tests for the Mini App API and Telegram initData validation."""

import hashlib
import hmac
import time
from urllib.parse import urlencode

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bot import service
from bot.webapp import app, validate_init_data


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    df = pd.DataFrame(
        [
            {
                "ad_id": 1,
                "brand": "toyota",
                "model": "camry",
                "modification": "Toyota Camry 70",
                "year": 2019,
                "price": 1_850_000,
                "predicted_price": 2_100_000,
                "discount_pct": 11.9,
                "score": 0.4,
                "deal_grade": "👍 Хорошая",
                "is_suspicious": False,
                "suspicious_reason": "",
                "mileage": 95_000,
                "engine_volume": 2.5,
                "horse_power": 181,
                "fuel_type": "бензин",
                "transmission": "автомат",
                "region": "Москва",
                "confidence": "high",
                "image_url": "",
                "url": "https://auto.ru/cars/used/sale/toyota/camry/1-a/",
                "scraped_at": pd.Timestamp("2026-06-12"),
            }
        ]
    )
    monkeypatch.setitem(service._cache, "df", df)
    monkeypatch.setitem(service._cache, "loaded_at", time.monotonic())
    return TestClient(app)


class TestApi:
    def test_top_returns_cards(self, client):
        cards = client.get("/api/top").json()
        assert cards[0]["title"] == "Toyota Camry 70"
        assert cards[0]["price"] == 1_850_000
        assert cards[0]["discount_pct"] == 11.9

    def test_top_price_filter(self, client):
        assert client.get("/api/top", params={"max_price": 1_000_000}).json() == []

    def test_pick_budget_window(self, client):
        cards = client.get(
            "/api/pick", params={"budget_from": 1_000_000, "budget_to": 2_000_000}
        ).json()
        assert len(cards) == 1

    def test_brands(self, client):
        assert client.get("/api/brands").json() == ["toyota"]

    def test_index_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Авторадар" in resp.text


class TestInitDataValidation:
    def _sign(self, params: dict, token: str) -> str:
        check = "\n".join(f"{k}={v}" for k, v in sorted(params.items()))
        secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
        params = dict(params)
        params["hash"] = hmac.new(
            secret, check.encode(), hashlib.sha256
        ).hexdigest()
        return urlencode(params)

    def test_valid_signature_accepted(self):
        token = "12345:TESTTOKEN"
        init_data = self._sign({"user": '{"id":1}', "auth_date": "1"}, token)
        assert validate_init_data(init_data, token)

    def test_tampered_signature_rejected(self):
        token = "12345:TESTTOKEN"
        init_data = self._sign({"user": '{"id":1}', "auth_date": "1"}, token)
        assert not validate_init_data(init_data + "x", token)
        assert not validate_init_data(init_data, "12345:OTHER")

    def test_garbage_rejected(self):
        assert not validate_init_data("not-a-querystring", "12345:TESTTOKEN")
