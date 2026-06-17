"""Tests for persistence and price-history tracking (in-memory SQLite)."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scraper.schemas import CarAdSchema
from scraper.storage import Base, PriceHistory, get_price_dynamics, save_ads


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _make_ad(ad_id: int = 1, price: int = 1_000_000) -> CarAdSchema:
    return CarAdSchema(
        ad_id=ad_id,
        brand="toyota",
        model="camry",
        year=2018,
        mileage=90_000,
        price=price,
        body_type="sedan",
        engine_volume=2.5,
        horse_power=181,
        transmission="automatic",
        owners_count=1,
        url=f"https://auto.ru/cars/used/sale/{ad_id}/",
        published_at=datetime(2026, 5, 1),
    )


class TestSaveAds:
    def test_new_ad_inserted_with_history_point(self, engine):
        inserted = save_ads([_make_ad()], engine)
        assert inserted == 1
        with Session(engine) as session:
            history = session.query(PriceHistory).filter_by(ad_id=1).all()
        assert len(history) == 1
        assert history[0].price == 1_000_000

    def test_price_change_appends_history_and_updates_ad(self, engine):
        save_ads([_make_ad(price=1_000_000)], engine)
        inserted = save_ads([_make_ad(price=900_000)], engine)
        assert inserted == 0  # update, not insert
        with Session(engine) as session:
            history = session.query(PriceHistory).filter_by(ad_id=1).all()
        assert [h.price for h in history] == [1_000_000, 900_000]

    def test_unchanged_price_adds_no_history(self, engine):
        save_ads([_make_ad()], engine)
        save_ads([_make_ad()], engine)
        with Session(engine) as session:
            count = session.query(PriceHistory).filter_by(ad_id=1).count()
        assert count == 1


class TestPriceDynamics:
    def test_drop_pct_positive_when_price_falls(self, engine):
        save_ads([_make_ad(price=1_000_000)], engine)
        save_ads([_make_ad(price=800_000)], engine)
        dynamics = get_price_dynamics(engine)
        row = dynamics[dynamics["ad_id"] == 1].iloc[0]
        assert row["price_drop_pct"] == pytest.approx(20.0)
        assert row["n_price_changes"] == 1

    def test_empty_history_returns_empty_frame(self, engine):
        dynamics = get_price_dynamics(engine)
        assert dynamics.empty
