"""Tests for persistence and price-history tracking (in-memory SQLite)."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scraper.schemas import CarAdSchema
from scraper.storage import (
    Base,
    CarAd,
    PriceHistory,
    get_price_dynamics,
    mark_ads_sold,
    save_ads,
)


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


class TestSoldFlag:
    def test_mark_ads_sold_flags_only_named_rows(self, engine):
        save_ads([_make_ad(1), _make_ad(2)], engine)
        flagged = mark_ads_sold({1}, engine)
        assert flagged == 1
        with Session(engine) as session:
            assert session.get(CarAd, 1).sold == 1
            assert session.get(CarAd, 2).sold == 0

    def test_mark_ads_sold_is_idempotent(self, engine):
        save_ads([_make_ad(1)], engine)
        mark_ads_sold({1}, engine)
        assert mark_ads_sold({1}, engine) == 0  # already sold, not re-counted

    def test_reseen_ad_clears_sold_flag(self, engine):
        save_ads([_make_ad(1)], engine)
        mark_ads_sold({1}, engine)
        save_ads([_make_ad(1)], engine)  # reappears in the feed → live again
        with Session(engine) as session:
            assert session.get(CarAd, 1).sold == 0


class TestAutoruBadge:
    def test_badge_persisted_on_insert(self, engine):
        ad = _make_ad(1)
        ad.autoru_badge, ad.autoru_discount_pct = "Ниже оценки на 10%", -10
        save_ads([ad], engine)
        with Session(engine) as session:
            row = session.get(CarAd, 1)
            assert row.autoru_badge == "Ниже оценки на 10%"
            assert row.autoru_discount_pct == -10

    def test_badge_refreshed_on_rescrape(self, engine):
        ad = _make_ad(1)
        ad.autoru_badge, ad.autoru_discount_pct = "Выше оценки на 5%", 5
        save_ads([ad], engine)
        ad2 = _make_ad(1)  # same ad, auto.ru re-rated it cheaper
        ad2.autoru_badge, ad2.autoru_discount_pct = "Ниже оценки на 8%", -8
        save_ads([ad2], engine)
        with Session(engine) as session:
            row = session.get(CarAd, 1)
            assert row.autoru_badge == "Ниже оценки на 8%"
            assert row.autoru_discount_pct == -8

    def test_no_badge_stays_null(self, engine):
        save_ads([_make_ad(1)], engine)
        with Session(engine) as session:
            row = session.get(CarAd, 1)
            assert row.autoru_badge is None
            assert row.autoru_discount_pct is None


class TestUpdateAutoruBadges:
    def test_updates_rating_from_detail_page(self, engine):
        from scraper.storage import update_autoru_badges

        save_ads([_make_ad(1), _make_ad(2)], engine)
        n = update_autoru_badges({1: ("Выше оценки на 16%", 16)}, engine)
        assert n == 1
        with Session(engine) as session:
            assert session.get(CarAd, 1).autoru_discount_pct == 16
            assert session.get(CarAd, 1).autoru_badge == "Выше оценки на 16%"
            assert session.get(CarAd, 2).autoru_discount_pct is None

    def test_empty_update_is_noop(self, engine):
        from scraper.storage import update_autoru_badges

        assert update_autoru_badges({}, engine) == 0
