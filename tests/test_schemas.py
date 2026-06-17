"""Tests for Pydantic schemas and raw-string parsing."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from scraper.schemas import (
    CarAdSchema,
    RawAdSchema,
    extract_generation,
    parse_raw_ad,
)


class TestExtractGeneration:
    @pytest.mark.parametrize(
        "title, expected",
        [
            ("Audi A8 Long III (D4) Рестайлинг", "iii (d4) рест"),
            ("BMW X4 30d II (G02)", "ii (g02)"),
            ("Genesis G70 I", "i"),
            ("Mercedes-Benz S-Класс 600 Long IV (W220)", "iv (w220)"),
            ("Toyota Land Cruiser Prado 150 Series", "150 series"),
            ("Volkswagen Passat B6", "b6"),
            ("Lada (ВАЗ) Granta I Рестайлинг", "i рест"),
            ("", ""),
        ],
    )
    def test_extracts_generation_key(self, title, expected):
        assert extract_generation(title) == expected

    def test_parse_raw_ad_fills_generation(self):
        raw = _make_raw(modification="BMW X4 30d II (G02)")
        ad = parse_raw_ad(raw)
        assert ad is not None
        assert ad.generation == "ii (g02)"


def _make_raw(**overrides) -> RawAdSchema:
    """Build a valid RawAdSchema, overriding selected fields."""
    defaults = dict(
        ad_id=1,
        brand="Toyota",
        model="Camry",
        year=2018,
        mileage_raw="150 000 км",
        price_raw="1 200 000 ₽",
        body_type="Sedan",
        engine_volume_raw="2.5 л",
        horse_power=181,
        transmission="Automatic",
        owners_count=1,
        url="https://auto.ru/cars/used/sale/1/",
        published_at=datetime(2026, 5, 1),
    )
    defaults.update(overrides)
    return RawAdSchema(**defaults)


class TestParseRawAd:
    def test_parses_mileage_string_to_int(self):
        ad = parse_raw_ad(_make_raw(mileage_raw="150 000 км"))
        assert ad is not None
        assert ad.mileage == 150_000

    def test_parses_price_string_to_int(self):
        ad = parse_raw_ad(_make_raw(price_raw="1 200 000 ₽"))
        assert ad is not None
        assert ad.price == 1_200_000

    def test_parses_engine_volume_to_float(self):
        ad = parse_raw_ad(_make_raw(engine_volume_raw="2.5 л"))
        assert ad is not None
        assert ad.engine_volume == pytest.approx(2.5)

    def test_normalizes_brand_and_model_to_lowercase(self):
        ad = parse_raw_ad(_make_raw(brand="  Toyota ", model=" CAMRY "))
        assert ad is not None
        assert ad.brand == "toyota"
        assert ad.model == "camry"

    def test_drops_ad_with_zero_price(self):
        assert parse_raw_ad(_make_raw(price_raw="0 ₽")) is None

    def test_drops_ad_with_unparseable_price(self):
        assert parse_raw_ad(_make_raw(price_raw="цена не указана")) is None

    def test_drops_ad_with_year_below_range(self):
        assert parse_raw_ad(_make_raw(year=1989)) is None

    def test_drops_ad_with_year_in_future(self):
        future = datetime.now().year + 1
        assert parse_raw_ad(_make_raw(year=future)) is None


class TestCarAdSchema:
    def _valid_kwargs(self) -> dict:
        return dict(
            ad_id=1,
            brand="kia",
            model="rio",
            year=2020,
            mileage=60_000,
            price=900_000,
            body_type="sedan",
            engine_volume=1.6,
            horse_power=123,
            transmission="automatic",
            owners_count=1,
            url="https://auto.ru/cars/used/sale/1/",
            published_at=datetime(2026, 5, 1),
        )

    def test_valid_ad_passes(self):
        ad = CarAdSchema(**self._valid_kwargs())
        assert ad.price == 900_000

    def test_negative_price_rejected(self):
        kwargs = self._valid_kwargs() | {"price": -100}
        with pytest.raises(ValidationError, match="price"):
            CarAdSchema(**kwargs)

    def test_negative_mileage_rejected(self):
        kwargs = self._valid_kwargs() | {"mileage": -1}
        with pytest.raises(ValidationError, match="mileage"):
            CarAdSchema(**kwargs)

    def test_fixture_ads_all_valid(self, sample_ads):
        for raw in sample_ads:
            ad = CarAdSchema(**raw)
            assert ad.price > 0
            assert 1990 <= ad.year <= datetime.now().year
