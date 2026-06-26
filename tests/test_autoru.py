"""Tests for the auto.ru listing parser against a real saved page fragment."""

from pathlib import Path

import pytest

from scraper.autoru import parse_autoru_page
from scraper.schemas import parse_raw_ad

FIXTURE = Path(__file__).parent / "fixtures" / "autoru_sample.html"


@pytest.fixture(scope="module")
def raw_ads():
    return parse_autoru_page(FIXTURE.read_text())


class TestParseAutoruPage:
    def test_extracts_all_cards(self, raw_ads):
        assert len(raw_ads) == 3

    def test_brand_model_from_url_slug(self, raw_ads):
        ad = raw_ads[0]
        assert ad.brand == "mercedes"
        assert ad.model == "gls klasse"

    def test_real_url_preserved(self, raw_ads):
        assert raw_ads[0].url.startswith("https://auto.ru/cars/used/sale/")
        assert raw_ads[0].ad_id == 1132757243

    def test_numeric_fields(self, raw_ads):
        ad = parse_raw_ad(raw_ads[0])
        assert ad is not None
        assert ad.year == 2025
        assert ad.mileage == 2579
        assert ad.price == 15_990_000
        assert ad.engine_volume == 3.0
        assert ad.horse_power == 367

    def test_spec_strings(self, raw_ads):
        ad = parse_raw_ad(raw_ads[1])
        assert ad is not None
        assert ad.transmission == "робот"
        assert "внедорожник" in ad.body_type

    def test_too_old_car_rejected_by_validation(self, raw_ads):
        chaika = next(r for r in raw_ads if r.year == 1960)
        assert parse_raw_ad(chaika) is None

    def test_region_extracted(self, raw_ads):
        assert raw_ads[0].region == "Москва"

    def test_drive_extracted(self, raw_ads):
        ad = parse_raw_ad(raw_ads[0])
        assert ad is not None
        assert ad.drive == "полный"
        drives = {parse_raw_ad(r).drive for r in raw_ads if parse_raw_ad(r)}
        assert drives <= {"передний", "задний", "полный"}

    def test_image_url_extracted_with_scheme(self, raw_ads):
        assert raw_ads[0].image_url.startswith(
            "https://avatars.mds.yandex.net/get-autoru-vos/"
        )

    def test_image_url_survives_validation(self, raw_ads):
        ad = parse_raw_ad(raw_ads[0])
        assert ad is not None
        assert ad.image_url == raw_ads[0].image_url

    def test_fuel_type_extracted(self, raw_ads):
        assert raw_ads[0].fuel_type == "дизель"
        assert raw_ads[1].fuel_type == "бензин"

    def test_modification_is_full_card_title(self, raw_ads):
        assert raw_ads[0].modification == "Mercedes-Benz GLS 450 d II (X167) Рестайлинг"

    def test_empty_html(self):
        assert parse_autoru_page("<html><body></body></html>") == []


class TestParseBadge:
    @staticmethod
    def _card(text: str, cls: str = "ListingItemUniversalPrice__fairPriceBadge-BsCSU"):
        from bs4 import BeautifulSoup

        return BeautifulSoup(f'<div><span class="{cls}">{text}</span></div>', "lxml")

    def test_below_estimate_is_negative(self):
        from scraper.autoru import _parse_badge

        result = _parse_badge(self._card("Ниже оценки на 15%"))
        assert result == ("Ниже оценки на 15%", -15)

    def test_above_estimate_is_positive(self):
        from scraper.autoru import _parse_badge

        result = _parse_badge(self._card("Выше оценки на 3%"))
        assert result == ("Выше оценки на 3%", 3)

    def test_fair_price_is_zero(self):
        from scraper.autoru import _parse_badge

        result = _parse_badge(self._card("Справедливая цена"))
        assert result == ("Справедливая цена", 0)

    def test_no_badge_is_none(self):
        from scraper.autoru import _parse_badge

        assert _parse_badge(self._card("x", cls="Other")) == (None, None)

    def test_badge_passes_through_validation(self):
        from scraper.schemas import RawAdSchema, parse_raw_ad

        raw = RawAdSchema(
            ad_id=1, brand="kia", model="rio", year=2019,
            mileage_raw="50 000", price_raw="900 000 ₽", body_type="седан",
            engine_volume_raw="1.6", horse_power=123, transmission="автомат",
            owners_count=1, url="https://auto.ru/cars/used/sale/kia/rio/1-a/",
            published_at=__import__("datetime").datetime(2026, 6, 1),
            autoru_badge="Ниже оценки на 12%", autoru_discount_pct=-12,
        )
        ad = parse_raw_ad(raw)
        assert ad is not None
        assert ad.autoru_badge == "Ниже оценки на 12%"
        assert ad.autoru_discount_pct == -12
