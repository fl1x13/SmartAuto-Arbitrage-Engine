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
