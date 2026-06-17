"""Tests for the HTML listing parser (demo-source round-trip)."""

import random

import pytest

from scraper.demo_source import render_listing_page
from scraper.parser import parse_listing_page
from scraper.schemas import parse_raw_ad


@pytest.fixture(autouse=True)
def _fixed_seed():
    random.seed(7)


class TestParseListingPage:
    def test_extracts_all_cards(self):
        html = render_listing_page(page=1, ads_per_page=20)
        raw_ads = parse_listing_page(html)
        assert len(raw_ads) == 20

    def test_raw_strings_survive_round_trip(self):
        html = render_listing_page(page=1, ads_per_page=5)
        raw = parse_listing_page(html)[0]
        assert "км" in raw.mileage_raw
        assert raw.ad_id == 1

    def test_validated_ads_have_numeric_fields(self):
        html = render_listing_page(page=2, ads_per_page=10)
        validated = [
            ad for raw in parse_listing_page(html) if (ad := parse_raw_ad(raw))
        ]
        assert validated, "at least some ads must parse"
        for ad in validated:
            assert isinstance(ad.mileage, int)
            assert ad.price > 0

    def test_empty_html_returns_empty_list(self):
        assert parse_listing_page("<html><body></body></html>") == []

    def test_malformed_card_skipped_not_raised(self):
        html = '<div class="listing-item">no data attributes</div>'
        assert parse_listing_page(html) == []
