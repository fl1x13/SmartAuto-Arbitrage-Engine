"""Tests for the single-ad page parser (og-metatag based)."""

from pathlib import Path

import pytest

from scraper.autoru_ad import is_sold_page, parse_ad_details, parse_ad_page

FIXTURE = Path(__file__).parent / "fixtures" / "autoru_ad_page.html"
URL = "https://auto.ru/cars/used/sale/toyota/land_cruiser/1046891338-4cf01/"


@pytest.fixture(scope="module")
def ad():
    parsed = parse_ad_page(FIXTURE.read_text(), URL)
    assert parsed is not None
    return parsed


class TestParseAdPage:
    def test_identity_from_url_slug(self, ad):
        assert ad.ad_id == 1046891338
        assert ad.brand == "toyota"
        assert ad.model == "land cruiser"

    def test_numeric_fields_from_description(self, ad):
        assert ad.year == 2004
        assert ad.mileage == 218_000
        assert ad.price == 3_200_000
        assert ad.engine_volume == 4.7
        assert ad.horse_power == 235

    def test_spec_strings(self, ad):
        assert ad.transmission == "автомат"
        assert ad.drive == "полный"
        assert ad.body_type == "внедорожник"

    def test_modification_from_title(self, ad):
        assert ad.modification == "Toyota Land Cruiser 100 Series Рестайлинг 1"

    def test_image_url_present(self, ad):
        assert ad.image_url.startswith("https://")

    def test_trim_code_not_mistaken_for_engine_volume(self):
        # BMW-style description: "двигатель 545i AT" — 545 is the trim,
        # the real displacement is absent → must come back as 0 (unknown).
        html = (
            '<html><head>'
            '<meta property="og:title" content="Смотрите, какая машина: '
            'BMW 5 серии 545i IV (E60) 2004 года за 850 000 рублей на Авто.ру!"/>'
            '<meta property="og:description" content="Седан BMW 5 серии 545i '
            'IV (E60) 2004 года, пробег 320 000 км, двигатель 545i AT '
            '(333 л.с.), цвет чёрный за 850 000 рублей."/>'
            "</head></html>"
        )
        ad = parse_ad_page(
            html, "https://auto.ru/cars/used/sale/bmw/5er/1111111111-abc123/"
        )
        assert ad is not None
        assert ad.engine_volume == 0.0
        assert ad.horse_power == 333

    def test_unparseable_page_returns_none(self):
        assert parse_ad_page("<html><head></head></html>", URL) is None

    def test_non_ad_url_returns_none(self):
        html = FIXTURE.read_text()
        assert parse_ad_page(html, "https://auto.ru/moskva/cars/used/") is None


# Minimal copy of the server-rendered card body (BEM classes carry build-hash
# suffixes, so the parser must match stable prefixes) for the reliability check.
_DETAILS_HTML = """
<html><body>
  <ul>
    <li class="CardInfoSummarySimpleRow-CY5TE">
      <span class="CardInfoSummarySimpleRow__label-uJbU8">Состояние</span>
      <span class="CardInfoSummarySimpleRow__content-IIKcj">Не требует ремонта</span>
    </li>
    <li class="CardInfoSummarySimpleRow-CY5TE">
      <span class="CardInfoSummarySimpleRow__label-uJbU8">Владельцы</span>
      <span class="CardInfoSummarySimpleRow__content-IIKcj">2 владельца</span>
    </li>
    <li class="CardInfoSummaryComplexRow-CngDv">
      <div class="CardInfoSummaryComplexRow__cellLabel-i9fmL">
        <div class="CardInfoSummaryComplexRow__cellTitle-S_R1k">Двигатель</div>
        <div class="CardInfoSummaryComplexRow__cellValue-Hka8p">3.6 л, 480 л.с.,
          <a class="Link" href="#">бензин</a></div>
      </div>
    </li>
  </ul>
  <div class="CardComplectationGroups__chips-pm0nm">Комфорт · Обзор</div>
  <div class="CardDescription__textInner-AbCdE">Отличная машина,   один
     владелец.</div>
</body></html>
"""


class TestParseAdDetails:
    @pytest.fixture(scope="class")
    def details(self):
        return parse_ad_details(_DETAILS_HTML)

    def test_simple_spec_rows(self, details):
        assert details["specs"]["Состояние"] == "Не требует ремонта"
        assert details["specs"]["Владельцы"] == "2 владельца"

    def test_complex_row_label_and_value(self, details):
        # cellTitle is the label, cellValue the value (incl. the nested link).
        assert details["specs"]["Двигатель"] == "3.6 л, 480 л.с., бензин"

    def test_complectation_and_description(self, details):
        assert details["complectation"] == "Комфорт · Обзор"
        # whitespace (incl. newlines) is collapsed.
        assert details["description"] == "Отличная машина, один владелец."

    def test_empty_page_yields_empty_fields(self):
        d = parse_ad_details("<html><body></body></html>")
        assert d == {"description": "", "specs": {}, "complectation": ""}


class TestIsSoldPage:
    def test_sold_banner_is_detected(self):
        assert is_sold_page("<html>Этот автомобиль УЖЕ ПРОДАН на auto.ru</html>")

    def test_removed_listing_banner_is_detected(self):
        assert is_sold_page("<html>Объявление снято с продажи</html>")

    def test_live_page_with_price_is_not_sold(self):
        html = (
            '<meta property="og:description" content="Седан Kia Rio 2018 '
            'года, пробег 90 000 км за 900 000 рублей."/>'
        )
        assert not is_sold_page(html)

    def test_og_description_without_price_is_sold(self):
        html = (
            '<meta property="og:description" content="Седан Kia Rio 2018 '
            'года, пробег 90 000 км."/>'
        )
        assert is_sold_page(html)

    def test_captcha_or_error_page_is_kept_live(self):
        # No sold marker and no og:description → ambiguous → never dropped.
        captcha = "<html><body>Подтвердите, что вы не робот</body></html>"
        assert not is_sold_page(captcha)
