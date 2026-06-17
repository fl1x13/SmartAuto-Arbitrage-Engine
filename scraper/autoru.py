"""Parser for real auto.ru listing pages.

Auto.ru renders listing cards server-side, so plain aiohttp + BeautifulSoup
works — no headless browser needed. CSS classes carry build-hash suffixes
(e.g. ``ListingItemUniversal-BAZaq``), so all lookups match on stable class
prefixes via regex.

Card limitations (fields absent from the listing page):
- ``owners_count`` is only shown on the ad detail page → defaults to 1.
- ``published_at`` is not rendered → set to scrape time.
"""

import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup, Tag

from scraper.schemas import RawAdSchema

logger = logging.getLogger(__name__)

# https://auto.ru/cars/used/sale/{brand}/{model}/{ad_id}-{hash}/
AD_URL_RE = re.compile(
    r"auto\.ru/cars/(?:used|new)/sale/(?P<brand>[^/]+)/(?P<model>[^/]+)/"
    r"(?P<ad_id>\d+)-[0-9a-f]+/?"
)
_ENGINE_RE = re.compile(r"(?P<volume>\d+\.\d+)\s*л(?:,|\b)")
_FUEL_RE = re.compile(r"\b(бензин|дизель|гибрид|электро|газ)\b", re.IGNORECASE)
_HP_RE = re.compile(r"(?P<hp>\d+)\s*л\.с\.")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MILEAGE_RE = re.compile(r"(?P<mileage>[\d\s ]+)\s*км")

_CARD_CLASS = re.compile(r"^ListingItemUniversal-")
_TITLE_LINK_CLASS = re.compile(r"^ListingItemTitle__link")
_SPECS_CLASS = re.compile(r"^ListingItemUniversalSpecs__specs")
_CONDITION_CLASS = re.compile(r"^ListingItemUniversalCondition")
_PRICE_CLASS = re.compile(r"^ListingItemUniversalPrice__title")
_REGION_CLASS = re.compile(r"^MetroListPlace__regionName")
_IMAGE_CLASS = re.compile(r"^LazyImage__image")


def _first_srcset_url(srcset: str) -> str:
    """Return the first URL of a srcset string ("url1 320w, url2 456w")."""
    first = srcset.split(",")[0].strip()
    return first.split()[0] if first else ""


def _parse_image_url(card: Tag) -> str:
    """Extract the first photo URL of a card ("" when absent).

    Lazy-loaded cards below the fold carry the URL in ``data-src`` or only
    in ``srcset`` (on the img or a sibling ``<source>``) instead of ``src``.
    Auto.ru serves images protocol-relative (``//avatars.mds.yandex.net/…``),
    so the scheme is prepended for direct use in the dashboard.
    """
    img = card.find("img", class_=_IMAGE_CLASS)
    src = ""
    if img is not None:
        src = (
            img.get("src")
            or img.get("data-src")
            or _first_srcset_url(img.get("srcset", ""))
        )
    if not src:
        source = card.find("source")
        if source is not None:
            src = _first_srcset_url(source.get("srcset", ""))
    return f"https:{src}" if src.startswith("//") else src


def _parse_card(card: Tag) -> RawAdSchema | None:
    """Extract one listing card into a RawAdSchema.

    Returns None for cards that lack mandatory data (promo blocks,
    price-on-request ads, etc.); the caller logs and skips them.
    """
    link = card.find("a", class_=_TITLE_LINK_CLASS)
    if link is None or not link.get("href"):
        return None

    url_match = AD_URL_RE.search(link["href"])
    if url_match is None:
        return None

    specs_node = card.find("div", class_=_SPECS_CLASS)
    if specs_node is None:
        return None
    spec_lines = [
        ch.get_text(" ", strip=True)
        for ch in specs_node.find_all(True, recursive=False)
    ]
    engine_line = next((s for s in spec_lines if _HP_RE.search(s)), "")
    engine_match = _ENGINE_RE.search(engine_line)
    hp_match = _HP_RE.search(engine_line)
    fuel_match = _FUEL_RE.search(engine_line)
    body_type = next(
        (s for s in spec_lines if s != engine_line and "привод" not in s.lower()), ""
    )
    drive_line = next((s for s in spec_lines if "привод" in s.lower()), "")
    drive = drive_line.lower().replace("привод", "").strip()
    transmission = next(
        (
            s
            for s in spec_lines
            if s.lower() in ("автомат", "механика", "робот", "вариатор")
        ),
        "",
    )

    condition_node = card.find("div", class_=_CONDITION_CLASS)
    condition_text = (
        condition_node.get_text(" | ", strip=True) if condition_node else ""
    )
    year_match = _YEAR_RE.search(condition_text)
    mileage_match = _MILEAGE_RE.search(condition_text)

    price_node = card.find("div", class_=_PRICE_CLASS)
    price_text = price_node.get_text(" ", strip=True) if price_node else ""
    # Keep only the leading amount: "15 990 000 ₽ Справедливая цена" → digits
    price_match = re.search(r"[\d\s ]+₽", price_text)

    if not (year_match and price_match and hp_match):
        return None

    region_node = card.find("span", class_=_REGION_CLASS)
    region = region_node.get_text(strip=True) if region_node else ""

    return RawAdSchema(
        ad_id=int(url_match["ad_id"]),
        brand=url_match["brand"],
        model=url_match["model"].replace("_", " "),
        year=int(year_match.group(1)),
        mileage_raw=mileage_match["mileage"] if mileage_match else "0",
        price_raw=price_match.group(0),
        body_type=body_type,
        engine_volume_raw=engine_match["volume"] if engine_match else "0",
        horse_power=int(hp_match["hp"]),
        transmission=transmission,
        owners_count=1,  # not rendered on listing cards; see module docstring
        url=link["href"],
        published_at=datetime.now(),
        region=region,
        drive=drive,
        image_url=_parse_image_url(card),
        fuel_type=fuel_match.group(1) if fuel_match else "",
        # Full card title ("Mercedes-Benz GLS 450 d II (X167) Рестайлинг")
        # identifies trim and generation — far more than the URL slug.
        modification=link.get_text(" ", strip=True),
    )


def parse_autoru_page(html: str) -> list[RawAdSchema]:
    """Extract raw ads from an auto.ru listing page.

    Args:
        html: Full HTML of an auto.ru listing page.

    Returns:
        List of raw ad schemas; malformed/promo cards are skipped with a log.
    """
    soup = BeautifulSoup(html, "lxml")
    raw_ads: list[RawAdSchema] = []
    seen_ids: set[int] = set()

    for card in soup.find_all("div", class_=_CARD_CLASS):
        try:
            raw = _parse_card(card)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Malformed auto.ru card skipped: %s", e)
            continue
        if raw is None:
            logger.debug("Card without mandatory fields skipped")
            continue
        if raw.ad_id in seen_ids:
            continue
        seen_ids.add(raw.ad_id)
        raw_ads.append(raw)

    return raw_ads
