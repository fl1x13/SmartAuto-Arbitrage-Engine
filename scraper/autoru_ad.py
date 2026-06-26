"""Fetch and parse a single auto.ru ad page (for evaluate-by-URL).

The ad page itself is a JS app, but its SEO og-metatags are server-rendered
and carry every field we need:

    og:description = "Внедорожник Toyota Land Cruiser 100 Series Рестайлинг 1
                      2004 года, пробег 218 000 км, двигатель 4.7 AT (235 л.с.)
                      4WD, цвет зелёный за 3 200 000 рублей."

Parsing the metatags is far more stable than the page markup.
"""

import asyncio
import logging
import random
import re
from datetime import datetime

import truststore

# Must run before aiohttp builds its default SSL context (see scraper.parser).
truststore.inject_into_ssl()

import aiohttp  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from config import cfg  # noqa: E402
from scraper.autoru import AD_URL_RE  # noqa: E402
from scraper.schemas import CarAdSchema, extract_generation  # noqa: E402

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\s+года")
_MILEAGE_RE = re.compile(r"пробег\s+([\d\s ]+)\s*км")
# Requires a decimal ("двигатель 4.4 AT"): BMW-style descriptions read
# "двигатель 545i AT" where 545 is the trim code, not litres.
_VOLUME_RE = re.compile(r"двигатель\s+(\d\.\d)\b")
_MAX_PLAUSIBLE_VOLUME = 9.0
_HP_RE = re.compile(r"\((\d+)\s*л\.с\.\)")
_PRICE_RE = re.compile(r"за\s+([\d\s ]+)\s*рубл")
_FUEL_RE = re.compile(r"\b(бензин|дизель|гибрид|электро|газ)\b", re.IGNORECASE)
_TITLE_RE = re.compile(r"машина:\s*(.+?)\s+(?:19|20)\d{2}\s+года")

_TRANSMISSIONS = {
    "AMT": "робот",  # before AT: "AMT" contains "AT" as substring
    "CVT": "вариатор",
    "AT": "автомат",
    "MT": "механика",
}


def _to_int(digits: str) -> int:
    return int(re.sub(r"\D", "", digits))


def parse_ad_page(html: str, url: str) -> CarAdSchema | None:
    """Parse one auto.ru ad page into a validated listing.

    Args:
        html: Full HTML of the ad page.
        url: The ad URL (brand, model and ad_id come from its slug).

    Returns:
        CarAdSchema, or None when the page lacks the required metatags
        (removed ad, captcha page, non-car URL).
    """
    url_match = AD_URL_RE.search(url)
    if url_match is None:
        return None

    soup = BeautifulSoup(html, "lxml")

    def _meta(prop: str) -> str:
        node = soup.find("meta", property=prop)
        return node.get("content", "") if node else ""

    description = _meta("og:description")
    title = _meta("og:title")
    if not description:
        return None

    year_match = _YEAR_RE.search(description)
    price_match = _PRICE_RE.search(description)
    hp_match = _HP_RE.search(description)
    if not (year_match and price_match and hp_match):
        return None

    mileage_match = _MILEAGE_RE.search(description)
    volume_match = _VOLUME_RE.search(description)
    fuel_match = _FUEL_RE.search(description)
    title_match = _TITLE_RE.search(title)
    transmission = next(
        (ru for code, ru in _TRANSMISSIONS.items() if f" {code} " in description),
        "",
    )

    image = _meta("twitter:image") or _meta("og:image")

    volume = float(volume_match.group(1)) if volume_match else 0.0
    if not 0.4 <= volume <= _MAX_PLAUSIBLE_VOLUME:
        volume = 0.0  # unknown beats a trim code mistaken for litres

    try:
        return CarAdSchema(
            ad_id=int(url_match["ad_id"]),
            brand=url_match["brand"].strip().lower(),
            model=url_match["model"].replace("_", " ").strip().lower(),
            year=int(year_match.group(1)),
            mileage=_to_int(mileage_match.group(1)) if mileage_match else 0,
            price=_to_int(price_match.group(1)),
            body_type=description.split()[0].lower(),
            engine_volume=volume,
            horse_power=int(hp_match.group(1)),
            transmission=transmission,
            owners_count=1,  # not exposed in the metatags
            url=url,
            published_at=datetime.now(),
            region="",
            drive="полный" if "4WD" in description else "",
            image_url=image,
            fuel_type=fuel_match.group(1).lower() if fuel_match else "",
            modification=title_match.group(1) if title_match else "",
            generation=extract_generation(
                title_match.group(1) if title_match else ""
            ),
        )
    except ValueError as e:
        logger.error("Ad page %s failed validation: %s", url, e)
        return None


# Full browser-like headers: auto.ru tarpits/blocks bare requests to individual
# sale pages (a plain User-Agent times out), so we mimic a real navigation.
_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


async def _fetch(url: str) -> str:
    headers = {"User-Agent": random.choice(cfg.user_agents), **_BROWSER_HEADERS}
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    last_exc: Exception = RuntimeError("no attempt made")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(3):
            try:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.text()
            except (aiohttp.ClientError, TimeoutError, OSError) as e:
                last_exc = e
                await asyncio.sleep(1.5 * (attempt + 1))
    raise last_exc


def fetch_ad_by_url(url: str) -> CarAdSchema | None:
    """Download and parse one auto.ru ad by its URL.

    Args:
        url: Full ad URL (https://auto.ru/cars/used/sale/...).

    Returns:
        Parsed listing, or None on network errors / unparseable page.
    """
    try:
        html = asyncio.run(_fetch(url))
    except (aiohttp.ClientError, TimeoutError, OSError) as e:
        # str(e) is often empty (e.g. bare TimeoutError), so log the type too.
        logger.error(
            "Failed to fetch ad page %s: %s: %s", url, type(e).__name__, e
        )
        return None
    return parse_ad_page(html, url)


# A sold/removed listing keeps its page live but drops the price and shows a
# "продан"/"снято с продажи" banner; a live ad always carries a price in its
# server-rendered og:description (verified against live auto.ru pages).
_SOLD_MARKERS = (
    "уже продан",
    "снято с продаж",
    "объявление снято",
    "больше не размещено",
)


def is_sold_page(html: str) -> bool:
    """Whether an auto.ru ad page shows the listing is sold/removed.

    Conservative by design: returns True only on a positive sold signal, so a
    transient captcha/error page (which carries none of these markers and no
    og:description) is treated as live — a listing is never dropped on a fluke.

    Args:
        html: Full HTML of the ad page.

    Returns:
        True when the page is confidently sold/removed, else False.
    """
    low = html.lower()
    if any(marker in low for marker in _SOLD_MARKERS):
        return True
    # A server-rendered og:description with no price is a removed listing;
    # its absence entirely (captcha/block) is ambiguous, so we keep the ad.
    match = re.search(r'property="og:description"\s+content="([^"]*)"', html)
    return bool(match and match.group(1) and not _PRICE_RE.search(match.group(1)))


async def _collect_sold(
    pairs: list[tuple[int, str]], concurrency: int = 8
) -> set[int]:
    """Fetch each ad page concurrently and return the ad_ids that are sold."""
    semaphore = asyncio.Semaphore(concurrency)
    sold: set[int] = set()

    async def _check(ad_id: int, url: str) -> None:
        async with semaphore:
            try:
                html = await _fetch(url)
            except (aiohttp.ClientError, TimeoutError, OSError):
                return  # fail open: a fetch error never drops a live ad
            if is_sold_page(html):
                sold.add(ad_id)

    await asyncio.gather(*(_check(ad_id, url) for ad_id, url in pairs))
    return sold


def find_sold_ad_ids(pairs: list[tuple[int, str]]) -> set[int]:
    """Return the subset of (ad_id, url) pairs whose listing is sold/removed.

    Runs the page fetches concurrently. Intended for sync/offline contexts
    (the prune job, the scheduler thread), not inside a running event loop.

    Args:
        pairs: (ad_id, url) tuples to verify.

    Returns:
        Set of ad_ids confirmed sold (empty on no input or total fetch failure).
    """
    if not pairs:
        return set()
    return asyncio.run(_collect_sold(pairs))


def _clean_ws(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def parse_ad_details(html: str) -> dict:
    """Extract the full human-readable ad body for the AI reliability check.

    Where :func:`parse_ad_page` pulls only the strict numeric schema from the
    SEO metatags, this scrapes the server-rendered card so the LLM sees
    everything a buyer would: the owner's free-text description, the complete
    spec table (condition, title/customs status, ownership duration, tax, full
    engine/drivetrain line) and the equipment list. All of it makes the
    reliability verdict specific to this exact car, not just the model.

    The card uses BEM class names with build-hash suffixes
    (``CardInfoSummarySimpleRow-CY5TE``); every lookup matches the stable
    prefix, like the listing parser in :mod:`scraper.autoru`.

    Args:
        html: Full HTML of the ad page.

    Returns:
        Dict with keys ``description`` (str), ``specs`` (dict label→value,
        insertion-ordered) and ``complectation`` (str). Missing parts are "".
    """
    soup = BeautifulSoup(html, "lxml")

    desc_el = soup.find(class_=re.compile(r"CardDescription__textInner"))
    description = _clean_ws(desc_el.get_text(" ")) if desc_el else ""

    specs: dict[str, str] = {}
    # Simple rows: a label + a content cell (year, mileage, owners, condition,
    # title, customs, tax, ...).
    for row in soup.find_all(class_=re.compile(r"CardInfoSummarySimpleRow-")):
        label = row.find(class_=re.compile(r"__label"))
        content = row.find(class_=re.compile(r"__content"))
        if label and content:
            key = _clean_ws(label.get_text(" "))
            val = _clean_ws(content.get_text(" "))
            if key and val:
                specs[key] = val
    # Complex rows: cellTitle is the label, cellValue the value (engine,
    # transmission, drive, body, colour, complectation count).
    for row in soup.find_all(class_=re.compile(r"CardInfoSummaryComplexRow-")):
        title = row.find(class_=re.compile(r"cellTitle"))
        value = row.find(class_=re.compile(r"cellValue"))
        if title and value:
            key = _clean_ws(title.get_text(" "))
            val = _clean_ws(value.get_text(" "))
            if key and val:
                specs.setdefault(key, val)

    chips = soup.find(class_=re.compile(r"CardComplectationGroups__chips"))
    complectation = _clean_ws(chips.get_text(" · ")) if chips else ""

    return {
        "description": description,
        "specs": specs,
        "complectation": complectation,
    }


def fetch_ad_details(url: str) -> dict | None:
    """Download one auto.ru ad and extract its full body for the AI check.

    Args:
        url: Full ad URL (https://auto.ru/cars/used/sale/...).

    Returns:
        The rich detail dict from :func:`parse_ad_details`, or None on network
        errors (so the caller can fall back to the stored fields).
    """
    try:
        html = asyncio.run(_fetch(url))
    except (aiohttp.ClientError, TimeoutError, OSError) as e:
        logger.error(
            "Failed to fetch ad details %s: %s: %s", url, type(e).__name__, e
        )
        return None
    return parse_ad_details(html)
