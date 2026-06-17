"""Async listing scraper: aiohttp + BeautifulSoup with retry and rate limiting."""

import asyncio
import logging
import random
from datetime import datetime

import truststore

# Use the OS trust store for TLS (must run before aiohttp creates its
# default SSL context at import time). Python's bundled CA list does not
# include locally-trusted proxy/VPN certificates, which breaks HTTPS on
# machines behind TLS-inspecting software even though curl/browsers work.
truststore.inject_into_ssl()

import aiohttp  # noqa: E402
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import cfg
from scraper.schemas import CarAdSchema, RawAdSchema, parse_raw_ad

logger = logging.getLogger(__name__)


def build_url(page: int) -> str:
    """Build a listing page URL for the configured source.

    Args:
        page: 1-based page number.

    Returns:
        Full URL of the listing page.
    """
    return f"{cfg.scraper_base_url.rstrip('/')}/?page={page}"


def build_autoru_urls(max_pages: int) -> list[str]:
    """Build auto.ru listing URLs for all configured regions.

    Pages are sorted freshest-first (``sort=cr_date-desc``), so shallow
    frequent scrapes catch newly posted cars across every region.

    Args:
        max_pages: Pages per region.

    Returns:
        Flat list of listing page URLs.
    """
    return [
        f"https://auto.ru/{region}/cars/used/?sort=cr_date-desc&page={page}"
        for region in cfg.scraper_regions
        for page in range(1, max_pages + 1)
    ]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError),
    reraise=True,
)
async def fetch_page(
    session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore
) -> str:
    """Fetch one page with concurrency limiting, politeness delay, and retries.

    Args:
        session: Shared aiohttp session.
        url: Page URL to fetch.
        sem: Semaphore capping concurrent requests.

    Returns:
        Raw HTML of the page.

    Raises:
        aiohttp.ClientError: After 3 failed attempts (exponential backoff).
    """
    async with sem:
        await asyncio.sleep(cfg.scraper_delay_seconds)
        headers = {"User-Agent": random.choice(cfg.user_agents)}
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.text()


def _text(card, selector: str) -> str:
    node = card.select_one(selector)
    return node.get_text(strip=True) if node else ""


def parse_listing_page(html: str) -> list[RawAdSchema]:
    """Extract raw ads from a listing page HTML.

    Cards that fail schema construction (missing/garbage fields) are
    skipped with an error log instead of aborting the whole page.

    Args:
        html: Listing page HTML.

    Returns:
        List of raw (unparsed-numerics) ad schemas.
    """
    soup = BeautifulSoup(html, "lxml")
    raw_ads: list[RawAdSchema] = []

    for card in soup.select("div.listing-item"):
        try:
            published_node = card.select_one("time.listing-published")
            published_at = datetime.fromisoformat(
                published_node["datetime"] if published_node else ""
            )
            raw_ads.append(
                RawAdSchema(
                    ad_id=int(card["data-ad-id"]),
                    brand=_text(card, ".listing-brand"),
                    model=_text(card, ".listing-model"),
                    year=int(_text(card, ".listing-year")),
                    mileage_raw=_text(card, ".listing-mileage"),
                    price_raw=_text(card, ".listing-price"),
                    body_type=_text(card, ".listing-body"),
                    engine_volume_raw=_text(card, ".listing-engine"),
                    horse_power=int(_text(card, ".listing-hp")),
                    transmission=_text(card, ".listing-transmission"),
                    drive=_text(card, ".listing-drive"),
                    owners_count=int(_text(card, ".listing-owners")),
                    url=card.select_one("a.listing-link")["href"],
                    published_at=published_at,
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Malformed listing card skipped: %s", e)

    return raw_ads


async def _collect_pages_html(max_pages: int) -> list[str]:
    """Fetch (or locally generate, in demo mode) listing pages HTML."""
    if not cfg.scraper_base_url:
        logger.info("No scraper_base_url configured — running in DEMO mode")
        from scraper.demo_source import render_listing_page

        return [render_listing_page(page) for page in range(1, max_pages + 1)]

    if "auto.ru" in cfg.scraper_base_url:
        urls = build_autoru_urls(max_pages)
    else:
        urls = [build_url(page) for page in range(1, max_pages + 1)]

    sem = asyncio.Semaphore(cfg.scraper_concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url, sem) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    pages: list[str] = []
    for url, result in zip(urls, results):
        if isinstance(result, BaseException):
            logger.error("Fetch failed (%s): %s", url, result)
        else:
            pages.append(result)
    return pages


async def run_scraper(max_pages: int | None = None) -> list[CarAdSchema]:
    """Scrape listing pages and return validated ads.

    Args:
        max_pages: Page count to scrape; defaults to cfg.scraper_max_pages.

    Returns:
        Validated ads ready for persistence.
    """
    max_pages = max_pages or cfg.scraper_max_pages
    pages_html = await _collect_pages_html(max_pages)

    if "auto.ru" in cfg.scraper_base_url:
        from scraper.autoru import parse_autoru_page

        parse_fn = parse_autoru_page
    else:
        parse_fn = parse_listing_page

    valid_ads: list[CarAdSchema] = []
    skipped = 0
    for html in pages_html:
        for raw in parse_fn(html):
            validated = parse_raw_ad(raw)
            if validated is not None:
                valid_ads.append(validated)
            else:
                skipped += 1

    logger.info("Scraping done. Valid: %d, Skipped: %d", len(valid_ads), skipped)
    return valid_ads
