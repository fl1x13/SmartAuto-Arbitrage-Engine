"""Pydantic schemas for raw and validated car listing data."""

import logging
import re
from datetime import datetime

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Roman numerals I..XV, longest-first so the regex prefers "III" over "I".
_ROMAN = r"(?:XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)"
# Generation marker inside an auto.ru card title, e.g. "III (D4) Рестайлинг",
# "II (G02)", "I". The roman numeral is the generation index; the optional
# parenthetical is the factory body code (W220, G02, D4) — the most precise
# generation key there is; "Рестайлинг" marks a facelift within a generation.
_GEN_RE = re.compile(
    rf"\b({_ROMAN})\b(?:\s*\(([^)]+)\))?(\s*Рестайлинг(?:\s+\d+)?)?",
    re.IGNORECASE,
)
# Fallbacks for titles that name the generation by body/series code instead of
# a roman numeral: "Passat B6", "Land Cruiser Prado 150 Series".
_SERIES_RE = re.compile(r"\b(\d{2,3}\s+series)\b", re.IGNORECASE)
_BODYCODE_RE = re.compile(r"\b([A-Z]{1,2}\d{1,3}[A-Z]?)\b")


def extract_generation(modification: str) -> str:
    """Extract a compact generation key from a full card title.

    The title (stored as ``modification``) already encodes the generation —
    e.g. "Mercedes-Benz GLS 450 d II (X167) Рестайлинг" → "ii (x167) рест".
    Pulling it into its own lower-cardinality feature lets the price model
    separate generations of the same model (Passat B6 vs B8) beyond what the
    raw year implies, and degrades gracefully to "" when nothing is found.

    Args:
        modification: Full listing title, or "" when not scraped.

    Returns:
        Lowercased generation key ("iv (w220)", "ii рест", "b6", "150 series")
        or "" when the title carries no generation marker.
    """
    if not modification:
        return ""
    m = _GEN_RE.search(modification)
    if m:
        parts = [m.group(1).upper()]
        if m.group(2):
            parts.append(f"({m.group(2).strip()})")
        if m.group(3):
            parts.append("рест")
        return " ".join(parts).lower()
    series = _SERIES_RE.search(modification)
    if series:
        return series.group(1).lower()
    code = _BODYCODE_RE.search(modification)
    if code:
        return code.group(1).lower()
    return ""


class RawAdSchema(BaseModel):
    """Raw scraped data before numeric parsing."""

    ad_id: int
    brand: str
    model: str
    year: int
    mileage_raw: str
    price_raw: str
    body_type: str
    engine_volume_raw: str
    horse_power: int
    transmission: str
    owners_count: int
    url: str
    published_at: datetime
    region: str = ""
    drive: str = ""
    image_url: str = ""
    fuel_type: str = ""
    modification: str = ""
    generation: str = ""
    autoru_badge: str | None = None
    autoru_discount_pct: int | None = None


class CarAdSchema(BaseModel):
    """Validated car listing with clean numeric fields."""

    ad_id: int
    brand: str
    model: str
    year: int
    mileage: int
    price: int
    body_type: str
    engine_volume: float
    horse_power: int
    transmission: str
    owners_count: int
    url: str
    published_at: datetime
    region: str = ""
    drive: str = ""
    image_url: str = ""
    fuel_type: str = ""
    modification: str = ""
    generation: str = ""
    # Auto.ru's own price rating (an independent fair-price second opinion):
    # raw badge text and its signed percent (negative = below the estimate).
    autoru_badge: str | None = None
    autoru_discount_pct: int | None = None

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"price must be > 0, got {v}")
        return v

    @field_validator("year")
    @classmethod
    def year_must_be_valid(cls, v: int) -> int:
        current = datetime.now().year
        if not (1990 <= v <= current):
            raise ValueError(f"year {v} out of range [1990, {current}]")
        return v

    @field_validator("mileage")
    @classmethod
    def mileage_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"mileage must be >= 0, got {v}")
        return v


def parse_raw_ad(raw: RawAdSchema) -> "CarAdSchema | None":
    """Convert RawAdSchema to CarAdSchema, returning None on parse failure.

    Args:
        raw: Raw scraped listing data.

    Returns:
        Validated CarAdSchema or None if numeric parsing fails.
    """
    try:
        mileage = int(re.sub(r"\D", "", raw.mileage_raw))
        price = int(re.sub(r"\D", "", raw.price_raw))
        engine_str = re.sub(r"[^\d.]", "", raw.engine_volume_raw)
        engine_volume = float(engine_str) if engine_str else 0.0
        modification = raw.modification.strip()

        return CarAdSchema(
            ad_id=raw.ad_id,
            brand=raw.brand.strip().lower(),
            model=raw.model.strip().lower(),
            year=raw.year,
            mileage=mileage,
            price=price,
            body_type=raw.body_type.strip().lower(),
            engine_volume=engine_volume,
            horse_power=raw.horse_power,
            transmission=raw.transmission.strip().lower(),
            owners_count=raw.owners_count,
            url=raw.url,
            published_at=raw.published_at,
            region=raw.region.strip(),
            drive=raw.drive.strip().lower(),
            image_url=raw.image_url,
            fuel_type=raw.fuel_type.strip().lower(),
            modification=modification,
            generation=extract_generation(modification),
            autoru_badge=raw.autoru_badge,
            autoru_discount_pct=raw.autoru_discount_pct,
        )
    except (ValueError, AttributeError) as e:
        logger.error("Failed to parse ad %s: %s", raw.ad_id, e)
        return None
