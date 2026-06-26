"""SQLAlchemy models and persistence layer for car listings."""

import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session

from config import cfg
from scraper.schemas import CarAdSchema

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class CarAd(Base):
    """ORM model for a validated car listing."""

    __tablename__ = "raw_ads"

    ad_id = Column(Integer, primary_key=True)
    brand = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer)
    mileage = Column(Integer)
    price = Column(Integer)
    body_type = Column(String)
    engine_volume = Column(Float)
    horse_power = Column(Integer)
    transmission = Column(String)
    owners_count = Column(Integer)
    url = Column(String)
    published_at = Column(DateTime)
    scraped_at = Column(DateTime)
    region = Column(String, default="")
    drive = Column(String, default="")
    image_url = Column(String, default="")
    fuel_type = Column(String, default="")
    modification = Column(String, default="")
    generation = Column(String, default="")
    # 1 once the listing is confirmed sold/removed (its page shows a "продан"
    # banner and drops the price). Sold ads stay in the DB for price history
    # but are excluded from every deal surface. Reset to 0 the moment the ad
    # reappears in a scrape feed — being listed again is proof it is live.
    sold = Column(Integer, default=0)


class PriceHistory(Base):
    """Every observed price point of a listing (for drop-rate signals)."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ad_id = Column(Integer, ForeignKey("raw_ads.ad_id"), index=True, nullable=False)
    price = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, nullable=False)


def get_engine():
    """Create and return a SQLAlchemy engine using the configured DB URL."""
    engine = create_engine(cfg.db_url)
    Base.metadata.create_all(engine)
    _migrate(engine)
    return engine


def _migrate(engine) -> None:
    """Apply additive schema changes to databases created by older versions."""
    from sqlalchemy import inspect, text

    columns = {c["name"] for c in inspect(engine).get_columns("raw_ads")}
    for column in (
        "region", "drive", "image_url", "fuel_type", "modification", "generation",
    ):
        if column not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f"ALTER TABLE raw_ads ADD COLUMN {column} VARCHAR DEFAULT ''"
                    )
                )
            logger.info("Migration: added raw_ads.%s column", column)
    if "sold" not in columns:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE raw_ads ADD COLUMN sold INTEGER DEFAULT 0")
            )
        logger.info("Migration: added raw_ads.sold column")


def save_ads(ads: list[CarAdSchema], engine=None) -> int:
    """Persist validated listings, tracking price changes of known ads.

    New ads are inserted with an initial price-history point. For an ad
    already in the DB with a different price, the current price is updated
    and the new price point is appended to price_history.

    Args:
        ads: List of validated CarAdSchema objects.
        engine: Optional SQLAlchemy engine; creates one from config if not given.

    Returns:
        Number of newly inserted records.
    """
    if engine is None:
        engine = get_engine()

    inserted = updated = 0
    now = _utcnow()
    with Session(engine) as session:
        for ad in ads:
            existing = session.get(CarAd, ad.ad_id)
            if existing is None:
                session.add(CarAd(**ad.model_dump(), scraped_at=now))
                session.add(
                    PriceHistory(ad_id=ad.ad_id, price=ad.price, recorded_at=now)
                )
                inserted += 1
            else:
                # Seen in the feed → it is live; clear any stale sold flag.
                existing.sold = 0
                # Backfill columns on rows scraped before they existed
                for column in (
                    "drive", "image_url", "fuel_type", "modification", "generation",
                ):
                    new_value = getattr(ad, column)
                    if new_value and not getattr(existing, column):
                        setattr(existing, column, new_value)
                if existing.price != ad.price:
                    existing.price = ad.price
                    existing.scraped_at = now
                    session.add(
                        PriceHistory(ad_id=ad.ad_id, price=ad.price, recorded_at=now)
                    )
                    updated += 1
        session.commit()

    logger.info(
        "Saved %d new ads, %d price updates (skipped %d unchanged)",
        inserted,
        updated,
        len(ads) - inserted - updated,
    )
    return inserted


def mark_ads_sold(ad_ids: set[int], engine=None) -> int:
    """Flag the given ads as sold so they drop out of every deal surface.

    Args:
        ad_ids: Listing identifiers confirmed sold/removed.
        engine: Optional SQLAlchemy engine; creates one from config if not given.

    Returns:
        Number of rows newly flagged (already-sold rows are not re-counted).
    """
    if not ad_ids:
        return 0
    if engine is None:
        engine = get_engine()
    flagged = 0
    with Session(engine) as session:
        for ad_id in ad_ids:
            ad = session.get(CarAd, ad_id)
            if ad is not None and not ad.sold:
                ad.sold = 1
                flagged += 1
        session.commit()
    logger.info("Marked %d listings sold", flagged)
    return flagged


def get_price_dynamics(engine=None) -> pd.DataFrame:
    """Aggregate price history into per-ad drop statistics.

    Args:
        engine: Optional SQLAlchemy engine; creates one from config if not given.

    Returns:
        DataFrame indexed by nothing, columns: ad_id, first_price, last_price,
        price_drop_pct (positive = price fell since first observation),
        n_price_changes.
    """
    if engine is None:
        engine = get_engine()

    history = pd.read_sql(
        "SELECT ad_id, price, recorded_at FROM price_history ORDER BY recorded_at",
        engine,
    )
    if history.empty:
        return pd.DataFrame(
            columns=[
                "ad_id", "first_price", "last_price",
                "price_drop_pct", "n_price_changes",
            ]
        )

    grouped = history.groupby("ad_id")["price"]
    dynamics = pd.DataFrame(
        {
            "first_price": grouped.first(),
            "last_price": grouped.last(),
            "n_price_changes": grouped.size() - 1,
        }
    ).reset_index()
    dynamics["price_drop_pct"] = (
        (dynamics["first_price"] - dynamics["last_price"])
        / dynamics["first_price"]
        * 100
    ).round(1)
    return dynamics
