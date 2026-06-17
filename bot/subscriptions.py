"""Persistence for Telegram deal-watch subscriptions."""

import logging
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, String
from sqlalchemy.orm import Session

from scraper.storage import Base, get_engine

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class BotSubscription(Base):
    """One chat subscribed to fresh hot-deal notifications."""

    __tablename__ = "bot_subscriptions"

    chat_id = Column(BigInteger, primary_key=True, autoincrement=False)
    max_price = Column(Integer, nullable=True)  # None = no ceiling
    brand = Column(String, default="")  # "" = any brand
    created_at = Column(DateTime, nullable=False)
    # Watcher cursor: only deals scraped after this moment are pushed
    last_sent_at = Column(DateTime, nullable=False)


def subscribe(chat_id: int, max_price: int | None = None, brand: str = "") -> None:
    """Create or update a subscription for a chat."""
    engine = get_engine()
    with Session(engine) as session:
        sub = session.get(BotSubscription, chat_id)
        now = _utcnow()
        if sub is None:
            session.add(
                BotSubscription(
                    chat_id=chat_id,
                    max_price=max_price,
                    brand=brand,
                    created_at=now,
                    last_sent_at=now,
                )
            )
        else:
            sub.max_price = max_price
            sub.brand = brand
        session.commit()
    logger.info(
        "Subscription saved: chat=%s brand=%r max=%s", chat_id, brand, max_price
    )


def unsubscribe(chat_id: int) -> bool:
    """Remove a chat's subscription. Returns True if one existed."""
    engine = get_engine()
    with Session(engine) as session:
        sub = session.get(BotSubscription, chat_id)
        if sub is None:
            return False
        session.delete(sub)
        session.commit()
    logger.info("Subscription removed: chat=%s", chat_id)
    return True


def all_subscriptions() -> list[BotSubscription]:
    """Every active subscription (detached objects)."""
    engine = get_engine()
    with Session(engine) as session:
        subs = session.query(BotSubscription).all()
        session.expunge_all()
        return subs


def mark_sent(chat_id: int, when: datetime | None = None) -> None:
    """Advance a chat's watcher cursor after a successful push."""
    engine = get_engine()
    with Session(engine) as session:
        sub = session.get(BotSubscription, chat_id)
        if sub is not None:
            sub.last_sent_at = when or _utcnow()
            session.commit()
