"""User feedback on deal predictions + the calibration it feeds back.

The Mini App lets a user rate the «почему это выгодно» verdict: was the model
right that this listing is a good deal? Those votes are stored here and turned
into a per-(brand, model) calibration factor that nudges the fair-price
estimate, so a segment users keep flagging as over-rated stops dominating the
deal ranking.

The correction is **downward-only**: negative feedback («неверный прогноз»)
demotes an over-optimistic prediction, but positive feedback never inflates a
discount — a correct verdict doesn't justify claiming an even bigger one.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    case,
    func,
)
from sqlalchemy.orm import Session

from scraper.storage import Base, get_engine

logger = logging.getLogger(__name__)

# Calibration tuning ---------------------------------------------------------
# ALPHA is the largest fair-price haircut a segment can earn (15%) when the
# consensus is unanimously "wrong". SMOOTHING adds pseudo-counts so a single
# vote barely moves a segment — it takes several agreeing votes to bite.
ALPHA = 0.15
SMOOTHING = 4.0

GOOD = "good"
BAD = "bad"
VERDICTS = (GOOD, BAD)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PredictionFeedback(Base):
    """One user's verdict on whether a listing's «good deal» call was right."""

    __tablename__ = "prediction_feedback"
    __table_args__ = (UniqueConstraint("ad_id", "chat_id", name="uq_ad_chat"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ad_id = Column(Integer, index=True, nullable=False)
    # 0 = anonymous (auth disabled); otherwise the Telegram user id.
    chat_id = Column(BigInteger, default=0, nullable=False)
    brand = Column(String, default="", index=True)
    model = Column(String, default="", index=True)
    verdict = Column(String, nullable=False)  # "good" | "bad"
    # Snapshot of what the model claimed when the vote was cast (for auditing).
    discount_pct = Column(Float, default=0.0)
    predicted_price = Column(Integer, default=0)
    price = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False)


def record_feedback(
    ad_id: int,
    verdict: str,
    *,
    brand: str = "",
    model: str = "",
    discount_pct: float = 0.0,
    predicted_price: int = 0,
    price: int = 0,
    chat_id: int = 0,
    engine=None,
) -> None:
    """Store (or update) one user's verdict on a listing's deal call.

    Re-voting on the same ad replaces the previous verdict for that user.

    Args:
        ad_id: Listing the vote is about.
        verdict: "good" (model was right) or "bad" (model over-rated it).
        brand/model: Segment keys (stored lowercase) used for calibration.
        discount_pct/predicted_price/price: Snapshot of the claim being judged.
        chat_id: Telegram user id; 0 for anonymous.
        engine: Optional SQLAlchemy engine (defaults to the shared DB).
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    engine = engine or get_engine()
    with Session(engine) as session:
        existing = (
            session.query(PredictionFeedback)
            .filter_by(ad_id=ad_id, chat_id=chat_id)
            .one_or_none()
        )
        if existing is None:
            session.add(
                PredictionFeedback(
                    ad_id=ad_id,
                    chat_id=chat_id,
                    brand=(brand or "").lower(),
                    model=(model or "").lower(),
                    verdict=verdict,
                    discount_pct=discount_pct,
                    predicted_price=predicted_price,
                    price=price,
                    created_at=_utcnow(),
                )
            )
        else:
            existing.verdict = verdict
            existing.brand = (brand or "").lower()
            existing.model = (model or "").lower()
            existing.discount_pct = discount_pct
            existing.predicted_price = predicted_price
            existing.price = price
            existing.created_at = _utcnow()
        session.commit()
    logger.info("Feedback saved: ad=%s verdict=%s chat=%s", ad_id, verdict, chat_id)


def feedback_for_ad(ad_id: int, chat_id: int = 0, engine=None) -> dict:
    """Vote tally for one ad plus this user's own verdict (if any).

    Returns: {"good": int, "bad": int, "mine": "good"|"bad"|None}.
    """
    engine = engine or get_engine()
    with Session(engine) as session:
        rows = (
            session.query(PredictionFeedback.verdict, func.count())
            .filter_by(ad_id=ad_id)
            .group_by(PredictionFeedback.verdict)
            .all()
        )
        counts = {GOOD: 0, BAD: 0}
        for verdict, count in rows:
            counts[verdict] = int(count)
        mine = (
            session.query(PredictionFeedback.verdict)
            .filter_by(ad_id=ad_id, chat_id=chat_id)
            .scalar()
        )
    return {"good": counts[GOOD], "bad": counts[BAD], "mine": mine}


def prediction_corrections(engine=None) -> dict[tuple[str, str], float]:
    """Per-(brand, model) fair-price multipliers learned from feedback.

    For each segment with votes, the net sentiment
    ``net = (good - bad) / (good + bad + SMOOTHING)`` lies in (-1, 1). The
    factor is ``1 + ALPHA * min(0, net)`` — at most 1.0 (no inflation) and at
    least ``1 - ALPHA`` (the worst-case haircut). Segments at exactly 1.0 are
    omitted so the caller only carries real corrections.

    Returns:
        Mapping ``(brand, model) -> factor`` with ``factor < 1.0``.
    """
    engine = engine or get_engine()
    with Session(engine) as session:
        rows = (
            session.query(
                PredictionFeedback.brand,
                PredictionFeedback.model,
                func.sum(case((PredictionFeedback.verdict == GOOD, 1), else_=0)),
                func.sum(case((PredictionFeedback.verdict == BAD, 1), else_=0)),
            )
            .group_by(PredictionFeedback.brand, PredictionFeedback.model)
            .all()
        )
    corrections: dict[tuple[str, str], float] = {}
    for brand, model, goods, bads in rows:
        goods, bads = int(goods or 0), int(bads or 0)
        total = goods + bads
        if total == 0:
            continue
        net = (goods - bads) / (total + SMOOTHING)
        factor = 1.0 + ALPHA * min(0.0, net)
        if factor < 1.0:
            corrections[(brand or "", model or "")] = round(factor, 4)
    return corrections
