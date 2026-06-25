"""Tests for prediction feedback storage and the calibration it produces."""

import pandas as pd
import pytest
from sqlalchemy import create_engine

from bot.feedback import (
    ALPHA,
    feedback_for_ad,
    prediction_corrections,
    record_feedback,
)
from model.predict import _apply_corrections
from scraper.storage import Base


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _vote(engine, ad_id, verdict, chat_id, brand="bmw", model="x5"):
    record_feedback(
        ad_id, verdict, brand=brand, model=model, chat_id=chat_id, engine=engine
    )


def test_no_votes_means_no_corrections(engine):
    assert prediction_corrections(engine) == {}


def test_re_vote_replaces_previous(engine):
    _vote(engine, ad_id=1, verdict="good", chat_id=42)
    _vote(engine, ad_id=1, verdict="bad", chat_id=42)
    tally = feedback_for_ad(1, chat_id=42, engine=engine)
    assert tally == {"good": 0, "bad": 1, "mine": "bad"}


def test_negative_consensus_lowers_fair_price(engine):
    for chat_id in range(3):
        _vote(engine, ad_id=chat_id, verdict="bad", chat_id=chat_id)
    corr = prediction_corrections(engine)
    factor = corr[("bmw", "x5")]
    # net = (0-3)/(3+4) ≈ -0.4286 → factor = 1 - ALPHA*0.4286
    assert factor == pytest.approx(1 - ALPHA * (3 / 7), abs=1e-3)
    assert 1 - ALPHA <= factor < 1.0


def test_positive_feedback_never_inflates(engine):
    for chat_id in range(5):
        _vote(engine, ad_id=chat_id, verdict="good", chat_id=chat_id)
    # Net positive → no upward correction; segment omitted entirely.
    assert ("bmw", "x5") not in prediction_corrections(engine)


def test_good_votes_offset_bad_votes(engine):
    _vote(engine, ad_id=1, verdict="good", chat_id=1)
    _vote(engine, ad_id=2, verdict="bad", chat_id=2)
    # Balanced → net 0 → factor 1.0 → omitted.
    assert prediction_corrections(engine) == {}


def test_corrections_only_touch_matching_segment(engine):
    _vote(engine, ad_id=1, verdict="bad", chat_id=1, brand="bmw", model="x5")
    corr = prediction_corrections(engine)
    df = pd.DataFrame(
        {
            "brand": ["bmw", "audi"],
            "model": ["x5", "q7"],
            "predicted_price": [2_000_000, 3_000_000],
        }
    )
    adjusted = _apply_corrections(df, corr)
    assert adjusted.iloc[0] < 2_000_000  # bmw x5 demoted
    assert adjusted.iloc[1] == 3_000_000  # audi q7 untouched


def test_apply_corrections_noop_without_corrections():
    df = pd.DataFrame(
        {"brand": ["bmw"], "model": ["x5"], "predicted_price": [2_000_000]}
    )
    pd.testing.assert_series_equal(_apply_corrections(df, None), df["predicted_price"])
