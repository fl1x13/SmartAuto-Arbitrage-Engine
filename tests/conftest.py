"""Shared fixtures for the test suite."""

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_ads() -> list[dict]:
    """Load the canned listing fixtures as a list of dicts."""
    with open(FIXTURES_DIR / "sample_ads.json") as f:
        return json.load(f)


@pytest.fixture()
def sample_df(sample_ads: list[dict]) -> pd.DataFrame:
    """Fixture ads as a DataFrame, mirroring a raw_ads DB read."""
    df = pd.DataFrame(sample_ads)
    df["published_at"] = pd.to_datetime(df["published_at"])
    if "region" not in df.columns:  # fixtures predate the region column
        df["region"] = "Москва"
    if "drive" not in df.columns:  # fixtures predate the drive column
        df["drive"] = "передний"
    return df
