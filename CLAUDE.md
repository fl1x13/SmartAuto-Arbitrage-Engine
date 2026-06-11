# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Car Market Arbitrage Analytics** — a production-ready pet project that scrapes used car listings, validates and stores them, trains a CatBoost price prediction model, and surfaces undervalued deals via a Streamlit dashboard.

## Stack

- Python 3.10+, async scraping via Playwright or aiohttp+BeautifulSoup4
- Pydantic v2 for data validation
- Pandas for data manipulation
- SQLAlchemy + SQLite (default) / PostgreSQL
- CatBoostRegressor for price prediction
- Streamlit + Plotly for the dashboard

## Project Structure

```
car_arbitrage/
├── scraper/       # Async scraping logic, Pydantic schemas, DB persistence
├── processing/    # DataPreprocessor class: dedup, outlier removal, feature engineering
├── model/         # CatBoost training pipeline, inference, arbitrage scoring
├── app/           # Streamlit dashboard (app.py)
└── config.py      # Settings: DB URL, model weights (w1=0.8, w2=0.2), liquidity map, paths
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run scraper (collect listings into DB)
python -m scraper.runner

# Preprocess data and engineer features
python -m processing.pipeline

# Train / retrain CatBoost model
python -m model.train

# Launch Streamlit dashboard
streamlit run app/app.py

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_processing.py -v
```

## Architecture Notes

### Data Flow
`scraper → SQLite (raw)` → `processing.DataPreprocessor` → `model.train` → `model.predict` + `calculate_arbitrage_score` → Streamlit dashboard

### Key Conventions
- All logging uses `logging.getLogger(__name__)` at INFO/ERROR levels — no `print()` calls.
- Type hints and Google-style docstrings are required on all public functions/classes.
- `config.py` is the single source of truth for weights, DB paths, liquidity coefficients, and feature lists.
- CatBoost receives `cat_features` directly (brand, model, body_type, transmission) — no manual encoding.

### Arbitrage Score Formula
```
Score = w1 * (P_pred - P_act) / P_pred + w2 * L
```
- `w1 = 0.8`, `w2 = 0.2` (defined in `config.py`)
- `L` = liquidity coefficient (1.2 for Toyota/Kia/Hyundai/BMW, 1.0 otherwise)

### Model Evaluation
Log MAPE and RMSE after every training run. MAPE is surfaced as a KPI in the dashboard.

### Dashboard Caching
All data-loading functions in `app/` use `@st.cache_data` to avoid re-querying the DB on every interaction.
