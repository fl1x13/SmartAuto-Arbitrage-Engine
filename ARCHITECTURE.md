# Architecture

Technical guide to the codebase: structure, commands, and the key design decisions behind the scraper, model, and dashboard.

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
├── bot/           # Telegram bot (aiogram): top deals, evaluate-by-link, hot-deal pushes
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

# Launch Telegram bot (token from @BotFather)
TELEGRAM_BOT_TOKEN=<token> python -m bot.main

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
- CatBoost receives `cat_features` directly (brand, model, body_type, transmission, region, drive, fuel_type, brand_segment, modification, generation) — no manual encoding.
- Engineered numeric features (in `DataPreprocessor.engineer_features`, shared by training and the valuation page): car_age, mileage_per_year, power_density (hp/litre, NaN for electric), owners_per_year.
- `fuel_type` and `modification` (full card title with trim/generation) are scraped from auto.ru; rows scraped before these columns existed hold `""` and are backfilled when the scraper sees the ad again.
- `generation` is a compact key (`"iv (w220)"`, `"b6"`, `"ii рест"`) parsed from the title by `scraper.schemas.extract_generation`; it separates generations of the same model (Passat B6 vs B8). `engineer_features` derives it on the fly from `modification` for rows that lack a stored value, so training benefits without a re-scrape.
- Outlier filtering (`DataPreprocessor._filter_outliers`) clips price gently per brand+model (`outlier_lower_q`/`upper_q` = 0.02/0.98) and keeps high-mileage cars (`mileage_upper_q` = 0.99, no low cut): cutting cheap/worn cars used to bias fair-price upward and invent discounts on old cars.

### Arbitrage Score Formula
```
Score = w1 * (P_pred - P_act) / P_pred + w2 * L + w3 * price_drop_share - w4 * mileage_penalty
```
- `w1 = 0.8`, `w2 = 0.2`, `w3 = 0.1`, `w4 = 0.15` (defined in `config.py`)
- `L` = liquidity coefficient from `model.predict.market_liquidity`: data-driven percentile rank of brand+model market depth scaled into `cfg.liquidity_range` (0.85–1.2); manual brand overrides in `cfg.liquidity_map` win.
- `mileage_penalty` = +1 per `cfg.mileage_penalty_scale` km above `cfg.mileage_penalty_start` — keeps cheap worn-out cars from crowding the top deals.
- Predictions are calibrated per mileage bucket **per market segment** (`mileage_calibration` in model_meta.json — `{buckets, global, by_segment}`, computed on the validation split at train time): the raw model overprices high-mileage cars, and depreciation-with-mileage differs sharply by segment (a 300k-km budget Lada vs a 300k-km premium SUV). `apply_mileage_calibration` uses the per-segment factor for a row, falling back to the global bucket factor (and still accepts the legacy flat-list artifact).

### Model Evaluation
Log MAPE and RMSE after every training run. MAPE is surfaced as a KPI in the dashboard.

### Dashboard Caching
All data-loading functions in `app/` use `@st.cache_data` to avoid re-querying the DB on every interaction.

### AI Reliability Check
`bot.reliability.reliability_report(row)` asks an LLM to look up a listing's exact engine/generation weak points, recalls and high-mileage issues, and return a structured verdict (`score` 0–10, `verdict`, `summary`, `weak_points`, `at_mileage`, `checklist`). Provider is picked by whichever key is set (`_provider`, in priority order): `ANTHROPIC_API_KEY` → Claude + web_search (paid); `OPENROUTER_API_KEY` → OpenRouter free tier (no live search, verdict from model knowledge; reachable where Gemini's free tier is geo-blocked); `GEMINI_API_KEY` → Gemini free tier + live Google Search grounding. OpenRouter is preferred over Gemini so a geo-blocked Gemini key left in `.env` doesn't shadow it. It is the single backend shared by the Telegram Mini App (`/api/reliability`) and the dashboard. In the dashboard it surfaces on the "🔍 Поиск сделок" page as the "🔧 Проверка надёжности" button under the selected car, via `app.components.car_card.render_reliability_check` + `app.data_loader.load_reliability_report` (cached). Keys come from a local `.env`, loaded dependency-free by `config._load_dotenv` (real env vars always win); with no key configured the UI shows a setup hint instead of failing. The Gemini call goes through truststore (so it survives a TLS-inspecting proxy) and honors an optional `GEMINI_PROXY` (http(s):// or socks5h://) for the call **only** — the Gemini free tier is geo-restricted (quota 0 from e.g. Russia), so it's routed through an exit-node abroad while the rest of the app stays direct.
