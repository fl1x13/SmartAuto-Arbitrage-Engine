# AGENTS.md

Working guide for AI coding agents on this repository. Read this first, then
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the system design and conventions.

## ⚠️ Golden rule — commit & push after every change

**After each completed feature or fix, commit and push to `origin/main`.** Do not
batch many changes or leave finished work uncommitted — the repository must always
hold the latest working version.

```bash
git add -A
git commit -m "<clear, descriptive message>"
git push origin main
```

- Verify the change first (run the relevant tests / smoke check) before committing.
- Use clear, descriptive commit messages. Do **not** add AI-tool attribution or
  co-author trailers to commits.
- Never commit secrets (`.env`), the SQLite DBs (`data/*.db`), trained model
  artifacts (`model/artifacts/`), or logs — these are git-ignored; keep them so.

## Project overview

Car-market arbitrage analytics: scrapes auto.ru used-car listings, validates and
stores them, trains a CatBoost fair-price model, and surfaces undervalued deals
via a Streamlit dashboard and a Telegram bot + Mini App. Full detail and design
decisions live in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Two user-facing surfaces — change the right one

There are **two** separate UIs. Confirm which one the user means before editing:

- **Streamlit dashboard** (`app/`) — desktop/local web UI on `:8501`. Multi-page:
  deal finder, market overview, valuation, car picker (`app/pages/`).
- **Telegram Mini App** (`bot/webapp.py` API + `bot/static/index.html` frontend)
  — the phone UI served on `:8050`, exposed publicly via Tailscale Funnel. This
  is what the owner actually uses day-to-day. Its picker, deals and reliability
  check are independent from the Streamlit pages — a change in one does **not**
  appear in the other.

## Commands

```bash
pip install -r requirements.txt          # install deps
python -m scraper.runner                 # scrape listings into the DB
python -m model.train                    # train / retrain the CatBoost model
streamlit run app/app.py                 # launch the Streamlit dashboard (:8501)
python -m bot.webapp                     # Mini App API (:8050)
python -m bot.main                       # Telegram bot
pytest tests/ -q                         # run the test suite
python -m ruff check .                   # lint
bash scripts/smoke.sh                    # seed → train → dashboard health check
```

## Deployment note

The stack auto-starts via launchd (`scripts/autoradar.sh`, `com.autoradar`).
Running processes read config from `.env` **only at start**, so after editing
`.env` or deploying new bot/Mini App code, restart so changes take effect:

```bash
launchctl kickstart -k gui/$(id -u)/com.autoradar   # restart whole stack
# or just the Mini App API: kill the `bot.webapp` process; the supervisor respawns it
```

## Conventions

- Logging via `logging.getLogger(__name__)` — no `print()`.
- Type hints + Google-style docstrings on public functions/classes.
- `config.py` is the single source of truth for weights, paths, and feature lists.
- Dashboard data-loading functions use `@st.cache_data`.
