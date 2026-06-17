# Техническое Задание: Система предиктивной аналитики и поиска арбитражных сделок на авторынке

## Цель проекта

Создать **production-ready** Python-приложение для резюме, демонстрирующее:
- Написание надёжных асинхронных ETL-пайплайнов
- Строгую валидацию данных через Pydantic
- Feature Engineering и обучение ML-модели (CatBoost)
- Построение интерактивных бизнес-дашбордов на Streamlit + Plotly
- Поиск арбитражных сделок (недооценённые авто для перепродажи)

**Итоговый продукт:** Платформа, которая в фоне собирает объявления с авторынка, рассчитывает справедливую рыночную стоимость каждого авто через ML и выделяет объявления с необоснованно заниженной ценой.

---

## Статус реализации (обновлено 2026-06-11, вечер)

> **ПРОЕКТ ЗАВЕРШЁН.** Все фазы 1–4 реализованы и проверены, подробный журнал — в `PROGRESS.md`.
> Не реализованы (осознанно отложены): Telegram-бот (8.6), PostgreSQL+Alembic (8.8) — кандидаты на следующую итерацию.

## Статус реализации (на утро 2026-06-11, исторический)

### Выполнено ✅
| Файл | Статус |
|------|--------|
| `config.py` | ✅ Полностью реализован |
| `scraper/schemas.py` | ✅ CarAdSchema + валидаторы |
| `scraper/storage.py` | ✅ SQLAlchemy + SQLite, dedup |
| `scraper/seed.py` | ✅ Генератор 2000 мок-объявлений (добавлен сверх ТЗ) |
| `processing/preprocessor.py` | ✅ dedup, outlier removal, feature engineering |
| `model/train.py` | ✅ CatBoost pipeline, метрики, сохранение артефактов |
| `model/predict.py` | ✅ Inference + arbitrage score |
| `app/app.py` | ✅ Streamlit, sys.path fix |
| `app/data_loader.py` | ✅ @st.cache_data |
| `app/components/sidebar.py` | ✅ Фильтры: цена, год, марка, модель |
| `app/components/kpi.py` | ✅ 4 KPI-карточки |
| `app/components/charts.py` | ✅ Scatter, feature importance, box plot |
| `app/components/table.py` | ✅ Топ-15 таблица |

### Не реализовано ❌
| Файл | Что нужно |
|------|-----------|
| `scraper/parser.py` | Асинхронный парсер (aiohttp + BS4) |
| `scraper/runner.py` | Точка входа `python -m scraper.runner` |
| `scraper/scheduler.py` | APScheduler каждые 6 часов |
| `model/metrics.py` | Отдельный модуль для MAPE/RMSE (сейчас в train.py) |
| `tests/test_schemas.py` | 3 теста на валидацию |
| `tests/test_preprocessor.py` | 3 теста на препроцессинг |
| `tests/test_model.py` | 3 теста на predict/score |
| `tests/fixtures/sample_ads.json` | 20-30 фиктивных объявлений для тестов |
| `README.md` | Документация для GitHub |
| `.env.example` | Шаблон переменных окружения |

### Definition of Done — текущий статус
- ❌ `pytest tests/ -v` — 0 тестов (нет тестовых файлов)
- ❌ `python -m scraper.runner` — модуль не существует
- ⚠️ `python -m model.train` — работает, MAPE=37.8% (на мок-данных, цель <20% на реальных)
- ✅ `streamlit run app/app.py` — дашборд работает, фильтры работают, топ-15 отображается
- ⚠️ Docstrings — есть на большинстве публичных функций, неполные в components
- ✅ Нет `print()` — везде `logging`

---

## Архитектура проекта

### Структура директорий

```
car_arbitrage/
│
├── scraper/
│   ├── __init__.py
│   ├── runner.py            # точка входа: запуск парсера (asyncio.run)
│   ├── parser.py            # асинхронная логика парсинга (aiohttp + BS4 или Playwright)
│   ├── schemas.py           # Pydantic-схемы для валидации сырых данных  ✅
│   ├── storage.py           # сохранение в БД через SQLAlchemy              ✅
│   └── seed.py              # генератор мок-данных для разработки            ✅
│
├── processing/
│   ├── __init__.py
│   └── preprocessor.py      # класс DataPreprocessor                         ✅
│
├── model/
│   ├── __init__.py
│   ├── train.py             # пайплайн обучения CatBoost                     ✅
│   ├── predict.py           # инференс и расчёт arbitrage score               ✅
│   └── metrics.py           # MAPE, RMSE, логирование метрик
│
├── app/
│   ├── __init__.py
│   ├── app.py               # главный файл Streamlit                          ✅
│   ├── components/
│   │   ├── sidebar.py       # логика фильтров                                 ✅
│   │   ├── kpi.py           # KPI-блоки метрик                                ✅
│   │   ├── charts.py        # все Plotly-графики                              ✅
│   │   └── table.py         # топ-15 таблица сделок                           ✅
│   └── data_loader.py       # кэшированные запросы к БД (@st.cache_data)      ✅
│
├── tests/
│   ├── test_schemas.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── fixtures/
│       └── sample_ads.json  # 20-30 фиктивных объявлений для тестов
│
├── config.py                # все настройки проекта                           ✅
├── requirements.txt                                                            ✅
├── .env.example
└── README.md
```

---

## Шаг 0. Конфигурация (`config.py`) ✅ ГОТОВО

---

## Шаг 1. Data Collection & Validation ✅ ГОТОВО (кроме parser.py)

---

## Шаг 2. Data Cleaning & Feature Engineering ✅ ГОТОВО

---

## Шаг 3. Machine Learning & Scoring ✅ ГОТОВО

---

## Шаг 4. Business Dashboard ✅ ГОТОВО

---

## Шаг 5. Автоматизация (APScheduler) ❌ НЕ РЕАЛИЗОВАНО

**Файл:** `scraper/scheduler.py`

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

async def scheduled_scrape_job():
    from scraper.runner import run_full_pipeline
    await run_full_pipeline()

def start_scheduler():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_scrape_job, "interval", hours=6)
    scheduler.start()
    return scheduler
```

---

## Шаг 6. Тесты ❌ НЕ РЕАЛИЗОВАНО

### `tests/fixtures/sample_ads.json`
20-30 фиктивных объявлений в формате CarAdSchema.

### `tests/test_schemas.py`
- Тест на корректный парсинг строки "150 000 км" → 150000
- Тест на дроп объявления с невалидной ценой (0 или отрицательной)
- Тест на дроп объявления с годом за пределами диапазона

### `tests/test_preprocessor.py`
- Тест удаления дубликатов по `ad_id`
- Тест что после `fit_transform` нет строк с `car_age < 0`
- Тест что `mileage_per_year` рассчитан корректно

### `tests/test_model.py`
- Тест `calculate_arbitrage_score`: если `p_pred > p_act`, score > 0
- Тест ликвидности: toyota должна давать score выше, чем аналогичное авто неизвестной марки
- Тест загрузки модели: `load_model()` не бросает исключение при наличии файла

---

## Шаг 7. README.md ❌ НЕ РЕАЛИЗОВАНО

README должен содержать:
1. GIF/скриншот дашборда
2. Краткое описание: что делает проект, какую проблему решает
3. Архитектурная схема (ASCII-арт)
4. Инструкция по запуску (seed → train → streamlit)
5. Примеры метрик модели (MAPE, RMSE)
6. Раздел "Tech Stack" с бейджами

---

## Шаг 8. Расширения — сделать проект максимально интересным

Ниже — идеи, упорядоченные по impact/сложности. Реализовывать последовательно.

---

### 8.1 Price History Tracking — отслеживание динамики цен

**Зачем:** продавец, который снижает цену — мотивирован продать быстро. Это дополнительный сигнал к arbitrage score.

**Что сделать:**
- Добавить таблицу `price_history(ad_id, price, recorded_at)` в БД
- В `storage.save_ads` при обновлении существующего объявления писать старую цену в историю
- Добавить колонку `price_drop_pct` (насколько цена снизилась с момента первого появления)
- В формулу arbitrage score добавить `w3 * price_drop_signal`
- На дашборде: новый graph_objects.Scatter по дням для выбранного объявления (hover в топ-таблице)

**Сложность:** средняя | **Impact:** высокий

---

### 8.2 Multipage Dashboard — несколько страниц

**Зачем:** демонстрирует архитектурные навыки Streamlit, делает приложение реальным продуктом.

**Страницы:**
1. `pages/01_market_overview.py` — текущий дашборд (главная)
2. `pages/02_deal_finder.py` — детальный поиск: расширенные фильтры (трансмиссия, объём двигателя, кузов, число владельцев), таблица с сортировкой, кнопка "Экспорт в CSV"
3. `pages/03_model_analytics.py` — метрики модели: learning curve, residual plot (predicted vs actual), feature importance, сравнение MAPE по маркам
4. `pages/04_price_trend.py` — история рынка (после 8.1): как менялась медианная цена по маркам за последние 30 дней

**Сложность:** средняя | **Impact:** высокий

---

### 8.3 SHAP-объяснения для каждого объявления

**Зачем:** превращает "чёрный ящик" в инструмент для аналитика. Уникальная фича для portfolio.

**Что сделать:**
- `pip install shap`
- В `model/predict.py` добавить `explain_prediction(model, row_df) -> dict` через `shap.TreeExplainer`
- На странице Deal Finder: при клике на строку таблицы показывать waterfall chart — почему цена занижена (пробег высокий, год старый, или владельцев много)

```python
import shap

def explain_prediction(model, row_df: pd.DataFrame) -> pd.Series:
    """Return SHAP values for a single row."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_df[cfg.cat_features + cfg.num_features])
    return pd.Series(shap_values[0], index=cfg.cat_features + cfg.num_features)
```

**Сложность:** средняя | **Impact:** высокий (wow-эффект на интервью)

---

### 8.4 Export to CSV/Excel + кнопки действий

**Зачем:** превращает дашборд в рабочий инструмент.

**Что сделать:**
- В `table.render_top_deals` добавить `st.download_button` с CSV топ-сделок
- Добавить кнопку "Обновить данные" (`st.button` → `st.cache_data.clear()`)
- Добавить кнопку "Переобучить модель" с `st.progress` и показом новых метрик после обучения

```python
csv_bytes = top_df.to_csv(index=False).encode("utf-8")
st.download_button("Скачать топ-сделки (CSV)", csv_bytes, "top_deals.csv", "text/csv")
```

**Сложность:** низкая | **Impact:** средний

---

### 8.5 Score confidence + сегментация сделок

**Зачем:** не все сделки с высоким score одинаково надёжны — малая выборка по марке/модели даёт ненадёжные предсказания.

**Что сделать:**
- Добавить `sample_count` (сколько авто данной марки+модели в обучающей выборке) как колонку
- Добавить `confidence` = `"low"/"medium"/"high"` в зависимости от sample_count
- Цветовая кодировка в таблице: красный (low confidence, игнорируй), жёлтый, зелёный
- Добавить `deal_grade` = `"🔥 Горячая"/"👍 Хорошая"/"⚠️ Рискованная"` — комбинация score + confidence

**Сложность:** средняя | **Impact:** средний

---

### 8.6 Telegram-бот для алертов о новых топ-сделках

**Зачем:** демонстрирует интеграцию с внешними API и event-driven архитектуру.

**Что сделать:**
- `pip install python-telegram-bot`
- `notifier/telegram_bot.py` — отправка сообщения с топ-3 сделками при запуске scheduler
- Настройки: `TELEGRAM_TOKEN` и `TELEGRAM_CHAT_ID` через `.env` / `pydantic.BaseSettings`
- В сообщении: марка, модель, год, цена, выгода%, ссылка

```python
async def notify_top_deals(deals: list[dict]) -> None:
    """Send top-3 arbitrage deals to Telegram channel."""
    bot = Bot(token=cfg.telegram_token)
    text = "🚗 Новые топ-сделки:\n\n"
    for d in deals[:3]:
        text += f"• {d['brand']} {d['model']} {d['year']} — {d['price']:,}₽ (−{d['discount_pct']}%)\n"
    await bot.send_message(chat_id=cfg.telegram_chat_id, text=text)
```

**Сложность:** средняя | **Impact:** высокий (уникально для portfolio)

---

### 8.7 Anomaly Detection — детектор подозрительных объявлений

**Зачем:** авто с восстановленным VIN, скрученным пробегом или угнанное часто стоит аномально дёшево. Нужно отличать настоящий arbitrage от ловушки.

**Что сделать:**
- `sklearn.ensemble.IsolationForest` на фичах `(mileage, price, year, engine_volume, horse_power)`
- Добавить колонку `is_suspicious` (bool) в `enrich_with_predictions`
- В топ-таблице: отмечать подозрительные объявления иконкой ⚠️
- На дашборде: KPI "Подозрительных объявлений: N"

```python
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """Return boolean mask of suspicious listings."""
    features = ["mileage", "price", "year", "engine_volume", "horse_power"]
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[features])
    return pd.Series(preds == -1, index=df.index, name="is_suspicious")
```

**Сложность:** низкая | **Impact:** средний

---

### 8.8 PostgreSQL + Alembic миграции

**Зачем:** демонстрирует production-ready подход с версионированием схемы БД.

**Что сделать:**
- Параметризировать `db_url` через `DATABASE_URL` env var (SQLite по умолчанию, PostgreSQL в prod)
- `alembic init alembic` → `alembic revision --autogenerate -m "initial"`
- `alembic/env.py` подхватывает модели из `scraper/storage.py`
- `docker-compose.yml` с PostgreSQL сервисом

**Сложность:** средняя | **Impact:** средний (но ценно для резюме)

---

### 8.9 CI/CD — GitHub Actions

**Зачем:** завершает "production-ready" образ проекта.

**Что сделать:**
- `.github/workflows/ci.yml`:
  - `pytest tests/ -v --cov=. --cov-report=xml`
  - `ruff check .` (линтер)
  - `mypy . --ignore-missing-imports`
- Badge в README: `pytest passing`, `coverage %`, `Python 3.10+`

**Сложность:** низкая | **Impact:** средний

---

### 8.10 Docker — контейнеризация

**Зачем:** `docker compose up` = полностью поднятый стенд. Критично для демо на собеседовании.

**Что сделать:**
- `Dockerfile` (multi-stage): builder + runtime, non-root user
- `docker-compose.yml`:
  - `db` — PostgreSQL
  - `seed` — одноразовый сервис для seed+train
  - `dashboard` — streamlit на порту 8501
- `.env.example` с `DATABASE_URL`, `TELEGRAM_TOKEN` и т.д.

**Сложность:** средняя | **Impact:** высокий

---

## Приоритизированный порядок реализации

### Фаза 1 — Закрыть ТЗ (обязательно)
| # | Задача | Файлы | Оценка |
|---|--------|-------|--------|
| 1 | Тесты + фикстуры | `tests/test_*.py`, `tests/fixtures/sample_ads.json` | 2-3ч |
| 2 | runner.py + заглушка parser.py | `scraper/runner.py`, `scraper/parser.py` | 1-2ч |
| 3 | README.md с скриншотом | `README.md` | 1ч |
| 4 | .env.example | `.env.example` | 15мин |

### Фаза 2 — Сделать интересным (высокий impact)
| # | Задача | Файлы | Оценка |
|---|--------|-------|--------|
| 5 | Multipage dashboard (стр. 2: Deal Finder) | `pages/02_deal_finder.py` | 3-4ч |
| 6 | Export CSV + кнопка обновления | `app/components/table.py` | 30мин |
| 7 | Anomaly detection (is_suspicious) | `model/predict.py` | 1ч |
| 8 | Score confidence + deal_grade | `model/predict.py`, `app/components/table.py` | 2ч |

### Фаза 3 — Wow-эффект (уникальные фичи)
| # | Задача | Файлы | Оценка |
|---|--------|-------|--------|
| 9 | Price history tracking | `scraper/storage.py`, новая таблица | 3-4ч |
| 10 | SHAP-объяснения | `model/predict.py`, `pages/02_deal_finder.py` | 2-3ч |
| 11 | Telegram-бот алерты | `notifier/telegram_bot.py` | 2ч |
| 12 | Model analytics page | `pages/03_model_analytics.py` | 3ч |

### Фаза 4 — Production-ready (для резюме)
| # | Задача | Файлы | Оценка |
|---|--------|-------|--------|
| 13 | APScheduler | `scraper/scheduler.py` | 1ч |
| 14 | CI/CD GitHub Actions | `.github/workflows/ci.yml` | 1ч |
| 15 | Docker + docker-compose | `Dockerfile`, `docker-compose.yml` | 3-4ч |
| 16 | PostgreSQL + Alembic | `alembic/`, `docker-compose.yml` | 3-4ч |

---

## Верификация (Definition of Done — финальная)

- [ ] `pytest tests/ -v --cov` — все тесты зелёные, coverage ≥ 70%
- [ ] `python -m scraper.runner` — логирует "Valid: N, Skipped: M"
- [ ] `python -m model.train` — логирует MAPE (на мок-данных ~38%, на реальных < 20%)
- [ ] `streamlit run app/app.py` — дашборд открывается, все страницы работают
- [ ] `docker compose up` — поднимает полный стенд одной командой
- [ ] Все публичные функции имеют type hints и Google-style docstrings
- [ ] Нет ни одного `print()` — только `logging`
- [ ] CI зелёный на GitHub
