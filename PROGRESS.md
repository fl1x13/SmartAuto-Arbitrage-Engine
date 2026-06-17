# Отчёт о ходе реализации

Файл для восстановления контекста после сбоя. Обновляется после каждого завершённого шага.

## Состояние на старте (2026-06-11)

Выполнено ранее: config.py, scraper/{schemas,storage,seed}.py, processing/preprocessor.py,
model/{train,predict}.py, app/ (полный дашборд). БД засеяна (2000 строк), модель обучена
(MAPE=37.84% на мок-данных). Smoke-скрипт: `scripts/smoke.sh`.

## Журнал шагов

| Шаг | Статус | Детали |
|-----|--------|--------|
| 1.1 Тесты + фикстуры | ✅ | 28 тестов зелёные: tests/{conftest,test_schemas,test_preprocessor,test_model}.py + fixtures/sample_ads.json (25 объявлений, seed=42). Попутно убран FutureWarning в preprocessor.py (явный выбор колонок после groupby). |
| 1.2 parser.py + runner.py | ✅ | scraper/parser.py (aiohttp+BS4+tenacity, semaphore, UA-ротация из config), scraper/demo_source.py (HTML-генератор мок-листингов, 5% битых карточек), scraper/runner.py (CLI --pages). `python -m scraper.runner` логирует "Valid: N, Skipped: M". +tests/test_parser.py (5 тестов, всего 33). Убраны deprecated utcnow() в seed/storage. config.py: +scraper_base_url, +user_agents. |
| 1.3 metrics.py + README + .env.example | ✅ | model/metrics.py (compute_regression_metrics, log_metrics, mape_by_group — для analytics-страницы), train.py использует его. README.md со скриншотом docs/dashboard.png (снят headless через Playwright — установлен pip3 playwright + chromium). .env.example + config.py читает env: DATABASE_URL, SCRAPER_BASE_URL, SCRAPER_DELAY_SECONDS/MAX_PAGES/CONCURRENCY. |
| 2.1 Anomaly detection + confidence + CSV export | ✅ | model/predict.py: detect_anomalies (IsolationForest), confidence (по sample_count: <20 low, <60 medium), deal_grade (🔥/👍/⚠️/—), score векторизован (iterrows убран). table.py: подозрительные исключаются из топа, колонки Оценка+Надёжность, st.download_button CSV топ-100. sidebar.py: кнопка "Обновить данные" (cache_data.clear). +5 тестов (всего 38). Проверено рендером через Playwright (/tmp/check_dashboard.py). |
| 2.2 Multipage (Deal Finder + Model Analytics + SHAP) | ✅ | app/pages/1_🔍_Поиск_сделок.py (расш. фильтры, st.dataframe с выбором строки → SHAP-waterfall) и 2_📊_Аналитика_модели.py (KPI, predicted-vs-actual c диагональю, MAPE по маркам, residuals histogram). SHAP через нативный CatBoost get_feature_importance(type="ShapValues") — без зависимости shap. model/predict.py: +explain_prediction(). Обе страницы и клик по строке проверены Playwright (/tmp/check_pages.py, /tmp/check_shap.py). В каждой странице — sys.path guard (как в app.py). |
| 3.1 Price history tracking | ✅ | storage.py: таблица PriceHistory, save_ads пишет точку при insert и при изменении цены (обновляет raw_ads.price), get_price_dynamics() → first/last price, price_drop_pct, n_price_changes. config: w3=0.1. predict.enrich принимает price_dynamics, score += w3·drop. data_loader: load_price_history(ad_id), merge dynamics. Deal Finder: график динамики цены выбранного объявления + подпись "продавец мотивирован". tests/test_storage.py (5 тестов, всего 43, in-memory SQLite). Двойной прогон runner: 197 точек истории, 97 объявлений с динамикой. |
| 4.1 APScheduler + CI + Docker | ✅ | scraper/scheduler.py (`python -m scraper.scheduler`, стартовый прогон + interval, проверен на 3-сек интервале; pip3 install apscheduler выполнен). config: scrape_interval_hours. ruff.toml (E402 ignore для app/ — sys.path guard) + весь код приведён к ruff clean. .github/workflows/ci.yml: ruff → pytest → smoke scraper → smoke train. Dockerfile (python:3.11-slim, libgomp1 для CatBoost, non-root, healthcheck) + docker-compose.yml (init→dashboard+scheduler, общие volumes) + .dockerignore. requirements.txt выровнен с реально протестированными версиями (pip freeze). Docker-стек собран (398MB) и проверен: `docker compose up` → init обучил модель, dashboard healthy на :8501, рендер-чек прошёл, scheduler скрапит. Docker Desktop стартует через `open -a Docker`. |
| 5.1 Реальный парсер auto.ru | ✅ (2026-06-12) | scraper/autoru.py: auto.ru рендерит карточки server-side → хватает aiohttp+BS4 (Playwright не нужен). Селекторы по regex-префиксам классов (хэш-суффиксы билда меняются). brand/model/ad_id из URL-слага. Ограничения карточки: owners_count=1 (только на детальной странице), published_at=время скрапинга. SSL-фикс: truststore.inject_into_ssl() ДО import aiohttp (на машине TLS-перехватчик; curl работал, aiohttp падал). Антибот: после ~30-40 страниц → showcaptcha, страницы пропускаются (не обходим). Собрано 1015 реальных объявлений Москвы (50 страниц, 2 прогона), синтетика → data/car_market_synthetic.db.bak. Модель переобучена: MAPE=37.1%, RMSE=2.1M (рынок 35К-50М ₽). Smoke изолирован: DATABASE_URL+MODEL_DIR → /tmp (добавлен MODEL_DIR env в config). +7 тестов на реальном HTML-фикстуре (tests/test_autoru.py, fixtures/autoru_sample.html) — всего 50. Скриншот README обновлён (реальные данные). Топ-ссылка проверена живьём: HTTP 200. |
| 5.2 Автообновление: свежесть + регионы + авто-retrain | ✅ (2026-06-12) | Выдача auto.ru сортируется по свежести (sort=cr_date-desc), регион 'rossiya' = вся Россия одной лентой. config: SCRAPER_REGIONS (default rossiya), SCRAPE_INTERVAL_MINUTES (default 15, backcompat с HOURS), RETRAIN_MIN_NEW_ADS (50). Схемы/ORM: +region (город из карточки, MetroListPlace__regionName), аддитивная миграция ALTER TABLE в get_engine. region добавлен в cat_features модели (региональные цены); старым московским записям region backfilled 'Москва'. scheduler: интервал в минутах + авто-переобучение в thread после порога новых объявлений (проверено живьём на 30-сек интервале: scrape→retrain цикл работает). Sidebar: фильтр 'Город'. seed: +region. docker-compose: env-переменные проброшены. Интервал 15 мин обоснован: недооценённая машина уходит за 30-60 мин, ~25 req/час безопасно для антибота. В БД 1255 реальных объявлений из 87 городов. +1 тест (51). Dashboard render-check с фильтром 'Город' пройден. |
| 6.1 Параметр drive + UX-улучшения дашборда | ✅ (2026-06-12) | Новый признак **drive** (привод) end-to-end: schemas, autoru-парсер (spec-строка «...привод»), demo_source/parser/seed, ORM + миграция, cat_features, save_ads бэкфиллит drive у старых записей при повторном скане. Модель переобучена (MAPE=40.3% — drive пока пуст у старых строк). **Sidebar**: фильтры пробег/КПП/кузов/привод/мин.выгода/скрыть подозрительные + expander «⚙️ Формула score» (живые w1/w2/w3 и пороги грейдов через rescore()). **KPI**: горячие сделки + потенциальная выгода (млн ₽). **Таблица топ-15**: st.dataframe c LinkColumn и ProgressColumn вместо HTML. **Новые страницы**: 3_📈_Обзор_рынка (treemap марки/модели, медианные цены по городам, кривая обесценивания, ₽/л.с.) и 4_🧮_Оценка_авто (форма → справедливая цена + SHAP + сравнение с рынком + вердикт по цене продавца). **Фикс грейдов**: deal_grade теперь от discount_pct (порог score включал w2·L≈0.2 → «Хорошая» получали даже переоценённые); пороги confidence ослаблены до 5/20 (на реальных данных все были low → горячих 0). Итог на реальной БД: 109 🔥 / 412 👍. Scatter: линия справедливой цены сглажена rolling-медианой. 52 теста, ruff clean, Playwright-чек всех 5 страниц + сабмит калькулятора. |
| 6.2 Большой датасет + лог-таргет: MAPE 40→16% | ✅ (2026-06-12) | **Данные ×6.3**: обход 16 крупных городов отдельными лентами (15 стр./город, SCRAPER_DELAY=3s, CONCURRENCY=3, пауза 150s между регионами + капча-проба перед каждым) → 1255 → 7884 объявлений, 500 городов, drive заполнен у 6780. Антибот: разовые блоки IP на 5-15 мин (включая connect-timeout — не только showcaptcha), пробу делать тем же aiohttp+truststore стеком (curl имеет другой TLS-отпечаток и может врать; truststore инжектить ДО import aiohttp). Города, попавшие под капчу целиком (Valid: 0), добираются вторым кругом. **Модель**: таргет = log(price) — рынок 35К-50М ₽, RMSE в рублях доминируют дорогие авто; обратное exp в predict_price() (model_meta.json: target_transform=log, predict.py читает; SHAP конвертируется в мультипликативные % — explain_prediction возвращает (вклады, база ₽, units)). Гиперпараметры по 5-fold CV (27 конфигов): lr=0.08, depth=6, l2=3, iterations=3000+es100. Динамика MAPE: 40.3% (база 1255, рубли) → 27.2% (лог) → 24.4% (CV+тюнинг) → **16.4% на 7884 объявлениях** (RMSE 1.50M, val 1348). Тест test_loaded_model_predicts_plausible_prices переведён на predict_price. 52 теста, ruff, Playwright-чек 5 страниц. |
| 6.3 Оценка авто: фикс st.form + потребительский UX | ✅ (2026-06-12) | **Баг**: страница была в st.form — виджеты внутри формы не перерисовываются до submit, поэтому список моделей отставал на одну марку (выбрал Ford — модели от прошлой марки). Форма убрана, всё live: марка → только её модели (у Ford теперь 26 моделей / 215 объявлений после расширения базы). **UX**: типичные параметры модели подставляются сами (медианы year/mileage/engine/hp, мода body/transmission/drive/region по выборке модели), детали спрятаны в expander, оценка пересчитывается мгновенно без кнопки. Диапазон оценки ±MAPE, вердикт по цене объявления простым языком, топ-5 похожих объявлений в продаже с живыми ссылками (ближайшие по году), SHAP-признаки переведены на русский (RU_FEATURE map). check_all_pages.py упрощён (нет сабмита). **HOWTO.md**: шпаргалка запуска/останова/перезапуска одной командой (nohup) наверху, обновлены страницы и цифры. Scheduler запущен фоном (SCRAPER_MAX_PAGES=10), первый цикл: +304 объявления → авторетрейн MAPE=16.24%, в базе 8188+. |

## Как проверить текущее состояние

```bash
python3 -m pytest tests/ -v                       # 52 теста
python3 -m ruff check .                           # линт
bash scripts/smoke.sh    # полный smoke (изолированная БД/модель)
DATABASE_URL="sqlite:////tmp/demo.db" \
  python3 -m scraper.runner --pages 5             # скрапер demo-режим (временная БД!)
SCRAPER_BASE_URL="https://auto.ru" \
  python3 -m scraper.runner --pages 5             # скрапер РЕАЛЬНОГО auto.ru (вся РФ, свежие)
SCRAPER_BASE_URL="https://auto.ru" \
  python3 -m scraper.scheduler                    # автообновление каждые 15 мин + retrain
docker compose up                                 # полный стек (нужен Docker Desktop)
```

## Итог (2026-06-11): ВСЕ ФАЗЫ ЗАВЕРШЕНЫ ✅

Definition of Done из plan.md:
- ✅ pytest — 43 теста зелёные
- ✅ python -m scraper.runner — "Valid: N, Skipped: M"
- ✅ python -m model.train — MAPE логируется (~38% на мок-данных)
- ✅ streamlit run app/app.py — 3 страницы, фильтры, топ-15, SHAP, история цен
- ✅ docker compose up — полный стенд одной командой (проверено)
- ✅ type hints + Google-style docstrings на публичных функциях
- ✅ ни одного print() в продакшн-коде — только logging
- ✅ CI workflow готов (.github/workflows/ci.yml) — проверится при пуше на GitHub

Сверх ТЗ реализовано: demo-источник HTML, anomaly detection (IsolationForest),
deal grades + confidence, SHAP-объяснения, price history + w3-сигнал,
multipage dashboard, CSV-экспорт, ruff, Docker.
