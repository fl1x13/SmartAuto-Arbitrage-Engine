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

## Архитектура проекта

### Структура директорий

```
car_arbitrage/
│
├── scraper/
│   ├── __init__.py
│   ├── runner.py            # точка входа: запуск парсера (asyncio.run)
│   ├── parser.py            # асинхронная логика парсинга (aiohttp + BS4 или Playwright)
│   ├── schemas.py           # Pydantic-схемы для валидации сырых данных
│   └── storage.py           # сохранение в БД через SQLAlchemy
│
├── processing/
│   ├── __init__.py
│   └── preprocessor.py      # класс DataPreprocessor
│
├── model/
│   ├── __init__.py
│   ├── train.py             # пайплайн обучения CatBoost
│   ├── predict.py           # инференс и расчёт arbitrage score
│   └── metrics.py           # MAPE, RMSE, логирование метрик
│
├── app/
│   ├── __init__.py
│   ├── app.py               # главный файл Streamlit
│   ├── components/
│   │   ├── sidebar.py       # логика фильтров
│   │   ├── kpi.py           # KPI-блоки метрик
│   │   ├── charts.py        # все Plotly-графики
│   │   └── table.py         # топ-15 таблица сделок
│   └── data_loader.py       # кэшированные запросы к БД (@st.cache_data)
│
├── tests/
│   ├── test_schemas.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── fixtures/
│       └── sample_ads.json  # 20-30 фиктивных объявлений для тестов
│
├── config.py                # все настройки проекта
├── requirements.txt
├── .env.example
└── README.md
```

### Поток данных

```
[Сайт авторынка]
       │
       ▼
scraper/parser.py       ← асинхронный сбор сырых данных
       │
       ▼
scraper/schemas.py      ← Pydantic-валидация (drop невалидных)
       │
       ▼
scraper/storage.py      ← SQLAlchemy → SQLite (таблица raw_ads)
       │
       ▼
processing/preprocessor.py  ← очистка, деdup, feature engineering
       │
       ▼
model/train.py          ← CatBoostRegressor, сохранение модели (.cbm)
       │
       ▼
model/predict.py        ← предсказание цен + arbitrage_score
       │
       ▼
app/app.py              ← Streamlit дашборд
```

---

## Шаг 0. Конфигурация (`config.py`)

**Файл:** `config.py`

Единственный источник истины для всех настроек. Используй `pydantic.BaseSettings` или датаклассы.

```python
# config.py
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent

@dataclass
class Config:
    # База данных
    db_url: str = f"sqlite:///{BASE_DIR}/data/car_market.db"

    # Пути к артефактам модели
    model_path: Path = BASE_DIR / "model" / "artifacts" / "catboost_model.cbm"
    feature_names_path: Path = BASE_DIR / "model" / "artifacts" / "feature_names.json"

    # Веса для arbitrage score
    w1: float = 0.8   # вес ценовой недооценки
    w2: float = 0.2   # вес ликвидности

    # Коэффициенты ликвидности по маркам
    liquidity_map: dict = field(default_factory=lambda: {
        "toyota": 1.2, "kia": 1.2, "hyundai": 1.2, "bmw": 1.2
    })
    default_liquidity: float = 1.0

    # Фичи для модели
    cat_features: list = field(default_factory=lambda: [
        "brand", "model", "body_type", "transmission"
    ])
    num_features: list = field(default_factory=lambda: [
        "year", "mileage", "engine_volume", "horse_power",
        "owners_count", "car_age", "mileage_per_year"
    ])

    # Параметры парсера
    scraper_delay_seconds: float = 1.5   # пауза между запросами
    scraper_max_pages: int = 50
    scraper_concurrency: int = 5         # asyncio.Semaphore

    # Фильтры выбросов (квантили)
    outlier_lower_q: float = 0.05
    outlier_upper_q: float = 0.95

cfg = Config()
```

---

## Шаг 1. Data Collection & Validation

### 1.1 Pydantic-схемы (`scraper/schemas.py`)

**Цель:** строгая типизация и автоматическая очистка входящих данных.

```python
from pydantic import BaseModel, field_validator, model_validator
from datetime import datetime
import re

class RawAdSchema(BaseModel):
    """Схема сырых данных из парсера — до очистки в числа."""
    ad_id: int
    brand: str
    model: str
    year: int
    mileage_raw: str        # "150 000 км"
    price_raw: str          # "1 200 000 ₽"
    body_type: str
    engine_volume_raw: str  # "1.5 л"
    horse_power: int
    transmission: str
    owners_count: int
    url: str
    published_at: datetime

class CarAdSchema(BaseModel):
    """Валидированная схема с числовыми полями."""
    ad_id: int
    brand: str
    model: str
    year: int
    mileage: int            # в км
    price: int              # в рублях
    body_type: str
    engine_volume: float    # в литрах
    horse_power: int
    transmission: str
    owners_count: int
    url: str
    published_at: datetime

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("price must be > 0")
        return v

    @field_validator("year")
    @classmethod
    def year_must_be_valid(cls, v):
        if not (1990 <= v <= datetime.now().year):
            raise ValueError(f"year {v} out of range")
        return v

def parse_raw_ad(raw: RawAdSchema) -> CarAdSchema | None:
    """
    Конвертирует RawAdSchema в CarAdSchema.
    Возвращает None при ошибке парсинга числовых полей (объявление дропается).

    Args:
        raw: Сырые данные объявления.

    Returns:
        Валидированный объект CarAdSchema или None при ошибке.
    """
    try:
        mileage = int(re.sub(r"\D", "", raw.mileage_raw))
        price = int(re.sub(r"\D", "", raw.price_raw))
        engine_volume = float(re.sub(r"[^\d.]", "", raw.engine_volume_raw))
        return CarAdSchema(
            **raw.model_dump(exclude={"mileage_raw", "price_raw", "engine_volume_raw"}),
            mileage=mileage, price=price, engine_volume=engine_volume
        )
    except (ValueError, AttributeError) as e:
        logging.error("Failed to parse ad %s: %s", raw.ad_id, e)
        return None
```

### 1.2 SQLAlchemy-модели (`scraper/storage.py`)

**Цель:** персистентность данных, поддержка PostgreSQL (через изменение `db_url`).

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class CarAd(Base):
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
    scraped_at = Column(DateTime)   # момент парсинга (для трекинга истории)

def save_ads(ads: list[CarAdSchema], engine) -> int:
    """
    Сохраняет список валидированных объявлений. Игнорирует дубликаты по ad_id.

    Returns:
        Количество реально вставленных записей.
    """
    with Session(engine) as session:
        inserted = 0
        for ad in ads:
            existing = session.get(CarAd, ad.ad_id)
            if not existing:
                session.add(CarAd(**ad.model_dump(), scraped_at=datetime.utcnow()))
                inserted += 1
        session.commit()
    return inserted
```

### 1.3 Асинхронный парсер (`scraper/parser.py`)

**Цель:** высокопроизводительный сбор объявлений с контролем скорости.

Реализация строится на `aiohttp` + `BeautifulSoup4` (без браузера — быстрее).
Для сайтов с JS-рендерингом — `Playwright` async API.

**Ключевые паттерны:**
- `asyncio.Semaphore(cfg.scraper_concurrency)` для ограничения параллелизма
- `asyncio.sleep(cfg.scraper_delay_seconds)` между запросами
- Retry-логика через `tenacity` (3 попытки, exponential backoff)
- User-Agent ротация через список строк в `config.py`

```python
async def fetch_page(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str:
    """Загружает страницу с учётом семафора и задержки."""
    async with sem:
        await asyncio.sleep(cfg.scraper_delay_seconds)
        async with session.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}) as resp:
            resp.raise_for_status()
            return await resp.text()

async def parse_listing_page(html: str) -> list[RawAdSchema]:
    """Парсит страницу листинга, возвращает список сырых объявлений."""
    soup = BeautifulSoup(html, "lxml")
    # ... селекторы зависят от конкретного сайта
    ...

async def run_scraper(max_pages: int = cfg.scraper_max_pages) -> list[CarAdSchema]:
    """
    Главная точка входа парсера. Собирает объявления с max_pages страниц.

    Returns:
        Список валидированных объявлений.
    """
    sem = asyncio.Semaphore(cfg.scraper_concurrency)
    valid_ads, skipped = [], 0
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, build_url(page), sem) for page in range(1, max_pages + 1)]
        pages_html = await asyncio.gather(*tasks, return_exceptions=True)
        for html in pages_html:
            if isinstance(html, Exception):
                logging.error("Page fetch failed: %s", html)
                continue
            raw_ads = await parse_listing_page(html)
            for raw in raw_ads:
                validated = parse_raw_ad(raw)
                if validated:
                    valid_ads.append(validated)
                else:
                    skipped += 1
    logging.info("Scraping done. Valid: %d, Skipped: %d", len(valid_ads), skipped)
    return valid_ads
```

---

## Шаг 2. Data Cleaning & Feature Engineering

### 2.1 Класс DataPreprocessor (`processing/preprocessor.py`)

**Цель:** воспроизводимая очистка данных перед обучением модели.

```python
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Очищает и обогащает датасет объявлений о продаже авто.

    Этапы:
        1. Удаление полных дубликатов по ad_id.
        2. Фильтрация выбросов по квантилям в разрезе brand+model.
        3. Feature Engineering: car_age, mileage_per_year, price_segment.
    """

    def __init__(self, lower_q: float = 0.05, upper_q: float = 0.95):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.current_year = datetime.now().year

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полный пайплайн обработки.

        Args:
            df: Сырой датафрейм из БД.

        Returns:
            Очищенный датафрейм с новыми признаками.
        """
        df = self._drop_duplicates(df)
        df = self._filter_outliers(df)
        df = self._engineer_features(df)
        logger.info("Preprocessing done. Shape: %s", df.shape)
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["ad_id"])
        logger.info("Duplicates removed: %d", before - len(df))
        return df

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Отсечение экстремальных цен и пробегов в разрезе brand+model."""
        before = len(df)
        def quantile_filter(group):
            low_p = group["price"].quantile(self.lower_q)
            high_p = group["price"].quantile(self.upper_q)
            low_m = group["mileage"].quantile(self.lower_q)
            high_m = group["mileage"].quantile(self.upper_q)
            return group[
                group["price"].between(low_p, high_p) &
                group["mileage"].between(low_m, high_m)
            ]
        df = df.groupby(["brand", "model"], group_keys=False).apply(quantile_filter)
        logger.info("Outliers removed: %d", before - len(df))
        return df.reset_index(drop=True)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["car_age"] = self.current_year - df["year"]
        df["mileage_per_year"] = df["mileage"] / (df["car_age"] + 1)
        df["price_segment"] = pd.cut(
            df["price"],
            bins=[0, 800_000, 2_500_000, float("inf")],
            labels=["budget", "mid", "premium"]
        )
        return df
```

---

## Шаг 3. Machine Learning & Scoring

### 3.1 Обучение модели (`model/train.py`)

**Цель:** воспроизводимый пайплайн обучения с логированием метрик.

```python
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

def train_model(df: pd.DataFrame) -> tuple[CatBoostRegressor, dict]:
    """
    Обучает CatBoostRegressor на подготовленном датафрейме.

    Args:
        df: Датафрейм после preprocessing.

    Returns:
        Кортеж (обученная модель, словарь метрик).
    """
    feature_cols = cfg.cat_features + cfg.num_features
    X, y = df[feature_cols], df["price"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Индексы категориальных фич в списке feature_cols
    cat_idx = [feature_cols.index(f) for f in cfg.cat_features]

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool)

    # Метрики
    preds = model.predict(X_val)
    metrics = {
        "rmse": float(np.sqrt(np.mean((preds - y_val) ** 2))),
        "mape": float(np.mean(np.abs((y_val - preds) / y_val)) * 100),
        "train_size": len(X_train),
        "val_size": len(X_val),
    }
    logger.info("Model trained. MAPE=%.2f%%, RMSE=%.0f", metrics["mape"], metrics["rmse"])

    # Сохранение
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(cfg.model_path))
    with open(cfg.feature_names_path, "w") as f:
        json.dump(feature_cols, f)
    logger.info("Model saved to %s", cfg.model_path)

    return model, metrics
```

### 3.2 Инференс и Arbitrage Score (`model/predict.py`)

**Формула Arbitrage Score:**

```
Score = w1 * (P_pred - P_act) / P_pred  +  w2 * L
```

- `P_pred` — предсказанная справедливая цена
- `P_act` — фактическая цена объявления
- `L` — коэффициент ликвидности марки (из `config.py`)
- `w1 = 0.8`, `w2 = 0.2`

**Интерпретация Score:**
- `Score > 0.3` — выгодная сделка (цена занижена ≥30% + хорошая ликвидность)
- `Score 0.1–0.3` — умеренно выгодная
- `Score < 0.1` — рыночная или завышенная цена

```python
from catboost import CatBoostRegressor
import pandas as pd

def load_model() -> CatBoostRegressor:
    """Загружает сохранённую модель с диска."""
    model = CatBoostRegressor()
    model.load_model(str(cfg.model_path))
    return model

def calculate_arbitrage_score(
    p_pred: float, p_act: float, brand: str
) -> float:
    """
    Рассчитывает арбитражный индекс привлекательности сделки.

    Args:
        p_pred: Предсказанная справедливая цена.
        p_act: Фактическая цена объявления.
        brand: Марка автомобиля (для коэффициента ликвидности).

    Returns:
        Скор от -inf до ~1.2. Чем выше — тем выгоднее сделка.
    """
    liquidity = cfg.liquidity_map.get(brand.lower(), cfg.default_liquidity)
    price_discount = (p_pred - p_act) / p_pred if p_pred > 0 else 0
    return cfg.w1 * price_discount + cfg.w2 * liquidity

def enrich_with_predictions(df: pd.DataFrame, model: CatBoostRegressor) -> pd.DataFrame:
    """
    Добавляет в датафрейм колонки predicted_price, score, discount_pct.

    Args:
        df: Датафрейм с признаками.
        model: Обученная модель.

    Returns:
        Датафрейм с дополнительными колонками аналитики.
    """
    feature_cols = cfg.cat_features + cfg.num_features
    df["predicted_price"] = model.predict(df[feature_cols]).astype(int)
    df["score"] = df.apply(
        lambda row: calculate_arbitrage_score(row["predicted_price"], row["price"], row["brand"]),
        axis=1
    )
    df["discount_pct"] = ((df["predicted_price"] - df["price"]) / df["predicted_price"] * 100).round(1)
    return df
```

---

## Шаг 4. Business Dashboard (Streamlit UI)

### 4.1 Главный файл (`app/app.py`)

```python
import streamlit as st
from app.components import sidebar, kpi, charts, table
from app.data_loader import load_enriched_data, load_model_metrics

st.set_page_config(
    page_title="Авторынок: Арбитражная аналитика",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Система поиска арбитражных сделок на авторынке")

# Загрузка данных
df = load_enriched_data()
metrics = load_model_metrics()

# Sidebar фильтры
filters = sidebar.render(df)
df_filtered = sidebar.apply_filters(df, filters)

# KPI блоки
kpi.render(df_filtered, metrics)

st.divider()

# Визуализации
col1, col2 = st.columns([2, 1])
with col1:
    charts.render_scatter(df_filtered)
with col2:
    charts.render_feature_importance()

st.divider()

# Топ-15 сделок
table.render_top_deals(df_filtered)
```

### 4.2 Кэшируемые загрузчики данных (`app/data_loader.py`)

```python
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import json

@st.cache_data(ttl=300)  # обновляется каждые 5 минут
def load_enriched_data() -> pd.DataFrame:
    """Загружает данные из БД и обогащает предсказаниями модели."""
    engine = create_engine(cfg.db_url)
    df = pd.read_sql("SELECT * FROM raw_ads", engine)
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    model = load_model()
    df = enrich_with_predictions(df, model)
    return df

@st.cache_data(ttl=3600)
def load_model_metrics() -> dict:
    """Загружает последние метрики обученной модели."""
    metrics_path = cfg.model_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"mape": None, "rmse": None}
```

### 4.3 Sidebar фильтры (`app/components/sidebar.py`)

```python
import streamlit as st
import pandas as pd

def render(df: pd.DataFrame) -> dict:
    """
    Отрисовывает панель фильтров. Возвращает словарь выбранных значений.

    Filters:
        - Диапазон цены (slider)
        - Диапазон года выпуска (slider)
        - Марка (multiselect)
        - Модель (multiselect, зависит от марки)
    """
    st.sidebar.header("Фильтры")
    price_range = st.sidebar.slider(
        "Цена (руб.)", int(df.price.min()), int(df.price.max()),
        (int(df.price.quantile(0.1)), int(df.price.quantile(0.9))),
        step=50_000, format="%d ₽"
    )
    year_range = st.sidebar.slider(
        "Год выпуска", int(df.year.min()), int(df.year.max()),
        (int(df.year.min()), int(df.year.max()))
    )
    brands = st.sidebar.multiselect("Марка", sorted(df.brand.unique()))
    models_available = df[df.brand.isin(brands)]["model"].unique() if brands else df["model"].unique()
    models = st.sidebar.multiselect("Модель", sorted(models_available))
    return {"price_range": price_range, "year_range": year_range, "brands": brands, "models": models}

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = (
        df.price.between(*filters["price_range"]) &
        df.year.between(*filters["year_range"])
    )
    if filters["brands"]:
        mask &= df.brand.isin(filters["brands"])
    if filters["models"]:
        mask &= df.model.isin(filters["models"])
    return df[mask]
```

### 4.4 KPI Блоки (`app/components/kpi.py`)

```python
import streamlit as st
import pandas as pd

def render(df: pd.DataFrame, metrics: dict) -> None:
    """Отрисовывает 4 KPI-карточки в одну строку."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Объявлений в базе", f"{len(df):,}")
    col2.metric("Средняя цена", f"{int(df.price.mean()):,} ₽")
    col3.metric("Медианная цена", f"{int(df.price.median()):,} ₽")
    mape = metrics.get("mape")
    col4.metric("MAPE модели", f"{mape:.1f}%" if mape else "N/A")
```

### 4.5 Plotly-визуализации (`app/components/charts.py`)

#### Scatter plot: Пробег vs Цена

```python
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

def render_scatter(df: pd.DataFrame) -> None:
    """
    Scatter: пробег (X) vs цена (Y), цвет = марка.
    Накладывает трендовую линию справедливой цены (predicted_price).
    """
    st.subheader("Пробег vs Цена по маркам")
    fig = px.scatter(
        df, x="mileage", y="price", color="brand",
        hover_data=["model", "year", "predicted_price", "score"],
        labels={"mileage": "Пробег (км)", "price": "Фактическая цена (₽)"},
        opacity=0.6, height=450
    )
    # Трендовая линия по predicted_price
    df_sorted = df.sort_values("mileage")
    fig.add_trace(go.Scatter(
        x=df_sorted["mileage"], y=df_sorted["predicted_price"],
        mode="lines", name="Справедливая цена (ML)",
        line=dict(color="red", width=2, dash="dash")
    ))
    st.plotly_chart(fig, use_container_width=True)
```

#### Feature Importance

```python
def render_feature_importance() -> None:
    """Горизонтальная столбчатая диаграмма важности признаков CatBoost."""
    st.subheader("Важность признаков")
    model = load_model()
    importances = model.get_feature_importance()
    feature_names = cfg.cat_features + cfg.num_features
    fig = px.bar(
        x=importances, y=feature_names,
        orientation="h", labels={"x": "Importance", "y": "Feature"},
        color=importances, color_continuous_scale="Blues"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
```

### 4.6 Топ-15 таблица (`app/components/table.py`)

```python
import streamlit as st
import pandas as pd

def render_top_deals(df: pd.DataFrame) -> None:
    """
    Отображает топ-15 объявлений с наибольшим arbitrage score.
    Колонки: Марка, Модель, Год, Факт. Цена, Предсказанная Цена, Выгода %, Ссылка.
    """
    st.subheader("🏆 Топ-15 лучших арбитражных сделок")
    top = (
        df.nlargest(15, "score")
          [["brand", "model", "year", "price", "predicted_price", "discount_pct", "url"]]
          .rename(columns={
              "brand": "Марка", "model": "Модель", "year": "Год",
              "price": "Факт. цена (₽)", "predicted_price": "Справедливая цена (₽)",
              "discount_pct": "Выгода (%)", "url": "Ссылка"
          })
    )
    # Кликабельные ссылки
    top["Ссылка"] = top["Ссылка"].apply(lambda u: f'<a href="{u}" target="_blank">Открыть</a>')
    st.write(top.to_html(escape=False, index=False), unsafe_allow_html=True)
```

---

## Шаг 5. Автоматизация (Bonus: APScheduler)

**Файл:** `scraper/scheduler.py`

Для демонстрации production-ready подхода — запуск скрапера по расписанию.

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import logging

logger = logging.getLogger(__name__)

async def scheduled_scrape_job():
    """Задача для планировщика: скрапинг + сохранение в БД."""
    from scraper.runner import run_full_pipeline
    logger.info("Scheduled scrape started")
    await run_full_pipeline()

def start_scheduler():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_scrape_job, "interval", hours=6)
    scheduler.start()
    logger.info("Scheduler started. Scraping every 6 hours.")
    return scheduler
```

---

## Шаг 6. Тесты (`tests/`)

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

## Шаг 7. Зависимости (`requirements.txt`)

```
# Core
pandas==2.2.2
numpy==1.26.4
sqlalchemy==2.0.30
pydantic==2.7.1

# Scraping
aiohttp==3.9.5
beautifulsoup4==4.12.3
lxml==5.2.2
playwright==1.44.0
tenacity==8.3.0

# ML
catboost==1.2.5
scikit-learn==1.5.0

# Dashboard
streamlit==1.35.0
plotly==5.22.0

# Scheduling (optional)
apscheduler==3.10.4

# Dev
pytest==8.2.2
pytest-asyncio==0.23.7
```

---

## Шаг 8. README.md (для GitHub)

README должен содержать:
1. **GIF/скриншот** дашборда (снять после запуска)
2. Краткое описание: что делает проект, какую проблему решает
3. Архитектурная схема (ASCII-арт как выше)
4. Инструкция по запуску (3 команды: `pip install -r requirements.txt`, `python -m scraper.runner`, `streamlit run app/app.py`)
5. Примеры метрик модели (MAPE, RMSE)
6. Раздел "Tech Stack" с бейджами

---

## Порядок реализации (рекомендуемый)

| # | Задача | Файлы |
|---|--------|-------|
| 1 | Конфигурация | `config.py` |
| 2 | SQLAlchemy модели + инит БД | `scraper/storage.py` |
| 3 | Pydantic-схемы + парсинг строк | `scraper/schemas.py` |
| 4 | Мок-данные для разработки (JSON) | `tests/fixtures/sample_ads.json` |
| 5 | DataPreprocessor | `processing/preprocessor.py` |
| 6 | Обучение CatBoost | `model/train.py` |
| 7 | Инференс + arbitrage score | `model/predict.py` |
| 8 | Streamlit дашборд | `app/` |
| 9 | Асинхронный парсер | `scraper/parser.py` |
| 10 | Тесты | `tests/` |
| 11 | Scheduler | `scraper/scheduler.py` |
| 12 | README + скриншоты | `README.md` |

---

## Верификация (Definition of Done)

- [ ] `pytest tests/ -v` — все тесты зелёные
- [ ] `python -m scraper.runner` — собирает ≥100 объявлений, логирует "Valid: N, Skipped: M"
- [ ] `python -m model.train` — логирует MAPE < 20% (при достаточном датасете)
- [ ] `streamlit run app/app.py` — дашборд открывается, фильтры работают, топ-15 отображается
- [ ] Все публичные функции имеют type hints и Google-style docstrings
- [ ] Нет ни одного `print()` — только `logging`
