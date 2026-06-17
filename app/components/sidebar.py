"""Sidebar filter controls for the Streamlit dashboard."""

import pandas as pd
import streamlit as st

from config import cfg


def _options(df: pd.DataFrame, column: str, title: bool = False) -> list:
    """Distinct non-empty values of a column, sorted (optionally title-cased)."""
    if column not in df.columns:
        return []
    values = [v for v in df[column].dropna().unique() if v]
    if title:
        values = [v.title() for v in values]
    return sorted(set(values))


def render(df: pd.DataFrame) -> dict:
    """Render sidebar filters and return the selected filter values.

    Args:
        df: Full enriched DataFrame (used to derive filter bounds).

    Returns:
        Dict with filter selections plus live scoring weights
        (keys: price_range, year_range, mileage_max, brands, models,
        regions, transmissions, body_types, drives, min_discount,
        hide_suspicious, weights).
    """
    st.sidebar.header("Фильтры")

    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider(
        "Цена (руб.)",
        min_value=price_min,
        max_value=price_max,
        value=(
            int(df["price"].quantile(0.05)),
            int(df["price"].quantile(0.95)),
        ),
        step=50_000,
        format="%d ₽",
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider(
        "Год выпуска",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    mileage_cap = int(df["mileage"].max())
    mileage_max = st.sidebar.slider(
        "Макс. пробег (км)",
        min_value=0,
        max_value=mileage_cap,
        value=mileage_cap,
        step=10_000,
        format="%d км",
    )

    brands = st.sidebar.multiselect("Марка", options=_options(df, "brand", title=True))

    models_pool = (
        df[df["brand"].str.title().isin(brands)]["model"].str.title().unique()
        if brands
        else df["model"].str.title().unique()
    )
    models = st.sidebar.multiselect("Модель", options=sorted(models_pool))

    regions = st.sidebar.multiselect("Город", options=_options(df, "region"))

    with st.sidebar.expander("Доп. параметры"):
        transmissions = st.multiselect(
            "Коробка передач", options=_options(df, "transmission")
        )
        body_types = st.multiselect("Кузов", options=_options(df, "body_type"))
        drives = st.multiselect("Привод", options=_options(df, "drive"))
        min_discount = st.slider(
            "Мин. выгода (%)",
            min_value=0,
            max_value=50,
            value=0,
            help="Скрыть объявления, где скидка к справедливой цене меньше порога",
        )
        hide_suspicious = st.checkbox(
            "Скрыть подозрительные",
            value=False,
            help=(
                "Объявления с ценой намного ниже рыночной оценки или "
                "неправдоподобно низким пробегом — признаки скрытых "
                "проблем или мошенничества."
            ),
        )

    with st.sidebar.expander("⚙️ Формула score"):
        st.caption(
            "Score = w1·скидка + w2·ликвидность + w3·снижение цены "
            "− w4·штраф за пробег"
        )
        w1 = st.slider("w1 — вес скидки", 0.0, 1.0, float(cfg.w1), 0.05)
        w2 = st.slider("w2 — вес ликвидности", 0.0, 1.0, float(cfg.w2), 0.05)
        w3 = st.slider("w3 — вес снижения цены", 0.0, 1.0, float(cfg.w3), 0.05)
        w4 = st.slider(
            "w4 — штраф за большой пробег",
            0.0,
            1.0,
            float(cfg.w4),
            0.05,
            help="Вычитается из score за каждые 150 тыс. км сверх 150 тыс. — "
            "дешёвые «миллионники» не должны вытеснять хорошие сделки.",
        )
        hot_threshold = st.slider(
            "Порог 🔥 Горячая (% выгоды)", 0.0, 50.0, 15.0, 1.0
        )
        good_threshold = st.slider(
            "Порог 👍 Хорошая (% выгоды)", 0.0, 50.0, 5.0, 1.0
        )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Обновить данные", help="Сбросить кэш и перечитать БД"):
        st.cache_data.clear()
        st.rerun()

    return {
        "price_range": price_range,
        "year_range": year_range,
        "mileage_max": mileage_max,
        "brands": [b.lower() for b in brands],
        "models": [m.lower() for m in models],
        "regions": regions,
        "transmissions": transmissions,
        "body_types": body_types,
        "drives": drives,
        "min_discount": min_discount,
        "hide_suspicious": hide_suspicious,
        "weights": {
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "w4": w4,
            "hot_threshold": hot_threshold,
            "good_threshold": good_threshold,
        },
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter selections to the DataFrame.

    Args:
        df: Full enriched DataFrame.
        filters: Dict returned by :func:`render`.

    Returns:
        Filtered DataFrame subset.
    """
    mask = df["price"].between(*filters["price_range"]) & df["year"].between(
        *filters["year_range"]
    )
    mask &= df["mileage"] <= filters["mileage_max"]
    if filters["brands"]:
        mask &= df["brand"].isin(filters["brands"])
    if filters["models"]:
        mask &= df["model"].isin(filters["models"])
    if filters.get("regions") and "region" in df.columns:
        mask &= df["region"].isin(filters["regions"])
    if filters.get("transmissions"):
        mask &= df["transmission"].isin(filters["transmissions"])
    if filters.get("body_types"):
        mask &= df["body_type"].isin(filters["body_types"])
    if filters.get("drives") and "drive" in df.columns:
        mask &= df["drive"].isin(filters["drives"])
    if filters.get("min_discount"):
        mask &= df["discount_pct"] >= filters["min_discount"]
    if filters.get("hide_suspicious"):
        mask &= ~df["is_suspicious"]
    return df[mask]
