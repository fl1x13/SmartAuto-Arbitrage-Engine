"""Car Valuation: consumer-friendly fair-price calculator.

No st.form on purpose: widgets inside a form don't rerun until submit,
so the model dropdown would lag one brand behind the brand dropdown.
Live widgets keep brand → model → defaults consistent, and the price
recalculates instantly on any change.
"""

import os
import sys
from datetime import datetime

# Same package-shadowing guard as in app.py (pages run as standalone scripts).
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if not sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.car_card import render_card
from app.data_loader import (
    fetch_ad_for_evaluation,
    load_enriched_data,
    load_model_metrics,
)
from model.predict import (
    evaluate_listing,
    explain_prediction,
    load_model,
    predict_price,
)
from processing.preprocessor import DataPreprocessor
from scraper.autoru import AD_URL_RE

st.set_page_config(page_title="Оценка авто", page_icon="🧮", layout="wide")
st.title("🧮 Сколько стоит машина?")

df = load_enriched_data()
metrics = load_model_metrics()

# --- Evaluate by listing URL ---
st.subheader("🔗 Оценка по ссылке на объявление")
ad_url = st.text_input(
    "Вставьте ссылку на объявление auto.ru — оценим конкретную машину",
    placeholder="https://auto.ru/cars/used/sale/toyota/camry/1234567890-abcdef/",
)
if ad_url.strip():
    url_match = AD_URL_RE.search(ad_url)
    if url_match is None:
        st.error(
            "Это не похоже на ссылку на объявление auto.ru. Нужен адрес вида "
            "`https://auto.ru/cars/used/sale/<марка>/<модель>/<id>-<код>/`."
        )
    else:
        in_db = df[df["ad_id"] == int(url_match["ad_id"])]
        if not in_db.empty:
            row = in_db.iloc[0]
            if not row.get("image_url"):
                # Older DB rows predate photo scraping — pull the photo
                # from the ad page metatags so the card isn't blank.
                fetched = fetch_ad_for_evaluation(ad_url.strip())
                if fetched and fetched.get("image_url"):
                    row = row.copy()
                    row["image_url"] = fetched["image_url"]
            st.caption("Объявление уже есть в нашей базе — оценка из мониторинга.")
        else:
            ad = fetch_ad_for_evaluation(ad_url.strip())
            row = None
            if ad is not None:
                row_df = DataPreprocessor().engineer_features(pd.DataFrame([ad]))
                row = evaluate_listing(row_df, df, load_model())
        if row is None:
            st.error(
                "Не удалось скачать объявление (снято с продажи, опечатка в "
                "ссылке или auto.ru показывает капчу). Попробуйте позже — "
                "или оцените машину вручную через форму ниже."
            )
        else:
            render_card(row, df)

st.divider()

# --- Manual valuation form ---
st.subheader("✍️ Оценка по параметрам")
st.caption(
    "Выберите марку и модель — типичные параметры подставятся сами, "
    "цена пересчитывается мгновенно при любом изменении."
)

RU_FEATURE = {
    "brand": "Марка",
    "model": "Модель",
    "body_type": "Кузов",
    "transmission": "Коробка передач",
    "region": "Город",
    "drive": "Привод",
    "year": "Год выпуска",
    "mileage": "Пробег",
    "engine_volume": "Объём двигателя",
    "horse_power": "Мощность",
    "owners_count": "Число владельцев",
    "car_age": "Возраст",
    "mileage_per_year": "Пробег за год",
    "fuel_type": "Тип топлива",
    "brand_segment": "Сегмент марки",
    "modification": "Модификация",
    "power_density": "Л.с. на литр",
    "owners_per_year": "Владельцев в год",
}


def _pool(frame: pd.DataFrame, column: str) -> list:
    """Distinct non-empty values of a column, sorted."""
    return sorted(v for v in frame[column].dropna().unique() if v)


def _mode_index(options: list, series: pd.Series) -> int:
    """Index of the most frequent value of series within options (0 if absent)."""
    if series.empty:
        return 0
    top = series.mode()
    if top.empty or top.iloc[0] not in options:
        return 0
    return options.index(top.iloc[0])


# --- Step 1: brand / model / year ---
col1, col2, col3 = st.columns(3)
with col1:
    brand = st.selectbox("Марка", options=_pool(df, "brand"), format_func=str.title)
similar = df[df["brand"] == brand]
with col2:
    model_name = st.selectbox(
        "Модель", options=_pool(similar, "model"), format_func=str.title
    )
similar = similar[similar["model"] == model_name]
with col3:
    year = st.number_input(
        "Год выпуска",
        1990,
        datetime.now().year,
        int(similar["year"].median()) if len(similar) else 2018,
    )

# --- Step 2: the rest, prefilled with typical values of this model ---
mileage_default = int(similar["mileage"].median()) if len(similar) else 100_000
with st.expander("Уточнить параметры (заполнены типичные для этой модели)"):
    col4, col5, col6 = st.columns(3)
    with col4:
        mileage = st.number_input(
            "Пробег (км)", 0, 1_500_000, mileage_default, step=5_000
        )
        body_pool = _pool(df, "body_type")
        body_type = st.selectbox(
            "Кузов", body_pool, index=_mode_index(body_pool, similar["body_type"])
        )
    with col5:
        trans_pool = _pool(df, "transmission")
        transmission = st.selectbox(
            "Коробка передач",
            trans_pool,
            index=_mode_index(trans_pool, similar["transmission"]),
        )
        drive_pool = _pool(df, "drive")
        drive = (
            st.selectbox(
                "Привод", drive_pool, index=_mode_index(drive_pool, similar["drive"])
            )
            if drive_pool
            else ""
        )
    with col6:
        engine_volume = st.number_input(
            "Объём двигателя (л) — 0 для электромобиля",
            0.0,
            8.0,
            float(similar["engine_volume"].median()) if len(similar) else 2.0,
            step=0.1,
        )
        horse_power = st.number_input(
            "Мощность (л.с.)",
            30,
            1500,
            int(similar["horse_power"].median()) if len(similar) else 150,
        )
        fuel_pool = ["", "бензин", "дизель", "гибрид", "электро", "газ"]
        fuel_type = st.selectbox(
            "Тип топлива",
            fuel_pool,
            index=_mode_index(fuel_pool, similar["fuel_type"]),
            format_func=lambda v: v or "не указан",
            help="Гибрид и электро ощутимо влияют на рыночную цену "
            "одной и той же модели.",
        )
    region_pool = _pool(df, "region")
    region = st.selectbox(
        "Город", region_pool, index=_mode_index(region_pool, similar["region"])
    )
    owners_count = st.number_input("Число владельцев", 1, 10, 1)

asking_price = st.number_input(
    "💬 Цена из объявления (₽) — необязательно: покажем, выгодна ли она",
    0,
    100_000_000,
    0,
    step=50_000,
)

# --- Prediction (derived features added by the shared pipeline step) ---
row = DataPreprocessor().engineer_features(
    pd.DataFrame(
        [
            {
                "brand": brand,
                "model": model_name,
                "body_type": body_type,
                "transmission": transmission,
                "region": region,
                "drive": drive,
                "year": int(year),
                "mileage": int(mileage),
                "engine_volume": float(engine_volume),
                "horse_power": int(horse_power),
                "owners_count": int(owners_count),
                "fuel_type": fuel_type,
            }
        ]
    )
)

model = load_model()
predicted = float(predict_price(row, model)[0])
mape = metrics.get("mape", 20.0)
low, high = predicted * (1 - mape / 100), predicted * (1 + mape / 100)

st.divider()
col_a, col_b, col_c = st.columns(3)
col_a.metric("Справедливая цена", f"{predicted:,.0f} ₽")
col_b.metric(
    "Диапазон оценки",
    f"{low / 1e6:.2f}–{high / 1e6:.2f} млн ₽",
    help=f"Типичная ошибка модели ±{mape:.0f}% (MAPE на валидации)",
)
if asking_price > 0:
    diff_pct = (predicted - asking_price) / predicted * 100
    col_c.metric(
        "Выгода к цене объявления",
        f"{diff_pct:+.1f}%",
        delta=f"{predicted - asking_price:+,.0f} ₽",
    )
    if diff_pct >= 10:
        st.success(
            "💰 Цена заметно ниже справедливой — похоже на выгодную покупку. "
            "Проверьте историю и состояние: слишком дёшево тоже бывает неспроста."
        )
    elif diff_pct <= -10:
        st.error(
            "⚠️ Просят существенно дороже справедливой цены — есть смысл торговаться."
        )
    else:
        st.info("Цена в пределах рынка.")

# --- Similar listings on sale right now ---
st.subheader(f"Похожие {brand.title()} {model_name.title()} в продаже сейчас")
if len(similar) == 0:
    st.info("Таких моделей в базе пока нет — оценка опирается на близкие модели марки.")
else:
    closest = similar.assign(_dist=(similar["year"] - int(year)).abs()).nsmallest(
        5, "_dist"
    )
    st.dataframe(
        closest[
            ["year", "mileage", "price", "predicted_price", "discount_pct", "url"]
        ].sort_values("price"),
        column_config={
            "year": st.column_config.NumberColumn("Год", format="%d"),
            "mileage": st.column_config.NumberColumn("Пробег", format="%d км"),
            "price": st.column_config.NumberColumn("Цена", format="%d ₽"),
            "predicted_price": st.column_config.NumberColumn(
                "Оценка модели", format="%d ₽"
            ),
            "discount_pct": st.column_config.NumberColumn("Выгода", format="%.1f%%"),
            "url": st.column_config.LinkColumn(
                "Объявление", display_text="Открыть →", width="small"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

# --- Explanation ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Из чего сложилась оценка")
    contributions, base_value, units = explain_prediction(row, model)
    contributions.index = [RU_FEATURE.get(f, f) for f in contributions.index]
    sorted_contrib = contributions.sort_values(key=abs, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=sorted_contrib.values,
            y=sorted_contrib.index,
            orientation="h",
            marker_color=["crimson" if v < 0 else "seagreen" for v in sorted_contrib],
        )
    )
    fig.update_layout(
        title=f"Средняя цена рынка {base_value:,.0f} ₽, поправки за параметры:",
        xaxis_title=f"Влияние на оценку, {units}",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    top = sorted_contrib.iloc[[-1]]
    direction = "повышает" if top.iloc[0] > 0 else "снижает"
    st.caption(
        f"Сильнее всего оценку {direction} **{top.index[0].lower()}** "
        f"({top.iloc[0]:+,.1f} {units})."
    )

with col_right:
    st.subheader("Где ваша цена на фоне рынка")
    if len(similar) < 3:
        st.info("Мало объявлений этой модели для сравнения с рынком.")
    else:
        fig = px.histogram(
            similar,
            x="price",
            nbins=30,
            labels={"price": "Цены объявлений этой модели (₽)"},
            height=380,
        )
        fig.add_vline(
            x=predicted, line_color="seagreen", line_width=3, annotation_text="Оценка"
        )
        if asking_price > 0:
            fig.add_vline(
                x=asking_price,
                line_color="crimson",
                line_dash="dash",
                annotation_text="Цена объявления",
                annotation_position="bottom right",
            )
        st.plotly_chart(fig, use_container_width=True)
        cheaper_share = float((similar["price"] < predicted).mean()) * 100
        st.caption(
            f"{cheaper_share:.0f}% объявлений {brand.title()} "
            f"{model_name.title()} в базе дешевле этой оценки."
        )
