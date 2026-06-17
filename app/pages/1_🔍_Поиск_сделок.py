"""Deal Finder: advanced filtering and per-listing SHAP explanations."""

import os
import sys

# Same package-shadowing guard as in app.py (pages run as standalone scripts).
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if not sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.car_card import (
    CONFIDENCE_RU,
    render_card,
    render_reliability_check,
    rub,
)
from app.data_loader import load_enriched_data, load_price_history
from model.predict import explain_prediction, load_model

st.set_page_config(page_title="Поиск сделок", page_icon="🔍", layout="wide")
st.title("🔍 Детальный поиск сделок")

df = load_enriched_data()

# --- Advanced filters ---
with st.expander("Фильтры", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        brands = st.multiselect("Марка", sorted(df["brand"].str.title().unique()))
        transmissions = st.multiselect(
            "Коробка передач", sorted(df["transmission"].unique())
        )
    with col2:
        body_types = st.multiselect("Кузов", sorted(df["body_type"].unique()))
        max_owners = st.slider("Макс. владельцев", 1, int(df["owners_count"].max()), 4)
    with col3:
        grades = st.multiselect(
            "Оценка сделки",
            ["🔥 Горячая", "👍 Хорошая", "— Рыночная", "⚠️ Подозрительная"],
            default=["🔥 Горячая", "👍 Хорошая"],
            help=(
                "🔥 Горячая — цена ≥15% ниже рыночной оценки при надёжной "
                "статистике; 👍 Хорошая — ≥5% ниже; — Рыночная — в пределах "
                "рынка; ⚠️ Подозрительная — цена или пробег выглядят "
                "неправдоподобно (причина показывается в карточке ниже)."
            ),
        )
        hide_suspicious = st.checkbox(
            "Скрыть подозрительные",
            value=True,
            help=(
                "Подозрительные — объявления с ценой намного ниже рыночной "
                "оценки или с неправдоподобно низким пробегом: так часто "
                "выглядят мошеннические объявления и авто со скрытыми "
                "проблемами."
            ),
        )

mask = pd.Series(True, index=df.index)
if brands:
    mask &= df["brand"].isin([b.lower() for b in brands])
if transmissions:
    mask &= df["transmission"].isin(transmissions)
if body_types:
    mask &= df["body_type"].isin(body_types)
if grades:
    mask &= df["deal_grade"].isin(grades)
if hide_suspicious:
    mask &= ~df["is_suspicious"]
mask &= df["owners_count"] <= max_owners

filtered = df[mask].sort_values("score", ascending=False)
st.caption(f"Найдено: {len(filtered)} объявлений")

display_cols = [
    "deal_grade", "brand", "model", "year", "mileage", "price",
    "predicted_price", "discount_pct", "score", "confidence", "url",
]
table_df = filtered[display_cols].reset_index(drop=True)
table_df["brand"] = table_df["brand"].str.title()
table_df["model"] = table_df["model"].str.title()
table_df["confidence"] = table_df["confidence"].map(CONFIDENCE_RU)
selection = st.dataframe(
    table_df,
    column_config={
        "deal_grade": "Оценка",
        "brand": "Марка",
        "model": "Модель",
        "year": st.column_config.NumberColumn("Год", format="%d"),
        "mileage": st.column_config.NumberColumn("Пробег", format="%d км"),
        "price": st.column_config.NumberColumn("Цена", format="%d ₽"),
        "predicted_price": st.column_config.NumberColumn(
            "Справедливая цена", format="%d ₽"
        ),
        "discount_pct": st.column_config.NumberColumn("Выгода", format="%.1f%%"),
        "score": st.column_config.NumberColumn("Score", format="%.3f"),
        "confidence": st.column_config.TextColumn(
            "Надёжность",
            help="Сколько похожих объявлений (та же марка и модель) видела "
            "модель: 🔴 <5, 🟡 5–20, 🟢 >20. Чем больше, тем точнее оценка.",
        ),
        "url": st.column_config.LinkColumn(
            "Ссылка", display_text="Открыть →", width="small"
        ),
    },
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    height=420,
)

# --- Detail card + SHAP explanation for the selected row ---
st.divider()
rows = selection.selection.rows if selection and selection.selection else []
if not rows:
    st.info(
        "Выберите строку в таблице, чтобы открыть карточку машины: "
        "фото, характеристики, ссылку на объявление и объяснение цены."
    )
else:
    row = filtered.iloc[rows[0]]
    render_card(row)

    st.divider()
    render_reliability_check(row)

    st.divider()
    st.subheader(
        f"Почему {row['brand'].title()} {row['model'].title()} {row['year']} "
        f"оценён в {rub(row['predicted_price'])}"
    )

    contributions, base_value, units = explain_prediction(
        row.to_frame().T, load_model()
    )
    contributions = contributions.sort_values(key=abs, ascending=True)

    fig = go.Figure(
        go.Bar(
            x=contributions.values,
            y=contributions.index,
            orientation="h",
            marker_color=["crimson" if v < 0 else "seagreen" for v in contributions],
        )
    )
    fig.update_layout(
        title=(
            f"Вклад признаков в цену (базовая цена рынка: {base_value:,.0f} ₽)"
        ),
        xaxis_title=f"Влияние на предсказанную цену, {units}",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    top_factor = contributions.iloc[[-1]]
    direction = "повышает" if top_factor.iloc[0] > 0 else "понижает"
    st.caption(
        f"Сильнее всего цену {direction} признак **{top_factor.index[0]}** "
        f"({top_factor.iloc[0]:+,.1f} {units})."
    )

    # --- Price history of the selected listing ---
    history = load_price_history(int(row["ad_id"]))
    if len(history) > 1:
        st.subheader("Динамика цены объявления")
        fig_hist = go.Figure(
            go.Scatter(
                x=history["recorded_at"],
                y=history["price"],
                mode="lines+markers",
                line=dict(color="royalblue"),
            )
        )
        fig_hist.update_layout(
            xaxis_title="Дата наблюдения", yaxis_title="Цена (₽)", height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        drop = row.get("price_drop_pct", 0)
        if drop > 0:
            st.caption(
                f"📉 Продавец снизил цену на **{drop:.1f}%** с первого наблюдения — "
                "вероятно, мотивирован продать быстрее."
            )
