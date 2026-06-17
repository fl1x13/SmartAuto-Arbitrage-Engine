"""Model Analytics: residuals, per-brand MAPE, and training metrics."""

import os
import sys

# Same package-shadowing guard as in app.py (pages run as standalone scripts).
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if not sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

import plotly.express as px
import streamlit as st

from app.data_loader import load_enriched_data, load_model_metrics
from model.metrics import mape_by_group

st.set_page_config(page_title="Аналитика модели", page_icon="📊", layout="wide")
st.title("📊 Аналитика качества модели")

df = load_enriched_data()
metrics = load_model_metrics()

# --- Training metrics row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAPE (валидация)", f"{metrics.get('mape', 0):.1f}%")
col2.metric("RMSE (валидация)", f"{metrics.get('rmse', 0):,.0f} ₽")
col3.metric("Обучающая выборка", f"{metrics.get('train_size', 0):,}")
col4.metric("Валидационная выборка", f"{metrics.get('val_size', 0):,}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Предсказанная vs фактическая цена")
    fig = px.scatter(
        df,
        x="price",
        y="predicted_price",
        color="brand",
        opacity=0.5,
        labels={
            "price": "Фактическая цена (₽)",
            "predicted_price": "Предсказанная цена (₽)",
            "brand": "Марка",
        },
        height=460,
    )
    price_max = float(max(df["price"].max(), df["predicted_price"].max()))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=price_max, y1=price_max,
        line=dict(color="grey", dash="dash"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Точки выше диагонали — модель считает авто недооценённым "
        "(потенциальный арбитраж)."
    )

with col_right:
    st.subheader("MAPE по маркам")
    brand_mape = mape_by_group(
        df["price"].to_numpy(),
        df["predicted_price"].to_numpy(),
        df["brand"].to_numpy(),
    )
    fig = px.bar(
        x=list(brand_mape.values()),
        y=[b.title() for b in brand_mape.keys()],
        orientation="h",
        labels={"x": "MAPE (%)", "y": "Марка"},
        color=list(brand_mape.values()),
        color_continuous_scale="RdYlGn_r",
        height=460,
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Марки с высоким MAPE предсказываются хуже — score по ним менее надёжен."
    )

st.divider()

st.subheader("Распределение ошибки (residuals)")
df_res = df.assign(
    residual_pct=(df["predicted_price"] - df["price"]) / df["price"] * 100
)
fig = px.histogram(
    df_res,
    x="residual_pct",
    nbins=60,
    labels={"residual_pct": "Ошибка предсказания (%)"},
    height=380,
)
fig.add_vline(x=0, line_color="crimson", line_dash="dash")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Симметричное распределение вокруг нуля — модель не имеет систематического "
    "смещения. Правый хвост — кандидаты в арбитраж."
)
