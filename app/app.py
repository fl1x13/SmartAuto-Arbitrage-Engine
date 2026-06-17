"""Main Streamlit dashboard entry point."""

import os
import sys

# Streamlit inserts the script's directory (app/) at sys.path[0], which makes
# `app.py` shadow the `app` package. Prepend the project root to fix it.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

import streamlit as st

from app.components import charts, kpi, sidebar, table
from app.data_loader import load_enriched_data, load_model_metrics
from model.predict import rescore

st.set_page_config(
    page_title="Авторынок: Арбитражная аналитика",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Система поиска арбитражных сделок на авторынке")
st.caption(
    "Мониторинг рынка подержанных авто · ML-оценка справедливой цены · "
    "Поиск недооценённых объявлений"
)

# --- Data loading ---
df_full = load_enriched_data()
metrics = load_model_metrics()

# --- Sidebar filters + live score weights ---
filters = sidebar.render(df_full)
df_full = rescore(df_full, **filters["weights"])
df = sidebar.apply_filters(df_full, filters)

if df.empty:
    st.warning("Нет данных для выбранных фильтров. Измените параметры поиска.")
    st.stop()

# --- KPI row ---
kpi.render(df, metrics)

st.divider()

# --- Charts ---
col_left, col_right = st.columns([3, 2])
with col_left:
    charts.render_scatter(df)
with col_right:
    charts.render_feature_importance()

st.divider()
charts.render_price_distribution(df)

st.divider()

# --- Top deals table ---
table.render_top_deals(df)

st.markdown(
    "<br><small style='color:grey'>Данные обновляются каждые 5 минут · "
    "Модель переобучается вручную командой <code>python -m model.train</code></small>",
    unsafe_allow_html=True,
)
