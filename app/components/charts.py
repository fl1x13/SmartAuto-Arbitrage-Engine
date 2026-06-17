"""Plotly chart components for the dashboard."""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import cfg
from model.predict import load_model

logger = logging.getLogger(__name__)


def render_scatter(df: pd.DataFrame) -> None:
    """Scatter plot of mileage vs price, colored by brand, with predicted-price trend.

    Args:
        df: Filtered enriched DataFrame.
    """
    st.subheader("Пробег vs Цена по маркам")

    fig = px.scatter(
        df,
        x="mileage",
        y="price",
        color="brand",
        hover_data=["model", "year", "predicted_price", "score"],
        labels={
            "mileage": "Пробег (км)",
            "price": "Факт. цена (₽)",
            "brand": "Марка",
        },
        opacity=0.55,
        height=460,
    )

    # Rolling median keeps the fair-price trend readable: raw predictions
    # jump between brands at neighbouring mileage values.
    df_sorted = df.sort_values("mileage")
    window = max(len(df_sorted) // 20, 5)
    trend = (
        df_sorted["predicted_price"]
        .rolling(window, center=True, min_periods=1)
        .median()
    )
    fig.add_trace(
        go.Scatter(
            x=df_sorted["mileage"],
            y=trend,
            mode="lines",
            name="Справедливая цена (ML, медианный тренд)",
            line=dict(color="crimson", width=2, dash="dash"),
        )
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance() -> None:
    """Horizontal bar chart of CatBoost feature importances.

    Falls back gracefully if the model is not yet trained.
    """
    st.subheader("Важность признаков")
    try:
        model = load_model()
        importances = model.get_feature_importance()
        feature_names = cfg.cat_features + cfg.num_features
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance")

        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Важность", "feature": "Признак"},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info("Модель ещё не обучена. Запустите `python -m model.train`.")


def render_price_distribution(df: pd.DataFrame) -> None:
    """Box plots of price distribution by brand.

    Args:
        df: Filtered enriched DataFrame.
    """
    st.subheader("Распределение цен по маркам")
    fig = px.box(
        df,
        x="brand",
        y="price",
        color="brand",
        labels={"brand": "Марка", "price": "Цена (₽)"},
        height=420,
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
