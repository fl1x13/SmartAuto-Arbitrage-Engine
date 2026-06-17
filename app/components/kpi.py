"""KPI metric cards for the dashboard header."""

import pandas as pd
import streamlit as st


def render(df: pd.DataFrame, metrics: dict) -> None:
    """Render KPI cards in a single row.

    Args:
        df: Filtered enriched DataFrame.
        metrics: Dict from :func:`app.data_loader.load_model_metrics`.
    """
    hot = df[(df["deal_grade"] == "🔥 Горячая") & ~df["is_suspicious"]]
    # Potential profit if hot deals were bought and resold at the fair price
    potential_profit = int((hot["predicted_price"] - hot["price"]).clip(lower=0).sum())

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Объявлений",
        value=f"{len(df):,}",
        help="Число объявлений после применения фильтров",
    )
    col2.metric(
        label="Медианная цена",
        value=f"{int(df['price'].median()):,} ₽",
    )
    col3.metric(
        label="🔥 Горячих сделок",
        value=f"{len(hot):,}",
        delta=f"{len(hot) / len(df) * 100:.1f}% выборки" if len(df) else None,
        delta_color="off",
    )
    col4.metric(
        label="Потенциальная выгода",
        value=f"{potential_profit / 1e6:.1f} млн ₽",
        help="Суммарная разница (справедливая − фактическая цена) по горячим сделкам",
    )

    mape = metrics.get("mape")
    col5.metric(
        label="MAPE модели",
        value=f"{mape:.1f}%" if mape is not None else "N/A",
        help="Mean Absolute Percentage Error на валидационной выборке",
    )
