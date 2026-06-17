"""Top-15 arbitrage deals table component with CSV export."""

import pandas as pd
import streamlit as st

_EXPORT_COLS = [
    "brand", "model", "year", "mileage", "price", "predicted_price",
    "discount_pct", "score", "deal_grade", "confidence", "url",
]


def render_top_deals(df: pd.DataFrame) -> None:
    """Display the 15 most attractive arbitrage listings with grades.

    Suspicious listings (anomaly-flagged) are excluded from the top but
    counted in the caption. A download button exports the filtered deals.

    Args:
        df: Filtered enriched DataFrame with score, deal_grade, confidence.
    """
    st.subheader("Топ-15 лучших арбитражных сделок")

    suspicious_count = int(df["is_suspicious"].sum())
    # Low-confidence rows (<6 comparable listings) have unreliable fair-price
    # estimates that invent large fake discounts on rare cars — keep them out
    # of the headline ranking, consistent with the bot's top deals.
    low_conf_count = int((~df["is_suspicious"] & (df["confidence"] == "low")).sum())
    clean = df[~df["is_suspicious"] & (df["confidence"] != "low")]
    top = clean.nlargest(15, "score")

    if suspicious_count:
        st.caption(
            f"⚠️ {suspicious_count} подозрительных объявлений исключено из топа "
            "(цена заметно ниже рыночной оценки или неправдоподобный пробег — "
            "признаки скрытых проблем или мошенничества)"
        )
    if low_conf_count:
        st.caption(
            f"ℹ️ {low_conf_count} объявлений с низкой достоверностью оценки "
            "(мало похожих машин на рынке) исключено из топа"
        )

    display = top[
        ["deal_grade", "brand", "model", "year", "mileage", "price",
         "predicted_price", "discount_pct", "confidence", "url"]
    ].reset_index(drop=True)
    display["brand"] = display["brand"].str.title()
    display["model"] = display["model"].str.title()
    display["confidence"] = display["confidence"].map(
        {"low": "🔴 низкая", "medium": "🟡 средняя", "high": "🟢 высокая"}
    )

    max_discount = max(float(display["discount_pct"].max()), 1.0)
    st.dataframe(
        display,
        column_config={
            "deal_grade": st.column_config.TextColumn("Оценка", width="small"),
            "brand": "Марка",
            "model": "Модель",
            "year": st.column_config.NumberColumn("Год", format="%d"),
            "mileage": st.column_config.NumberColumn("Пробег", format="%d км"),
            "price": st.column_config.NumberColumn("Факт. цена", format="%d ₽"),
            "predicted_price": st.column_config.NumberColumn(
                "Справедливая цена", format="%d ₽"
            ),
            "discount_pct": st.column_config.ProgressColumn(
                "Выгода",
                format="%.1f%%",
                min_value=0.0,
                max_value=max_discount,
            ),
            "confidence": "Надёжность",
            "url": st.column_config.LinkColumn(
                "Ссылка", display_text="Открыть →", width="small"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    csv_bytes = (
        clean.nlargest(100, "score")[_EXPORT_COLS]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        label="📥 Скачать топ-100 сделок (CSV)",
        data=csv_bytes,
        file_name="top_deals.csv",
        mime="text/csv",
    )
