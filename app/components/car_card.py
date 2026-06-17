"""Reusable car-listing card components (photo, specs, deal metrics)."""

import re

import pandas as pd
import streamlit as st

CONFIDENCE_RU = {"low": "🔴 низкая", "medium": "🟡 средняя", "high": "🟢 высокая"}

# auto.ru photo URLs end in a size token ("…/320x240"); request a larger one.
_IMG_SIZE_RE = re.compile(r"/\d{2,4}x\d{2,4}$")


def _hires(url: str) -> str:
    """Upgrade a thumbnail photo URL to a larger size."""
    return _IMG_SIZE_RE.sub("/1200x900", url) if url else url


_PLACEHOLDER = (
    "<div style='border:1px dashed grey;border-radius:12px;height:{h}px;"
    "display:flex;align-items:center;justify-content:center;"
    "font-size:{fs}px'>🚗</div>"
)


def rub(value: float) -> str:
    """Format rubles with thin-space thousands: 1 234 567 ₽."""
    return f"{value:,.0f} ₽".replace(",", " ")


def _specs_line(row: pd.Series) -> str:
    engine = (
        f"{row['engine_volume']} л / {row['horse_power']} л.с."
        if row["engine_volume"]
        else f"{row['horse_power']} л.с."
    )
    if row.get("fuel_type"):
        engine += f" ({row['fuel_type']})"
    specs = {
        "Пробег": f"{row['mileage']:,} км".replace(",", " "),
        "Двигатель": engine,
        "Коробка": str(row["transmission"]).capitalize() or "—",
        "Привод": str(row["drive"]).capitalize() or "—",
        "Кузов": str(row["body_type"]).capitalize() or "—",
        "Владельцев": row["owners_count"],
        "Регион": row.get("region") or "—",
    }
    return " · ".join(f"**{k}:** {v}" for k, v in specs.items())


def _title(row: pd.Series) -> str:
    if row.get("modification"):
        return f"{row['modification']}, {row['year']}"
    return f"{row['brand'].title()} {row['model'].title()}, {row['year']}"


def render_deal_report(row: pd.Series, market_df: pd.DataFrame) -> None:
    """Expander explaining WHY a listing is (or isn't) a good deal.

    Shows the SHAP price factors and "looks the same but costs more"
    comparables — the same report the Telegram Mini App serves, so both
    surfaces tell the identical story.

    Args:
        row: Enriched listing row.
        market_df: Enriched market frame (for the comparable listings).
    """
    from bot.service import build_report

    try:
        rep = build_report(row, market_df)
    except Exception:  # noqa: BLE001 — never let the report break the card
        return

    with st.expander("📊 Подробнее — почему это выгодно"):
        disc = rep["discount_pct"]
        if disc > 0:
            st.markdown(
                f"**Дешевле рыночной оценки на {disc:.1f}%** — выгода "
                f"{rub(rep['predicted_price'] - rep['price'])}."
            )
        else:
            st.markdown(f"Дороже рыночной оценки на {abs(disc):.1f}%.")
        st.caption(
            f"{rep['grade']} · оценка по {rep['sample_count']} похожим объявлениям"
        )

        st.markdown("**Из чего складывается справедливая цена**")
        st.caption(
            f"Базовая цена сегмента {rub(rep['base_price'])}, далее — поправки "
            "за параметры именно этой машины:"
        )
        for factor in rep["factors"]:
            mark = "🟢" if factor["pct"] >= 0 else "🔴"
            st.markdown(f"{mark} {factor['label']}: **{factor['pct']:+.1f}%**")

        st.markdown(f"**{rep.get('comp_title', 'Похожие в продаже')}**")
        if rep["comparables"]:
            cmp_df = pd.DataFrame(rep["comparables"])
            cmp_df["diff"] = cmp_df["diff_pct"].map(
                lambda v: f"{'+' if v >= 0 else '−'}{abs(v):.1f}%"
            )
            st.dataframe(
                cmp_df[["title", "year", "mileage", "price", "diff", "url"]],
                column_config={
                    "title": "Модель",
                    "year": st.column_config.NumberColumn("Год", format="%d"),
                    "mileage": st.column_config.NumberColumn("Пробег", format="%d км"),
                    "price": st.column_config.NumberColumn("Цена", format="%d ₽"),
                    "diff": "Разница",
                    "url": st.column_config.LinkColumn(
                        "Объявление", display_text="Открыть →", width="small"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption(
                "Похожих объявлений этой модели в продаже сейчас мало — "
                "сравнивать не с чем."
            )


# Verdict text from the LLM → leading emoji for the reliability card.
_VERDICT_EMOJI = {
    "стоит брать": "🟢",
    "брать с осторожностью": "🟡",
    "рискованно": "🔴",
}


def _render_reliability_verdict(rep: dict) -> None:
    """Render the structured AI reliability verdict (score, weak points, …)."""
    score = rep.get("score")
    verdict = str(rep.get("verdict") or "").strip()
    emoji = _VERDICT_EMOJI.get(verdict.lower(), "⚪")

    col_score, col_text = st.columns([1, 3], gap="large")
    with col_score:
        if isinstance(score, (int, float)):
            st.metric("Надёжность", f"{score:g}/10")
            st.progress(min(max(float(score) / 10, 0.0), 1.0))
    with col_text:
        if verdict:
            st.markdown(f"### {emoji} {verdict.capitalize()}")
        if rep.get("summary"):
            st.markdown(rep["summary"])

    sections = (
        ("⚠️ Типичные болячки этого поколения и двигателя", rep.get("weak_points")),
        ("🛠 Что обычно требует внимания на таком пробеге", rep.get("at_mileage")),
        ("✅ Что проверить перед покупкой", rep.get("checklist")),
    )
    for header, items in sections:
        if items:
            st.markdown(f"**{header}**")
            for item in items:
                st.markdown(f"- {item}")

    st.caption(
        "Оценка сгенерирована ИИ по открытым источникам (форумы, отзывы "
        "владельцев, отзывные кампании) и носит ознакомительный характер — "
        "не заменяет очный осмотр и диагностику."
    )


def render_reliability_check(row: pd.Series) -> None:
    """Button + AI reliability verdict for the selected listing.

    The price model only says whether a listing is *cheap*; this says whether
    the car is worth *owning*. On click, an LLM (free Gemini with live Google
    Search, or Claude when ANTHROPIC_API_KEY is set) looks up the exact
    engine/generation's known weak points, recalls and high-mileage issues,
    then returns a 0–10 score, a buy verdict, what fails at this mileage, and a
    pre-purchase checklist. Same verdict the Telegram Mini App serves; the
    result is cached, so re-clicking the same car is instant.

    Args:
        row: Enriched listing row (needs ad_id and the raw spec columns).
    """
    from app.data_loader import load_reliability_report

    flag = f"reliab_{int(row['ad_id'])}"
    st.subheader("🔧 Проверка надёжности")
    st.caption(
        "ИИ-агент считывает всё объявление с auto.ru (полные характеристики, "
        "комплектацию, состояние и описание продавца), сопоставляет с типичными "
        "болячками этого двигателя и поколения — и выносит вердикт, стоит ли "
        "брать именно эту машину с таким пробегом и за эту цену."
    )
    if st.button(
        "Проверить надёжность этой машины",
        key=f"btn_{flag}",
        type="primary",
        use_container_width=True,
    ):
        st.session_state[flag] = True
    if not st.session_state.get(flag):
        return

    with st.spinner(
        "Считываю полное объявление с auto.ru (характеристики, комплектация, "
        "описание) и анализирую слабые места модели…"
    ):
        rep = load_reliability_report(
            ad_id=int(row["ad_id"]),
            brand=str(row["brand"]),
            model=str(row["model"]),
            generation=str(row.get("generation") or ""),
            modification=str(row.get("modification") or ""),
            year=int(row["year"]),
            mileage=int(row["mileage"]),
            engine_volume=float(row.get("engine_volume") or 0),
            horse_power=int(row.get("horse_power") or 0),
            fuel_type=str(row.get("fuel_type") or ""),
            transmission=str(row.get("transmission") or ""),
            drive=str(row.get("drive") or ""),
            price=int(row["price"]),
            url=str(row.get("url") or ""),
        )

    if rep.get("error") == "no_key":
        st.info(
            "🔑 Проверка надёжности не настроена. Добавьте бесплатный "
            "`GEMINI_API_KEY` в файл `.env` (получить ключ — "
            "https://aistudio.google.com/app/apikey) и перезапустите дашборд."
        )
        return
    if rep.get("error"):
        st.error(
            "Не удалось получить оценку надёжности — попробуйте ещё раз позже."
        )
        return

    _render_reliability_verdict(rep)


def render_card(row: pd.Series, market_df: pd.DataFrame | None = None) -> None:
    """Full detail card: photo, deal metrics, specs, listing link, warnings.

    Args:
        row: Enriched listing row (predicted_price, discount_pct, deal_grade,
            confidence, sample_count, suspicious fields present).
        market_df: Enriched market frame; when given, appends the
            "why this is a good deal" report under the card.
    """
    col_photo, col_info = st.columns([2, 3], gap="large")
    with col_photo:
        if row.get("image_url"):
            st.image(_hires(row["image_url"]), use_container_width=True)
        else:
            st.markdown(
                _PLACEHOLDER.format(h=220, fs=64), unsafe_allow_html=True
            )
            st.caption(
                "Фото пока не собрано — появится после следующего "
                "прохода скрапера."
            )
        st.link_button(
            "🔗 Открыть объявление", row["url"], use_container_width=True
        )
    with col_info:
        st.subheader(_title(row))
        st.markdown(
            f"{row['deal_grade']} · Надёжность оценки: "
            f"{CONFIDENCE_RU.get(row['confidence'], row['confidence'])} "
            f"({row['sample_count']} похожих объявлений)"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Цена продавца", rub(row["price"]))
        m2.metric("Справедливая цена", rub(row["predicted_price"]))
        m3.metric(
            "Выгода",
            f"{row['discount_pct']:.1f}%",
            delta=rub(row["predicted_price"] - row["price"]),
        )
        st.markdown(_specs_line(row))
        if row["is_suspicious"]:
            st.warning(
                f"⚠️ Почему объявление подозрительное: "
                f"{row['suspicious_reason']}"
            )
    if market_df is not None:
        render_deal_report(row, market_df)


def render_result_row(row: pd.Series, market_df: pd.DataFrame | None = None) -> None:
    """Compact horizontal card for pick-a-car result lists.

    Args:
        row: Enriched listing row.
        market_df: Enriched market frame; when given, appends the
            "why this is a good deal" report under the card.
    """
    with st.container(border=True):
        col_photo, col_info, col_deal = st.columns([1.2, 3, 1.3])
        with col_photo:
            if row.get("image_url"):
                st.image(_hires(row["image_url"]), use_container_width=True)
            else:
                st.markdown(
                    _PLACEHOLDER.format(h=90, fs=36), unsafe_allow_html=True
                )
        with col_info:
            st.markdown(f"**{_title(row)}**  \n{row['deal_grade']}")
            st.caption(_specs_line(row))
        with col_deal:
            st.markdown(
                f"**{rub(row['price'])}**  \n"
                f"рынок: {rub(row['predicted_price'])}  \n"
                f"выгода: **{row['discount_pct']:+.1f}%**"
            )
            st.link_button("Открыть ↗", row["url"], use_container_width=True)
        if market_df is not None:
            render_deal_report(row, market_df)
