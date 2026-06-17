"""Market Overview: persona-focused analytics (flipper / buyer / dealer)."""

import os
import sys

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
from app.data_loader import load_enriched_data

st.set_page_config(page_title="Обзор рынка", page_icon="📈", layout="wide")
st.title("📈 Обзор рынка")
st.caption(
    "Аналитика под задачу: перекупу — где маржа и что быстро продаётся, "
    "покупателю — что теряет машина в цене, автосалону — где объём рынка."
)

df = load_enriched_data()

tab_flip, tab_buyer, tab_dealer = st.tabs(
    ["🔁 Перекупу", "🛒 Покупателю", "🏢 Автосалону"]
)


def _brand_pick(frame: pd.DataFrame, label: str, key: str) -> str:
    """Brand selectbox (title-cased), defaults to the most listed brand."""
    options = frame["brand"].value_counts().index.tolist()
    return st.selectbox(
        label, options, format_func=str.title, key=key
    )


def _annual_rate(p_from: float, p_to: float, years: float) -> float:
    """Average annual depreciation (%) between two price points."""
    return (1 - (p_to / p_from) ** (1 / max(years, 1))) * 100


def _depreciation_insight(dep: pd.DataFrame, label: str) -> str:
    """Turn a median-price-by-age curve into a written recommendation.

    Args:
        dep: DataFrame with car_age and median_price, sorted by car_age.
        label: Human-readable name of the brand/model being analyzed.

    Returns:
        Markdown text with the analysis and a buy-age recommendation.
    """
    ages = dep["car_age"].to_numpy()
    prices = dep["median_price"].to_numpy()
    overall = _annual_rate(prices[0], prices[-1], ages[-1] - ages[0])

    # Steepest stretch between adjacent observed ages
    seg_rates = [
        (_annual_rate(prices[i], prices[i + 1], ages[i + 1] - ages[i]), i)
        for i in range(len(ages) - 1)
    ]
    steep_rate, steep_i = max(seg_rates)
    steep_from, steep_to = int(ages[steep_i]), int(ages[steep_i + 1])

    # Plateau: first age from which the rest of the curve loses <5%/yr
    plateau_age = None
    for i in range(len(ages) - 1):
        if _annual_rate(prices[i], prices[-1], ages[-1] - ages[i]) < 5:
            plateau_age = int(ages[i])
            break

    lines = [
        f"🤖 **Вывод по графику:** {label} дешевеет в среднем на "
        f"**~{overall:.0f}% в год**.",
        f"- Самое резкое падение — между **{steep_from} и {steep_to} годами** "
        f"(−{steep_rate:.0f}%/год): продавать машину этого возраста "
        "невыгодно, а покупателю выгодно дождаться, пока она его пройдёт.",
    ]
    if plateau_age is not None:
        lines.append(
            f"- После **{plateau_age} лет** цена почти перестаёт падать "
            "(<5%/год): машина уже «на дне», при перепродаже почти "
            "не потеряете."
        )
        buy_from = max(steep_to, min(plateau_age, steep_to + 2))
        lines.append(
            f"- **Рекомендация:** оптимальный возраст покупки — "
            f"**{buy_from}–{buy_from + 3} лет**: основное обесценивание "
            "уже оплатил прежний владелец, а ликвидность ещё высокая."
        )
    else:
        lines.append(
            f"- Цена падает по всей кривой — если важна перепродажа, "
            f"берите {label} не моложе {steep_to} лет, основное падение "
            "к этому возрасту уже произошло."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- Перекупу
with tab_flip:
    st.subheader("Где сейчас недооценённые машины")
    st.caption(
        "Модели с наибольшим числом выгодных объявлений (🔥 + 👍) прямо "
        "сейчас — что мониторить для перепродажи."
    )
    deals = df[df["deal_grade"].isin(["🔥 Горячая", "👍 Хорошая"])]
    hot_models = (
        deals.assign(name=deals["brand"].str.title() + " " + deals["model"].str.title())
        .groupby("name", as_index=False)
        .agg(
            deals_count=("price", "size"),
            median_discount=("discount_pct", "median"),
            median_price=("price", "median"),
        )
        .nlargest(12, "deals_count")
        .sort_values("deals_count")
    )
    if hot_models.empty:
        st.info("Сейчас нет объявлений с оценкой 🔥/👍 — загляните позже.")
    else:
        fig = px.bar(
            hot_models,
            x="deals_count",
            y="name",
            orientation="h",
            color="median_discount",
            color_continuous_scale="RdYlGn",
            labels={
                "deals_count": "Выгодных объявлений сейчас",
                "name": "",
                "median_discount": "Медианная выгода, %",
            },
            hover_data={"median_price": ":,.0f"},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Ликвидность с возрастом")
    flip_brand = _brand_pick(df, "Марка", key="flip_brand")
    seg = df[df["brand"] == flip_brand].copy()
    seg["age_bucket"] = pd.cut(
        seg["car_age"],
        bins=[-1, 2, 5, 8, 12, 100],
        labels=["до 3 лет", "3–5", "6–8", "9–12", "13+"],
    )
    liq = seg.groupby("age_bucket", observed=True).agg(
        count=("price", "size"),
        spread=("price", lambda p: p.std() / p.median() * 100 if len(p) > 2 else None),
    ).reset_index()
    fig = go.Figure()
    fig.add_bar(
        x=liq["age_bucket"].astype(str),
        y=liq["count"],
        name="Объявлений на рынке (глубина)",
        marker_color="steelblue",
    )
    fig.add_scatter(
        x=liq["age_bucket"].astype(str),
        y=liq["spread"],
        name="Разброс цен, % от медианы",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="crimson", width=3),
    )
    fig.update_layout(
        yaxis=dict(title="Объявлений"),
        yaxis2=dict(title="Разброс цен, %", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Глубина рынка падает, а разброс цен растёт с возрастом — старую "
        "машину дольше продавать и труднее оценить. Берите на перепродажу "
        "возраст, где столбики ещё высокие, а красная линия ещё низкая."
    )

    st.divider()
    st.subheader("Географический арбитраж")
    geo_models = (
        df.assign(name=df["brand"].str.title() + " " + df["model"].str.title())
        .groupby("name")["region"]
        .agg(lambda r: (r != "").sum())
    )
    geo_options = geo_models[geo_models >= 10].index.tolist()
    geo_pick = st.selectbox("Модель", sorted(geo_options), key="geo_model")
    geo_df = df.assign(
        name=df["brand"].str.title() + " " + df["model"].str.title()
    ).query("name == @geo_pick and region != ''")
    by_city = (
        geo_df.groupby("region", as_index=False)
        .agg(median_price=("price", "median"), count=("price", "size"))
        .query("count >= 3")
        .sort_values("median_price")
    )
    if len(by_city) < 2:
        st.info("Мало городов с ≥3 объявлениями этой модели.")
    else:
        spread_pct = (
            (by_city["median_price"].max() / by_city["median_price"].min()) - 1
        ) * 100
        fig = px.bar(
            by_city,
            x="median_price",
            y="region",
            orientation="h",
            text="count",
            labels={"median_price": "Медианная цена (₽)", "region": ""},
            height=max(260, 36 * len(by_city)),
        )
        fig.update_traces(texttemplate="%{text} объявл.", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Разница между самым дешёвым и самым дорогим городом — "
            f"**{spread_pct:.0f}%**. Минус перегон и оформление — остальное "
            "потенциальная маржа."
        )

# ---------------------------------------------------------------- Покупателю
with tab_buyer:
    st.subheader("Сколько машина теряет в цене с возрастом")
    col_b, col_m = st.columns(2)
    with col_b:
        buyer_brand = _brand_pick(df, "Марка", key="buyer_brand")
    bseg = df[df["brand"] == buyer_brand]
    with col_m:
        dep_models = ["Все модели"] + bseg["model"].value_counts().index.tolist()
        dep_model = st.selectbox(
            "Модель",
            dep_models,
            format_func=lambda m: m if m == "Все модели" else m.title(),
            key="dep_model",
        )
    dep_seg = bseg if dep_model == "Все модели" else bseg[bseg["model"] == dep_model]
    dep_label = (
        buyer_brand.title()
        if dep_model == "Все модели"
        else f"{buyer_brand.title()} {dep_model.title()}"
    )
    dep = (
        dep_seg.groupby("car_age", as_index=False)
        .agg(median_price=("price", "median"), count=("price", "size"))
        .query("count >= 3 and car_age <= 15")
        .sort_values("car_age")
    )
    if len(dep) < 3:
        st.info(
            "Мало данных для кривой обесценивания — выберите «Все модели» "
            "или другую модель."
        )
    else:
        fig = px.line(
            dep,
            x="car_age",
            y="median_price",
            markers=True,
            labels={
                "car_age": "Возраст (лет)",
                "median_price": "Медианная цена (₽)",
            },
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(_depreciation_insight(dep, dep_label))

    st.divider()
    st.subheader("Цена vs пробег: где торговаться")
    buyer_models = bseg["model"].value_counts().index.tolist()
    buyer_model = st.selectbox(
        "Модель", buyer_models, format_func=str.title, key="buyer_model"
    )
    mseg = bseg[bseg["model"] == buyer_model]
    if len(mseg) < 5:
        st.info("Мало объявлений этой модели для графика.")
    else:
        fig = px.scatter(
            mseg,
            x="mileage",
            y="price",
            color="deal_grade",
            custom_data=["ad_id"],
            hover_data=["year", "predicted_price", "discount_pct"],
            labels={
                "mileage": "Пробег (км)",
                "price": "Цена (₽)",
                "deal_grade": "Оценка",
            },
            height=420,
        )
        trend = mseg.sort_values("mileage")
        fig.add_scatter(
            x=trend["mileage"],
            y=trend["predicted_price"]
            .rolling(max(len(trend) // 10, 3), center=True, min_periods=1)
            .median(),
            mode="lines",
            name="Справедливая цена (ML)",
            line=dict(color="crimson", dash="dash", width=2),
        )
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="price_mileage_scatter",
        )
        st.caption(
            "Точки выше красной линии — переоценённые объявления (повод "
            "торговаться), ниже — потенциально выгодные. **Кликните по "
            "точке, чтобы открыть карточку машины.**"
        )
        points = (
            event.selection.points if event and event.selection else []
        )

        def _clicked_ad_id(point: dict) -> int | None:
            # px packs custom_data + hover_data into customdata; Streamlit
            # delivers it as a dict with string keys ('0' = ad_id). The ML
            # trend line has no customdata at all.
            data = point.get("customdata")
            if isinstance(data, dict):
                data = data.get("0")
            elif isinstance(data, (list, tuple)):
                data = data[0]
            return int(data) if data is not None else None

        clicked_ids = [
            ad_id for p in points if (ad_id := _clicked_ad_id(p)) is not None
        ]
        if clicked_ids:
            clicked = mseg[mseg["ad_id"] == clicked_ids[0]]
            if not clicked.empty:
                render_card(clicked.iloc[0])

    st.divider()
    st.subheader("Сколько добавляет коробка передач")
    by_trans = (
        bseg[bseg["transmission"] != ""]
        .groupby("transmission", as_index=False)
        .agg(median_price=("price", "median"), count=("price", "size"))
        .query("count >= 5")
        .sort_values("median_price")
    )
    if len(by_trans) < 2:
        st.info("Мало данных по типам коробки для этой марки.")
    else:
        fig = px.bar(
            by_trans,
            x="transmission",
            y="median_price",
            text="count",
            labels={
                "transmission": "Коробка передач",
                "median_price": "Медианная цена (₽)",
            },
            color="median_price",
            color_continuous_scale="Blues",
            height=360,
        )
        fig.update_traces(texttemplate="%{text} объявл.", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Разница медианных цен показывает, сколько рынок доплачивает "
            "за автомат/робот у этой марки — и сколько вы потеряете при "
            "перепродаже механики."
        )

# ---------------------------------------------------------------- Автосалону
with tab_dealer:
    st.subheader("Структура рынка: марки и модели")
    structure = (
        df.assign(
            brand=df["brand"].str.title(),
            model=df["model"].str.title(),
        )
        .groupby(["brand", "model"], as_index=False)
        .agg(count=("price", "size"), median_price=("price", "median"))
    )
    fig = px.treemap(
        structure,
        path=["brand", "model"],
        values="count",
        color="median_price",
        color_continuous_scale="RdYlGn_r",
        labels={"median_price": "Медианная цена (₽)"},
        height=460,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Размер — число объявлений, цвет — медианная цена.")

    st.divider()
    st.subheader("Сегменты: объём рынка × цена × выгодные закупки")
    seg_stats = (
        df.assign(name=df["brand"].str.title() + " " + df["model"].str.title())
        .groupby(["name", "brand_segment"], as_index=False)
        .agg(
            count=("price", "size"),
            median_price=("price", "median"),
            hot=("deal_grade", lambda g: g.isin(["🔥 Горячая", "👍 Хорошая"]).sum()),
        )
        .query("count >= 10")
    )
    fig = px.scatter(
        seg_stats,
        x="median_price",
        y="count",
        size="hot",
        color="brand_segment",
        hover_name="name",
        log_x=True,
        labels={
            "median_price": "Медианная цена (₽, лог-шкала)",
            "count": "Объявлений на рынке",
            "brand_segment": "Сегмент",
            "hot": "Выгодных закупок",
        },
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Правый верх — массовые дорогие сегменты (объём + чек); размер "
        "точки — сколько машин прямо сейчас можно закупить ниже рынка."
    )

    st.divider()
    st.subheader("Цена лошадиной силы по маркам")
    hp_value = (
        df[df["horse_power"] > 0]
        .assign(price_per_hp=lambda d: d["price"] / d["horse_power"])
        .groupby("brand", as_index=False)
        .agg(price_per_hp=("price_per_hp", "median"), count=("price", "size"))
        .query("count >= 5")
        .sort_values("price_per_hp")
        .assign(brand=lambda d: d["brand"].str.title())
    )
    fig = px.bar(
        hp_value,
        x="brand",
        y="price_per_hp",
        labels={"brand": "Марка", "price_per_hp": "₽ за 1 л.с. (медиана)"},
        color="price_per_hp",
        color_continuous_scale="RdYlGn_r",
        height=380,
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Чем левее марка, тем дешевле обходится мощность на вторичном рынке."
    )
