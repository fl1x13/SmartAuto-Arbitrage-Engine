"""Car picker: detailed search across every brand/model + paged live listings."""

import os
import sys

# Same package-shadowing guard as in app.py (pages run as standalone scripts).
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if not sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

import streamlit as st

from app.components.car_card import render_result_row
from app.data_loader import load_enriched_data

st.set_page_config(page_title="Подбор машины", page_icon="🎯", layout="wide")

PAGE = 10  # listings revealed per "show more" click

st.title("🎯 Подбор машины")
st.caption(
    "Детальный поиск по всем маркам и моделям из живых объявлений: задайте "
    "любые условия — покажем лучшие варианты с максимальной выгодой к "
    "рыночной цене и ликвидностью. Можно догружать новые объявления."
)

head_l, head_r = st.columns([4, 1])
with head_r:
    if st.button("🔄 Обновить", use_container_width=True, help="Сбросить кэш и "
                 "подтянуть свежие объявления, собранные скрапером."):
        load_enriched_data.clear()
        st.session_state.pop("picker_sig", None)
        st.rerun()

df = load_enriched_data()

# --- Criteria ---
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_from, budget_to = st.slider(
            "Бюджет (₽)",
            min_value=0,
            max_value=int(df["price"].max()),
            value=(0, int(df["price"].max())),
            step=50_000,
            format="%d",
        )
        brands = st.multiselect(
            "Марка (пусто — любая)",
            sorted(df["brand"].str.title().unique()),
        )
        # Models depend on the chosen brand(s); empty brand → every model.
        model_pool = (
            df[df["brand"].isin([b.lower() for b in brands])] if brands else df
        )
        models = st.multiselect(
            "Модель (пусто — любая)",
            sorted(model_pool["model"].str.title().unique()),
        )
    with col2:
        year_from = st.number_input(
            "Год выпуска не старше", 1990, int(df["year"].max()), 1990
        )
        mileage_to = st.number_input(
            "Пробег до (км)", 0, 1_500_000, 1_500_000, step=10_000
        )
        owners_to = st.slider("Владельцев не больше", 1, 6, 6)
    with col3:
        body_types = st.multiselect("Кузов", sorted(df["body_type"].unique()))
        transmissions = st.multiselect(
            "Коробка передач", sorted(df["transmission"].unique())
        )
        fuel_options = sorted(v for v in df["fuel_type"].unique() if v)
        fuels = st.multiselect("Тип топлива", fuel_options)

    col4, col5, col6 = st.columns(3)
    with col4:
        regions = st.multiselect(
            "Город", sorted(v for v in df["region"].unique() if v)
        )
    with col5:
        sort_by = st.selectbox(
            "Сначала показывать",
            ["Лучшие сделки (score)", "Максимальная выгода (%)", "Дешевле",
             "Новее", "Меньше пробег"],
            help="Score учитывает выгоду к рыночной цене, ликвидность "
            "сегмента и снижение цены продавцом.",
        )
    with col6:
        reliable_only = st.checkbox(
            "Только надёжные оценки",
            value=False,
            help="Скрыть машины, по которым у модели мало похожих "
            "объявлений (оценка цены менее точна).",
        )
        st.checkbox(
            "Исключить подозрительные",
            value=True,
            key="no_suspicious",
            help="Слишком дешёвые и с неправдоподобным пробегом.",
        )

mask = df["price"].between(budget_from, budget_to)
mask &= df["year"] >= year_from
mask &= df["mileage"] <= mileage_to
mask &= df["owners_count"] <= owners_to
if brands:
    mask &= df["brand"].isin([b.lower() for b in brands])
if models:
    mask &= df["model"].isin([m.lower() for m in models])
if body_types:
    mask &= df["body_type"].isin(body_types)
if transmissions:
    mask &= df["transmission"].isin(transmissions)
if fuels:
    mask &= df["fuel_type"].isin(fuels)
if regions:
    mask &= df["region"].isin(regions)
if reliable_only:
    mask &= df["confidence"] != "low"
if st.session_state.get("no_suspicious", True):
    mask &= ~df["is_suspicious"]

matches = df[mask]

sort_col, ascending = {
    "Лучшие сделки (score)": ("score", False),
    "Максимальная выгода (%)": ("discount_pct", False),
    "Дешевле": ("price", True),
    "Новее": ("year", False),
    "Меньше пробег": ("mileage", True),
}[sort_by]
matches = matches.sort_values(sort_col, ascending=ascending)

# Reset the page size whenever the search changes, so "show more" always
# starts from the top of a fresh result set.
filter_sig = (
    budget_from, budget_to, year_from, mileage_to, owners_to,
    tuple(brands), tuple(models), tuple(body_types), tuple(transmissions),
    tuple(fuels), tuple(regions), reliable_only,
    st.session_state.get("no_suspicious", True), sort_by,
)
if st.session_state.get("picker_sig") != filter_sig:
    st.session_state["picker_sig"] = filter_sig
    st.session_state["picker_shown"] = PAGE

shown = min(st.session_state.get("picker_shown", PAGE), len(matches))

st.subheader(
    f"Подходит {len(matches)} объявлений"
    + (f" — показано {shown}" if shown else "")
)
if matches.empty:
    st.info(
        "Под такие условия ничего не нашлось. Расширьте бюджет, год "
        "или уберите часть фильтров."
    )
else:
    if matches.head(shown)["discount_pct"].max() < 0:
        st.caption(
            "⚠️ Все показанные варианты дороже своей рыночной оценки — "
            "возможно, стоит подождать новых объявлений или поторговаться."
        )
    for _, row in matches.head(shown).iterrows():
        render_result_row(row, df)

    if shown < len(matches):
        remaining = len(matches) - shown
        if st.button(
            f"Показать ещё {min(PAGE, remaining)} (осталось {remaining})",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["picker_shown"] = shown + PAGE
            st.rerun()
