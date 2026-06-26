"""Render enriched listing rows as Telegram HTML messages."""

import html

import pandas as pd

CONFIDENCE_RU = {"low": "🔴 низкая", "medium": "🟡 средняя", "high": "🟢 высокая"}


def rub(value: float) -> str:
    """Format rubles with thin-space thousands: 1 234 567 ₽."""
    return f"{value:,.0f} ₽".replace(",", " ")


def _title(row: pd.Series) -> str:
    if row.get("modification"):
        return f"{row['modification']}, {row['year']}"
    return f"{str(row['brand']).title()} {str(row['model']).title()}, {row['year']}"


def deal_caption(row: pd.Series, header: str = "") -> str:
    """One listing as a compact phone-friendly HTML caption.

    Args:
        row: Enriched listing row.
        header: Optional first line (e.g. "🔔 Новая горячая сделка").

    Returns:
        HTML string within Telegram's 1024-char caption limit.
    """
    engine = (
        f"{row['engine_volume']} л / {row['horse_power']} л.с."
        if row["engine_volume"]
        else f"{row['horse_power']} л.с."
    )
    if row.get("fuel_type"):
        engine += f", {row['fuel_type']}"
    # Benefit (and the discount it mirrors) is measured against the landed
    # price, so a far-east import is not flattered by the delivery it omits.
    surcharge = int(row.get("delivery_surcharge", 0) or 0)
    landed = row["price"] + surcharge
    benefit = row["predicted_price"] - landed
    sign = "−" if benefit < 0 else ""
    lines = []
    if header:
        lines.append(f"<b>{html.escape(header)}</b>")
    lines += [
        f"<b>{html.escape(_title(row))}</b>  {row['deal_grade']}",
        f"💰 <b>{rub(row['price'])}</b> · рынок: {rub(row['predicted_price'])}",
    ]
    if surcharge:
        lines.append(
            f"🚚 +{rub(surcharge)} доставка из {row.get('region') or '—'} "
            f"→ {rub(landed)} под ключ"
        )
    lines += [
        (
            f"📉 Выгода: <b>{row['discount_pct']:+.1f}%</b> "
            f"({sign}{rub(abs(benefit))})"
        ),
        f"🛣 {row['mileage']:,} км · {engine} · {row['transmission'] or '—'}".replace(
            ",", " "
        ),
        (
            f"📍 {row.get('region') or '—'} · Надёжность: "
            f"{CONFIDENCE_RU.get(row['confidence'], row['confidence'])}"
        ),
    ]
    if row.get("is_suspicious"):
        lines.append(f"⚠️ {html.escape(str(row['suspicious_reason']))}")
    return "\n".join(lines)


def deals_summary(rows: pd.DataFrame, title: str) -> str:
    """Header line for a list of deals."""
    return f"<b>{html.escape(title)}</b> — {len(rows)} шт., листайте ниже 👇"
