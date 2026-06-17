"""Telegram Mini App backend: JSON API over the deal pipeline + static UI.

Serves the single-page app from bot/static/ and a small read-only API that
reuses bot.service (same cache as the chat bot). Telegram opens the page via
a web_app button; the page calls the API with the WebApp initData header,
which is HMAC-validated against the bot token when WEBAPP_REQUIRE_AUTH=1.

Run: python -m bot.webapp   (default port 8050; needs HTTPS tunnel for
Telegram, e.g. `cloudflared tunnel --url http://localhost:8050`).
"""

import asyncio
import hashlib
import hmac
import logging
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bot import service
from config import cfg

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
REQUIRE_AUTH = os.getenv("WEBAPP_REQUIRE_AUTH", "0") == "1"

# auto.ru/yandex photo URLs end in a size token ("…/320x240"); swap it for a
# larger one so the Mini App cards show crisp, immersive photos.
_IMG_SIZE_RE = re.compile(r"/\d{2,4}x\d{2,4}$")


def _hires_image(url: str) -> str:
    """Upgrade a thumbnail photo URL to a larger size ("" stays "")."""
    return _IMG_SIZE_RE.sub("/1200x900", url) if url else ""


# Brand emblems (carlogos dataset on jsDelivr) shown until a listing photo is
# scraped. Most slugs match after _→-; these differ. Unknown brands still get a
# URL — the Mini App falls back to a 🚗 glyph if it 404s.
_LOGO_BASE = (
    "https://cdn.jsdelivr.net/gh/filippofilip95/"
    "car-logos-dataset@master/logos/optimized"
)
_LOGO_SLUG = {
    "vaz": "lada",
    "mercedes": "mercedes-benz",
    "ssang_yong": "ssangyong",
    "moscvich": "moskvich",
    "vw": "volkswagen",
}


def _logo_url(brand: str) -> str:
    """Brand-emblem image URL for the placeholder before a real photo exists."""
    slug = _LOGO_SLUG.get(brand, brand.replace("_", "-"))
    return f"{_LOGO_BASE}/{slug}.png"

app = FastAPI(title="Car Arbitrage Mini App", docs_url=None, redoc_url=None)


def validate_init_data(init_data: str, bot_token: str) -> bool:
    """Verify Telegram WebApp initData per the official HMAC scheme.

    Args:
        init_data: Raw query string from Telegram.WebApp.initData.
        bot_token: The bot token the hash is keyed against.

    Returns:
        True when the hash matches.
    """
    try:
        pairs = dict(parse_qsl(init_data, keep_blank_values=True))
        received_hash = pairs.pop("hash")
    except (ValueError, KeyError):
        return False
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_hash)


def _check_auth(init_data: str | None) -> None:
    if not REQUIRE_AUTH:
        return
    if not init_data or not validate_init_data(init_data, cfg.telegram_token):
        raise HTTPException(status_code=401, detail="invalid Telegram initData")


def _row_to_json(row: pd.Series) -> dict:
    """Project an enriched listing row onto the Mini App card shape."""
    engine = (
        f"{row['engine_volume']} л · {row['horse_power']} л.с."
        if row["engine_volume"]
        else f"{row['horse_power']} л.с."
    )
    if row.get("fuel_type"):
        engine += f" · {row['fuel_type']}"
    title = (
        f"{row['modification']}"
        if row.get("modification")
        else f"{str(row['brand']).title()} {str(row['model']).title()}"
    )
    return {
        "ad_id": int(row["ad_id"]) if pd.notna(row.get("ad_id")) else None,
        "title": title,
        "year": int(row["year"]),
        "price": int(row["price"]),
        "predicted_price": int(row["predicted_price"]),
        "discount_pct": float(row["discount_pct"]),
        "grade": row["deal_grade"],
        "mileage": int(row["mileage"]),
        "engine": engine,
        "transmission": row["transmission"] or "—",
        "region": row.get("region") or "—",
        "confidence": row["confidence"],
        "image_url": _hires_image(row.get("image_url") or ""),
        "logo_url": _logo_url(str(row["brand"])),
        "url": row["url"],
        "suspicious_reason": row.get("suspicious_reason") or "",
    }


@app.get("/api/top")
async def api_top(
    max_price: int | None = Query(default=None, ge=0),
    min_price: int | None = Query(default=None, ge=0),
    brand: str | None = None,
    model: str | None = None,
    category: str | None = None,
    n: int = Query(default=10, le=20),
    offset: int = Query(default=0, ge=0),
    x_telegram_init_data: str | None = Header(default=None),
) -> list[dict]:
    """Best current deals for the «Сделки» tab (offset pages the ranking)."""
    _check_auth(x_telegram_init_data)
    deals = await asyncio.to_thread(
        service.top_deals, n, max_price, brand, model, min_price, offset, category
    )
    return [_row_to_json(row) for _, row in deals.iterrows()]


@app.get("/api/brands")
async def api_brands(
    x_telegram_init_data: str | None = Header(default=None),
) -> list[str]:
    """Popular brands for the picker chips."""
    _check_auth(x_telegram_init_data)
    return await asyncio.to_thread(service.popular_brands, 12)


@app.get("/api/models")
async def api_models(
    brand: str,
    x_telegram_init_data: str | None = Header(default=None),
) -> list[str]:
    """Popular models of a brand for the picker chips."""
    _check_auth(x_telegram_init_data)
    return await asyncio.to_thread(service.models_for_brand, brand, 14)


@app.get("/api/pick")
async def api_pick(
    budget_from: int = Query(ge=0),
    budget_to: int = Query(gt=0),
    brand: str | None = None,
    model: str | None = None,
    x_telegram_init_data: str | None = Header(default=None),
) -> list[dict]:
    """Picker results for a budget window."""
    _check_auth(x_telegram_init_data)
    deals = await asyncio.to_thread(
        service.pick_cars, budget_from, budget_to, brand, model
    )
    return [_row_to_json(row) for _, row in deals.iterrows()]


@app.get("/api/explain")
async def api_explain(
    ad_id: int,
    x_telegram_init_data: str | None = Header(default=None),
) -> dict:
    """'Why this is a good deal' report for one listing already in the base."""
    _check_auth(x_telegram_init_data)
    report = await asyncio.to_thread(service.deal_report, ad_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Объявление не найдено")
    return report


@app.get("/api/reliability")
async def api_reliability(
    ad_id: int,
    x_telegram_init_data: str | None = Header(default=None),
) -> dict:
    """AI reliability check (Claude + web search) for one listing."""
    _check_auth(x_telegram_init_data)
    from bot.reliability import NoApiKey

    try:
        report = await asyncio.to_thread(service.reliability_for_ad, ad_id)
    except NoApiKey:
        raise HTTPException(
            status_code=503,
            detail="Проверка надёжности не настроена: добавьте бесплатный "
            "OPENROUTER_API_KEY (openrouter.ai/keys) или GEMINI_API_KEY "
            "(aistudio.google.com) в .env и перезапустите сервис.",
        ) from None
    except Exception:
        logger.exception("Reliability check failed")
        raise HTTPException(
            status_code=502,
            detail="Не удалось проверить надёжность — попробуйте позже.",
        ) from None
    if report is None:
        raise HTTPException(status_code=404, detail="Объявление не найдено")
    return report


@app.get("/api/evaluate")
async def api_evaluate(
    url: str,
    x_telegram_init_data: str | None = Header(default=None),
) -> dict:
    """Evaluate one auto.ru listing URL."""
    _check_auth(x_telegram_init_data)
    row = await asyncio.to_thread(service.evaluate_url, url)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Не удалось получить объявление: auto.ru не отдаёт страницу "
            "(защита от ботов или объявление снято). Попробуйте позже.",
        )
    card = _row_to_json(row)
    # Attach the "why" report inline: freshly fetched ads aren't in the market
    # frame, so the detail screen can't fetch it later by ad_id.
    try:
        card["report"] = await asyncio.to_thread(
            service.build_report, row, service.load_market()
        )
    except Exception:  # noqa: BLE001 — a report failure must not block the card
        logger.exception("Report build failed for evaluated ad")
        card["report"] = None
    return card


@app.get("/")
async def index() -> FileResponse:
    # Telegram's WebView caches Mini App assets aggressively; no-store forces
    # it to refetch so UI updates actually reach users.
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("WEBAPP_PORT", "8050")))
