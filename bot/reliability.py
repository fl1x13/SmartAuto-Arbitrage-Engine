"""AI reliability check for a specific used car — with live web search.

Given one listing's exact specs (brand/model/generation/engine/mileage), asks
an LLM to look up the model's known weak points, recalls, and owner feedback,
then return a plain-language buy verdict.

Providers, picked by whichever key is set (see ``_provider`` for order):
  * ANTHROPIC_API_KEY — Claude + web_search (paid, highest quality).
  * OPENROUTER_API_KEY — OpenRouter free tier (no web search; verdict from the
    model's own knowledge). Reachable from regions where Gemini's free tier is
    geo-blocked (quota 0), so it needs no proxy/server.
  * GEMINI_API_KEY (or GOOGLE_API_KEY) — Google Gemini, free tier, with free
    Google Search grounding. Best where its free tier is available; otherwise
    route via GEMINI_PROXY.

Raises ``NoApiKey`` when neither is configured, so the API returns a clear
"needs setup" message instead of a 500.
"""

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import truststore

logger = logging.getLogger(__name__)

# Use the OS trust store for the Gemini HTTPS call. On machines behind a
# TLS-inspecting proxy (corporate MITM), Python's bundled CA list doesn't
# trust the proxy's cert and urllib raises CERTIFICATE_VERIFY_FAILED; the OS
# store does include that CA. Same reason scraper.parser injects truststore.
_SSL_CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

# Optional outbound proxy for the Gemini call only. Google's Gemini API free
# tier is geo-restricted (e.g. quota 0 / "denied access" from some countries),
# so route just this request through a proxy/exit-node in an allowed region.
# Supports http://, https://, socks5://, socks5h:// (remote DNS), socks4://.
# Everything else in the app keeps its normal direct connection.
_GEMINI_PROXY = os.getenv("GEMINI_PROXY", "").strip()
_opener_cache: urllib.request.OpenerDirector | None = None


def _build_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener honoring GEMINI_PROXY (direct when unset).

    Returns:
        Opener that sends the Gemini HTTPS request directly or via the
        configured HTTP/SOCKS proxy.

    Raises:
        RuntimeError: SOCKS proxy requested but PySocks isn't installed, or
            the proxy scheme is unsupported.
    """
    https = urllib.request.HTTPSHandler(context=_SSL_CTX)
    if not _GEMINI_PROXY:
        return urllib.request.build_opener(https)
    parsed = urllib.parse.urlparse(_GEMINI_PROXY)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        proxy = urllib.request.ProxyHandler(
            {"http": _GEMINI_PROXY, "https": _GEMINI_PROXY}
        )
        return urllib.request.build_opener(proxy, https)
    if scheme.startswith("socks"):
        try:
            import socks
            from sockshandler import SocksiPyHandler
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "GEMINI_PROXY uses SOCKS but PySocks isn't installed "
                "(pip install PySocks)."
            ) from e
        ptype = socks.SOCKS4 if scheme.startswith("socks4") else socks.SOCKS5
        # socks5h/socks4a → resolve DNS on the proxy side (remote region).
        rdns = scheme in ("socks5h", "socks4a")
        # Through SOCKS, TLS is end-to-end to Google (the proxy only tunnels
        # TCP), so the local MITM can't intercept — the handler's default
        # context with standard CAs verifies the real Google cert.
        return urllib.request.build_opener(
            SocksiPyHandler(
                ptype,
                parsed.hostname,
                parsed.port,
                rdns=rdns,
                username=parsed.username,
                password=parsed.password,
            )
        )
    raise RuntimeError(f"Unsupported GEMINI_PROXY scheme: {scheme!r}")


def _opener() -> urllib.request.OpenerDirector:
    """Return the cached proxy-aware opener (built on first use)."""
    global _opener_cache
    if _opener_cache is None:
        _opener_cache = _build_opener()
    return _opener_cache

ANTHROPIC_MODEL = "claude-opus-4-8"
GEMINI_MODEL = "gemini-2.0-flash"
# Free OpenRouter model (override via OPENROUTER_MODEL). Routed through
# OpenRouter's infra, so it works from regions where Gemini's free tier is
# geo-blocked. Free tier has no live web search, so the verdict relies on the
# model's own knowledge — strong for well-documented car reliability issues.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
# Free models are frequently rate-limited (429) or retired (404); fall back
# through these in order so the feature stays up without a config change.
_OPENROUTER_FALLBACKS = [
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

_SYSTEM = (
    "Ты — опытный автоэксперт-диагност и подборщик б/у авто. Твоя задача — "
    "честно оценить надёжность конкретной машины перед покупкой. Найди типичные "
    "болячки именно этого поколения и двигателя, отзывные кампании, отзывы "
    "владельцев и ресурс на больших пробегах. Внимательно используй ВСЕ данные "
    "объявления: точную комплектацию, тип двигателя и КПП, состояние, ПТС, "
    "таможню, число владельцев, срок владения и особенно текст описания от "
    "продавца — ищи в нём тревожные сигналы (замены агрегатов, ДТП, тюнинг, "
    "«не на ходу») и сопоставляй с известными слабыми местами модели. Пиши "
    "простым языком для обычного покупателя, без воды. Опирайся на факты, не "
    "выдумывай."
)

_SCHEMA = (
    "Верни ТОЛЬКО один JSON-объект (без markdown, без пояснений вокруг), строго:\n"
    "{\n"
    '  "score": <целое 0-10, насколько надёжна и беспроблемна>,\n'
    '  "verdict": "<стоит брать | брать с осторожностью | рискованно>",\n'
    '  "summary": "<2-3 предложения: итог простым языком>",\n'
    '  "weak_points": ["<типичные болячки этого поколения/двигателя>", ...],\n'
    '  "at_mileage": ["<что обычно требует внимания на таком пробеге>", ...],\n'
    '  "checklist": ["<что проверить перед покупкой именно этой машины>", ...]\n'
    "}"
)


class NoApiKey(RuntimeError):
    """Raised when no LLM provider key is configured."""


def _gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _openrouter_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def _provider() -> str | None:
    """Pick a provider by available key.

    Order: paid Claude (best, has web search) → OpenRouter (free, reachable
    from regions where Gemini's free tier is geo-blocked) → Gemini direct.
    OpenRouter wins over Gemini when both keys are set, so a geo-blocked
    GEMINI_API_KEY left in .env doesn't shadow a working OpenRouter key.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _openrouter_key():
        return "openrouter"
    if _gemini_key():
        return "gemini"
    return None


def _car_brief(row: pd.Series) -> str:
    """Human-readable spec line for the prompt."""
    title = row.get("modification") or (
        f"{str(row['brand']).title()} {str(row['model']).title()}"
    )
    bits = [f"Автомобиль: {title}", f"Год: {int(row['year'])}"]
    if row.get("generation"):
        bits.append(f"Поколение: {row['generation']}")
    bits.append(f"Пробег: {int(row['mileage']):,} км".replace(",", " "))
    if row.get("engine_volume"):
        bits.append(f"Двигатель: {row['engine_volume']} л")
    bits.append(f"Мощность: {int(row['horse_power'])} л.с.")
    if row.get("fuel_type"):
        bits.append(f"Топливо: {row['fuel_type']}")
    if row.get("transmission"):
        bits.append(f"Коробка: {row['transmission']}")
    if row.get("drive"):
        bits.append(f"Привод: {row['drive']}")
    bits.append(f"Цена: {int(row['price']):,} ₽".replace(",", " "))
    return "\n".join(bits)


def _parse_json(text: str) -> dict:
    """Pull the JSON object out of the model's final text."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in model response")
    return json.loads(match.group(0))


def _call_gemini(system: str, user: str) -> str:
    """Google Gemini (free tier) with Google Search grounding → text."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={_gemini_key()}"
    )
    base = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1600},
    }

    def _post(body: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _opener().open(req, timeout=70) as resp:
            return json.loads(resp.read())

    try:  # prefer live Google Search grounding (free on Gemini 2.0)
        out = _post({**base, "tools": [{"google_search": {}}]})
    except urllib.error.HTTPError:
        try:  # grounding unavailable on this key/tier → model knowledge only
            out = _post(base)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini HTTP {e.code}") from e
    parts = out["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _call_openrouter(system: str, user: str) -> str:
    """OpenRouter (free tier, OpenAI-compatible, no web search) → text.

    Tries the configured model, then the free-model fallbacks on 429/404/5xx,
    since free models are routinely rate-limited or retired.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_openrouter_key()}",
        # Optional attribution headers recommended by OpenRouter.
        "HTTP-Referer": "https://github.com/car-arbitrage",
        "X-Title": "Car Arbitrage Reliability",
    }
    # Configured model first, then the fallbacks (deduped, order preserved).
    models = list(dict.fromkeys([OPENROUTER_MODEL, *_OPENROUTER_FALLBACKS]))
    last_exc: Exception = RuntimeError("no OpenRouter model attempted")
    for model in models:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
            "max_tokens": 1600,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with _opener().open(req, timeout=70) as resp:
                out = json.loads(resp.read())
            return out["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:200]
            last_exc = RuntimeError(f"OpenRouter HTTP {e.code}: {detail}")
            if e.code in (404, 408, 429, 500, 502, 503):
                logger.warning("OpenRouter model %s unavailable (%s), trying "
                               "next", model, e.code)
                continue
            raise last_exc from e
    raise last_exc


def _call_anthropic(system: str, user: str) -> str:
    """Claude (paid) with the web_search server tool → text."""
    import anthropic

    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": user}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    resp = None
    for _ in range(4):  # server-side web_search may pause_turn; resume the loop
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1600,
            system=system,
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    return "".join(b.text for b in resp.content if b.type == "text")


def _details_block(details: dict) -> str:
    """Format scraped ad details (full spec table, equipment, owner text).

    Args:
        details: Output of ``scraper.autoru_ad.parse_ad_details``.

    Returns:
        A prompt-ready block, or "" when there's nothing usable. The owner
        description is capped so a verbose ad can't blow up the prompt.
    """
    parts: list[str] = []
    specs = details.get("specs") or {}
    if specs:
        parts.append("Характеристики из объявления:")
        parts += [f"- {k}: {v}" for k, v in specs.items()]
    compl = (details.get("complectation") or "").strip()
    if compl:
        parts.append(f"Комплектация: {compl[:800]}")
    desc = (details.get("description") or "").strip()
    if desc:
        parts.append(f"Описание от продавца: {desc[:2500]}")
    return "\n".join(parts)


def _fetch_ad_details(url: str) -> dict | None:
    """Best-effort live fetch of full ad details; None on any failure."""
    if not url or "auto.ru" not in url:
        return None
    try:
        from scraper.autoru_ad import fetch_ad_details

        return fetch_ad_details(url)
    except Exception as e:  # noqa: BLE001 — enrichment is optional, never fatal
        logger.warning("Ad detail fetch failed for %s: %s", url, e)
        return None


def reliability_report(
    row: pd.Series, details: dict | None = None, fetch: bool = True
) -> dict:
    """Return an AI reliability verdict for one listing.

    For maximum precision the verdict is built from the *full* ad, not just the
    stored columns: when ``fetch`` is on (and ``details`` isn't supplied), the
    live auto.ru page is scraped for the complete spec table, equipment and the
    owner's free-text description, all fed to the model alongside the known
    weak points of this generation/engine. A failed fetch degrades gracefully
    to the stored fields.

    Args:
        row: Listing row with the raw spec columns (incl. ``url``).
        details: Pre-scraped ad details (``parse_ad_details`` output) to use
            instead of fetching; skips the network call.
        fetch: When True and no ``details`` given, fetch them live from
            ``row['url']``.

    Returns:
        Dict: score, verdict, summary, weak_points, at_mileage, checklist, title.

    Raises:
        NoApiKey: no provider key configured.
    """
    provider = _provider()
    if provider is None:
        raise NoApiKey

    if details is None and fetch:
        details = _fetch_ad_details(str(row.get("url") or ""))

    brief = _car_brief(row)
    detail_text = _details_block(details) if details else ""
    user = (
        f"{brief}\n"
        + (f"\n{detail_text}\n" if detail_text else "")
        + "\nПроанализируй надёжность ИМЕННО этой машины с учётом всех данных "
        "выше (комплектация, двигатель, состояние, история, описание продавца). "
        "Сначала вспомни/найди типичные болячки этого поколения и двигателя, "
        "сопоставь их с пробегом, состоянием и тем, что написал владелец, "
        f"потом дай вердикт.\n\n{_SCHEMA}"
    )
    if provider == "anthropic":
        text = _call_anthropic(_SYSTEM, user)
    elif provider == "openrouter":
        text = _call_openrouter(_SYSTEM, user)
    else:
        text = _call_gemini(_SYSTEM, user)
    data = _parse_json(text)
    data["title"] = brief.splitlines()[0].replace("Автомобиль: ", "")
    return data
