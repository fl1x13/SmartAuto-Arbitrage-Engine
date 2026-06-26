"""Central configuration for the car arbitrage analytics platform.

Values can be overridden via environment variables (see .env.example):
DATABASE_URL, SCRAPER_BASE_URL, SCRAPER_DELAY_SECONDS, SCRAPER_MAX_PAGES,
SCRAPER_CONCURRENCY.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into the process environment.

    A tiny, dependency-free loader: it only fills variables that aren't
    already set, so real environment variables (e.g. injected by launchd or
    Docker) always win. Lets every entry point — dashboard, bot, scraper —
    pick up secrets like GEMINI_API_KEY from a local .env without each one
    wiring up python-dotenv.

    Args:
        path: Path to the .env file (skipped silently when it doesn't exist).
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


_load_dotenv(BASE_DIR / ".env")

# Market segment per brand: lets the model pool rare brands with their price
# peers (13 Bentley ads alone say little; "luxury" as a group says a lot).
_BRAND_SEGMENTS = {
    "luxury": [
        "porsche", "bentley", "rolls_royce", "ferrari", "lamborghini",
        "maserati", "mclaren", "aston_martin", "maybach", "aurus", "lotus",
        "alpina", "hummer",
    ],
    "premium": [
        "bmw", "mercedes", "audi", "lexus", "land_rover", "infiniti",
        "volvo", "cadillac", "jaguar", "genesis", "mini", "acura",
        "lincoln", "tesla", "hongqi", "jeep", "gmc",
    ],
    "mass": [
        "toyota", "volkswagen", "kia", "hyundai", "nissan", "mazda",
        "ford", "skoda", "renault", "chevrolet", "honda", "mitsubishi",
        "opel", "suzuki", "subaru", "peugeot", "citroen", "dodge", "ram",
        "fiat", "seat", "cupra", "mg", "samsung", "daihatsu", "smart",
        "dacia", "chrysler", "buick", "pontiac", "isuzu", "alfa_romeo",
        "lancia", "ssang_yong", "kgm", "solaris", "jetta",
    ],
    "budget": [
        "vaz", "uaz", "gaz", "daewoo", "datsun", "zaz", "lifan", "ravon",
        "tagaz", "vortex", "brilliance", "moscvich", "izh", "212",
    ],
    "china": [
        "haval", "geely", "chery", "changan", "zeekr", "lixiang", "exeed",
        "lynk_co", "jetour", "trumpchi", "omoda", "gac", "byd", "tank",
        "voyah", "great_wall", "denza", "xiaomi", "kaiyi", "jac", "xcite",
        "belgee", "jaecoo", "dongfeng", "livan", "forthing", "wuling",
        "baic", "bestune", "baw", "arcfox", "weltmeister", "swm", "seres",
        "haima", "foton", "evolute", "avatr", "aito", "aion", "faw",
        "tenet", "knewstar", "sollers", "rox",
    ],
}


def _build_brand_segment_map() -> dict:
    """Invert _BRAND_SEGMENTS into a flat brand → segment lookup."""
    return {
        brand: segment
        for segment, brands in _BRAND_SEGMENTS.items()
        for brand in brands
    }


@dataclass
class Config:
    # Database
    db_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR}/data/car_market.db"
    )

    # Model artifacts (dir overridable via MODEL_DIR, e.g. for smoke tests)
    model_dir: Path = Path(
        os.getenv("MODEL_DIR", str(BASE_DIR / "model" / "artifacts"))
    )
    model_path: Path = model_dir / "catboost_model.cbm"
    metrics_path: Path = model_dir / "metrics.json"
    feature_names_path: Path = model_dir / "feature_names.json"
    model_meta_path: Path = model_dir / "model_meta.json"

    # Arbitrage score weights
    w1: float = 0.8
    w2: float = 0.2
    w3: float = 0.1  # price-drop signal (seller motivated to sell fast)
    w4: float = 0.15  # high-mileage penalty (worn cars are hard to resell)
    w5: float = 0.5  # suspicious-listing penalty (scam/hidden-problem cars
    # must sink below honest deals, not crown the ranking)
    w6: float = 0.4  # weight of auto.ru's own price rating — an independent
    # fair-price second opinion. Rewards listings auto.ru also calls cheap and
    # demotes a model discount auto.ru rates as actually above its estimate
    # (a fake discount). Zero contribution when a listing carries no badge.
    # Penalty kicks in above this mileage and grows by w4 per scale step:
    # 300k km → -w4, 450k km → -2*w4
    mileage_penalty_start: int = 150_000
    mileage_penalty_scale: int = 150_000
    # Discount reward saturates here: the deal value of a price gap rises up
    # to ~30% below market, then flattens. A 90% "discount" is almost always
    # a scam or a hidden-problem car, not 3x the deal of a 30% one — without
    # the cap, linear reward hands the top of the ranking to exactly the
    # listings we least want to surface.
    discount_reward_cap: float = 0.30
    # Per-segment discount "noise allowance". Premium/luxury price dispersion
    # is far wider than mass-market, so a given % below the model's estimate is
    # mostly noise there — and a cheap premium car usually hides a costly reason
    # (accident, grey import, looming repairs). This much discount is shaved off
    # before it earns score or a hot grade, so premium must be dramatically
    # cheaper (not merely 30% below) to reach the top deals. Segments absent
    # here (mass, budget, china, other) get no haircut.
    segment_discount_noise: dict = field(default_factory=lambda: {
        "premium": 0.15,
        "luxury": 0.20,
    })
    # Age discount haircut. The price model overvalues old cars (its
    # depreciation curve is too shallow at the far end and the data thins out),
    # so a 30% "discount" on a 12-year-old car is mostly model error, not a
    # deal — hand-labelling showed the misses concentrated almost entirely in
    # cars older than this. Shave age_discount_per_year of discount for each
    # year past age_discount_start, capped, before it earns score or a hot
    # grade, so an old car must be dramatically cheap to reach the top.
    age_discount_start: int = 8
    age_discount_per_year: float = 0.02
    age_discount_cap: float = 0.22

    # Liquidity: data-driven from market depth (see model.predict.market_liquidity),
    # scaled into liquidity_range; liquidity_map holds manual brand overrides.
    liquidity_range: tuple = (0.85, 1.2)
    liquidity_map: dict = field(default_factory=lambda: {
        "toyota": 1.2,
        "kia": 1.2,
        "hyundai": 1.2,
        "bmw": 1.2,
    })
    default_liquidity: float = 1.0

    # Brand → market segment (luxury/premium/mass/budget/china)
    brand_segment_map: dict = field(default_factory=_build_brand_segment_map)
    default_brand_segment: str = "other"

    # Features
    cat_features: list = field(default_factory=lambda: [
        "brand", "model", "body_type", "transmission", "region", "drive",
        "fuel_type", "brand_segment", "modification", "generation",
    ])
    num_features: list = field(default_factory=lambda: [
        "year", "mileage", "engine_volume", "horse_power",
        "owners_count", "car_age", "mileage_per_year",
        "power_density", "owners_per_year",
    ])

    # Scraper
    scraper_base_url: str = os.getenv("SCRAPER_BASE_URL", "")  # empty → demo mode
    scraper_delay_seconds: float = float(os.getenv("SCRAPER_DELAY_SECONDS", "1.5"))
    scraper_max_pages: int = int(os.getenv("SCRAPER_MAX_PAGES", "50"))
    scraper_concurrency: int = int(os.getenv("SCRAPER_CONCURRENCY", "5"))
    # Regions for auto.ru ("rossiya" = nationwide feed; or city slugs
    # like "moskva,sankt-peterburg"). Listing is sorted freshest-first,
    # so frequent shallow scrapes catch newly posted cars.
    scraper_regions: list = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("SCRAPER_REGIONS", "rossiya").split(",")
            if r.strip()
        ]
    )
    scrape_interval_minutes: float = float(
        os.getenv("SCRAPE_INTERVAL_MINUTES")
        or float(os.getenv("SCRAPE_INTERVAL_HOURS", "0")) * 60
        or 15
    )
    # Scheduler retrains the model once this many new ads have accumulated
    retrain_min_new_ads: int = int(os.getenv("RETRAIN_MIN_NEW_ADS", "50"))
    user_agents: list = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/124.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/123.0 Safari/537.36",
    ])

    # Outlier quantiles. Price is clipped on both tails to drop typos and
    # scam/parts listings, but only lightly on the low side: cutting the
    # cheapest 5% per brand+model removed legitimately cheap old/worn cars and
    # biased the fair-price estimate upward (fake "discounts" on old cars).
    # Mileage is NOT clipped at the top — high-mileage cars are exactly the
    # depreciation signal the model needs; only absurd odometer values go.
    outlier_lower_q: float = 0.02
    outlier_upper_q: float = 0.98
    mileage_lower_q: float = 0.0
    mileage_upper_q: float = 0.99
    # Absolute price floor applied before group-quantile filtering. A rare
    # brand+model with <10 ads skips quantile filtering entirely, so a
    # 3 750 ₽ typo/scam on a 2024 car would otherwise reach the top deals.
    # No running car sells below this — anything cheaper is parts/typo/fraud.
    min_plausible_price: int = 30_000

    # Far-from-centre regions: a car here is cheaper at source, but a central-
    # Russia buyer pays to bring it across the country — and these far-east hubs
    # are the Korea/Japan import channel (RHD, "на заказ", price often quoted
    # only as far as Vladivostok). Its sticker therefore understates the landed
    # cost, so the discount is measured against price + delivery, not the
    # sticker alone — otherwise these listings always look the cheapest.
    import_delivery_surcharge: int = 150_000
    import_regions: set = field(default_factory=lambda: {
        "Владивосток", "Уссурийск", "Артём", "Находка", "Хабаровск",
        "Благовещенск", "Комсомольск-на-Амуре", "Южно-Сахалинск",
        "Чита", "Улан-Удэ", "Биробиджан",
    })

    # Sold-listing prune: how many of the highest-scoring ads to verify live
    # each cycle. Only ads that could surface as deals are checked, so the
    # network cost stays bounded; a sold ad's page is fetched and, if its
    # "продан" banner shows, it is flagged out of every deal surface.
    liveness_check_top_n: int = 120

    # Telegram bot
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    bot_top_n: int = 5  # deals per reply (phone-friendly)
    bot_cache_ttl_seconds: int = 300
    bot_watch_interval_minutes: float = float(
        os.getenv("BOT_WATCH_INTERVAL_MINUTES", "15")
    )
    # Public HTTPS URL of the Mini App (python -m bot.webapp behind a tunnel);
    # when set, the bot shows an "open app" button and a menu button.
    webapp_url: str = os.getenv("WEBAPP_URL", "")


cfg = Config()
