"""Generate realistic mock data and seed the database for development/demo."""

import logging
import random
from datetime import datetime, timedelta, timezone

from faker import Faker

from scraper.schemas import CarAdSchema
from scraper.storage import get_engine, save_ads

logger = logging.getLogger(__name__)
fake = Faker("ru_RU")

BRANDS_MODELS = {
    "toyota": ["camry", "corolla", "rav4", "land cruiser", "highlander"],
    "kia": ["rio", "ceed", "sportage", "sorento", "k5"],
    "hyundai": ["solaris", "elantra", "tucson", "santa fe", "accent"],
    "bmw": ["3 series", "5 series", "x5", "x3", "1 series"],
    "volkswagen": ["polo", "passat", "tiguan", "golf", "touareg"],
    "skoda": ["octavia", "rapid", "kodiaq", "karoq", "superb"],
    "nissan": ["qashqai", "x-trail", "juke", "almera", "patrol"],
    "mercedes": ["c-class", "e-class", "glc", "gle", "a-class"],
    "lada": ["vesta", "granta", "niva", "largus", "kalina"],
    "renault": ["logan", "sandero", "duster", "kaptur", "arkana"],
}

BODY_TYPES = ["sedan", "hatchback", "suv", "wagon", "crossover", "minivan"]
TRANSMISSIONS = ["automatic", "manual", "robot", "variator"]
DRIVES = ["передний", "задний", "полный"]
REGIONS = [
    "Москва", "Санкт-Петербург", "Екатеринбург", "Новосибирск",
    "Казань", "Краснодар", "Нижний Новгород", "Самара",
]

BASE_PRICES = {
    "toyota": (900_000, 4_500_000),
    "kia": (700_000, 3_000_000),
    "hyundai": (700_000, 3_200_000),
    "bmw": (1_500_000, 7_000_000),
    "volkswagen": (800_000, 3_500_000),
    "skoda": (750_000, 2_800_000),
    "nissan": (800_000, 3_800_000),
    "mercedes": (1_800_000, 8_000_000),
    "lada": (350_000, 900_000),
    "renault": (500_000, 1_800_000),
}


def generate_ad(ad_id: int) -> CarAdSchema:
    """Generate a single realistic car listing."""
    brand = random.choice(list(BRANDS_MODELS.keys()))
    model = random.choice(BRANDS_MODELS[brand])
    year = random.randint(2010, 2024)
    car_age = 2024 - year
    mileage = random.randint(car_age * 5_000, car_age * 30_000 + 10_000)

    price_low, price_high = BASE_PRICES[brand]
    # Newer = more expensive; slight random noise
    age_factor = max(0.3, 1 - car_age * 0.06)
    base = random.randint(price_low, price_high)
    price = int(base * age_factor * random.uniform(0.85, 1.15) / 10_000) * 10_000

    # ~8% of listings are deliberately underpriced (arbitrage candidates)
    if random.random() < 0.08:
        price = int(price * random.uniform(0.55, 0.75))

    return CarAdSchema(
        ad_id=ad_id,
        brand=brand,
        model=model,
        year=year,
        mileage=mileage,
        price=price,
        body_type=random.choice(BODY_TYPES),
        engine_volume=round(random.choice([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5]), 1),
        horse_power=random.choice([90, 110, 122, 140, 150, 177, 190, 220, 249, 306]),
        transmission=random.choice(TRANSMISSIONS),
        owners_count=random.randint(1, 4),
        url=f"https://auto.ru/cars/used/sale/{ad_id}/",
        published_at=datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(days=random.randint(0, 90)),
        region=random.choice(REGIONS),
        drive=random.choice(DRIVES),
        fuel_type=random.choices(
            ["бензин", "дизель", "гибрид", "электро"],
            weights=[70, 20, 7, 3],
        )[0],
    )


def seed_database(n: int = 2000) -> None:
    """Populate the database with n mock listings.

    Args:
        n: Number of listings to generate.
    """
    logger.info("Generating %d mock car listings...", n)
    ads = [generate_ad(i + 1) for i in range(n)]
    engine = get_engine()
    inserted = save_ads(ads, engine)
    logger.info("Seeded database with %d new records.", inserted)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seed_database()
