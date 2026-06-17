"""Demo listing-page generator: renders mock ads as HTML.

Lets the real BS4 parsing path run end-to-end without an external site.
A small share of cards is rendered broken (missing price) to exercise
the validation/drop logic in the pipeline.
"""

import random

from scraper.seed import generate_ad

ADS_PER_PAGE = 20
BROKEN_AD_SHARE = 0.05

_CARD_TEMPLATE = """
<div class="listing-item" data-ad-id="{ad_id}">
  <a class="listing-link" href="{url}">
    <span class="listing-brand">{brand}</span>
    <span class="listing-model">{model}</span>
  </a>
  <span class="listing-year">{year}</span>
  <span class="listing-mileage">{mileage}</span>
  <span class="listing-price">{price}</span>
  <span class="listing-body">{body_type}</span>
  <span class="listing-engine">{engine}</span>
  <span class="listing-hp">{horse_power}</span>
  <span class="listing-transmission">{transmission}</span>
  <span class="listing-drive">{drive}</span>
  <span class="listing-owners">{owners_count}</span>
  <time class="listing-published" datetime="{published_at}"></time>
</div>
"""


def _format_number(value: int, suffix: str) -> str:
    """Render an int the way listing sites do: '150 000 км'."""
    return f"{value:,}".replace(",", " ") + f" {suffix}"


def render_listing_page(page: int, ads_per_page: int = ADS_PER_PAGE) -> str:
    """Render one listing page of mock ads as HTML.

    Args:
        page: 1-based page number; drives deterministic ad_id ranges.
        ads_per_page: Number of ad cards per page.

    Returns:
        HTML string mimicking a marketplace listing page.
    """
    cards = []
    start_id = (page - 1) * ads_per_page + 1
    for ad_id in range(start_id, start_id + ads_per_page):
        ad = generate_ad(ad_id)
        price_str = _format_number(ad.price, "₽")
        if random.random() < BROKEN_AD_SHARE:
            price_str = "цена не указана"

        cards.append(
            _CARD_TEMPLATE.format(
                ad_id=ad.ad_id,
                url=ad.url,
                brand=ad.brand.title(),
                model=ad.model.title(),
                year=ad.year,
                mileage=_format_number(ad.mileage, "км"),
                price=price_str,
                body_type=ad.body_type,
                engine=f"{ad.engine_volume} л",
                horse_power=ad.horse_power,
                transmission=ad.transmission,
                drive=ad.drive,
                owners_count=ad.owners_count,
                published_at=ad.published_at.isoformat(),
            )
        )

    return f"<html><body><div class='listings'>{''.join(cards)}</div></body></html>"
