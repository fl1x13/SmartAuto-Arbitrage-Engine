"""Telegram bot: monitor the car-arbitrage pipeline from your phone.

Features:
- 🔥 Топ сделок — best current deals as photo cards with listing links
- 🎯 Подбор машины — budget/brand wizard over live listings
- Оценка по ссылке — paste an auto.ru ad URL, get the fair-price verdict
- 🔔 Подписка — push notification when the scraper finds a fresh hot deal

Run: TELEGRAM_BOT_TOKEN=<token from @BotFather> python -m bot.main
"""

import asyncio
import logging

import pandas as pd
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    MenuButtonWebApp,
    Message,
    ReplyKeyboardMarkup,
    WebAppInfo,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from bot import formatting, service, subscriptions
from config import cfg

logger = logging.getLogger(__name__)

# Bump on every Mini App UI change: Telegram caches the WebView per URL, so a
# new ?v= makes it fetch the fresh page instead of the stale cached one.
WEBAPP_VERSION = 10


def _webapp_url() -> str:
    """Mini App URL with a cache-busting version query for Telegram."""
    sep = "&" if "?" in cfg.webapp_url else "?"
    return f"{cfg.webapp_url}{sep}v={WEBAPP_VERSION}"

MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🔥 Топ сделок"), KeyboardButton(text="🎯 Подбор машины")],
        [KeyboardButton(text="🔔 Подписаться"), KeyboardButton(text="🔕 Отписаться")],
        [KeyboardButton(text="ℹ️ Помощь")],
    ],
    resize_keyboard=True,
)

BUDGETS = [
    ("до 500 тыс.", 0, 500_000),
    ("500 тыс. – 1 млн", 500_000, 1_000_000),
    ("1–2 млн", 1_000_000, 2_000_000),
    ("2–4 млн", 2_000_000, 4_000_000),
    ("от 4 млн", 4_000_000, 1_000_000_000),
]

HELP_TEXT = (
    "Что я умею:\n\n"
    "🔥 <b>Топ сделок</b> — лучшие недооценённые машины прямо сейчас\n"
    "🎯 <b>Подбор машины</b> — укажите бюджет и марку, покажу варианты\n"
    "🔗 <b>Оценка по ссылке</b> — просто пришлите ссылку на объявление "
    "auto.ru, скажу справедливую цену и выгоду\n"
    "🔔 <b>Подписка</b> — пришлю уведомление, как только сканер найдёт "
    "новую горячую сделку (мониторинг каждые "
    f"{cfg.bot_watch_interval_minutes:.0f} мин)"
)


class PickCar(StatesGroup):
    budget = State()
    brand = State()


class SubscribeFlow(StatesGroup):
    budget = State()


def _budget_keyboard(prefix: str, extra: tuple[str, str] | None = None):
    rows = [
        [InlineKeyboardButton(text=label, callback_data=f"{prefix}:{lo}:{hi}")]
        for label, lo, hi in BUDGETS
    ]
    if extra:
        rows.append([InlineKeyboardButton(text=extra[0], callback_data=extra[1])])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_deal(bot: Bot, chat_id: int, row: pd.Series, header: str = "") -> None:
    """Send one listing as a photo card (text fallback) with a link button."""
    caption = formatting.deal_caption(row, header)
    button = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Открыть на auto.ru ↗", url=row["url"])]
        ]
    )
    image_url = row.get("image_url")
    if image_url:
        try:
            await bot.send_photo(
                chat_id, image_url, caption=caption, reply_markup=button
            )
            return
        except TelegramAPIError:
            logger.info("Photo rejected by Telegram, falling back to text")
    await bot.send_message(
        chat_id, caption, reply_markup=button, disable_web_page_preview=True
    )


async def _send_deals(bot: Bot, chat_id: int, deals: pd.DataFrame, title: str) -> None:
    if deals.empty:
        await bot.send_message(
            chat_id, "Ничего не нашлось — попробуйте другие условия."
        )
        return
    await bot.send_message(chat_id, formatting.deals_summary(deals, title))
    for _, row in deals.iterrows():
        await _send_deal(bot, chat_id, row)


def build_dispatcher() -> Dispatcher:
    """Register all handlers on a fresh Dispatcher."""
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(
            "Привет! Я слежу за рынком авто и нахожу недооценённые "
            "объявления.\n\n" + HELP_TEXT,
            reply_markup=MENU,
        )
        if cfg.webapp_url:
            await message.answer(
                "🚀 Удобнее всего — в мини-приложении: лента сделок, "
                "подбор и оценка в пару касаний.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="🚗 Открыть Авторадар",
                                web_app=WebAppInfo(url=_webapp_url()),
                            )
                        ]
                    ]
                ),
            )

    @dp.message(Command("help"))
    @dp.message(F.text == "ℹ️ Помощь")
    async def help_cmd(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=MENU)

    @dp.message(Command("top"))
    @dp.message(F.text == "🔥 Топ сделок")
    async def top(message: Message) -> None:
        await message.answer("Считаю лучшие сделки…")
        deals = await asyncio.to_thread(service.top_deals)
        await _send_deals(
            message.bot, message.chat.id, deals, "🔥 Лучшие сделки сейчас"
        )

    # --- Подбор машины: бюджет → марка → результаты ---
    @dp.message(Command("find"))
    @dp.message(F.text == "🎯 Подбор машины")
    async def pick_start(message: Message, state: FSMContext) -> None:
        await state.set_state(PickCar.budget)
        await message.answer(
            "Какой бюджет?", reply_markup=_budget_keyboard("pickbudget")
        )

    @dp.callback_query(PickCar.budget, F.data.startswith("pickbudget:"))
    async def pick_budget(call: CallbackQuery, state: FSMContext) -> None:
        _, lo, hi = call.data.split(":")
        await state.update_data(budget_from=int(lo), budget_to=int(hi))
        brands = await asyncio.to_thread(service.popular_brands)
        rows = [
            [
                InlineKeyboardButton(
                    text=b.title(), callback_data=f"pickbrand:{b}"
                )
                for b in brands[i : i + 2]
            ]
            for i in range(0, len(brands), 2)
        ]
        rows.append(
            [InlineKeyboardButton(text="Любая марка", callback_data="pickbrand:any")]
        )
        await state.set_state(PickCar.brand)
        await call.message.edit_text(
            "Какая марка?",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )
        await call.answer()

    @dp.callback_query(PickCar.brand, F.data.startswith("pickbrand:"))
    async def pick_brand(call: CallbackQuery, state: FSMContext) -> None:
        brand = call.data.split(":", 1)[1]
        data = await state.get_data()
        await state.clear()
        await call.message.edit_text("Подбираю варианты…")
        deals = await asyncio.to_thread(
            service.pick_cars,
            data["budget_from"],
            data["budget_to"],
            None if brand == "any" else brand,
        )
        await _send_deals(
            call.bot, call.message.chat.id, deals, "🎯 Лучшее под ваш запрос"
        )
        await call.answer()

    # --- Подписка на новые горячие сделки ---
    @dp.message(Command("subscribe"))
    @dp.message(F.text == "🔔 Подписаться")
    async def subscribe_start(message: Message, state: FSMContext) -> None:
        await state.set_state(SubscribeFlow.budget)
        await message.answer(
            "На какой бюджет присылать новые 🔥 сделки?",
            reply_markup=_budget_keyboard(
                "subbudget", extra=("Без ограничений", "subbudget:0:0")
            ),
        )

    @dp.callback_query(SubscribeFlow.budget, F.data.startswith("subbudget:"))
    async def subscribe_budget(call: CallbackQuery, state: FSMContext) -> None:
        _, _, hi = call.data.split(":")
        await state.clear()
        max_price = int(hi) or None
        await asyncio.to_thread(
            subscriptions.subscribe, call.message.chat.id, max_price
        )
        limit = f" до {formatting.rub(max_price)}" if max_price else ""
        await call.message.edit_text(
            f"🔔 Готово! Пришлю уведомление, как только появится новая "
            f"горячая сделка{limit}. Отписаться: 🔕"
        )
        await call.answer()

    @dp.message(Command("unsubscribe"))
    @dp.message(F.text == "🔕 Отписаться")
    async def unsubscribe(message: Message) -> None:
        existed = await asyncio.to_thread(
            subscriptions.unsubscribe, message.chat.id
        )
        await message.answer(
            "🔕 Подписка отключена." if existed else "У вас нет активной подписки."
        )

    # --- Оценка по ссылке: любое сообщение с URL auto.ru ---
    @dp.message(F.text.contains("auto.ru/"))
    async def evaluate(message: Message) -> None:
        await message.answer("Оцениваю объявление…")
        row = await asyncio.to_thread(service.evaluate_url, message.text.strip())
        if row is None:
            await message.answer(
                "Не получилось скачать объявление: проверьте ссылку, либо "
                "auto.ru временно показывает капчу — попробуйте позже."
            )
            return
        await _send_deal(message.bot, message.chat.id, row, header="🧮 Оценка")

    @dp.message()
    async def fallback(message: Message) -> None:
        await message.answer(
            "Пришлите ссылку на объявление auto.ru или выберите действие в меню.",
            reply_markup=MENU,
        )

    return dp


async def watch_new_deals(bot: Bot) -> None:
    """Push fresh hot deals to every subscriber (runs on a schedule)."""
    subs = await asyncio.to_thread(subscriptions.all_subscriptions)
    if not subs:
        return
    await asyncio.to_thread(service.load_market, True)  # one refresh per tick
    for sub in subs:
        deals = await asyncio.to_thread(
            service.new_hot_deals,
            sub.last_sent_at,
            sub.max_price,
            sub.brand or None,
        )
        if deals.empty:
            continue
        for _, row in deals.head(3).iterrows():
            try:
                await _send_deal(
                    bot, sub.chat_id, row, header="🔔 Новая горячая сделка"
                )
            except TelegramAPIError as e:
                logger.error("Push to chat %s failed: %s", sub.chat_id, e)
                break
        else:
            await asyncio.to_thread(subscriptions.mark_sent, sub.chat_id)
    logger.info("Deal watcher tick done: %d subscribers", len(subs))


async def main() -> None:
    """Start polling and the deal-watcher scheduler."""
    if not cfg.telegram_token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN не задан. Создайте бота у @BotFather и "
            "запустите: TELEGRAM_BOT_TOKEN=<токен> python3 -m bot.main"
        )
    bot = Bot(cfg.telegram_token, default=DefaultBotProperties(parse_mode="HTML"))
    dp = build_dispatcher()

    if cfg.webapp_url:
        # Mini App entry next to the message field (works in private chats)
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="Авторадар", web_app=WebAppInfo(url=_webapp_url())
            )
        )
        logger.info("Mini App menu button set: %s", _webapp_url())

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        watch_new_deals,
        "interval",
        minutes=cfg.bot_watch_interval_minutes,
        args=[bot],
    )
    scheduler.start()

    logger.info("Bot started, polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
