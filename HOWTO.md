# Как запустить проект

## 🟢 Самый простой путь — запустить ВСЁ одной командой

Этот проект уже настроен на **автозапуск при входе в систему** через launchd
(`com.autoradar`). Одна команда поднимает весь стек и сама держит его живым:
дашборд, авто-обновление базы, Telegram-бот и публичный адрес для Mini App.

```bash
# ЗАПУСТИТЬ / ПЕРЕЗАПУСТИТЬ ВСЁ:
launchctl kickstart -k gui/$(id -u)/com.autoradar

# ПРОВЕРИТЬ, ЧТО РАБОТАЕТ (должно показать рабочий https-адрес):
cat ~/Pet_project_fable/.autoradar/current_url.txt
tail -20 ~/Pet_project_fable/.autoradar/supervisor.log

# ОСТАНОВИТЬ ВСЁ:
launchctl bootout gui/$(id -u)/com.autoradar 2>/dev/null
# Включить обратно (и автозапуск при входе):
launchctl bootstrap gui/$(id -u) ~/Pet_project_fable/scripts/com.autoradar.plist
```

Что поднимает `kickstart`:

| Компонент | Где | Зачем |
|-----------|-----|-------|
| Дашборд + scheduler | Docker, `:8501` | веб-аналитика + авто-скрапинг каждые 15 мин |
| Mini App API | `bot.webapp`, `:8050` | бэкенд телеграм-приложения |
| Telegram-бот | `bot.main` | бот @avtorader_bot и кнопка Mini App |
| Публичный адрес | туннель | по нему Mini App открывается с телефона |

Всё перезапускается само, если падает. После правок `.env` или кода —
повтори `kickstart`, чтобы процессы перечитали изменения.

---

## 📱 Почему Mini App иногда не открывается (и как это лечится)

Telegram требует **публичный HTTPS-адрес** для Mini App. Адрес даёт туннель
(приложение крутится на этом Mac, наружу его пускает туннель). Проблема: твой
провайдер по очереди блокирует известные туннель-домены (`trycloudflare.com`,
`ngrok.io`, а сейчас и Tailscale `*.ts.net`).

Поэтому supervisor сам **подбирает рабочий туннель** и переключается, если
текущий заблокировали:

1. **localhost.run** (`*.lhr.life`) — основной. SSH-ключ
   `~/.ssh/lhr_tunnel_ed25519` зарегистрирован на бесплатном аккаунте
   localhost.run, поэтому адрес **постоянный**:
   **`https://7f32a8b1dc13f4.lhr.life`** — он не меняется при переподключениях.
2. **Tailscale Funnel** (`*.ts.net`, из `.env` `WEBAPP_URL`) — аварийный запас на
   случай, если localhost.run будет недоступен.

Watchdog каждые 90 секунд проверяет, что текущий адрес реально отдаёт твоё
приложение. При обрыве localhost.run переподключается на **тот же** постоянный
адрес (благодаря ключу), поэтому бот не дёргается. Текущий рабочий адрес всегда
лежит в `~/Pet_project_fable/.autoradar/current_url.txt`.

> **Постоянство адреса не зависит от VPN.** localhost.run узнаёт тебя по
> SSH-ключу, а не по IP/стране — можно менять страны VPN и на телефоне, и на
> Mac, адрес остаётся тем же. Нужно лишь, чтобы Mac был включён, с интернетом,
> и мог достучаться до localhost.run по SSH (порт 22).

### Если адрес снова станет случайным

Это значит ключ перестал распознаваться (в `~/Pet_project_fable/.autoradar/lhr.log`
не будет строки `user … authenticated`). Перепроверь, что ключ на месте на
<https://admin.localhost.run/> (раздел SSH keys). Публичный ключ:
```bash
cat ~/.ssh/lhr_tunnel_ed25519.pub
```

> Самый надёжный вариант (если появится возможность) — свой домен + Cloudflare
> Tunnel: обычные edge-адреса Cloudflare провайдеры не блокируют по SNI. Это
> требует домена (~150–200 ₽/год), туннель `cloudflared` остаётся бесплатным.

---

## 🛠 Запуск вручную по частям (для разработки/отладки)

Если нужно гонять компоненты по отдельности (не через автозапуск):

```bash
cd ~/Pet_project_fable
pip install -r requirements.txt           # один раз — поставить зависимости

# Дашборд → http://localhost:8501
python3 -m streamlit run app/app.py

# Авто-обновление базы + переобучение модели (не закрывать терминал)
SCRAPER_BASE_URL="https://auto.ru" SCRAPER_MAX_PAGES=10 python3 -m scraper.scheduler

# Mini App API → http://localhost:8050
python3 -m bot.webapp

# Telegram-бот (нужен WEBAPP_URL = публичный адрес туннеля)
WEBAPP_URL="https://<адрес-из-current_url.txt>" python3 -m bot.main
```

Разовый скрапинг + обучение без расписания:

```bash
SCRAPER_BASE_URL="https://auto.ru" python3 -m scraper.runner --pages 5
python3 -m model.train
```

---

## ✅ Проверки

```bash
python3 -m pytest tests/ -q     # тесты
python3 -m ruff check .         # линтер
bash scripts/smoke.sh           # seed → train → проверка дашборда
```

---

## 📊 Данные

Проект работает на **реальных** данных auto.ru (`data/car_market.db`, 10 000+
объявлений). Scheduler каждые 15 минут докачивает свежие объявления и
переобучает модель после накопления `RETRAIN_MIN_NEW_ADS` (по умолчанию 50)
новых записей. Демо-режим (без интернета) включается, если `SCRAPER_BASE_URL`
пуст — генерируются мок-листинги.

---

## ⚙️ Переменные окружения

Скопируй `.env.example` → `.env` и заполни (`.env` в git не попадает):

| Переменная | По умолчанию | Смысл |
|-----------|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | — | токен бота от @BotFather |
| `WEBAPP_URL` | — | постоянный адрес туннеля (Tailscale Funnel) |
| `WEBAPP_PORT` | `8050` | порт Mini App API |
| `DATABASE_URL` | `sqlite:///data/car_market.db` | БД |
| `SCRAPER_BASE_URL` | пусто (demo) | `https://auto.ru` для реальных данных |
| `SCRAPER_REGIONS` | `rossiya` | регионы скрапинга |
| `SCRAPER_MAX_PAGES` | `5` | страниц за прогон |
| `SCRAPE_INTERVAL_MINUTES` | `15` | интервал автообновления |
| `RETRAIN_MIN_NEW_ADS` | `50` | порог авто-переобучения |
