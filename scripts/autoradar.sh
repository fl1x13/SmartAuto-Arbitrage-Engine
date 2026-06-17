#!/bin/bash
# Autoradar supervisor — keeps the whole stack up with zero manual steps.
#
#   * dashboard + scheduler        -> docker compose (background, self-healing)
#   * Mini App API (bot.webapp)    -> local :8050, restarted if it dies
#   * public HTTPS URL             -> Tailscale Funnel (fixed *.ts.net URL in
#                                     WEBAPP_URL; Cloudflare/ngrok are blocked on
#                                     this network, Funnel works over 443/relay)
#   * Telegram bot (bot.main)      -> launched once with the fixed WEBAPP_URL,
#                                     restarted if it dies. URL never changes.
#
# Run directly:  ./scripts/autoradar.sh
# Or via launchd (auto-start at login): see scripts/com.autoradar.plist
set -u

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

STATE_DIR="$PROJECT_DIR/.autoradar"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/supervisor.log"
URL_FILE="$STATE_DIR/current_url.txt"

PY="$(command -v python3)"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

# --- load .env -------------------------------------------------------------
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a; . "$PROJECT_DIR/.env"; set +a
else
  log "FATAL: .env not found"; exit 1
fi
WEBAPP_PORT="${WEBAPP_PORT:-8050}"

# --- child PIDs we manage --------------------------------------------------
WEBAPP_PID=""; BOT_PID=""

cleanup() {
  log "Shutting down…"
  [ -n "$BOT_PID" ] && kill "$BOT_PID" 2>/dev/null
  [ -n "$WEBAPP_PID" ] && kill "$WEBAPP_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# --- 1. dashboard + scheduler (docker) -------------------------------------
# Backgrounded: the `init` service (first scrape+train) can take minutes, and
# the Mini App / bot read the DB + model straight off disk, so they must not
# wait on docker.
if command -v docker >/dev/null 2>&1; then
  log "Starting docker compose (dashboard + scheduler) in background…"
  ( docker compose up -d >>"$LOG" 2>&1 || log "WARN: docker compose up failed" ) &
else
  log "WARN: docker not found — skipping dashboard/scheduler"
fi

# --- 2. Mini App API, supervised in the background -------------------------
start_webapp() {
  while true; do
    log "Starting bot.webapp on :$WEBAPP_PORT…"
    "$PY" -m bot.webapp >>"$STATE_DIR/webapp.log" 2>&1
    log "bot.webapp exited — restarting in 3s"
    sleep 3
  done
}
start_webapp & WEBAPP_PID=$!

# give the API a moment to bind before the tunnel points at it
sleep 5

# --- 3. permanent public URL via Tailscale Funnel --------------------------
# Funnel gives a fixed *.ts.net HTTPS URL (in WEBAPP_URL) that never changes,
# and tailscaled keeps it up across reboots. We just (idempotently) re-assert
# it on start; the URL is static, so the bot is launched once.
TS_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
[ -x "$TS_CLI" ] || TS_CLI="$(command -v tailscale)"

if [ -z "${WEBAPP_URL:-}" ]; then
  log "FATAL: WEBAPP_URL not set in .env"; exit 1
fi
echo "$WEBAPP_URL" >"$URL_FILE"

if [ -n "$TS_CLI" ] && [ -x "$TS_CLI" ]; then
  log "Asserting Tailscale Funnel → :$WEBAPP_PORT"
  "$TS_CLI" funnel --bg "$WEBAPP_PORT" >>"$LOG" 2>&1 \
    || log "WARN: 'tailscale funnel' failed (Tailscale down or Funnel disabled?)"
else
  log "WARN: tailscale CLI not found — assuming Funnel already configured"
fi

# --- 4. Telegram bot, supervised (fixed URL) -------------------------------
log "Starting bot.main with WEBAPP_URL=$WEBAPP_URL"
while true; do
  WEBAPP_URL="$WEBAPP_URL" "$PY" -m bot.main >>"$STATE_DIR/bot.log" 2>&1
  log "bot.main exited — restarting in 3s"
  sleep 3
done
