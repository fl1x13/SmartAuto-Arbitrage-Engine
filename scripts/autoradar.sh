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

# Tailscale CLI (GUI app bundle first, then PATH)
TS_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
[ -x "$TS_CLI" ] || TS_CLI="$(command -v tailscale)"

# --- child PIDs we manage --------------------------------------------------
WEBAPP_PID=""; BOT_PID=""

cleanup() {
  log "Shutting down…"
  [ -n "$BOT_PID" ] && kill "$BOT_PID" 2>/dev/null
  [ -n "$WEBAPP_PID" ] && kill "$WEBAPP_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# --- prerequisites: bring up Docker Desktop & Tailscale, then wait ----------
# launchd starts us at login before these background apps are ready, so we
# launch and wait for each one instead of assuming it is already running.

ensure_docker() {
  command -v docker >/dev/null 2>&1 || { log "WARN: docker not installed — skipping"; return 1; }
  if docker info >/dev/null 2>&1; then return 0; fi
  log "Docker daemon down — launching Docker Desktop…"
  open -a Docker >/dev/null 2>&1 || true
  for i in $(seq 1 60); do
    docker info >/dev/null 2>&1 && { log "Docker ready (after $((i*2))s)"; return 0; }
    sleep 2
  done
  log "WARN: Docker did not become ready in ~120s"; return 1
}

ensure_tailscale() {
  [ -n "$TS_CLI" ] && [ -x "$TS_CLI" ] || { log "WARN: tailscale CLI not found — skipping"; return 1; }
  if "$TS_CLI" status >/dev/null 2>&1; then return 0; fi   # already connected
  log "Tailscale stopped — launching app & connecting…"
  open -a Tailscale >/dev/null 2>&1 || true
  sleep 3
  "$TS_CLI" up >/dev/null 2>&1 || true
  for i in $(seq 1 30); do
    "$TS_CLI" status >/dev/null 2>&1 && { log "Tailscale connected (after $((i*2))s)"; return 0; }
    sleep 2
  done
  log "WARN: Tailscale did not connect in ~60s (Funnel will be retried anyway)"; return 1
}

# --- 1. dashboard + scheduler (docker) -------------------------------------
# Backgrounded: the `init` service (first scrape+train) can take minutes, and
# the Mini App / bot read the DB + model straight off disk, so they must not
# wait on docker.
if ensure_docker; then
  log "Starting docker compose (dashboard + scheduler) in background…"
  ( docker compose up -d >>"$LOG" 2>&1 || log "WARN: docker compose up failed" ) &
else
  log "WARN: Docker unavailable — skipping dashboard/scheduler"
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
if [ -z "${WEBAPP_URL:-}" ]; then
  log "FATAL: WEBAPP_URL not set in .env"; exit 1
fi
echo "$WEBAPP_URL" >"$URL_FILE"

ensure_tailscale
if [ -n "$TS_CLI" ] && [ -x "$TS_CLI" ]; then
  log "Asserting Tailscale Funnel → :$WEBAPP_PORT"
  "$TS_CLI" funnel --bg "$WEBAPP_PORT" >>"$LOG" 2>&1 \
    || log "WARN: 'tailscale funnel' failed (Tailscale down or Funnel disabled?)"
else
  log "WARN: tailscale CLI not found — assuming Funnel already configured"
fi

# --- 4. Telegram bot, supervised (fixed URL) -------------------------------
log "Starting bot.main with WEBAPP_URL=$WEBAPP_URL"
start_bot() {
  while true; do
    WEBAPP_URL="$WEBAPP_URL" "$PY" -m bot.main >>"$STATE_DIR/bot.log" 2>&1
    log "bot.main exited — restarting in 3s"
    sleep 3
  done
}
start_bot & BOT_PID=$!

# --- 5. Tailscale relay watchdog --------------------------------------------
# This ISP occasionally intercepts the specific DERP relay tailscaled picked
# (self-signed cert instead of the real one), which breaks Funnel's inbound
# NAT traversal until tailscaled is restarted and picks a different relay.
# Detect that health condition and self-heal automatically. (Separate issue
# from the ISP blocking *.ts.net by SNI outright — that one needs Funnel
# traffic to route over the private tailnet instead, e.g. via Tailscale on
# the client device; restarting tailscaled here can't fix that case.)
watch_tailscale_health() {
  [ -n "$TS_CLI" ] && [ -x "$TS_CLI" ] || return 0
  while true; do
    sleep 120
    if "$TS_CLI" status 2>&1 | grep -qi "intercepted connection"; then
      log "Tailscale relay intercepted by ISP — restarting to pick a clean relay…"
      "$TS_CLI" down >>"$LOG" 2>&1
      sleep 3
      "$TS_CLI" up >>"$LOG" 2>&1
      sleep 3
      "$TS_CLI" funnel --bg "$WEBAPP_PORT" >>"$LOG" 2>&1
    fi
  done
}
watch_tailscale_health &

wait
