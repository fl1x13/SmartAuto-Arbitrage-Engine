#!/bin/bash
# Autoradar supervisor — keeps the whole stack up with zero manual steps.
#
#   * dashboard + scheduler        -> docker compose (background, self-healing)
#   * Mini App API (bot.webapp)    -> local :8050, restarted if it dies
#   * public HTTPS URL             -> auto-picked from a list of free tunnel
#                                     providers (Tailscale Funnel first, then
#                                     localhost.run), health-checked and switched
#                                     automatically if the ISP blocks the current
#                                     one (Russian DPI blocklists known tunnel
#                                     domains one at a time, not all at once).
#   * Telegram bot (bot.main)      -> restarted whenever the active public URL
#                                     changes, so its Mini App button always
#                                     points at whatever currently works.
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
BOT_PID_FILE="$STATE_DIR/bot.pid"

PY="$(command -v python3)"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG" >&2; }

# --- load .env -------------------------------------------------------------
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a; . "$PROJECT_DIR/.env"; set +a
else
  log "FATAL: .env not found"; exit 1
fi
WEBAPP_PORT="${WEBAPP_PORT:-8050}"
PRIMARY_URL="${WEBAPP_URL:-}"   # the Tailscale Funnel URL configured in .env

# Tailscale CLI (GUI app bundle first, then PATH)
TS_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
[ -x "$TS_CLI" ] || TS_CLI="$(command -v tailscale)"

# --- child PIDs we manage --------------------------------------------------
WEBAPP_PID=""; BOT_LOOP_PID=""

cleanup() {
  log "Shutting down…"
  [ -n "$BOT_LOOP_PID" ] && kill "$BOT_LOOP_PID" 2>/dev/null
  [ -f "$BOT_PID_FILE" ] && kill "$(cat "$BOT_PID_FILE")" 2>/dev/null
  [ -n "$WEBAPP_PID" ] && kill "$WEBAPP_PID" 2>/dev/null
  stop_fallback_tunnels
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

# give the API a moment to bind before any tunnel points at it
sleep 5

# --- 3. public HTTPS URL: try providers in order, keep whichever works -----
# The ISP (Beeline) has blocked trycloudflare.com and ngrok.io before, and now
# blocks Tailscale's *.ts.net by SNI too — it blocks one well-known tunnel
# domain at a time, not the whole internet. So instead of betting on one
# provider, we health-check the primary Funnel URL and fall back to free
# SSH-based tunnels (no account, no domain needed) when it's unreachable.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=20 -o ExitOnForwardFailure=yes -o ConnectTimeout=8"

health_ok() {
  local url="$1"
  [ -n "$url" ] || return 1
  # Require the tunnel to return *our* Mini App, not just any HTTP 200 — some
  # providers answer 200 with their own login/interstitial page. Retry a few
  # times: free tunnels (lhr.life) are occasionally slow, and a single timed-out
  # probe must NOT tear down an otherwise-working tunnel (that caused the URL to
  # churn to a new random subdomain every cycle).
  local i
  for i in 1 2 3; do
    if curl -s --max-time 12 "$url/" 2>/dev/null | grep -q "<title>Авторадар</title>"; then
      return 0
    fi
    sleep 3
  done
  return 1
}

assert_tailscale_funnel() {
  ensure_tailscale
  if [ -n "$TS_CLI" ] && [ -x "$TS_CLI" ]; then
    "$TS_CLI" funnel --bg "$WEBAPP_PORT" >>"$LOG" 2>&1 \
      || log "WARN: 'tailscale funnel' failed (Tailscale down or Funnel disabled?)"
  fi
}

# If this key is registered at https://admin.localhost.run/ the assigned
# *.lhr.life subdomain is PERMANENT (survives reconnects); otherwise we connect
# anonymously and get a fresh random subdomain each time. Either way it works.
LHR_KEY="$HOME/.ssh/lhr_tunnel_ed25519"

start_localhost_run() {
  pkill -f "ssh.*localhost.run" 2>/dev/null
  rm -f "$STATE_DIR/lhr.log"
  if [ -f "$LHR_KEY" ]; then
    ssh -i "$LHR_KEY" $SSH_OPTS -R 80:localhost:"$WEBAPP_PORT" localhost.run >"$STATE_DIR/lhr.log" 2>&1 &
  else
    ssh $SSH_OPTS -R 80:localhost:"$WEBAPP_PORT" nokey@localhost.run >"$STATE_DIR/lhr.log" 2>&1 &
  fi
  local pid=$!
  for i in $(seq 1 15); do
    url=$(grep -oE 'https://[a-zA-Z0-9.-]+\.lhr\.life' "$STATE_DIR/lhr.log" 2>/dev/null | head -1)
    [ -n "$url" ] && { echo "$url"; return 0; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill "$pid" 2>/dev/null
  return 1
}

stop_fallback_tunnels() {
  pkill -f "ssh.*localhost.run" 2>/dev/null
}

# Tries, in order: the primary Tailscale Funnel URL, then localhost.run. Prints
# the first one that actually answers with HTTP 200 *and serves our app* (not a
# provider login/interstitial page), or nothing if all are blocked/down.
# (serveo.net used to be a no-account fallback but now forces sign-in, so it is
# intentionally not in the chain — it would only show a login page.)
pick_working_url() {
  assert_tailscale_funnel
  if health_ok "$PRIMARY_URL"; then
    stop_fallback_tunnels
    echo "$PRIMARY_URL"
    return 0
  fi
  log "Primary URL ($PRIMARY_URL) unreachable — trying localhost.run…"
  local lhr; lhr=$(start_localhost_run)
  if health_ok "$lhr"; then echo "$lhr"; return 0; fi
  return 1
}

if [ -z "$PRIMARY_URL" ]; then
  log "FATAL: WEBAPP_URL not set in .env"; exit 1
fi

ACTIVE_URL="$(pick_working_url)"
if [ -z "$ACTIVE_URL" ]; then
  log "WARN: no public tunnel reachable right now — keeping primary URL, will keep retrying"
  ACTIVE_URL="$PRIMARY_URL"
fi
echo "$ACTIVE_URL" >"$URL_FILE"
log "Active public URL: $ACTIVE_URL"

# --- 4. Telegram bot, restarted whenever the active URL changes -----------
start_bot() {
  while true; do
    local current_url
    current_url="$(cat "$URL_FILE" 2>/dev/null)"
    log "Starting bot.main with WEBAPP_URL=$current_url"
    WEBAPP_URL="$current_url" "$PY" -m bot.main >>"$STATE_DIR/bot.log" 2>&1 &
    local pid=$!
    echo "$pid" >"$BOT_PID_FILE"
    wait "$pid"
    log "bot.main exited — restarting in 3s"
    sleep 3
  done
}
start_bot & BOT_LOOP_PID=$!

# --- 5. public-URL watchdog --------------------------------------------------
# Every 90s, re-check the active URL. If it stopped working, try the fallback
# chain again and switch (updating the URL file + restarting bot.main) the
# moment something else starts working. If the primary Tailscale URL comes
# back, switch back to it and drop the SSH fallback tunnels.
watch_url_health() {
  while true; do
    sleep 90
    local cur
    cur="$(cat "$URL_FILE" 2>/dev/null)"
    if health_ok "$cur"; then
      continue
    fi
    log "Active URL ($cur) stopped responding — re-picking a working tunnel…"
    local new_url
    new_url="$(pick_working_url)"
    if [ -z "$new_url" ]; then
      log "WARN: still no public tunnel reachable — keeping $cur, will retry"
      continue
    fi
    if [ "$new_url" != "$cur" ]; then
      log "Switching public URL: $cur -> $new_url"
    fi
    echo "$new_url" >"$URL_FILE"
    [ -f "$BOT_PID_FILE" ] && kill "$(cat "$BOT_PID_FILE")" 2>/dev/null
  done
}
watch_url_health &

# --- 6. Tailscale relay watchdog --------------------------------------------
# This ISP occasionally intercepts the specific DERP relay tailscaled picked
# (self-signed cert instead of the real one), which breaks Funnel's inbound
# NAT traversal until tailscaled is restarted and picks a different relay.
# Detect that health condition and self-heal automatically. (Separate issue
# from the ISP blocking *.ts.net by SNI outright, which watch_url_health
# above handles by switching to a fallback tunnel instead.)
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
