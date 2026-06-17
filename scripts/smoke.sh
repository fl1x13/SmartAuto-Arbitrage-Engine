#!/usr/bin/env bash
# Smoke test for car-arbitrage: seeds DB, trains model, checks dashboard HTTP 200.
# Run from project root: bash scripts/smoke.sh
set -euo pipefail

PORT=${1:-8510}
cd "$(dirname "$0")/.."   # project root

# Isolated DB + model dir: never pollute data/car_market.db or model/artifacts
# (they may hold real auto.ru data and a model trained on it)
SMOKE_DB="/tmp/car_market_smoke.db"
rm -f "$SMOKE_DB"
export DATABASE_URL="sqlite:///$SMOKE_DB"
export MODEL_DIR="/tmp/car_market_smoke_model"
rm -rf "$MODEL_DIR"

echo "=== 1/3  Seed DB (isolated: $SMOKE_DB) ==="
python3 -m scraper.seed

echo "=== 2/3  Train model ==="
python3 -m model.train 2>&1 | grep -E "MAPE|RMSE|saved"

echo "=== 3/3  Dashboard health ==="
python3 -m streamlit run app/app.py \
  --server.headless true \
  --server.port "$PORT" \
  --server.address 127.0.0.1 > /tmp/streamlit_smoke.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 20); do
  sleep 1
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then
    echo "  Server up on :$PORT (after ${i}s)"
    break
  fi
  if [ "$i" -eq 20 ]; then
    echo "  TIMEOUT: server did not start" >&2
    cat /tmp/streamlit_smoke.log >&2
    kill "$SERVER_PID" 2>/dev/null
    exit 1
  fi
done

# Check page loads and has no Python error in title
TITLE=$(curl -s "http://127.0.0.1:$PORT/" | grep -o '<title>[^<]*</title>')
if echo "$TITLE" | grep -qi "error"; then
  echo "  ERROR: page title indicates an error: $TITLE" >&2
  kill "$SERVER_PID" 2>/dev/null
  exit 1
fi

# Wait for app script to execute (Streamlit lazy-runs on first WebSocket connection)
sleep 8
if grep -q "Uncaught app execution" /tmp/streamlit_smoke.log; then
  echo "  ERROR: app raised an exception:" >&2
  grep -A 10 "Uncaught app execution" /tmp/streamlit_smoke.log >&2
  kill "$SERVER_PID" 2>/dev/null
  exit 1
fi

kill "$SERVER_PID" 2>/dev/null
echo "=== All smoke checks passed ==="
