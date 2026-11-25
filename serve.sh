#!/bin/bash
# Ternary Bonsai 2 27B (V100 sm_70 optimized, router-mode server with Vision + DFlash2).
# Usage: ./serve.sh {start|stop|restart|status|load|unload|key} [model]
D="$(cd "$(dirname "$0")" && pwd)"
BIN="$D/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="$D/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PORT="${BONSAI_PORT:-8110}"
HOST="${BONSAI_HOST:-0.0.0.0}"
LOG="$D/router.log"
KEYFILE="${LLAMA_API_KEY_FILE:-$D/.api-key}"
MODEL="${2:-bonsai-2-27b}"   # load/unload target: bonsai-2-27b | bonsai-2.5-27b

[ -n "${LLAMA_API_KEY:-}" ] && (umask 077; printf "%s\n" "$LLAMA_API_KEY" > "$KEYFILE")
if [ ! -s "$KEYFILE" ]; then
  if [ -s "$D/.api_key" ]; then
    cp -p "$D/.api_key" "$KEYFILE"
  else
    (umask 077; openssl rand -hex 24 > "$KEYFILE")
  fi
fi
ln -sf ".api-key" "$D/.api_key" 2>/dev/null || true

loaded() {
  curl -s -H "Authorization: Bearer $(head -1 "$KEYFILE")" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -o '"value": *"loaded"' | wc -l
}

pid_cmd() { pgrep -f "$BIN --models-preset $D/models.ini"; }

start() {
  if pid_cmd >/dev/null; then echo "already running"; return 0; fi
  # Stop any legacy standalone instance if running
  pkill -f "$BIN -m .*Ternary-Bonsai-2-27B" 2>/dev/null || true
  setsid nohup "$BIN" --models-preset "$D/models.ini" --models-max 1 \
    --api-key-file "$KEYFILE" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 30); do
    if curl -s -H "Authorization: Bearer $(head -1 "$KEYFILE")" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "ready: http://localhost:$PORT/v1  (router mode, loads on first request: bonsai-2-27b / bonsai-2.5-27b)"
      return 0
    fi
    pid_cmd >/dev/null || { echo "failed to start, see $LOG"; return 1; }
    sleep 1
  done
  echo "timeout, see $LOG"; return 1
}

stop() {
  pkill -f "$BIN --models-preset $D/models.ini" && echo "stopped" || echo "not running"
  pkill -f "$BIN -m .*Ternary-Bonsai-2-27B" 2>/dev/null || true
  sleep 3
}

status() {
  if pid_cmd >/dev/null; then
    echo "router: running (:$PORT), loaded models: $(loaded)/1"
  else
    echo "router: stopped"
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

load_model() {
  curl -s -X POST "http://127.0.0.1:$PORT/models/load" \
    -H "Authorization: Bearer $(head -1 "$KEYFILE")" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\"}"
  echo
}

unload_model() {
  curl -s -X POST "http://127.0.0.1:$PORT/models/unload" \
    -H "Authorization: Bearer $(head -1 "$KEYFILE")" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\"}"
  echo
}

case "${1:-status}" in
  start) start;;
  stop) stop;;
  restart) stop; start;;
  status) status;;
  load) load_model;;
  unload) unload_model;;
  key) cat "$KEYFILE";;
  *) echo "usage: $0 {start|stop|restart|status|load|unload|key} [model]"; exit 1;;
esac
