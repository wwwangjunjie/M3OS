#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${M3OS_MAIN_MCP_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
SERVER_SCRIPT="${PROJECT_ROOT}/mcp_server_improved.py"
DEFAULT_PORT=8010
PORT="${MCP_SERVER_PORT:-$DEFAULT_PORT}"
HOST="${MCP_SERVER_HOST:-127.0.0.1}"
REPLACE=false

while (($#)); do
  case "$1" in
    --replace)
      REPLACE=true
      shift
      ;;
    --port)
      [[ $# -ge 2 ]] || { printf '%s\n' '--port requires a value' >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    [0-9]*)
      PORT="$1"
      shift
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  printf 'Invalid port: %s\n' "$PORT" >&2
  exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { printf 'Python is not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
[[ -f "$SERVER_SCRIPT" ]] || { printf 'Server script not found: %s\n' "$SERVER_SCRIPT" >&2; exit 1; }

RUNTIME_DIR="${PROJECT_ROOT}/tmp/main_mcp"
PID_FILE="${RUNTIME_DIR}/main_mcp.pid"
LOG_FILE="${RUNTIME_DIR}/main_mcp.log"
mkdir -p "$RUNTIME_DIR"

process_is_managed_server() {
  local pid="$1"
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq -- "$SERVER_SCRIPT"
}

stop_managed_server() {
  local pid="$1"
  kill -TERM "$pid"
  for _ in {1..50}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -KILL "$pid"
}

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] \
    && kill -0 "$existing_pid" 2>/dev/null \
    && process_is_managed_server "$existing_pid"; then
    if [[ "$REPLACE" == false ]]; then
      printf 'M3OS main MCP is already running (PID %s).\n' "$existing_pid"
      exit 0
    fi
    printf 'Stopping existing M3OS main MCP (PID %s).\n' "$existing_pid"
    stop_managed_server "$existing_pid"
  fi
fi

setsid "$PYTHON_BIN" "$SERVER_SCRIPT" \
  --transport streamable-http \
  --host "$HOST" \
  --port "$PORT" \
  > "$LOG_FILE" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$PID_FILE"

# Do not report a successful launch if Python fails immediately (bad import,
# occupied port, invalid transport, and similar startup errors).
sleep 1
if ! kill -0 "$server_pid" 2>/dev/null; then
  printf 'M3OS main MCP exited during startup. See: %s\n' "$LOG_FILE" >&2
  rm -f "$PID_FILE"
  if [[ -s "$LOG_FILE" ]]; then
    tail -n 40 "$LOG_FILE" >&2
  fi
  exit 1
fi

printf 'M3OS main MCP started. PID: %s, URL: http://%s:%s/mcp\n' \
  "$server_pid" "$HOST" "$PORT"
printf 'Log: %s\n' "$LOG_FILE"
