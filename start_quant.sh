#!/bin/bash
# ==========================================================
#   A股量化工作台 · 唯一启动入口
#   用法: bash start_quant.sh
# ==========================================================
set -u

ROOT="${QUANT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${QUANT_PYTHON:-$(command -v python3)}"
PORT="${QUANT_WEB_PORT:-8600}"
RUNTIME_DIR="$ROOT/generated/run"
LOG="${QUANT_WEB_LOG:-$RUNTIME_DIR/quant-web.log}"
PIDFILE="${QUANT_WEB_PIDFILE:-$RUNTIME_DIR/quant-web.pid}"
BIND_HOST="${QUANT_WEB_BIND_HOST:-127.0.0.1}"
# 本机回环启动默认开启匿名开发模式；云端通过 EnvironmentFile 提供 API key。
if [ "$BIND_HOST" = "127.0.0.1" ] || [ "$BIND_HOST" = "localhost" ] || [ "$BIND_HOST" = "::1" ]; then
  export QUANT_WEB_ALLOW_UNAUTH="${QUANT_WEB_ALLOW_UNAUTH:-1}"
fi
SERVER_CMD="${PYTHON_BIN} ${ROOT}/quant_web/server.py ${PORT}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python 运行时不存在: $PYTHON_BIN"
  exit 2
fi
if [ "$BIND_HOST" != "127.0.0.1" ] && [ "$BIND_HOST" != "localhost" ] && [ "$BIND_HOST" != "::1" ] && [ -z "${QUANT_WEB_API_KEY:-}" ]; then
  echo "❌ 拒绝启动：非回环地址必须设置 QUANT_WEB_API_KEY"
  exit 2
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
touch "$LOG"

is_owned_pid() {
  local pid="$1"
  [ -r "/proc/$pid/cmdline" ] || return 1
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$cmd" == *"quant_web/server.py $PORT"* ]] || return 1
  # proc cmdline may contain the absolute workspace path; resolve identity below.
  # Keep the explicit script/path check as the authoritative ownership test.
  [[ "$cmd" == *"python"* ]] || return 1
  # Resolve the script argument against the process cwd so both historical
  # relative invocations and the current absolute invocation are recognized.
  local script_arg script_path proc_cwd
  script_arg=$(printf '%s\n' "$cmd" | sed -n 's#.* \([^ ]*quant_web/server.py\) [0-9][0-9]*.*#\1#p')
  proc_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  if [[ "$script_arg" = /* ]]; then
    script_path=$(readlink -f "$script_arg" 2>/dev/null || true)
  else
    script_path=$(readlink -f "$proc_cwd/$script_arg" 2>/dev/null || true)
  fi
  [ "$script_path" = "$ROOT/quant_web/server.py" ] || return 1
}

stop_owned_pid() {
  local pid="$1"
  if is_owned_pid "$pid"; then
    echo "🛑 停止受本入口管理的旧服务 PID $pid..."
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# 只处理 PID 文件中由本入口启动的实例；不再无条件 fuser，避免误杀其他服务。
if [ -s "$PIDFILE" ]; then
  OLD_PID=$(head -1 "$PIDFILE" | tr -cd '0-9')
  [ -n "$OLD_PID" ] && stop_owned_pid "$OLD_PID"
fi
: > "$PIDFILE"

# 若端口已有实例，只有在命令行确认是本服务时才停止；否则安全退出并提示。
PORT_PIDS=$(fuser -n tcp "$PORT" 2>/dev/null || true)
for pid in $PORT_PIDS; do
  if is_owned_pid "$pid"; then
    stop_owned_pid "$pid"
  else
    echo "❌ 端口 $PORT 已被其他进程占用（PID $pid），未强杀。"
    echo "   请检查: ss -ltnp | grep :$PORT"
    exit 3
  fi
done

# 清理本服务字节码不是启动必需条件，保持失败也不阻断服务。
find "$ROOT/quant_system/__pycache__" "$ROOT/quant_web/__pycache__" -name '*.pyc' -delete 2>/dev/null || true

cd "$ROOT"
echo "🚀 启动唯一 quant_web 实例: $SERVER_CMD"
QUANT_WEB_BIND_HOST="$BIND_HOST" PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" "$ROOT/quant_web/server.py" "$PORT" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

# 只等轻量 liveness；以服务端返回的真实 Python PID 校验身份，避免记录错误的父进程 PID。
for i in $(seq 1 20); do
  sleep 1
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:${PORT}/api/livez" 2>/dev/null || true)
  VERSION_PID=$(curl -s --max-time 2 "http://127.0.0.1:${PORT}/api/version" 2>/dev/null | sed -n 's/.*"pid":[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  if [ "$CODE" = "200" ] && [ -n "$VERSION_PID" ] && is_owned_pid "$VERSION_PID"; then
    echo "$VERSION_PID" > "$PIDFILE"
    echo "✅ 服务已监听且身份一致 (${i}s) → http://${BIND_HOST}:${PORT}"
    echo "📋 日志: $LOG"
    echo "🔎 服务版本: http://127.0.0.1:${PORT}/api/version"
    exit 0
  fi
done

# 启动子进程未成功时不留下失真的 PID 文件；保留日志供定位。
: > "$PIDFILE"
echo "❌ 20秒仍未确认由本入口管理的 quant_web 服务，最近日志:"
tail -30 "$LOG"
exit 1
