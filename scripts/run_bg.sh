#!/bin/sh
# 后台跑重活 + 只看摘要，避免 Agent 在一轮里干等十几分钟。
#
#   scripts/run_bg.sh pytest -- python3 -m pytest -q     # 启动（-- 之后的都是命令）
#   scripts/run_bg.sh --status                           # 所有任务状态
#   scripts/run_bg.sh --tail pytest 30                   # 看日志尾部
#   scripts/run_bg.sh --wait pytest 900                  # 阻塞等待（最多 900s）
#   scripts/run_bg.sh --list                             # 任务名列表
#
# 日志/PID/退出码都放在 ${BG_DIR:-/tmp/quant-bg}。
set -u

DIR="${BG_DIR:-/tmp/quant-bg}"
mkdir -p "$DIR"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

alive() { [ -f "$DIR/$1.pid" ] && kill -0 "$(cat "$DIR/$1.pid")" 2>/dev/null; }

case "${1:-}" in
  ""|-h|--help) usage ;;
  --list) ls "$DIR"/*.pid 2>/dev/null | sed "s|$DIR/||;s|\.pid$||" ;;
  --status)
    found=0
    for p in "$DIR"/*.pid; do
      [ -e "$p" ] || continue
      found=1
      n=$(basename "$p" .pid)
      if alive "$n"; then
        printf '%-20s 运行中  pid=%s  %ss  最后: %s\n' "$n" "$(cat "$p")" \
          "$(( $(date +%s) - $(stat -c %Y "$p" 2>/dev/null || echo 0) ))" \
          "$(tail -n 1 "$DIR/$n.log" 2>/dev/null | cut -c1-70)"
      else
        code=$(cat "$DIR/$n.exit" 2>/dev/null || echo '?')
        printf '%-20s 已结束 exit=%s  最后: %s\n' "$n" "$code" \
          "$(tail -n 1 "$DIR/$n.log" 2>/dev/null | cut -c1-70)"
      fi
    done
    [ "$found" = 1 ] || echo "（没有任务）"
    ;;
  --tail)
    n="${2:?用法: --tail <name> [行数]}"; k="${3:-20}"
    tail -n "$k" "$DIR/$n.log" 2>/dev/null || { echo "没有 $n 的日志"; exit 1; }
    ;;
  --wait)
    n="${2:?用法: --wait <name> [超时秒]}"; t="${3:-900}"; i=0
    while alive "$n"; do
      [ "$i" -ge "$t" ] && { echo "等待超时（${t}s），$n 仍在跑"; exit 2; }
      sleep 2; i=$((i + 2))
    done
    echo "$n 已结束 exit=$(cat "$DIR/$n.exit" 2>/dev/null || echo '?')  用时约 ${i}s"
    tail -n 20 "$DIR/$n.log" 2>/dev/null
    ;;
  *)
    n="$1"; shift
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "缺少命令：run_bg.sh <name> -- <cmd...>" >&2; exit 2; }
    if alive "$n"; then
      echo "同名任务已在跑：$n (pid $(cat "$DIR/$n.pid"))，日志 $DIR/$n.log" >&2; exit 1
    fi
    rm -f "$DIR/$n.exit"
    : > "$DIR/$n.log"
    { ( "$@" ) >>"$DIR/$n.log" 2>&1; echo $? > "$DIR/$n.exit"; } &
    echo $! > "$DIR/$n.pid"
    echo "已启动 $n  pid=$(cat "$DIR/$n.pid")"
    echo "日志 $DIR/$n.log"
    echo "查看 $0 --tail $n 20    等待 $0 --wait $n 900    状态 $0 --status"
    ;;
esac
