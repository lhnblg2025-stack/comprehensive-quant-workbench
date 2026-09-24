#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""health_guardian.py — 监控的监控（HealthGuardian，独立于主系统）

阶段1 待办 A3：系统"监控本身"缺乏监控（复盘扫描断链需人工发现）。
本进程每 N 分钟检查：
  1. 腾讯云 schtasks 最近运行时间（quant_pipeline_daily/quant_industry_sw/quant_data_health/quant_rt_snapshot）
  2. 腾讯云 C:\\quant\\generated 每日关键产出物（battle_map/multi_agent/macro_veto/
     emotion/data_health/rs_strength/pattern_report）
  3. 本机 quant-web 8600 存活
  4. 本机白名单进程内存（RSS>2GB 连续3次告警防 OOM）
断链 → 飞书告警（经 alert_dedup 网关去重，2026-08-14：同签名 2h 内静默）。

用法:
  python3 scripts/health_guardian.py          # 单次检查（openclaw 网关 cron 每5分钟）
  python3 scripts/health_guardian.py --loop   # 常驻循环
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import psutil
except ImportError:  # 可选依赖；缺失时降级到 /proc/[pid]/status
    psutil = None

ROOT = Path(os.environ.get("QUANT_ROOT", str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
RUN_DIR = Path(os.environ.get("QUANT_RUN_DIR", str(ROOT / "generated" / "run"))).expanduser()
STATE_FILE = Path(os.environ.get("HEALTH_GUARDIAN_STATE", str(ROOT / "data_warehouse" / "health_guardian_state.json"))).expanduser()
LOG_FILE = Path(os.environ.get("HEALTH_GUARDIAN_LOG", str(ROOT / "generated" / "logs" / "health_guardian.log"))).expanduser()
LOCK_FILE = Path(os.environ.get("HEALTH_GUARDIAN_LOCK", str(RUN_DIR / "health_guardian.lock"))).expanduser()

# 进程内存监控白名单（按 cmdline 关键字匹配）
MEM_WATCH_KEYWORDS = (
    "realtime_snapshot.py",
    "update_kline_tencent.py",
    "update_valuation_baostock.py",
    "temperature_replay.py",
    "calibrate_position_map.py",
    "quant_web",
)
try:
    MEM_WARN_MB = int(os.environ.get("HEALTH_MEM_WARN_MB", "2048") or "2048")
except ValueError:
    MEM_WARN_MB = 2048
MEM_OVER_COUNT_KEY = "mem_over_count"

# 腾讯云 schtasks 关键任务（阶段1：上云首轮验证后启用）
TX_TASKS = {
    "quant_pipeline_daily": {"max_age_h": 30, "note": "11体系综合复盘", "weekday_only": True},
    "quant_industry_sw": {"max_age_h": 30, "note": "行业数据更新", "weekday_only": True},
    "quant_data_health": {"max_age_h": 26, "note": "数据健康看板"},
    "quant_rt_snapshot": {"max_age_h": 6, "note": "盘中快照(交易日)"},
}

# 腾讯云 ssh 访问（Windows OpenSSH）
CLOUD_HOST = os.environ.get("QUANT_CLOUD_HOST", "").strip()
CLOUD_PEM = os.environ.get("QUANT_CLOUD_KEY", "").strip()
KNOWN_HOSTS = Path(os.environ.get("QUANT_KNOWN_HOSTS", str(Path.home() / ".ssh" / "known_hosts"))).expanduser()
CLOUD_GENERATED = os.environ.get("QUANT_CLOUD_GENERATED", r"C:\quant\generated")
# P1-3 新增探得的云端产物目录（2026-08-15 实探，非猜测）：
#  - CLOUD_MNT_REPORT : quant_html_report 输出 A股多因子复盘日报_<YYYY-MM-DD>.html
#  - CLOUD_OUT_REPORT : quant_daily_report 输出 <YYYY-MM-DD>-A股量化日报.md
#  - CLOUD_OUT_SNAPSHOT: quant_snapshot  输出 daily_snapshot.{parquet,json}
#  - CLOUD_DATA_MARKET: quant_lhb_daily/quant_index_daily 输出 lhb_*.parquet / index_daily_*.parquet
CLOUD_MNT_REPORT = r"C:\mnt\hgfs\研报共享"
CLOUD_OUT_REPORT = r"C:\quant\out\每日任务\A股量化日报"
CLOUD_OUT_SNAPSHOT = r"C:\quant\out\daily_snapshot"
CLOUD_DATA_MARKET = r"C:\quant\data_warehouse\market"

# 本机日志（复盘扫描断链检测，支持 glob 模式）
# daily_report/html_report/after_close 已于 8/12 迁往腾讯云 schtasks，
# 由云端任务自身保证，本机不再检查其日志。
LOCAL_LOGS = []

# 每日关键产出物账本（文件名中的日期应等于最近交易日；来源=腾讯云）
# - required 产物带 grace_days：云端从未产出时从首次检查起给宽限期，避免"修复未上云前"刷屏；
#   出现过后再缺失则立即告警。
# - optional 且云端无历史产出时跳过。
# - after_hour：该小时之前不检查当天产物（避免凌晨把"应为今天"当作缺失误报）。
# - dir：产物所在云端目录（缺省 CLOUD_GENERATED）；P1-3 起支持 C:\quant 下的其他产物目录。
ARTIFACT_CHECKS = (
    # 盘后产物（18:35-20:30 生成）：after_hour=19.0 保证 19:00 前不检查当天产物，
    # 避免凌晨误报"产物缺失应为今天"。19:00 后到次日凌晨检查当天产物正常。
    {"name": "battle_map", "patterns": ("battle_map_*.md",), "grace_days": 2, "after_hour": 19.0},
    {"name": "multi_agent", "patterns": ("multi_agent_*.json",), "grace_days": 2, "after_hour": 19.0},
    {"name": "macro_veto", "patterns": ("macro_veto_*.json",), "grace_days": 2, "after_hour": 19.0},
    {"name": "emotion_report", "patterns": ("emotion_report_*.json", "emotion_report_*.md"),
     "optional": True, "module": "emotion_system.py", "after_hour": 19.0},
    {"name": "data_health", "patterns": ("data_health_*.md",), "after_hour": 20.5},
    {"name": "rs_strength", "patterns": ("rs_strength_*.md",), "optional": True, "after_hour": 19.0},
    {"name": "pattern_report", "patterns": ("pattern_report_*.md",), "optional": True, "after_hour": 19.0},
    # P1-3 实探补录（2026-08-15）：quant_html_report(17:30) / quant_daily_report(16:30) /
    # quant_meta_reviewer(19:30) 的日期型文件名产物，此前无产物级守护。
    # 目录与文件名均已 ssh dir 实探确认，非猜测。
    # optional=True: 与 rs_strength/pattern_report 同款"无历史则跳过、有历史才守护"，
    # 避免新条目在无历史/测试场景误报；云端实测已存在历史(2026-08-14) → 会真实启用。
    {"name": "html_report", "patterns": ("*多因子复盘日报_*.html",), "dir": CLOUD_MNT_REPORT,
     "optional": True, "after_hour": 18.0, "note": "quant_html_report"},
    {"name": "daily_report", "patterns": ("*-A股量化日报.md",), "dir": CLOUD_OUT_REPORT,
     "optional": True, "after_hour": 16.5, "note": "quant_daily_report"},
    {"name": "meta_review", "patterns": ("meta_review_*.md", "meta_review_*.json"),
     "dir": r"C:\quant\generated\meta_review", "recursive": True, "optional": True,
     "after_hour": 20.0, "note": "quant_meta_reviewer(generated\\meta_review\\<YYYYMMDD>\\ 下)"},
)

# P1-3 实探：云端"结果 0 但产物 mtime 陈旧"的 mtime 型产物（文件名不含日期，需按最新 mtime 判新鲜度）。
# after_min_hour：当日该小时后才检查（避免把"当天还没到生成点"误报为陈旧）；
# max_age_h：产物最新 mtime 距今超过该值即视为陈旧。
# dir_family/mtime 探针均以"目录下按 mtime 最新匹配文件"为准，天然规避季度滚动命名边界问题。
DAILY_PRODUCT_MTIME_CHECKS = (
    {"name": "msnapshot", "dir": CLOUD_OUT_SNAPSHOT, "pattern": "summary.json",
     "after_min_hour": 18.0, "max_age_h": 48, "note": "quant_snapshot(C:\\quant\\out\\daily_snapshot)"},
    {"name": "kline", "dir": r"C:\quant\data_warehouse\kline", "pattern": "*.parquet",
     "after_min_hour": 18.0, "max_age_h": 60, "note": "quant_after_close 盘后K线更新"},
    {"name": "lhb_daily", "dir": CLOUD_DATA_MARKET, "pattern": r"lhb_*.parquet",
     "after_min_hour": 18.0, "max_age_h": 48, "note": "quant_lhb_daily(季滚动文件取最新 mtime)"},
    {"name": "index_daily", "dir": CLOUD_DATA_MARKET, "pattern": r"index_daily_*.parquet",
     "after_min_hour": 17.0, "max_age_h": 48, "note": "quant_index_daily(逐指数 parquet 取最新 mtime)"},
)


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _feishu(text: str) -> None:
    """飞书告警（复用 feishu_sender，失败不阻断）"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from feishu_sender import send_simple_text  # noqa: PLC0415
        send_simple_text(text, title="⚠️ HealthGuardian 告警")
    except Exception as e:  # noqa: BLE001
        _log(f"飞书告警失败: {e}")


def _match_mem_keyword(cmdline: object) -> str | None:
    """返回 cmdline 命中的第一个监控关键字；无命中返回 None。"""
    if isinstance(cmdline, str):
        text = cmdline
    else:
        text = " ".join(str(part) for part in (cmdline or []))
    for keyword in MEM_WATCH_KEYWORDS:
        if keyword in text:
            return keyword
    return None


def _proc_label(pid: object, keyword: str) -> str:
    return f"{keyword}(pid={pid})"


def _collect_process_rss_psutil() -> list[tuple[str, float]]:
    """用 psutil 扫描白名单进程，返回 [(进程标签, RSS_MB), ...]。"""
    results: list[tuple[str, float]] = []
    for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            info = proc.info
            cmdline = info.get("cmdline") or []
            keyword = _match_mem_keyword(cmdline)
            if not keyword:
                continue
            mem_info = info.get("memory_info")
            rss_bytes = int(getattr(mem_info, "rss", 0) or 0)
            if rss_bytes <= 0:
                continue
            results.append((_proc_label(info.get("pid"), keyword), rss_bytes / 1024 / 1024))
        except Exception:  # noqa: BLE001 - 单进程退出/权限不足不应影响其他进程扫描
            continue
    return results


def _parse_vmrss_kb(status_text: str) -> int | None:
    """解析 /proc/[pid]/status 中的 VmRSS（单位 kB）。"""
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lower() == "kb":
                try:
                    return int(parts[-2])
                except ValueError:
                    return None
    return None


def _collect_process_rss_proc(proc_root: Path | None = None) -> list[tuple[str, float]]:
    """降级方案：遍历 /proc/[pid]，解析 cmdline 与 status 的 VmRSS。"""
    root = proc_root or Path("/proc")
    results: list[tuple[str, float]] = []
    try:
        entries = list(root.iterdir())
    except OSError as e:
        _log(f"/proc 进程扫描失败: {e}")
        return results
    for pid_dir in entries:
        if not pid_dir.name.isdigit():
            continue
        try:
            raw_cmdline = (pid_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
        keyword = _match_mem_keyword(raw_cmdline.split("\0"))
        if not keyword:
            continue
        try:
            status_text = (pid_dir / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rss_kb = _parse_vmrss_kb(status_text)
        if rss_kb is None:
            continue
        results.append((_proc_label(pid_dir.name, keyword), rss_kb / 1024))
    return results


def _collect_process_rss() -> list[tuple[str, float]]:
    """扫描白名单进程 RSS；psutil 失败时降级 /proc，均失败则返回空列表。"""
    if psutil is not None:
        try:
            return _collect_process_rss_psutil()
        except Exception as e:  # noqa: BLE001
            _log(f"psutil 进程扫描失败，降级 /proc: {e}")
    return _collect_process_rss_proc()


def _is_weekend_run(last_run: str) -> bool:
    """上次运行时间是否落在周六/周日（2026-08-23 新增周末豁免）。
    非交易日空数据/空接口导致的退出1属正常业务现象, 不告警。"""
    try:
        ds = str(last_run).strip().split(" ")[0]
        dt = datetime.strptime(ds, "%Y/%m/%d")
        return dt.weekday() >= 5
    except Exception:  # noqa: BLE001
        return False


def _cloud_config_error() -> str | None:
    if not CLOUD_HOST or not CLOUD_PEM:
        return "QUANT_CLOUD_HOST/QUANT_CLOUD_KEY 未配置"
    if not KNOWN_HOSTS.is_file():
        return f"known_hosts 不存在: {KNOWN_HOSTS}"
    return None


def _cloud_ssh(cmd: str, timeout: int = 20) -> tuple[int, list[str]] | None:
    """ssh 在腾讯云执行 Windows 命令，返回 (退出码, stdout 行列表)。

    连接失败/超时/不可达返回 None（调用方据此跳过检查，避免误报）。
    Windows cmd 输出为 GBK，按 gbk→utf-8 依次尝试解码。
    """
    config_error = _cloud_config_error()
    if config_error:
        _log(f"腾讯云 ssh 配置错误: {config_error}")
        return None
    try:
        raw = subprocess.run(
            ["ssh", "-i", CLOUD_PEM, "-o", "StrictHostKeyChecking=yes",
             "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
             "-o", "ConnectTimeout=10", CLOUD_HOST, cmd],
            capture_output=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - 超时/网络异常不阻断，跳过即可
        _log(f"腾讯云 ssh 不可达: {e}")
        return None
    if raw.returncode == 255:  # ssh 自身连接失败（非远程命令失败）
        _log(f"腾讯云 ssh 连接失败(255): {cmd}")
        return None
    text = raw.stdout.decode("gbk", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return raw.returncode, lines


def _cloud_reachable() -> bool:
    """探测腾讯云 ssh 是否可用；产出物检查前置，不可达时跳过避免误报。"""
    return _cloud_ssh(f'dir /b "{CLOUD_GENERATED}"') is not None


def _cloud_web_ok() -> bool | None:
    """探测腾讯云 quant-web 8600 是否健康（None=无法探测/ssh失败）。"""
    r = _cloud_ssh('curl -s -o nul -w "%{http_code}" http://127.0.0.1:8600/ 2>nul', timeout=25)
    if r is None:
        return None
    rc, lines = r
    if rc != 0 or not lines:
        return None
    try:
        code = int(lines[-1].strip())
    except ValueError:
        return None
    return code == 200


def _cloud_artifact_latest(pattern: str, directory: str | None = None,
                           recursive: bool = False) -> date | None:
    """ssh 拉取云端 <directory>\\<pattern> 文件列表，解析最新文件名日期。

    directory 缺省为 CLOUD_GENERATED（P1-3 起允许指向 C:\\quant 下其他产物目录）；
    recursive=True 时用 `dir /s /b`（用于 meta_review 日期子目录嵌套产物）。
    ssh 失败/不可达返回 None（配合 _cloud_reachable 整体跳过，不误报）；
    云端无匹配文件同样返回 None（此时按"产出物缺失"告警）。
    """
    base = directory or CLOUD_GENERATED
    listing = ("dir", "/s", "/b") if recursive else ("dir", "/b")
    result = _cloud_ssh(f'{listing[0]} {listing[1]} "{base}\\{pattern}"')
    if result is None:
        return None
    _, lines = result
    dates: list[date] = []
    for line in lines:
        if line.lower().startswith("file not found"):
            continue
        parsed = _parse_artifact_date(Path(line))
        if parsed is not None:
            dates.append(parsed)
    return max(dates) if dates else None


def _cloud_latest_artifacts(patterns: tuple[str, ...], directory: str | None = None,
                            recursive: bool = False) -> date | None:
    """多个 glob 模式取最新日期（用于 emotion_report 多扩展名/meta_review 多类型）。"""
    latest: date | None = None
    for pattern in patterns:
        day = _cloud_artifact_latest(pattern, directory, recursive)
        if day is not None and (latest is None or day > latest):
            latest = day
    return latest


def _check_artifacts(state: dict) -> list[str]:
    """检查腾讯云 C:\\quant\\generated 下每日关键产出物是否存在且日期新鲜。

    仅在最近自然日是交易日时校验；周末/节假日自动跳过。交易日历优先复用
    quant_system.data_quality.get_trade_calendar，不可用时按周末粗筛降级。
    ssh 不可达时跳过产出物检查（不误报）。
    """
    today = _artifact_now().date()
    calendar = _load_trade_calendar()
    if not _is_trading_day(today, calendar):
        return []
    if not _cloud_reachable():
        # Optional remote host unavailable: skip this secondary check. The
        # resident node's local release gate remains the production authority.
        return []

    now = _artifact_now()
    alerts: list[str] = []
    for cfg in ARTIFACT_CHECKS:
        if cfg.get("optional") and not _optional_artifact_enabled(cfg):
            continue
        after_hour = cfg.get("after_hour")
        if after_hour is not None and now.hour + now.minute / 60 < after_hour:
            continue
        latest = _cloud_latest_artifacts(cfg["patterns"], cfg.get("dir"), cfg.get("recursive") is True)
        grace_days = cfg.get("grace_days") or 0
        if latest is None and grace_days and not _artifact_grace_expired(
                cfg["name"], today, state, grace_days):
            continue
        if latest is None or latest < today:
            latest_text = latest.isoformat() if latest else "无"
            alerts.append(f"⚠️ 产出物缺失: {cfg['name']} 最新 {latest_text}, 应为 {today.isoformat()}")
    return alerts


def _cloud_mtime_latest(directory: str, pattern: str) -> datetime | None:
    """ssh 拉取云端 <directory> 下匹配 pattern 的文件最新 mtime（目录 + 文件名拼接）。

    用 PowerShell Get-ChildItem 取按 LastWriteTime 最新的一个文件。
    ssh 失败/不可达/无匹配返回 None（调用方据此跳过或按陈旧告警）。
    """
    # 用 .ToString('yyyy-MM-dd HH:mm:ss') 强制统一格式，规避中文区域(年/月/日)差异
    ps = ("powershell -NoProfile -Command \"Get-ChildItem -Path '{dir}' -Filter '{pat}' "
          "-File | Sort-Object LastWriteTime -Descending | Select-Object -First 1 "
          "-ExpandProperty LastWriteTime | ForEach-Object {{ $_.ToString('yyyy-MM-dd HH:mm:ss') }}\""
          ).format(dir=directory, pat=pattern)
    result = _cloud_ssh(ps, timeout=25)
    if result is None:
        return None
    rc, lines = result
    if rc != 0 or not lines:
        return None
    # 取第一行解析 "yyyy-MM-dd HH:mm:ss"
    raw = lines[0].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _check_cloud_daily_products(state: dict) -> list[str]:
    """P1-3: "结果 0 但产物 mtime 陈旧"的 mtime 型每日产物新鲜度检查。

    针对 quant_snapshot / quant_after_close(K线) / quant_lhb_daily / quant_index_daily：
    文件名不含日期、无法用 `_check_artifacts` 的日期新鲜度判断，改按最新 mtime 判定。
    - after_min_hour: 当日在该小时前不检查（避免把"当天还没到生成点"误报为陈旧）；
    - max_age_h: 最新 mtime 距今超过即告警（周末/节假日产物不更新时靠此宽容）。
    仅在最近自然日(周末粗筛)&&ssh 可达时检查；不可达跳过不误报。
    """
    today = _artifact_now().date()
    calendar = _load_trade_calendar()
    if not _is_trading_day(today, calendar):
        # 非交易日(周末)产物本就停更，直接跳过避免陈旧误报
        return []
    if not _cloud_reachable():
        # The remote Windows node is optional. A resident Linux node validates
        # its own release/artifacts locally, so do not emit a false outage for
        # an intentionally unconfigured secondary host.
        return []

    now = _artifact_now()
    alerts: list[str] = []
    for cfg in DAILY_PRODUCT_MTIME_CHECKS:
        if now.hour + now.minute / 60 < cfg["after_min_hour"]:
            continue
        latest = _cloud_mtime_latest(cfg["dir"], cfg["pattern"])
        if latest is None:
            # 目录/文件不存在 → "无产物"（缺失，独立于陈旧）
            alerts.append(f"⚠️ 产出物缺失: {cfg['name']}({cfg['note']}) 目录下无 {cfg['pattern']}")
            continue
        age_h = (now - latest).total_seconds() / 3600
        if age_h > cfg["max_age_h"]:
            alerts.append(
                f"⚠️ 产物陈旧: {cfg['name']}({cfg['note']}) 最新 {latest:%m-%d %H:%M} "
                f"(> {cfg['max_age_h']}h 未更新)")
    return alerts


def _artifact_now() -> datetime:
    """当前时间（拆出便于测试注入固定日期）。"""
    return datetime.now()


def _load_trade_calendar() -> set[str] | None:
    """加载交易日历；失败返回 None，由调用方降级为周末粗筛。"""
    try:
        sys.path.insert(0, str(ROOT))
        from quant_system.data_quality import get_trade_calendar  # noqa: PLC0415

        calendar = get_trade_calendar()
        if calendar:
            return {str(day) for day in calendar}
    except Exception as e:  # noqa: BLE001
        _log(f"交易日历加载失败，产出物检查降级为周末粗筛: {e}")
    return None


def _is_trading_day(day: date, calendar: set[str] | None) -> bool:
    if calendar:
        return day.isoformat() in calendar
    return day.weekday() < 5


def _parse_artifact_date(path: Path) -> date | None:
    """从文件名解析日期。

    兼容两类命名：
      - 日期后缀式: battle_map_2026-08-12 / data_health_20260812 / meta_review_2026-08-14
      - 日期前缀式: 2026-08-14-A股量化日报（quant_daily_report）
    统一策略：取 stem 中第一个形如 YYYY-MM-DD 或 YYYYMMDD 且校验合法的日期片段。
    中国汉字文件名经 GBK 往返会乱码，但日期片段是纯 ASCII，不影响解析。
    """
    stem = path.stem
    for fmt in (r"%Y-%m-%d", r"%Y%m%d"):
        needle_chars = ("-" if fmt == r"%Y-%m-%d" else "",)
        # YYYY-MM-DD 长度 10；YYYYMMDD 长度 8
        win_size = 10 if fmt == r"%Y-%m-%d" else 8
        for i in range(len(stem) - win_size + 1):
            frag = stem[i:i + win_size]
            if not frag.replace("-", "").isdigit():
                continue
            try:
                return datetime.strptime(frag, fmt).date()
            except ValueError:
                continue
    return None


def _artifact_grace_expired(name: str, today: date, state: dict, grace_days: int) -> bool:
    """required 产物从未出现过时，从首次检查起给 grace_days 天宽限。

    首次检查记录 first_seen 并返回 False（宽限内不告警）；宽限期满仍未出现返回 True；
    产物出现过（latest 非 None）后走严格检查，不经过此函数。
    """
    bucket = state.setdefault("artifact_grace", {}).setdefault(name, {})
    first = bucket.get("first_seen")
    if not first:
        bucket["first_seen"] = today.isoformat()
        return False
    try:
        first_dt = date.fromisoformat(first)
    except ValueError:
        bucket["first_seen"] = today.isoformat()
        return False
    return (today - first_dt).days >= grace_days


def _optional_artifact_enabled(cfg: dict) -> bool:
    """可选产出物检查启用条件：云端已有该产物历史（从未产出过则跳过）。

    cfg 带 module 键时，本机对应模块缺失也跳过（避免把"未部署模块"误报为缺失）。
    """
    module = cfg.get("module")
    if module and not (ROOT / "quant_system" / "analysis_core" / module).exists():
        return False
    return _cloud_latest_artifacts(cfg["patterns"], cfg.get("dir"),
                                   cfg.get("recursive") is True) is not None


def _check_process_memory(state: dict) -> list[str]:
    """检查白名单单进程 RSS；同进程连续 3 次超阈值才告警，恢复后重置计数。"""
    try:
        procs = _collect_process_rss()
    except Exception as e:  # noqa: BLE001
        _log(f"进程内存检查跳过: {e}")
        return []

    last_mem = state.setdefault("last_mem", {})
    over_count = state.setdefault(MEM_OVER_COUNT_KEY, {})
    now = datetime.now().isoformat(timespec="seconds")
    alerts: list[str] = []

    for proc_name, rss_mb in procs:
        last_mem[proc_name] = {"rss_mb": round(rss_mb, 2), "ts": now}
        if rss_mb > MEM_WARN_MB:
            over_count[proc_name] = int(over_count.get(proc_name, 0)) + 1
            if over_count[proc_name] == 3:
                alerts.append(f"⚠️ OOM风险: {proc_name} RSS {rss_mb:.0f}MB > {MEM_WARN_MB}MB")
        else:
            over_count[proc_name] = 0
    return alerts


def _rt_snapshot_due(now: datetime) -> bool:
    """盘中快照仅在交易日 9:30-15:05 检查（盘前/盘后/周末/节假日跳过，避免误报）。"""
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    if not (9 * 60 + 30 <= minutes <= 15 * 60 + 5):
        return False
    return _is_trading_day(now.date(), _load_trade_calendar())


def _check_tx_schtasks(state: dict) -> list[str]:
    """检查腾讯云 schtasks 最近运行时间（需本机已配置 ssh key）"""
    alerts = []
    now = datetime.now()
    for task, cfg in TX_TASKS.items():
        if task == "quant_rt_snapshot" and not _rt_snapshot_due(now):
            continue
        if cfg.get("weekday_only"):
            # W2.5 飞书刷屏修复：交易日任务周末停更 + 当日 19:00 运行窗口前不判陈旧，
            # 否则周六日/周一早会每小时误报"上次运行 45h 前"→ 飞书告警刷屏
            if now.weekday() >= 5 or now.hour < 19:
                continue
        last = state.get("tx", {}).get(task)
        if not last:
            continue  # 首轮只记录，不告警
        last_dt = datetime.fromisoformat(last)
        age_h = (now - last_dt).total_seconds() / 3600
        if age_h > cfg["max_age_h"]:
            alerts.append(f"腾讯云 {task}({cfg['note']}) 上次运行 {age_h:.0f}h 前（> {cfg['max_age_h']}h）")
    return alerts


def _check_local_logs(state: dict) -> list[str]:
    """检查本机日志是否持续更新（支持 glob 模式取最新文件）"""
    import glob  # noqa: PLC0415
    alerts = []
    now = datetime.now()
    for pattern, cfg in LOCAL_LOGS:
        d = cfg.get("dir", str(ROOT / "generated" / "logs"))
        matches = sorted(glob.glob(str(Path(d) / pattern)), key=lambda p: Path(p).stat().st_mtime)
        if not matches:
            alerts.append(f"日志缺失: {d}/{pattern}({cfg['note']})")
            continue
        p = Path(matches[-1])
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        age_h = (now - mtime).total_seconds() / 3600
        if age_h > cfg["max_age_h"]:
            alerts.append(f"{cfg['note']} 日志 {age_h:.0f}h 未更新（{p}）")
    return alerts


def _check_local_kline_freshness() -> list[str]:
    """检查本机仓库 K线文件最近写入时间（盘后更新链的黄金信号）。

    2026-08-14: 网关 cron PATH 无 anaconda 导致本机 K线更新静默失败一整天
    （data_warehouse/kline 停在 8/12）而 health_guardian 毫无察觉——日志检查
    只能看到"启动"行，看不到子进程崩溃。此检查直接盯产物 mtime：
    交易日 17:30 后 K线最新 mtime 超过 30h → 告警。
    """
    try:
        kdir = ROOT / "data_warehouse" / "kline"
        if not kdir.exists():
            return ["本机仓库 K线目录缺失 (data_warehouse/kline)——盘后更新链未落盘"]
        files = list(kdir.glob("*.parquet"))
        if not files:
            return ["本机仓库 K线目录为空 (data_warehouse/kline)——盘后更新链未落盘"]
        latest = max(p.stat().st_mtime for p in files)
        age_h = (time.time() - latest) / 3600
        # 交易日且已过 17:30（盘后更新窗口后）才严格检查
        now = datetime.now()
        if now.weekday() < 5 and now.hour >= 18 and age_h > 30:
            return [f"本机仓库K线 {age_h:.0f}h 未更新（最新 {datetime.fromtimestamp(latest):%m-%d %H:%M}）——盘后更新链可能静默失败"]
    except Exception as e:  # noqa: BLE001
        return [f"K线新鲜度检查失败: {e}"]
    return []


def _check_core_data_health() -> list[str]:
    """读取最新 data_health 产物，核心基座降级即告警（不重复全量扫描）。"""
    try:
        import re as _re
        files = [p for p in ROOT.glob("generated/data_health_*.json")
                 if _re.fullmatch(r"data_health_\d{4}-\d{2}-\d{2}\.json", p.name)]
        if not files:
            return ["⚠️ 数据健康: 无 data_health 产物（数据健康 digest 未运行）"]
        d = json.loads(sorted(files)[-1].read_text(encoding="utf-8"))
        if d.get("overall_ok"):
            return []
        degraded = [c.get("name", "?") for c in d.get("checks", []) if not c.get("ok")]
        if not degraded:
            return [f"⚠️ 数据健康降级（ok_count={d.get('ok_count')}）但详情缺失，请复核 data_health 产物"]
        return [f"⚠️ 数据基座降级 {d.get('ok_count', 0)}/{d.get('total_checks', 0)}: " + ", ".join(degraded[:8])]
    except Exception as e:  # noqa: BLE001
        return [f"数据健康检查失败: {e}"]


def _check_web() -> list[str]:
    """检查本机 quant-web 8600 存活"""
    try:
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                            "--max-time", "5", "http://127.0.0.1:8600/"],
                           capture_output=True, text=True, timeout=10)
        if r.stdout.strip() != "200":
            return ["quant-web 8600 非 200: " + r.stdout.strip()]
    except Exception as e:  # noqa: BLE001
        return [f"quant-web 检查失败: {e}"]
    return []


# 2026-08-14: openclaw 网关 cron 全部 headless 任务引用的脚本清单。
# 历史教训：8/13 归档 68 脚本时误删 4 个仍被 cron 引用的脚本（quant_scan_alert/
# pull_cloud_data/enrich_hot_rank/desktop_audit）→ 任务每天静默失败。此检查
# 在"再被归档/删除"时立即告警，防止同坑复发。
CRON_SCRIPT_CHECKS = (
    "scripts/health_guardian.py",
    "scripts/factor_extreme_alert.py",
    "scripts/after_close_update.sh",
    "scripts/update_kline_tencent.py",
    "scripts/update_valuation_baostock.py",
    "scripts/pull_cloud_data.py",
    "scripts/pull_cloud_reports.py",
    "scripts/enrich_hot_rank.py",
    "scripts/stamp_data_freshness.py",
    "scripts/data_health_digest.py",
    "scripts/desktop_audit.py",
    "scripts/refresh_factor_direction.py",
    "scripts/ic_vectorized.py",
)


def _check_cron_scripts() -> list[str]:
    """检查 openclaw 网关 cron 引用的脚本都存在（防归档误删后静默失败）。"""
    missing = [s for s in CRON_SCRIPT_CHECKS if not (ROOT / s).exists()]
    if missing:
        return ["cron 引用脚本缺失（任务将静默失败）: " + ", ".join(missing)]
    return []


def _update_tx_state() -> None:
    """从腾讯云拉取 schtasks 最近运行时间（尽力而为）"""
    try:
        config_error = _cloud_config_error()
        if config_error:
            _log(f"腾讯云状态更新跳过: {config_error}")
            return
        cmd = "schtasks /query /fo CSV /v /nh 2>nul"
        # 第一次调用 text=True 会在 UTF-8 解码时崩（GBK 输出），改为直接 bytes
        raw = subprocess.run(["ssh", "-i", CLOUD_PEM, "-o", "BatchMode=yes",
                              "-o", "StrictHostKeyChecking=yes",
                              "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                              "-o", "ConnectTimeout=10", CLOUD_HOST, cmd],
                             capture_output=True, timeout=30)
        if raw.returncode != 0:
            return
        import csv  # noqa: PLC0415
        import io  # noqa: PLC0415
        for enc in ("gbk", "utf-8"):
            try:
                text = raw.stdout.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        rows = list(csv.reader(io.StringIO(text)))
        # /v 模式列: [1]=TaskName, [5]=LastRunTime, [6]=LastResult
        for row in rows:
            if len(row) < 7:
                continue
            task = row[1].strip().strip('"').replace("\\", "")
            if task in TX_TASKS:
                last_run = row[5].strip().strip('"')
                if last_run and last_run != "N/A":
                    try:
                        dt = datetime.strptime(last_run, "%Y/%m/%d %H:%M:%S")
                        # Windows 新建任务未运行过显示 1999/11/30 → 跳过（首轮不告警）
                        if dt.year < 2020:
                            continue
                        state = _load_state()
                        state.setdefault("tx", {})[task] = dt.isoformat()
                        _save_state(state)
                    except ValueError:
                        pass
    except Exception as e:  # noqa: BLE001
        _log(f"腾讯云状态拉取失败: {e}")


def _check_cloud_task_results(state: dict) -> list[str]:
    """云端 schtasks 执行结果检查: Last Result != 0x0 且任务曾运行 → 告警。

    2026-08-15 新增(评分短板④): 任务失败需主动告警闭环, 不能只靠产物检查
    (产物可能由旧数据生成, 掩盖任务失败)。失败集合记录进 state,
    新增失败告警, 恢复后自动解除。
    """
    out = _cloud_ssh('schtasks /query /fo csv /v', timeout=30)
    if out is None:
        # 审计修复(2026-08-24): 云端不可达 → UNKNOWN 显式告警，禁止静默返回空。
        return ["⚠️ UNKNOWN: 腾讯云 SSH 不可达，云端任务状态无法核验"]
    rc, lines = out
    if rc != 0 or not lines:
        return ["⚠️ UNKNOWN: 云端 schtasks 查询失败(rc={rc})，任务状态无法核验".format(rc=rc or "?")]
    failures: list[str] = []
    for line in lines:
        if "quant_" not in line and "crawler_" not in line and "cninfo_" not in line:
            continue
        # CSV /v 列序(GBK): 0主机名,1任务名,2下次运行,3模式,4登录,5上次运行时间,
        # 6上次结果,7作者,8要运行任务,9开始,10计划类型,11计划任务状态
        # V12.3 审计 P1-1: csv.reader 解析(防命令含逗号错位), 状态取列11(非模式列3)
        import csv as _csv
        try:
            _parts = next(_csv.reader([line]))
        except Exception:  # noqa: BLE001
            continue
        parts = _parts
        if len(parts) < 12:
            continue
        task = parts[1].strip('"')
        status = parts[11].strip('"')  # 计划任务状态(列11)
        last_run = parts[5].strip('"')
        last_result = parts[6].strip('"')
        # 已禁用/从未运行/成功 跳过
        if status == "已禁用" or last_result in ("N/A", "", "0x0", "0"):
            continue
        try:
            code = int(last_result, 16) if last_result.lower().startswith("0x") else int(last_result)
        except ValueError:
            continue
        # 排除非失败错误码: 0x41303 未运行 / 0x41301 运行中 / 0x41306 被同实例忽略
        if code in (267011, 267009, 267014):
            continue
        # 已知超时重任务(非核心)豁免告警: ic_direction(全量因子超时)/opp_1530(重扫描)
        if code != 0 and task.lstrip("\\") not in ("quant_ic_direction", "quant_opp_1530",
                                                    "quant_ic_run2", "quant_ic_run3"):
            # 2026-08-23 稳定化: 周末(周六/日)任务失败多为非交易日空数据/空接口致退出1,
            # 属正常业务现象(实测 08-22 周六 5 任务全失败、20+ 工作日全成功) → 豁免告警
            if _is_weekend_run(last_run):
                state.setdefault("weekend_skips", {})[task] = f"周末空数据疑似(0x{code:X} {last_run})"
                state.setdefault("task_failures", {}).pop(task, None)  # 清除历史周末条目
                _log(f"{task} 周末失败 0x{code:X} {last_run} → 豁免告警(非交易日空数据)")
                continue
            # 保活任务 quant_web_ensure 失败但云端 web 仍 200 → 视为"保活已生效"，
            # 不告警（该任务本身可能因前一实例/返回码问题失败，服务仍健康）
            if task.lstrip("\\") == "quant_web_ensure":
                web_ok = _cloud_web_ok()
                if web_ok is True:
                    state.setdefault("task_failures", {}).pop("quant_web_ensure", None)
                    _log(f"quant_web_ensure 上次失败 0x{code:X}，但云端 web 200 → 豁免")
                    continue
                if web_ok is None:
                    # 无法探测云端 web：保留告警但标注待核
                    failures.append(f"{task} 上次运行失败 (0x{code:X}, {last_run}) [云端web探测不可用]")
                    continue
            failures.append(f"{task} 上次运行失败 (0x{code:X}, {last_run})")
    if not failures:
        prev = state.get("task_failures", {})
        if prev:
            # 2026-08-23: 清理历史周末假失败(如 08-22 周六 5 任务) -- 周末条目移入 weekend_skips
            wk = state.setdefault("weekend_skips", {})
            kept = {}
            for t, desc in prev.items():
                if _is_weekend_run(desc):
                    wk[t] = f"周末假失败已清理:{desc[:60]}"
                else:
                    kept[t] = desc
            if len(kept) != len(prev):
                state["task_failures"] = kept
                _log(f"已清理 {len(prev)-len(kept)} 条周末假失败告警")
            elif not kept:
                state["task_failures"] = {}
                _log("云端任务失败已全部恢复, 解除告警标记")
        return []
    # 只告警新增/变化的失败
    prev = state.get("task_failures", {})
    alerts = []
    for f in failures:
        task = f.split(" ")[0]
        if prev.get(task) != f:
            alerts.append(f"❌ {f}")
    state["task_failures"] = {f.split(" ")[0]: f for f in failures}
    _log(f"云端任务失败 {len(failures)} 个: {failures}")
    return alerts


# P1-4: 云端 config/.env.secrets 存在性 + 必需 KEY 探针（fail-loud）
# 背景: 云端 .env.secrets 靠 sync_to_cloud.sh 手动同步、无探针, 缺失/损坏时飞书链路
# 静默退化(此前靠硬编码兜底)。守护需在配置缺位时告警而非掩蔽。
# 只校验"文件存在 + 必需 key 名出现", 绝不打印/回传 key 的 value。
CLOUD_SECRETS_PATH = r"C:\quant\config\.env.secrets"
CLOUD_SECRETS_REQUIRED_KEYS = ("FEISHU_APP_SECRET", "FEISHU_CHAT_ID")


def _check_cloud_secrets() -> list[str]:
    """探针: 云端 config/.env.secrets 存在且含必需飞书 KEY；缺失直接告警(fail-loud)。

    - 文件不存在 → "配置缺失"告警（此前静默兜底掩盖该问题）；
    - 文件存在但缺必需 KEY → "内容缺失"告警；
    - ssh 不可达 → 跳过不误报（与产出物检查一致的健壮性策略）。
    """
    if not _cloud_reachable():
        return []
    # 1) 存在性
    exist_cmd = (
        f'cmd /c "if exist {CLOUD_SECRETS_PATH} (echo PRESENT) else (echo MISSING)"'
    )
    res = _cloud_ssh(exist_cmd)
    if res is None:
        # 审计修复(2026-08-24): 云端 SSH 不可达 → UNKNOWN，禁止静默返回空
        return ["⚠️ UNKNOWN: 腾讯云 SSH 不可达，云端密钥存在性无法核验"]
    _, lines = res
    present = any("PRESENT" in line.upper() for line in lines)
    if not present:
        return ["❌ 云端密钥缺失: config/.env.secrets 不存在 —— 飞书链路将 fail-loud，"
                "请先运行 sync_to_cloud.sh 同步"]

    # 2) 必需 key 名校验（findstr /i /b 按行首匹配 key 名，不回传 value）
    missing = []
    for key in CLOUD_SECRETS_REQUIRED_KEYS:
        check = _cloud_ssh(
            f'findstr /i /b "{key}" {CLOUD_SECRETS_PATH} >nul 2>&1'
        )
        if check is None:
            return ["⚠️ UNKNOWN: 腾讯云 SSH 不可达，云端密钥 key 校验无法完成"]  # 审计修复: 禁止静默跳过
        if check[0] != 0:
            missing.append(key)
    if missing:
        return [f"❌ 云端密钥内容缺失: config/.env.secrets 缺必需 KEY: {', '.join(missing)} —— "
                "飞书链路将 fail-loud，请补齐后重新同步"]
    return []


# ── 自愈：失败任务自动重触发（2026-08-21 用户要求"报错要自己修改"）──
_REMEDY_TASKS = {  # 失败可安全重跑的任务（重计算/长任务除外）
    "quant_daily_report": 1, "quant_after_close": 1,
    "quant_lhb_daily": 1, "quant_index_daily": 1, "quant_html_report": 1,
    "quant_margin_detail": 1, "quant_review_fusion": 1,
    "quant_opp_0945": 1, "quant_opp_1000": 1,
}


def _remedy_cloud_tasks() -> list[str]:
    """检测云端失败任务 → 自动触发重跑（最多每任务每天1次，写入 state）。"""
    try:
        state = _load_state()
        remedied = state.get("remedy_log", {})
        today = datetime.now().strftime("%Y-%m-%d")  # 本地时区即可，自愈按天维度
        out = []
        # 已由 _check_cloud_task_results 得到的 failure 列表复用
        import subprocess as _sp
        config_error = _cloud_config_error()
        if config_error:
            return [f"自愈未执行: {config_error}"]
        ssh_base = ["ssh", "-i", CLOUD_PEM, "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=yes",
                    "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                    "-o", "ConnectTimeout=12", CLOUD_HOST]
        res = _sp.run(
            [*ssh_base, 'cmd /c schtasks /query /fo csv 2>nul | findstr /i "quant_"'],
            capture_output=True, timeout=50)
        txt = res.stdout.decode("gbk", errors="replace")
        for line in txt.splitlines():
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 3:
                continue
            name = parts[0].lstrip("\\")
            if name not in _REMEDY_TASKS:
                continue
            # 只处理明确失败（状态列非就绪）
            status = parts[2] if len(parts) > 2 else ""
            if "失败" not in status and "准备就绪" not in status and "就绪" not in status:
                continue
            key = f"{name}:{today}"
            if remedied.get(key):
                continue
            r2 = _sp.run([*ssh_base, f'schtasks /run /tn {name}'],
                         capture_output=True, timeout=40)
            _so = r2.stdout.decode("gbk", errors="replace").upper()
            ok = ("成功" in _so) or ("SUCCESS" in _so)
            remedied[key] = {"ts": datetime.now().isoformat(), "ok": ok}
            out.append(f"🔧 自愈触发: {name} {'✅' if ok else '❌'}")
        state["remedy_log"] = remedied
        _save_state(state)
        return out
    except Exception as e:  # noqa: BLE001
        return [f"自愈失败: {str(e)[:80]}"]


_FAST = False  # 云端/高并发场景: 只查关键产物+任务失败(2026-08-21 防超时被杀)


def run_once() -> int:
    state = _load_state()
    alerts = []
    cloud_error = _cloud_config_error()
    cloud_enabled = cloud_error is None
    if _FAST:
        # Fast mode always validates local data; remote checks are optional so a
        # self-hosted resident node does not fail merely for lacking legacy SSH.
        if cloud_enabled:
            alerts += _check_cloud_daily_products(state)
            alerts += _check_cloud_task_results(state)
        alerts += _check_core_data_health()
    else:
        alerts += _check_local_logs(state)
        alerts += _check_web()
        if cloud_enabled:
            alerts += _check_tx_schtasks(state)
        alerts += _check_process_memory(state)
        alerts += _check_artifacts(state)
        if cloud_enabled:
            alerts += _check_cloud_daily_products(state)
        alerts += _check_cron_scripts()
        alerts += _check_local_kline_freshness()
        alerts += _check_core_data_health()
        if cloud_enabled:
            alerts += _check_cloud_task_results(state)
            alerts += _check_cloud_secrets()
    try:
        _save_state(state)
    except Exception as e:  # noqa: BLE001
        _log(f"健康状态保存失败: {e}")
    # 拉取腾讯云最新状态（每次都拉，写入 state 供下次对比）
    _update_tx_state()

    # 结构化告警落盘（供前端/审计追溯，不再只有飞书文本）
    try:
        out_dir = ROOT / "generated" / "health_guardian"
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "fast" if _FAST else "full",
            "ok": not alerts,
            "alert_count": len(alerts),
            "alerts": alerts,
        }
        (out_dir / "alerts_latest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        _log(f"告警快照落盘失败: {e}")

    if alerts:
        msg = "\n".join(alerts)
        _log("告警:\n" + msg)
        # 自愈: 先尝试自动修复再告警
        remedies = _remedy_cloud_tasks()
        if remedies:
            msg += "\n\n" + "\n".join(remedies)
        _send_deduped_alert(msg)
        # A failed check must remain visible to the scheduler; the structured
        # snapshot and Feishu alert retain the detailed diagnostics.
        return 1
    _log("全部正常")
    return 0


def _send_deduped_alert(msg: str) -> None:
    """经 alert_dedup 网关发送告警。

    2026-08-14 审计修复: 旧版每次告警都直发飞书 → 持续故障时每5分钟刷屏。
    现按内容签名去重: 新签名立即发, 同签名 2h 内静默, 超 2h 重发一次提醒;
    夜间(23-8点)非 danger 静默; 每小时硬顶 10 条。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from alert_dedup import get_deduper, make_sig, normalize_price  # noqa: PLC0415
        sig = make_sig("health_guardian", normalize_price(msg))
        # 内存超限/严重告警夜间也要发
        level = "danger" if ("内存" in msg or "OOM" in msg.upper()) else "warn"
        if get_deduper().should_send("health_guardian", sig, level=level):
            _feishu("系统健康异常:\n" + msg)
        else:
            _log("告警去重: 同签名在静默窗口内, 跳过飞书推送")
    except Exception as e:  # noqa: BLE001 - 去重失败不应阻断告警
        _log(f"告警去重失败(直发兜底): {e}")
        _feishu("系统健康异常:\n" + msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="HealthGuardian 监控的监控")
    ap.add_argument("--loop", action="store_true", help="常驻循环（默认单次）")
    ap.add_argument("--interval", type=int, default=300, help="循环间隔秒（默认 300）")
    ap.add_argument("--fast", action="store_true", help="快速模式: 只查云产物+任务失败(防超时)")
    args = ap.parse_args()
    global _FAST
    _FAST = bool(getattr(args, "fast", False))
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_FILE.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log(f"已有 HealthGuardian 实例运行: {LOCK_FILE}")
        return 75
    if not args.loop:
        return run_once()
    while True:
        try:
            run_once()
        except Exception as e:  # noqa: BLE001
            _log(f"运行异常: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
