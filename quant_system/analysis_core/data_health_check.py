"""
data_health_check — 数据健康检查（V11 决策可信度地基）

盘前/盘后全链路运行前先自检：每个数据集的新鲜度、规模、完整性，
输出 trust_scores 供决策层降权（数据不新鲜 → 依赖它的模块自动降权）。

检查项:
  1. zt_daily_stats   最新日期 vs 最近交易日（情绪周期/天梯地基）
  2. zt_pool_history  行数 + 日期覆盖（全量 vs 被覆盖的近期表）
  3. zt_pool_em_daily 东财三池最新日期
  4. fund_forces      四路资金最新日期
  5. theme_cycle      题材周期最新日期
  6. fusion           信号融合最新日期
  7. social_sentiment 社交情绪是否有可用数据
  8. 关键 parquet 文件大小突变检测（与昨日快照对比，防静默丢失）
  9. baostock 登录可用性（慢测，可跳过）
  10. 名称映射 stock_names 新鲜度

输出: generated/data_health_{date}.json + 状态快照 generated/data_health_state.json
  决策层读取 trust_scores 对模块降权:
    - 数据集滞后 >2 交易日 → 对应模块 weight * 0.5
    - 文件大小突变 >30% → weight * 0.5 并告警

用法:
  python3 -m quant_system.analysis_core.data_health_check [--skip-baostock]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import (  # noqa: E402
    MARKET_DIR, ZT_HISTORY, ZT_EM_DAILY, ZT_DAILY_STATS, NAME_MAP,
)

CST = timezone(timedelta(hours=8))
STATE_FILE = ROOT / "generated" / "data_health_state.json"
DW = MARKET_DIR.parent  # data_warehouse 根

# 数据集 → 依赖它的模块（降权映射）
DATASET_MODULES = {
    "zt_daily_stats": ["emotion_cycle", "ladder", "scenario", "decision_card", "battle_map"],
    "zt_pool_history": ["ladder", "theme_cycle", "short_term_extra", "battle_map"],
    "zt_pool_em_daily": ["short_term_extra", "battle_map"],
    "fund_forces": ["fund_forces", "fusion", "battle_map"],
    "theme_cycle": ["theme_cycle", "fusion", "resonance_scorer", "battle_map"],
    "fusion": ["fusion", "battle_map"],
    "social_sentiment": ["social_sentiment", "battle_map"],
    "stock_names": ["zt_pool_history", "broker_gaming"],
    # 基础数据目录（2026-08-23 W0.7 补盲，按各自数据节奏设阈值）
    "kline": ["factors", "ml", "backtest"],
    "valuation": ["valuation", "ml"],
    "financial": ["valuation", "fundamental", "stock_lens"],
    "industry": ["industry_roll", "factor_rotation"],
    "quarterly": ["institutional"],
    "cninfo": ["announcement"],
    "events": ["event_pulse"],
    "macro": ["macro"],
    "hot_rank": ["social_sentiment"],
    "social": ["social_sentiment"],
    "lhb_hist": ["short_term_extra"],
    "zt_history": ["ladder"],
}

# 基座目录 → (路径, 最大陈旧天数, 依赖模块) —— 天数按数据节奏定：日频≤5、周频≤7、月频≤31、季频≤120
BASE_DIR_CHECKS: dict[str, tuple[Path, int]] = {
    "kline": (DW / "kline", 5),
    "valuation": (DW / "valuation", 5),
    "financial": (DW / "financial", 7),
    "industry": (DW / "industry", 5),
    "quarterly": (DW / "quarterly", 120),
    "cninfo": (DW / "cninfo", 7),
    "events": (DW / "events", 7),
    "market": (DW / "market", 4),
    "macro": (DW / "macro", 31),
    "hot_rank": (DW / "hot_rank", 3),
    "social": (DW / "social", 7),
    "lhb_hist": (DW / "lhb_hist", 7),
    "zt_history": (DW / "zt_history", 7),
}


def _latest_date(df: pd.DataFrame, col: str = "date") -> pd.Timestamp | None:
    if df is None or df.empty:
        return None
    if col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return s.max()


def _read(path: Path, cols: list[str] | None = None) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        return pd.read_parquet(path, columns=cols) if cols else pd.read_parquet(path)
    except Exception as e:
        print(f"  ⚠️ 读取失败 {path.name}: {str(e)[:80]}")
        return None


def _size_mutation(path: Path, name: str, old_state: dict) -> float | None:
    """返回文件大小相对昨日快照的变化率（无快照返回 None）。
    键统一为 f"size::{name}"（与快照写入一致，防静默丢失检测失效）。
    """
    if not path.exists():
        return None
    cur = path.stat().st_size
    key = f"size::{name}"
    prev = old_state.get(key)
    if prev is None or prev <= 0:
        return None
    return (cur - prev) / prev


def _date_from_filename(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", path.name)
    if not match:
        return None
    try:
        return pd.Timestamp("-".join(match.groups()))
    except ValueError:
        return None


def _latest_business_date(dir_path: Path) -> pd.Timestamp | None:
    """返回目录中的最新业务日期；绝不使用复制/恢复产生的 mtime。"""
    if not dir_path.is_dir():
        return None
    latest: pd.Timestamp | None = None
    files = sorted(
        (p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".json", ".csv"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:128]
    for path in files:
        candidate = _date_from_filename(path)
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(path).names
                column = next((c for c in (
                    "date", "trade_date", "日期", "交易日期", "上榜日", "计入日期",
                    "报告期", "report_date", "公告日期", "最新日期",
                ) if c in schema), None)
                if column:
                    frame = pd.read_parquet(path, columns=[column])
                    values = pd.to_datetime(frame[column], errors="coerce").dropna()
                    if not values.empty:
                        candidate = values.max()
            except Exception:
                pass
        if candidate is not None:
            candidate = pd.Timestamp(candidate).tz_localize(None) if pd.Timestamp(candidate).tzinfo else pd.Timestamp(candidate)
            latest = candidate if latest is None or candidate > latest else latest
    return latest


def run(skip_baostock: bool = True) -> dict:
    # The recovery environment can have a host clock one day ahead of the
    # business audit date.  Allow an explicit as-of date so reports never
    # claim a future date while preserving the normal clock-based default.
    as_of = os.environ.get("QUANT_HEALTH_AS_OF", "").strip()
    if as_of:
        try:
            as_of_date = datetime.fromisoformat(as_of).date()
        except ValueError as exc:
            raise ValueError(f"QUANT_HEALTH_AS_OF 无效: {as_of!r}") from exc
        now = datetime.combine(as_of_date, datetime.min.time(), tzinfo=CST).replace(hour=16)
    else:
        now = datetime.now(CST)
    today = now.date().isoformat()
    old_state: dict = {}
    if STATE_FILE.exists():
        try:
            old_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            old_state = {}

    checks: list[dict] = []
    degraded: list[str] = []
    trust: dict[str, float] = {m: 1.0 for mods in DATASET_MODULES.values() for m in mods}

    # 最近参照日: 审计 2026-08-16 改用真实交易日历（不再用工作日粗筛，
    # 长假不再把所有盘后产物误报 stale）
    recent_td = now - timedelta(days=4)
    closed_today = now.hour > 15 or (now.hour == 15 and now.minute >= 30)
    try:
        from quant_system.market_clock import latest_trading_day, is_trading_day
        base = latest_trading_day(now)
        # 若今天未收盘且今天本身是交易日，参照基准应取上一交易日
        if is_trading_day(now) and not closed_today:
            from quant_system.market_clock import prev_trading_day
            base = prev_trading_day(now) or base
        import pandas as _pd
        recent_td = _pd.Timestamp(base).to_pydatetime()
    except Exception:
        # 降级：工作日粗筛
        for back in range(0, 5):
            d = now - timedelta(days=back)
            if d.weekday() < 5:
                if back == 0 and not closed_today:
                    continue
                recent_td = d
                break

    def _check(name: str, path: Path, ok: bool, detail: str, weight_key: str | None = None):
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok and weight_key:
            degraded.append(weight_key)
            for m in DATASET_MODULES.get(weight_key, []):
                trust[m] = min(trust[m], 0.5)

    # 1. zt_daily_stats
    df = _read(ZT_DAILY_STATS, ["date"])
    ld = _latest_date(df)
    ok = ld is not None and ld.date().isoformat() >= recent_td.date().isoformat()
    _check("zt_daily_stats", ZT_DAILY_STATS, ok,
           f"最新 {ld.date() if ld is not None else '无'} (要求 ≥ {recent_td.date()})",
           "zt_daily_stats")
    # 1b. 独立新鲜度兜底（防止日历粗筛在长假期间全部误判）: 数据距今天数
    if ld is not None:
        staleness = (now.replace(tzinfo=None) - pd.Timestamp(ld)).days
        if staleness > 12:
            _check("zt_daily_stats_fresh", ZT_DAILY_STATS, False,
                   f"最新数据 {ld.date()} 距今 {staleness} 天 > 12 天 → 数据源可能停更",
                   "zt_daily_stats")

    # 2. zt_pool_history（重点: 是否仍是 2 千行的近期表）
    df = _read(ZT_HISTORY, ["date"])
    ld = _latest_date(df)
    rows = 0 if df is None else len(df)
    ok = ld is not None and ld.date().isoformat() >= recent_td.date().isoformat() and rows > 100000
    _check("zt_pool_history", ZT_HISTORY, ok,
           f"{rows} 行 | 最新 {ld.date() if ld is not None else '无'} | {'✅全量' if rows > 100000 else '⚠️近期表(需重建)'}",
           "zt_pool_history")

    # 3. zt_pool_em_daily
    df = _read(ZT_EM_DAILY, ["date"])
    ld = _latest_date(df)
    ok = ld is not None and ld.date().isoformat() >= recent_td.date().isoformat()
    _check("zt_pool_em_daily", ZT_EM_DAILY, ok,
           f"最新 {ld.date() if ld is not None else '无'}", "zt_pool_em_daily")

    # 4. fund_forces
    df = _read(MARKET_DIR / "fund_forces.parquet", ["date"])
    ld = _latest_date(df)
    ok = ld is not None and ld.date().isoformat() >= recent_td.date().isoformat()
    _check("fund_forces", MARKET_DIR / "fund_forces.parquet", ok,
           f"最新 {ld.date() if ld is not None else '无'}", "fund_forces")

    # 5. theme_cycle
    df = _read(MARKET_DIR / "theme_cycle.parquet", ["date"])
    ld = _latest_date(df)
    ok = ld is not None and ld.date().isoformat() >= recent_td.date().isoformat()
    _check("theme_cycle", MARKET_DIR / "theme_cycle.parquet", ok,
           f"最新 {ld.date() if ld is not None else '无'}", "theme_cycle")

    # 6. fusion
    df = _read(MARKET_DIR / "fusion.parquet", ["date"])
    ld = _latest_date(df)
    ok = ld is not None and ld.date().isoformat() >= recent_td.date().isoformat()
    _check("fusion", MARKET_DIR / "fusion.parquet", ok,
           f"最新 {ld.date() if ld is not None else '无'}", "fusion")

    # 7. social_sentiment
    df = _read(MARKET_DIR / "social_sentiment.parquet")
    ok = df is not None and len(df) > 0
    _check("social_sentiment", MARKET_DIR / "social_sentiment.parquet", ok,
           f"{0 if df is None else len(df)} 行数据", "social_sentiment")

    # 8. 文件大小突变检测
    for name, path in [("zt_daily_stats", ZT_DAILY_STATS),
                       ("zt_pool_history", ZT_HISTORY),
                       ("fund_forces", MARKET_DIR / "fund_forces.parquet"),
                       ("theme_cycle", MARKET_DIR / "theme_cycle.parquet")]:
        if not path.exists():
            continue
        chg = _size_mutation(path, name, old_state)
        if chg is not None and abs(chg) > 0.3:
            _check(f"size::{name}", path, False,
                   f"文件大小突变 {chg:+.0%} (疑似静默丢失/重复覆盖)", name)
        else:
            checks.append({"name": f"size::{name}", "ok": True,
                           "detail": f"{path.stat().st_size/1024:.0f}KB" + (f" (Δ{chg:+.0%})" if chg else "")})

    # 9. baostock 可用性（默认跳过，慢）
    if not skip_baostock:
        try:
            import baostock as bs
            lg = bs.login()
            ok = lg.error_code == "0"
            bs.logout()
            checks.append({"name": "baostock", "ok": ok, "detail": "登录成功" if ok else lg.error_msg})
        except Exception as e:
            checks.append({"name": "baostock", "ok": False, "detail": str(e)[:80]})
            for m in ["valuation"]:
                trust[m] = min(trust.get(m, 1.0), 0.5)
                degraded.append("baostock")

    # 10. stock_names
    df = _read(NAME_MAP)
    ok = df is not None and len(df) > 5000
    _check("stock_names", NAME_MAP, ok, f"{0 if df is None else len(df)} 只", "stock_names")

    # 11. 基础数据目录新鲜度（2026-08-23 W0.7 补盲）
    for bname, (bdir, max_days) in BASE_DIR_CHECKS.items():
        latest_date = _latest_business_date(bdir)
        if latest_date is None:
            _check(f"base::{bname}", bdir, False, "目录缺失或空", bname)
            continue
        stale_days = max(0.0, (now.replace(tzinfo=None) - latest_date.to_pydatetime()).total_seconds() / 86400.0)
        ok = stale_days <= max_days
        _check(f"base::{bname}", bdir, ok,
               f"最新业务日期 {latest_date.date()}，距今 {stale_days:.1f} 天 (阈值 {max_days} 天)", bname)

    # 保存状态快照（供明日突变检测）
    new_state = {k: v for k, v in old_state.items() if not k.startswith("size::")}
    for name, path in [("zt_daily_stats", ZT_DAILY_STATS),
                       ("zt_pool_history", ZT_HISTORY),
                       ("fund_forces", MARKET_DIR / "fund_forces.parquet"),
                       ("theme_cycle", MARKET_DIR / "theme_cycle.parquet")]:
        if path.exists():
            new_state[f"size::{name}"] = path.stat().st_size
    new_state["last_run"] = now.isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    n_ok = sum(1 for c in checks if c["ok"])
    overall_ok = n_ok == len(checks)
    res = {
        "date": today,
        "overall_ok": overall_ok,
        "ok_count": n_ok,
        "total_checks": len(checks),
        "checks": checks,
        "degraded_datasets": degraded,
        "trust_scores": {k: round(v, 2) for k, v in sorted(trust.items())},
    }
    out = ROOT / "generated" / f"data_health_{today}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def print_report(r: dict) -> str:
    lines = [
        f"🩺 数据健康检查 {r['date']} | {'✅ 全绿' if r['overall_ok'] else '⚠️ ' + str(len(r['degraded_datasets'])) + ' 项降级'}",
        "-" * 46,
    ]
    for c in r["checks"]:
        lines.append(f"  {'✅' if c['ok'] else '❌'} {c['name']}: {c['detail']}")
    if r["degraded_datasets"]:
        lines.append(f"\n  降级数据集: {', '.join(r['degraded_datasets'])}")
        lines.append(f"  受影响模块权重: { {k: v for k, v in r['trust_scores'].items() if v < 1.0} }")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V11 数据健康检查")
    ap.add_argument("--check-baostock", action="store_true", default=False,
                    help="额外执行 baostock 慢测（默认跳过）")
    args = ap.parse_args()
    r = run(skip_baostock=not args.check_baostock)
    print(print_report(r))
