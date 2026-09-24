#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pattern_gate — 规律显著性门禁（Bootstrap）+ 规律半衰期机制（V12.1 域P）

pattern_engine/macro_learner 发现的规律不能凭观察样本直接进实时权重：
  ① Bootstrap 验证器   : 对规律触发后的收益做 1000 次有放回重采样，按 H0 重中心化，
                          计算 P(null >= observed)（单侧）；p < 0.05 才允许写规律库。
  ② 半衰期机制        : 规律注册携带 birth_date；半衰期 20 个交易日（可配）。
                          超期后用最近 20 日窗口算胜率；胜率 >= 55% 续期 active，
                          否则 archived 并退出实时权重。
  ③ 集成点            : 提供 validate_pattern()/gate_patterns() 供 pattern_engine 调用；
                          不改 pattern_engine 主流程，只新增门禁函数。
  ④ 输出              : generated/pattern_gate/YYYYMMDD/pattern_gate_{date}.json|md。

统计口径（重要）:
  - bootstrap 分布必须在 H0 下重中心化，不能直接从原始收益重采样。
    均值检验：原始收益减样本均值后重采样；胜率检验：0/1 胜率序列中心化到 0.5 后重采样。
    否则 P(boot >= obs) 会恒在 0.5 附近，显著性门禁完全失效。
  - 有 scipy 时用 scipy.stats.percentileofscore 计算经验 p 值；无 scipy 回退 numpy 分位数法。

用法:
  python3 -m quant_system.analysis_core.pattern_gate --patterns x.json --history h.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as SCIPY_STATS
except Exception:  # pragma: no cover - 环境缺少 scipy 时回退 numpy
    SCIPY_STATS = None

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
DEFAULT_OUT_DIR = ROOT / "generated"
CST = timezone(timedelta(hours=8))

BOOTSTRAP_ITERATIONS = 1000
P_THRESHOLD = 0.05
HALF_LIFE_DAYS = 20
RECENT_WINDOW = 20
RENEW_WIN_RATE = 0.55

logger = logging.getLogger(__name__)


# ── 基础工具 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _date_str(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return _today()


def _r4(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None


def _as_finite(returns) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _is_short(direction) -> bool:
    return str(direction or "").lower() in ("down", "short", "sell", "空", "看空")


def _directional_returns(returns, direction) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    return -arr if _is_short(direction) else arr


def _bootstrap_p(boot_stats, observed) -> float | None:
    """H0 分布中 P(stat >= observed)，有 scipy 用 scipy，否则 numpy 分位数法。"""
    boot = np.asarray(boot_stats, dtype=float).reshape(-1)
    if boot.size == 0 or observed is None or not np.isfinite(observed):
        return None
    if SCIPY_STATS is not None:
        # percentileofscore(kind='strict') = P(x < observed)；
        # 1 - strict = P(x >= observed)
        pct = float(SCIPY_STATS.percentileofscore(boot, observed, kind="strict"))
        return float(1.0 - pct / 100.0)
    return float(np.mean(boot >= observed))


# ── Bootstrap 验证器 ──────────────────────────────────────────────────────
def bootstrap_test(returns, n_iterations: int = BOOTSTRAP_ITERATIONS,
                   seed: int | None = None, p_threshold: float = P_THRESHOLD,
                   direction: str = "long") -> dict:
    """对收益序列做 H0 中心化 bootstrap，返回胜率/平均收益的 p 值与分布。

    returns 若为空/全非有限 → degraded=True，不崩。
    通过条件：胜率单侧 bootstrap p < p_threshold（默认 0.05）。
    """
    arr = _as_finite(_directional_returns(returns, direction))
    n = int(arr.size)
    if n == 0:
        return {
            "n_samples": 0,
            "n_iterations": int(n_iterations),
            "direction": direction,
            "observed_win_rate": None,
            "observed_avg_ret": None,
            "bootstrap_win_rates": [],
            "bootstrap_avg_rets": [],
            "p_value": None,
            "avg_ret_p_value": None,
            "p_threshold": float(p_threshold),
            "passed": False,
            "degraded": True,
        }
    if int(n_iterations) < 1:
        raise ValueError("n_iterations 必须 >= 1")

    rng = np.random.default_rng(seed)
    observed_win = float((arr > 0).mean())
    observed_avg = float(arr.mean())

    # H0 中心化：均值检验中心化到 0，胜率检验中心化到 0.5。
    avg_null_data = arr - observed_avg
    win_null_data = (arr > 0).astype(float) - observed_win + 0.5

    boot_avg = np.empty(int(n_iterations), dtype=float)
    boot_win = np.empty(int(n_iterations), dtype=float)
    for i in range(int(n_iterations)):
        idx = rng.integers(0, n, size=n)
        boot_avg[i] = float(avg_null_data[idx].mean())
        boot_win[i] = float(win_null_data[idx].mean())

    p_win = _bootstrap_p(boot_win, observed_win)
    p_avg = _bootstrap_p(boot_avg, observed_avg)
    return {
        "n_samples": n,
        "n_iterations": int(n_iterations),
        "direction": direction,
        "observed_win_rate": _r4(observed_win),
        "observed_avg_ret": _r4(observed_avg),
        "bootstrap_win_rates": boot_win.tolist(),
        "bootstrap_avg_rets": boot_avg.tolist(),
        "p_value": _r4(p_win),
        "avg_ret_p_value": _r4(p_avg),
        "p_threshold": float(p_threshold),
        "passed": bool(p_win is not None and p_win < float(p_threshold)),
        "degraded": False,
    }


# ── 历史数据 → 规律触发收益 ────────────────────────────────────────────────
def extract_returns(history, signal_col: str = "signal", return_col: str = "return",
                    direction: str = "long", pattern_id: str | None = None) -> np.ndarray:
    """从 DataFrame/list[dict]/dict mapping/数组 中提取规律触发后的收益序列。

    DataFrame 可选含 pattern_id/signal/return；signal 为 0/假值时剔除。
    direction=short/down 时收益取反，使正收益恒代表方向正确。
    """
    if history is None:
        return np.asarray([], dtype=float)

    if isinstance(history, dict):
        key = str(pattern_id) if pattern_id is not None else None
        val = history.get(key) if key is not None else None
        if val is None and "*" in history:
            val = history["*"]
        if val is None and "returns" in history:
            val = history["returns"]
        if val is None:
            return np.asarray([], dtype=float)
        return extract_returns(val, signal_col=signal_col, return_col=return_col,
                               direction=direction, pattern_id=pattern_id)

    if isinstance(history, pd.DataFrame):
        df = history
        if pattern_id is not None and "pattern_id" in df.columns:
            df = df[df["pattern_id"].astype(str) == str(pattern_id)]
        if signal_col in df.columns:
            signal = pd.to_numeric(df[signal_col], errors="coerce").fillna(0)
            df = df[signal > 0]
        cols = [return_col, "ret", "forward_return", "return_1d"]
        col = next((c for c in cols if c in df.columns), None)
        if col is None:
            return np.asarray([], dtype=float)
        return _directional_returns(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float),
                                    direction)

    if isinstance(history, (list, tuple, np.ndarray, pd.Series)):
        items = list(history)
        if items and isinstance(items[0], dict):
            out = []
            for row in items:
                if pattern_id is not None and "pattern_id" in row and str(row.get("pattern_id")) != str(pattern_id):
                    continue
                sig = row.get(signal_col)
                if sig is not None:
                    try:
                        if float(sig) <= 0:
                            continue
                    except (TypeError, ValueError):
                        if not bool(sig):
                            continue
                val = row.get(return_col)
                if val is None:
                    val = row.get("ret", row.get("forward_return", row.get("return_1d")))
                if val is not None:
                    out.append(float(val))
            return _directional_returns(np.asarray(out, dtype=float), direction)
        return _directional_returns(pd.to_numeric(pd.Series(items), errors="coerce").to_numpy(dtype=float),
                                    direction)

    return np.asarray([], dtype=float)


# ── 规律注册与半衰期机制 ──────────────────────────────────────────────────
def register_pattern(pattern: dict, birth_date=None, as_of_date=None) -> dict:
    """返回带 birth_date 的规律副本；缺省依次取 first_seen/signal_date/当前日。"""
    out = dict(pattern)
    if not out.get("birth_date"):
        out["birth_date"] = (out.get("first_seen") or birth_date or out.get("signal_date")
                             or as_of_date or _today())
    out.setdefault("status", "active")
    return out


def trading_day_age(birth_date, as_of_date=None, trading_dates=None) -> int | None:
    """计算 birth_date → as_of_date 的交易日龄。

    trading_dates 提供时只数 (birth, as_of] 内的交易日；否则用 numpy 工作日历近似。
    """
    if birth_date is None:
        return None
    try:
        birth = pd.Timestamp(birth_date).normalize()
    except Exception:
        return None
    end = pd.Timestamp(as_of_date or _today()).normalize()

    if trading_dates is not None:
        try:
            days = pd.to_datetime(list(trading_dates), errors="coerce").dropna().normalize()
            if len(days):
                return int(((days > birth) & (days <= end)).sum())
        except Exception:
            pass
    return int(np.busday_count(birth.date(), end.date()))


def recent_win_rate(returns, window: int = RECENT_WINDOW, direction: str = "long") -> float | None:
    arr = _as_finite(_directional_returns(returns, direction))[-int(window):]
    if arr.size == 0:
        return None
    return float((arr > 0).mean())


def half_life_status(pattern: dict, as_of_date=None, trading_dates=None,
                     half_life_days: int = HALF_LIFE_DAYS,
                     recent_window: int = RECENT_WINDOW,
                     renew_win_rate: float = RENEW_WIN_RATE) -> dict:
    """规律半衰期状态：未到期 active；到期看最近窗口胜率续期/归档。"""
    p = dict(pattern)
    birth = p.get("birth_date") or p.get("first_seen") or p.get("signal_date")
    direction = p.get("direction", "long")
    age = trading_day_age(birth, as_of_date=as_of_date, trading_dates=trading_dates)
    returns = p.get("recent_returns") if p.get("recent_returns") is not None else p.get("returns")
    wr = recent_win_rate(returns, window=int(recent_window), direction=direction) if returns is not None else None

    base = {
        "pattern_id": p.get("pattern_id"),
        "pattern_type": p.get("pattern_type"),
        "birth_date": birth,
        "as_of_date": _date_str(as_of_date or _today()),
        "age_days": age,
        "half_life_days": int(half_life_days),
        "recent_window": int(recent_window),
        "renew_win_rate": float(renew_win_rate),
        "win_rate": _r4(wr),
        "renewed": False,
        "status": "active",
        "reason": "not_expired",
    }

    if age is None or age < int(half_life_days):
        base["reason"] = "not_expired"
        return base
    if wr is None:
        base["status"] = "archived"
        base["reason"] = "no_recent_returns"
        return base
    if wr >= float(renew_win_rate):
        base["status"] = "active"
        base["reason"] = "renewed"
        base["renewed"] = True
    else:
        base["status"] = "archived"
        base["reason"] = "low_win_rate"
    return base


def real_time_weights(patterns, as_of_date=None, trading_dates=None,
                      half_life_days: int = HALF_LIFE_DAYS,
                      recent_window: int = RECENT_WINDOW,
                      renew_win_rate: float = RENEW_WIN_RATE) -> dict[str, float]:
    """只给 active 规律实时权重；archived 自动退出权重并归一化。"""
    active: list[tuple[str, float]] = []
    for p in patterns:
        st = half_life_status(p, as_of_date=as_of_date, trading_dates=trading_dates,
                              half_life_days=half_life_days, recent_window=recent_window,
                              renew_win_rate=renew_win_rate)
        if st["status"] != "active":
            continue
        pid = str(st.get("pattern_id") or "")
        w = float(p.get("weight", 1.0) or 1.0)
        if w > 0:
            active.append((pid, w))
    total = sum(w for _, w in active)
    if total <= 0:
        return {}
    return {pid: w / total for pid, w in active}


# ── 门禁集成与输出 ────────────────────────────────────────────────────────
def validate_pattern(pattern: dict, history=None, returns=None, as_of_date=None,
                     trading_dates=None, birth_date=None,
                     n_iterations: int = BOOTSTRAP_ITERATIONS,
                     p_threshold: float = P_THRESHOLD,
                     half_life_days: int = HALF_LIFE_DAYS,
                     recent_window: int = RECENT_WINDOW,
                     renew_win_rate: float = RENEW_WIN_RATE,
                     seed: int | None = None) -> dict:
    """门禁包装：bootstrap 显著性 + 半衰期状态 → active|archived。

    返回 dict 同时保留 bootstrap 明细与 half_life 明细，供 pattern_engine 写库前调用。
    """
    p = register_pattern(pattern, birth_date=birth_date, as_of_date=as_of_date)
    direction = p.get("direction", "long")
    raw_returns = returns
    if raw_returns is None:
        raw_returns = p.get("recent_returns") if p.get("recent_returns") is not None else p.get("returns")
    if raw_returns is None:
        raw_returns = extract_returns(history, direction="long", pattern_id=p.get("pattern_id"))
    gate = bootstrap_test(raw_returns, n_iterations=n_iterations, seed=seed,
                          p_threshold=p_threshold, direction=direction)
    arr = _as_finite(_directional_returns(raw_returns, direction))
    p["recent_returns"] = arr
    hs = half_life_status(p, as_of_date=as_of_date, trading_dates=trading_dates,
                          half_life_days=half_life_days, recent_window=recent_window,
                          renew_win_rate=renew_win_rate)
    passed = bool(gate.get("passed"))
    status = "active" if passed and hs["status"] == "active" else "archived"
    return {
        "pattern_id": p.get("pattern_id"),
        "pattern_type": p.get("pattern_type"),
        "name": p.get("name"),
        "direction": direction,
        "bootstrap": gate,
        "win_rate": hs.get("win_rate"),
        "status": status,
        "half_life": hs,
    }


def _degraded_result(date: str, reason: str) -> dict:
    return {
        "schema": "pattern_gate/v1",
        "date": date,
        "generated_at": _now(),
        "degraded": True,
        "reason": reason,
        "config": {
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "p_threshold": P_THRESHOLD,
            "half_life_days": HALF_LIFE_DAYS,
            "recent_window": RECENT_WINDOW,
            "renew_win_rate": RENEW_WIN_RATE,
        },
        "n_patterns": 0,
        "n_active": 0,
        "n_archived": 0,
        "patterns": [],
    }


def gate_patterns(patterns, history=None, *, as_of_date=None, trading_dates=None,
                  n_iterations: int = BOOTSTRAP_ITERATIONS,
                  p_threshold: float = P_THRESHOLD,
                  half_life_days: int = HALF_LIFE_DAYS,
                  recent_window: int = RECENT_WINDOW,
                  renew_win_rate: float = RENEW_WIN_RATE,
                  seed: int | None = None,
                  out_dir=None) -> dict:
    """批量门禁：对每条规律 validate_pattern，并可选落盘 json/md。"""
    date = _date_str(as_of_date or _today())
    if not patterns:
        result = _degraded_result(date, "no_patterns")
        if out_dir is not None:
            write_outputs(result, Path(out_dir))
        return result

    evaluated = []
    for p in patterns:
        try:
            ev = validate_pattern(
                p, history=history, as_of_date=as_of_date, trading_dates=trading_dates,
                n_iterations=n_iterations, p_threshold=p_threshold,
                half_life_days=half_life_days, recent_window=recent_window,
                renew_win_rate=renew_win_rate, seed=seed)
        except Exception as exc:  # noqa: BLE001 - 单条规律失败不阻断批量
            logger.warning("pattern_gate 校验失败 %s: %s", p.get("pattern_id"), exc)
            ev = {
                "pattern_id": p.get("pattern_id"),
                "pattern_type": p.get("pattern_type"),
                "name": p.get("name"),
                "direction": p.get("direction", "long"),
                "bootstrap": {
                    "p_value": None, "avg_ret_p_value": None, "passed": False,
                    "degraded": True, "observed_win_rate": None, "observed_avg_ret": None,
                },
                "win_rate": None,
                "status": "archived",
                "half_life": {"status": "archived", "reason": "gate_error",
                              "as_of_date": date, "age_days": None},
            }
        evaluated.append(ev)

    pattern_rows = []
    for ev in evaluated:
        bs = ev.get("bootstrap") or {}
        hs = ev.get("half_life") or {}
        pattern_rows.append({
            "pattern_id": ev.get("pattern_id"),
            "pattern_type": ev.get("pattern_type"),
            "name": ev.get("name"),
            "direction": ev.get("direction"),
            "bootstrap_p_value": bs.get("p_value"),
            "bootstrap_passed": bool(bs.get("passed")),
            "avg_ret_p_value": bs.get("avg_ret_p_value"),
            "observed_win_rate": bs.get("observed_win_rate"),
            "observed_avg_ret": bs.get("observed_avg_ret"),
            "win_rate": ev.get("win_rate"),
            "status": ev.get("status"),
            "half_life": hs,
        })

    result = {
        "schema": "pattern_gate/v1",
        "date": date,
        "generated_at": _now(),
        "degraded": False,
        "config": {
            "bootstrap_iterations": int(n_iterations),
            "p_threshold": float(p_threshold),
            "half_life_days": int(half_life_days),
            "recent_window": int(recent_window),
            "renew_win_rate": float(renew_win_rate),
        },
        "n_patterns": len(evaluated),
        "n_active": sum(1 for ev in evaluated if ev.get("status") == "active"),
        "n_archived": sum(1 for ev in evaluated if ev.get("status") == "archived"),
        "patterns": pattern_rows,
    }
    if out_dir is not None:
        write_outputs(result, Path(out_dir))
    return result


def write_outputs(result: dict, out_dir) -> tuple[Path, Path]:
    """写 generated/pattern_gate/YYYYMMDD/pattern_gate_{date}.json|md。"""
    out = Path(out_dir)
    day = result["date"].replace("-", "") if result.get("date") else "unknown"
    sub = out / "pattern_gate" / day
    sub.mkdir(parents=True, exist_ok=True)
    json_path = sub / f"pattern_gate_{result['date']}.json"
    md_path = sub / f"pattern_gate_{result['date']}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    md_path.write_text(_to_md(result), encoding="utf-8")
    return json_path, md_path


def _to_md(result: dict) -> str:
    lines = [f"# 规律显著性门禁 {result.get('date', '')}（V12.1 域P）", ""]
    if result.get("degraded"):
        lines.append(f"> 降级：{result.get('reason', '')}，无规律可写，结果为空。")
        return "\n".join(lines)
    cfg = result.get("config", {})
    lines += [
        f"- Bootstrap {cfg.get('bootstrap_iterations')} 次 / p<{cfg.get('p_threshold')} "
        f"/ 半衰期 {cfg.get('half_life_days')} 交易日 / 续期胜率 {cfg.get('renew_win_rate')}",
        f"- 规律数 {result.get('n_patterns')} | active {result.get('n_active')} | "
        f"archived {result.get('n_archived')}",
        "",
        "| pattern_id | 类型 | 方向 | bootstrap p | 通过 | 胜率 | 状态 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for p in result.get("patterns", []):
        pv = "—" if p.get("bootstrap_p_value") is None else f"{p['bootstrap_p_value']:.4f}"
        wr = "—" if p.get("win_rate") is None else f"{p['win_rate']:.1%}"
        lines.append(
            f"| `{p.get('pattern_id')}` | {p.get('pattern_type') or '—'} | "
            f"{p.get('direction') or '—'} | {pv} | {'yes' if p.get('bootstrap_passed') else 'no'} | "
            f"{wr} | {p.get('status')} |")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="规律显著性门禁 + 半衰期机制（V12.1 域P）")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，缺省今天")
    ap.add_argument("--patterns", default=None, help="规律 JSON 文件（list[dict]）")
    ap.add_argument("--history", default=None, help="历史数据 JSON 文件（list[dict] 或 DataFrame 风格 dict）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出根目录（默认 generated）")
    args = ap.parse_args(argv)

    patterns = []
    if args.patterns:
        patterns = json.loads(Path(args.patterns).read_text(encoding="utf-8"))
    history = None
    if args.history:
        try:
            history = json.loads(Path(args.history).read_text(encoding="utf-8"))
        except Exception:
            history = pd.read_json(args.history)

    try:
        result = gate_patterns(patterns, history=history, as_of_date=args.date,
                               out_dir=args.out_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pattern_gate 执行失败: %s", exc)
        return 1

    if result.get("degraded"):
        print(f"[pattern_gate] {result.get('date')} 降级（{result.get('reason')}）：空结果")
        return 0
    print(f"[pattern_gate] {result['date']} 规律数={result['n_patterns']} "
          f"active={result['n_active']} archived={result['n_archived']}")
    for p in result["patterns"]:
        print(f"  {p['pattern_id']} p={p['bootstrap_p_value']} "
              f"pass={p['bootstrap_passed']} status={p['status']}")
    out = Path(args.out_dir) / "pattern_gate" / result["date"].replace("-", "")
    print(f"[pattern_gate] 输出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
