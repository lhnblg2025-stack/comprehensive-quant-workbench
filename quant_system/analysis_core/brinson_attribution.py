#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""brinson_attribution — Brinson 业绩归因四维拆解（V12.1 域Q）

把组合超额收益拆成四个可执行维度：
  - 择时贡献    : (组合总仓位 - 1) * 基准收益，衡量仓位择时
  - 行业配置贡献 : 全仓口径下，行业权重相对基准行业权重的贡献
  - 选股贡献    : 全仓口径下，行业内股票收益相对基准行业收益的 alpha
  - 交易摩擦    : 实际收益 - 理论收益，覆盖滑点/费用/成交偏差

口径说明（保证四维相加等于实际收益）：
  theoretical_return = benchmark_return + timing + allocation + selection
  friction = actual_return - theoretical_return
  其中 portfolio 行业权重按组合内部仓位归一（sum=1），再乘总仓位 W。

数据均为 mock 可控，默认无真实组合/行业基准文件时输出降级结果。

用法:
  python3 -m quant_system.analysis_core.brinson_attribution --date 2026-08-11
  python3 -m quant_system.analysis_core.brinson_attribution \
    --portfolio /tmp/portfolio.json --benchmark /tmp/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
DEFAULT_OUT_DIR = ROOT / "generated"
DEFAULT_PORTFOLIO_PATH = ROOT / "data_warehouse" / "attribution" / "portfolio_daily.json"
DEFAULT_BENCHMARK_PATH = ROOT / "data_warehouse" / "attribution" / "benchmark_industries.json"
DEFAULT_INDUSTRY_MAP_PATH = ROOT / "data_warehouse" / "market" / "sw_industry_map.parquet"

CST = timezone(timedelta(hours=8))
LOCK_WINDOW = 3


# ── 基础工具 ──────────────────────────────────────────────────────────────
def _today() -> str:
    return datetime.now(CST).date().isoformat()


def _now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _norm_date(value: str | None) -> str:
    value = (value or _today()).strip().replace("/", "-")
    if len(value) == 8 and value.isdigit():
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _f(value, nd: int = 8):
    """转有限 float 并四舍五入，保证 JSON 可读；None/非数值原样返回。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def _as_records(data):
    """统一把 DataFrame/list[dict]/dict 转成 list[dict]。"""
    if data is None:
        return []
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _read_table(path: Path | str | None):
    """读取 json/parquet/csv；缺失返回 None。"""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return None


def _load_industry_map(industry_map):
    """code -> industry；支持 dict / DataFrame / json / parquet 路径。"""
    if industry_map is None:
        return {}
    if isinstance(industry_map, dict):
        return {str(k).zfill(6): str(v) for k, v in industry_map.items()}
    if isinstance(industry_map, pd.DataFrame):
        industry_map = industry_map.to_dict("records")
    if isinstance(industry_map, (str, Path)):
        industry_map = _read_table(Path(industry_map))
    if not isinstance(industry_map, list):
        return {}
    out = {}
    for r in industry_map:
        if not isinstance(r, dict):
            continue
        code = r.get("code")
        industry = r.get("industry") or r.get("industry_name")
        if code is not None and industry is not None:
            out[str(code).zfill(6)] = str(industry)
    return out


# ── 数据归一化 ────────────────────────────────────────────────────────────
def _normalize_portfolio(portfolio):
    """转成 [{date, actual_return, positions:[{code,industry,weight,return}]}]。"""
    rows = _as_records(portfolio)
    out: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    actual_by_date: dict[str, float | None] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        date = _norm_date(row.get("date"))
        positions = row.get("positions")
        if positions is None and {"weight", "return"}.issubset(row.keys()):
            positions = [row]
        if not isinstance(positions, list):
            continue
        grouped.setdefault(date, [])
        for p in positions:
            if not isinstance(p, dict):
                continue
            grouped[date].append(p)
        if "actual_return" in row:
            actual_by_date[date] = _f(row.get("actual_return"))

    for date in sorted(grouped):
        positions = []
        for p in grouped[date]:
            weight = _f(p.get("weight"))
            ret = _f(p.get("return"))
            if weight is None or ret is None:
                continue
            positions.append({
                "code": str(p.get("code") or "").zfill(6) or None,
                "industry": p.get("industry") or p.get("industry_name"),
                "weight": weight,
                "return": ret,
            })
        if positions:
            out.append({"date": date, "actual_return": actual_by_date.get(date), "positions": positions})
    return out


def _normalize_benchmark(benchmark):
    """转成 [{date, overall_return, industries:[{industry,weight,return}]}]。"""
    rows = _as_records(benchmark)
    out: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    overall_by_date: dict[str, float | None] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        date = _norm_date(row.get("date"))
        industries = row.get("industries") or row.get("industry_rows")
        if industries is None and {"industry", "weight", "return"}.issubset(row.keys()):
            industries = [row]
        if isinstance(industries, list):
            grouped.setdefault(date, [])
            for it in industries:
                if isinstance(it, dict):
                    grouped[date].append(it)
        if "return" in row and "industries" not in row and "industry" not in row:
            overall_by_date[date] = _f(row.get("return"))
        elif "benchmark_return" in row:
            overall_by_date[date] = _f(row.get("benchmark_return"))

    for date in sorted(grouped):
        industries = []
        for it in grouped[date]:
            industry = it.get("industry") or it.get("industry_name")
            weight = _f(it.get("weight"))
            ret = _f(it.get("return"))
            if industry is None or weight is None or ret is None:
                continue
            industries.append({"industry": str(industry), "weight": weight, "return": ret})
        if industries:
            out.append({"date": date, "overall_return": overall_by_date.get(date), "industries": industries})
    return out


# ── Brinson 核心 ──────────────────────────────────────────────────────────
def _degraded_result(date: str, reason: str) -> dict:
    return {
        "schema": "brinson_attribution/v1",
        "date": date,
        "generated_at": _now(),
        "degraded": True,
        "reason": reason,
        "config": {"lock_window": LOCK_WINDOW, "benchmark": "沪深300"},
        "daily": [],
        "cumulative": {"timing": 0.0, "allocation": 0.0, "selection": 0.0, "friction": None},
        "lock_allocator": False,
        "lock_allocator_advice": "",
        "degraded_reasons": [reason],
    }


def compute_attribution(portfolio, benchmark, industry_map=None, date=None,
                        lock_window: int = LOCK_WINDOW) -> dict:
    """计算四维 Brinson 归因。

    portfolio: list[dict] / DataFrame；每条含 date、positions 或长表列。
    benchmark: list[dict] / DataFrame；每条含 date、industries，可含总 return。
    industry_map: dict[code, industry]，portfolio 缺 industry 时补全。
    """
    date = _norm_date(date)
    portfolio_rows = _normalize_portfolio(portfolio)
    benchmark_rows = _normalize_benchmark(benchmark)
    if not portfolio_rows:
        return _degraded_result(date, "no_portfolio_data")
    if not benchmark_rows:
        return _degraded_result(date, "no_benchmark_data")

    industry_lookup = _load_industry_map(industry_map)
    bench_map = {b["date"]: b for b in benchmark_rows}
    daily: list[dict] = []
    cum = {"timing": 0.0, "allocation": 0.0, "selection": 0.0, "friction": 0.0}
    friction_seen = False
    timing_streak = 0
    max_streak = 0

    for port in portfolio_rows:
        day = port["date"]
        bench = bench_map.get(day)
        if bench is None:
            continue

        # 组合行业聚合：用原始仓位聚合，行业内收益率按仓位加权。
        p_raw: dict[str, dict] = {}
        total_weight = 0.0
        for pos in port.get("positions", []):
            industry = pos.get("industry")
            if not industry and pos.get("code"):
                industry = industry_lookup.get(pos.get("code"), "未知")
            if not industry:
                industry = "未知"
            w = _f(pos.get("weight")) or 0.0
            r = _f(pos.get("return")) or 0.0
            bucket = p_raw.setdefault(industry, {"weight": 0.0, "weighted_return": 0.0})
            bucket["weight"] += w
            bucket["weighted_return"] += w * r
            total_weight += w
        if total_weight <= 0:
            continue

        p_industries = {}
        for industry, bucket in p_raw.items():
            w_i = bucket["weight"]
            p_industries[industry] = {
                "weight": w_i / total_weight,
                "return": bucket["weighted_return"] / w_i if w_i else 0.0,
            }

        # 基准行业权重归一，避免 mock 数据权重和不为 1 时口径漂移。
        raw_bench = [(it.get("industry"), _f(it.get("weight")) or 0.0,
                      _f(it.get("return")) or 0.0)
                     for it in bench.get("industries", [])]
        raw_bench = [(i, w, r) for i, w, r in raw_bench if i and w >= 0]
        b_total = sum(w for _, w, _ in raw_bench)
        if b_total <= 0:
            continue
        b_industries = {i: {"weight": w / b_total, "return": r} for i, w, r in raw_bench}

        benchmark_return = _f(bench.get("overall_return"))
        if benchmark_return is None:
            benchmark_return = sum(b["weight"] * b["return"] for b in b_industries.values())

        universe = sorted(set(p_industries) | set(b_industries))
        allocation = 0.0
        selection = 0.0
        for industry in universe:
            w_p = p_industries.get(industry, {}).get("weight", 0.0)
            r_p = p_industries.get(industry, {}).get("return", 0.0)
            w_b = b_industries.get(industry, {}).get("weight", 0.0)
            r_b = b_industries.get(industry, {}).get("return", 0.0)
            allocation += (w_p - w_b) * r_b
            selection += w_p * (r_p - r_b)

        allocation = total_weight * allocation
        selection = total_weight * selection
        timing = (total_weight - 1.0) * benchmark_return
        theoretical = benchmark_return + timing + allocation + selection
        actual = port.get("actual_return")
        friction = None
        if actual is not None:
            friction = _f(actual) - _f(theoretical)
            friction_seen = True

        if timing < 0:
            timing_streak += 1
        else:
            timing_streak = 0
        max_streak = max(max_streak, timing_streak)

        daily.append({
            "date": day,
            "portfolio_weight": _f(total_weight),
            "benchmark_return": _f(benchmark_return),
            "actual_return": _f(actual),
            "theoretical_return": _f(theoretical),
            "contributions": {
                "timing": _f(timing),
                "allocation": _f(allocation),
                "selection": _f(selection),
                "friction": _f(friction),
            },
            "timing_negative_streak": timing_streak,
        })
        cum["timing"] += timing
        cum["allocation"] += allocation
        cum["selection"] += selection
        if friction is not None:
            cum["friction"] += friction

    if not daily:
        return _degraded_result(date, "no_matched_attribution_dates")

    lock = max_streak >= int(lock_window)
    return {
        "schema": "brinson_attribution/v1",
        "date": date,
        "generated_at": _now(),
        "degraded": False,
        "reason": None,
        "config": {"lock_window": int(lock_window), "benchmark": "沪深300"},
        "daily": daily,
        "cumulative": {
            "timing": _f(cum["timing"]),
            "allocation": _f(cum["allocation"]),
            "selection": _f(cum["selection"]),
            "friction": _f(cum["friction"]) if friction_seen else None,
        },
        "lock_allocator": lock,
        "lock_allocator_advice": ("择时贡献连续 3 日及以上为负，建议锁定 allocator 的仓位调整"
                                  if lock else ""),
        "degraded_reasons": [],
    }


# ── 输出 ──────────────────────────────────────────────────────────────────
def write_outputs(result: dict, out_dir) -> tuple[Path, Path]:
    out = Path(out_dir)
    day = result["date"].replace("-", "") if result.get("date") else "unknown"
    sub = out / "attribution" / day
    sub.mkdir(parents=True, exist_ok=True)
    json_path = sub / f"attribution_{result['date']}.json"
    md_path = sub / f"attribution_{result['date']}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    md_path.write_text(_to_md(result), encoding="utf-8")
    return json_path, md_path


def _to_md(result: dict) -> str:
    lines = [f"# Brinson 业绩归因 {result.get('date', '')}（V12.1 域Q）", ""]
    if result.get("degraded"):
        lines.append(f"> 降级：{result.get('reason', '')}，无四维归因可输出。")
        return "\n".join(lines)
    lines += [
        "| 日期 | 总仓位 | 基准收益 | 择时 | 行业配置 | 选股 | 交易摩擦 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in result.get("daily", []):
        c = d.get("contributions", {})
        def _s(v):
            return "—" if v is None else f"{float(v):.6f}"
        lines.append(
            f"| {d.get('date')} | {d.get('portfolio_weight'):.4f} | "
            f"{d.get('benchmark_return'):.4f} | {_s(c.get('timing'))} | "
            f"{_s(c.get('allocation'))} | {_s(c.get('selection'))} | {_s(c.get('friction'))} |")
    if result.get("lock_allocator"):
        lines.append("")
        lines.append(f"> ⚠️ {result.get('lock_allocator_advice')}")
    return "\n".join(lines)


def run_attribution(portfolio, benchmark, industry_map=None, date=None, out_dir=None,
                    lock_window: int = LOCK_WINDOW) -> dict:
    """compute_attribution + 可选写 generated/attribution/YYYYMMDD/*。"""
    result = compute_attribution(portfolio, benchmark, industry_map=industry_map,
                                 date=date, lock_window=lock_window)
    if out_dir is not None:
        write_outputs(result, out_dir)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Brinson 业绩归因四维拆解（V12.1 域Q）")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，缺省今天")
    ap.add_argument("--portfolio", default=None, help="组合持仓 JSON/parquet/csv")
    ap.add_argument("--benchmark", default=None, help="行业基准 JSON/parquet/csv")
    ap.add_argument("--industry-map", default=None, help="code->industry JSON/parquet/csv")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出根目录（默认 generated）")
    ap.add_argument("--lock-window", type=int, default=LOCK_WINDOW)
    args = ap.parse_args(argv)

    portfolio = _read_table(args.portfolio or DEFAULT_PORTFOLIO_PATH)
    benchmark = _read_table(args.benchmark or DEFAULT_BENCHMARK_PATH)
    industry_map = _read_table(args.industry_map or DEFAULT_INDUSTRY_MAP_PATH)
    if portfolio is None:
        portfolio = []
    if benchmark is None:
        benchmark = []
    if industry_map is None:
        industry_map = {}

    result = run_attribution(portfolio, benchmark, industry_map=industry_map,
                             date=args.date, out_dir=args.out_dir,
                             lock_window=args.lock_window)
    if result.get("degraded"):
        print(f"[brinson] {result['date']} 降级（{result.get('reason')}）：无四维归因")
    else:
        n = len(result["daily"])
        print(f"[brinson] {result['date']} 日序列={n} | 择时累计={result['cumulative']['timing']} "
              f"| lock_allocator={result['lock_allocator']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
