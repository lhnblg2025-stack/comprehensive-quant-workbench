#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""error_book — 强看多暴跌错题本 + 失效环境特征反向扫描（V12.1 域Q）

把 battle_map 的“看多但失败”案例沉淀为可执行规则：
  ① 入册 : 方向强看多（置信度 >= 0.7）且次日实际收益 <= -3%
  ② 反向扫描: 对错题案例，统计当日环境特征在错题集中的占比；
    占比超过阈值的单特征/两两特征组合生成规则：
    “当 X 且 Y 时，动量方向失效概率 Z%”。
  ③ 输出 : generated/error_book/YYYYMMDD/error_book_{date}.json|md

环境特征支持中文键，内部统一为英文 canonical 键：
  涨跌家数比 -> breadth_ratio
  炸板率 -> blast_rate
  社融发布 -> social_financing_release
  北向方向 -> northbound_direction
  风格象限 -> style_quadrant

用法:
  python3 -m quant_system.analysis_core.error_book --date 2026-08-11
  python3 -m quant_system.analysis_core.error_book \
    --history /tmp/history.json --returns /tmp/returns.json --environment /tmp/env.json
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
DEFAULT_INDEX_PATH = ROOT / "data_warehouse" / "market" / "index_daily_沪深300.parquet"

CST = timezone(timedelta(hours=8))
CONFIDENCE_THRESHOLD = 0.7
RETURN_THRESHOLD = -0.03
RULE_THRESHOLD = 0.6

FEATURE_ALIASES = {
    "涨跌家数比": "breadth_ratio",
    "涨跌比": "breadth_ratio",
    "炸板率": "blast_rate",
    "社融发布": "social_financing_release",
    "社融": "social_financing_release",
    "北向方向": "northbound_direction",
    "北向": "northbound_direction",
    "风格象限": "style_quadrant",
    "风格": "style_quadrant",
}
FEATURE_LABELS = {
    "breadth_ratio": "涨跌家数比",
    "blast_rate": "炸板率",
    "social_financing_release": "社融发布",
    "northbound_direction": "北向方向",
    "style_quadrant": "风格象限",
}


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


def _canonical_feature(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    key = name.strip()
    if key in FEATURE_LABELS:
        return key
    return FEATURE_ALIASES.get(key)


def _feature_label(key: str) -> str:
    return FEATURE_LABELS.get(key, key)


def _is_bullish_direction(direction) -> bool:
    text = str(direction or "").strip().lower()
    return text in {"多", "看多", "进攻", "增持", "long", "bull", "bullish", "buy", "买入"}


def _normalize_history(history):
    """转成 [{date, direction, confidence, features?}]，过滤缺字段行。"""
    rows = _as_records(history)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        date = r.get("date") or r.get("signal_date") or r.get("recommend_date")
        if not date:
            continue
        confidence = _f(r.get("confidence"))
        if confidence is None:
            confidence = _f(r.get("conf"))
        if confidence is None:
            confidence = 0.0
        out.append({
            "date": _norm_date(date),
            "direction": str(r.get("direction") or r.get("view") or r.get("recommended") or ""),
            "confidence": confidence,
            "features": r.get("features") if isinstance(r.get("features"), dict) else None,
            "source": r.get("source"),
        })
    return out


def _normalize_returns(returns):
    """转成 {date: return}，按日期排序后的 dict。"""
    if returns is None:
        return {}
    if isinstance(returns, pd.DataFrame):
        returns = returns.to_dict("records")
    out: dict[str, float] = {}
    if isinstance(returns, dict):
        for key, value in returns.items():
            if isinstance(value, dict):
                date = value.get("date") or key
                ret = value.get("return", value.get("actual_return"))
            else:
                date, ret = key, value
            ret = _f(ret)
            if ret is not None:
                out[_norm_date(date)] = ret
    elif isinstance(returns, list):
        for r in returns:
            if not isinstance(r, dict):
                continue
            date = r.get("date")
            ret = r.get("return", r.get("actual_return"))
            if date is None or ret is None:
                continue
            ret = _f(ret)
            if ret is not None:
                out[_norm_date(date)] = ret
    return dict(sorted(out.items()))


def _normalize_environment(environment):
    """转成 {date: {canonical_feature: value}}。"""
    out: dict[str, dict] = {}
    if environment is None:
        return out
    if isinstance(environment, pd.DataFrame):
        environment = environment.to_dict("records")
    if isinstance(environment, dict):
        # dict 可能是 {date: {features}}，也可能是单条记录。
        if "date" in environment:
            environment = [environment]
        else:
            for key, value in environment.items():
                if isinstance(value, dict):
                    out[_norm_date(key)] = _canonicalize_features(value)
                else:
                    out[_norm_date(key)] = {key: value}
            return out
    if isinstance(environment, list):
        for r in environment:
            if not isinstance(r, dict):
                continue
            date = r.get("date")
            if not date:
                continue
            features = _canonicalize_features(r)
            out[_norm_date(date)] = features
    return out


def _canonicalize_features(raw: dict) -> dict:
    out = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if key in {"date", "direction", "confidence", "return", "actual_return"}:
            continue
        canonical = _canonical_feature(key)
        if canonical is not None:
            out[canonical] = value
    return out


def _next_return_date(date: str, return_dates: list[str]):
    for candidate in return_dates:
        if candidate > date:
            return candidate
    return None


# ── 错题案例 ──────────────────────────────────────────────────────────────
def find_errors(history, returns, environment=None,
                confidence_threshold: float = CONFIDENCE_THRESHOLD,
                return_threshold: float = RETURN_THRESHOLD) -> list[dict]:
    """识别强看多 + 次日收益 <= return_threshold 的错题案例。"""
    history_rows = _normalize_history(history)
    returns_by_date = _normalize_returns(returns)
    env_by_date = _normalize_environment(environment)
    return_dates = list(returns_by_date)
    cases = []

    for row in history_rows:
        if not _is_bullish_direction(row["direction"]):
            continue
        if row["confidence"] < float(confidence_threshold):
            continue
        next_date = _next_return_date(row["date"], return_dates)
        if next_date is None:
            continue
        next_return = returns_by_date[next_date]
        if next_return <= float(return_threshold):
            features = _canonicalize_features(row["features"] or {})
            if not features:
                features = env_by_date.get(row["date"], {})
            cases.append({
                "date": row["date"],
                "next_date": next_date,
                "direction": row["direction"],
                "confidence": row["confidence"],
                "next_return": next_return,
                "features": features,
            })
    return cases


# ── 特征统计与规则提炼 ────────────────────────────────────────────────────
def _numeric_buckets(feature: str, values: list):
    """数值特征按中位数拆成 >= 与 < 两桶；非数值按取值拆桶。"""
    if not values:
        return []
    try:
        nums = sorted(float(v) for v in values)
        is_numeric = True
    except (TypeError, ValueError):
        is_numeric = False

    if is_numeric:
        median = nums[len(nums) // 2]
        return [
            {"operator": ">=", "value": _f(median, 4)},
            {"operator": "<", "value": _f(median, 4)},
        ]

    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    return [{"operator": "=", "value": v} for v in uniq]


def _condition_matches(value, cond: dict) -> bool:
    raw = _f(value)
    threshold = _f(cond.get("value"))
    if raw is None or threshold is None:
        return str(value) == str(cond.get("value"))
    op = cond.get("operator")
    if op == ">=":
        return raw >= threshold
    if op == "<":
        return raw < threshold
    return raw == threshold or str(value) == str(cond.get("value"))


def _extract_rules(cases: list[dict], rule_threshold: float) -> tuple[list[dict], list[dict]]:
    """返回 (rules, feature_stats)。"""
    n_cases = len(cases)
    if n_cases == 0:
        return [], []

    feature_names = []
    for case in cases:
        for key in case.get("features", {}):
            if key not in feature_names:
                feature_names.append(key)

    condition_map: dict[tuple[str, str, object], dict] = {}
    for feature in feature_names:
        values = [case.get("features", {}).get(feature) for case in cases
                  if feature in case.get("features", {})]
        for cond in _numeric_buckets(feature, values):
            condition_map[(feature, cond["operator"], cond["value"])] = {
                "feature": feature,
                "label": _feature_label(feature),
                "operator": cond["operator"],
                "value": cond["value"],
            }

    condition_rows = []
    for key, meta in condition_map.items():
        matches = sum(1 for case in cases
                      if meta["feature"] in case.get("features", {})
                      and _condition_matches(case["features"][meta["feature"]], meta))
        ratio = matches / n_cases
        text = f"{meta['label']} {meta['operator']} {meta['value']}"
        condition_rows.append({
            "key": key,
            "meta": meta,
            "text": text,
            "matches": matches,
            "ratio": ratio,
        })

    individual = [c for c in condition_rows if c["ratio"] > float(rule_threshold)]
    individual.sort(key=lambda x: (-x["ratio"], x["text"]))

    rules = []
    for c in individual:
        rules.append({
            "conditions": [{
                "feature": c["meta"]["feature"],
                "label": c["meta"]["label"],
                "operator": c["meta"]["operator"],
                "value": c["meta"]["value"],
                "text": c["text"],
            }],
            "match_count": c["matches"],
            "case_count": n_cases,
            "ratio": c["ratio"],
        })

    pair_rows = []
    for i, a in enumerate(individual):
        for b in individual[i + 1:]:
            if a["meta"]["feature"] == b["meta"]["feature"]:
                continue
            matches = 0
            for case in cases:
                fa = case.get("features", {})
                if (a["meta"]["feature"] in fa and _condition_matches(fa[a["meta"]["feature"]], a["meta"])
                        and b["meta"]["feature"] in fa
                        and _condition_matches(fa[b["meta"]["feature"]], b["meta"])):
                    matches += 1
            ratio = matches / n_cases
            if ratio > float(rule_threshold):
                pair_rows.append({
                    "conditions": [
                        {
                            "feature": a["meta"]["feature"],
                            "label": a["meta"]["label"],
                            "operator": a["meta"]["operator"],
                            "value": a["meta"]["value"],
                            "text": a["text"],
                        },
                        {
                            "feature": b["meta"]["feature"],
                            "label": b["meta"]["label"],
                            "operator": b["meta"]["operator"],
                            "value": b["meta"]["value"],
                            "text": b["text"],
                        },
                    ],
                    "match_count": matches,
                    "case_count": n_cases,
                    "ratio": ratio,
                })

    pair_rows.sort(key=lambda x: (-x["ratio"], " 且 ".join(c["text"] for c in x["conditions"])))
    rules.extend(pair_rows)
    rules.sort(key=lambda x: (-x["ratio"], " 且 ".join(c["text"] for c in x["conditions"])))
    for idx, rule in enumerate(rules, 1):
        rule["rule_id"] = f"R{idx:03d}"
        pct = rule["ratio"] * 100
        prefix = " 且 ".join(c["text"] for c in rule["conditions"])
        rule["failure_probability"] = round(rule["ratio"], 4)
        rule["confidence"] = round(rule["ratio"], 4)
        rule["statement"] = f"当 {prefix} 时，动量方向失效概率 {pct:.1f}%"

    feature_stats = []
    for feature in feature_names:
        values = [case.get("features", {}).get(feature) for case in cases
                  if feature in case.get("features", {})]
        stats = []
        for cond in _numeric_buckets(feature, values):
            matches = sum(1 for v in values if _condition_matches(v, cond))
            stats.append({
                "operator": cond["operator"],
                "value": cond["value"],
                "match_count": matches,
                "case_count": n_cases,
                "ratio": round(matches / n_cases, 4) if n_cases else 0.0,
            })
        feature_stats.append({
            "feature": feature,
            "label": _feature_label(feature),
            "present_count": len(values),
            "case_count": n_cases,
            "conditions": stats,
        })

    return rules, feature_stats


# ── 结果与输出 ─────────────────────────────────────────────────────────────
def _degraded_result(date: str, reason: str) -> dict:
    return {
        "schema": "error_book/v1",
        "date": date,
        "generated_at": _now(),
        "degraded": True,
        "reason": reason,
        "config": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "return_threshold": RETURN_THRESHOLD,
            "rule_threshold": RULE_THRESHOLD,
        },
        "total_recommendations": 0,
        "strong_bullish_count": 0,
        "n_cases": 0,
        "n_rules": 0,
        "cases": [],
        "rules": [],
        "feature_stats": [],
        "degraded_reasons": [reason],
    }


def run_error_book(history, returns, environment=None, date=None,
                   confidence_threshold: float = CONFIDENCE_THRESHOLD,
                   return_threshold: float = RETURN_THRESHOLD,
                   rule_threshold: float = RULE_THRESHOLD,
                   out_dir=None) -> dict:
    """识别错题 + 反向扫描特征 + 可选落盘。"""
    date = _norm_date(date)
    history_rows = _normalize_history(history)
    returns_by_date = _normalize_returns(returns)
    if not history_rows:
        result = _degraded_result(date, "no_history_data")
        if out_dir is not None:
            write_outputs(result, out_dir)
        return result
    if not returns_by_date:
        result = _degraded_result(date, "no_return_data")
        if out_dir is not None:
            write_outputs(result, out_dir)
        return result

    cases = find_errors(history_rows, returns_by_date, environment=environment,
                        confidence_threshold=confidence_threshold,
                        return_threshold=return_threshold)
    rules, feature_stats = _extract_rules(cases, rule_threshold)
    strong_bullish_count = sum(1 for r in history_rows
                               if _is_bullish_direction(r["direction"])
                               and r["confidence"] >= float(confidence_threshold))

    result = {
        "schema": "error_book/v1",
        "date": date,
        "generated_at": _now(),
        "degraded": False,
        "reason": None,
        "config": {
            "confidence_threshold": float(confidence_threshold),
            "return_threshold": float(return_threshold),
            "rule_threshold": float(rule_threshold),
        },
        "total_recommendations": len(history_rows),
        "strong_bullish_count": strong_bullish_count,
        "n_cases": len(cases),
        "n_rules": len(rules),
        "cases": cases,
        "rules": rules,
        "feature_stats": feature_stats,
        "degraded_reasons": [],
    }
    if out_dir is not None:
        write_outputs(result, out_dir)
    return result


def write_outputs(result: dict, out_dir) -> tuple[Path, Path]:
    out = Path(out_dir)
    day = result["date"].replace("-", "") if result.get("date") else "unknown"
    sub = out / "error_book" / day
    sub.mkdir(parents=True, exist_ok=True)
    json_path = sub / f"error_book_{result['date']}.json"
    md_path = sub / f"error_book_{result['date']}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    md_path.write_text(_to_md(result), encoding="utf-8")
    return json_path, md_path


def _to_md(result: dict) -> str:
    lines = [f"# 错题本 {result.get('date', '')}（V12.1 域Q）", ""]
    if result.get("degraded"):
        lines.append(f"> 降级：{result.get('reason', '')}，无错题案例可沉淀。")
        return "\n".join(lines)
    lines += [
        f"- 建议总数 {result.get('total_recommendations')} | 强看多 {result.get('strong_bullish_count')} "
        f"| 错题案例 {result.get('n_cases')} | 提炼规则 {result.get('n_rules')}",
        "",
        "## 案例",
        "| 建议日 | 次日 | 方向 | 置信度 | 次日收益 |",
        "|---|---|---:|---:|---:|",
    ]
    for c in result.get("cases", []):
        lines.append(f"| {c.get('date')} | {c.get('next_date')} | {c.get('direction')} "
                     f"| {c.get('confidence'):.2f} | {c.get('next_return'):.4f} |")
    lines += ["", "## 提炼规则"]
    if result.get("rules"):
        for rule in result["rules"]:
            lines.append(f"- {rule['rule_id']}：{rule['statement']} "
                         f"（{rule['match_count']}/{rule['case_count']}，置信 {rule['confidence']:.1%}）")
    else:
        lines.append("- 无超过阈值的失效环境特征组合")
    return "\n".join(lines)


# ── 默认数据加载 ───────────────────────────────────────────────────────────
def _battle_map_direction(map_data: dict) -> str:
    if "direction" in map_data:
        return str(map_data["direction"])
    rec = str(map_data.get("recommended") or "")
    mapping = {"进攻": "看多", "看多": "看多", "多": "看多", "增持": "看多",
               "防守": "看空", "看空": "看空", "空": "看空"}
    return mapping.get(rec, "震荡")


def load_battle_map_history(paths=None) -> list[dict]:
    """从 generated/battle_map_*.json 抽取 date/direction/confidence。"""
    if paths is None:
        paths = sorted(ROOT.glob("generated/battle_map_*.json"))
    else:
        paths = [Path(p) for p in paths if Path(p).exists()]
    out = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        date = data.get("date") or data.get("analyze_date")
        if not date:
            continue
        out.append({
            "date": date,
            "direction": _battle_map_direction(data),
            "confidence": _f(data.get("confidence")) or 0.0,
            "source": path.name,
        })
    return out


def load_index_returns(path: Path | str | None = None) -> dict[str, float]:
    """沪深300 日收益 dict，供 CLI 默认 next-return 使用。"""
    path = Path(path or DEFAULT_INDEX_PATH)
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        df = df.sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["return"] = df["close"].pct_change()
        return {r["date"]: _f(r["return"]) for _, r in df.iterrows() if _f(r["return"]) is not None}
    except Exception:
        return {}


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="错题本 + 失效环境反向扫描（V12.1 域Q）")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，缺省今天")
    ap.add_argument("--history", default=None, help="battle_map 历史 JSON（缺省自动读 generated/battle_map_*.json）")
    ap.add_argument("--returns", default=None, help="次日收益 JSON/parquet/csv（缺省沪深300 日收益）")
    ap.add_argument("--environment", default=None, help="环境特征 JSON/parquet/csv")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出根目录（默认 generated）")
    ap.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--return-threshold", type=float, default=RETURN_THRESHOLD)
    ap.add_argument("--rule-threshold", type=float, default=RULE_THRESHOLD)
    args = ap.parse_args(argv)

    history = _read_table(args.history) if args.history else load_battle_map_history()
    returns = _read_table(args.returns) if args.returns else load_index_returns()
    environment = _read_table(args.environment) if args.environment else None

    result = run_error_book(history, returns, environment=environment, date=args.date,
                            confidence_threshold=args.confidence,
                            return_threshold=args.return_threshold,
                            rule_threshold=args.rule_threshold,
                            out_dir=args.out_dir)
    if result.get("degraded"):
        print(f"[error_book] {result['date']} 降级（{result.get('reason')}）：无错题案例")
    else:
        print(f"[error_book] {result['date']} 案例={result['n_cases']} "
              f"规则={result['n_rules']} 强看多={result['strong_bullish_count']}")
        for rule in result["rules"]:
            print(f"  {rule['rule_id']} {rule['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
