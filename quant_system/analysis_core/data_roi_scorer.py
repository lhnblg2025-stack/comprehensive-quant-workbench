"""data_roi_scorer — Data ROI 评分器（V12.1 阶段3 域N）。

目标:
  - 从 107 数据集中找出真正参与决策的高边际贡献源
  - 低效用源不绝对删除，而是降级到冷管道
  - 行业专属数据按场景独立评分：全局 ROI 低但特定场景 ROI 高 → 场景特种兵

方法（简化可复现，mock 数据驱动）:
  - 所有源先 z-score 标准化，再取等权 ensemble 作为基线信号
  - 逐源剔除后重算 ensemble 与目标的相关性、方向命中率
  - raw 贡献 = 0.5 * corr_delta + 0.5 * dir_delta
  - 跨源 z-score 后映射到 0~100 ROI 分；tier 按 Top20%/Bottom20% 分档

用法:
  python3 -m quant_system.analysis_core.data_roi_scorer --date 2026-08-13 --mock
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = ROOT / "generated" / "data_roi"
CST = timezone(__import__("datetime").timedelta(hours=8))

HOT_RATIO = 0.20
COLD_RATIO = 0.20


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _as_date(value: str | None) -> str:
    value = value or _today()
    return str(value).strip().replace("/", "-")


def _align_series(value, index: pd.Index, name: str | None = None) -> pd.Series:
    """DataFrame/Series/list → 与目标 index 对齐的 Series。"""
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        out = value.reindex(index)
    else:
        out = pd.Series(list(value), index=index)
    out = pd.to_numeric(out, errors="coerce")
    out.name = name
    return out


def _coerce_sources(dataset, index: pd.Index) -> dict[str, pd.Series]:
    """输入映射或 DataFrame → {source_name: aligned Series}。"""
    if isinstance(dataset, pd.DataFrame):
        items = [(str(col), dataset[col]) for col in dataset.columns]
    else:
        items = [(str(name), series) for name, series in dict(dataset).items()]
    return {name: _align_series(series, index, name) for name, series in items}


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index, name=s.name)
    return ((s - s.mean()) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    corr = a.corr(b)
    return float(corr) if np.isfinite(corr) else 0.0


def _direction_hit_rate(pred: pd.Series, target: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "target": target}).dropna()
    frame = frame[(frame["pred"] != 0) & (frame["target"] != 0)]
    if frame.empty:
        return 0.0
    return float(np.mean(np.sign(frame["pred"]) == np.sign(frame["target"])))


def _ensemble_signal(sources: dict[str, pd.Series]) -> pd.Series:
    if not sources:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({name: _zscore(s) for name, s in sources.items()})
    return frame.mean(axis=1)


def _eval(sources: dict[str, pd.Series], target: pd.Series) -> dict[str, float]:
    signal = _ensemble_signal(sources)
    if signal.empty or len(signal) < 2:
        return {"corr": 0.0, "direction": 0.0}
    return {
        "corr": _pearson(signal, target),
        "direction": _direction_hit_rate(signal, target),
    }


def _raw_contribution(
    target_series: pd.Series,
    source_name: str,
    sources: dict[str, pd.Series],
    baseline: dict[str, float] | None = None,
) -> float:
    """单源剔除后相对 baseline 的 raw 边际贡献。"""
    leave = {name: s for name, s in sources.items() if name != source_name}
    if not leave:
        return 0.0
    base = baseline if baseline is not None else _eval(sources, target_series)
    left = _eval(leave, target_series)
    corr_delta = base["corr"] - left["corr"]
    dir_delta = base["direction"] - left["direction"]
    return 0.5 * corr_delta + 0.5 * dir_delta


def _raws_to_scores(raws: Sequence[float]) -> list[float]:
    """跨源 raw 贡献 z-score 后映射到 0~100，保持相对排序。"""
    vals = np.asarray(raws, dtype=float)
    if len(vals) == 0:
        return []
    if len(vals) == 1 or not np.isfinite(vals.std()):
        return [50.0] * len(vals)
    std = float(vals.std())
    if std == 0:
        return [50.0] * len(vals)
    z = (vals - vals.mean()) / std
    scores = 50.0 + 50.0 * np.tanh(z)
    return [round(float(min(max(x, 0.0), 100.0)), 3) for x in scores]


def _tier_from_scores(scores: Sequence[float]) -> list[str]:
    """Top20% hot，Bottom20% cold，其余 warm。"""
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    n_hot = max(1, int(round(n * HOT_RATIO))) if n >= 5 else (1 if n == 1 else 0)
    n_cold = max(1, int(round(n * COLD_RATIO))) if n >= 5 else (1 if n >= 2 else 0)
    tiers = ["warm"] * n
    for pos, idx in enumerate(order):
        if pos < n_hot:
            tiers[idx] = "hot"
        elif pos >= n - n_cold:
            tiers[idx] = "cold"
    return tiers


def _sign(score: float) -> str:
    if score > 50.0 + 1e-9:
        return "positive"
    if score < 50.0 - 1e-9:
        return "negative"
    return "neutral"


def _normalize_scopes(scopes, source: str | None = None) -> list[str]:
    if isinstance(scopes, str):
        out = [scopes]
    elif isinstance(scopes, (list, tuple, set)):
        out = [str(x) for x in scopes]
    else:
        out = ["global"]
    out = [x for x in out if x]
    if not out:
        out = ["global"]
    if "global" not in out:
        out.insert(0, "global")
    return out


def build_mock_dataset(n_sources: int = 10, n: int = 120, seed: int = 7) -> dict:
    """构造可控 mock 数据：含强源/弱源/噪声源/反源/场景特种兵源。

    返回:
      dataset: {source_00..: Series}
      target: 全局目标（次日全A涨跌的连续代理）
      source_scopes: 每源场景标签（source_09 为 chain_auto）
      scenario_targets: {chain_auto: Series}
    """
    rng = np.random.default_rng(seed)
    target = pd.Series(rng.normal(0, 1, n), name="target")
    dataset: dict[str, pd.Series] = {}

    # 前 2 个强源
    for i in range(2):
        dataset[f"source_{i:02d}"] = pd.Series(
            target.to_numpy() + rng.normal(0, 0.12, n),
            index=target.index, name=f"source_{i:02d}")
    # 中间 4 个中等源
    for i in range(2, 6):
        dataset[f"source_{i:02d}"] = pd.Series(
            target.to_numpy() + rng.normal(0, 0.9, n),
            index=target.index, name=f"source_{i:02d}")
    # 3 个噪声源
    for i in range(6, 9):
        dataset[f"source_{i:02d}"] = pd.Series(
            rng.normal(0, 1, n), index=target.index, name=f"source_{i:02d}")

    # 场景特种兵源：对全局目标为弱反信号（全局应降冷），对 chain_auto 是强正信号
    special_source = pd.Series(
        -target.to_numpy() * 0.8 + rng.normal(0, 0.10, n),
        index=target.index, name="source_09")
    dataset["source_09"] = special_source
    scenario_targets = {
        "chain_auto": pd.Series(
            special_source.to_numpy() * 0.9 + rng.normal(0, 0.05, n),
            index=target.index, name="chain_auto"),
    }

    source_scopes = {name: ["global"] for name in dataset}
    source_scopes["source_09"] = ["global", "chain_auto"]
    return {
        "dataset": dataset,
        "target": target,
        "source_scopes": source_scopes,
        "scenario_targets": scenario_targets,
    }


def _score_block(
    target_series: pd.Series,
    sources: dict[str, pd.Series],
    names: list[str],
) -> tuple[list[float], list[str]]:
    baseline = _eval(sources, target_series)
    raws = [_raw_contribution(target_series, name, sources, baseline) for name in names]
    scores = _raws_to_scores(raws)
    tiers = _tier_from_scores(scores)
    return scores, tiers


def score_sources(
    dataset,
    target: pd.Series,
    source_scopes: Mapping[str, Sequence[str] | str] | None = None,
    scenario_targets: Mapping[str, pd.Series] | None = None,
    date: str | None = None,
) -> dict:
    """逐源计算 ROI 分、tier、场景标签，返回完整报告 dict。"""
    date = _as_date(date)
    if target is None:
        target = pd.Series(dtype=float)

    sources = _coerce_sources(dataset, target.index if isinstance(target, pd.Series) else pd.Index([]))
    if not sources or len(target) < 2:
        return {
            "date": date,
            "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
            "sources": [],
            "summary": {
                "n_sources": 0,
                "tier_counts": {"hot": 0, "warm": 0, "cold": 0},
                "special_forces": [],
                "degraded": True,
                "note": "无可用数据，已降级为空报告",
            },
        }

    target = _align_series(target, target.index)
    scenario_targets = {str(k): _align_series(v, target.index, str(k))
                        for k, v in dict(scenario_targets or {}).items()}
    source_scopes = dict(source_scopes or {})
    names = sorted(sources)

    global_scores, global_tiers = _score_block(target, sources, names)
    scenario_blocks: dict[str, tuple[list[float], list[str]]] = {}
    for scenario_name, scenario_target in scenario_targets.items():
        scenario_blocks[scenario_name] = _score_block(scenario_target, sources, names)

    rows: list[dict] = []
    for pos, name in enumerate(names):
        global_score = global_scores[pos]
        global_tier = global_tiers[pos]
        scenario_roi: dict[str, float] = {"global": round(global_score, 3)}
        scenario_tiers: dict[str, str] = {"global": global_tier}
        for scenario_name, (scores, tiers) in scenario_blocks.items():
            scenario_roi[str(scenario_name)] = round(scores[pos], 3)
            scenario_tiers[str(scenario_name)] = tiers[pos]

        scopes = _normalize_scopes(source_scopes.get(name), name)
        special_forces = [
            scenario_name
            for scenario_name in scenario_blocks
            if scenario_name in scopes
            and global_tier == "cold"
            and scenario_tiers[scenario_name] == "hot"
        ]
        if special_forces:
            suggestion = "场景特种兵"
        elif global_tier == "hot":
            suggestion = "hot核心"
        elif global_tier == "warm":
            suggestion = "warm观察"
        else:
            suggestion = "cold低效用降级冷管道"

        rows.append({
            "source": name,
            "roi": round(global_score, 3),
            "sign": _sign(global_score),
            "tier": global_tier,
            "scopes": scopes,
            "scenario_roi": scenario_roi,
            "scenario_tiers": scenario_tiers,
            "special_forces": special_forces,
            "suggestion": suggestion,
        })

    rows.sort(key=lambda row: (-row["roi"], row["source"]))
    tier_counts = {"hot": 0, "warm": 0, "cold": 0}
    special_names: list[str] = []
    for row in rows:
        tier_counts[row["tier"]] += 1
        if row["special_forces"]:
            special_names.append(row["source"])

    return {
        "date": date,
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "sources": rows,
        "summary": {
            "n_sources": len(rows),
            "tier_counts": tier_counts,
            "special_forces": sorted(special_names),
            "degraded": False,
            "note": "mock 边际贡献 ROI 评分完成",
        },
    }


def write_report(result: dict, output_dir: Path | str | None = None) -> Path:
    """写 data_roi_{date}.json 和 .md，返回日期子目录。"""
    date_key = str(result.get("date") or _today()).replace("-", "")
    root = Path(output_dir) if output_dir is not None else OUT_ROOT
    out_dir = root / date_key
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"data_roi_{date_key}.json"
    md_path = out_dir / f"data_roi_{date_key}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    return out_dir


def _render_markdown(result: dict) -> str:
    date = result.get("date") or ""
    summary = result.get("summary", {})
    lines = [f"# Data ROI 评分器 — {date}", ""]
    if summary.get("degraded"):
        lines += ["- 状态: 无可用数据，已降级为空报告", ""]
        return "\n".join(lines) + "\n"

    counts = summary.get("tier_counts", {})
    lines += [
        f"- 数据源: {summary.get('n_sources', 0)}",
        f"- tier: hot {counts.get('hot', 0)} / warm {counts.get('warm', 0)} / cold {counts.get('cold', 0)}",
        f"- 场景特种兵: {', '.join(summary.get('special_forces', [])) or '无'}",
        "",
        "| source | ROI | sign | tier | scopes | special_forces | suggestion |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in result.get("sources", []):
        scopes = ",".join(row.get("scopes", []))
        special = ",".join(row.get("special_forces", [])) or "-"
        lines.append(
            f"| {row['source']} | {row['roi']:.3f} | {row['sign']} | {row['tier']} "
            f"| {scopes} | {special} | {row['suggestion']} |"
        )
    return "\n".join(lines) + "\n"


def run(
    date: str | None = None,
    dataset=None,
    target: pd.Series | None = None,
    source_scopes: Mapping[str, Sequence[str] | str] | None = None,
    scenario_targets: Mapping[str, pd.Series] | None = None,
    output_dir: Path | str | None = None,
) -> dict:
    """评分并落盘，返回报告 dict。"""
    date = _as_date(date)
    result = score_sources(
        dataset if dataset is not None else {},
        target if target is not None else pd.Series(dtype=float),
        source_scopes=source_scopes,
        scenario_targets=scenario_targets,
        date=date,
    )
    write_report(result, output_dir=output_dir)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Data ROI 评分器")
    ap.add_argument("--date", default=None, help="目标日期 YYYY-MM-DD，默认今天")
    ap.add_argument("--mock", action="store_true", help="使用 mock 数据跑通（离线）")
    ap.add_argument("--output-dir", default=None, help="输出根目录，默认 generated/data_roi")
    args = ap.parse_args()
    if args.mock:
        data = build_mock_dataset(seed=7)
        run(date=args.date, dataset=data["dataset"], target=data["target"],
            source_scopes=data["source_scopes"], scenario_targets=data["scenario_targets"],
            output_dir=args.output_dir)
    else:
        run(date=args.date, output_dir=args.output_dir)
