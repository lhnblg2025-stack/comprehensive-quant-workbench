"""
factor_generator — 规则因子生成器（轻量版，V11 研究工具）

Spec: 项目文档/量化交易系统/spec_factor_generator.md

定位: 研究工具，不接入决策主链，输出报告供参考；只写 generated/ 报告，不改任何配置。

方法:
  1. 因子模板库（表达式骨架 + 参数网格）→ generate_candidates() 全组合展开（41 个）
  2. 固定 seed 随机抽样 200 只股票，近 250 交易日，全向量化
  3. 每日截面: 因子值 → 次日收益 Spearman rank IC
  4. 筛选: |IC均值|>0.02 且 ICIR>0.3 且 覆盖度>80% → 有效因子
  5. 输出 generated/factor_generator_report_{date}.json + .md

指标口径:
  ic_mean    = 评估窗口内每日截面 rank IC 的均值
  ic_std     = 每日 IC 的标准差
  icir       = ic_mean / ic_std（日频，未年化）
  icir_ann   = icir * sqrt(252)（年化，仅供参考）
  win_rate   = IC 与 ic_mean 同号的交易日占比
  coverage   = 评估窗口内因子非 NaN 单元格占比（全 NaN 股票先剔除）
  每日 IC 要求同截面有效样本 >= 10；因子统计要求有效天数 >= 20

数据: data_warehouse/kline/*.parquet（本地，禁止网络）

用法:
  python3 -m quant_system.analysis_core.factor_generator --list    # 打印候选因子
  python3 -m quant_system.analysis_core.factor_generator --report  # 生成报告
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
KLINE_DIR = ROOT / "data_warehouse" / "kline"
KLINE_COLUMNS = ["date", "high", "low", "close", "volume", "pct_chg"]
OUT_DIR = ROOT / "generated"

# ── 配置 ────────────────────────────────────────────────────
SEED = 42                 # 抽样固定 seed
MIN_ROWS = 60             # 股票最少历史行数（不足跳过）
WARMUP = 170              # 评估窗口前额外留的 warmup 行（覆盖 bias_144 等长窗口）
MIN_STOCKS_PER_DAY = 10   # 每日截面有效样本下限
MIN_VALID_DAYS = 20       # 因子统计有效天数下限
IC_MEAN_TH = 0.02         # 筛选: |IC均值| 阈值
ICIR_TH = 0.30            # 筛选: ICIR 阈值
COV_TH = 0.80             # 筛选: 覆盖度阈值

# ── 因子模板库（表达式骨架 + 参数网格）───────────────────────
# spec 规定的五类 + 补充骨架（位置/量均线比/偏度/振幅/累计涨跌），
# 参数网格全组合展开后共 33 个基础因子 + 8 个合成因子 = 41 个候选。
FAMILY_TEMPLATES: list[dict] = [
    {"family": "动量",     "name": "mom",       "formula": "close/close.shift(N)-1",
     "params": [5, 10, 20, 60, 120]},
    {"family": "反转",     "name": "rev",       "formula": "-(close/close.shift(N)-1)",
     "params": [5, 10, 20]},
    {"family": "波动",     "name": "vol",       "formula": "ret.rolling(N).std()",
     "params": [5, 10, 20, 60]},
    {"family": "量能",     "name": "vol_ratio", "formula": "volume/volume.rolling(N).mean()",
     "params": [5, 10, 20, 60]},
    {"family": "乖离",     "name": "bias",      "formula": "close/close.rolling(N).mean()-1",
     "params": [10, 20, 60, 144]},
    {"family": "位置",     "name": "pos",       "formula": "(close-low.rolling(N).min())/(high.rolling(N).max()-low.rolling(N).min())",
     "params": [10, 20, 60]},
    {"family": "量均线比", "name": "vma_ratio", "formula": "volume.rolling(N).mean()/volume.rolling(2*N).mean()-1",
     "params": [5, 10, 20]},
    {"family": "收益偏度", "name": "skew",      "formula": "ret.rolling(N).skew()",
     "params": [10, 20]},
    {"family": "振幅",     "name": "amp",       "formula": "((high-low)/close).rolling(N).mean()",
     "params": [5, 20]},
    {"family": "累计涨跌", "name": "pct",       "formula": "pct_chg.rolling(N).sum()",
     "params": [5, 10, 20]},
]

SYNTHETIC_TEMPLATES: list[dict] = [
    {"family": "合成", "name": "mom20_div_vol20",       "formula": "mom_20/vol_20"},
    {"family": "合成", "name": "mom5_div_vol10",        "formula": "mom_5/vol_10"},
    {"family": "合成", "name": "mom60_div_vol60",       "formula": "mom_60/vol_60"},
    {"family": "合成", "name": "mom20_div_vol60",       "formula": "mom_20/vol_60"},
    {"family": "合成", "name": "rev5_div_vol10",        "formula": "rev_5/vol_10"},
    {"family": "合成", "name": "mom20_mul_vol_ratio20", "formula": "mom_20*vol_ratio_20"},
    {"family": "合成", "name": "bias20_div_vol20",      "formula": "bias_20/vol_20"},
    {"family": "合成", "name": "mom20_minus_mom5",      "formula": "mom_20-mom_5"},
]

_BASE_NS: dict[str, list[int]] = {t["name"]: t["params"] for t in FAMILY_TEMPLATES}


# ────────────────────────────────────────────────────────────
# 候选展开
# ────────────────────────────────────────────────────────────
def generate_candidates() -> list[dict]:
    """[{name, formula, params}] 全组合展开。"""
    out: list[dict] = []
    for t in FAMILY_TEMPLATES:
        for n in t["params"]:
            out.append({
                "name": f"{t['name']}_{n}",
                "formula": t["formula"].replace("N", str(n)),
                "params": {"family": t["family"], "N": n},
            })
    for t in SYNTHETIC_TEMPLATES:
        out.append({
            "name": t["name"],
            "formula": t["formula"],
            "params": {"family": t["family"], "expr": t["formula"]},
        })
    return out


# ────────────────────────────────────────────────────────────
# 数据准备与因子计算（向量化）
# ────────────────────────────────────────────────────────────
def _prepare(raw: pd.DataFrame, days: int) -> pd.DataFrame | None:
    """清洗并按 日期索引 + 尾部切片（评估窗口 + warmup），返回 None 表示样本不足。"""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = (df.drop_duplicates("date", keep="last")
            .sort_values("date")
            .set_index("date"))
    df = df[df["close"] > 0]
    if len(df) < MIN_ROWS:
        return None
    return df.tail(days + WARMUP)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """向量除法，除数为 0/NaN 时结果置 NaN（避免 inf 污染 rank IC）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        res = a / b
    return res.where(np.isfinite(res))


def _compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """在单只股票序列上一次性向量化计算全部候选因子 + 次日收益。"""
    close = df["close"]
    high, low = df["high"], df["low"]
    volume = df["volume"]
    ret = close.pct_change()

    cols: dict[str, pd.Series] = {}
    ns = _BASE_NS
    mom: dict[int, pd.Series] = {}
    for n in ns["mom"]:
        mom[n] = close / close.shift(n) - 1.0
        cols[f"mom_{n}"] = mom[n]
    for n in ns["rev"]:
        cols[f"rev_{n}"] = -mom[n]
    for n in ns["vol"]:
        cols[f"vol_{n}"] = ret.rolling(n).std()
    for n in ns["vol_ratio"]:
        cols[f"vol_ratio_{n}"] = _safe_div(volume, volume.rolling(n).mean())
    for n in ns["bias"]:
        cols[f"bias_{n}"] = close / close.rolling(n).mean() - 1.0
    for n in ns["pos"]:
        hi = high.rolling(n).max()
        lo = low.rolling(n).min()
        den = (hi - lo).replace(0.0, np.nan)
        cols[f"pos_{n}"] = (close - lo) / den
    for n in ns["vma_ratio"]:
        cols[f"vma_ratio_{n}"] = (
            _safe_div(volume.rolling(n).mean(), volume.rolling(2 * n).mean()) - 1.0
        )
    for n in ns["skew"]:
        cols[f"skew_{n}"] = ret.rolling(n).skew()
    for n in ns["amp"]:
        cols[f"amp_{n}"] = ((high - low) / close).rolling(n).mean()
    for n in ns["pct"]:
        cols[f"pct_{n}"] = df["pct_chg"].rolling(n).sum()

    # 合成类（引用上面已计算的基础因子列）
    cols["mom20_div_vol20"] = _safe_div(cols["mom_20"], cols["vol_20"])
    cols["mom5_div_vol10"] = _safe_div(cols["mom_5"], cols["vol_10"])
    cols["mom60_div_vol60"] = _safe_div(cols["mom_60"], cols["vol_60"])
    cols["mom20_div_vol60"] = _safe_div(cols["mom_20"], cols["vol_60"])
    cols["rev5_div_vol10"] = _safe_div(cols["rev_5"], cols["vol_10"])
    cols["mom20_mul_vol_ratio20"] = cols["mom_20"] * cols["vol_ratio_20"]
    cols["bias20_div_vol20"] = _safe_div(cols["bias_20"], cols["vol_20"])
    cols["mom20_minus_mom5"] = cols["mom_20"] - cols["mom_5"]

    cols["next_ret"] = close.shift(-1) / close - 1.0
    return pd.DataFrame(cols, index=df.index)


# ────────────────────────────────────────────────────────────
# IC 统计（截面 Spearman，向量化）
# ────────────────────────────────────────────────────────────
def _spearman_wide(f: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
    """宽表逐日 Spearman: 对因子/收益逐列 rank 后按列算 Pearson。
    返回每个交易日一行的 IC Series（有效样本不足的日子为 NaN）。"""
    fr = f.rank(axis=0, na_option="keep")
    rr = r.rank(axis=0, na_option="keep")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=0)
    fr = fr.where(valid)
    rr = rr.where(valid)
    fcen = fr - fr.mean(axis=0)
    rcen = rr - rr.mean(axis=0)
    num = (fcen * rcen).sum(axis=0)
    den = np.sqrt((fcen.pow(2).sum(axis=0)) * (rcen.pow(2).sum(axis=0)))
    return (num / den).where(n >= MIN_STOCKS_PER_DAY)


def _factor_stats(f: pd.DataFrame, r: pd.DataFrame) -> dict:
    """单因子统计: ic_mean/ic_std/icir/icir_ann/win_rate/coverage/n_days/n_stocks。"""
    f = f.loc[:, f.notna().any()]          # 因子值全 NaN 的股票跳过
    r = r.loc[:, f.columns]
    n_stocks = int(f.shape[1])
    if n_stocks == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "icir_ann": np.nan, "win_rate": np.nan, "coverage": np.nan,
                "n_days": 0, "n_stocks": 0}
    ic = _spearman_wide(f, r).dropna()
    coverage = float(f.notna().sum().sum() / (f.shape[0] * f.shape[1]))
    n_days = int(len(ic))
    if n_days < MIN_VALID_DAYS:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "icir_ann": np.nan, "win_rate": np.nan, "coverage": coverage,
                "n_days": n_days, "n_stocks": n_stocks}
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    if ic_mean != 0.0:
        win_rate = float((np.sign(ic.to_numpy()) == np.sign(ic_mean)).mean())
    else:
        win_rate = np.nan
    return {
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "icir_ann": icir * np.sqrt(252.0) if np.isfinite(icir) else np.nan,
        "win_rate": win_rate, "coverage": coverage,
        "n_days": n_days, "n_stocks": n_stocks,
    }


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────
def evaluate(symbols_sample: int = 200, days: int = 250) -> pd.DataFrame:
    """抽样 symbols_sample 只 × 近 days 交易日，逐因子算 IC 统计并排序。"""
    candidates = generate_candidates()
    files = sorted(KLINE_DIR.glob("*.parquet"))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(files))

    stocks: dict[str, pd.DataFrame] = {}
    for j in order:
        if len(stocks) >= symbols_sample:
            break
        path = files[int(j)]
        try:
            raw = pd.read_parquet(path, columns=KLINE_COLUMNS)
        except Exception as e:
            logging.getLogger(__name__).error(f"[factor_generator] 操作失败: {e}", exc_info=True)
            continue
        df = _prepare(raw, days)
        if df is None:
            continue
        stocks[path.stem] = _compute_factors(df)
    if len(stocks) < 30:
        raise RuntimeError(f"有效股票样本不足: {len(stocks)}")

    all_dates = sorted({d for fdf in stocks.values() for d in fdf.index})
    eval_dates = all_dates[-days:]
    frames = {sym: fdf.reindex(eval_dates) for sym, fdf in stocks.items()}
    ret_wide = pd.DataFrame({sym: fr["next_ret"] for sym, fr in frames.items()},
                            index=eval_dates)

    rows: list[dict] = []
    for cand in candidates:
        name = cand["name"]
        f = pd.DataFrame({sym: fr[name] for sym, fr in frames.items()},
                         index=eval_dates)
        stats = _factor_stats(f, ret_wide)
        valid = (
            np.isfinite(stats["ic_mean"]) and abs(stats["ic_mean"]) > IC_MEAN_TH
            and np.isfinite(stats["icir"]) and stats["icir"] > ICIR_TH
            and stats["coverage"] > COV_TH
        )
        row = {
            "name": name,
            "formula": cand["formula"],
            "family": cand["params"]["family"],
            "params": cand["params"],
            **{k: (round(v, 4) if isinstance(v, float) and np.isfinite(v)
                   else (None if isinstance(v, float) else v))
               for k, v in stats.items()},
            "is_valid": bool(valid),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        "ic_mean", key=lambda s: s.abs(), ascending=False, na_position="last"
    )
    return df.reset_index(drop=True)


def _render_md(out: dict) -> str:
    lines = [
        "# 因子生成器研究报告（factor_generator）",
        "",
        f"- 生成日期: {out['date']}",
        f"- 样本: {out['config']['symbols_sample']} 只股票 × 近 {out['config']['days']} 交易日"
        f"（seed={out['config']['seed']}）",
        f"- 候选因子: {out['n_candidates']} 个 / 有效因子: {out['n_valid']} 个",
        f"- 耗时: {out['runtime_seconds']} s",
        f"- 筛选阈值: |IC均值|>{out['config']['ic_mean_th']}，ICIR>{out['config']['icir_th']}，"
        f"覆盖度>{out['config']['coverage_th']}",
        "",
    ]
    if out["valid_factors"]:
        lines += ["## 有效因子", "", "| 因子 | 公式 | IC均值 | ICIR | 胜率 | 覆盖度 |",
                  "|---|---|---|---|---|---|"]
        for v in out["valid_factors"]:
            lines.append(
                f"| {v['name']} | {v['formula']} | {v['ic_mean']:.4f} | {v['icir']:.3f} | "
                f"{v['win_rate']:.3f} | {v['coverage']:.3f} |"
            )
        lines.append("")
    lines += ["## 全部因子统计（按 |IC均值| 降序）", "",
              "| 因子 | 家族 | IC均值 | ICIR | ICIR年化 | 胜率 | 覆盖度 | 有效天数 | 股票数 | 有效 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["factors"]:
        def fmt(x: float | None) -> str:
            return "-" if x is None else f"{x:.4f}"
        lines.append(
            f"| {r['name']} | {r['family']} | {fmt(r['ic_mean'])} | {fmt(r['icir'])} | "
            f"{fmt(r['icir_ann'])} | {fmt(r['win_rate'])} | {fmt(r['coverage'])} | "
            f"{r['n_days']} | {r['n_stocks']} | {'是' if r['is_valid'] else ''} |"
        )
    lines += ["", "## 结论", "", out["conclusion"], ""]
    return "\n".join(lines)


def run_report() -> dict:
    """汇总评估结果，写 generated/factor_generator_report_{date}.json + .md，返回摘要。"""
    t0 = time.perf_counter()
    ev = evaluate()
    elapsed = round(time.perf_counter() - t0, 2)

    valid = ev[ev["is_valid"]]
    date_s = datetime.now(CST).date().isoformat()
    key_cols = ["name", "formula", "ic_mean", "icir", "win_rate", "coverage"]
    valid_factors = [{k: v for k, v in r.items() if k in key_cols}
                     for r in valid.to_dict("records")]

    if valid_factors:
        top = "，".join(
            f"{v['name']}(IC={v['ic_mean']:.4f}/ICIR={v['icir']:.3f}/胜率={v['win_rate']:.2f}/"
            f"覆盖度={v['coverage']:.2f})"
            for v in valid_factors[:10]
        )
        conclusion = (f"有效因子 {len(valid_factors)} 个（按 |IC均值| 降序）: {top}"
                      + ("…" if len(valid_factors) > 10 else ""))
    else:
        conclusion = ("无有效因子: 当前阈值 |IC均值|>0.02、ICIR>0.3、覆盖度>80% 下，"
                      "无候选因子通过筛选。")

    out = {
        "module": "factor_generator",
        "date": date_s,
        "config": {
            "symbols_sample": 200, "days": 250, "seed": SEED,
            "n_candidates": int(len(ev)),
            "ic_mean_th": IC_MEAN_TH, "icir_th": ICIR_TH, "coverage_th": COV_TH,
            "min_stocks_per_day": MIN_STOCKS_PER_DAY, "min_valid_days": MIN_VALID_DAYS,
        },
        "runtime_seconds": elapsed,
        "n_candidates": int(len(ev)),
        "n_valid": int(len(valid_factors)),
        "valid_factors": valid_factors,
        "factors": ev.to_dict("records"),
        "conclusion": conclusion,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pj = OUT_DIR / f"factor_generator_report_{date_s}.json"
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pm = OUT_DIR / f"factor_generator_report_{date_s}.md"
    pm.write_text(_render_md(out), encoding="utf-8")
    print(f"[factor_generator] 已输出 {pj} / {pm}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="规则因子生成器（轻量版，研究工具）")
    ap.add_argument("--list", action="store_true", help="打印候选因子")
    ap.add_argument("--report", action="store_true", help="生成报告（默认）")
    args = ap.parse_args()
    if args.list:
        for c in generate_candidates():
            print(f"{c['name']:<24} {c['formula']}")
        return
    if not args.report:
        args.report = True
    print(json.dumps(run_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
