#!/usr/bin/env python3
"""ic_oos_report.py — 因子方向样本外验证报告 (V12.3 分析闭环)

目的: 证明"IC 校准的 direction"不是全样本过拟合——用前 60% 交易日
计算 direction, 后 40% 交易日验证 IC 符号保持率/ICIR/胜率。

方法:
  1. 抽样 N 只股票(seed 固定可复现) → 公共交易日序列
  2. 切分 train(前60%) / test(后40%)
  3. train: 全因子面板 → forward 5 日收益 → 逐日 IC → direction=sign(ic_mean)
  4. test:  同样计算 → 验证段 ic_mean/icir/winrate, 与 train direction 比对
  5. 汇总: 符号一致率 / 翻转列表 / 平均|IC|保持 / 稳定 Top 因子
  6. 输出 generated/ic_report/IC_OOS_REPORT.md + .json

用法:
  python3 scripts/ic_oos_report.py [--n 150] [--seed 42] [--forward 5]
  (云端月度任务 quant_ic_oos, 全量建议 --n 300)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KLINE_DIR = ROOT / "data_warehouse" / "kline"
VALUATION_DIR = ROOT / "data_warehouse" / "valuation"
FINANCIAL_DIR = ROOT / "data_warehouse" / "financial"
OUT_DIR = ROOT / "generated" / "ic_report"


def _load_sample(n: int, seed: int) -> dict[str, pd.DataFrame]:
    """抽样 N 只股票的日K(需足够历史)。"""
    kl = {}
    files = sorted(KLINE_DIR.glob("*.parquet"))
    rng = random.Random(seed)
    rng.shuffle(files)
    for f in files:
        if len(kl) >= n:
            break
        try:
            df = pd.read_parquet(f)
            if len(df) < 500:  # 至少 500 交易日供切分
                continue
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            kl[f.stem] = df.sort_values("date")
        except Exception:  # noqa: BLE001
            continue
    return kl


def _split_dates(kl: dict[str, pd.DataFrame], test_frac: float = 0.4) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """公共交易日 → train/test 切分。"""
    common = None
    for code, df in kl.items():
        d = pd.DatetimeIndex(df["date"].drop_duplicates())
        common = d if common is None else common.intersection(d)
    if common is None or len(common) < 120:
        raise RuntimeError("公共交易日不足")
    cut = int(len(common) * (1 - test_frac))
    return common[:cut], common[cut:]


def _ic_stats(kl: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, names: list[str],
              series_fn, forward: int) -> pd.DataFrame:
    """指定时段: 全因子面板 → 逐日 IC → 每因子 ic_mean/icir/winrate。"""
    import ic_vectorized as icv
    kl_sub = {}
    for code, df in kl.items():
        sub = df[(df["date"] >= dates[0]) & (df["date"] <= dates[-1])]
        if len(sub) >= 60:
            kl_sub[code] = sub
    if not kl_sub:
        return pd.DataFrame()
    common, codes = icv._prepare_common(kl_sub, dates)
    if common is None or len(common) < 30:
        return pd.DataFrame()
    panels = icv._build_panels(kl_sub, names, series_fn, common, codes,
                               None, "oos")
    rows = []
    for name in names:
        fw = panels.get(name)
        if fw is None:
            continue
        rets = icv.build_forward_returns(kl_sub, forward)
        ret_wide = rets.reindex(fw.index)
        ic = icv.compute_ic_fast(fw, ret_wide, min_n=30)
        ic = ic.dropna()
        if len(ic) < 20:
            continue
        rows.append({
            "factor": name,
            "ic_mean": float(ic.mean()),
            "icir": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else 0.0,
            "winrate": float((ic > 0).mean()),
            "n_days": int(len(ic)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="因子方向样本外验证")
    ap.add_argument("--n", type=int, default=150, help="抽样股票数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--forward", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kl = _load_sample(args.n, args.seed)
    if not kl:
        print("[oos] 无样本数据", flush=True)
        return 1
    train_dates, test_dates = _split_dates(kl, args.test_frac)
    print(f"[oos] 股票 {len(kl)} 只 | train {len(train_dates)} 日 → test {len(test_dates)} 日", flush=True)

    # 因子清单(与 ic_vectorized 主流程一致)
    import ic_vectorized as icv
    from quant_system.ic_factors import registry as _reg
    _reg.ensure_loaded()
    names = list(icv.ZOO_FACTOR_NAMES)

    # train 段 direction
    train_df = _ic_stats(kl, train_dates, names, icv._zoo_series_all, args.forward)
    if train_df.empty:
        print("[oos] train 段无有效 IC", flush=True)
        return 1
    train_df["direction"] = np.sign(train_df["ic_mean"]).astype(int)

    # test 段验证
    test_df = _ic_stats(kl, test_dates, names, icv._zoo_series_all, args.forward)
    if test_df.empty:
        print("[oos] test 段无有效 IC", flush=True)
        return 1

    merged = train_df[["factor", "ic_mean", "direction"]].merge(
        test_df[["factor", "ic_mean", "icir", "winrate"]], on="factor", suffixes=("_train", "_test"))
    merged["sign_keep"] = np.sign(merged["ic_mean_test"]) == merged["direction"]
    keep_rate = float(merged["sign_keep"].mean()) if len(merged) else 0.0
    flips = merged[~merged["sign_keep"]]["factor"].tolist()
    stable = merged[merged["sign_keep"]].sort_values("icir", ascending=False).head(10)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"n_stocks": len(kl), "forward": args.forward,
                   "train_days": len(train_dates), "test_days": len(test_dates),
                   "test_frac": args.test_frac, "seed": args.seed},
        "n_factors": int(len(merged)),
        "sign_keep_rate": round(keep_rate, 3),
        "flipped_factors": flips,
        # W2.4：持久化逐因子 in-sample(ic_mean_train) vs OOS(ic_mean_test)，
        # 供 oos_pool_rule 硬规则消费（原报告只有 stable_top + 翻转名单，无逐因子 OOS IC）。
        "factors": merged.to_dict(orient="records"),
        "stable_top": stable.to_dict(orient="records"),
        "test_ic_mean_avg": round(float(merged["ic_mean_test"].mean()), 4),
        "test_icir_avg": round(float(merged["icir"].mean()), 3),
    }
    jf = OUT_DIR / "IC_OOS_REPORT.json"
    jf.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    md = [f"# 因子方向样本外验证报告 ({datetime.now().strftime('%Y-%m-%d')})", "",
          f"- 股票 {len(kl)} 只(seed={args.seed}) | 前瞻 {args.forward} 日 | train {len(train_dates)} 日 → test {len(test_dates)} 日",
          f"- 因子数 {len(merged)} | **符号保持率 {keep_rate:.1%}** | 翻转 {len(flips)} 个",
          f"- test 段平均 IC {report['test_ic_mean_avg']} | 平均 ICIR {report['test_icir_avg']}", "",
          "## 方向翻转因子(需复核)", ""]
    md += [f"- {f}" for f in flips] or ["- (无)"]
    md += ["", "## 稳定 Top10(test 段 ICIR)", "",
           "| factor | train_ic | test_ic | test_icir | winrate |",
           "|---|---|---|---|---|"]
    for r in stable.to_dict(orient="records"):
        md.append("| {f} | {it:.4f} | {ie:.4f} | {ic:.2f} | {w:.2f} |".format(
            f=r['factor'], it=r['ic_mean_train'], ie=r['ic_mean_test'],
            ic=r['icir'], w=r['winrate']))
    mf = OUT_DIR / "IC_OOS_REPORT.md"
    mf.write_text("\n".join(md), encoding="utf-8")
    print(f"[oos] 完成: 符号保持率 {keep_rate:.1%} ({len(flips)} 翻转) → {mf.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
