#!/usr/bin/env python3
"""factor_dedup_report.py — P1-4 因子去冗余（共线性压缩）报告工具

审计_因子层.md P1-4：281 注册因子内多族高相关，缺相关聚类/压缩。本工具：
  1. 读因子面板 → 算两两截面 Spearman 相关矩阵；
  2. 层次聚类分族（可调 threshold）；
  3. 每族内按 ICIR 降序选代表因子；
  4. 族内可做 Gram-Schmidt 正交化（--orthogonalize 产出正交化面板供复合层选用）；
  5. 输出缩并报告到 data_cache/factor_health/dedup/（各族列出相关因子 + 保留代表）。

重要：这是**报告/分析工具**，不删注册表因子、不破坏下游——只产出缩并建议。

用法:
  python3 scripts/factor_dedup_report.py                       # 全量
  python3 scripts/factor_dedup_report.py --codes 000001,600000
  python3 scripts/factor_dedup_report.py --threshold 1.0 --orthogonalize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor_dedup_report")

_ROOT = Path(__file__).resolve().parent
WORKSPACE = _ROOT.parent


def load_panels(codes: list[str] | None, max_stocks: int = 0) -> dict[str, pd.DataFrame]:
    """构建因子面板（复用 ic_vectorized 面板构建，非 spill）。"""
    sys.path.insert(0, str(WORKSPACE))
    import scripts.ic_vectorized as icv

    kl = icv._load_from_warehouse(codes)
    if not kl:
        return {}
    if max_stocks:
        kl = {c: kl[c] for c in sorted(kl)[:max_stocks]}
    common, common_codes = icv._prepare_common(kl)
    if not common_codes:
        return {}
    panels: dict = {}
    panels.update(icv.build_tech_panels(kl, common=common, codes=common_codes,
                                        spill_dir=None, sample_dates=None, reuse_dir=None))
    panels.update(icv.build_zoo_panels(kl, common=common, codes=common_codes,
                                       spill_dir=None, sample_dates=None, reuse_dir=None))
    return panels


def load_icir(codes: list[str] | None) -> dict[str, float]:
    """读取已生成的 IC 报告 CSV（generated/ic_report/*.csv），用于 ICIR 选代表。"""
    from pathlib import Path as _P
    cands = sorted(_P(WORKSPACE / "generated" / "ic_report").glob("*.csv")) if \
        _P(WORKSPACE / "generated" / "ic_report").exists() else []
    out: dict[str, float] = {}
    for p in cands:
        try:
            df = pd.read_csv(p)
            if "factor" in df.columns and "icir" in df.columns:
                for _, r in df.iterrows():
                    if pd.notna(r["icir"]):
                        out.setdefault(str(r["factor"]), float(r["icir"]))
        except Exception as e:  # noqa: BLE001
            log.warning(f"IC 报告读取失败 {p}: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="因子去冗余（共线性压缩）报告")
    ap.add_argument("--codes", default="")
    ap.add_argument("--max-stocks", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="层次聚类距离阈值（1-|corr|），越小分族越细")
    ap.add_argument("--max-per-family", type=int, default=1,
                    help="每族保留代表因子数（默认1）")
    ap.add_argument("--orthogonalize", action="store_true",
                    help="额外产出族内 Gram-Schmidt 正交化面板")
    ap.add_argument("--out-dir", default=str(WORKSPACE / "data_cache" / "factor_health" / "dedup"))
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    panels = load_panels(codes, args.max_stocks)
    if not panels:
        log.error("无因子面板，去冗余报告未生成")
        return 2
    icir = load_icir(codes)

    from quant_system.ic_factors.dedup import dedup_report, decorrelate_families
    report = dedup_report(panels, icir=icir, threshold=args.threshold,
                          max_per_family=args.max_per_family)
    corr = report["corr"]
    families = report["families"]
    representatives = report["representatives"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 主报告 Markdown
    lines = ["# 因子去冗余报告（P1-4 共线性压缩）", "",
             f"- 因子数: {len(panels)} | 族数: {len(families)} "
             f"| 距离阈值: {args.threshold} (1-|corr|)",
             f"- 至少 2 因子族: {sum(1 for f in families.values() if len(f) > 1)} 个 "
             f"（压缩前因子覆盖 {len(panels)}，代表性因子 {len(representatives)}）",
             "", "## 代表性因子（每族择 1 或多只）", "", "| 代表因子 | 族内因子(同族高相关) | 族大小 | ICIR |", "|---|---|---|---|"]
    for rep in representatives:
        fam_list = next((fv for fv in families.values() if rep in fv), [rep])
        icir_v = f"{icir.get(rep, np.nan):.2f}" if rep in icir else "-"
        lines.append(f"| {rep} | {', '.join(fam_list)} | {len(fam_list)} | {icir_v} |")
    lines += ["", "## 相关性最高的因子对（max 15）", "", "| 因子A | 因子B | |r| |", "|---|---|---|"]
    pairs = []
    for a in corr.index:
        for b in corr.index:
            if a < b:
                v = corr.loc[a, b]
                if pd.notna(v):
                    pairs.append((abs(v), a, b, v))
    for _, a, b, v in sorted(pairs, reverse=True)[:15]:
        lines.append(f"| {a} | {b} | {v:.3f} |")
    report_path = out_dir / "dedup_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    corr.to_parquet(out_dir / "factor_corr.parquet")
    # 结构明细 CSV
    rows = []
    for fid, members in sorted(families.items()):
        rep = next((m for m in members if m in representatives), None)
        rows.append({"family_id": fid, "size": len(members),
                     "representative": rep, "members": ";".join(members)})
    pd.DataFrame(rows).to_csv(out_dir / "families.csv", index=False, encoding="utf-8-sig")

    # 正交化面板（--orthogonalize）供复合层选用
    orth_path = None
    if args.orthogonalize:
        # prefer 顺序：代表因子优先（按 ICIR 全体降序已 implicit），
        # 族内 decorrelate_families 内部已按 ICIR 排 order
        orth = decorrelate_families(panels, families, icir=icir)
        if orth:
            orth_dir = out_dir / "orth"
            orth_dir.mkdir(parents=True, exist_ok=True)
            for k, v in orth.items():
                v.to_parquet(orth_dir / f"{k}.parquet")
            orth_path = orth_dir
        else:
            log.warning("正交化面板为空（无多因子族）")

    print("\n=== 因子去冗余报告（P1-4）===")
    summary = report["summary"]
    print(f"因子 {len(panels)} 个 → 分成 {len(families)} 族，代表性因子 {len(representatives)} 个")
    for s in sorted(summary, key=lambda x: -x["size"]):
        if s["size"] > 1:
            print(f"  族{s['family_id']}: 大小{s['size']} 代表={s['representative']}")
    print(f"\n报告: {report_path}")
    print(f"相关矩阵: {out_dir}/factor_corr.parquet | 族结构: {out_dir}/families.csv")
    if orth_path:
        print(f"正交化面板: {orth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
