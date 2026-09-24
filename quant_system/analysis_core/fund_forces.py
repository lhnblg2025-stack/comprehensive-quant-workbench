"""
fund_forces — 四路资金合力（V11 短线 M7）

四路: 游资(龙虎榜净买-机构) / 北向(股通席位) / 机构(机构专用席位) / 融资(两融余额)

合力指数 0-100:
  - 四路同向 → 90+；三路 → 70；2:2 → 分歧
  - 背离时标注"谁在主导"（历史统计: 游资 vs 北向背离时游资对次日影响更大）

数据（全部本地已有，滚动更新）:
  lhb_20*.parquet 龙虎榜个股明细 | lhb_jgmmtj_em.parquet 机构买卖
  lhb_hyyyb_em.parquet 席位日聚合(北向) | margin_detail_sh/*.parquet 两融个股明细

用法:
  python3 -m quant_system.analysis_core.fund_forces --build
  python3 -m quant_system.analysis_core.fund_forces --latest
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402

CST = timezone(timedelta(hours=8))
OUT = MARKET_DIR / "fund_forces.parquet"


def _load_lhb_detail() -> pd.DataFrame:
    files = sorted(MARKET_DIR.glob("lhb_20*.parquet"))
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    df = pd.concat(frames, ignore_index=True)
    df["上榜日"] = pd.to_datetime(df["上榜日"])
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


def build_forces(days: int = 90) -> pd.DataFrame:
    detail = _load_lhb_detail()
    # V12.3 审计 P2-7: jg/broker 任一文件缺失用 try/except 优雅降级(该路置空),
    # 不中断 --build(对比 chain_map/battle_map 的降级要求)。缺文件时 note 由调用方
    # 或后续"数据缺失"通路说明, 不再 throw 崩溃。
    try:
        jg = pd.read_parquet(MARKET_DIR / "lhb_jgmmtj_em.parquet")
        jg["上榜日期"] = pd.to_datetime(jg["上榜日期"])
        jg["代码"] = jg["代码"].astype(str).str.zfill(6)
    except Exception:  # noqa: BLE001 - 数据缺失降级
        logging.getLogger(__name__).warning("[fund_forces] lhb_jgmmtj_em 缺失, 机构侧置空")
        jg = pd.DataFrame(columns=["上榜日期", "代码", "机构买入净额"])
    try:
        broker = pd.read_parquet(MARKET_DIR / "lhb_hyyyb_em.parquet")
        broker["上榜日"] = pd.to_datetime(broker["上榜日"])
    except Exception:  # noqa: BLE001 - 数据缺失降级
        logging.getLogger(__name__).warning("[fund_forces] lhb_hyyyb_em 缺失, 营业部侧置空")
        broker = pd.DataFrame(columns=["上榜日", "营业部名称", "总买卖净额"])

    start = pd.Timestamp(datetime.now(CST).date()) - pd.Timedelta(days=days * 2 + 10)
    detail = detail[detail["上榜日"] >= start]
    jg = jg[jg["上榜日期"] >= start]
    broker = broker[broker["上榜日"] >= start]

    dates = sorted(detail["上榜日"].unique())
    rows = []
    for d in dates:
        dd = detail[detail["上榜日"] == d]
        jj = jg[jg["上榜日期"] == d]
        bb = broker[broker["上榜日"] == d]

        lhb_net = float(dd["龙虎榜净买额"].sum()) if len(dd) else 0.0
        jg_net = float(jj["机构买入净额"].sum()) if len(jj) else 0.0
        youzi_net = lhb_net - jg_net  # 游资 ≈ 龙虎榜总净买 - 机构净买

        north = bb[bb["营业部名称"].str.contains("股通专用", na=False)]
        north_net = float(north["总买卖净额"].sum()) if len(north) else 0.0

        rows.append({"date": d, "youzi_net": youzi_net, "jg_net": jg_net,
                     "north_net": north_net})

    df = pd.DataFrame(rows)

    # 融资余额（margin_detail_sh 按日文件，取最近天数）
    margin_rows = []
    files = sorted(MARKET_DIR.glob("margin_detail_sh/*.parquet"))
    for f in files[-days:]:
        try:
            m = pd.read_parquet(f)
            d = pd.Timestamp(f.stem)
            if d < start:
                continue
            col = [c for c in m.columns if "融资余额" in c]
            if col:
                margin_rows.append({"date": d, "margin_bal": float(m[col[0]].sum())})
        except Exception as e:
            logging.getLogger(__name__).error(f"[fund_forces] 操作失败: {e}", exc_info=True)
            continue
    if margin_rows:
        mg = pd.DataFrame(margin_rows).sort_values("date")
        mg["margin_delta"] = mg["margin_bal"].diff() / 1e8  # 亿元
        df = df.merge(mg[["date", "margin_delta"]], on="date", how="left")

    # 合力指数（近30日口径滚动，用当日符号与强度）
    def _sign(x):
        if pd.isna(x) or x == 0:
            return 0
        return 1 if x > 0 else -1

    idx_list: list[int] = []
    sign_list: list[str] = []
    nlegs_list: list[int] = []
    for _, r in df.iterrows():
        # 2026-08-21 审计: 腿缺失(NA/0)静默当中性会让合力分数失真且不可辨。
        # 记录实际参与腿数 n_legs 与缺失腿，缺失腿从符号判定中排除(仍记 sign=0)。
        raw = {"游资": r["youzi_net"], "机构": r["jg_net"], "北向": r["north_net"],
               "融资": r.get("margin_delta")}
        signs = [_sign(v) for v in raw.values()]
        n_legs = sum(1 for v in raw.values() if pd.notna(v) and v != 0)
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        agree = pos + neg
        if agree >= 4:
            idx = 95 if (pos == 4 or neg == 4) else 75
        elif agree == 3:
            idx = 70 if (pos == 3 or neg == 3) else 50
        elif agree == 2:
            idx = 55 if (pos == 2 or neg == 2) else 40
        else:
            idx = 30
        # 腿缺失惩罚: 参与腿 <3 时力度降档，避免"三腿同向"伪装成"四腿同向"
        if n_legs < 3:
            idx = int(idx * 0.85)
        # 强度修正: 以游资净买规模锚定
        scale = min(abs(r["youzi_net"]) / 5e8, 1.0)  # 5亿封顶
        idx = int(idx * (0.7 + 0.3 * scale))
        idx_list.append(idx)
        sign_list.append(json.dumps({"游资": signs[0], "机构": signs[1], "北向": signs[2],
                                     "融资": signs[3] if len(signs) > 3 else None}))
        nlegs_list.append(n_legs)
    df["force_index"] = idx_list
    df["signs"] = sign_list
    df["n_legs"] = nlegs_list

    df.to_parquet(OUT, index=False)
    print(f"[forces] {len(df)} 个交易日 → {OUT}")
    return df


def latest(as_of: str | None = None, n: int = 5) -> pd.DataFrame:
    if not OUT.exists():
        raise FileNotFoundError("先运行 --build")
    df = pd.read_parquet(OUT)
    if as_of:
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)]
    return df.tail(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="四路资金合力")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    if args.build:
        build_forces()
    if args.latest:
        df = latest()
        for _, r in df.iloc[::-1].iterrows():
            print(f"{r['date'].date()} 合力{r['force_index']} | 游资{r['youzi_net']/1e8:.2f}亿 "
                  f"机构{r['jg_net']/1e8:.2f}亿 北向{r['north_net']/1e8:.2f}亿 "
                  f"融资Δ{r.get('margin_delta', float('nan')):.1f}亿" if pd.notna(r.get('margin_delta')) else
                  f"{r['date'].date()} 合力{r['force_index']} | 游资{r['youzi_net']/1e8:.2f}亿 "
                  f"机构{r['jg_net']/1e8:.2f}亿 北向{r['north_net']/1e8:.2f}亿")
    if not (args.build or args.latest):
        ap.print_help()
