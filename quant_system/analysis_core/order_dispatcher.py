"""
order_dispatcher — 指令转化层（V11 决策 → 可执行订单）

1. 把作战地图攻击分组 → 结构化订单 {code, name, ratio, strategy, price_cond}
2. 组合相关性约束: 推荐股票两两日收益相关 >0.7 → 总仓位上限压缩，
   强制分散到弱相关方向（防一损俱损）
3. 归因链: 每张订单记录信号来源（共振评分/情绪/席位），供日后验证

相关性数据: data_warehouse/kline/*.parquet（本地已有）或回退日线缓存。

输出: generated/orders_{date}.json

用法:
  python3 -m quant_system.analysis_core.order_dispatcher --date 2026-08-10 [--capital 1000000]
"""

from __future__ import annotations

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
CORR_THRESHOLD = 0.7


def _kline_close(code: str, days: int = 60) -> pd.Series | None:
    """个股近 days 日收盘（本地 kline parquet，失败回退 None）。"""
    code = code.zfill(6)
    f = MARKET_DIR.parent / "kline" / f"{code}.parquet"
    if not f.exists():
        f = MARKET_DIR / "kline" / f"{code}.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
    else:
        df = pd.read_parquet(f)
    if "close" not in df.columns or "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days)
    return df.set_index("date")["close"].astype(float)


def _correlation_constraint(orders: list[dict]) -> dict:
    """相关性约束: 返回 {cap_note, final_ratio_scale}。"""
    codes = [o["code"] for o in orders if o.get("code")]
    if len(codes) < 2:
        return {"note": "订单<2，无相关性约束", "scale": 1.0}
    closes = {}
    for c in codes:
        s = _kline_close(c)
        if s is not None and len(s) > 20:
            closes[c] = s.pct_change().dropna()
    if len(closes) < 2:
        return {"note": "相关性数据不足", "scale": 1.0}
    df = pd.DataFrame(closes).dropna(how="all")
    if len(df) < 15:
        return {"note": "对齐样本不足", "scale": 1.0}
    corr = df.corr()
    high_pairs = []
    names = {o["code"]: o["name"] for o in orders}
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            if a in corr.index and b in corr.columns:
                c = corr.loc[a, b]
                if abs(c) > CORR_THRESHOLD:
                    high_pairs.append((names.get(a, a), names.get(b, b), round(float(c), 2)))
    if high_pairs:
        return {
            "note": f"高相关组合 {len(high_pairs)} 对 → 总仓位上限压缩30%: {high_pairs[:3]}",
            "scale": 0.7,
        }
    return {"note": "无高相关组合", "scale": 1.0}


def dispatch(date: str | None = None, capital: float = 1_000_000) -> dict:
    date = date or datetime.now(CST).date().isoformat()
    bm_path = ROOT / "generated" / f"battle_map_{date}.json"
    if not bm_path.exists():
        return {"date": date, "error": "无作战地图，先运行 battle_map"}
    bm = json.loads(bm_path.read_text(encoding="utf-8"))

    orders = []
    pos_hi = float(bm["position_range"].split("-")[1].rstrip("%")) / 100.0
    core_list = bm.get("attack_groups", {}).get("core", [])
    for g in core_list:
        ld = g.get("leader")
        if not ld:
            continue
        orders.append({
            "code": ld["code"], "name": ld["name"],
            "boards": ld.get("boards", 1),
            "score": g.get("score", 0),
            "ratio": 0.0,  # 去重归一后填
            "strategy": g.get("strategy", "趋势低吸"),
            "signal_source": f"共振{g.get('score', 0)}+{bm['emotion_stage']}+{bm['macro_veto']}",
            "price_cond": "竞价>3%不炸板" if "打板" in g.get("strategy", "") else "回踩不破均线",
            "stop": "-2.5%" if "震荡" in bm.get("regime", "") else ("-3.5%" if "高波" in bm.get("regime", "") else "-5%"),
        })

    # 按 code 去重: 保留最高共振（2026-08-10 审计: 原保留第一个且分母用去重前 n）
    by_code: dict[str, dict] = {}
    for o in orders:
        if o["code"] not in by_code or o["score"] > by_code[o["code"]]["score"]:
            by_code[o["code"]] = o
    orders = list(by_code.values())
    # 去重后按实际订单数等权归一（总仓位上限 = position_range 上限）
    if orders:
        n = len(orders)
        each = round(pos_hi / n, 4)
        for o in orders:
            o["ratio"] = each
            o["amount"] = round(capital * each)

    # 相关性约束
    corr = _correlation_constraint(orders)
    if corr["scale"] < 1.0:
        for o in orders:
            o["ratio"] = round(o["ratio"] * corr["scale"], 4)
            o["amount"] = int(o["amount"] * corr["scale"])

    res = {
        "date": date,
        "orders": orders,
        "correlation": corr,
        "total_exposure": round(sum(o["ratio"] for o in orders), 3),
        "note": "模拟盘执行前需人工确认；归因链已随单记录 signal_source",
    }
    out = ROOT / "generated" / f"orders_{date}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="指令转化层")
    ap.add_argument("--date", default=None)
    ap.add_argument("--capital", type=float, default=1_000_000)
    args = ap.parse_args()
    r = dispatch(args.date, args.capital)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
