"""
quant_platform.commodity_link — 商品股-商品价格联动分析（V3 用户 #12/#13）

用户需求：商品股要看对应商品价格联动。
- 短线看龙虎榜；中长线看价值+期货；商品股必须看对应商品价格联动。
- 本模块维护"股票 → 关键商品"映射表，从数据仓库 commodity/futures/kline
  计算联动：涨跌同步率、近 N 日相关性、商品价格趋势对股价的指引。

数据源（全部仓库优先，零网络）：
- commodity: data_warehouse/commodity__*.parquet（东财商品数据，列含 收盘价/涨跌幅/日期）
- futures:   data_warehouse/market/futures_basis.parquet（期货基差）
- kline:     data_warehouse/kline/{code}.parquet（个股日线）

用法（openclaw 会话内）：
    from quant_platform.commodity_link import commodity_linkage, stock_commodities
    commodity_linkage("601899")   # 紫金矿业 → 黄金/铜联动
    stock_commodities("600028")   # 中国石化 → 原油
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── 股票 → 关键商品映射（可扩展；code 为 6 位）────────────────
# 商品名需要能在 commodity 数据集列名/数据中模糊匹配，或在 futures contract 中匹配
STOCK_COMMODITY_MAP: dict[str, dict] = {
    "601899": {"name": "紫金矿业", "commodities": ["黄金", "铜"], "note": "金铜双驱动，金占比更高"},
    "600547": {"name": "山东黄金", "commodities": ["黄金"], "note": "纯金矿标的"},
    "600988": {"name": "赤峰黄金", "commodities": ["黄金"], "note": "黄金成长股"},
    "601988": {"name": "中国神华", "commodities": ["动力煤"], "note": "煤炭龙头"},
    "601088": {"name": "中国神华", "commodities": ["动力煤"], "note": "煤炭+电力"},
    "600028": {"name": "中国石化", "commodities": ["原油"], "note": "炼化一体化"},
    "601857": {"name": "中国石油", "commodities": ["原油"], "note": "上游油气"},
    "600019": {"name": "宝钢股份", "commodities": ["螺纹钢", "铁矿石"], "note": "普钢龙头"},
    "000708": {"name": "中信特钢", "commodities": ["特钢"], "note": "特钢龙头"},
    "601600": {"name": "中国铝业", "commodities": ["铝"], "note": "电解铝龙头"},
    "000807": {"name": "云铝股份", "commodities": ["铝"], "note": "水电铝"},
    "600362": {"name": "江西铜业", "commodities": ["铜"], "note": "铜矿+冶炼"},
    "603993": {"name": "洛阳钼业", "commodities": ["铜", "钼", "钴"], "note": "铜钼钴多金属"},
    "002460": {"name": "赣锋锂业", "commodities": ["碳酸锂"], "note": "锂盐龙头"},
    "002466": {"name": "天齐锂业", "commodities": ["碳酸锂"], "note": "锂矿"},
    "601012": {"name": "隆基绿能", "commodities": ["硅料", "硅片"], "note": "光伏硅片"},
    "600438": {"name": "通威股份", "commodities": ["硅料"], "note": "多晶硅龙头"},
    "002714": {"name": "牧原股份", "commodities": ["生猪", "豆粕", "玉米", "大豆"], "note": "生猪产品价格 + 豆粕/玉米饲料成本；大豆为豆粕上游代理，避免重复解读"},
    "000895": {"name": "双汇发展", "commodities": ["生猪"], "note": "肉制品"},
    "600309": {"name": "万华化学", "commodities": ["MDI", "原油"], "note": "化工茅，MDI 全球龙头"},
    "601225": {"name": "陕西煤业", "commodities": ["动力煤"], "note": "动力煤"},
    "600900": {"name": "长江电力", "commodities": ["水电"], "note": "水电（弱商品属性）"},
    "601919": {"name": "中远海控", "commodities": ["运价", "BDI"], "note": "集运周期"},
    "600150": {"name": "中国船舶", "commodities": ["新船价格"], "note": "造船周期"},
    "600585": {"name": "海螺水泥", "commodities": ["水泥"], "note": "水泥龙头"},
    "601668": {"name": "中国建筑", "commodities": ["螺纹钢"], "note": "基建（弱联动）"},
}


def stock_commodities(code: str) -> dict | None:
    """查询某只股票的关联商品映射。"""
    code = str(code).strip().zfill(6)
    return STOCK_COMMODITY_MAP.get(code)


# 商品文件名标识（commodity__{tag}.parquet）→ 中文别名，用于匹配 STOCK_COMMODITY_MAP
_COMMODITY_TAG_ALIAS: dict[str, list[str]] = {
    "crude": ["原油"],
    "bdi": ["BDI", "运价"],
    "lh": ["生猪"],
    "gold": ["黄金"],
    "copper": ["铜"],
    "alu": ["铝"],
    "coal": ["动力煤", "煤炭"],
    "meal": ["豆粕"],
    "soybean": ["大豆", "黄大豆"],
    "corn": ["玉米"],
}


def _load_commodity_series(name: str) -> pd.DataFrame | None:
    """从 commodity 数据集加载某商品序列（先按文件名标识匹配，再按列名模糊匹配）。"""
    try:
        files = sorted((ROOT / "data_warehouse" / "market").glob("commodity__*.parquet")) or \
                sorted((ROOT / "data_warehouse").glob("commodity__*.parquet"))
        if not files:
            return None
        # 1) 文件名标识匹配：commodity__{tag}.parquet
        for f in files:
            tag = f.stem.replace("commodity__", "")
            aliases = _COMMODITY_TAG_ALIAS.get(tag, [])
            if any(name.lower() in a.lower() for a in aliases) or name.lower() == tag.lower():
                df = pd.read_parquet(f)
                if df.empty:
                    return None
                date_col = next((c for c in ("日期", "date") if c in df.columns), df.columns[0])
                out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce")})
                # 兼容中英文列名（中文: 收盘价/最新值；英文: close）
                close_col = next((c for c in ("收盘价", "最新值", "close") if c in df.columns), df.columns[1])
                out["price"] = pd.to_numeric(df[close_col], errors="coerce")
                return out.dropna().sort_values("date").reset_index(drop=True)
        # 2) 列名匹配（宽表格式：列名含商品名）
        df = pd.concat([pd.read_parquet(f) for f in files[-2:]], ignore_index=True)
        if df.empty:
            return None
        cols = df.columns.tolist()
        hit = [c for c in cols if name.lower() in str(c).lower()]
        if not hit:
            return None
        out = pd.DataFrame({"date": pd.to_datetime(df["日期"], errors="coerce")})
        close_col = next((c for c in hit if "收盘" in str(c) or "最新" in str(c)), hit[0])
        out["price"] = pd.to_numeric(df[close_col], errors="coerce")
        return out.dropna().sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _load_stock_kline(code: str) -> pd.DataFrame | None:
    try:
        p = ROOT / "data_warehouse" / "kline" / f"{code}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if df.empty:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def commodity_linkage(code: str, lookback: int = 120) -> dict:
    """计算个股与其关键商品的联动度。

    指标：
    - 涨跌同步率：同向（商品涨→股涨）的交易日占比
    - 近 N 日收益相关性（pearson）
    - 商品价格趋势（近20日涨跌幅）
    - 联动结论：强/中/弱（|corr| 阈值 0.3/0.15）
    """
    meta = stock_commodities(code)
    if not meta:
        return {"code": code, "error": f"未登记商品映射（可加入 STOCK_COMMODITY_MAP）",
                "registered": sorted({v["name"] for v in STOCK_COMMODITY_MAP.values()})}
    k = _load_stock_kline(code)
    if k is None or len(k) < 30:
        return {"code": code, "error": f"K线数据不足（{0 if k is None else len(k)} 行）"}

    results = []
    for cname in meta["commodities"]:
        cs = _load_commodity_series(cname)
        if cs is None or len(cs) < 30:
            results.append({"commodity": cname, "error": "商品数据不足（需先抓 commodity 数据）"})
            continue
        # 对齐日期（内连接）
        k_ret = k[["date", "close"]].copy()
        k_ret["stock_ret"] = k_ret["close"].pct_change()
        cs_ret = cs.copy()
        cs_ret["comm_ret"] = cs_ret["price"].pct_change()
        m = k_ret.merge(cs_ret, on="date", how="inner").dropna(subset=["stock_ret", "comm_ret"])
        if len(m) < 20:
            results.append({"commodity": cname, "error": f"对齐后仅 {len(m)} 行"})
            continue
        m = m.tail(lookback)
        corr = m["stock_ret"].corr(m["comm_ret"])
        synch = float(((m["stock_ret"] > 0) == (m["comm_ret"] > 0)).mean())
        comm_trend = float(cs_ret["comm_ret"].tail(20).sum() * 100)
        strength = "强" if abs(corr) >= 0.3 else ("中" if abs(corr) >= 0.15 else "弱")
        results.append({
            "commodity": cname,
            "corr": round(float(corr), 3) if pd.notna(corr) else None,
            "同步率": round(synch, 2),
            "商品近20日累计涨跌%": round(comm_trend, 2),
            "联动强度": strength,
            "样本交易日": int(len(m)),
        })
    return {
        "code": code,
        "name": meta["name"],
        "note": meta["note"],
        "commodities": results,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def commodity_market_overview() -> dict:
    """商品市场总览：主要商品最新价/趋势（供日报/周报引用）。"""
    out = {}
    for cname in ("黄金", "铜", "原油", "铝", "碳酸锂", "生猪", "螺纹钢", "动力煤", "硅料"):
        cs = _load_commodity_series(cname)
        if cs is None or cs.empty:
            continue
        last = cs.iloc[-1]
        trend20 = cs["price"].pct_change(20).iloc[-1] * 100 if len(cs) > 20 else None
        out[cname] = {
            "最新价": round(float(last["price"]), 2) if pd.notna(last["price"]) else None,
            "近20日涨跌%": round(float(trend20), 2) if trend20 is not None and pd.notna(trend20) else None,
            "最新日期": str(last["date"].date()),
        }
    return {"商品": out, "ts": datetime.now().strftime("%Y-%m-%d %H:%M")}


if __name__ == "__main__":
    import json
    print("=== 商品联动自检 ===")
    for code in ("601899", "002714", "600028", "601088"):
        r = commodity_linkage(code)
        print(json.dumps(r, ensure_ascii=False)[:260])
    print("\n=== 商品市场总览 ===")
    print(json.dumps(commodity_market_overview(), ensure_ascii=False)[:400])
