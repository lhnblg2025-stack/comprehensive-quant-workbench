"""
量化交易系统 — 自选股看板模块 (409只大中盘主板股票)

提供：
  - get_watchlist()          → 全量409只股票基础信息
  - fetch_quotes()           → 拉取实时行情（腾讯接口，批量60只/请求）
  - filter_stocks(criteria)  → 按条件过滤（PE/PB/涨跌幅/位置等）
  - group_by_tier()          → 按市值档次分类汇总
  - group_by_sector()        → 按行业关键词近似分组
  - summary()                → 完整摘要（涨跌/分档/行业/成交额/价值洼地/超跌/新高）
  - format_report()          → 完整日报格式输出

数据源：Tencent qt.gtimg.cn (批量60只/请求, 5~8秒扫完409只)
"""

from __future__ import annotations

import json
import time as _time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# ── paths ──
ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT.parent / "config"
DATA_DIR = ROOT.parent / "quant_system"

# ── constants ──
TENCENT_HEADERS = {"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"}
CST = timezone(timedelta(hours=8))

TIER_NAMES = [
    "超大盘(5000亿+)",
    "大盘A(2000-5000亿)",
    "大盘B(1000-2000亿)",
    "中大盘A(500-1000亿)",
    "中大盘B(300-500亿)",
]

_WATCHLIST_CACHE: list[dict] | None = None
_QUOTE_CACHE: list[dict] | None = None
_QUOTE_CACHE_TIME: float = 0
_QUOTE_CACHE_TTL: float = 60  # seconds


# ════════════════════════════════════════════════════════════════
# 1. 加载自选股基础信息
# ════════════════════════════════════════════════════════════════

def get_watchlist(refresh: bool = False) -> list[dict]:
    """
    返回全量409只大中盘股票基础信息列表。
    每个元素: {symbol, name, market_cap_yi, pe_ttm, pb}
    """
    global _WATCHLIST_CACHE
    if _WATCHLIST_CACHE is not None and not refresh:
        return _WATCHLIST_CACHE

    pool_path = DATA_DIR / "large_cap_stocks.json"
    if pool_path.exists():
        with open(pool_path, encoding="utf-8") as f:
            data = json.load(f)
        _WATCHLIST_CACHE = [{
            "symbol": s["symbol"],
            "name": s["name"],
            "market_cap_yi": s.get("market_cap_yi", 0),
            "pe_ttm": s.get("pe_ttm", 0),
            "pb": s.get("pb", 0),
        } for s in data]
    else:
        # Fallback: load from config watchlist
        config_path = CONFIG_DIR / "watchlist_personal.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                wl = json.load(f)
            _WATCHLIST_CACHE = wl.get("_flat", [])
        else:
            _WATCHLIST_CACHE = []

    return _WATCHLIST_CACHE


def count_watchlist() -> dict:
    """返回自选股统计数据"""
    stocks = get_watchlist()
    tiers: dict[str, int] = Counter()
    total_mv = 0
    for s in stocks:
        tier = _cap_tier(s.get("market_cap_yi", 0))
        tiers[tier] += 1
        total_mv += s.get("market_cap_yi", 0)
    return {
        "total": len(stocks),
        "total_market_cap_yi": round(total_mv, 0),
        "tiers": dict(tiers),
    }


# ════════════════════════════════════════════════════════════════
# 2. 拉取实时行情
# ════════════════════════════════════════════════════════════════

def fetch_quotes(symbols: list[str] | None = None, force: bool = False) -> list[dict]:
    """
    从腾讯批量拉取股票实时行情。

    Args:
        symbols: 股票列表，None=拉取全量409只
        force: 是否强制刷新缓存

    Returns:
        每只股票: {symbol, name, price, change_pct, market_cap_yi,
                   pe_ttm, pb, volume, turnover, high_52w, low_52w, ...}
    """
    global _QUOTE_CACHE, _QUOTE_CACHE_TIME

    if symbols is None:
        # Use cached quotes if fresh enough
        if not force and _QUOTE_CACHE is not None and _time.time() - _QUOTE_CACHE_TIME < _QUOTE_CACHE_TTL:
            return _QUOTE_CACHE
        stocks = get_watchlist()
        symbols_to_fetch = [s["symbol"] for s in stocks]
    else:
        symbols_to_fetch = symbols

    result: list[dict] = []
    chunks = [symbols_to_fetch[i:i+60] for i in range(0, len(symbols_to_fetch), 60)]

    for chunk in chunks:
        codes = ",".join(
            # V11 审计修复（Medium）: 北交所（4/8/920 开头）原被拼 sz → 腾讯查不到静默缺失。
            # 修正: 北交所用 bj 前缀（腾讯支持 bj8xxxxx）。
            ("bj" + s) if s.startswith(("4", "8", "920")) else
            (f"sh{s}" if s.startswith("6") else f"sz{s}")
            for s in chunk
        )
        # P2-Q26-fix: 行情接口默认走 https（加密传输），失败兜底 http（明文）并各带超时；
        # 原实现仅 http 且单点失败直接丢弃整个批次（无降级/重试）。
        resp = None
        for _url in (f"https://qt.gtimg.cn/q={codes}",
                     f"http://qt.gtimg.cn/q={codes}"):
            try:
                resp = requests.get(_url, headers=TENCENT_HEADERS, timeout=10)
                break
            except requests.RequestException:
                continue
        if resp is None:
            continue
        r = resp
        try:
            for line in r.text.strip().split(";\n"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                parts = line.split("~")
                if len(parts) < 50:
                    continue
                try:
                    result.append({
                        "symbol": parts[2],
                        "name": parts[1],
                        "price": float(parts[3]),
                        "prev_close": float(parts[4]),
                        "open": float(parts[5]),
                        "high": float(parts[33]) if parts[33] else float(parts[4]),
                        "low": float(parts[34]) if parts[34] else float(parts[4]),
                        # P1-Q26-fix: parts[6] 为成交量(手)，×100 转为股，
                        # 保证 summary() 的 most_active_10(volume*price=成交额) 及量比计算口径一致。
                        "volume": float(parts[6]) * 100 if parts[6] else 0,
                        "change": float(parts[31]) if parts[31] else 0,
                        "change_pct": float(parts[32]) if parts[32] else 0,
                        # P1-Q26-fix: 换手率字段错位——parts[49] 为量比，真实换手率在 parts[38]，
                        # 直接进入 opportunity.py 选股过滤与 intraday_monitor.py 预警，必须纠正。
                        "turnover": float(parts[38]) if len(parts) > 38 and parts[38] else 0,
                        # P2-Q26-fix: 振幅字段错位——parts[70] 实测为量比类指标(13.07)，
                        # 真实振幅在 parts[43]（(高-低)/昨收，实测2.20），供 intraday_monitor 等预警使用。
                        "amplitude": float(parts[43]) if len(parts) > 43 and parts[43] else 0,
                        "pe_ttm": float(parts[39]) if parts[39] else 0,
                        "pb": float(parts[46]) if parts[46] else 0,
                        # P2-Q26-fix: 市值口径统一为总市值 parts[45]（原 parts[44] 为流通市值）。
                        # 分档(_cap_tier)/筛选/日报语义均为总市值；全流通股两者相等，部分流通股按总市值分档更准确。
                        "market_cap_yi": float(parts[45]) if len(parts) > 45 and parts[45] else 0,
                        # V11 审计修复（High, 实网实证）: parts[47]/[48] 是当日涨停/跌停价
                        #（茅台 1308.55×1.1=1439.41 / ×0.9=1177.70 精确吻合），
                        # 原实现误作 52 周高低 → oversold/value_picks 伪信号刷屏。
                        # 修正: 52 周高低在 parts[67]/[68]（茅台 1539.98/1151.01 与真实区间吻合）。
                        "high_52w": float(parts[67]) if len(parts) > 67 and parts[67] else 0,
                        "low_52w": float(parts[68]) if len(parts) > 68 and parts[68] else 0,
                        # P1-Q26-fix: parts[37] 为成交额(万元)，×10000 转为元。
                        "amount": float(parts[37]) * 10000 if len(parts) > 37 and parts[37] else 0,
                    })
                except (ValueError, IndexError):
                    pass
        except requests.RequestException:
            pass

    if symbols is None:
        _QUOTE_CACHE = result
        _QUOTE_CACHE_TIME = _time.time()

    return result


# ════════════════════════════════════════════════════════════════
# 3. 分类与过滤
# ════════════════════════════════════════════════════════════════

def _cap_tier(mv: float) -> str:
    if mv >= 5000:
        return "超大盘(5000亿+)"
    elif mv >= 2000:
        return "大盘A(2000-5000亿)"
    elif mv >= 1000:
        return "大盘B(1000-2000亿)"
    elif mv >= 500:
        return "中大盘A(500-1000亿)"
    else:
        return "中大盘B(300-500亿)"


def _52w_position(q: dict) -> tuple[float, str]:
    """返回52周分位(%)和标记emoji"""
    high52 = q.get("high_52w", 0)
    low52 = q.get("low_52w", 0)
    price = q.get("price", 0)
    if high52 > low52 > 0:
        pos = (price - low52) / (high52 - low52) * 100
        pos = max(0, min(100, pos))
        if pos > 95:
            mark = "🔝"
        elif pos < 10:
            mark = "🔻"
        elif pos < 30:
            mark = "📉"
        elif pos > 75:
            mark = "📈"
        else:
            mark = "➖"
        return round(pos, 1), mark
    return 50, "➖"


def filter_stocks(
    quotes: list[dict] | None = None,
    *,
    min_cap: float = 0,
    max_cap: float = float("inf"),
    max_pe: float = float("inf"),
    min_pe: float = 0,
    max_pb: float = float("inf"),
    min_pb: float = 0,
    min_change: float = float("-inf"),
    max_change: float = float("inf"),
    near_52w_low_pct: float = 100,
    near_52w_high_pct: float = 100,
    min_turnover: float = 0,
    keyword: str = "",
    tier: str = "",
    top_n: int = 0,
    sort_by: str = "change_pct",
    ascending: bool = False,
) -> list[dict]:
    """
    按条件过滤自选股。

    Args:
        quotes: 行情数据列表（None=自动拉取最新）
        near_52w_low_pct: 距52周低点百分比阈值 (如 5 = 距低点+5%以内)
        near_52w_high_pct: 距52周高点百分比阈值
        keyword: 名称包含关键字
        tier: 市值档位名称
        top_n: 返回前N条
        sort_by: 排序字段
        ascending: 升序

    Returns:
        过滤后的行情列表
    """
    if quotes is None:
        quotes = fetch_quotes()

    valid = [q for q in quotes if q.get("price", 0) > 0]

    filtered = []
    for q in valid:
        mv = q.get("market_cap_yi", 0)
        pe = q.get("pe_ttm", 0)
        pb = q.get("pb", 0)
        price = q.get("price", 0)
        chg = q.get("change_pct", 0)
        high52 = q.get("high_52w", 0)
        low52 = q.get("low_52w", 0)
        turnover = q.get("turnover", 0)

        # Market cap filter
        if mv < min_cap or mv > max_cap:
            continue
        # PE filter (skip if PE=0/negative unless min_pe allows)
        if pe <= 0 and min_pe > 0:
            continue
        if pe < min_pe or pe > max_pe:
            continue
        # PB filter
        if pb < min_pb or pb > max_pb:
            continue
        # Change filter
        if chg < min_change or chg > max_change:
            continue
        # Turnover filter
        if turnover < min_turnover:
            continue
        # 52w position filters
        if low52 > 0 and price > 0:
            pct_from_low = (price / low52 - 1) * 100
            if pct_from_low > near_52w_low_pct:
                continue
        if high52 > 0 and price > 0:
            pct_from_high = (price / high52 - 1) * 100
            if abs(pct_from_high) > near_52w_high_pct:
                continue
        # Tier filter
        if tier and tier not in _cap_tier(mv):
            continue
        # Keyword filter
        if keyword and keyword not in q.get("name", "") and keyword not in q.get("symbol", ""):
            continue

        filtered.append(q)

    # Sort
    reverse = not ascending
    filtered.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

    if top_n > 0:
        filtered = filtered[:top_n]

    return filtered


# ════════════════════════════════════════════════════════════════
# 4. 按分类汇总
# ════════════════════════════════════════════════════════════════

def group_by_tier(quotes: list[dict] | None = None) -> dict[str, dict]:
    """
    按市值档次分类汇总。

    Returns:
        {tier_name: {"quotes": [...], "up": N, "down": N,
                     "avg_change": x, "best": {}, "worst": {}}}
    """
    if quotes is None:
        quotes = fetch_quotes()

    valid = [q for q in quotes if q.get("price", 0) > 0]
    groups: dict[str, dict] = {}

    for tier_name in TIER_NAMES:
        tier_q = [q for q in valid if _cap_tier(q["market_cap_yi"]) == tier_name]
        if not tier_q:
            continue

        up = sum(1 for q in tier_q if q["change_pct"] > 0)
        down = sum(1 for q in tier_q if q["change_pct"] < 0)
        avg_chg = sum(q["change_pct"] for q in tier_q) / len(tier_q) if tier_q else 0
        best = max(tier_q, key=lambda q: q["change_pct"]) if tier_q else {}
        worst = min(tier_q, key=lambda q: q["change_pct"]) if tier_q else {}

        groups[tier_name] = {
            "count": len(tier_q),
            "up": up,
            "down": down,
            "flat": len(tier_q) - up - down,
            "avg_change": round(avg_chg, 2),
            "best": best,
            "worst": worst,
            "quotes": tier_q,
        }

    return groups


def group_by_sector(quotes: list[dict] | None = None) -> list[dict]:
    """
    按行业关键词分组（近似行业分类）。
    注意：这是基于名称关键字的粗略分类，非证监会行业代码。
    """
    if quotes is None:
        quotes = fetch_quotes()

    keywords = {
        "银行": ["银行", "农行", "工行", "中行", "招行", "兴业", "浦发",
                 "民生", "光大", "平安银行", "中信银行", "华夏", "邮储",
                 "宁波", "南京", "北京", "上海", "江苏", "杭州", "成都"],
        "券商": ["证券", "券商", "中信建投", "东方财富", "指南针"],
        "保险": ["保险", "人寿", "新华保险"],
        "白酒": ["茅台", "五粮液", "泸州", "古井", "今世缘", "汾酒",
                 "洋河", "舍得", "酒鬼", "水井坊"],
        "半导体/电子": ["半导体", "芯片", "华创", "长电", "兆易", "中芯",
                      "沪电", "深南", "东山精密", "鹏鼎", "生益",
                      "立讯", "歌尔", "蓝思", "京东方"],
        "医药": ["医药", "药", "医疗", "生物", "健康", "恒瑞", "迈瑞",
                "同仁堂", "片仔癀", "白云山"],
        "新能源": ["能源", "光伏", "风电", "锂", "电池", "新能源",
                  "比亚迪", "隆基", "通威", "天合", "阳光"],
        "地产": ["地产", "万科", "保利", "招商蛇口", "华侨城", "绿地",
                "华发", "滨江"],
        "食品饮料": ["食品", "饮料", "乳业", "伊利", "东鹏", "双汇",
                    "海天", "安井", "涪陵"],
        "有色金属": ["有色", "矿业", "黄金", "铜", "铝", "钢铁", "神华",
                   "紫金", "江铜", "山东黄金", "洛阳钼业"],
        "汽车": ["汽车", "比亚迪", "上汽", "长安", "长城", "江淮",
                "广汽", "一汽", "赛力斯"],
        "基建/建筑": ["建筑", "中铁", "中交", "中建", "中铁建", "电建",
                     "能建", "交建", "中国交建"],
        "家电": ["家电", "美的", "海尔", "格力", "TCL", "海信", "科沃斯"],
    }

    valid = [q for q in quotes if q.get("price", 0) > 0]
    sectors: dict[str, list[dict]] = defaultdict(list)

    for q in valid:
        name = q.get("name", "")
        matched = False
        for sector, kws in keywords.items():
            if any(kw in name for kw in kws):
                sectors[sector].append(q)
                matched = True
                break
        if not matched:
            # Try to broadly classify by symbol prefix or other data
            sectors["其他"].append(q)

    result = []
    for sector_name, sq in sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True):
        up = sum(1 for q in sq if q["change_pct"] > 0)
        down = sum(1 for q in sq if q["change_pct"] < 0)
        avg = sum(q["change_pct"] for q in sq) / len(sq) if sq else 0
        result.append({
            "sector": sector_name,
            "count": len(sq),
            "up": up,
            "down": down,
            "avg_change": round(avg, 2),
            "quotes": sorted(sq, key=lambda q: abs(q["change_pct"]), reverse=True)[:5],
        })

    return result


def summary(quotes: list[dict] | None = None) -> dict[str, Any]:
    """
    返回自选股完整摘要。

    Returns:
        {
            "timestamp": "2026-07-28 15:30",
            "session": "盘后",
            "total": 409,
            "up": N, "down": N, "flat": N,
            "avg_change": x, "median_change": x,
            "tiers": {...},  # group_by_tier 结果
            "sectors": [...],  # group_by_sector 结果
            "best_10": [...], # 涨幅前10
            "worst_10": [...], # 跌幅前10
            "most_active_10": [...], # 成交额前10
            "value_picks": [...], # 价值洼地
            "oversold": [...], # 超跌
            "momentum": [...], # 新高
        }
    """
    if quotes is None:
        quotes = fetch_quotes()

    valid = [q for q in quotes if q.get("price", 0) > 0]

    now = datetime.now(CST)
    h = now.hour
    if 9 <= h < 11 or 13 <= h < 15:
        session = "盘中"
    elif h < 9:
        session = "盘前"
    else:
        session = "盘后"

    up = sum(1 for q in valid if q["change_pct"] > 0)
    down = sum(1 for q in valid if q["change_pct"] < 0)
    flat_n = len(valid) - up - down

    changes = sorted(q["change_pct"] for q in valid)
    avg_chg = sum(changes) / len(changes) if changes else 0
    med_chg = changes[len(changes)//2] if changes else 0

    sorted_up = sorted(valid, key=lambda q: q["change_pct"], reverse=True)
    sorted_down = sorted(valid, key=lambda q: q["change_pct"])

    # Value picks
    value = [q for q in valid if 0 < q.get("pe_ttm", 100) < 15
             and 0 < q.get("pb", 10) < 1.5
             and q.get("price", 0) < q.get("high_52w", 0) * 0.85
             and q.get("price", 0) > 3]
    value.sort(key=lambda q: q["pe_ttm"])

    # Oversold
    oversold = [q for q in valid if q.get("price", 0) > 2
                and q.get("low_52w", 0) > 0
                and q["price"] / q["low_52w"] < 1.05]
    oversold.sort(key=lambda q: q["price"] / q["low_52w"])

    # Momentum (near 52w high + positive)
    momentum = [q for q in valid if q.get("high_52w", 0) > 0
                and q["price"] >= q["high_52w"] * 0.97
                and q["change_pct"] > 0
                and q.get("price", 0) > 5]
    momentum.sort(key=lambda q: q["change_pct"], reverse=True)

    # Most active by turnover
    by_volume = sorted(valid, key=lambda q: q.get("volume", 0) * q.get("price", 0), reverse=True)

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M"),
        "session": session,
        "total": len(valid),
        "up": up,
        "down": down,
        "flat": flat_n,
        "avg_change": round(avg_chg, 2),
        "median_change": round(med_chg, 2),
        "tiers": group_by_tier(valid),
        "sectors": group_by_sector(valid),
        "best_10": sorted_up[:10],
        "worst_10": sorted_down[:10],
        "most_active_10": by_volume[:10],
        "value_picks": value[:10],
        "oversold": oversold[:15],
        "momentum": momentum[:10],
    }


# ════════════════════════════════════════════════════════════════
# 5. 格式化输出
# ════════════════════════════════════════════════════════════════

def format_report(summary_dict: dict | None = None, brief: bool = False) -> str:
    """
    生成自选股日报文本（可直接投递飞书/微信）。

    Args:
        summary_dict: summary() 返回的字典，None=自动生成
        brief: 简要版（仅概述+极端值）

    Returns:
        格式化文本
    """
    if summary_dict is None:
        summary_dict = summary()

    lines = []
    s = summary_dict

    # Header
    hdr = f"📊 **自选股看板** ({s['total']}只主板≥300亿)"
    if s["session"]:
        hdr += f" {s['timestamp']} {'🟢盘中' if s['session']=='盘中' else '🔴盘后' if s['session']=='盘后' else '🌙盘前'}"
    lines.append(hdr)
    lines.append(f"📈 🟢{s['up']}涨 🔴{s['down']}跌 ⚪{s['flat']}平 | 平均{s['avg_change']:+.2f}% 中位数{s['median_change']:+.2f}%")
    lines.append("")

    if brief:
        # Brief mode: just best/worst + tier overview
        lines.append("**分档概览**")
        for tier_name in TIER_NAMES:
            tier = s["tiers"].get(tier_name)
            if not tier:
                continue
            lines.append(f"  {tier_name}: {tier['count']}只 🟢{tier['up']} 🔴{tier['down']} 平均{tier['avg_change']:+.2f}%")
        lines.append("")
        if s["best_10"]:
            q = s["best_10"][0]
            lines.append(f"🏆 最佳: {q['name']}({q['symbol']}) {q['change_pct']:+.2f}%")
        if s["worst_10"]:
            q = s["worst_10"][0]
            lines.append(f"📉 最差: {q['name']}({q['symbol']}) {q['change_pct']:+.2f}%")
        return "\n".join(lines)

    # === Full report ===

    # Sector overview
    lines.append("**行业风向**")
    for sg in s["sectors"]:
        arrow = "🟢" if sg["avg_change"] > 0 else "🔴"
        lines.append(f"  {arrow} {sg['sector']}: {sg['count']}只 | 平均{sg['avg_change']:+.2f}% | 🟢{sg['up']} 🔴{sg['down']}")
    lines.append("")

    # Tier overview
    lines.append("**分档概览**")
    for tier_name in TIER_NAMES:
        tier = s["tiers"].get(tier_name)
        if not tier:
            continue
        best_q = tier["best"]
        worst_q = tier["worst"]
        lines.append(
            f"  {tier_name}: {tier['count']}只 🟢{tier['up']} 🔴{tier['down']} 平均{tier['avg_change']:+.2f}%"
            f"  ↑{best_q.get('name','')}+{best_q.get('change_pct',0):+.2f}%"
            f"  ↓{worst_q.get('name','')}{worst_q.get('change_pct',0):+.2f}%"
        )
    lines.append("")

    # Top gainers
    lines.append(f"**🏆 涨幅TOP8** ({s['timestamp']})")
    for q in s["best_10"][:8]:
        pos, mark = _52w_position(q)
        name = q["name"]
        lines.append(
            f"  {q['symbol']} {name:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
            f"  {mark} | {q['market_cap_yi']:>5.0f}亿 PE={q['pe_ttm']:.1f}"
        )
    lines.append("")

    # Worst decliners
    lines.append(f"**📉 跌幅TOP8**")
    for q in s["worst_10"][:8]:
        pos, mark = _52w_position(q)
        name = q["name"]
        lines.append(
            f"  {q['symbol']} {name:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
            f"  {mark} | {q['market_cap_yi']:>5.0f}亿 PE={q['pe_ttm']:.1f}"
        )
    lines.append("")

    # Value picks
    if s["value_picks"]:
        lines.append(f"**💰 价值洼地** (PE<15, PB<1.5, 距高点-15%+, {len(s['value_picks'])}只显示)")
        for q in s["value_picks"]:
            pct_from_high = (q["price"] / q["high_52w"] - 1) * 100 if q.get("high_52w") else 0
            lines.append(
                f"  {q['symbol']} {q['name']:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                f"  PE={q['pe_ttm']:.1f} PB={q['pb']:.2f} 距高点{pct_from_high:.0f}%"
            )
        lines.append("")

    # Oversold
    if s["oversold"]:
        lines.append(f"**🔻 接近52周低点** ({len(s['oversold'])}只)")
        for q in s["oversold"][:8]:
            pct_from_low = (q["price"] / q["low_52w"] - 1) * 100 if q.get("low_52w") else 0
            lines.append(
                f"  {q['symbol']} {q['name']:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                f"  距低点+{pct_from_low:.1f}% PE={q['pe_ttm']:.1f} PB={q['pb']:.2f}"
            )
        lines.append("")

    # Momentum
    if s["momentum"]:
        lines.append(f"**🔥 接近52周高点** ({len(s['momentum'])}只)")
        for q in s["momentum"]:
            lines.append(
                f"  {q['symbol']} {q['name']:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                f"  PE={q['pe_ttm']:.1f} PB={q['pb']:.2f} 换手{q['turnover']:.1f}%"
            )

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 6. CLI 入口
# ════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="自选股看板 (量化交易系统模块)")
    parser.add_argument("--brief", action="store_true", help="简要版输出")
    parser.add_argument("--deep", type=str, help="深度分析某只股票")
    parser.add_argument("--tier", type=str, help="按市值档查看")
    parser.add_argument("--keyword", type=str, help="按名称关键字筛选")
    parser.add_argument("--value", action="store_true", help="价值洼地筛选")
    parser.add_argument("--oversold", action="store_true", help="超跌股票")
    parser.add_argument("--momentum", action="store_true", help="动量股票")
    parser.add_argument("--top", type=int, default=0, help="只看涨幅前N")
    args = parser.parse_args()

    if args.deep:
        from quant_system.intraday_decision import decide_stock, format_decision
        result = decide_stock(args.deep)
        print(format_decision(result))
        return

    if args.tier:
        quotes = fetch_quotes()
        tier_q = [q for q in quotes if args.tier in _cap_tier(q.get("market_cap_yi", 0))]
        print(f"\n📊 {args.tier} ({len(tier_q)}只)")
        for q in sorted(tier_q, key=lambda x: x["market_cap_yi"], reverse=True):
            pos, mark = _52w_position(q)
            print(f"  {q['symbol']} {q['name'][:8]:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                  f"  {mark} {pos}% | {q['market_cap_yi']:>5.0f}亿 PE={q['pe_ttm']:.1f} PB={q['pb']:.2f}")
        return

    if args.keyword:
        quotes = fetch_quotes()
        kw_q = [q for q in quotes if args.keyword in q.get("name", "")]
        print(f"\n📂 「{args.keyword}」({len(kw_q)}只)")
        for q in sorted(kw_q, key=lambda x: x["market_cap_yi"], reverse=True):
            pos, mark = _52w_position(q)
            print(f"  {q['symbol']} {q['name'][:8]:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                  f"  {mark} {pos}% | {q['market_cap_yi']:>5.0f}亿 PE={q['pe_ttm']:.1f} PB={q['pb']:.2f}")
        return

    if args.value:
        quotes = fetch_quotes()
        value = [q for q in quotes if 0 < q.get("pe_ttm", 100) < 15
                 and 0 < q.get("pb", 10) < 1.5
                 and q.get("price", 0) > 3]
        value.sort(key=lambda q: q["pe_ttm"])
        print(f"\n💰 价值洼地 ({len(value)}只)")
        for q in value[:20]:
            print(f"  {q['symbol']} {q['name']:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                  f"  PE={q['pe_ttm']:.1f} PB={q['pb']:.2f} {q['market_cap_yi']:>5.0f}亿")
        return

    if args.oversold:
        quotes = fetch_quotes()
        oversold = [q for q in quotes if q.get("low_52w", 0) > 0
                    and q["price"] / q["low_52w"] < 1.05 and q.get("price", 0) > 2]
        oversold.sort(key=lambda q: q["price"] / q["low_52w"])
        print(f"\n🔻 接近52周低点 ({len(oversold)}只)")
        for q in oversold[:20]:
            pct = (q["price"] / q["low_52w"] - 1) * 100
            print(f"  {q['symbol']} {q['name']:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                  f"  距低点+{pct:.1f}% PE={q['pe_ttm']:.1f} PB={q['pb']:.2f}")
        return

    if args.momentum:
        quotes = fetch_quotes()
        momentum = [q for q in quotes if q.get("high_52w", 0) > 0
                    and q["price"] >= q["high_52w"] * 0.97
                    and q["change_pct"] > 0]
        momentum.sort(key=lambda q: q["change_pct"], reverse=True)
        print(f"\n🔥 接近新高 ({len(momentum)}只)")
        for q in momentum[:20]:
            print(f"  {q['symbol']} {q['name']:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                  f"  PE={q['pe_ttm']:.1f} PB={q['pb']:.2f} {q['market_cap_yi']:>5.0f}亿")
        return

    # Default: full report
    if args.top > 0:
        quotes = fetch_quotes()
        sorted_q = sorted(quotes, key=lambda q: q.get("change_pct", 0), reverse=True)
        print(f"\n🏆 涨幅TOP{args.top}")
        for q in sorted_q[:args.top]:
            pos, mark = _52w_position(q)
            print(f"  {q['symbol']} {q['name'][:8]:<8} ¥{q['price']:<8.2f} {q['change_pct']:>+7.2f}%"
                  f"  {mark} {pos}% | {q['market_cap_yi']:>5.0f}亿 PE={q['pe_ttm']:.1f}")
    else:
        print(format_report(brief=args.brief))


if __name__ == "__main__":
    main()
