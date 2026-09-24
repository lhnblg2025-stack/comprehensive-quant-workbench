"""
量化交易系统 — 基本面筛选。

数据源: 东方财富个股财务数据 + 腾讯PE/PB
覆盖: 营收增速/净利润增速/ROE/分红率/毛利率

用法:
  python3 -m quant_system.fundamental --symbol 600519
  python3 -m quant_system.fundamental --screener

D5收敛登记 (2026-08-11): 基本面域收敛（保守策略）——与 fundamental_analysis.py /
financial_data.py 的数值转换重复已由 D1(utils.safe_float) 收敛；本模块报表抓取
fetch_financials 为独立能力保留（THS摘要→东财push2→新浪兜底链）；PE/PB 与
financial_data(收盘价/每股TTM)、fundamental_analysis(市值/年报) 为同名异口径保留。
"""

from __future__ import annotations
import logging

import json
import sys
import time as _time
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from quant_system.utils import safe_float as _safe_float_impl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))
EASTMONEY_H = {"User-Agent": "Mozilla/5.0", "Referer": "https://emweb.securities.eastmoney.com/"}


def _safe_float(val: Any) -> Optional[float]:
    """Parse finance numeric fields without leaking raw strings into callers.

    D1 收敛: 转发 quant_system.utils.safe_float（default=None 保留原 None 语义，
    units 保留中文单位 亿/万 倍数解析）。
    """
    return _safe_float_impl(
        val,
        default=None,
        units={
            "亿元": 100000000.0,
            "亿": 100000000.0,
            "万元": 10000.0,
            "万": 10000.0,
        },
    )


# P2-Q3-fix(L385): 按板块前缀区分交易所 —— 北交所(4/8开头 + 新代码920开头)此前被
# 当作 sz，沪市 6/5/9 为 sh（900 为沪B），深市 0/2/3 为 sz。
def tencent_prefix(symbol: str) -> str:
    """交易所前缀（腾讯/新浪风格：bj/sh/sz）。

    D5收敛登记: 独立能力保留（含北交所 920 修正，无等价实现）。
    """
    if symbol.startswith(("4", "8")) or symbol.startswith("920"):
        return f"bj{symbol}"
    if symbol.startswith(("6", "5", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _em_secid(symbol: str) -> str:
    """东财 push2 secid 前缀：深市/北交所 → 0.xxx，沪市 → 1.xxx。

    D5收敛: 内部真实重复 —— fetch_financials 兜底与 _backfill_market_cap 两处
    相同内联表达式收敛为唯一真源（行为不变）。
    B1实测: 东财 push2 北交所 secid 用 0. 前缀（0.920001 纬达光电 / 0.832566
    梓橦宫 正确，1. 前缀取错值），故 4/8/92 开头北交所代码归 0.；6/9/688
    沪市归 1.。
    """
    if symbol.startswith(("0", "3", "4", "8")) or symbol.startswith("92"):
        return f"0.{symbol}"
    return f"1.{symbol}"


# P2-Q3-fix(M386): fetch_financials 结果 TTL 缓存（screener 等批量场景避免重复网络请求）
_FETCH_CACHE_TTL = 600.0  # 秒
_FETCH_CACHE: dict[str, tuple[float, dict]] = {}


def fetch_financials(symbol: str) -> dict[str, Any]:
    """
    从AKShare获取基础财务数据，失败后尝试东方财富/新浪。

    D5收敛登记: 独立能力保留 —— 与 financial_data.fetch_financial_indicators
    异名异实现：本函数取 THS 摘要字段 + 东财 push2 行情估值 + 新浪 F10 兜底
    (内存600s TTL)，彼取 akshare financial_abstract 80指标×多期(SQLite 缓存)。

    P2-Q3-fix(M386): 600s TTL 结果缓存，重复调用不重复请求网络。

    Returns: {symbol, name, revenue_growth, profit_growth, roe,
              gross_margin, debt_ratio, dividend, ...} or error dict
    """
    cached = _FETCH_CACHE.get(symbol)
    if cached and _time.time() - cached[0] < _FETCH_CACHE_TTL:
        return dict(cached[1])

    prefix = tencent_prefix(symbol)

    # 1. Try AKShare first
    _akshare_err = ""
    try:
        import akshare as ak
        df = ak.stock_financial_abstract_ths(symbol=symbol)
        if df is not None and not df.empty:
            try:
                # V4.1 fix: sort descending by date to ensure iloc[0]=latest, iloc[1]=prev
                if "报告期" in df.columns:
                    df = df.sort_values("报告期", ascending=False).reset_index(drop=True)
                latest = df.iloc[0]
                prev = df.iloc[1] if len(df) > 1 else None
                result = {
                    "symbol": symbol,
                    "name": "",
                    "report_date": str(latest.get("报告期", ""))[:10],
                    "source": "akshare_ths",
                }
                # Map Chinese columns to English
                for cn, en in {"营业总收入": "revenue", "营业总收入同比增长率": "revenue_yoy",
                                 "净利润": "profit", "净利润同比增长率": "profit_yoy",
                                 "基本每股收益": "eps", "每股净资产": "bps",
                                 "净资产收益率": "roe", "销售毛利率": "gross_margin",
                                 "销售净利率": "net_margin", "资产负债率": "debt_ratio"}.items():
                    if cn in df.columns:
                        val = _safe_float(latest.get(cn))
                        if val is not None:
                            result[en] = val
                # P2-Q3-fix(M382): prev 行此前被取但从未使用，format_fundamentals 的
                # prev_roe/prev_revenue_yoy/prev_profit_yoy/deducted_ratio 分支全是死代码。
                # 现在从 prev 行提取上期对比字段，并计算扣非占比（扣非净利润/净利润）。
                if prev is not None:
                    prev_map = {
                        "roe": "净资产收益率",
                        "revenue_yoy": "营业总收入同比增长率",
                        "profit_yoy": "净利润同比增长率",
                    }
                    for en_key, cn_name in prev_map.items():
                        if cn_name in df.columns:
                            pv = _safe_float(prev.get(cn_name))
                            if pv is not None:
                                result[f"prev_{en_key}"] = pv
                # 扣非占比：优先最新期，缺扣非列则用上一期
                for row_ in (latest, prev):
                    if row_ is None:
                        continue
                    deducted = _safe_float(row_.get("扣非净利润")) if "扣非净利润" in df.columns else None
                    net_profit = _safe_float(row_.get("净利润"))
                    if deducted is not None and net_profit not in (None, 0):
                        result["deducted_ratio"] = deducted / net_profit
                        break
                _FETCH_CACHE[symbol] = (_time.time(), dict(result))
                return result
            except Exception as inner_e:
                return {"symbol": symbol, "source": "akshare_parse_error", "error": str(inner_e)[:100]}
    except Exception as e:
        _akshare_err = f"akshare: {e}"

    # 2. Fallback: East Money push2 for PE/PB
    try:
        secid = _em_secid(symbol)
        # V4.1 fix: correct PE/PB field mapping (f57=代码, f58=名称)
        # 东方财富push2字段对照: f162=静态PE, f167=动态PE, f164=PB, f58=名称
        # D5收敛: 同名异口径保留 —— PE/PB 为东财行情字段口径(静态/动态PE、PB, /100)，
        # 与 financial_data.pe_ttm/pb(收盘价/每股TTM)、fundamental_analysis 市值口径不同。
        url = f"https://push2delay.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f57,f58,f162,f164,f167,f170,f171"
        r = requests.get(url, headers=EASTMONEY_H, timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", {})
            # P2-Q3-fix(L384): 提取 f58 名称；"-"/None 数值字段用 _safe_float 守卫，
            # 不再触发 "-"/100 TypeError 被静默吞掉后直接落到 no_data。
            name = data.get("f58") or ""
            price_raw = _safe_float(data.get("f43"))
            pe_static = _safe_float(data.get("f162"))
            pe_dynamic = _safe_float(data.get("f167"))
            pb_raw = _safe_float(data.get("f164"))
            if price_raw is None:
                print(f"  ⚠️ 东财 {symbol}: 价格字段异常值={data.get('f43')!r}", file=sys.stderr)
            result = {
                "symbol": symbol,
                "name": name,
                "price": (price_raw or 0) / 100,
                "pe": (pe_static or 0) / 100 if pe_static else ((pe_dynamic or 0) / 100 if pe_dynamic else 0),
                "pb": (pb_raw or 0) / 100,
                "high": (_safe_float(data.get("f44")) or 0) / 100,
                "low": (_safe_float(data.get("f45")) or 0) / 100,
                "open": (_safe_float(data.get("f46")) or 0) / 100,
                "volume": data.get("f47", 0),
                "amount": data.get("f48", 0),
                "source": "eastmoney_push2",
            }
            _FETCH_CACHE[symbol] = (_time.time(), dict(result))
            return result
    except Exception as e:
        logging.getLogger(__name__).error(f"[fundamental] 操作失败: {e}", exc_info=True)

    # 3. Fallback: 新浪 F10（P2-Q3-fix L383: 此前该函数从未被调用，死代码；现接入兜底链）
    sina = _fetch_financials_v2(symbol)
    if "error" not in sina:
        _FETCH_CACHE[symbol] = (_time.time(), dict(sina))
        return sina

    # 4. Last resort: generic message
    return {"symbol": symbol, "source": "no_data", "error": "基本面数据不可用，盘中请刷新", "note": "try_again"}


def _fetch_financials_v2(symbol: str) -> dict:
    """Fallback: use Sina F10 API for basic financials."""
    prefix = tencent_prefix(symbol)
    
    # Sina F10 financial summary
    url = f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/FinanceSummary.getFinanceSummary?symbol={prefix}"
    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn",
        }, timeout=8)
        if r.status_code == 200 and r.text.strip() not in ("", "null"):
            data = json.loads(r.text)
            return {
                "symbol": symbol,
                "source": "sina_fallback",
                **{k: v for k, v in data.items() if isinstance(v, (str, int, float, type(None)))},
            }
    except Exception as e:
        logging.getLogger(__name__).error(f"[fundamental] 操作失败: {e}", exc_info=True)
    
    return {"symbol": symbol, "error": "无法获取财务数据，可能需要本地数据源"}


def format_fundamentals(data: dict) -> str:
    """Format fundamentals into readable text.

    D5收敛登记: 独立能力保留（展示层格式化，无等价实现）。
    """
    if "error" in data:
        return f"⚠️ {data['error']}"
    
    name = data.get("name", data.get("symbol", ""))
    lines = [f"📊 **{name}({data['symbol']})** 基本面"]
    report_date = data.get("report_date", "")
    if report_date:
        lines.append(f"  报告期: {report_date[:10]}")
    
    # ROE
    roe = _safe_float(data.get("roe"))
    if roe is not None:
        roe_str = f"{roe:.2f}%"
        prev_roe = _safe_float(data.get("prev_roe"))
        if prev_roe is not None:
            roe_str += f" (上期{prev_roe:.2f}%)"
        lines.append(f"  ROE: {roe_str}")
        if roe > 20:
            lines.append(f"       💎 ROE>20% 优秀")
        elif roe > 15:
            lines.append(f"       ✅ ROE>15% 良好")
        elif roe > 10:
            lines.append(f"       ➖ ROE>10% 中等")
        else:
            lines.append(f"       ⚠️ ROE<10% 偏低")
    
    # Revenue growth
    rev = _safe_float(data.get("revenue_yoy"))
    if rev is not None:
        rev_str = f"{rev:.2f}%"
        prev_rev = _safe_float(data.get("prev_revenue_yoy"))
        if prev_rev is not None:
            change = rev - prev_rev
            arrow = "⬆️" if change > 0 else "⬇️"
            rev_str += f" (上期{prev_rev:.2f}% {arrow})"
        lines.append(f"  营收增速: {rev_str}")
        if rev > 30:
            lines.append(f"       🚀 高速增长")
        elif rev > 15:
            lines.append(f"       ✅ 稳定增长")
        elif rev > 0:
            lines.append(f"       ➖ 正增长")
        else:
            lines.append(f"       🔴 负增长")
    
    # Profit growth
    profit = _safe_float(data.get("profit_yoy"))
    if profit is not None:
        profit_str = f"{profit:.2f}%"
        prev_profit = _safe_float(data.get("prev_profit_yoy"))
        if prev_profit is not None:
            change = profit - prev_profit
            arrow = "⬆️" if change > 0 else "⬇️"
            profit_str += f" (上期{prev_profit:.2f}% {arrow})"
        lines.append(f"  净利润增速: {profit_str}")
    
    # Deducted ratio (quality)
    dr = _safe_float(data.get("deducted_ratio"))
    if dr is not None:
        lines.append(f"  扣非净利润占比: {dr:.1%}")
        if dr > 0.9:
            lines.append(f"       ✅ 利润质量高(扣非占比>90%)")
        elif dr > 0.7:
            lines.append(f"       ➖ 利润质量中等")
        else:
            lines.append(f"       ⚠️ 利润靠非经常性损益支撑")
    
    # Gross margin
    gm = _safe_float(data.get("gross_margin"))
    if gm is not None:
        lines.append(f"  毛利率: {gm:.2f}%")
        if gm > 60:
            lines.append(f"       💎 高毛利率(护城河)")
        elif gm > 30:
            lines.append(f"       ✅ 毛利率良好")
        else:
            lines.append(f"       ➖ 毛利率一般")
    
    # Debt ratio
    dr2 = _safe_float(data.get("debt_ratio"))
    if dr2 is not None:
        lines.append(f"  资产负债率: {dr2:.2f}%")
        if dr2 > 70:
            lines.append(f"       ⚠️ 负债率偏高")
        elif dr2 > 50:
            lines.append(f"       ➖ 中等负债")
        else:
            lines.append(f"       ✅ 负债率较低")
    
    # EPS
    eps = _safe_float(data.get("eps"))
    if eps is not None:
        lines.append(f"  每股收益: {eps:.3f}")
    
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 基本面筛选器 (基于已有的409只,加财务指标过滤)
# ════════════════════════════════════════════════════════════════

def _backfill_market_cap(symbol: str) -> float | None:
    """用东财 push2 接口补齐市值（亿元）；失败返回 None。

    D5收敛登记: 独立能力保留（screener 市值补齐专用）。
    """
    secid = _em_secid(symbol)
    try:
        r = requests.get(
            f"https://push2delay.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f116",
            headers=EASTMONEY_H, timeout=5,
        )
        if r.status_code == 200:
            d = r.json().get("data") or {}
            mc = _safe_float(d.get("f116"))
            if mc is not None and mc > 0:
                return mc / 1e8  # 元 → 亿元
    except Exception as e:
        logging.getLogger(__name__).error(f"[fundamental] 操作失败: {e}", exc_info=True)
    return None


def screener(min_roe: float = 10, min_revenue_growth: float = 5,
             min_profit_growth: float = 5, max_stocks: int = 20) -> list[dict]:
    """
    基本面筛选器: 从409只股票中找符合财务条件的。

    P2-Q3-fix(M386):
      ① watchlist 条目缺 market_cap_yi（此前默认取 0）会导致排序失真：
         缺键的先用东财接口补齐市值，补齐失败则按 -1 排在最后；
      ② fetch_financials 已带 600s TTL 缓存；
      ③ 请求间加 0.2s 限速，避免触发东财限流。

    D5收敛登记: 独立能力保留（基于 watchlist 池的筛选器，无等价实现）。

    ⚠️ 注意: 东方财富API并发有限,会分批查询,耗时较长。
    """
    from quant_system.watchlist import get_watchlist
    stocks = get_watchlist()

    # 补齐缺失市值，避免排序失真
    cleaned = []
    for s in stocks:
        if not s.get("market_cap_yi") or s.get("market_cap_yi") <= 0:
            mc = _backfill_market_cap(s["symbol"])
            if mc:
                s = dict(s, market_cap_yi=mc)
            else:
                s = dict(s, market_cap_yi=-1.0)  # 补齐失败 → 排最后
        cleaned.append(s)

    # Take top 100 by market cap for speed
    top_stocks = sorted(cleaned, key=lambda s: s.get("market_cap_yi", -1.0), reverse=True)[:100]

    results = []
    for i, s in enumerate(top_stocks):
        sym = s["symbol"]
        fin = fetch_financials(sym)
        if "error" in fin:
            continue
        
        roe = _safe_float(fin.get("roe")) or 0
        rev = _safe_float(fin.get("revenue_yoy")) or 0
        profit = _safe_float(fin.get("profit_yoy")) or 0
        
        if roe >= min_roe and rev >= min_revenue_growth and profit >= min_profit_growth:
            results.append({
                "symbol": sym,
                "name": fin.get("name", s.get("name", "")),
                "roe": roe,
                "revenue_yoy": rev,
                "profit_yoy": profit,
                "gross_margin": _safe_float(fin.get("gross_margin")),
                "market_cap_yi": s.get("market_cap_yi", 0),
            })

        # P2-Q3-fix(M386-③): 请求限速，避免触发东财/同花顺限流
        if i < len(top_stocks) - 1:
            _time.sleep(0.2)

    results.sort(key=lambda r: r["roe"], reverse=True)
    return results[:max_stocks]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="个股基本面查询")
    parser.add_argument("--screener", action="store_true", help="基本面筛选")
    args = parser.parse_args()
    
    if args.screener:
        print("📊 **基本面筛选 (ROE≥10%, 营收增速≥5%, 利润增速≥5%)**")
        print("⚠️ 从市值前100只中筛选,可能需要等待...\n")
        t0 = _time.time()
        results = screener()
        elapsed = _time.time() - t0
        print(f"⏱ 耗时 {elapsed:.0f}s, 找到 {len(results)} 只\n")
        for r in results:
            print(f"  ✅ {r['symbol']} {r['name'][:8]:<8} ROE={r['roe']:.1f}% "
                  f"营收{r['revenue_yoy']:+.1f}% 利润{r['profit_yoy']:+.1f}% "
                  f"毛利{r.get('gross_margin',0):.1f}% {r['market_cap_yi']:.0f}亿")
    
    elif args.symbol:
        t0 = _time.time()
        data = fetch_financials(args.symbol)
        elapsed = _time.time() - t0
        print(format_fundamentals(data))
        print(f"\n⏱ {elapsed:.0f}s")
    
    else:
        parser.print_help()
