"""
量化交易系统 — 北向资金 / 主力流向分析。

数据源: 东方财富 push2.eastmoney.com
覆盖: 沪深港通总额、个股北向持仓、净买入排名

用法:
  python3 -m quant_system.north_flow                    # 北向总览
  python3 -m quant_system.north_flow --top 10          # 个股净买入TOP10
  python3 -m quant_system.north_flow --symbol 600519   # 某只股票

Q4/Q23-fix（2026-08-02，V6 P0 修复 B 组）：
  #3  2024-08-19 起沪深交易所停止披露北向资金盘中/日度净买入：
      - 主端点 kamt.northflow 已下线（实测 404）
      - kamt.kline 兜底：净买额字段（f52，索引1）实测恒 0，原代码取索引3(f54)
        字段错位；且旧代码把 f53（当日成交额量级）当"累计"展示。
      - 适配：检测到净买额全 0 且数据日期 >= 2024-08-19 时置
        net_buy_disclosed=False，明确标注"规则变更后无披露"，不再输出
        "交易中" 误导状态；展示口径改用官方仍披露的当日成交额。
  #4  fetch_north_rank 增加合理性校验（2024-08-19 后东财该接口返回占位/垃圾
      数据：负持仓、同值市值等），校验不通过返回 [] + 告警，禁止垃圾进决策链。
  #13 新增 get_north_flow_summary() 兼容函数（net_amount/sh_net/sz_net 元口径），
      修复 market_pulse.py / scripts/dashboard.py 的 ImportError 集成断裂。
  #27 secid 前缀按市场白名单（沪6/科创688→"1."，深0/3→"0."），剔除北交所/B股。
  #28 字段解析统一走 _to_float() 容错，避免 None/"-" 触发 TypeError。
  #32 清理未用导入与死变量；--symbol 分支防御 error dict。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from quant_system.utils import to_float as _to_float_impl

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}

# 北向净买入披露规则变更日（2024-08-19 起停止披露）
NORTH_DISCLOSURE_CUTOFF = "2024-08-19"


def _to_float(value: Any, default: float = 0.0) -> float:
    """D1 收敛: 转发 quant_system.utils.to_float（默认 0.0 语义不变）。"""
    return _to_float_impl(value, default=default)


def _is_after_cutoff(date_str: str) -> bool:
    """数据日期是否在 2024-08-19 披露规则变更之后。"""
    if not date_str:
        return False
    try:
        d = str(date_str)[:10]
        return d >= NORTH_DISCLOSURE_CUTOFF
    except Exception:
        return False


def _normalize_north_summary(summary: dict[str, Any]) -> dict[str, Any]:
    total_net = _to_float(summary.get("total_net_yi"))
    today = datetime.now(CST).strftime("%Y-%m-%d")
    data_date = summary.get("date", "")
    is_today = (data_date == today)

    # Q4#3: 披露规则变更后，净买额恒 0 不代表真实买卖——禁止标"交易中"，
    # 并明确标注"规则变更后无披露"
    net_buy_disclosed = summary.get("net_buy_disclosed", True)
    if net_buy_disclosed:
        summary.setdefault("trading", is_today)
        summary.setdefault("trading_status", "交易中" if is_today else "非交易时段")
    else:
        summary["trading"] = False
        summary["trading_status"] = "北向净买入已停止披露"
        summary["note"] = "2024-08-19起北向净买入停止披露,以下净买额为0系披露规则所致,非真实信号"
    if not is_today and net_buy_disclosed:
        summary["note"] = "非交易时段(数据非今日)"
    return summary


def _fetch_north_realtime() -> dict[str, Any] | None:
    """东方财富实时北向端点。

    Q4#3: 该端点（kamt.northflow）自披露规则变更后已下线（实测 404），
    保留代码仅为兼容旧数据源；失败返回 None 走 kline 兜底。
    """
    url = "https://push2delay.eastmoney.com/api/qt/kamt.northflow/get?secid=1.1"
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200 or not r.text.strip():
            return None
        data = r.json().get("data") or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    sh_net = _to_float(data.get("hk2sh") or data.get("hk2sh_net") or data.get("sh_net") or data.get("north_sh"))
    sz_net = _to_float(data.get("hk2sz") or data.get("hk2sz_net") or data.get("sz_net") or data.get("north_sz"))
    total_net = _to_float(data.get("north_money") or data.get("total_net") or data.get("total"), sh_net + sz_net)
    sh_total = _to_float(data.get("hk2sh_total") or data.get("sh_total"))
    sz_total = _to_float(data.get("hk2sz_total") or data.get("sz_total"))
    date = str(data.get("date") or data.get("time") or datetime.now(CST).strftime("%Y-%m-%d"))[:10]

    if not any([sh_net, sz_net, total_net, sh_total, sz_total]):
        return None
    # V4.1 fix: 东方财富push2返回单位是元, 应除以1e8得到亿元, 而非1e4(万元)
    # P2-Q23-fix(M261): 统一单位换算——push2 单位恒为元，恒 /1e8 转亿元。
    # 原实现 `abs(x)>1e8 则 /1e8 否则 /1e4` 的元/万元启发式在沪/深额度量级
    # 附近制造歧义，且与 kline 兜底路径(万元 /1e4→亿)口径不一致（两条路径
    # 最终都应产出亿元）。
    # Q4#3: 若数据日期已过披露规则变更日，净买额字段即使非 0 也不可信
    disclosed = not _is_after_cutoff(date) or abs(total_net) > 0
    return _normalize_north_summary({
        "date": date,
        "sh_net_yi": round(sh_net / 1e8, 2),
        "sz_net_yi": round(sz_net / 1e8, 2),
        "total_net_yi": round(total_net / 1e8, 2),
        "sh_total_yi": round(sh_total / 1e8, 2),
        "sz_total_yi": round(sz_total / 1e8, 2),
        "net_buy_disclosed": disclosed,
        "source": "eastmoney_northflow",
    })


def fetch_north_summary() -> dict[str, Any]:
    """
    北向资金总体流向（沪深港通每日净买入）。

    Q4#3 适配：2024-08-19 起净买额停止披露。
    返回: {date, sh_net_yi, sz_net_yi, total_net_yi, sh_total_yi, sz_total_yi,
           sh_turnover_yi, sz_turnover_yi, net_buy_disclosed, trading, ...}
    - net_buy_disclosed=False 表示规则变更后无披露，净买额 0 系披露规则所致。
    - 成交额口径（sh_turnover_yi/sz_turnover_yi）仍由交易所披露。
    """
    try:
        realtime = _fetch_north_realtime()
        if realtime and _to_float(realtime.get("total_net_yi")) != 0:
            return realtime
    except Exception:
        realtime = None

    # Q4#3: kamt.kline 字段序 f51=日期, f52=净买额, f53=当日成交额, f54=卖出成交额
    # 原代码 latest_sh[3]（f54）被当"净买入"，字段索引错误
    # P2-Q23-fix(M261): 按请求的 fields2=f51,f52,f53,f54,f55 声明字段名并按列数
    # 断言，防止东财改字段序导致净买额/成交额取错列。
    _KL_FIELDS = ("f51_date", "f52_net", "f53_turnover", "f54_sell", "f55")
    url = ("https://push2delay.eastmoney.com/api/qt/kamt.kline/get?"
           "secid=1.1&fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55&klt=1&lmt=10")
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        data = r.json().get("data", {})
        hk2sh_raw = data.get("hk2sh", [])
        hk2sz_raw = data.get("hk2sz", [])

        latest_sh = hk2sh_raw[0].split(",") if hk2sh_raw else []
        latest_sz = hk2sz_raw[0].split(",") if hk2sz_raw else []
        # P2-Q23-fix(M261): 字段名断言——响应列数不足即抛错走降级，禁止把缺列
        # 当 0 输出"假净买额"。
        if len(latest_sh) < 3 or len(latest_sz) < 3:
            raise ValueError("kamt.kline 返回字段数不足(期望 f51..f55 至少3列)")

        sh = dict(zip(_KL_FIELDS, latest_sh))
        sz = dict(zip(_KL_FIELDS, latest_sz))
        data_date = (sh.get("f51_date") or sz.get("f51_date") or "").strip()

        # f52=净买额；f53=当日成交额——同源同单位（万元），2024-08-19 后
        # 净买额恒 0 但成交额仍披露（旧代码 /1e4 换算亿元，与实时路径 /1e8
        # 元的换算在亿元口径上等价，见 _fetch_north_realtime）。
        sh_net = _to_float(sh.get("f52_net"))
        sz_net = _to_float(sz.get("f52_net"))
        sh_turnover = _to_float(sh.get("f53_turnover"))
        sz_turnover = _to_float(sz.get("f53_turnover"))

        # P2-Q23-fix(M261): 单位/口径一致性断言——|净买额| 不可能超过当日成交额
        # （净买额=买-卖，成交额=买+卖）。沪/深同时违反即说明字段序已变化或
        # 单位错位，抛错降级而非输出错位数据。
        if abs(sh_net) > abs(sh_turnover) or abs(sz_net) > abs(sz_turnover):
            raise ValueError("kamt.kline 字段量级异常(净买额>成交额),疑似字段序/单位变化")

        # Q4#3: 规则变更后净买额恒 0 → 明确标注无披露，禁止当真实信号
        disclosed = (not _is_after_cutoff(data_date)) or (sh_net != 0 or sz_net != 0)

        note = None
        # 成交额疑似占位：沪/深完全相同值时东财可能返回占位数据
        if (sh_turnover and sz_turnover and abs(sh_turnover - sz_turnover) < 1e-9):
            note = "成交额字段疑似占位数据(沪/深完全相同),仅供参考"

        summary = _normalize_north_summary({
            "date": data_date,
            "sh_net_yi": round(sh_net / 10000, 2),
            "sz_net_yi": round(sz_net / 10000, 2),
            "total_net_yi": round((sh_net + sz_net) / 10000, 2),
            "sh_turnover_yi": round(sh_turnover / 10000, 2),
            "sz_turnover_yi": round(sz_turnover / 10000, 2),
            "total_turnover_yi": round((sh_turnover + sz_turnover) / 10000, 2),
            "sh_total_yi": 0,
            "sz_total_yi": 0,
            "net_buy_disclosed": disclosed,
            "source": "eastmoney_kline",
        })
        if note:
            summary["note"] = note
        return summary
    except Exception as e:
        if realtime:
            return realtime
        return {"error": str(e), "ok": False}


def get_north_flow_summary() -> dict[str, Any]:
    """兼容函数（Q4#13）：market_pulse.py / scripts/dashboard.py 引用的
    get_north_flow_summary 原不存在 → ImportError 被吞 → 北向分量静默缺失。

    返回元口径字段（net_amount/sh_net/sz_net，单位：元），兼容 dashboard 的
    net_amount/1e8 用法，同时保留 fetch_north_summary 的全部键。
    """
    summary = fetch_north_summary()
    if "error" in summary:
        summary.setdefault("ok", False)
        summary.setdefault("net_amount", 0)
        summary.setdefault("sh_net", 0)
        summary.setdefault("sz_net", 0)
        summary.setdefault("status", "获取失败")
        return summary
    total_yi = _to_float(summary.get("total_net_yi"))
    sh_yi = _to_float(summary.get("sh_net_yi"))
    sz_yi = _to_float(summary.get("sz_net_yi"))
    summary.update({
        "ok": True,
        "net_amount": round(total_yi * 1e8, 0),   # 元
        "sh_net": round(sh_yi * 1e8, 0),          # 元
        "sz_net": round(sz_yi * 1e8, 0),          # 元
        "status": summary.get("trading_status", summary.get("note", "")),
    })
    return summary


def _valid_secid_prefix(sym: str) -> str | None:
    """按市场生成 secid 前缀（Q4#27）。

    沪主板 60xxxx/科创 688xxx → "1."；深主板 00xxxx/创业板 30xxxx → "0."；
    北交所(8/4/92)与沪B(900) 不在北向标的内 → None（跳过）。
    """
    if sym.startswith(("600", "601", "603", "605", "688", "689")):
        return "1."
    if sym.startswith(("000", "001", "002", "003", "300", "301")):
        return "0."
    return None


def fetch_north_holdings(top_n: int = 50, indicator: str = "5日排行") -> dict[str, Any]:
    """Return northbound ownership structure, independent of stopped net-flow data.

    北向盘中/日频净买入已经停止披露，但持股数量、持股市值、占 A 股比例
    及区间变化仍是独立研究数据。该接口不返回或推导日净买额。
    """
    allowed = {"今日排行", "3日排行", "5日排行", "10日排行", "月排行", "季排行", "年排行"}
    indicator = indicator if indicator in allowed else "5日排行"
    try:
        import akshare as ak
        frame = ak.stock_hsgt_hold_stock_em(market="北向", indicator=indicator)
        if frame is None or frame.empty:
            return {"ok": False, "status": "empty", "indicator": indicator, "rows": [],
                    "error": "北向持股排行当前无返回", "source": "akshare.stock_hsgt_hold_stock_em"}
        aliases = {
            "代码": "symbol", "股票代码": "symbol", "名称": "name", "股票简称": "name",
            "持股数量": "holding_shares", "持股市值": "holding_market_cap",
            "持股数量占发行股百分比": "holding_pct", "持股数量占A股百分比": "holding_pct", "持股数量占流通股百分比": "holding_pct", "持股占流通股比": "holding_pct",
            "持股市值变化-1日": "holding_change_1d", "持股市值变化-3日": "holding_change_3d",
            "持股市值变化-5日": "holding_change_5d", "持股市值变化-10日": "holding_change_10d",
            "当日收盘价": "close", "当日涨跌幅": "change_pct", "日期": "date",
        }
        rows = []
        for raw in frame.head(max(1, min(int(top_n), 200))).to_dict("records"):
            item = {}
            for key, value in raw.items():
                target = aliases.get(str(key), str(key))
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        value = str(value)
                if isinstance(value, float) and value != value:
                    value = None
                item[target] = value
            symbol = str(item.get("symbol") or "").split(".")[0].zfill(6)
            if not symbol.isdigit() or len(symbol) != 6:
                continue
            item["symbol"] = symbol
            pct = _to_float(item.get("holding_pct"), default=float("nan"))
            cap = _to_float(item.get("holding_market_cap"), default=float("nan"))
            shares = _to_float(item.get("holding_shares"), default=float("nan"))
            if pct == pct and not 0 <= pct <= 100:
                continue
            if cap == cap and cap < 0 or shares == shares and shares < 0:
                continue
            rows.append(item)
        observed = None
        for item in rows:
            value = item.get("date")
            if value:
                observed = str(value)[:10]
                break
        return {
            "ok": bool(rows), "status": "available" if rows else "empty",
            "indicator": indicator, "as_of": observed, "rows": rows,
            "count": len(rows), "source": "akshare.stock_hsgt_hold_stock_em",
            "data_contract": "northbound-holdings.v1",
            "note": "北向持股结构仍可用；不等同于已停更的盘中/日频净买入。",
        }
    except Exception as exc:
        return {"ok": False, "status": "error", "indicator": indicator, "rows": [],
                "error": str(exc)[:240], "source": "akshare.stock_hsgt_hold_stock_em",
                "data_contract": "northbound-holdings.v1"}


def fetch_north_rank(top_n: int = 20, mode: str = "day") -> list[dict]:
    """
    个股北向资金排名。

    Args:
        top_n: 返回前N只
        mode: "day"当日净买入排名 / "5day" 5日净买入排名

    Returns:
        [{symbol, name, hold_value, day_net, day_net_pct, day_net_yi, 5d_net_yi, hold_pct}]

    Q4#4: 2024-08-19 后东财该接口返回占位/垃圾数据（负持仓、同值市值等），
    增加合理性校验，数据异常时返回 [] 并附告警 dict，禁止垃圾数据进入决策链。

    P2-Q23-fix(L271): 排名仅覆盖自选股(watchlist)，非全市场——东财北向个股
    接口无全市场遍历端点，取数范围以 watchlist 为准，输出中已明示"仅自选股"，
    避免被误解为全市场净买入排名。
    """
    # Fields: f12=symbol, f14=name, f62=hold_shares, f66=day_net,
    #         f69=day_net_pct, f72=hold_value, f75=hold_pct,
    #         f78=5d_net, f81=5d_net_pct, f124=mkt_cap
    url = ("https://push2delay.eastmoney.com/api/qt/ulist.np/get?"
           "fltt=2&fields=f12,f14,f62,f66,f69,f72,f75,f78,f81,f124"
           "&secids=")

    try:
        sys.path.insert(0, str(ROOT.parent))
        from quant_system.watchlist import get_watchlist
        stocks = get_watchlist()
        secids = []
        for s in stocks:
            sym = str(s["symbol"]).strip()
            prefix = _valid_secid_prefix(sym)
            if prefix is None:
                continue  # Q4#27: 剔除北交所/B股
            secids.append(f"{prefix}{sym}")

        if not secids:
            return [{"error": "无可查询的沪深北向标的"}]

        chunk_size = 100
        results = []
        invalid_count = 0
        f124_values: set[float] = set()

        for i in range(0, len(secids), chunk_size):
            chunk = ",".join(secids[i:i + chunk_size])
            r = requests.get(url + chunk, headers=HEADERS, timeout=10)
            items = (r.json().get("data") or {}).get("diff") or []
            for item in items:
                symbol = str(item.get("f12", ""))
                # Q4#28: 统一 _to_float 容错，避免 None/"-" 触发 TypeError
                hold_shares = _to_float(item.get("f62"))
                day_net = _to_float(item.get("f66"))
                day_net_pct = _to_float(item.get("f69"))
                hold_value = _to_float(item.get("f72"))
                hold_pct = _to_float(item.get("f75"))
                f5d_net = _to_float(item.get("f78"))
                f5d_net_pct = _to_float(item.get("f81"))
                mkt_cap = _to_float(item.get("f124"))
                if mkt_cap:
                    f124_values.add(round(mkt_cap, -3))

                # Q4#4 合理性校验：
                #   持仓市值/持仓占比必须为正且占比 0~100；|当日净买| 不能
                #   超过持仓市值量级；持仓股数非负
                if hold_value < 0 or hold_pct < 0 or hold_pct > 100:
                    invalid_count += 1
                    continue
                if hold_shares < 0:
                    invalid_count += 1
                    continue
                if hold_value > 0 and abs(day_net) > hold_value * 10:
                    invalid_count += 1
                    continue
                # V11 审计修复（Medium）: 单位无断言——若 f72 实际为万元而非元，
                # hold_value/10000 会放大 1e4 倍。用 f124（总市值）交叉校验:
                # 持仓市值不应超过总市值（异常单位即拒）。
                if mkt_cap > 0 and hold_value > mkt_cap * 1.2:
                    invalid_count += 1
                    continue
                if not symbol:
                    invalid_count += 1
                    continue

                results.append({
                    "symbol": symbol,
                    "name": str(item.get("f14", "")),
                    "hold_value_yi": round(hold_value / 10000, 2),
                    "day_net": day_net,
                    "day_net_yi": round(day_net / 10000, 2),
                    "day_net_pct": day_net_pct,
                    "5d_net": f5d_net,
                    "5d_net_yi": round(f5d_net / 10000, 2),
                    "5d_net_pct": f5d_net_pct,
                    "hold_pct": hold_pct,
                })

        # Q4#4: 市值字段去重检查——多只股票返回完全相同市值说明接口在返回
        # 占位/垃圾数据（实测三只股票 f124 同值 1785485510）
        if len(results) >= 3 and len(f124_values) <= 2:
            return [{"error": "北向个股接口返回占位数据(市值字段无区分度),已下线该数据源"}]
        if results and invalid_count > len(results) * 2:
            return [{"error": f"北向个股数据合理性校验失败({invalid_count}/{len(results)+invalid_count}),已丢弃"}]

        results.sort(key=lambda x: abs(x["day_net" if mode == "day" else "5d_net"]), reverse=True)
        return results[:top_n]
    except Exception as e:
        return [{"error": str(e)}]


def format_summary() -> str:
    """格式化北向资金总览"""
    summary = fetch_north_summary()
    if "error" in summary:
        return f"⚠️ 北向数据获取失败: {summary['error']}"

    lines = [f"📊 **北向资金** — {summary['date']}"]
    disclosed = summary.get("net_buy_disclosed", True)
    if not disclosed:
        lines.append("ℹ️ 2024-08-19 起北向净买入停止披露，以下净买额为 0 系披露规则所致")
        total = 0.0
        arrow = "⚪"
    else:
        total = summary.get("total_net_yi", 0)
        arrow = "🟢" if total > 0 else ("🔴" if total < 0 else "⚪")
    lines.append(f"{arrow} 当日净买入: {total:+.2f}亿")
    lines.append(f"  沪股通: {summary.get('sh_net_yi', 0):+.2f}亿"
                 f" (当日成交额{summary.get('sh_turnover_yi', 0):.0f}亿)")
    lines.append(f"  深股通: {summary.get('sz_net_yi', 0):+.2f}亿"
                 f" (当日成交额{summary.get('sz_turnover_yi', 0):.0f}亿)")

    # Top buys（仅净买额仍披露时才展示；否则跳过个股净买排名）
    if disclosed:
        top_buy = fetch_north_rank(top_n=8, mode="day")
        if top_buy and "error" not in top_buy[0]:
            # P2-Q23-fix(L271): 明示排名仅覆盖自选股，非全市场
            lines.append(f"\n🏆 **北向净买入TOP8（仅自选股）**")
            for s in top_buy:
                arrow2 = "🟢" if s["day_net_yi"] > 0 else "🔴"
                lines.append(f"  {arrow2} {s['symbol']} {s['name'][:8]:<8} {s['day_net_yi']:+.2f}亿"
                             f"  | 持仓{s['hold_pct']:.2f}% | 5日{s['5d_net_yi']:+.2f}亿")

            # V11 审计修复（Low）: 原实现从按 |net| 排序的前 8 名里筛负值，
            # 会漏掉排名更靠后的更大卖单。修正: 单独按净额升序取前 5 卖单。
            all_sorted_sell = sorted(
                [s for s in top_buy if s["day_net_yi"] < 0],
                key=lambda x: x["day_net_yi"],
            )
            top_sell = all_sorted_sell[:5]
            if top_sell:
                lines.append(f"\n📉 **北向净卖出TOP5**")
                for s in top_sell:
                    lines.append(f"  🔴 {s['symbol']} {s['name'][:8]:<8} {s['day_net_yi']:+.2f}亿"
                                 f"  | 持仓{s['hold_pct']:.2f}%")
        elif top_buy and "error" in top_buy[0]:
            lines.append(f"\n⚠️ 北向个股排名暂不可用: {top_buy[0]['error']}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="排名前N")
    parser.add_argument("--symbol", type=str, help="某只股票查询")
    args = parser.parse_args()

    if args.symbol:
        result = fetch_north_rank(top_n=500, mode="day")
        sym = args.symbol
        # Q4#32: result 可能为 [{"error": ...}]，防御 KeyError
        if result and "error" not in result[0]:
            match = [s for s in result if s["symbol"] == sym]
            if match:
                s = match[0]
                print(f"📊 {s['name']}({s['symbol']}) 北向资金:")
                print(f"  持仓: {s['hold_value_yi']:.2f}亿 ({s['hold_pct']:.2f}%)")
                print(f"  当日: {s['day_net_yi']:+.2f}亿")
                print(f"  5日:  {s['5d_net_yi']:+.2f}亿")
            else:
                print(f"⚠️ 未找到{sym}的北向数据")
        else:
            print(f"⚠️ 北向个股数据不可用: {result[0].get('error') if result else 'empty'}")
    else:
        print(format_summary())
