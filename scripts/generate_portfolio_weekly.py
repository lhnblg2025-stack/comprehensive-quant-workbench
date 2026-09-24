#!/usr/bin/env python3
"""Hard weekly reports for Zijin, Muyuan, pig cycle, and China Internet ETF."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from report_delivery import finalize_report  # noqa: E402
from report_paths import iso_week, report_path  # noqa: E402

CONFIG = {
    "zijin_weekly": {"title": "紫金矿业周报", "code": "601899", "report_key": "zijin_weekly"},
    "muyuan_weekly": {"title": "牧原股份周报", "code": "002714", "report_key": "muyuan_weekly"},
    "pig_weekly": {"title": "生猪周期周报", "code": "002714", "report_key": "pig_weekly"},
    "china_internet_weekly": {"title": "159792港股通互联网ETF定投周报", "code": "159792", "report_key": "china_internet_weekly"},
}


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
        return "-" if not math.isfinite(number) else f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _stock_history(code: str, limit: int = 180) -> pd.DataFrame:
    path = ROOT / "data_warehouse" / "kline" / f"{code}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "close"]).sort_values("date").tail(limit)


def _valuation(code: str) -> dict[str, Any]:
    path = ROOT / "data_warehouse" / "valuation" / f"{code}.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    row = frame.sort_values("date").iloc[-1].to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def _etf_history(code: str) -> pd.DataFrame:
    path = ROOT / "data_warehouse" / "events" / "etf_state.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame[frame["code"].astype(str).str.zfill(6) == code].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date")


def _commodity(asset: str, limit: int = 60) -> pd.DataFrame:
    path = ROOT / "data_warehouse" / "market" / f"commodity__{asset}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    date_col = next((key for key in ("date", "日期") if key in frame.columns), None)
    close_col = next((key for key in ("close", "收盘价", "最新值") if key in frame.columns), None)
    if not date_col or not close_col:
        return pd.DataFrame()
    result = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"),
                           "close": pd.to_numeric(frame[close_col], errors="coerce")})
    return result.dropna().sort_values("date").tail(limit)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    close = frame["close"].astype(float)
    latest = float(close.iloc[-1])
    returns = close.pct_change().dropna()
    peak = close.cummax()
    drawdown = close / peak - 1
    def ret(period: int):
        return (latest / float(close.iloc[-period - 1]) - 1) * 100 if len(close) > period and close.iloc[-period - 1] else None
    return {
        "as_of": frame["date"].iloc[-1].date().isoformat(), "close": latest,
        "ret5": ret(5), "ret20": ret(20),
        "ma20": float(close.tail(20).mean()), "ma60": float(close.tail(60).mean()),
        "vol20": float(returns.tail(20).std() * math.sqrt(252) * 100) if len(returns) >= 2 else None,
        "max_drawdown": float(drawdown.min() * 100),
        "high20": float(close.tail(20).max()), "low20": float(close.tail(20).min()),
    }


def _history_table(frame: pd.DataFrame, limit: int = 20) -> list[str]:
    if frame.empty:
        return ["（本地历史不可用）"]
    columns = [column for column in ("date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg") if column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"]
    for row in frame.tail(limit).to_dict("records"):
        values = []
        for column in columns:
            value = row.get(column)
            values.append(str(value)[:10] if column == "date" else _fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _asset_report(task: str) -> str:
    cfg = CONFIG[task]
    code = cfg["code"]
    frame = _stock_history(code)
    metric = _metrics(frame)
    valuation = _valuation(code)
    pe = valuation.get("pe_ttm") if valuation.get("pe_ttm") is not None else valuation.get("peTTM")
    pb = valuation.get("pb") if valuation.get("pb") is not None else valuation.get("pbMRQ")
    title = cfg["title"]
    lines = [f"# {title}", "", f"生成日期：{date.today().isoformat()} · 周次：{iso_week()}", "",
             "## 核心结论与市场状态"]
    if not metric:
        lines.append("- 本地K线不可用，本周只保留观察，不生成方向性交易结论。")
    else:
        trend = "多头" if metric["close"] >= metric["ma20"] >= metric["ma60"] else ("修复" if metric["close"] >= metric["ma20"] else "偏弱")
        lines += [
            f"- 数据截至 {metric['as_of']}，收盘 {_fmt(metric['close'])}，5日 {_fmt(metric['ret5'])}% / 20日 {_fmt(metric['ret20'])}%。",
            f"- 趋势 {trend}：MA20 {_fmt(metric['ma20'])}，MA60 {_fmt(metric['ma60'])}，20日区间 {_fmt(metric['low20'])}-{_fmt(metric['high20'])}。",
            f"- 风险：20日年化波动 {_fmt(metric['vol20'])}%，180日最大回撤 {_fmt(metric['max_drawdown'])}%。",
            f"- 估值：PE(TTM) {_fmt(pe)}，PB {_fmt(pb)}；估值缺失时不补零。",
        ]
    if task == "zijin_weekly":
        gold, copper = _metrics(_commodity("gold")), _metrics(_commodity("copper"))
        lines += ["", "## 铜金周期与利润弹性",
                  f"- 沪金截至 {gold.get('as_of','-')}，20日变化 {_fmt(gold.get('ret20'))}%，相对MA20 {'上方' if gold and gold['close'] >= gold['ma20'] else '下方/缺失'}。",
                  f"- 沪铜截至 {copper.get('as_of','-')}，20日变化 {_fmt(copper.get('ret20'))}%，相对MA20 {'上方' if copper and copper['close'] >= copper['ma20'] else '下方/缺失'}。",
                  "- 执行：铜金同强且股价回踩MA20承接时才增加交易仓；铜跌破MA20或海外项目公告硬风险命中时降仓。"]
    elif task == "muyuan_weekly":
        pig = _metrics(_commodity("lh"))
        lines += ["", "## 猪价、成本与公司映射",
                  f"- 生猪期货截至 {pig.get('as_of','-')}，5日 {_fmt(pig.get('ret5'))}% / 20日 {_fmt(pig.get('ret20'))}%。",
                  "- 公司层必须同时观察出栏量、完全成本、仔猪销售、现金流和负债；股价上涨不能替代成本兑现。",
                  "- 执行：猪价上行、成本下降、股价站稳MA20三项至少两项确认才增加观察仓；跌破20日低点或成本反弹则退出交易仓。"]
    lines += ["", "## 下周行动卡",
              "1. 只在价格、产业驱动和风险门控同向时执行，不追单日脉冲。",
              "2. 回踩MA20不破且成交没有失速，可分两次试探；跌破20日低点视为失效。",
              "3. 公告、监管、现金流或行业价格出现反向证据时，优先减仓而不是解释。",
              "", "## 最近20个交易日原始明细", *_history_table(frame), "",
              "## 数据口径", f"- K线：data_warehouse/kline/{code}.parquet。", f"- 估值：data_warehouse/valuation/{code}.parquet。", "- 所有日期按本地真实观测；缺失不补零、不改标。"]
    return "\n".join(lines)


def _pig_report() -> str:
    frame = _stock_history("002714")
    pig = _commodity("lh")
    stock_metric, pig_metric = _metrics(frame), _metrics(pig)
    lines = ["# 生猪周期周报", "", f"生成日期：{date.today().isoformat()} · 周次：{iso_week()}", "",
             "## 周期判断",
             f"- 生猪期货截至 {pig_metric.get('as_of','-')}，5日 {_fmt(pig_metric.get('ret5'))}% / 20日 {_fmt(pig_metric.get('ret20'))}%，MA20 {_fmt(pig_metric.get('ma20'))}。",
             f"- 牧原股份截至 {stock_metric.get('as_of','-')}，5日 {_fmt(stock_metric.get('ret5'))}% / 20日 {_fmt(stock_metric.get('ret20'))}%，最大回撤 {_fmt(stock_metric.get('max_drawdown'))}%。",
             "- 周期链：能繁母猪存栏 -> 仔猪价格 -> 出栏体重 -> 猪价 -> 自繁自养利润 -> 现金流 -> 产能去化。缺一项不做强结论。",
             "", "## 情景与行动",
             "- 猪价站上MA20且牧原股价同步站上MA20：周期修复确认，观察仓分批建立。",
             "- 猪价反弹但股价/现金流不确认：只看行业，不执行个股。",
             "- 猪价跌破20日低点或饲料成本快速上行：周期结论失效，降低养殖链暴露。",
             "", "## 生猪期货最近20日", *_history_table(pig), "", "## 牧原股份最近20日", *_history_table(frame),
             "", "## 数据口径", "- 生猪：data_warehouse/market/commodity__lh.parquet。", "- 牧原：data_warehouse/kline/002714.parquet。", "- 能繁母猪等月度数据缺失时明确降级，不用时间周期推算冒充数据。"]
    return "\n".join(lines)


def _internet_report() -> str:
    frame = _etf_history("159792")
    latest = frame.iloc[-1].to_dict() if not frame.empty else {}
    lines = ["# 159792港股通互联网ETF定投周报", "", f"生成日期：{date.today().isoformat()} · 周次：{iso_week()}", "",
             "## 本周状态",
             f"- 数据截至 {str(latest.get('date','-'))[:10]}，价格 {_fmt(latest.get('price'))}，涨跌 {_fmt(latest.get('change_pct'))}%。",
             f"- 成交额 {_fmt((latest.get('amount') or 0)/1e8)}亿元，主力净额 {_fmt((latest.get('main_net') or 0)/1e8)}亿元，份额变化 {_fmt(latest.get('shares_delta'),0)}份。",
             f"- 折溢价 {_fmt(latest.get('discount_pct'))}%，IOPV {_fmt(latest.get('iopv'),4)}；折溢价字段按源数据保留。",
             "", "## 定投与风险纪律",
             "- 定投只解决择时分散，不解决估值、汇率和行业盈利风险；港股互联网必须同时看恒生科技趋势、美元流动性和平台盈利。",
             "- 价格下跌但份额持续增加、主力净额转正时可维持定投；价格上涨而份额下降、主力流出时不追。",
             "- 单周急涨不加速定投；跌破长期趋势且基本面下修时降低频率，而不是机械摊平。",
             "", "## ETF日频明细"]
    if frame.empty:
        lines.append("（ETF状态历史不可用）")
    else:
        cols = [key for key in ("date", "price", "change_pct", "amount", "main_net", "shares", "shares_delta", "discount_pct") if key in frame.columns]
        lines += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols)-1)) + "|"]
        for row in frame.tail(20).to_dict("records"):
            lines.append("| " + " | ".join(str(row.get(key))[:10] if key == "date" else _fmt(row.get(key)) for key in cols) + " |")
    lines += ["", "## 数据口径", "- data_warehouse/events/etf_state.parquet，真实观测日期，不用股票快照冒充ETF日频。"]
    return "\n".join(lines)


def generate(task: str) -> str:
    if task in ("zijin_weekly", "muyuan_weekly"):
        return _asset_report(task)
    if task == "pig_weekly":
        return _pig_report()
    if task == "china_internet_weekly":
        return _internet_report()
    raise KeyError(task)


def _fallback_path(task: str) -> Path:
    week = iso_week()
    return ROOT / "研究报告" / "周报" / week / f"{CONFIG[task]['title']}_{week}.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成硬核组合周报")
    parser.add_argument("--task", required=True, choices=sorted(CONFIG))
    parser.add_argument("--out")
    parser.add_argument("--no-deliver", action="store_true")
    args = parser.parse_args()
    report = generate(args.task)
    required = {
        "zijin_weekly": ("核心结论", "铜金周期", "下周行动卡", "数据口径"),
        "muyuan_weekly": ("核心结论", "猪价、成本", "下周行动卡", "数据口径"),
        "pig_weekly": ("周期判断", "情景与行动", "数据口径"),
        "china_internet_weekly": ("本周状态", "定投与风险纪律", "数据口径"),
    }[args.task]
    if not all(marker in report for marker in required):
        raise RuntimeError(f"周报缺少核心决策章节: {[marker for marker in required if marker not in report]}")
    if args.out:
        path = Path(args.out).expanduser()
    else:
        try:
            path = report_path(CONFIG[args.task]["report_key"])
        except OSError:
            path = _fallback_path(args.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    result = finalize_report(path, deliver=not args.no_deliver, channels=["feishu"],
                             title=CONFIG[args.task]["title"], body=report,
                             backup_youdao=not args.no_deliver)
    print(json.dumps({"ok": True, "task": args.task, "path": str(path), "bytes": path.stat().st_size,
                      "delivered": result.get("delivered", False)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
