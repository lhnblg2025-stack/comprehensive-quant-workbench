#!/usr/bin/env python3
"""Skills analysis engine — applies financial models to report_data_base data.

Reads from generated/report_data_base/latest.json, runs each applicable model,
and writes structured analysis + conclusions back to the data base.

Output buckets:
  - skills_analysis/: per-model analysis JSON
  - skills_analysis_latest/: per-ticker condensed signals for A-share assets
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_data_base import collect_report_data_base, _write_json, BASE_DIR

TZ = ZoneInfo("Asia/Shanghai")
SKILLS_DIR = BASE_DIR / "skills_analysis"
SIGNALS_DIR = BASE_DIR / "skills_signals"
LATEST_SKILLS = BASE_DIR / "skills_latest.json"
LATEST_SIGNALS = BASE_DIR / "skills_signals_latest.json"

SKILLS_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Loader ──

def _load_db() -> dict:
    """Load latest report_data_base."""
    try:
        return collect_report_data_base(update=False)
    except Exception:
        return {"data": {}, "snapshot": {}}


def _get(db: dict, *keys, default=None):
    """Safe nested get."""
    d = db
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return default
    return d if d else default


def _f(item: dict, key: str):
    """Get feature from item."""
    return item.get("features", {}).get(key) if item else None


# ── Model Analysis Functions ──

def analyze_gold_real_rate(db: dict) -> dict:
    """gold_real_rate_usd model: gold vs real rates vs USD."""
    gc = _get(db, "data", "yahoo_assets", "GC=F")
    si = _get(db, "data", "yahoo_assets", "SI=F")
    us10y = _get(db, "data", "fred", "us10y")
    dx = _get(db, "data", "yahoo_assets", "DX-Y.NYB")
    cl = _get(db, "data", "yahoo_assets", "CL=F")
    cpi = _get(db, "data", "macro", "cpi_ppi")

    gold_p = _f(gc, "latest")
    gold_ma144 = _f(gc, "ma144")
    gold_ma300 = _f(gc, "ma300")
    gold_20d = _f(gc, "return_20d_pct")
    gold_60d = _f(gc, "return_60d_pct")
    silver_p = _f(si, "latest")
    us10y_val = _f(us10y, "latest")
    cl_val = _f(cl, "latest")

    # Build analysis
    signals = {}
    conclusions = []

    # Trend
    if gold_p and gold_ma144:
        pct_144 = (gold_p - gold_ma144) / gold_ma144 * 100
        signals["gold_vs_ma144_pct"] = round(pct_144, 2)
        signals["gold_trend"] = "bearish" if pct_144 < -5 else "neutral" if pct_144 < 5 else "bullish"
        conclusions.append(
            f"黄金 ${gold_p:,.0f}/oz，距MA144 {pct_144:+.1f}%（{'偏空' if pct_144 < -3 else '中性' if pct_144 < 5 else '偏多'}）。"
            f"60日涨跌 {gold_60d:+.1f}%，中期趋势偏弱。"
        )

    if gold_p and gold_ma300:
        pct_300 = (gold_p - gold_ma300) / gold_ma300 * 100
        signals["gold_vs_ma300_pct"] = round(pct_300, 2)
        conclusions.append(
            f"距MA300 {pct_300:+.1f}%，长期趋势{'偏多' if pct_300 > 0 else '偏空'}。"
        )

    # Real rate
    if us10y_val:
        real_rate_est = us10y_val - 2.3  # approximate TIPS using 2.3% inflation expectation
        signals["real_rate_est"] = round(real_rate_est, 2)
        if real_rate_est > 2.0:
            rr_signal = "压制"
        elif real_rate_est > 1.5:
            rr_signal = "偏压制"
        else:
            rr_signal = "中性偏多"
        signals["real_rate_signal"] = rr_signal
        conclusions.append(
            f"10Y名义利率 {us10y_val:.2f}%，估算实际利率约{real_rate_est:.2f}%，{rr_signal}黄金估值。"
        )

    # Silver
    if silver_p and gold_p:
        gsr = gold_p / silver_p
        signals["gold_silver_ratio"] = round(gsr, 1)
        if gsr > 85:
            gsr_signal = "白银相对低估，关注补涨"
        elif gsr < 60:
            gsr_signal = "白银相对高估"
        else:
            gsr_signal = "中性区间"
        signals["gsr_signal"] = gsr_signal
        conclusions.append(f"金银比 {gsr:.1f}（{gsr_signal}）。")

    # Multi-factor conclusion
    if gold_p and gold_ma144 and gold_ma300:
        # Combined assessment
        if gold_p < gold_ma144 and gold_p < gold_ma300:
            overall = "黄金短期中期均偏弱，处于MA144和MA300下方，趋势承压。关注$4,000支撑。"
        elif gold_p < gold_ma144 and gold_p > gold_ma300:
            overall = "黄金短期回调但长期趋势仍在，MA300提供支撑。关注MA144能否收复。"
        else:
            overall = "黄金站上MA144和MA300，多头趋势完好。"
        signals["overall"] = overall
        conclusions.append(overall)

    return {
        "model": "gold_real_rate_usd",
        "timestamp": datetime.now(TZ).isoformat(),
        "assets": ["GC=F", "SI=F", "DX-Y.NYB"],
        "raw": {"gold_price": gold_p, "silver_price": silver_p, "us10y": us10y_val},
        "signals": signals,
        "conclusions": conclusions,
        "strategies": [
            f"黄金趋势偏空期间，关注A股黄金股（紫金矿业）的贝塔风险。",
            f"若金价守住$4,000并收复MA144，可重新评估黄金资产配置。",
            f"实际利率若回落至2.0%以下，黄金将获得更清晰的宏观多头组合。",
        ] if signals.get("gold_trend") in ("bearish", "neutral") else [
            f"黄金多头趋势中，A股黄金矿企盈利中枢上移。",
            f"关注金价能否站稳MA300上方。",
        ],
    }


def analyze_macro_four_driver(db: dict) -> dict:
    """macro_four_driver: growth, inflation, monetary, external."""
    cpi = _get(db, "data", "macro", "cpi_ppi")
    sf = _get(db, "data", "macro", "social_finance_m2")
    us10y = _get(db, "data", "fred", "us10y")
    fed = _get(db, "data", "fred", "fed_funds")

    conclusions = []

    # Growth (proxy: 社融)
    sf_p = sf.get("features", {}) if sf else {}
    sf_yoy = sf_p.get("sf_yoy")
    if sf_yoy:
        conclusions.append(f"社融存量同比 {sf_yoy:.1f}%，增长动能{'偏强' if sf_yoy > 9 else '中性' if sf_yoy > 8 else '偏弱'}。")

    # Inflation
    cpi_p = cpi.get("features", {}) if cpi else {}
    cpi_yoy = cpi_p.get("cpi_yoy")
    ppi_yoy = cpi_p.get("ppi_yoy")
    if cpi_yoy:
        conclusions.append(f"CPI同比 {cpi_yoy:.1f}%，PPI同比 {ppi_yoy:.1f}%，通胀{'温和偏上' if cpi_yoy > 1.5 else '温和'}。")

    # Monetary
    us10y_val = _f(us10y, "latest")
    fed_val = _f(fed, "latest")
    if us10y_val and fed_val:
        spread = us10y_val - fed_val
        conclusions.append(f"10Y-FF利差 {spread:.2f}%，{'倒挂' if spread < 0 else '正常陡峭'}。")
        conclusions.append(f"利率环境对周期资产{'偏压制' if us10y_val > 4.5 else '中性'}。")

    return {
        "model": "macro_four_driver",
        "timestamp": datetime.now(TZ).isoformat(),
        "conclusions": conclusions,
        "signals": {
            "us10y": us10y_val,
            "fed_funds": fed_val,
            "growth_signal": "待社融数据更新",
        },
    }


def analyze_oil_supply_demand(db: dict) -> dict:
    """oil_supply_demand_curve: oil prices, inventory, macro."""
    cl = _get(db, "data", "yahoo_assets", "CL=F")
    brent = _get(db, "data", "yahoo_assets", "BZ=F")

    cl_p = _f(cl, "latest")
    cl_ma144 = _f(cl, "ma144")
    cl_20d = _f(cl, "return_20d_pct")
    br_p = _f(brent, "latest")

    conclusions = []
    signals = {}

    if cl_p:
        conclusions.append(f"WTI ${cl_p:.2f}/桶")
        if cl_ma144:
            pct = (cl_p - cl_ma144) / cl_ma144 * 100
            signals["wti_vs_ma144_pct"] = round(pct, 2)
            conclusions.append(f"  MA144 ${cl_ma144:.2f}，{pct:+.1f}%{'偏强' if pct > 5 else '中性' if pct > -5 else '偏弱'}。")
        if cl_20d:
            conclusions.append(f"  20日涨跌 {cl_20d:+.1f}%，短期动能{'偏强' if cl_20d > 5 else '中性'}。")

    if br_p:
        spread = (br_p - cl_p) if (br_p and cl_p) else None
        if spread:
            signals["brent_wti_spread"] = round(spread, 2)
            conclusions.append(f"Brent ${br_p:.2f}/桶，Brent-WTI价差 ${spread:.2f}。")

    return {
        "model": "oil_supply_demand_curve",
        "timestamp": datetime.now(TZ).isoformat(),
        "assets": ["CL=F", "BZ=F"],
        "conclusions": conclusions,
        "signals": signals,
    }


def analyze_zijin_commodity_sensitivity(db: dict) -> dict:
    """zijin_commodity_sensitivity: 紫金矿业 sensitivity to gold/copper."""
    zijin = _get(db, "data", "a_share_stocks", "601899.SH")
    gc = _get(db, "data", "yahoo_assets", "GC=F")
    hg = _get(db, "data", "yahoo_assets", "HG=F")

    zijin_p = _f(zijin, "latest")
    zijin_ma144 = _f(zijin, "ma144")
    gold_p = _f(gc, "latest")
    copper_p = _f(hg, "latest")

    conclusions = []
    signals = {}

    if zijin_p:
        conclusions.append(f"紫金矿业 ¥{zijin_p:.2f}")
        if zijin_ma144:
            pct = (zijin_p - zijin_ma144) / zijin_ma144 * 100
            signals["zijin_vs_ma144_pct"] = round(pct, 2)
            conclusions.append(f"  MA144 ¥{zijin_ma144:.2f}，{pct:+.1f}%{'偏多' if pct > 0 else '偏空'}。")

    if gold_p and copper_p:
        # Commodity sensitivity proxy
        conclusions.append(f"  - 黄金 ${gold_p:,.0f}/oz（核心驱动）")
        conclusions.append(f"  - 铜 ${copper_p:.2f}/lbs（第二驱动）")

    signals["gold_copper_ratio"] = round(gold_p / copper_p, 0) if gold_p and copper_p else None

    return {
        "model": "zijin_commodity_sensitivity",
        "timestamp": datetime.now(TZ).isoformat(),
        "assets": ["601899.SH", "GC=F", "HG=F"],
        "conclusions": conclusions,
        "signals": signals,
        "strategies": [
            "紫金矿业同时受黄金和铜价驱动，黄金占比更高。",
            "金价站上MA144前，紫金矿业估值承压。",
            "关注紫金矿业A/H价差，若H股折价扩大可能是买入信号。",
        ],
    }


def analyze_hog_cycle(db: dict) -> dict:
    """hog_capacity_cycle: pig cycle from available data."""
    muyuan = _get(db, "data", "a_share_stocks", "002714.SZ")

    my_p = _f(muyuan, "latest")
    my_ma144 = _f(muyuan, "ma144")

    conclusions = []
    signals = {}

    if my_p:
        conclusions.append(f"牧原股份 ¥{my_p:.2f}")
        if my_ma144:
            pct = (my_p - my_ma144) / my_ma144 * 100
            signals["muyuan_vs_ma144_pct"] = round(pct, 2)
            conclusions.append(f"  MA144 ¥{my_ma144:.2f}，{pct:+.1f}%。")

    conclusions.extend([
        "猪周期核心判断需能繁母猪/仔猪/猪粮比数据（待数据源接入）。",
        "饲料成本（玉米/大豆）高位运行加速产能去化。",
    ])

    return {
        "model": "hog_capacity_cycle",
        "timestamp": datetime.now(TZ).isoformat(),
        "assets": ["002714.SZ"],
        "conclusions": conclusions,
        "signals": signals,
    }


# ── Master Analysis ──

MODELS = {
    "gold_real_rate_usd": analyze_gold_real_rate,
    "macro_four_driver": analyze_macro_four_driver,
    "oil_supply_demand_curve": analyze_oil_supply_demand,
    "zijin_commodity_sensitivity": analyze_zijin_commodity_sensitivity,
    "hog_capacity_cycle": analyze_hog_cycle,
}


def run_all_models(db: dict = None) -> dict:
    """Run all analysis models and return combined output."""
    if db is None:
        db = _load_db()
    results = {}
    for name, fn in MODELS.items():
        try:
            results[name] = fn(db)
        except Exception as e:
            results[name] = {"model": name, "error": str(e), "conclusions": []}
    return results


def generate_signals(results: dict) -> dict:
    """Extract condensed signals for A-share commodity assets."""
    signals = {}

    # Gold signals
    gold = results.get("gold_real_rate_usd", {})
    if gold.get("signals"):
        signals["gold"] = gold["signals"]
        signals["gold"]["conclusions"] = gold.get("conclusions", [])

    # Oil signals
    oil = results.get("oil_supply_demand_curve", {})
    if oil.get("signals"):
        signals["oil"] = oil["signals"]
        signals["oil"]["conclusions"] = oil.get("conclusions", [])

    # 紫金矿业
    zijin = results.get("zijin_commodity_sensitivity", {})
    if zijin.get("signals"):
        signals["zijin_601899"] = zijin["signals"]
        signals["zijin_601899"]["conclusions"] = zijin.get("conclusions", [])

    # 牧原
    hog = results.get("hog_capacity_cycle", {})
    if hog.get("signals"):
        signals["muyuan_002714"] = hog["signals"]
        signals["muyuan_002714"]["conclusions"] = hog.get("conclusions", [])

    # Macro context
    macro = results.get("macro_four_driver", {})
    if macro.get("conclusions"):
        signals["macro_context"] = macro["conclusions"]

    return signals


def save_results(results: dict, signals: dict) -> dict:
    """Save analysis to disk."""
    now = datetime.now(TZ)
    date_str = now.strftime("%Y%m%d")

    # Save per-model analysis
    for name, data in results.items():
        path = SKILLS_DIR / f"{name}_{date_str}.json"
        _write_json(path, data)

    # Save overall results
    combined = {
        "generated_at": now.isoformat(),
        "model_count": len(results),
        "models": {k: {"conclusions": v.get("conclusions", [])} for k, v in results.items() if v.get("conclusions")},
    }
    _write_json(LATEST_SKILLS, combined)

    # Save signals
    sig_data = {
        "generated_at": now.isoformat(),
        "signals": signals,
    }
    _write_json(SIGNALS_DIR / f"signals_{date_str}.json", sig_data)
    _write_json(LATEST_SIGNALS, sig_data)

    return {
        "skills_path": str(LATEST_SKILLS),
        "signals_path": str(LATEST_SIGNALS),
        "models_run": list(results.keys()),
        "models_ok": sum(1 for v in results.values() if "error" not in v),
    }


def generate_report_md(results: dict, signals: dict) -> str:
    """Generate a detailed markdown analysis report."""
    now = datetime.now(TZ)
    lines = [
        f"# 共用数据基座技能分析报告",
        f"生成时间：{now.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ═══════════════════════════════════════
    # 综合结论
    # ═══════════════════════════════════════
    lines.append("## 一、综合结论")
    lines.append("")

    gold = results.get("gold_real_rate_usd", {})
    if gold.get("conclusions"):
        for c in gold["conclusions"]:
            lines.append(f"- {c}")

    macro = results.get("macro_four_driver", {})
    if macro.get("conclusions"):
        lines.append("")
        lines.append("### 宏观环境")
        for c in macro["conclusions"]:
            lines.append(f"- {c}")

    oil = results.get("oil_supply_demand_curve", {})
    if oil.get("conclusions"):
        lines.append("")
        lines.append("### 原油")
        for c in oil["conclusions"]:
            lines.append(f"- {c}")

    # ═══════════════════════════════════════
    # A股大宗资产信号
    # ═══════════════════════════════════════
    lines.extend(["", "## 二、A股大宗资产信号"])

    zijin = results.get("zijin_commodity_sensitivity", {})
    if zijin.get("conclusions"):
        lines.append("")
        lines.append("### 紫金矿业 (601899)")
        for c in zijin["conclusions"]:
            lines.append(f"- {c}")
        if zijin.get("strategies"):
            lines.append("")
            lines.append("策略：")
            for s in zijin["strategies"]:
                lines.append(f"- {s}")

    hog = results.get("hog_capacity_cycle", {})
    if hog.get("conclusions"):
        lines.append("")
        lines.append("### 牧原股份 (002714) / 猪周期")
        for c in hog["conclusions"]:
            lines.append(f"- {c}")

    # ═══════════════════════════════════════
    # 关键信号信号汇总
    # ═══════════════════════════════════════
    lines.extend(["", "## 三、关键信号"])
    lines.append("")
    lines.append("| 资产 | 信号 | 值 | 含义 |")
    lines.append("|------|------|-----|------|")

    sig_rows = []

    # Gold signals
    gs = signals.get("gold", {})
    if gs.get("gold_trend"):
        sig_rows.append(("黄金", "趋势", gs["gold_trend"], {"bearish": "偏空", "neutral": "中性", "bullish": "偏多"}.get(gs["gold_trend"], "")))
    if gs.get("real_rate_signal"):
        sig_rows.append(("黄金", "实际利率压制", "是" if "压制" in gs["real_rate_signal"] else "否", gs["real_rate_signal"]))
    if gs.get("gold_silver_ratio"):
        sig_rows.append(("白银", "金银比", f'{gs["gold_silver_ratio"]:.1f}', gs.get("gsr_signal", "")))

    # 紫金
    zs = signals.get("zijin_601899", {})
    if zs.get("zijin_vs_ma144_pct"):
        val = zs["zijin_vs_ma144_pct"]
        sig_rows.append(("紫金矿业", "距MA144", f"{val:+.1f}%", "偏多" if val > 0 else "偏空"))

    # Oil
    os = signals.get("oil", {})
    if os.get("wti_vs_ma144_pct"):
        val = os["wti_vs_ma144_pct"]
        sig_rows.append(("WTI原油", "距MA144", f"{val:+.1f}%", "偏强" if val > 5 else "中性" if val > -5 else "偏弱"))

    # 牧原
    ms = signals.get("muyuan_002714", {})
    if ms.get("muyuan_vs_ma144_pct"):
        val = ms["muyuan_vs_ma144_pct"]
        sig_rows.append(("牧原股份", "距MA144", f"{val:+.1f}%", "待周期确认" if val < 0 else "偏多"))

    for asset, indicator, value, meaning in sig_rows:
        lines.append(f"| {asset} | {indicator} | {value} | {meaning} |")

    # ═══════════════════════════════════════
    # 模型元数据
    # ═══════════════════════════════════════
    lines.extend(["", "## 四、模型与技能"])
    lines.append("")
    model_meta = {
        "gold_real_rate_usd": "黄金/实际利率/美元四象限分析",
        "macro_four_driver": "增长/通胀/货币/外部宏观四驱分析",
        "oil_supply_demand_curve": "原油供需/库存/价差分析",
        "zijin_commodity_sensitivity": "紫金矿业商品敏感度分析",
        "hog_capacity_cycle": "猪周期/养殖产能分析",
    }
    for mname, mdesc in model_meta.items():
        mr = results.get(mname, {})
        status = "✅" if "error" not in mr else "❌"
        lines.append(f"- {status} **{mname}**: {mdesc}")

    return "\n".join(lines)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="update report_data_base first")
    parser.add_argument("--report", action="store_true", help="print markdown report to stdout")
    parser.add_argument("--save", action="store_true", help="save analysis to data base")
    args = parser.parse_args()

    db = collect_report_data_base(update=args.update, light=not args.update)
    results = run_all_models(db)
    signals = generate_signals(results)

    if args.save:
        saved = save_results(results, signals)
        print(f"Saved: {saved['skills_path']}")
        print(f"Signals: {saved['signals_path']}")
        print(f"Models: {saved['models_ok']}/{saved['models_run']} OK")

    if args.report:
        md = generate_report_md(results, signals)
        print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
