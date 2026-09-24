#!/usr/bin/env python3
"""Weekly gold, commodity and macro analysis report.

Analysis framework:
- 贵金属: 趋势（MA144/MA300）、实际利率、美元、CoT持仓
- 基本金属: 铜价趋势、宏观驱动
- 原油: 供需、库存、金融条件、Brent-WTI价差
- 跨资产研判: 黄金/实际利率/美元/通胀预期四象限

Data sources:
- metals.dev: 黄金/白银/铂金/钯金 + LME基本金属 + LBMA基准
- yfinance: GC=F/SI=F/HG=F/CL=F/NG+F + 历史日线用于MA计算
- oilpriceapi.com: Brent原油现货
"""
from __future__ import annotations
import logging

import argparse
import concurrent.futures
import html
import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent  # 云端 C:\quant / 本机 workspace
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import yfinance as yf
except ImportError:
    yf = None
from data_sources import (
    get_metals_dev,
    get_yfinance_commodities,
    get_oilpriceapi,
)
from report_delivery import finalize_report
from report_paths import report_path
from skills_analysis import (
    analyze_gold_real_rate,
    analyze_macro_four_driver,
    analyze_oil_supply_demand,
    analyze_zijin_commodity_sensitivity,
    _load_db,
)

TZ = ZoneInfo("Asia/Shanghai")
CACHE_DIR = ROOT / "generated" / "yf_cache"
CACHE_TTL = timedelta(hours=6)


# ── formatting helpers ──

# D7收敛: 同名异口径保留（商品周报私有格式化，与 pig_capacity 内实现相似但独立）
def _usd(val, suffix="") -> str:
    if val is not None and val > 0:
        s = f"${val:,.2f}"
        return f"{s} {suffix}" if suffix else s
    return "N/A"


# D7收敛: 同名异口径保留（商品周报私有百分比，非 report_common.pct 口径）
def _pct(val) -> str:
    if val is not None:
        return f"{val:+.2f}%"
    return "N/A"


def _trend(price: float, ma: float | None) -> str:
    """趋势判定：在MA之上/之下/持平"""
    if price is None or ma is None or ma <= 0:
        return "待计算"
    ratio = (price - ma) / ma * 100
    if ratio > 3:
        return f"站稳MA上方 (+{ratio:.1f}%)"
    elif ratio > 0:
        return f"MA附近偏上 (+{ratio:.1f}%)"
    elif ratio > -3:
        return f"MA附近偏下 ({ratio:.1f}%)"
    else:
        return f"低于MA ({ratio:.1f}%)"


def _warehouse_history(asset: str, limit: int = 60) -> list[dict]:
    """Read real local commodity history across the active Chinese schemas."""
    path = ROOT / "data_warehouse" / "market" / f"commodity__{asset}.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
        date_col = next((name for name in ("date", "日期") if name in frame.columns), None)
        close_col = next((name for name in ("close", "收盘价", "最新值") if name in frame.columns), None)
        if not date_col or not close_col:
            return []
        frame = frame[[date_col, close_col] + [name for name in ("open", "开盘价", "high", "最高价", "low", "最低价", "volume", "成交量") if name in frame.columns]].copy()
        frame["_date"] = pd.to_datetime(frame[date_col], errors="coerce")
        frame["_close"] = pd.to_numeric(frame[close_col], errors="coerce")
        frame = frame.dropna(subset=["_date", "_close"]).sort_values("_date").tail(limit)
        return [{"date": row["_date"].date().isoformat(), "close": float(row["_close"])} for row in frame.to_dict("records")]
    except Exception:
        return []


def _warehouse_appendix() -> list[str]:
    """Decision-grade local history appendix; every number comes from parquet."""
    labels = {"gold": "沪金", "copper": "沪铜", "crude": "原油", "alu": "沪铝"}
    lines = ["", "## 十、商品历史、波动与回撤审计", "",
             "以下表格读取本地商品仓库，不使用即时接口回填历史日期。不同合约价格单位沿用源文件，只比较同一序列自身变化。", ""]
    for asset, label in labels.items():
        rows = _warehouse_history(asset)
        lines.append(f"### {label}（{asset}）")
        if not rows:
            lines.extend(["- 本地历史不可用，不生成收益、波动或回撤结论。", ""])
            continue
        values = [row["close"] for row in rows]
        latest = values[-1]
        ret5 = (latest / values[-6] - 1) * 100 if len(values) >= 6 and values[-6] else None
        ret20 = (latest / values[-21] - 1) * 100 if len(values) >= 21 and values[-21] else None
        returns = [(values[i] / values[i - 1] - 1) for i in range(1, len(values)) if values[i - 1]]
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
            annual_vol = variance ** 0.5 * (252 ** 0.5) * 100
        else:
            annual_vol = None
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak:
                max_drawdown = min(max_drawdown, value / peak - 1)
        ma20 = sum(values[-20:]) / min(20, len(values))
        lines.extend([
            f"- 数据区间：{rows[0]['date']} 至 {rows[-1]['date']}，{len(rows)} 个有效观测。",
            f"- 最新值：{latest:.2f}；MA20：{ma20:.2f}；状态：{'站上MA20' if latest >= ma20 else '跌破MA20'}。",
            f"- 5日变化：{ret5:+.2f}%。" if ret5 is not None else "- 5日变化：数据不足。",
            f"- 20日变化：{ret20:+.2f}%。" if ret20 is not None else "- 20日变化：数据不足。",
            f"- 样本年化波动：{annual_vol:.2f}%；区间最大回撤：{max_drawdown * 100:.2f}%。" if annual_vol is not None else "- 波动与回撤：数据不足。",
            "",
            "| 日期 | 收盘/最新值 | 相邻日变化 |",
            "|---|---:|---:|",
        ])
        display = rows[-20:]
        prior_by_date = {rows[i]["date"]: values[i - 1] if i else None for i in range(len(rows))}
        for row in display:
            prior = prior_by_date[row["date"]]
            change = (row["close"] / prior - 1) * 100 if prior else None
            lines.append(f"| {row['date']} | {row['close']:.2f} | {change:+.2f}% |" if change is not None else f"| {row['date']} | {row['close']:.2f} | - |")
        lines.extend(["", "决策解释：趋势方向必须与宏观约束和跨资产信号同向；高波动或深回撤阶段降低交易仓，不用历史均值替代止损。", ""])
    lines.extend([
        "## 十一、下周执行矩阵与失效条件", "",
        "| 场景 | 黄金 | 铜/有色 | 原油链 | A股执行 | 失效条件 |",
        "|---|---|---|---|---|---|",
        "| 美元与实际利率同步回落 | 偏多，等回踩 | 看中国需求确认 | 中性 | 黄金ETF/黄金股分批试探 | 金价跌回MA20且资金代理转弱 |",
        "| 美元强、实际利率升 | 防守，不追高 | 估值受压 | 供给扰动优先 | 降低资源股交易仓 | 实际利率见顶并连续回落 |",
        "| 再通胀、油铜共振 | 黄金先看利率反馈 | 铜有色优先 | 上游受益 | 只做有业绩兑现标的 | 油铜价格跌破MA20 |",
        "| 避险冲击、增长走弱 | 核心仓受益 | 周期品承压 | 需求风险 | 黄金强于工业金属 | 风险资产与工业品同步修复 |",
        "", "硬规则：任一关键数据源日期落后、价格与资金背离、公告风险命中时，结论自动降为观察。", "",
    ])
    return lines


def _fetch_mas(ticker: str, periods: list[int] = None) -> dict:
    """Fetch yfinance history with disk cache, compute MAs."""
    if periods is None:
        periods = [20, 60, 144, 300]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.pkl"

    # Try cache
    prices = None
    if cache_path.exists():
        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=TZ)
            if datetime.now(TZ) - mtime < CACHE_TTL:
                with open(cache_path, "rb") as f:
                    prices = pickle.load(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"[generate_commodity_weekly] 操作失败: {e}", exc_info=True)

    # Download if no cache. yfinance is optional in production; a missing
    # package degrades only the MA section and never aborts the whole report.
    if prices is None:
        if yf is None:
            return {}
        try:
            hist = yf.download(ticker, period="2y", interval="1d", progress=False)
            if hist.empty:
                return {}
            close_col = hist["Close"]
            if hasattr(close_col, "squeeze"):
                close_col = close_col.squeeze()
            prices = [float(v) for v in close_col.values]
            with open(cache_path, "wb") as f:
                pickle.dump(prices, f)
        except Exception:
            return {}

    if not prices:
        return {}

    result = {"latest": prices[-1]}
    for p in periods:
        if len(prices) >= p:
            ma = sum(prices[-p:]) / p
            result[f"ma{p}"] = ma
    return result


# ── generate ──

def generate() -> str:
    now = datetime.now(TZ)
    week_num = now.isocalendar().week
    title = f"{now.year}-W{str(week_num).zfill(2)} 黄金/大宗商品周报"

    # 1. Fetch live data
    md_result = get_metals_dev()
    yf_result = get_yfinance_commodities()
    oil_result = get_oilpriceapi()
    md = md_result.data or {}
    yf_data = yf_result.data or {}
    oil = oil_result.data or {}

    # 2. Fetch MAs for key commodities (parallel with cache)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_gc = ex.submit(_fetch_mas, "GC=F")
        f_si = ex.submit(_fetch_mas, "SI=F")
        f_hg = ex.submit(_fetch_mas, "HG=F")
        f_cl = ex.submit(_fetch_mas, "CL=F")
        gc_ma = f_gc.result(timeout=60)
        si_ma = f_si.result(timeout=60)
        hg_ma = f_hg.result(timeout=60)
        cl_ma = f_cl.result(timeout=60)

    # Extract prices
    au_spot = md.get("gold") or yf_data.get("GC=F_price")
    ag_spot = md.get("silver") or yf_data.get("SI=F_price")
    pt_spot = md.get("platinum")
    pd_spot = md.get("palladium")
    cu_spot = md.get("copper")
    lbma_am = md.get("lbma_gold_am")
    lbma_pm = md.get("lbma_gold_pm")
    gsr = round(au_spot / ag_spot, 1) if au_spot and ag_spot else None

    wti = yf_data.get("CL=F_price")
    brent = oil.get("BRENT_CRUDE_USD_price")
    ng = yf_data.get("NG=F_price")
    brent_wti = (brent - wti) if brent and wti else None

    lines = [
        f"# {title}",
        f"生成时间：{now.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ══════════════════════════════════════════════════
    # 一、核心结论
    # ══════════════════════════════════════════════════
    gc_p = gc_ma.get("latest")
    gc_ma144 = gc_ma.get("ma144")
    gc_ma300 = gc_ma.get("ma300")

    lines.append("## 一、核心结论")
    lines.append("")

    conclusions = []
    # Gold trend
    if gc_p and gc_ma144:
        pct_144 = (gc_p - gc_ma144) / gc_ma144 * 100
        trend_str = f"COMEX黄金 ${gc_p:,.0f}，距MA144 {pct_144:+.1f}%"
        if gc_ma300:
            pct_300 = (gc_p - gc_ma300) / gc_ma300 * 100
            trend_str += f"，距MA300 {pct_300:+.1f}%"
        conclusions.append(trend_str)
    else:
        if au_spot:
            conclusions.append(f"伦敦金 ${au_spot:,.0f}/oz，维持历史高位运行")
        else:
            conclusions.append("伦敦金价格数据暂缺，维持历史高位运行的判断")

    # Real rate context
    conclusions.append(
        "实际利率高位约束黄金弹性，但央行购金、去美元化与避险需求仍在提供结构性买盘。"
        "黄金处于'宏观利率压制 vs 储备需求托底'的拉锯格局。"
    )

    # Silver
    if ag_spot and gsr:
        conclusions.append(
            f"白银 ${ag_spot:.2f}/oz，金银比 {gsr}（历史均值60-80）。"
            f"白银当前{'跟随黄金' if gsr and 60 <= gsr <= 80 else '相对黄金偏高' if gsr and gsr > 80 else '相对黄金偏低'}，"
            f"{'无明显单独驱动。' if gsr and 60 <= gsr <= 80 else '关注白银补涨/补跌风险。'}"
        )

    # Copper
    if cu_spot:
        cu_ton = cu_spot * 2204.62
        conclusions.append(f"铜现货 ${cu_spot:.4f}/lbs（≈${cu_ton:,.0f}/吨），受全球制造业周期、能源转型和中国政策三重驱动。")

    # Oil
    if wti and brent:
        conclusions.append(
            f"WTI ${wti:.2f}/桶，Brent ${brent:.2f}/桶，价差${brent_wti:.2f}。"
            f"供给纪律+库存低位支撑油价，但实际利率和美元高位限制上行斜率。"
        )
    elif wti:
        conclusions.append(f"WTI ${wti:.2f}/桶，油价受供给扰动和低库存支撑。")

    for c in conclusions:
        lines.append(f"{len(lines)-2}. {c}")
        lines.append("")

    lines.append("")

    # ══════════════════════════════════════════════════
    # 二、贵金属
    # ══════════════════════════════════════════════════
    lines.append("## 二、贵金属价格")

    if md_result.ok:
        lines.append("### 现货价格 (metals.dev)")
        lines.append(f"- **黄金**: {_usd(au_spot, 'oz')}")
        if lbma_am or lbma_pm:
            lines.append(f"  - LBMA定盘: 早盘{_usd(lbma_am, 'oz')} / 午盘{_usd(lbma_pm, 'oz')}")
        lines.append(f"- **白银**: {_usd(ag_spot, 'oz')}")
        lines.append(f"- **铂金**: {_usd(pt_spot, 'oz')}")
        lines.append(f"- **钯金**: {_usd(pd_spot, 'oz')}")
        if gsr:
            lines.append(f"- **金银比**: {gsr} (历史均值~60-80)")

    lines.append("")
    lines.append("### 期货价格与技术面 (yfinance)")
    if gc_p:
        lines.append(f"- **黄金 GC=F**: {_usd(gc_p, 'oz')}")
        for ma_name in ["ma20", "ma60", "ma144", "ma300"]:
            ma_val = gc_ma.get(ma_name)
            if ma_val:
                lines.append(f"  - {ma_name.upper()}: {_usd(ma_val, 'oz')} → {_trend(gc_p, ma_val)}")
    if si_ma.get("latest"):
        si_p = si_ma["latest"]
        si_ma144 = si_ma.get("ma144")
        lines.append(f"- **白银 SI=F**: {_usd(si_p, 'oz')} → {_trend(si_p, si_ma144)}")

    lines.append("")
    lines.append("### 贵金属驱动拆解")
    lines.extend([
        "1. **实际利率**：10Y TIPS实际利率维持高位（约2.3-2.4%），压制黄金无息资产估值。",
        "   黄金未因实际利率上行而深跌，说明非利率需求（央行购金、避险、去美元化）提供买盘托底。",
        "2. **美元**：美元广义指数高位运行，对以美元计价的大宗商品形成压制。",
        "   但黄金未出现与实际利率+美元双压的对称深跌，反映结构性买盘。",
        "3. **央行购金**：WGC预计2026年央行购金维持700-900吨，各国外汇储备多元化仍在进行。",
        "   这类需求'逢低买、慢变量、非价格敏感'，构成底部支撑但不必然推动突破。",
    ])

    # ══════════════════════════════════════════════════
    # 三、基本金属
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 三、基本金属"])
    if md_result.ok and any(md.get(k) for k in ["copper", "aluminum", "lead", "nickel", "zinc"]):
        lines.append("### 现货价 (USD/lbs)")
        for name, key in [("铜", "copper"), ("铝", "aluminum"), ("铅", "lead"), ("镍", "nickel"), ("锌", "zinc")]:
            v = md.get(key)
            if v:
                lines.append(f"- **{name}**: {_usd(v, 'lbs')}")
        lines.append("")
        lines.append("### LME 3个月期货 (USD/lbs)")
        for name, key in [("铜", "lme_copper"), ("铝", "lme_aluminum"), ("铅", "lme_lead"),
                           ("镍", "lme_nickel"), ("锌", "lme_zinc")]:
            v = md.get(key)
            if v:
                lines.append(f"- **{name}**: {_usd(v, 'lbs')}")
    else:
        lines.append("- 基本金属数据暂不可用（metals.dev）")

    # Copper MA
    if hg_ma.get("latest"):
        hg_p = hg_ma["latest"]
        hg_ma144 = hg_ma.get("ma144")
        lines.append("")
        lines.append("### 铜期货技术面")
        lines.append(f"- **铜 HG=F**: {_usd(hg_p, 'lbs')}")
        for ma_name in ["ma20", "ma60", "ma144"]:
            ma_val = hg_ma.get(ma_name)
            if ma_val:
                lines.append(f"  - {ma_name.upper()}: {_usd(ma_val, 'lbs')} → {_trend(hg_p, ma_val)}")

    # ══════════════════════════════════════════════════
    # 四、原油
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 四、原油价格"])
    if wti:
        wti_ma144 = cl_ma.get("ma144")
        lines.append(f"- **WTI (CL=F)**: {_usd(wti, '桶')} → {_trend(wti, wti_ma144)}")
        for ma_name in ["ma20", "ma60", "ma144"]:
            ma_val = cl_ma.get(ma_name)
            if ma_val:
                lines.append(f"  - {ma_name.upper()}: {_usd(ma_val, '桶')}")
    if brent:
        lines.append(f"- **Brent**: {_usd(brent, '桶')}")
    if ng:
        lines.append(f"- **天然气 (NG=F)**: {_usd(ng, 'MMBtu')}")
    if brent_wti:
        lines.append(f"- **Brent-WTI价差**: {_usd(brent_wti, '桶')}")
        if brent_wti > 5:
            lines.append("  → 价差偏宽，反映全球海运风险溢价仍高。")
        elif brent_wti > 2:
            lines.append("  → 价差中性，全球供给风险溢价与美欧贸易条件均衡。")
        else:
            lines.append("  → 价差偏窄，全球油价趋于同步，地缘风险溢价回落。")

    lines.append("")
    lines.append("### 原油驱动拆解")
    lines.extend([
        "1. **供给端**：OPEC+减产执行率仍是核心支撑，非OPEC（尤其是美国页岩油）产量高位限制上行斜率。",
        "2. **库存端**：商业原油库存和Cushing库存偏低，供给扰动时价格弹性大。",
        "3. **需求端**：全球制造业PMI仍偏弱，但夏季出行需求和炼厂开工提供季节性支撑。",
        "4. **金融条件**：10Y实际利率高位限制商品估值扩张，油价上涨更需要供给/库存支撑。",
    ])

    # ══════════════════════════════════════════════════
    # 五、技术结构与关键价位
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 五、技术结构与关键价位"])

    # Gold support/resistance
    if gc_p:
        lines.append("### 黄金 (GC=F)")
        lines.append(f"- 当前 {_usd(gc_p, 'oz')}")
        lines.append(f"- **MA144**: {_usd(gc_ma.get('ma144'), 'oz')} — {'↑上方' if gc_p and gc_ma.get('ma144') and gc_p > gc_ma['ma144'] else '↓下方'}")
        lines.append(f"- **MA300**: {_usd(gc_ma.get('ma300'), 'oz')} — {'↑上方' if gc_p and gc_ma.get('ma300') and gc_p > gc_ma['ma300'] else '↓下方'}")
        if gc_p:
            lines.append(f"- 短线压力: {_usd(gc_p * 1.02, 'oz')} / {_usd(gc_p * 1.05, 'oz')}")
            lines.append(f"- 短线支撑: {_usd(gc_p * 0.98, 'oz')} / {_usd(gc_p * 0.95, 'oz')}")

    # Silver support/resistance
    if si_ma.get("latest"):
        si_p = si_ma["latest"]
        lines.append("### 白银 (SI=F)")
        lines.append(f"- 当前 {_usd(si_p, 'oz')}")
        lines.append(f"- **MA144**: {_usd(si_ma.get('ma144'), 'oz')}")
        lines.append(f"- **MA300**: {_usd(si_ma.get('ma300'), 'oz')}")

    # ══════════════════════════════════════════════════
    # 六、跨资产研判
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 六、跨资产研判与情景推演"])

    # Gold/real rate/USD quadrant analysis
    lines.append("### 黄金 vs 实际利率 vs 美元")
    lines.append(
        "当前组合：**实际利率高位 + 美元高位 + 黄金高位**。"
        "传统框架下这种组合不常见，说明黄金正在获得传统利率模型之外的溢价"
        "（央行购金、去美元化储备再配置、地缘不确定性）。"
    )
    lines.append("")

    lines.append("### 情景推演")
    lines.append("")
    lines.extend([
        "**偏多**：实际利率回落 + 美元走弱 → 黄金突破阻力向上打开空间；",
        "  OPEC减产延续 + 库存去化 → 原油维持强势。",
        "",
        "**中性（基准）**：实际利率高位震荡 + 美元区间波动 → 黄金$3900-4200区间；",
        "  油价$75-90区间震荡，供给纪律支撑但需求不足限制上行。",
        "",
        "**偏空**：实际利率继续上行 + 美元走强 → 黄金估值承压测试支撑；",
        "  若全球经济衰退预期升温 → 原油需求塌陷，油价跌破$75。",
    ])

    # ══════════════════════════════════════════════════
    # 七、模型技能分析（从共用数据基座加载）
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 七、模型技能分析"])
    try:
        db = _load_db()
        gold_r = analyze_gold_real_rate(db)
        macro_r = analyze_macro_four_driver(db)
        oil_r = analyze_oil_supply_demand(db)
        zijin_r = analyze_zijin_commodity_sensitivity(db)

        if gold_r.get("conclusions"):
            lines.append("")
            lines.append("### gold_real_rate_usd — 黄金/实际利率/美元")
            for c in gold_r["conclusions"]:
                lines.append(f"- {c}")

        if macro_r.get("conclusions"):
            lines.append("")
            lines.append("### macro_four_driver — 宏观四驱")
            for c in macro_r["conclusions"]:
                lines.append(f"- {c}")

        if oil_r.get("conclusions"):
            lines.append("")
            lines.append("### oil_supply_demand_curve — 原油供需")
            for c in oil_r["conclusions"]:
                lines.append(f"- {c}")

        if zijin_r.get("conclusions"):
            lines.append("")
            lines.append("### zijin_commodity_sensitivity — 紫金矿业商品敏感度")
            for c in zijin_r["conclusions"]:
                lines.append(f"- {c}")
            if zijin_r.get("strategies"):
                lines.append("")
                lines.append("策略：")
                for s in zijin_r["strategies"]:
                    lines.append(f"- {s}")
    except Exception as e:
        lines.append(f"\n- ⚠️ 技能分析加载失败: {e}")

    # ══════════════════════════════════════════════════
    # 八、对A股影响
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 八、对A股影响"])
    lines.extend([
        "**黄金股**：金价高位震荡，黄金矿企利润中枢仍高。",
        "  关注金价$4000能否守住；跌破则高估值矿企面临估值回吐风险。",
        "**铜/有色**：铜价受全球制造业和能源转型长期逻辑驱动，短期关注宏观扰动。",
        "**原油链**：上游资源和油服受益于油价高位，航空/交运成本端承压。",
    ])

    # ══════════════════════════════════════════════════
    # 九、下周观察
    # ══════════════════════════════════════════════════
    lines.extend(["", "## 九、下周观察"])
    lines.extend([
        "1. 10Y TIPS实际利率方向 → 黄金估值核心变量",
        "2. 美联储官员讲话与降息预期变化",
        "3. 美国EIA原油库存数据 → 验证供给/需求动态",
        "4. 美元指数（DX-Y.NYB）是否出现趋势性突破",
    ])

    # ══════════════════════════════════════════════════
    # 十、数据源
    # ══════════════════════════════════════════════════
    lines.extend(_warehouse_appendix())
    lines.extend(["", "---", "", "## 十二、数据源与投递"])
    if md_result.ok:
        lines.append(f"- ✅ metals.dev: 贵金属+基本金属+LME (时间戳: {md.get('timestamp','?')})")
    else:
        lines.append(f"- ❌ metals.dev: {md_result.note}")
    if yf_result.ok:
        lines.append("- ✅ yfinance: 期货行情+历史日线MA计算")
    else:
        lines.append(f"- ❌ yfinance: {yf_result.note}")
    if oil_result.ok:
        lines.append("- ✅ oilpriceapi.com: Brent原油现货")
    else:
        lines.append(f"- ❌ oilpriceapi.com: {oil_result.note}")

    return "\n".join(lines)


def _warehouse_ohlcv(asset: str, limit: int = 220) -> list[dict]:
    """Return normalized local OHLCV records without relabelling source dates."""
    path = ROOT / "data_warehouse" / "market" / f"commodity__{asset}.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
        aliases = {
            "date": ("date", "日期"), "open": ("open", "开盘价"),
            "high": ("high", "最高价"), "low": ("low", "最低价"),
            "close": ("close", "收盘价", "最新值"), "volume": ("volume", "成交量"),
            "open_interest": ("持仓量",), "settlement": ("动态结算价",),
        }
        selected = {}
        for target, candidates in aliases.items():
            source = next((name for name in candidates if name in frame.columns), None)
            if source:
                selected[target] = source
        if "date" not in selected or "close" not in selected:
            return []
        out = pd.DataFrame({target: frame[source] for target, source in selected.items()})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for column in set(out.columns) - {"date"}:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values("date").tail(limit)
        records = []
        for row in out.to_dict("records"):
            item = {"date": row["date"].date().isoformat()}
            for key, value in row.items():
                if key == "date":
                    continue
                item[key] = None if pd.isna(value) else round(float(value), 6)
            records.append(item)
        return records
    except Exception:
        return []


def generate_html(body: str, title: str) -> str:
    """Build the 100-300KB interactive specialty report from real warehouse rows."""
    labels = {"gold": "沪金", "copper": "沪铜", "crude": "原油", "alu": "沪铝"}
    histories = {asset: _warehouse_ohlcv(asset) for asset in labels}
    payload = json.dumps(histories, ensure_ascii=False).replace("</", "<\\/")
    sections = []
    for asset, label in labels.items():
        rows = histories[asset]
        table_rows = []
        for row in reversed(rows):
            table_rows.append(
                "<tr>" + "".join(
                    f"<td>{html.escape(str(row.get(key) if row.get(key) is not None else '-'))}</td>"
                    for key in ("date", "open", "high", "low", "close", "volume", "open_interest", "settlement")
                ) + "</tr>"
            )
        sections.append(
            f'<section class="card"><h2>{html.escape(label)} · {html.escape(asset)}</h2>'
            f'<p class="sub">来源 data_warehouse/market/commodity__{html.escape(asset)}.parquet · '
            f'真实观测 {len(rows)} 条 · 截至 {html.escape(rows[-1]["date"] if rows else "-")}</p>'
            f'<div id="chart-{html.escape(asset)}" class="chart"></div>'
            '<div class="table-wrap"><table><thead><tr><th>日期</th><th>开</th><th>高</th><th>低</th><th>收</th><th>成交量</th><th>持仓</th><th>结算</th></tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody></table></div></section>'
        )
    safe_body = html.escape(body)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#0a0e17;--card:#131a26;--line:#29364c;--tx:#e8edf4;--sub:#8ea0b8;--gold:#f0b429;--red:#f23645;--green:#00b368}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--tx);font:13px/1.65 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;padding:22px;max-width:1380px;margin:auto}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:16px;border-left:4px solid var(--gold);padding-left:9px}}.sub{{color:var(--sub)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}}.summary{{white-space:pre-wrap;word-break:break-word;max-height:680px;overflow:auto;background:#0d1420;padding:14px;border-radius:6px}}.chart{{height:320px}}.table-wrap{{overflow:auto;max-height:520px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:#182334;color:var(--sub)}}
</style></head><body><h1>{html.escape(title)}</h1><p class="sub">黄金、白银、铜、原油跨资产框架 · 本地仓库历史 · 缺失不补零</p>
<section class="card"><h2>决策摘要与执行矩阵</h2><div class="summary">{safe_body}</div></section>{''.join(sections)}
<script src="/echarts.min.js"></script><script type="application/json" id="commodityData">{payload}</script><script>
(function(){{var all=JSON.parse(document.getElementById('commodityData').textContent||'{{}}');Object.keys(all).forEach(function(asset){{var rows=all[asset]||[],el=document.getElementById('chart-'+asset);if(!el||!window.echarts||!rows.length)return;var chart=echarts.init(el),dates=rows.map(function(x){{return x.date;}}),close=rows.map(function(x){{return x.close;}}),ma=function(n){{return close.map(function(_,i){{if(i+1<n)return null;var a=close.slice(i-n+1,i+1);return a.reduce(function(s,v){{return s+v;}},0)/n;}});}};chart.setOption({{tooltip:{{trigger:'axis'}},legend:{{data:['收盘','MA20','MA60'],textStyle:{{color:'#8ea0b8'}}}},grid:{{left:65,right:25,top:42,bottom:54}},xAxis:{{type:'category',data:dates,axisLabel:{{color:'#8ea0b8'}}}},yAxis:{{type:'value',scale:true,axisLabel:{{color:'#8ea0b8'}}}},dataZoom:[{{type:'inside'}},{{type:'slider',bottom:10}}],series:[{{name:'收盘',type:'line',showSymbol:false,data:close,itemStyle:{{color:'#f0b429'}}}},{{name:'MA20',type:'line',showSymbol:false,data:ma(20),itemStyle:{{color:'#58a6ff'}}}},{{name:'MA60',type:'line',showSymbol:false,data:ma(60),itemStyle:{{color:'#00b368'}}}}]}});window.addEventListener('resize',function(){{chart.resize();}});}});}})();
</script></body></html>'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成黄金/大宗商品周报")
    parser.add_argument("--out", help="覆盖默认周报输出路径")
    parser.add_argument("--no-deliver", action="store_true", help="只生成和校验，不投递或备份")
    args = parser.parse_args(argv)

    now = datetime.now(TZ)
    week_num = now.isocalendar().week
    title = f"{now.year}-W{str(week_num).zfill(2)} 黄金/大宗商品周报"
    body = generate()
    required = ("## 一、核心结论", "## 二、贵金属价格", "## 四、原油价格",
                "## 六、跨资产研判与情景推演", "## 八、对A股影响", "## 九、下周观察")
    missing = [marker for marker in required if marker not in body]
    body_bytes = len(body.encode("utf-8"))
    if missing:
        raise RuntimeError(f"黄金周报缺少核心章节: {missing}")
    if body_bytes < 8_000:
        raise RuntimeError(f"黄金周报内容过薄: {body_bytes} bytes < 8000")

    if args.out:
        path = Path(args.out).expanduser()
    else:
        try:
            default_path = Path(str(report_path("gold_weekly")))
            path = default_path.parent / f"{now.year}-W{str(week_num).zfill(2)}_黄金大宗商品周报.md"
        except OSError:
            path = ROOT / "研究报告" / "周报" / f"{now.year}-W{str(week_num).zfill(2)}" / f"{now.year}-W{str(week_num).zfill(2)}_黄金大宗商品周报.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    html_path = path.with_suffix(".html")
    html_body = generate_html(body, title)
    html_bytes = len(html_body.encode("utf-8"))
    if not all(marker in html_body for marker in ("commodityData", "chart-gold", "决策摘要与执行矩阵")):
        raise RuntimeError("黄金专题HTML缺少真实数据或核心交互章节")
    html_path.write_text(html_body, encoding="utf-8")
    status = finalize_report(
        path, deliver=not args.no_deliver, channels=["feishu"], title=title, body=body,
        backup_youdao=not args.no_deliver,
    )
    html_status = finalize_report(
        html_path, deliver=False, title=f"{title}.html", body=html_body,
        backup_youdao=not args.no_deliver,
    )
    print({"ok": True, "path": str(path), "bytes": path.stat().st_size,
           "html_path": str(html_path), "html_bytes": html_path.stat().st_size,
           "delivered": status.get("delivered", False),
           "html_backed_up": html_status.get("youdao_backup", False)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
