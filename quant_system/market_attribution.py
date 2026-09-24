"""
market_attribution.py - Daily Market Factor Attribution Engine

Analyzes A-share market meta JSON data and produces factor-level attribution
for a trading day. Uses only built-in Python + datetime.

Factors computed:
  - Market Factor (overall equal-weighted return and breadth)
  - Size Factor (small-cap vs large-cap tilt via breadth spread)
  - Value Factor (value vs growth sector performance)
  - Momentum Factor (sector dispersion / continuation)
  - Sector Factors (specific top/bottom sector contributions)

Contribution 口径 (P2-Q27-fix M338):
  - 每个因子统一输出 `normalized` 字段（-1..1，跨因子可横向比较）与 `unit` 字段
    （"pct"=原始百分比口径 / "norm"=已归一化 -1..1），避免市场/规模用原始 %、
    价值/动量用归一化值、板块用原始 % 导致量纲不一致无法比较。

Calling convention:
    from quant_system.market_attribution import compute_daily_attribution
    result = compute_daily_attribution("20260722")
"""

import json
import os
from datetime import datetime

from quant_system.utils import to_float as _safe_float

# ── helpers ──────────────────────────────────────────────────────────────────

_META_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "generated", "a_share_data"
)

_SECTOR_TAGS = {
    # Value / defensive sectors (low PE, stable)
    "value": {
        "银行", "煤炭开采加工", "石油加工贸易", "油气开采及服务",
        "电力", "公用事业", "公路铁路运输", "燃气", "钢铁",
        "建筑材料", "建筑装饰", "房地产", "保险", "港口航运",
        "交通运输", "基础化学",
    },
    # Growth / tech sectors (high PE, high beta)
    "growth": {
        "半导体", "电子化学品", "元件", "通信设备", "消费电子",
        "光学光电子", "电池", "计算机设备", "软件开发", "IT服务",
        "军工电子", "军工装备", "航天装备", "医疗器械", "生物制品",
        "医疗服务", "化学制药", "汽车零部件",
    },
    # Cyclical / commodity sectors
    "cyclical": {
        "贵金属", "小金属", "能源金属", "金属新材料", "非金属材料",
        "工业金属", "化学原料", "化学制品", "农化制品", "钢铁",
        "工业金属",
    },
    # Defensive / consumer staples
    "defensive": {
        "白酒", "食品加工制造", "饮料制造", "农产品加工", "养殖业",
        "医药商业", "中药", "服装家纺", "家用轻工",
    },
}


def _latest_meta_path():
    """返回 _META_DIR 中日期最新的 meta 文件路径；目录不存在/无文件返回 None。"""
    try:
        files = [f for f in os.listdir(_META_DIR) if f.endswith("-meta.json")]
        if not files:
            return None
        files.sort()
        return os.path.join(_META_DIR, files[-1])
    except OSError:
        return None


def _load_meta(date_str=None, meta_data=None):
    """Load meta JSON; return (meta_dict, date_string)."""
    if meta_data is not None:
        date_str = meta_data.get("trade_date", date_str or "unknown")
        return meta_data, date_str

    explicit = date_str is not None  # 调用方是否显式指定日期
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    meta_path = os.path.join(_META_DIR, f"{date_str}-meta.json")
    if not os.path.exists(meta_path):
        # P2-Q27-fix(M340): 默认取当天日期时，周末/节假日无当日 meta 文件 →
        # 回退最近交易日文件，避免 FileNotFoundError 冒泡导致调用方(quant_web) 500。
        # 显式指定的历史日期仍抛错，避免静默替换为错误日期数据。
        if explicit:
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        fallback = _latest_meta_path()
        if fallback is None:
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        meta_path = fallback

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta, meta.get("trade_date", date_str)


def _direction(value, up_thresh=0.3, down_thresh=-0.3):
    """Return direction label for a numeric contribution."""
    if value > up_thresh:
        return "up"
    if value < down_thresh:
        return "down"
    return "neutral"


def _normalize_contribution(val, scale=1.0):
    """Scale a raw value onto a -1..1 contribution range, clamped."""
    raw = val / max(scale, 1e-9)
    return max(-1.0, min(1.0, round(raw, 2)))


def _norm_or_none(val, scale=10.0):
    """归一化到 -1..1；val 为 None 时返回 None。

    P2-Q27-fix(M338): 各因子统一输出 normalized 字段（-1..1）用于横向比较，
    contribution 保留原始口径（pct 或 norm），通过 unit 字段注明。
    """
    if val is None:
        return None
    return _normalize_contribution(val, scale=scale)


# ── core attribution ─────────────────────────────────────────────────────────

def compute_daily_attribution(date_str=None, meta_data=None):
    """
    Compute factor attribution for a trading day.

    Parameters
    ----------
    date_str : str, optional
        Date in YYYYMMDD format.  Defaults to today.
    meta_data : dict, optional
        Pre-loaded meta JSON.  If given, *date_str* is ignored.

    Returns
    -------
    dict
        {
            "date": str,
            "factors": list[dict] — factor breakdowns,
            "top_drivers": list[str] — plain-English explanations,
            "summary": str — one-line market summary
        }
    """
    meta, date = _load_meta(date_str, meta_data)

    bread = meta.get("breadth") or {}
    raw_bread = meta.get("raw_breadth") or {}
    sec_up = meta.get("sector_up") or []
    sec_down = meta.get("sector_down") or []
    exchange = meta.get("exchange_summary") or {}
    margin = meta.get("margin_metrics") or {}
    risk = meta.get("risk", 1)

    # P1-Q27-fix: 检测宽度数据缺失。
    # 当日 breadth 可能为 {"口径":..., "数据说明":"empty"}（实时行情抓取失败），
    # 此时必须显式标注"数据缺失"，禁止把缺失值当 0 输出伪 0.00% 归因。
    _BREADTH_KEYS = (
        "全A等权涨跌幅%", "涨跌幅中位数%", "上涨占比%",
        "股票数", "上涨", "下跌", "涨超5%", "跌超5%",
    )
    breadth_missing = not (
        any(k in bread for k in _BREADTH_KEYS)
        or any(k in raw_bread for k in _BREADTH_KEYS)
    )

    # Extract key breadth metrics
    eq_return = None if breadth_missing else _safe_float(
        bread.get("全A等权涨跌幅%", raw_bread.get("全A等权涨跌幅%", 0)))
    med_return = None if breadth_missing else _safe_float(
        bread.get("涨跌幅中位数%", raw_bread.get("涨跌幅中位数%", eq_return)))
    advance_pct = None if breadth_missing else _safe_float(
        bread.get("上涨占比%", raw_bread.get("上涨占比%", 50)))
    total_stocks = 0 if breadth_missing else int(bread.get("股票数", raw_bread.get("股票数", 0)))
    up_stocks = 0 if breadth_missing else int(bread.get("上涨", raw_bread.get("上涨", 0)))
    dn_stocks = 0 if breadth_missing else int(bread.get("下跌", raw_bread.get("下跌", 0)))
    strong_up = 0 if breadth_missing else int(bread.get("涨超5%", raw_bread.get("涨超5%", 0)))
    strong_dn = 0 if breadth_missing else int(bread.get("跌超5%", raw_bread.get("跌超5%", 0)))

    factors = []

    # ── 1. Market Factor ────────────────────────────────────────────────
    if breadth_missing:
        factors.append({
            "name": "市场因子 (Market)",
            "contribution": None,
            "unit": "pct",          # P2-Q27-fix(M338): contribution 口径注明
            "normalized": None,
            "direction": "neutral",
            "description": "宽度数据缺失（breadth/raw_breadth 无有效指标），市场因子无法归因",
        })
    else:
        market_dir = _direction(eq_return, up_thresh=0.3, down_thresh=-0.3)
        market_contrib = round(eq_return, 2)
        factors.append({
            "name": "市场因子 (Market)",
            "contribution": market_contrib,
            "unit": "pct",          # P2-Q27-fix(M338)
            "normalized": _norm_or_none(market_contrib, scale=10.0),
            "direction": market_dir,
            "description": (
                f"全A等权涨跌幅 {eq_return:+.2f}%, "
                f"上涨占比 {advance_pct:.1f}% "
                f"({up_stocks}/{total_stocks} 只上涨)"
            ),
        })

    # ── 2. Size Factor (small-cap vs large-cap tilt) ────────────────────
    #   The equal-weighted average is tilted toward small/mid caps.
    #   The median is the "typical stock" — if median > equal-weight,
    #   small/mid outperformed large (positive size factor).
    #   If equal-weight > median, a few mega-caps dragged the average up
    #   (negative size factor = large outperforming).
    if breadth_missing:
        size_spread = None
        size_dir = "neutral"
        size_note_parts = ["宽度数据缺失，规模因子无法计算"]
    else:
        size_spread = round(med_return - eq_return, 2)
        size_dir = "small_cap" if size_spread > 0.2 else (
            "large_cap" if size_spread < -0.2 else "neutral"
        )

        # PE ratio spread between STAR (科创, small/growth) and Main Board (主板, large/value)
        sse_items = {}
        for item in exchange.get("sse", []):
            if isinstance(item, dict) and "项目" in item:
                sse_items[item["项目"]] = item
        pe_main = _safe_float(sse_items.get("平均市盈率", {}).get("主板", 15))
        pe_star = _safe_float(sse_items.get("平均市盈率", {}).get("科创板", 80))
        pe_spread = pe_star / max(pe_main, 0.1)

        size_note_parts = [
            f"中位数 {med_return:+.2f}%, 等权 {eq_return:+.2f}%",
            f"科创板PE/主板PE {pe_spread:.1f}x",
        ]
    factors.append({
        "name": "规模因子 (Size)",
        "contribution": size_spread,
        "unit": "pct",          # P2-Q27-fix(M338)
        "normalized": _norm_or_none(size_spread, scale=10.0),
        "direction": size_dir,
        "description": ", ".join(size_note_parts),
    })

    # ── 3. Value Factor (value vs growth sector dominance) ──────────────
    up_top5 = [s["板块"] for s in sec_up[:5]]
    dn_top5 = [s["板块"] for s in sec_down[:5]]

    def sector_tag_score(sector_list, tag_set):
        """Count how many sectors in *sector_list* belong to *tag_set*."""
        return sum(1 for s in sector_list if s in tag_set)

    value_up = sector_tag_score(up_top5, _SECTOR_TAGS["value"])
    value_dn = sector_tag_score(dn_top5, _SECTOR_TAGS["value"])
    growth_up = sector_tag_score(up_top5, _SECTOR_TAGS["growth"])
    growth_dn = sector_tag_score(dn_top5, _SECTOR_TAGS["growth"])

    # +1 per value sector in top-up, +1 per growth sector in top-down (value tailwind)
    # -1 per growth sector in top-up, -1 per value sector in top-down (growth tailwind)
    value_score_raw = (
        (value_up - value_dn) / 5.0
        + (growth_dn - growth_up) / 5.0
    )
    value_contrib = _normalize_contribution(value_score_raw, scale=1.0)
    value_dir = (
        "value" if value_contrib > 0.15
        else ("growth" if value_contrib < -0.15 else "neutral")
    )

    val_desc_parts = []
    if value_up:
        val_desc_parts.append(f"{value_up}个价值板块领涨")
    if growth_up:
        val_desc_parts.append(f"{growth_up}个成长板块领涨")
    if value_dn:
        val_desc_parts.append(f"{value_dn}个价值板块领跌")
    if growth_dn:
        val_desc_parts.append(f"{growth_dn}个成长板块领跌")
    val_desc = "; ".join(val_desc_parts) if val_desc_parts else "板块风格分化不明显"

    factors.append({
        "name": "价值因子 (Value)",
        "contribution": value_contrib,
        "unit": "norm",         # P2-Q27-fix(M338): 价值因子 contribution 已为 -1..1
        "normalized": value_contrib,
        "direction": value_dir,
        "description": val_desc,
    })

    # ── 4. Momentum Factor (sector dispersion & continuation) ───────────
    if sec_up and sec_down:
        top_pct = _safe_float(sec_up[0].get("涨跌幅", 0))
        bot_pct = _safe_float(sec_down[0].get("涨跌幅", 0))
        sector_spread = top_pct - bot_pct
        top_name = sec_up[0]["板块"]
        bot_name = sec_down[0]["板块"]
    else:
        sector_spread = 0
        top_pct = bot_pct = 0
        top_name = bot_name = "N/A"

    # Momentum: extreme dispersion > 3.5% suggests strong momentum/trend
    # Low dispersion < 2% suggests reversal/choppy market
    # P2-Q27-fix(M337): sector_spread = top - bot ≥ 0 恒成立，原 mom_contrib < 0.0
    # 分支不可达（死代码），"低分化<2% 判 reversal" 的注释与阈值矛盾。
    # 改为 mom_contrib < 0.2（对应 2% 分化度），使 reversal 分支可达且与注释一致。
    mom_contrib = _normalize_contribution(sector_spread, scale=10.0)
    mom_dir = (
        "momentum" if mom_contrib > 0.35
        else ("reversal" if mom_contrib < 0.2 else "neutral")
    )

    factors.append({
        "name": "动量因子 (Momentum)",
        "contribution": mom_contrib,
        "unit": "norm",         # P2-Q27-fix(M338)
        "normalized": mom_contrib,
        "direction": mom_dir,
        "description": (
            f"行业分化度 {sector_spread:.1f}%: "
            f"最强「{top_name}」{top_pct:+.2f}%, "
            f"最弱「{bot_name}」{bot_pct:+.2f}%"
        ),
    })

    # ── 5. Sector Factors (leading & lagging sectors) ───────────────────
    sector_factors = compute_sector_attribution(sec_up[:3] + sec_down[:3])
    factors.extend(sector_factors)

    # ── Assemble top drivers ────────────────────────────────────────────
    top_drivers = _build_top_drivers(
        factors, eq_return, med_return, advance_pct,
        strong_up, strong_dn,
        sec_up, sec_down, margin, risk,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    summary = _build_summary(
        factors, date, eq_return, advance_pct,
        strong_up, strong_dn, risk,
    )

    return {
        "date": date,
        "factors": factors,
        "top_drivers": top_drivers,
        "summary": summary,
    }


def compute_sector_attribution(sector_data):
    """
    Rank individual sector contributions from a sector list.

    Each entry in *sector_data* should have keys:
        "板块" (str), "涨跌幅" (float), "总成交额" (float)

    Returns
    -------
    list[dict]  — sorted by absolute contribution descending.
    """
    contributions = []
    for sec in sector_data:
        name = sec.get("板块", "未知")
        pct = _safe_float(sec.get("涨跌幅", 0))
        amt = _safe_float(sec.get("总成交额", 0))

        # Normalize contribution: a ±10% move is extreme
        contrib = _normalize_contribution(pct, scale=10.0)
        if abs(contrib) < 0.01:
            continue

        direction = "up" if pct > 0 else "down"
        amt_str = f"成交额{amt:.0f}亿" if amt else ""
        contributions.append({
            "name": f"板块-{name}",
            "contribution": round(pct, 2),
            "unit": "pct",          # P2-Q27-fix(M338): 板块 contribution 为原始 %
            "normalized": contrib,
            "direction": direction,
            "description": f"「{name}」{pct:+.2f}%" + (f" ({amt_str})" if amt_str else ""),
        })

    # Sort by absolute contribution descending
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


# ── report formatting ───────────────────────────────────────────────────────

def format_attribution_report(attribution):
    """
    Convert an attribution dict into a human-readable Markdown report.

    Parameters
    ----------
    attribution : dict — output of *compute_daily_attribution*

    Returns
    -------
    str
    """
    date = attribution.get("date", "未知日期")
    factors = attribution.get("factors", [])
    top_drivers = attribution.get("top_drivers", [])
    summary = attribution.get("summary", "")

    lines = [
        f"## 📊 A股市场归因分析 — {date}",
        "",
        summary,
        "",
        "### 因子分解",
        "",
        "| 因子 | 贡献 | 方向 | 说明 |",
        "|------|------|------|------|",
    ]

    for f in factors:
        contrib = f.get("contribution")
        # P1-Q27-fix: 数据缺失的因子显式标注，而不是显示伪 0.00%
        contrib_str = f"{contrib:+.2f}" if isinstance(contrib, (int, float)) else "数据缺失"
        dir_icon = {
            "up": "🟢",
            "down": "🔴",
            "neutral": "⚪",
            "small_cap": "📈",
            "large_cap": "📉",
            "value": "💎",
            "growth": "🚀",
            "momentum": "🔥",
            "reversal": "🔄",
        }.get(f["direction"], "⚪")
        lines.append(
            f"| {f['name']} | {contrib_str} | {dir_icon} {f['direction']} | {f['description']} |"
        )

    lines.extend(["", "### 核心驱动逻辑", ""])
    for i, driver in enumerate(top_drivers, 1):
        lines.append(f"{i}. {driver}")

    lines.append("")
    return "\n".join(lines)


# ── internal helpers ────────────────────────────────────────────────────────

def _build_top_drivers(factors, eq_return, med_return, advance_pct,
                       strong_up, strong_dn,
                       sec_up, sec_down, margin, risk):
    """Generate plain-English top-driver bullets."""
    drivers = []

    # P1-Q27-fix: 宽度数据缺失时不做伪归因，仅说明数据缺失
    if eq_return is None:
        drivers.append("市场宽度数据缺失（breadth/raw_breadth 为空），等权涨跌幅/上涨占比无法归因")

    # Market direction
    if eq_return is not None:
        if eq_return > 1.5:
            drivers.append(f"市场整体强势，等权上涨 {eq_return:+.2f}%")
        elif eq_return > 0.5:
            drivers.append(f"市场温和上涨 {eq_return:+.2f}%")
        elif eq_return < -1.5:
            drivers.append(f"市场整体承压，等权下跌 {eq_return:+.2f}%")
        elif eq_return < -0.5:
            drivers.append(f"市场小幅回调 {eq_return:+.2f}%")
        else:
            drivers.append(f"市场窄幅震荡，等权涨跌幅 {eq_return:+.2f}%")

    # Breadth quality
    if advance_pct is not None:
        if advance_pct > 70:
            drivers.append(f"上涨家数占比 {advance_pct:.0f}%，市场赚钱效应广泛")
        elif advance_pct < 30:
            drivers.append(f"上涨家数仅 {advance_pct:.0f}%，市场赚钱效应差，个股普跌")

    # Median vs equal-weight → size story
    if eq_return is not None and med_return is not None and abs(med_return - eq_return) > 0.5:
        if med_return > eq_return:
            drivers.append(
                f"中位数涨幅({med_return:+.2f}%)高于等权平均({eq_return:+.2f}%)，"
                f"中小盘表现优于大盘权重股"
            )
        else:
            drivers.append(
                f"等权平均({eq_return:+.2f}%)高于中位数({med_return:+.2f}%)，"
                f"大盘权重股拉动指数，中小盘跟涨不足"
            )

    # Extreme moves
    # P2-Q27-fix(M339): strong_up/strong_dn 依赖 breadth 键，breadth 缺失时为 0，
    # 在此跳过相关 driver，避免输出"涨超5%个股0只"的失真文案。
    if advance_pct is not None:
        if strong_up > 500:
            drivers.append(f"涨超5%个股达{strong_up}只，市场情绪极度亢奋")
        elif strong_up > 200:
            drivers.append(f"涨超5%个股{strong_up}只，赚钱效应突出")
        if strong_dn > 500:
            drivers.append(f"跌超5%个股达{strong_dn}只，市场恐慌情绪蔓延")

    # Sector extremes
    if sec_up:
        top = sec_up[0]
        if top["涨跌幅"] > 5:
            drivers.append(
                f"领涨板块「{top['板块']}」涨幅 {top['涨跌幅']:+.2f}% "
                f"(成交额{top.get('总成交额',0):.0f}亿)，板块效应显著"
            )
    if sec_down:
        bot = sec_down[0]
        if bot["涨跌幅"] < -3:
            drivers.append(
                f"领跌板块「{bot['板块']}」跌幅 {bot['涨跌幅']:+.2f}% "
                f"(成交额{bot.get('总成交额',0):.0f}亿)"
            )

    # Margin / leverage signal
    # P2-Q27-fix(M339): 原变量名 margin_change 实际取的是"全A融资余额(亿)"（余额），
    # 文案中括号展示的却是变化额，展示错位。此处拆分为 balance(余额) 与 change_yi(较前日变化)，
    # 变化额取 "全A较前日变化(亿)"，余额单独展示。
    if isinstance(margin, dict) and "全A融资余额(亿)" in margin:
        balance = _safe_float(margin.get("全A融资余额(亿)", 0))
        change_yi = _safe_float(margin.get("全A较前日变化(亿)", 0))
        margin_pct = _safe_float(margin.get("全A较前日变化%", 0))
        if margin_pct < -1:
            drivers.append(
                f"两融余额较前日下降 {abs(margin_pct):.1f}% "
                f"(变化 {change_yi:+.0f}亿，余额 {balance:.0f}亿)，杠杆资金加速离场"
            )
        elif margin_pct > 1:
            drivers.append(
                f"两融余额较前日上升 {margin_pct:+.1f}% "
                f"(变化 {change_yi:+.0f}亿，余额 {balance:.0f}亿)，杠杆资金积极加仓"
            )

    # Risk level
    risk_labels = {1: "低风险震荡", 2: "中性偏弱", 3: "中性", 4: "中性偏强",
                   5: "偏强", 6: "强", 7: "高风险/恐慌", 8: "极端风险", 9: "危机"}
    risk_label = risk_labels.get(risk, f"风险级{risk}")
    drivers.append(f"系统风险等级: {risk} — {risk_label}")

    return drivers


def _build_summary(factors, date, eq_return, advance_pct,
                   strong_up, strong_dn, risk):
    """One-line market summary."""
    # P1-Q27-fix: 宽度数据缺失时输出显式提示，禁止伪 0.00% 摘要
    if eq_return is None:
        return (f"📆 {date} 市场宽度数据缺失：无法计算全A等权涨跌幅/上涨占比归因 "
                f"(breadth/raw_breadth 为空，实时行情可能抓取失败)")

    # Market regime
    if eq_return > 1.5 and advance_pct > 65:
        regime = "强势普涨"
    elif eq_return > 0.5 and advance_pct > 55:
        regime = "温和上涨"
    elif eq_return < -1.5 and advance_pct < 35:
        regime = "弱势普跌"
    elif eq_return < -0.5 and advance_pct < 40:
        regime = "承压回调"
    elif abs(eq_return) <= 0.5:
        regime = "窄幅震荡"
    else:
        regime = "结构分化"

    # Risk suffix
    risk_suffix = f" (风险等级 {risk})" if risk > 3 else ""

    summary = (
        f"📆 {date} 市场{regime}：等权涨跌幅 {eq_return:+.2f}%, "
        f"上涨占比 {advance_pct:.1f}%, "
        f"涨超5% {strong_up}只 / 跌超5% {strong_dn}只"
        f"{risk_suffix}"
    )
    return summary


# ── CLI convenience ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    att = compute_daily_attribution(date_arg)
    print(format_attribution_report(att))
