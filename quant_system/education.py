"""
量化交易系统 — 信号教育与解释层。

把技术信号转成人能看懂的操作建议。
适用于: intraday_decision.py 的输出增强、买入机会的自然语言解读。
"""

from __future__ import annotations
import logging



# ════════════════════════════════════════════════════════════════
# 1. 信号分类与中文解释
# ════════════════════════════════════════════════════════════════

_SIGNAL_EXPLANATIONS = {
    "rsi_oversold": {
        "title": "RSI超卖",
        "desc": "相对强弱指标进入超卖区(≤35)，意味着短期跌多了，有反弹需求",
        "action": "关注 — 分批轻仓试探",
        "probability": "中等偏高",
        "wait_for": "RSI回升到40以上+收阳线再加重仓位",
        "risk": "超卖后可能继续超卖(下跌趋势中RSI可以在低位钝化)",
        "examples": [
            "贵州茅台RSI=28 历史上RSI<30后5日上涨概率约65%",
            "工商银行RSI=25 银行股RSI超卖准确率较高",
        ],
    },
    "cci_oversold": {
        "title": "CCI超卖",
        "desc": "商品通道指标低于-80，说明价格偏离均值太远，有回归需求",
        "action": "关注 — 等CCI回升到-50以上考虑入场",
        "probability": "中等",
        "wait_for": "CCI连续两根回升 + 价格企稳",
        "risk": "趋势向下时CCI可以持续在-100以下",
        "examples": [],
    },
    "rsi_turning_up": {
        "title": "RSI触底回升",
        "desc": "RSI从低位开始反弹，是动能转向的早期信号",
        "action": "建仓信号 — 配合其他信号使用",
        "probability": "较高(配合放量)",
        "wait_for": "确认RSI连续回升2天以上",
        "risk": "反弹初期可能只是小反弹，还不是反转",
        "examples": [],
    },
    "near_ma60": {
        "title": "MA60接近支撑",
        "desc": "股价接近60日均线——中期趋势的重要支撑位",
        "action": "观察 — 看能否守住",
        "probability": "趋势向上时较高",
        "wait_for": "在MA60附近出现阳线+放量再入场",
        "risk": "趋势向下时MA60是压力位不是支撑位",
        "examples": [
            "60日均线是机构调仓的重要参考线",
            "强势股的MA60支撑通常有效3-5次",
        ],
    },
    "near_ma144": {
        "title": "MA144接近支撑",
        "desc": "股价接近144日均线——长期牛熊分界线（很多大资金的生命线）",
        "action": "重要支撑 — 缩量触碰可试仓",
        "probability": "较高",
        "wait_for": "触碰MA144后收长下影线或阳线反包",
        "risk": "跌破MA144通常意味着中期趋势转弱",
        "examples": [
            "A股历史上MA144是重要牛熊分界线",
            "跌破MA144后通常需要数月才能修复",
        ],
    },
    "near_boll_lower": {
        "title": "BOLL下轨支撑",
        "desc": "价格触及布林带下轨，统计学上偏离均值-2倍标准差",
        "action": "观察 — 缩量触碰布林下轨可关注",
        "probability": "震荡市中较高",
        "wait_for": "触碰下轨后反弹+站回中轨",
        "risk": "单边下跌市中布林下轨会不断下移",
        "examples": [
            "BOLL下轨在横盘震荡时支撑效果最好",
            "牛市中的BOLL下轨触碰是很好的买点",
        ],
    },
    "bullish_divergence": {
        "title": "底背离(最强信号⭐)",
        "desc": "价格创新低但MACD/RSI指标不创新低——下跌动能衰竭,最可靠的反转信号之一",
        "action": "买入! — 这是最值得信赖的买入信号之一",
        "probability": "高(约65-75%成功率)",
        "wait_for": "底背离形成后出现放量阳线确认",
        "risk": "底背离可能多次背离后才真正反转(三次背离最常见)",
        "examples": [
            "底背离是Gerald Appel三重滤网的核心买入信号",
            "白马股底背离成功率高于小盘股",
        ],
    },
    "macd_bullish_cross": {
        "title": "MACD金叉",
        "desc": "MACD快线上穿慢线,柱由负转正——趋势由跌转涨",
        "action": "买入信号 — 0轴上方金叉最强",
        "probability": "0轴上方:高 | 0轴下方:中",
        "wait_for": "金叉次日收阳确认",
        "risk": "0轴下方金叉可能是反弹不是反转",
        "examples": [
            "0轴上方金叉(强势区): 强烈买入信号",
            "0轴下方金叉(弱势区): 先当反弹做",
        ],
    },
    "volume_confirmation": {
        "title": "放量确认",
        "desc": "成交量大于均量1.5倍以上,说明资金在活跃参与",
        "action": "配合使用 — 不作为独立信号",
        "probability": "N/A(辅助信号)",
        "wait_for": "放量+阳线才是真买盘",
        "risk": "放量下跌=出货,放量滞涨=分歧大",
        "examples": [],
    },
    "today_dropping": {
        "title": "今日下跌",
        "desc": "当日跌幅超过2%,短期超跌",
        "action": "观察 — 等企稳信号",
        "probability": "低(仅作提醒)",
        "wait_for": "次日不再创新低",
        "risk": "下跌趋势中每天都可以跌2%",
        "examples": [],
    },
    "low_pe": {
        "title": "低估值(PE较低)",
        "desc": "市盈率低于行业平均水平,具备安全边际",
        "action": "价值支撑 — 适合底仓配置",
        "probability": "长期有效",
        "wait_for": "配合技术面信号一起使用",
        "risk": "低PE陷阱: 利润大幅下滑导致的被动低PE要回避",
        "examples": [
            "银行PE<8通常意味着已具备较高安全边际",
            "消费股PE<15可视为低估区域",
        ],
    },
    "low_pb": {
        "title": "破净/低市净率",
        "desc": "PB<1.2,股价接近甚至低于净资产",
        "action": "安全边际强 — 低于1.0可逐步建仓",
        "probability": "极高",
        "wait_for": "PB<1+盈利为正+分红稳定 = 黄金坑",
        "risk": "银行/钢铁等重资产行业PB天然低,不说明任何问题",
        "examples": [
            "中国能建PB=0.94 已破净,下行空间有限",
            "兴业银行PB=0.48 极其低估",
        ],
    },
    "near_52w_low": {
        "title": "接近52周低点",
        "desc": "股价处于近一年最低位附近",
        "action": "观察 — 确认不破前低",
        "probability": "下跌趋势中低",
        "wait_for": "不破前低+放量反弹",
        "risk": "低点下面还有低点",
        "examples": [],
    },
    "today_up": {
        "title": "今日收涨",
        "desc": "当日收盘上涨,短线企稳",
        "action": "积极信号 — 确认其他买入信号",
        "probability": "N/A",
        "wait_for": "连续2天以上上涨更可靠",
        "risk": "一天上涨不代表趋势反转",
        "examples": [],
    },
    "rsi_turning_down": {
        "title": "RSI从高位回落",
        "desc": "RSI从超买区回落,动能减弱",
        "action": "减仓/止盈信号",
        "probability": "较高",
        "wait_for": "RSI跌破70并伴随价格下跌",
        "risk": "强势股票RSI可以在70以上钝化很久",
        "examples": [],
    },
    "cci_oversold_turn": {
        "title": "CCI从超卖回升",
        "desc": "CCI从低于-100回升,是超卖后反弹的确认信号",
        "action": "买入 — 超卖+回升=入场时机",
        "probability": "中等偏高",
        "wait_for": "CCI回升超过-50",
        "risk": "大趋势向下时反弹高度有限",
        "examples": [],
    },
}


def explain_signal(signal_key: str, symbol: str = "", price: float = 0) -> str:
    """Generate a human-readable explanation for a single signal.

    # P2-Q25-fix(L302): symbol/price 参数此前从未使用, 现并入输出文案
    (标的与价格), 参数默认值保持空/0 以便既有调用方不受影响。
    """
    info = _SIGNAL_EXPLANATIONS.get(signal_key)
    if not info:
        return f"⚠️ 未知信号: {signal_key}"

    header = f"📌 **{info['title']}**"
    if symbol:
        header += f" | {symbol}"
        if price:
            header += f" ¥{price:.2f}"

    lines = [
        header,
        f"   {info['desc']}",
        f"   📋 建议: {info['action']}",
        f"   📊 胜率: {info['probability']}",
    ]
    if info["wait_for"]:
        lines.append(f"   ⏳ 确认条件: {info['wait_for']}")
    lines.append(f"   ⚠️ 风险: {info['risk']}")
    if info["examples"]:
        lines.append(f"   💡 参考: {info['examples'][0]}")
    return "\n".join(lines)


def explain_buy_setup(signals: dict[str, bool], symbol: str, name: str,
                      price: float, rsi: float = 0, cci: float = 0) -> str:
    """
    综合解释一组买入信号,生成完整的交易建议。
    
    Args:
        signals: {signal_key: True} 的字典
        symbol/name/price: 股票信息
        rsi/cci: 指标值
    
    Returns:
        自然语言操作建议
    """
    active_signals = [k for k, v in signals.items() if v]
    if not active_signals:
        return ""
    
    lines = []
    
    # Header
    lines.append(f"📊 **{name}({symbol})** — ¥{price:.2f}")
    lines.append(f"{'─'*40}")
    
    # Signal strength
    score = len(active_signals)
    has_divergence = signals.get("bullish_divergence", False)
    has_oversold = any(signals.get(k, False) for k in ["rsi_oversold", "cci_oversold"])
    has_support = any(signals.get(k, False) for k in ["near_ma60", "near_ma144", "near_boll_lower"])
    
    # Overall judgement
    if has_divergence and has_oversold:
        lines.append("🟢 **强烈买入信号** — 底背离+超卖双重确认")
    elif has_divergence:
        lines.append("🟢 **买入信号** — 底背离出现")
    elif has_oversold and has_support:
        lines.append("🟡 **关注买入** — 超卖+均线支撑,等企稳")
    elif has_oversold:
        lines.append("🟡 **谨慎关注** — 超卖区域,等反弹信号")
    else:
        lines.append("⚪ **一般关注** — 有支撑但无明确反转信号")
    
    # Signal details
    lines.append(f"\n**信号明细 ({score})**")
    for key in active_signals:
        info = _SIGNAL_EXPLANATIONS.get(key, {})
        if info:
            lines.append(f"  ✅ {info['title']}: {info['desc'][:30]}…")
        else:
            lines.append(f"  ✅ {key}")
    
    # Technical values
    tech_parts = []
    if rsi > 0:
        tech_parts.append(f"RSI={rsi:.0f}")
    if cci != 0:
        tech_parts.append(f"CCI={cci:.0f}")
    if tech_parts:
        lines.append(f"\n**技术值**: {' | '.join(tech_parts)}")
    
    # Action plan
    lines.append(f"\n**📋 操作计划**")
    
    if has_divergence:
        lines.append(f"  1️⃣ 总仓位: 计划仓位的50%")
        lines.append(f"  2️⃣ 入场: 现价¥{price:.2f} 首批建仓")
        lines.append(f"  3️⃣ 加仓: 放量阳线确认后加剩余50%")
        stop = round(price * 0.95, 2)
        lines.append(f"  4️⃣ 止损: ¥{stop} (-5%)")
        target = round(price * 1.10, 2)
        lines.append(f"  5️⃣ 目标: ¥{target} (+10%)")
    elif has_oversold:
        lines.append(f"  1️⃣ 分两批: 现价¥{price:.2f}首批30%, 等CCI回升再补70%")
        stop = round(price * 0.95, 2)
        lines.append(f"  2️⃣ 止损: ¥{stop} (-5%)")
    else:
        lines.append(f"  1️⃣ 等待: 等价格企稳+放量信号再入场")
        lines.append(f"  2️⃣ 入场条件: 连续2天收阳+站上5日均线")
    
    # Risk reminder
    lines.append(f"\n**⚠️ 风险提示**")
    lines.append("  - 任何信号都不是100%准确,严格止损")
    lines.append("  - 新手建议只用闲钱投资,首次仓位不超过总资金20%")
    lines.append("  - 分批建仓比一把梭更好")
    
    return "\n".join(lines)


def format_decision_with_education(decision_text: str, signals: dict = None) -> str:
    """增强版决策格式化: 在技术决策文本基础上追加信号解释层。

    Args:
        decision_text: 已有决策文本(如 intraday_decision 输出的技术指标行)。
        signals: {signal_key: True} 字典, 用于在文本尾部追加对应信号的中文解释。

    Returns:
        追加解释层后的完整文本。

    # P2-Q25-fix(L303): 修正断头引用 — 原注释声称"after format_decision()",
    但项目不存在该函数; 且 `signals` 参数此前从未使用。现补全 signals 解释层。
    """
    lines = decision_text.split("\n")

    # 综合评分行: 附加评分解读
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "综合分" in line and "信号" in line:
            try:
                score = float(line.split("=")[-1].split()[0])
                if score >= 80:
                    new_lines.append("  📝 综合评分高(>80),可重点关注")
                elif score <= 30:
                    new_lines.append("  📝 综合评分低(<30),当前不适合操作")
            except Exception as e:
                logging.getLogger(__name__).error(f"[education] 操作失败: {e}", exc_info=True)

    # 追加 signals 解释层
    if signals:
        active = [k for k, v in signals.items() if v]
        if active:
            new_lines.append("")
            new_lines.append("【信号解释】")
            for key in active:
                info = _SIGNAL_EXPLANATIONS.get(key)
                if info:
                    new_lines.append(f"  ✅ {info['title']}: {info['desc'][:40]}…")
                    new_lines.append(f"     📋 建议: {info['action']}")
                else:
                    new_lines.append(f"  ✅ {key}")

    return "\n".join(new_lines)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="信号教育解释层")
    parser.add_argument("--signal", type=str, help="查看某个信号的解释")
    parser.add_argument("--list", action="store_true", help="列出所有信号")
    args = parser.parse_args()
    
    if args.list:
        print("📚 **可用的买入信号**\n")
        for key, info in sorted(_SIGNAL_EXPLANATIONS.items()):
            print(f"  {key}: {info['title']} — {info['desc'][:40]}...")
            print(f"      建议: {info['action']} | 胜率: {info['probability']}")
            print()
    
    elif args.signal:
        print(explain_signal(args.signal))
    
    else:
        # Demo: show a sample explanation
        test_signals = {
            "rsi_oversold": True,
            "cci_oversold": True,
            "bullish_divergence": True,
            "rsi_turning_up": True,
        }
        print(explain_buy_setup(test_signals, "600519", "贵州茅台", 1320.00, rsi=23, cci=-98))
