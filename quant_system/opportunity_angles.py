#!/usr/bin/env python3
"""
opportunity_angles.py — 机会多视角评估层 (2026-08-14)
====================================================
用户核心诉求: 机会提示不能只给一个"信号分", 要从不同角度衡量并突出侧重。
五视角:
  A. 行情防御 (event/timing)   — 大盘/板块环境 + 涨跌停/缩量位置: 该标的此刻是否"能买"
  B. 短线趋势 (short)          — MA/趋势/突破: 短线动量跟随
  C. 中长线价值 (value)        — PE/PB/ROE/毛利率/负债率: 低估价值
  D. 资金/筹码 (flow)          — 放量/缩量/换手: 资金面
  E. 技术动量 (momentum)       — RSI/CCI/MACD/KDJ: 超卖反转/金叉

对每只候选机会, 把现有 signals/signal_reasons/因子 归类到五视角,
每个视角输出: 判断(看多/看空/中性) + 置信度(0-1) + 关键理由(2条内),
并选主导视角(侧重明确) + 综合置信度。供 opportunity_monitor 飞书推送
与前端展示使用, 替换原单调的 signal_summary。

设计原则:
  - 纯函数, 零 IO, 方便单测
  - 不改变原有 signal_count 计算(兼容现有消费方)
  - 视角判定逻辑透明、可调阈值(集中在 _CONF 配置)
"""
from __future__ import annotations

# ── 视角置信度阈值 (可调, 中心化) ──
_CONF = {
    "defensive_min": 0.15,     # 风险信号触发置信
    "short_ma_bias": 0.15,
    "value_pe_ok": 20.0,
    "value_pb_ok": 1.5,
    "value_roe_ok": 10.0,
    "flow_vol_ratio": 1.2,
    "momentum_rsi_sell": 75.0,
}


# 信号 → 视角 归属表 (weight 为该信号对此视角的贡献)
# 视角 key: defense(行情防御) / short(短线趋势) / value(中长线价值)
#           flow(资金筹码) / momentum(技术动量)
_ANGLE_MAP: dict[str, dict[str, float]] = {
    # ── defense 行情防御 ──
    "limit_up": {"defense": 0.5, "short": -0.3},     # 涨停封不住/一字板难交易
    "limit_down": {"defense": -0.6},                  # 跌停=逃不出
    "downtrend": {"short": -0.5, "defense": -0.2},
    "near_low52": {"defense": 0.4, "value": 0.3},
    "today_drop": {"defense": -0.2, "short": -0.2},
    "active": {"flow": 0.2},
    # ── short 短线趋势 ──
    "uptrend": {"short": 0.6, "momentum": 0.2},
    "ma60_support": {"short": 0.4, "defense": 0.2},
    "ma144_support": {"short": 0.5, "defense": 0.25},
    "fib_support": {"short": 0.4},
    "swing_low_support": {"short": 0.3},
    "breakout": {"short": 0.6, "momentum": 0.2},
    "ma20_dev": {"short": 0.2},
    # ── value 中长线价值 ──
    "low_pe": {"value": 0.5},
    "low_pb": {"value": 0.6, "defense": 0.2},
    "fundamental_roe": {"value": 0.5},
    "fundamental_growth": {"value": 0.4},
    "fundamental_margin": {"value": 0.3},
    "fundamental_safety": {"value": 0.3},
    # ── flow 资金筹码 ──
    "volume_price_up": {"flow": 0.6, "short": 0.2},
    "volume_price_down": {"flow": 0.3, "defense": 0.15},
    "active": {"flow": 0.2},
    # ── momentum 技术动量 ──
    "rsi_oversold": {"momentum": 0.5, "defense": 0.15},
    "rsi_near_oversold": {"momentum": 0.3},
    "rsi_turning_up": {"momentum": 0.5},
    "cci_oversold": {"momentum": 0.5, "defense": 0.15},
    "macd_bullish_cross": {"momentum": 0.6, "short": 0.2},
    "macd_bullish_diverging": {"momentum": 0.3},
    "bullish_divergence": {"momentum": 0.7, "short": 0.2},
    "boll_lower_band": {"momentum": 0.4, "defense": 0.15},
    "boll_near_lower": {"momentum": 0.25},
    "today_up": {"momentum": 0.2, "short": 0.1},
}

# 视角中文名 + 顺位(主导判定用)
_ANGLES = [
    ("value", "中长线价值"),
    ("short", "短线趋势"),
    ("momentum", "技术动量"),
    ("flow", "资金筹码"),
    ("defense", "行情防御"),
]

_ANGLE_DESC = {
    "value": "PE/PB/ROE/毛利/负债低估",
    "short": "均线趋势/突破/支撑",
    "momentum": "RSI/CCI/MACD 反转/金叉",
    "flow": "放量/缩量/换手资金",
    "defense": "大盘环境/涨跌停/位置",
}


def _trim_reasons(reasons: list[str], n: int = 2) -> str:
    """取前 n 条理由拼接(短句, 供飞书紧凑展示)。"""
    out = []
    for r in reasons:
        r = str(r).strip()
        if not r or r in out:
            continue
        out.append(r)
        if len(out) >= n:
            break
    return "；".join(out)


def evaluate_angles(sig: dict, reasons: dict | None = None,
                    extra: dict | None = None) -> dict:
    """把单只机会的信号归类为五视角评估。

    Args:
        sig:     signals dict (signal_name -> 权重/贡献)
        reasons: signal_reasons dict (signal_name -> 展示文本), 可选
        extra:   额外字段(pe/pb/roe/trend/vol_ratio/factor_score 等), 可选

    Returns:
        {
          "angles": {angle: {"label","judge","conf","reason","signals"}},
          "dominant_angle": angle_key,   # 主导视角
          "dominant_label": 主导中文名,
          "composite_conf": 0-10 综合置信度(兼容原signal_count量纲),
          "summary": 一句话侧重结论,
        }
    """
    sig = sig or {}
    reasons = reasons or {}
    extra = extra or {}

    # 信号名→归属 汇总
    # 2026-08-14 逐行审计修复(两轮):
    #  ① 负贡献取 abs → 下降趋势反而加分;
    #  ② 权重w已编码方向(downtrend:short=-0.5), contrib 再乘 w 会双符号翻转。
    #  现改为: score += w (权重含方向), contrib 只判断触发与正负理由。
    angle_score: dict[str, float] = {a: 0.0 for a, _ in _ANGLES}
    angle_trig: dict[str, list[str]] = {a: [] for a, _ in _ANGLES}
    angle_neg: dict[str, list[str]] = {a: [] for a, _ in _ANGLES}
    for sname, contrib in sig.items():
        mapping = _ANGLE_MAP.get(sname)
        if not mapping:
            continue
        contrib = float(contrib)
        if contrib == 0:
            continue
        for ang, w in mapping.items():
            angle_score[ang] += w
            txt = str(reasons.get(sname, sname))
            if txt == sname:
                txt = sname
            if contrib > 0:
                angle_trig[ang].append(txt)
            else:
                angle_neg[ang].append(txt)

    # 额外硬指标补充
    pe = safe_f(extra.get("pe_ttm"))
    pb = safe_f(extra.get("pb"))
    roe = safe_f(extra.get("roe"))
    vol_ratio = safe_f(extra.get("vol_ratio"))
    factor_score = safe_f(extra.get("factor_score"))
    price = safe_f(extra.get("price"))
    ma60 = safe_f(extra.get("ma60"))

    reasons_out: dict[str, list[str]] = {a: list(angle_trig[a]) for a, _, in _ANGLES}

    # value 视角: 低估程度量化
    if 0 < pe < _CONF["value_pe_ok"]:
        angle_score["value"] += (1 - pe / _CONF["value_pe_ok"]) * 0.8
        reasons_out["value"].append(f"PE={pe:.1f}")
    if 0 < pb < _CONF["value_pb_ok"]:
        angle_score["value"] += (1 - pb / _CONF["value_pb_ok"]) * 0.9
        reasons_out["value"].append(f"PB={pb:.2f}")
    if roe and roe > _CONF["value_roe_ok"]:
        angle_score["value"] += 0.4
        reasons_out["value"].append(f"ROE={roe:.1f}%")

    # flow 视角: 量能
    if vol_ratio:
        if vol_ratio >= _CONF["flow_vol_ratio"]:
            angle_score["flow"] += min(0.8, (vol_ratio - 1) * 0.5)
        reasons_out["flow"] = angle_trig["flow"] or ([f"量比={vol_ratio:.1f}x"] if vol_ratio else [])

    # short 视角: 均线距离
    if ma60 and price:
        dist = (price / ma60 - 1) * 100
        if -3 <= dist <= 6:
            angle_score["short"] += 0.3
            if not reasons_out["short"]:
                reasons_out["short"].append(f"距MA60 {dist:+.1f}%")

    # factor_score 补强综合(方向算动量/价值近似)
    if factor_score and factor_score > 5:
        angle_score["momentum"] += min(0.5, (factor_score - 5) * 0.15)

    # 归一置信度到 0-1
    # 2026-08-14 逐行审计修复: 原"低置信过滤"为死代码(round未变值), 删除;
    # conf 恒非负(score/2 归一), 删除永假的"看空"分支。
    conf: dict[str, float] = {}
    for a, _ in _ANGLES:
        conf[a] = min(1.0, max(0.0, angle_score[a] / 2.0))  # 2.0 视为满分 1.0

    # 判断(judge): 正分看多, 负分看空; defense 语义=风险高低(分高=警惕)
    result_angles = {}
    for a, label in _ANGLES:
        c = conf[a]
        raw = angle_score[a]
        if a == "defense":
            if raw >= 1.0:
                judge = "警惕"  # 风险高
            elif raw >= 0.4:
                judge = "中性"
            else:
                judge = "安全"
        else:
            if raw >= 1.0:
                judge = "看多"
            elif raw >= 0.3:
                judge = "偏多"
            elif raw <= -0.4:
                judge = "看空"
            else:
                judge = "中性"
        # 展示理由: 正贡献优先, 无则补负面理由
        shown = angle_trig[a][:2]
        if not shown and angle_neg[a]:
            shown = [f"负: {x}" for x in angle_neg[a][:1]]
        result_angles[a] = {
            "label": label, "judge": judge, "conf": round(c, 2),
            "signals": shown,
            "desc": _ANGLE_DESC[a],
        }

    # 主导视角: 非defense中 conf 最高 + judge>=偏多, 且 conf>0.2; 否则用综合
    positives = [(a, c) for a, c in conf.items()
                 if a != "defense" and c >= 0.2 and result_angles[a]["judge"] in ("看多", "偏多")]
    if positives:
        dominant = max(positives, key=lambda x: x[1])[0]
    else:
        # 全部中性: 取相对的动量/价值
        fallback = max([(a, c) for a, c in conf.items() if a != "defense"],
                       key=lambda x: x[1])
        dominant = fallback[0]

    composite = sum(v for v in angle_score.values() if v > 0)
    composite = round(min(10.0, composite * 1.6), 1)  # 映射到 0-10

    # 一句话侧重结论
    d = result_angles[dominant]
    summary = f"侧重**{d['label']}**（{d['judge']}，置信{d['conf']:.0%}）"
    if d["signals"]:
        summary += " - " + d["signals"][0]

    return {
        "angles": result_angles,
        "dominant_angle": dominant,
        "dominant_label": result_angles[dominant]["label"],
        "composite_conf": composite,
        "summary": summary,
    }


def safe_f(v) -> float:
    """安全转 float, 无效返回 0。"""
    try:
        if v is None:
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0
