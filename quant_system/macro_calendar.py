"""
宏观事件日历+策略联动 — 经济数据发布跟踪与参数自动调整

功能:
  1. 宏观日历: PMI/CPI/PPI/社融/M2/信贷/进出口 发布日程
  2. 数据对比: 实际值 vs 预期值 vs 前值
  3. 策略联动: 数据超预期→自动调整策略参数
  4. 发布提醒: 标记即将发布的经济数据

用法:
  python3 -m quant_system.macro_calendar               # 本月宏观日历
  python3 -m quant_system.macro_calendar --today       # 今日发布
  python3 -m quant_system.macro_calendar --upcoming    # 即将发布
  python3 -m quant_system.macro_calendar --load-all    # 加载所有历史数据
  python3 -m quant_system.macro_calendar --strategy    # 策略参数建议
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# 宏观数据发布日历（每月例行）
MACRO_SCHEDULE = {
    "PMI": {
        "publish_day": "月末最后一日",
        "source_desc": "国家统计局 09:30发布",
        "channel": "制造业PMI/非制造业PMI/综合PMI",
        "threshold": {"bull": 51, "bear": 48},
    },
    "CPI/PPI": {
        "publish_day": "每月9日左右",
        "source_desc": "国家统计局 09:30发布",
        "channel": "CPI同比/环比/核心CPI, PPI同比",
        "threshold": {"bull_cpi": 3.0, "bear_cpi": 1.5},
    },
    "社融/M2": {
        "publish_day": "每月10-15日",
        "source_desc": "中国人民银行 16:00发布",
        "channel": "社融增量/M2增速/新增人民币贷款",
        "threshold": {"bull_she": 25000, "bear_she": 15000},  # 亿
    },
    "进出口": {
        "publish_day": "每月7日左右",
        "source_desc": "海关总署 11:00发布",
        "channel": "出口同比/进口同比/贸易顺差",
        "threshold": {"bull_exp": 5, "bear_exp": -5},
    },
    "社消零售": {
        "publish_day": "每月15日左右",
        "source_desc": "国家统计局 10:00发布",
        "channel": "社消总额同比",
        "threshold": {"bull": 5, "bear": 3},
    },
    "工业增加值": {
        "publish_day": "每月15日左右",
        "source_desc": "国家统计局 10:00发布",
        "channel": "规上工业增加值同比",
        "threshold": {"bull": 5.5, "bear": 4},
    },
    "GDP": {
        "publish_day": "每季度后15日左右",
        "source_desc": "国家统计局 10:00发布",
        "channel": "GDP同比",
        "threshold": {"bull": 5.5, "bear": 4.5},
    },
}

# 策略参数联动规则
STRATEGY_LINKS = {
    "bull_momentum": {
        "trigger": ["PMI>51", "社融>25000亿", "工业增加值>5.5%"],
        "effect": {"momentum_lookback": 60, "position_limit": 0.9, "signal_threshold": 0.03},
        "desc": "经济扩张→长周期动量+高仓位+低信号门槛",
    },
    "bear_momentum": {
        "trigger": ["PMI<48", "社融<15000亿", "工业增加值<4%"],
        "effect": {"momentum_lookback": 21, "position_limit": 0.3, "signal_threshold": 0.06},
        "desc": "经济收缩→短周期动量+低仓位+高信号门槛",
    },
    "inflation_warning": {
        "trigger": ["CPI>3.0%"],
        "effect": {"position_limit": 0.5, "stop_loss_pct": 0.05},
        "desc": "通胀过高→降仓位+收紧止损",
    },
    "deflation_warning": {
        "trigger": ["CPI<1.5%", "PPI<0%"],
        "effect": {"momentum_lookback": 42, "position_limit": 0.6},
        "desc": "通缩风险→中等动量周期+适中仓位",
    },
    "export_boom": {
        "trigger": ["出口同比>10%"],
        "effect": {"sector_focus": "外贸相关", "position_limit": 0.8},
        "desc": "出口强劲→聚焦外贸板块",
    },
}


def get_macro_calendar(month: int = None, year: int = None) -> list[dict]:
    """生成本月宏观数据发布日历

    Returns:
        list[dict]: 每项包含数据名称、预计发布时间、状态
    """
    import re as _re
    now = datetime.now(CST)
    year = year or now.year
    month = month or now.month

    calendar = []
    import calendar as cal
    for name, info in MACRO_SCHEDULE.items():
        day_str = info['publish_day']
        # 解析发布日
        # V6 fix (Q3 MEDIUM: macro_calendar.py:125-132):
        #   V5.4 用 ''.join(数字) 提取 → "每月10-15日" 得 "1015" → 被钳成 28 号；
        #   V6 用正则取首个数字，月末不钳位（PMI 实际 31 号），季度项标注。
        # P2-Q3-fix(M407-③): 季度发布项（GDP「每季度后15日左右」）此前被硬塞进
        # 当月。现按「本季度结束后次月 + 发布日」计算下期发布日（Q1→4月, Q2→7月,
        # Q3→10月, Q4→次年1月），并标注「季度发布」。
        if "季度" in day_str:
            q = (month - 1) // 3 + 1          # 本季度 1-4
            pub_month = q * 3 + 1             # 季度结束后次月
            pub_year = year
            if pub_month > 12:
                pub_month = 1
                pub_year = year + 1
            m_digit = _re.search(r"\d{1,2}", day_str)
            pub_day = int(m_digit.group()) if m_digit else 15
            pub_day = min(pub_day, cal.monthrange(pub_year, pub_month)[1])
            pub_date = datetime(pub_year, pub_month, pub_day, 10, 0, tzinfo=CST)
            is_past = pub_date < now
            is_today = pub_date.date() == now.date()
            calendar.append({
                'name': name,
                'publish_date': pub_date,
                'day_str': day_str + "（季度发布）",
                'is_past': is_past,
                'is_today': is_today,
                'days_until': (pub_date - now).days,
                'source': info['source_desc'],
                'channel': info['channel'],
            })
            continue

        if "月末" in day_str:
            pub_day = cal.monthrange(year, month)[1]  # 月末不钳位
        else:
            m_digit = _re.search(r"\d{1,2}", day_str)
            pub_day = int(m_digit.group()) if m_digit else 15  # 取首个数字，"10-15日"→10
        # 发布日为1-2位数字；非法/超限时兜底为15
        if pub_day < 1 or pub_day > 31:
            pub_day = 15
        pub_date = datetime(year, month, min(pub_day, 31), 10, 0, tzinfo=CST)
        is_past = pub_date < now
        is_today = pub_date.date() == now.date()

        calendar.append({
            'name': name,
            'publish_date': pub_date,
            'day_str': day_str,
            'is_past': is_past,
            'is_today': is_today,
            'days_until': (pub_date - now).days,
            'source': info['source_desc'],
            'channel': info['channel'],
        })

    calendar.sort(key=lambda x: x['publish_date'])
    return calendar


# V6 fix (Q3 CRITICAL/HIGH: macro_calendar.py:158-179):
#   V5.4 假定 df.iloc[-1] 是最新一期并取最后一列 → 对 akshare 1.18.64 实测
#   PMI/M2/GDP（新→旧降序）取到 2008/2006 年最旧一期，且取到的是非主指标列
#   （非制造业-同比增长 / M0-同比增长 / 第三产业-同比增长）；CPI 升序但取到「前值」。
#   analyze_macro_regime() 曾拿 2008 年 PMI=-2.15 判「收缩」→ 策略联动方向性反转。
#   V6: 接口名映射到 akshare 1.18.64 实际存在的函数，按列名取数（先探测列名再取值，
#   禁止位置索引），按日期列降序取最新非 NaN 行；接口缺失时显式返回 error 字段
#   （降级可见，不静默返回 None）。
#   实测列名（akshare 1.18.64）：
#     PMI: 月份/制造业-指数/制造业-同比增长/非制造业-指数/非制造业-同比增长（降序）
#     CPI: 商品/日期/今值/预测值/前值（升序）
#     PPI: 月份/当月/当月同比增长/累计
#     社融: 月份/社会融资规模增量/...（升序）
#     M2: 月份/货币和准货币(M2)-同比增长/...（降序）
#     社消零售: 月份/当月/同比增长/...
#     工业增加值: 商品/日期/今值/预测值/前值（升序）
#     进出口(出口同比): 商品/日期/今值/预测值/前值（升序）
#     GDP: 季度/国内生产总值-同比增长/...（降序）
_MACRO_API_SPECS = {
    "PMI":     ("macro_china_pmi",                ["月份"], ["制造业-指数"]),
    "CPI":     ("macro_china_cpi",                ["月份"], ["全国-同比增长"]),
    "PPI":     ("macro_china_ppi",                ["月份"], ["当月同比增长", "当月"]),
    "社融":    ("macro_china_shrzgm",             ["月份"], ["社会融资规模增量"]),
    "M2":      ("macro_china_money_supply",       ["月份"], ["货币和准货币(M2)-同比增长"]),
    "社消零售": ("macro_china_consumer_goods_retail", ["月份"], ["同比增长", "当月"]),
    "工业增加值": ("macro_china_industrial_production_yoy", ["日期"], ["今值"]),
    "进出口":  ("macro_china_hgjck",              ["月份"], ["当月出口额-同比增长"]),
    "GDP":     ("macro_china_gdp",                ["季度"], ["国内生产总值-同比增长"]),
}

# 触发条件关键词 → 数据名别名（如 STRATEGY_LINKS 里写「出口同比」对应数据名「进出口」）
_MACRO_NAME_ALIASES = {
    "进出口": ["出口", "进口"],
}


import re as _re


def _parse_macro_date(val: Any) -> datetime | None:
    """把宏观数据日期列解析为可排序 datetime（月份/季度/日期 混合格式）。"""
    if val is None:
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val).to_pydatetime()
    s = str(val).strip()
    # 季度："2026年第1-2季度" / "2006年第1季度"
    m = _re.search(r"(\d{4})年第?(\d+)(?:-(\d+))?季度", s)
    if m:
        year = int(m.group(1))
        q2 = int(m.group(3)) if m.group(3) else int(m.group(2))
        return datetime(year, min(q2, 4) * 3, 1)
    # 月度："2026年07月份" / "2026-07" / "201501"
    m = _re.search(r"(\d{4})\D?(\d{1,2})", s)
    if m:
        return datetime(int(m.group(1)), min(int(m.group(2)), 12), 1)
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return None


def _extract_latest_macro(df: pd.DataFrame, date_cols: list[str],
                          value_cols: list[str]) -> dict | None:
    """按列名取数：按日期列降序取最新非 NaN 行及其前值（禁止位置索引）。"""
    if df is None or df.empty:
        return None
    date_col = next((c for c in date_cols if c in df.columns), None)
    value_col = next((c for c in value_cols if c in df.columns), None)
    if date_col is None or value_col is None:
        return {
            "error": f"akshare 返回列与预期不符: 日期列={date_cols} 指标列={value_cols}，实际列={list(df.columns)}",
        }
    try:
        df = df.copy()
        df["_dt"] = df[date_col].map(_parse_macro_date)
        df = df.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)
        vals = pd.to_numeric(df[value_col], errors="coerce")
        df = df.assign(_val=vals)
        latest = df[df["_val"].notna()]
        if latest.empty:
            return {"error": f"{value_col} 全为 NaN（数据未发布）"}
        row0 = latest.iloc[0]
        prev = latest.iloc[1] if len(latest) > 1 else None
        trend = None
        if prev is not None and pd.notna(row0["_val"]) and pd.notna(prev["_val"]):
            trend = "up" if float(row0["_val"]) > float(prev["_val"]) else "down"
        return {
            "latest_value": float(row0["_val"]),
            "latest_date": str(row0[date_col]),
            "prev_value": float(prev["_val"]) if prev is not None else None,
            "prev_date": str(prev[date_col]) if prev is not None else None,
            "trend": trend,
            "error": None,
        }
    except Exception as exc:
        return {"error": f"宏观数据解析失败: {exc}"}


def get_macro_data(name: str) -> dict | None:
    """获取特定宏观数据的历史值

    Args:
        name: 宏观数据名称 (PMI/CPI/PPI/社融/M2/进出口/GDP/社消零售/工业增加值)

    Returns:
        dict: latest_value/latest_date/prev_value/prev_date/trend/error
              接口缺失或解析失败时返回含 error 字段的 dict（降级可见）。
    """
    spec = _MACRO_API_SPECS.get(name)
    if spec is None:
        return {"error": f"未知宏观数据名称: {name}"}
    fn_name, date_cols, value_cols = spec
    try:
        import akshare as ak
        if not hasattr(ak, fn_name):
            # V6 fix (Q3 HIGH: macro_calendar.py:158-166): 接口缺失显式告警，
            # 不再被 except Exception 静默吞成 None
            return {
                "error": f"akshare 无接口 {fn_name}()（V5.4 用的 {fn_name} 不存在，"
                         f"已映射到实际接口）",
            }
        df = getattr(ak, fn_name)()
        return _extract_latest_macro(df, date_cols, value_cols)
    except Exception as exc:
        return {"error": f"{name} 获取失败: {exc}"}


def get_all_macro() -> dict:
    """获取所有宏观数据最新值

    V6 fix (Q3 HIGH): V5.4 遍历 MACRO_SCHEDULE 的组合键（"CPI/PPI"、"社融/M2"），
    与 apis 字典的单键永远匹配不上 → 全部永远无数据。V6 改为遍历规范数据名。
    """
    result = {}
    for name in _MACRO_API_SPECS:
        data = get_macro_data(name)
        if data:
            result[name] = data
    return result


def analyze_macro_regime() -> str:
    """基于宏观数据判断当前经济阶段"""
    data = get_all_macro()
    signals = []

    # PMI判断
    pmi = data.get('PMI', {}).get('latest_value')
    if pmi is not None:
        try:
            pmi_v = float(pmi) if not isinstance(pmi, (int, float)) else pmi
            if pmi_v > 51:
                signals.append("扩张")
            elif pmi_v < 48:
                signals.append("收缩")
            else:
                signals.append("平稳")
        except (ValueError, TypeError):
            pass

    # CPI判断
    cpi = data.get('CPI', {}).get('latest_value')
    if cpi is not None:
        try:
            cpi_v = float(cpi) if not isinstance(cpi, (int, float)) else cpi
            if cpi_v > 3:
                signals.append("通胀")
            elif cpi_v < 1.5:
                signals.append("通缩风险")
        except (ValueError, TypeError):
            pass

    return " + ".join(signals) if signals else "无法判断"


def get_strategy_adjustments() -> list[dict]:
    """基于宏观数据生成策略调整建议"""
    data = get_all_macro()
    suggestions = []

    for name, link in STRATEGY_LINKS.items():
        triggered = False
        trigger_details = []
        for trigger in link['trigger']:
            # V6 fix (Q3 CRITICAL): 用规范数据名 + 别名匹配触发条件
            #   （V5.4 遍历 MACRO_SCHEDULE 的组合键 "CPI/PPI" 等永远匹配不上
            #   "CPI>3.0%" 之类的条件 → 4 条联动规则永不触发）
            for macro_name in _MACRO_API_SPECS:
                keys = [macro_name] + _MACRO_NAME_ALIASES.get(macro_name, [])
                if not any(k in trigger for k in keys):
                    continue
                macro_data = data.get(macro_name, {})
                latest = macro_data.get('latest_value')
                if latest is None:
                    continue
                try:
                    v = float(latest) if not isinstance(latest, (int, float)) else latest
                    # 解析trigger条件
                    if '>' in trigger:
                        threshold = float(trigger.split('>')[1].replace('%', '').replace('亿', ''))
                        if v > threshold:
                            triggered = True
                            trigger_details.append(f"{macro_name}={v}")
                    elif '<' in trigger:
                        threshold = float(trigger.split('<')[1].replace('%', '').replace('亿', ''))
                        if v < threshold:
                            triggered = True
                            trigger_details.append(f"{macro_name}={v}")
                except (ValueError, TypeError):
                    pass

        if triggered:
            suggestions.append({
                'name': name,
                'desc': link['desc'],
                'triggered_by': trigger_details,
                'effect': link['effect'],
            })

    return suggestions


# ───────── 格式化 ─────────

def format_calendar(upcoming_only: bool = False) -> str:
    """格式化宏观日历"""
    calendar = get_macro_calendar()
    if upcoming_only:
        calendar = [c for c in calendar if not c['is_past']]

    lines = ["\n## 📅 宏观数据发布日历\n"]
    lines.append(f"{'数据名称':<12}{'预计发布':<10}{'剩余天数':>8}{'来源'}")
    lines.append("-" * 55)
    for item in calendar:
        name = item['name']
        day = item['day_str']
        days = item['days_until']
        source = item['source'][:20]
        if item['is_today']:
            status = " 🔴 今日发布！"
        elif item['is_past']:
            status = " ✅ 已发布"
        else:
            status = f" ⏳ 还有{days}天"
        lines.append(f"  {name:<10} {day:<10} {status}")
    return "\n".join(lines)


def format_macro_status() -> str:
    """格式化宏观数据最新状态"""
    data = get_all_macro()
    lines = ["\n## 📈 宏观经济数据最新值\n"]
    if not data:
        return "⚠️ 暂无宏观数据"

    for name, info in data.items():
        latest = info.get('latest_value', '')
        prev = info.get('prev_value', '')
        trend = info.get('trend')
        err = info.get('error')
        if err:
            lines.append(f"  ⚠️ {name}: {err}")
            continue
        # V6 fix (Q3 LOW): trend 为 None（无前值可比）时显示「—」，不再误导为 🔴↓
        trend_arrow = "🟢↑" if trend == 'up' else ("🔴↓" if trend == 'down' else "➖—")
        lines.append(f"  {trend_arrow} {name}: 最新={latest}, 前值={prev}")
    return "\n".join(lines)


def format_strategy_advice() -> str:
    """格式化策略参数调整建议"""
    suggestions = get_strategy_adjustments()
    lines = ["\n## 🎯 宏观驱动的策略参数建议\n"]
    if not suggestions:
        lines.append("✅ 当前宏观环境无显著偏差，维持默认参数")
    else:
        for s in suggestions:
            lines.append(f"\n🟢 {s['desc']}")
            lines.append(f"  触发条件: {', '.join(s['triggered_by'])}")
            lines.append(f"  调整参数:")
            for k, v in s['effect'].items():
                lines.append(f"    {k} = {v}")
    return "\n".join(lines)


def format_full_report() -> str:
    parts = []
    now = datetime.now(CST)
    parts.append(f"# 📊 宏观日历+策略联动 ({now.strftime('%Y-%m-%d')})")
    parts.append(f"{'='*50}")

    # 宏观日历
    parts.append(format_calendar())
    # 最新数据
    parts.append(format_macro_status())
    # 经济阶段
    parts.append(f"\n## 🏭 经济阶段\n  {analyze_macro_regime()}")
    # 策略建议
    parts.append(format_strategy_advice())

    return "\n".join(parts)


# ───────── CLI ─────────

def main():
    args = set(sys.argv[1:])

    if "--today" in args:
        calendar = get_macro_calendar()
        today_items = [c for c in calendar if c['is_today']]
        if today_items:
            print("\n## 🔴 今日发布数据\n")
            for item in today_items:
                print(f"  {item['name']} - {item['source']}")
                data = get_macro_data(item['name'])
                # V11 审计修复（Medium）: get_macro_data 失败返回含 error 的 dict，
                # 原实现 data['latest_value'] 直接 KeyError 崩溃。修正: 防御性读取。
                if data and "latest_value" in data:
                    print(f"  最新值: {data['latest_value']}")
                    print(f"  前值: {data.get('prev_value', 'N/A')}")
                elif data and "error" in data:
                    print(f"  ⚠️ 数据获取失败: {data['error']}")
                else:
                    print("  ⚠️ 数据为空")
        else:
            print("\n✅ 今日无重要宏观数据发布")
    elif "--upcoming" in args:
        print(format_calendar(upcoming_only=True))
    elif "--load-all" in args:
        data = get_all_macro()
        print(format_macro_status())
    elif "--strategy" in args:
        print(format_strategy_advice())
    else:
        print(format_full_report())


if __name__ == "__main__":
    main()
