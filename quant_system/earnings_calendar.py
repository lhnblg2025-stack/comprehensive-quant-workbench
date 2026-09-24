"""
季度业绩预告/快报日历 — 个股财报发布时间跟踪

功能:
  1. 业绩预告: 最新一期业绩预告（预增/预减/扭亏/首亏）
  2. 业绩快报: 最新一期正式快报数据
  3. 日历视图: 按发布日排列
  4. 重点关注: 超预期/扭亏/大幅增长

用法:
  python3 -m quant_system.earnings_calendar              # 本期报告
  python3 -m quant_system.earnings_calendar --surprise   # 超预期
  python3 -m quant_system.earnings_calendar --all        # 全量
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

FORECAST_REQUIRED_COLUMNS = {"code", "name", "forecast_type", "announce_date"}


def _missing_forecast_columns(df: pd.DataFrame) -> list[str]:
    missing = FORECAST_REQUIRED_COLUMNS - set(df.columns)
    has_change_range = "change_range" in df.columns or {"profit_change_low", "profit_change_high"}.issubset(df.columns)
    if not has_change_range:
        missing.add("change_range")
    return sorted(missing)


def _latest_quarter() -> str:
    """返回最近一期财报季度

    P2-Q3-fix(M404): 按 A 股披露时点修正边界 ——
      1-4月:   上年年报(1231) 披露期
      5-6月:   一季报(0331) 已披露完毕、中报(0630) 7月起才披露 → 返回当年 0331
      7-9月:   中报(0630) 披露期（截止 8/31）→ 返回当年 0630
      10-12月: 三季报(0930) 披露期（10月起）→ 返回当年 0930
    旧逻辑 5-6 月返回 0630、9 月返回 0930，会拿到尚未披露的季度数据窗口。
    """
    now = datetime.now(CST)
    m = now.month
    if m <= 4:
        return f"{now.year-1}1231"
    elif m <= 6:
        return f"{now.year}0331"
    elif m <= 9:
        return f"{now.year}0630"
    else:
        return f"{now.year}0930"


def get_earnings_forecast(quarter: str = None) -> pd.DataFrame:
    """获取业绩预告

    Args:
        quarter: 季度码，如 "20260630"

    Returns:
        DataFrame: 预告类型、净利润变动、上年同期
    """
    quarter = quarter or _latest_quarter()
    try:
        import akshare as ak
        df = ak.stock_yjyg_em(date=quarter)
        if df is not None and not df.empty:
            # 标准化列名
            col_map = {
                '股票代码': 'code', '股票简称': 'name',
                '预告类型': 'forecast_type', '业绩预告类型': 'forecast_type',
                '预测指标': 'forecast_metric', '预测数值': 'forecast_value',
                '业绩变动': 'change_type', '业绩变动幅度': 'change_range',
                '业绩变动原因': 'summary', '业绩预告摘要': 'summary',
                '报告日期': 'report_date', '公告日期': 'announce_date', '最新公告日期': 'announce_date',
                '上年同期值': 'previous_value',
                '净利润变动幅度下限': 'profit_change_low',
                '净利润变动幅度上限': 'profit_change_high',
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            missing = _missing_forecast_columns(df)
            if missing:
                raise ValueError(f"业绩预告缺少关键列: {missing}; 原始列={list(df.columns)}")
            # P2-Q3-fix(L405): 不再静默 head(200) 截断全市场预告；返回全量，
            # 展示层 format_surprises/get_surprises 各自再做 top_n 截断。
            return df
    except ValueError:
        raise
    except Exception as e:
        print(f"⚠️ 获取业绩预告失败: {e}")
    return pd.DataFrame()


def get_earnings_express(quarter: str = None) -> pd.DataFrame:
    """获取业绩快报"""
    quarter = quarter or _latest_quarter()
    try:
        import akshare as ak
        df = ak.stock_yjkb_em(date=quarter)
        if df is not None and not df.empty:
            # P2-Q3-fix(L405): 返回全量快报，不再静默截断前 200 只
            return df
    except Exception as e:
        print(f"⚠️ 获取业绩快报失败: {e}")
    return pd.DataFrame()


def get_surprises(top_n: int = 20) -> pd.DataFrame:
    """获取超预期个股

    策略: 预告类型='预增'或'扭亏'，且变动幅度大的
    """
    df = get_earnings_forecast()
    if df.empty:
        return df

    interesting = ['预增', '扭亏', '大幅上升', '略增']
    missing = _missing_forecast_columns(df)
    if missing:
        raise ValueError(f"业绩预告缺少关键列: {missing}")
    mask = df['forecast_type'].isin(interesting)
    result = df[mask].head(top_n)
    return result


def _fmt_change(chg) -> str:
    """P2-Q3-fix(L406): NaN 为真、数值 0 为假 —— 用 pd.notna 判断并过滤空值。"""
    if pd.isna(chg):
        return ""
    s = str(chg).strip()
    if s.lower() in ("", "nan", "none", "-"):
        return ""
    return s


def format_surprises(df: pd.DataFrame) -> str:
    if df.empty:
        return "⚠️ 暂无数据"
    lines = ["# 💥 业绩超预期\n"]
    for _, r in df.head(20).iterrows():
        name = r.get('name', r.get('股票简称', ''))
        ftype = r.get('forecast_type', r.get('业绩预告类型', r.get('预告类型', '')))
        change_range = r.get('change_range', r.get('业绩变动幅度', ''))
        chg_low = r.get('profit_change_low', r.get('净利润变动幅度下限', ''))
        chg_high = r.get('profit_change_high', r.get('净利润变动幅度上限', ''))
        emoji = {"预增": "🟢", "扭亏": "🔄", "略增": "🟡", "大幅上升": "🚀"}.get(ftype, "⚪")
        # P2-Q3-fix(L406): NaN 为真、数值 0 为假 —— 用 pd.notna 判断并过滤空值
        change_range_s = _fmt_change(change_range)
        chg_low_s = _fmt_change(chg_low)
        chg_high_s = _fmt_change(chg_high)
        if change_range_s:
            chg_str = change_range_s
        elif chg_low_s and chg_high_s:
            chg_str = f"{chg_low_s}%~{chg_high_s}"
        else:
            chg_str = ""
        lines.append(f"  {emoji} {name}: {ftype} {chg_str}")
    return "\n".join(lines)


def format_full_report() -> str:
    parts = [f"# 📅 业绩日历 ({_latest_quarter()[:4]}-{_latest_quarter()[4:6]})"]
    parts.append("=" * 50)

    df = get_surprises()
    parts.append(format_surprises(df))

    return "\n".join(parts)


def main():
    args = set(sys.argv[1:])
    if "--surprise" in args:
        df = get_surprises()
        print(format_surprises(df))
    elif "--all" in args:
        df = get_earnings_forecast()
        if not df.empty:
            print(df.head(50).to_string())
        else:
            print("⚠️ 暂无数据")
    else:
        print(format_full_report())


if __name__ == "__main__":
    main()
