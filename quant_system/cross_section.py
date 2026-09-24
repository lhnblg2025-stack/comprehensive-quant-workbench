"""
因子截面分析 — 409只全量横截面排名

功能:
  1. ROE百分位: 盈利能力横截面对比
  2. PE分位数: 估值在历史中的位置 vs 全市场对比
  3. 动量排名: 20日/60日动量在409只中的分位
  4. 综合因子分: 多因子合成
  5. 因子暴露: 每只股票在各因子上的Z-score

用法:
  python3 -m quant_system.cross_section              # 综合排名
  python3 -m quant_system.cross_section --factor PE   # 指定因子排名
  python3 -m quant_system.cross_section --distribution # 因子分布
"""

from __future__ import annotations
import logging

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.utils import to_float as _to_float_impl

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))


def _to_float(val: object, default: float = 0.0) -> float:
    """P2-Q6-fix (Q6-L486): 无法解析返回 NaN（不进截面）。

    D1 收敛: 转发 quant_system.utils.to_float（默认 NaN 语义不变）。
    """
    if val is None:
        return float("nan")
    if isinstance(val, str):
        text = val.replace(",", "").strip()
        if text in ("", "-", "--", "None", "nan", "NaN"):
            return float("nan")
        val = text
    return _to_float_impl(val, default=float("nan"), finite=True)


def get_fundamental_snapshot(stocks: list[str] = None) -> pd.DataFrame:
    """获取409只股票的基本面截面数据"""
    try:
        from quant_system.watchlist import get_watchlist
        if stocks is None:
            wl = get_watchlist()
            stocks = [str(s.get('code', '')) for s in wl[:200]]
    except ImportError:
        return pd.DataFrame()

    rows = []
    import akshare as ak
    for i, code in enumerate(stocks):
        try:
            info = ak.stock_individual_info_em(symbol=code)
            if info is None:
                continue
            d = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
            rows.append({
                'code': code,
                'name': d.get('股票简称', ''),
                # P2-Q6-fix (Q6-L486): akshare 缺失返回 '-'/''，float() 抛错会吞掉整只
                #   股票；改用 _to_float 清洗，解析失败置 NaN（不进截面）
                'pe': _to_float(d.get('市盈率-动态')),
                'pb': _to_float(d.get('市净率')),
                'mkt_cap': _to_float(d.get('总市值')),
                'industry': d.get('行业', ''),
            })
        except Exception as e:
            logging.getLogger(__name__).error(f"[cross_section] 操作失败: {e}", exc_info=True)
            continue
        if (i + 1) % 40 == 0:
            import time
            time.sleep(0.5)

    df = pd.DataFrame(rows)
    return df


def compute_cross_sectional_factors(df: pd.DataFrame, pct_ranking: bool = True) -> pd.DataFrame:
    """计算截面因子Z-score和百分位

    Args:
        df: 包含 pe, pb, mkt_cap 等列
        pct_ranking: 返回百分位排名(True)或Z-score(False)

    Returns:
        DataFrame: 含各因子排名
    """
    if df.empty:
        return df

    result = df.copy()

    for col in ['pe', 'pb', 'mkt_cap']:
        if col in result.columns:
            # 替换无穷/空值
            mask = np.isfinite(result[col].values) & (result[col] > 0)
            valid = result.loc[mask, col]
            if len(valid) > 0:
                if pct_ranking:
                    result[f'{col}_pct'] = 0
                    result.loc[mask, f'{col}_pct'] = valid.rank(pct=True) * 100
                else:
                    mean_v = valid.mean()
                    std_v = valid.std()
                    result[f'{col}_z'] = 0
                    if std_v > 0:
                        result.loc[mask, f'{col}_z'] = (valid - mean_v) / std_v

    return result


def get_momentum_ranking(stocks: list[str], asof_date: datetime | None = None) -> pd.DataFrame:
    """计算动量截面排名
    
    Parameters
    ----------
    stocks : list[str]
        股票代码列表
    asof_date : datetime, optional
        # V4.1 fix: 新增 asof_date 参数替代硬编码 datetime.now()
        # 原代码 datetime.now() 在回测中会使用当前日期而非回测日期，
        # 导致前瞻偏差。调用方应传入回测日期。
    """
    import akshare as ak
    if asof_date is None:
        asof_date = datetime.now(CST)
    rows = []
    for i, code in enumerate(stocks):
        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                       start_date=(asof_date - timedelta(days=120)).strftime('%Y%m%d'),
                                       end_date=asof_date.strftime('%Y%m%d'),
                                       adjust="qfq")
            if hist is not None and len(hist) > 20:
                close = hist['收盘'].values
                ret_20d = close[-1] / close[-20] - 1 if len(close) >= 20 else 0
                # P2-Q6-fix (Q6-L487): 数据不足 60 日时原实现静默回退 ret_20d，
                #   方向/量纲错位；改为 NaN 并在排名中剔除（rank 默认 skipna）
                ret_60d = close[-1] / close[-60] - 1 if len(close) >= 60 else np.nan
                vol = hist['成交量'].values
                vol_ma = np.mean(vol[-5:])
                vol_ratio = vol[-1] / vol_ma if vol_ma > 0 else 1
                rows.append({
                    'code': code,
                    'ret_20d': round(ret_20d * 100, 2),
                    'ret_60d': round(ret_60d * 100, 2),
                    'vol_ratio': round(vol_ratio, 2),
                })
        except Exception as e:
            logging.getLogger(__name__).error(f"[cross_section] 操作失败: {e}", exc_info=True)
            continue
        if (i + 1) % 40 == 0:
            import time
            time.sleep(0.5)

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ['ret_20d', 'ret_60d']:
            if col in df.columns:
                df[f'{col}_pct'] = df[col].rank(pct=True) * 100
        if 'vol_ratio' in df.columns:
            df['vol_ratio_pct'] = df['vol_ratio'].rank(pct=True) * 100
    return df


def get_composite_ranking(top_n: int = 20) -> pd.DataFrame:
    """综合排名（价值+成长+动量）"""
    try:
        from quant_system.watchlist import get_watchlist
        wl = get_watchlist()
        stocks = [str(s.get('code', '')) for s in wl[:150]]
    except ImportError:
        return pd.DataFrame()

    # 基本面截面
    fund = get_fundamental_snapshot(stocks)

    # 动量截面
    mom = get_momentum_ranking(stocks)

    if fund.empty and mom.empty:
        return pd.DataFrame()

    # 合并
    if not fund.empty and not mom.empty:
        merged = fund.merge(mom, on='code', how='left')
    elif not fund.empty:
        merged = fund
    else:
        merged = mom

    # 综合得分
    # P2-Q6-fix (Q6-M484): 原综合分取全部 *_pct 列（含 mkt_cap_pct 市值、vol_ratio_pct
    #   量比），"价值+成长+动量"排名被大市值/高换手污染，方向含义混乱。改为只取明确
    #   因子：估值（低 PE/PB 为价值，用 100-分位翻转）+ 动量（ret_20d/ret_60d 高分优）。
    parts = []
    if 'pe_pct' in merged.columns:
        parts.append((100 - merged['pe_pct']).rank(pct=True))
    if 'pb_pct' in merged.columns:
        parts.append((100 - merged['pb_pct']).rank(pct=True))
    if 'ret_20d_pct' in merged.columns:
        parts.append(merged['ret_20d_pct'].rank(pct=True))
    if 'ret_60d_pct' in merged.columns:
        parts.append(merged['ret_60d_pct'].rank(pct=True))
    if parts:
        merged['composite'] = pd.concat(parts, axis=1).mean(axis=1)
        merged = merged.sort_values('composite', ascending=False)

    return merged.head(top_n)


def format_composite(df: pd.DataFrame) -> str:
    if df.empty:
        return "⚠️ 暂无数据"
    lines = ["# 📊 因子截面综合排名\n"]
    lines.append(f"{'股票':<14}{'综合分':>8}{'PE百分位':>10}{'动量20d':>10}{'动量60d':>10}")
    lines.append("-" * 55)
    for _, r in df.iterrows():
        name = f"{r.get('name','')}({r.get('code','')[:6]})"
        comp = f"{r.get('composite',0):.1f}"
        pe = f"{r.get('pe_pct',0):.0f}%" if 'pe_pct' in r else "-"
        m20 = f"{r.get('ret_20d_pct',0):.0f}%" if 'ret_20d_pct' in r else "-"
        m60 = f"{r.get('ret_60d_pct',0):.0f}%" if 'ret_60d_pct' in r else "-"
        lines.append(f"  {name:<12} {comp:>7} {pe:>9} {m20:>9} {m60:>9}")
    return "\n".join(lines)


def format_distribution(df: pd.DataFrame, factor: str = 'pe') -> str:
    if df.empty:
        return "⚠️ 暂无数据"
    vals = df[factor].dropna()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lines = [f"# {factor.upper()} 截面分布\n"]
    lines.append(f"  均值: {vals.mean():.1f}")
    lines.append(f"  中位数: {vals.median():.1f}")
    lines.append(f"  p10: {vals.quantile(0.1):.1f}")
    lines.append(f"  p90: {vals.quantile(0.9):.1f}")
    lines.append(f"  样本: {len(vals)}")
    return "\n".join(lines)


def main():
    args = set(sys.argv[1:])
    if "--factor" in args:
        # P2-Q6-fix (Q6-L485): 原实现含 argv[0]（脚本路径），无参数时 factor=脚本路径
        #   静默无输出；改为仅从 sys.argv[1:] 取非 flag 参数
        non_flag = [a for a in sys.argv[1:] if not a.startswith('-')]
        factor = non_flag[-1] if non_flag else 'pe'
        try:
            from quant_system.watchlist import get_watchlist
            wl = get_watchlist()
            stocks = [str(s.get('code', '')) for s in wl[:200]]
        except ImportError:
            stocks = []
        df = get_fundamental_snapshot(stocks)
        if not df.empty and factor in df.columns:
            print(format_distribution(df, factor))
    elif "--distribution" in args:
        try:
            from quant_system.watchlist import get_watchlist
            wl = get_watchlist()
            stocks = [str(s.get('code', '')) for s in wl[:200]]
        except ImportError:
            stocks = []
        df = get_fundamental_snapshot(stocks)
        if not df.empty:
            for col in ['pe', 'pb']:
                print(format_distribution(df, col))
    else:
        df = get_composite_ranking(20)
        print(format_composite(df))


if __name__ == "__main__":
    main()
