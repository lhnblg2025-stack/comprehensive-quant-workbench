"""
A股特色数据 — 龙虎榜/大宗交易/股东增减持/概念板块

功能:
  1. 龙虎榜数据: 上榜个股、游资营业部、净买入排名
  2. 大宗交易: 折溢价率、成交量、成交额
  3. 股东增减持: 董监高变动、大股东增持/减持
  4. 概念板块: 热点概念成分股、涨幅排名
  5. 牛熊分界: 涨停/跌停/连板统计

用法:
  python3 -m quant_system.ashare_special              # 完整报告
  python3 -m quant_system.ashare_special --dragon     # 龙虎榜
  python3 -m quant_system.ashare_special --block-trade # 大宗交易
  python3 -m quant_system.ashare_special --holder      # 股东增减持
  python3 -m quant_system.ashare_special --concept     # 概念板块热力图
"""

from __future__ import annotations
import logging

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# P2-Q4-fix(L429): 删除从未使用的 _cached 装饰器与 _CACHE/_CACHE_TTL 全局缓存
# （原实现定义了缓存却无任何调用点，属死代码），同时清理随之失效的
# time/_time、typing.Any、numpy 导入。

# ───────── 1. 龙虎榜 ─────────

def get_dragon_tiger_daily(date: str = None) -> pd.DataFrame:
    """获取龙虎榜每日上榜数据

    Args:
        date: 日期 YYYYMMDD，默认最新交易日

    Returns:
        DataFrame: 上榜个股、净买入、营业部
    """
    if date is None:
        date = (datetime.now(CST) - timedelta(days=1)).strftime('%Y%m%d')
    try:
        import akshare as ak
        for api_name in ['stock_lhb_detail_daily_sina', 'stock_lhb_detail_em']:
            try:
                if api_name.endswith('_sina'):
                    df = getattr(ak, api_name)(date=date)
                else:
                    df = getattr(ak, api_name)(start_date=date, end_date=date)
                if df is not None and len(df) > 0:
                    df = df.copy()
                    df['source'] = api_name
                    return df
            except Exception as e:
                logging.getLogger(__name__).error(f"[ashare_special] 操作失败: {e}", exc_info=True)
                continue
    except Exception as e:
        print(f"⚠️ 获取龙虎榜数据失败: {e}")
    return pd.DataFrame()


def get_top_dragon_tiger(top_n: int = 10) -> pd.DataFrame:
    """龙虎榜净买入排名。"""
    df = get_dragon_tiger_daily()
    if df.empty:
        return df
    net_buy_col = next((c for c in df.columns if '净买' in c), None)
    if net_buy_col:
        ranked = df.copy()
        ranked[net_buy_col] = pd.to_numeric(ranked[net_buy_col], errors='coerce')
        return ranked.sort_values(net_buy_col, ascending=False).head(top_n)
    return df.head(top_n)


# ───────── 2. 大宗交易 ─────────

def get_block_trades(date: str = None) -> pd.DataFrame:
    """获取大宗交易数据

    Returns:
        DataFrame: 大宗交易折溢价、成交量
    """
    if date is None:
        date = (datetime.now(CST) - timedelta(days=1)).strftime('%Y%m%d')
    date = str(date).replace('-', '')
    try:
        import akshare as ak
        for api_name in ['stock_dzjy_mrmx', 'stock_dzjy_mrtj']:
            try:
                if api_name == 'stock_dzjy_mrmx':
                    df = getattr(ak, api_name)(symbol='A股', start_date=date, end_date=date)
                else:
                    df = getattr(ak, api_name)(start_date=date, end_date=date)
                if df is not None and len(df) > 0:
                    df = df.copy()
                    df['source'] = api_name
                    return df
            except Exception as e:
                logging.getLogger(__name__).error(f"[ashare_special] 操作失败: {e}", exc_info=True)
                continue
    except Exception as e:
        print(f"⚠️ 获取大宗交易失败: {e}")
    return pd.DataFrame()


def get_premium_block_trades(top_n: int = 10) -> pd.DataFrame:
    """折价率最高的大宗交易"""
    df = get_block_trades()
    if df.empty:
        return df
    # 找折价率列
    premium_col = next((c for c in df.columns if '折价' in c or '溢价' in c), None)
    if premium_col:
        df = df.sort_values(premium_col, ascending=True).head(top_n)
    return df


# ───────── 3. 股东增减持 ─────────

def get_shareholder_changes(stock: str = None, top_n: int = 20) -> pd.DataFrame:
    """获取股东增减持数据

    Args:
        stock: 股票代码，None为全市场
        top_n: 返回条数

    Returns:
        DataFrame: 董监高变动/大股东增减持
    """
    try:
        import akshare as ak
        if stock:
            df = ak.stock_shareholder_change_ths(symbol=stock)
            if df is not None and len(df) > 0:
                df = df.copy()
                df['source'] = 'stock_shareholder_change_ths'
                return df.head(top_n)
        return pd.DataFrame(
            columns=['source', 'error'],
            data=[['stock_shareholder_change_ths', 'akshare当前仅提供个股股东增减持接口，请传入stock参数']],
        )
    except Exception as e:
        print(f"⚠️ 获取股东增减持失败: {e}")
    return pd.DataFrame()


# ───────── 4. 概念板块 ─────────

def get_concept_board_top(rank_n: int = 20) -> pd.DataFrame:
    """获取概念板块涨幅排名"""
    try:
        import akshare as ak
        df = ak.stock_board_concept_name_em()
        if df is not None and len(df) > 0:
            # P2-Q4-fix(L429): 删除计算后从未使用的 rank_col 死变量
            pct_col = next((c for c in df.columns if '涨跌幅' in c), None)
            if pct_col:
                df = df.sort_values(pct_col, ascending=False).head(rank_n)
            return df
    except Exception as e:
        print(f"⚠️ 获取概念板块排名失败: {e}")
    return pd.DataFrame()


def get_concept_fund_flow(indicator: str = "今日") -> pd.DataFrame:
    """概念板块资金流向"""
    try:
        import akshare as ak
        df = ak.stock_sector_fund_flow_rank(
            indicator=indicator,
            sector_type="概念资金流"
        )
        return df
    except Exception as e:
        print(f"⚠️ 获取概念资金流失败: {e}")
    return pd.DataFrame()


# ───────── 5. 综合报告 ─────────

def format_dragon_tiger(df: pd.DataFrame) -> str:
    if df.empty:
        return "⚠️ 暂无龙虎榜数据"
    lines = ["\n## 🐲 龙虎榜/涨停连板\n"]
    for _, r in df.head(10).iterrows():
        name = str(r.get('名称', r.get('股票名称', '')))
        code = str(r.get('代码', r.get('股票代码', '')))
        lb = r.get('连板数', '-')
        if '涨停封单额' in r:
            amt = r['涨停封单额']
            amt_str = f"{amt/1e8:.2f}亿" if isinstance(amt, (int, float)) and amt >= 1e8 else str(amt)
            lines.append(f"  {name}({code}) 📈 连板{lb} 封单{amt_str}")
        else:
            lines.append(f"  {name}({code})")
    return "\n".join(lines)


def format_concept_top(df: pd.DataFrame) -> str:
    if df.empty:
        return "⚠️ 暂无概念板块数据"
    lines = ["\n## 💡 热点概念TOP20\n"]
    pct_col = next((c for c in df.columns if '涨跌幅' in c), None)
    name_col = next((c for c in df.columns if '名称' in c), None)
    if not name_col:
        return "⚠️ 无法识别概念板块列名"
    for _, r in df.head(20).iterrows():
        name = r[name_col]
        pct = r.get(pct_col, 0)
        arrow = "🟢" if isinstance(pct, (int, float)) and pct > 0 else "🔴"
        lines.append(f"  {arrow} {name}: {pct}%")
    return "\n".join(lines)


def format_block_trades(df: pd.DataFrame) -> str:
    if df.empty:
        return "⚠️ 暂无大宗交易数据"
    lines = ["\n## 📦 大宗交易\n"]
    for _, r in df.head(10).iterrows():
        stock = str(r.get('证券简称', r.get('股票名称', '')))
        price = r.get('成交价', r.get('价格', ''))
        vol = r.get('成交量', r.get('成交量(股)', ''))
        premium = r.get('折溢价率', r.get('溢价率', ''))
        lines.append(f"  {stock} 价{price} 量{vol} 折溢价{premium}")
    return "\n".join(lines)


def format_full_report() -> str:
    parts = []
    now = datetime.now(CST)
    parts.append(f"# 📊 A股特色数据 ({now.strftime('%Y-%m-%d %H:%M')})")
    parts.append(f"{'='*50}")

    # 龙虎榜
    dragon = get_top_dragon_tiger()
    parts.append(format_dragon_tiger(dragon))

    # 概念板块
    concept = get_concept_board_top()
    parts.append(format_concept_top(concept))

    # 概念资金流
    cf = get_concept_fund_flow()
    if not cf.empty:
        parts.append(f"\n概念资金流数据已获取 ({len(cf)}条)")

    # 大宗交易
    block = get_block_trades()
    parts.append(format_block_trades(block))

    return "\n".join(parts)


# ───────── CLI ─────────

def main():
    args = set(sys.argv[1:])

    if "--dragon" in args:
        df = get_top_dragon_tiger()
        print(format_dragon_tiger(df))
    elif "--block-trade" in args:
        df = get_block_trades()
        print(format_block_trades(df))
    elif "--holder" in args:
        df = get_shareholder_changes()
        print("\n## 📋 股东增减持\n")
        if not df.empty:
            print(df.head(20).to_string())
        else:
            print("⚠️ 暂无数据")
    elif "--concept" in args:
        df = get_concept_board_top()
        print(format_concept_top(df))
    else:
        print(format_full_report())


if __name__ == "__main__":
    main()
