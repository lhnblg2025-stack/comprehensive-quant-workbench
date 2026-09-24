"""
动态精选池 — 行业龙头+基本面前20%+技术面有信号自动入池

功能:
  1. 基础池: 从409只中按基本面/流动性过滤
  2. 技术池: 有买入技术信号的自动加入
  3. 评分池: 多维度评分排序（ROE/PB/动量/波动）
  4. 自动更新: 每日收盘后自动轮换

用法:
  python3 -m quant_system.stock_pool                # 当前精选池
  python3 -m quant_system.stock_pool --refresh      # 强制刷新
  python3 -m quant_system.stock_pool --watch        # 持续监控新信号
"""

from __future__ import annotations
import logging

import json
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
CACHE_FILE = ROOT.parent / "config/stock_pool.json"

# P2-Q26-fix: 打分取样上限。全量 409 只打分需 ~400 次 akshare 调用（约7-10分钟），
# 因此只对 large_cap_stocks.json 前 _SCORE_LIMIT 只（按总市值降序 → 样本偏超大市值）打分，
# 该采样口径已在 build_pool docstring 注明。如需全量请调大此值。
_SCORE_LIMIT = 150


def _score_stock(code: str) -> dict:
    """对单只股票打分"""
    score = 0
    details = []
    try:
        import akshare as ak
        # 基本面
        try:
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None:
                info_dict = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
                pe = float(info_dict.get('市盈率-动态', 0) or 0)
                pb = float(info_dict.get('市净率', 0) or 0)
                mkt = float(info_dict.get('总市值', 0) or 0) / 1e8
                if 5 < pe < 40:
                    score += 2
                    details.append(f"PE={pe:.0f}合理")
                if pb < 3:
                    score += 1
                    details.append(f"PB={pb:.1f}低")
                if mkt > 500:
                    score += 1
                    details.append("大盘股")
                score = min(score, 10)
        except Exception as e:
            logging.getLogger(__name__).error(f"[stock_pool] 操作失败: {e}", exc_info=True)

        # 技术面
        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                       start_date=(datetime.now(CST)-timedelta(days=120)).strftime('%Y%m%d'),
                                       end_date=datetime.now(CST).strftime('%Y%m%d'),
                                       adjust="qfq")
            if hist is not None and len(hist) > 20:
                close = hist['收盘'].values
                vol = hist['成交量'].values
                # 动量
                chg_20d = close[-1] / close[-20] - 1 if len(close) >= 20 else 0
                if chg_20d > 0.05:
                    score += 1
                    details.append(f"20日+{chg_20d*100:.1f}%")
                # 成交量
                vol_ratio = vol[-1] / np.mean(vol[-5:-1]) if len(vol) >= 5 else 1
                if vol_ratio > 1.5:
                    score += 1
                    details.append("放量")
                # RSI
                delta = np.diff(close)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_g = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
                avg_l = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
                rsi = 100 - 100 / (1 + avg_g / avg_l) if avg_l != 0 else 100
                if 30 < rsi < 45:
                    score += 2
                    details.append(f"RSI{rsi:.0f}超卖区")
        except Exception as e:
            logging.getLogger(__name__).error(f"[stock_pool] 操作失败: {e}", exc_info=True)

    except Exception as e:
        logging.getLogger(__name__).error(f"[stock_pool] 操作失败: {e}", exc_info=True)

    return {'code': code, 'score': min(score, 10), 'details': details}


def build_pool(top_n: int = 50) -> list[dict]:
    """构建精选池。

    P2-Q26-fix: 打分样本为 watchlist 前 _SCORE_LIMIT=150 只（large_cap_stocks.json
    按总市值降序排列 → 样本偏超大市值），非全量 409 只——这是性能折衷
    （每只含 2 次 akshare 调用，全量需 7-10 分钟），已显式文档化。
    """
    cached = None
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            age = (_time.time() - cached.get('ts', 0)) / 60
            pool = cached.get('pool', [])
            # 修复 Q26: 空池缓存不生效（历史遗留的空池缓存直接跳过）
            if age < 120 and pool:
                return pool[:top_n]
        except Exception as e:
            logging.getLogger(__name__).error(f"[stock_pool] 操作失败: {e}", exc_info=True)

    # 从 watchlist 获取基础池
    try:
        from quant_system.watchlist import get_watchlist
        stocks = get_watchlist()
    except ImportError:
        return []

    # 打分（取样上限见模块常量 _SCORE_LIMIT）
    results = []
    for i, s in enumerate(stocks[:_SCORE_LIMIT]):
        # 修复 Q26: watchlist.get_watchlist() 返回键为 'symbol'
        # （large_cap_stocks.json 同），原实现读 'code' 导致全部跳过 → 恒空池
        code = str(s.get('symbol') or s.get('code') or '').strip()
        if not code:
            continue
        result = _score_stock(code)
        result['name'] = str(s.get('name', ''))
        results.append(result)
        if (i + 1) % 30 == 0:
            _time.sleep(1)

    results.sort(key=lambda x: x['score'], reverse=True)
    pool = results[:top_n]

    # 修复 Q26: 空池不落缓存，避免固化空池 2 小时（原 :99-106 直接缓存）
    if pool:
        CACHE_FILE.write_text(json.dumps({'ts': _time.time(), 'pool': pool}, ensure_ascii=False))
    return pool


def format_pool(pool: list[dict]) -> str:
    lines = ["# 🎯 精选池\n"]
    if not pool:
        lines.append("⚠️ 暂无数据，请稍后重试")
        return "\n".join(lines)
    lines.append(f"{'股票':<14}{'评分':>6}{'理由'}")
    lines.append("-" * 50)
    for p in pool[:20]:
        name = f"{p.get('name','')}({p.get('code','')})"
        score = p.get('score', 0)
        bar = "🟩" * score + "⬜" * (10 - score)
        details = ", ".join(p.get('details', []))[:40] or "-"
        lines.append(f"  {name:<12} {score}/10 {bar} {details}")
    return "\n".join(lines)


def main():
    args = set(sys.argv[1:])
    if "--refresh" in args:
        CACHE_FILE.unlink(missing_ok=True)
    pool = build_pool()
    print(format_pool(pool))


if __name__ == "__main__":
    main()
