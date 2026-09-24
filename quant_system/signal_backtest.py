"""
量化交易系统 — 信号回测验证。

对买入信号进行历史回测：RSI超卖后买入的胜率？CCI底背离的胜率？MA60支撑的胜率？

基于 backtesting-framework skill 的纯 NumPy 回测引擎。

用法:
  python3 -m quant_system.signal_backtest --signal rsi_oversold --symbol 600519
  python3 -m quant_system.signal_backtest --signal all --top 10
"""

from __future__ import annotations

import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))
TENCENT_HEADERS = {"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"}


# ════════════════════════════════════════════════════════════════
# 1. 数据获取
# ════════════════════════════════════════════════════════════════

def fetch_history(symbol: str, days: int = 500) -> np.ndarray | None:
    """
    从AKShare获取历史日K线。

    # P2-Q9-fix (Q9-M555): days 参数原先从未使用、start_date 硬编码 20230101，
    # 样本期锁死、days=500 形同虚设。现用 days 推算起始日历日（交易日≈日历
    # 日×1.6），并裁剪到最近 days 根K线，使回测窗口可配置。

    Returns: np.array of [open, close, high, low, volume] per row
    """
    try:
        import akshare as ak
        start = (datetime.now(CST) - timedelta(days=int(days * 1.6) + 30)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start,
                                adjust="qfq")
        if df is None or len(df) < 100:
            return None

        arr = np.array([
            [float(row["开盘"]), float(row["收盘"]),
             float(row["最高"]), float(row["最低"]), float(row["成交量"])]
            for _, row in df.iterrows()
        ])
        if len(arr) > days:
            arr = arr[-days:]
        return arr
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# 2. 信号回测引擎
# ════════════════════════════════════════════════════════════════

def _sma(data: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average."""
    result = np.zeros(len(data))
    for i in range(len(data)):
        if i < n:
            result[i] = np.mean(data[:i+1])
        else:
            result[i] = np.mean(data[i-n+1:i+1])
    return result


def _rsi(prices: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI indicator."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    rs = np.zeros(len(prices))
    rsi = np.zeros(len(prices))
    
    for i in range(n, len(prices)):
        if i == n:
            avg_gain[i] = np.mean(gains[:n])
            avg_loss[i] = np.mean(losses[:n])
        else:
            avg_gain[i] = (avg_gain[i-1] * (n-1) + gains[i-1]) / n
            avg_loss[i] = (avg_loss[i-1] * (n-1) + losses[i-1]) / n
        
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - 100 / (1 + rs[i])
        else:
            rsi[i] = 100
    
    return rsi


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 20) -> np.ndarray:
    """CCI indicator."""
    tp = (high + low + close) / 3
    cci = np.zeros(len(tp))
    for i in range(n, len(tp)):
        # P2-Q9-fix (Q9-L558): 窗口改为 tp[i-n+1:i+1]（含当前 bar），与标准
        # CCI 定义一致；原 tp[i-n:i] 不含当前 bar，指标系统性滞后 1 根。
        mean_tp = np.mean(tp[i-n+1:i+1])
        md = np.mean(np.abs(tp[i-n+1:i+1] - mean_tp))
        if md > 0:
            cci[i] = (tp[i] - mean_tp) / (0.015 * md)
    return cci


# ════════════════════════════════════════════════════════════════
# 3. 信号定义 & 回测
# ════════════════════════════════════════════════════════════════

_SIGNAL_DEFS = {
    "rsi_oversold": {
        "name": "RSI超卖买入",
        "desc": "RSI(14) < 35时买入,持有5个交易日",
        "hold_days": 5,
    },
    "rsi_oversold_10d": {
        "name": "RSI超卖买入(10日)",
        "desc": "RSI(14) < 35时买入,持有10个交易日",
        "hold_days": 10,
    },
    "cci_oversold": {
        "name": "CCI超卖买入",
        "desc": "CCI(20) < -100时买入,持有5个交易日",
        "hold_days": 5,
    },
    "boll_lower": {
        "name": "BOLL下轨买入",
        "desc": "价格触及BOLL(20,2)下轨时买入,持有5日",
        "hold_days": 5,
    },
    "ma60_bounce": {
        "name": "MA60支撑买入",
        "desc": "价格回调至MA60附近(±1%)且收涨时买入,持有5日",
        "hold_days": 5,
    },
}


def _cost_rates(notional: float | None = None) -> tuple[float, float]:
    """返回 (买入成本率, 卖出成本率) 占成交金额的比例。

    # P2-Q9-fix (Q9-M557): 回测原先不含交易成本，胜率/收益系统性高估。
    # 按 config.PortfolioConfig 计入：佣金(万0.85，最低5元/笔)、卖出印花税
    # (万5)、过户费(双边0.001%)、滑点(0.1%)。本回测只算收益率、不跟踪资金
    # 规模，单笔名义金额按默认组合单笔上限 initial_cash*max_position_pct
    # 折算最低佣金（默认 5/200000=0.0025% < 万0.85，故最低佣金通常不触发）。

    # P1-6 (审计回测层) 注意——最低佣金口径假设（已在函数名/文档明示）：
    #   本回测不跟踪资金规模/成交笔数，无法按"实际每笔 max(amount*rate, 5元)"
    #   计算最低佣金，故把单笔最低佣金 5 元按"单笔名义金额 notional"折算成
    #   百分比率摊入（min_commission / notional）。这与 backtest.py/backtest_pro
    #   按实盘每笔成交额 max(amount*rate, 5) 的计法在数值上等价，仅当单笔
    #   实际成交额 = notional 时严格成立；若单笔成交额显著小于 notional，则
    #   本折算会低估最低佣金，属于已知假设，不做静默差异。
    """
    try:
        from quant_system.config import PortfolioConfig
    except ImportError:
        # 直接运行脚本（不在包内）时的兜底，避免 import 失败拖垮回测
        from config import PortfolioConfig
    cfg = PortfolioConfig()
    if notional is None:
        notional = cfg.initial_cash * cfg.max_position_pct
    comm = max(cfg.commission_pct, cfg.min_commission / max(notional, 1e-9))
    buy = comm + cfg.transfer_fee_pct + cfg.slippage_pct
    sell = comm + cfg.stamp_tax_pct + cfg.transfer_fee_pct + cfg.slippage_pct
    return buy, sell


def backtest_signal(
    ohlcv: np.ndarray,
    signal_name: str,
    hold_days: int = 5,
    lookback: int = 60,
) -> dict:
    """
    对单个信号的通用回测。

    # P2-Q9-fix (Q9-M557): 已计入 A股交易成本（佣金/卖出印花税/过户费/滑点，
    # 见 _cost_rates），收益/胜率不再系统性高估。注意幸存者偏差：回测样本仅
    # 含当前仍上市的标的，退市/被ST标的未纳入，实际信号胜率通常低于回测值。

    Args:
        ohlcv: [open, close, high, low, volume] per row
        signal_name: signal identifier
        hold_days: 持有天数
        lookback: 至少需要多少根K线作为预热

    Returns:
        {total_trades, win_rate, avg_return, total_return, sharpe,
         max_drawdown, profit_factor, ...}
    """
    opens = ohlcv[:, 0]
    closes = ohlcv[:, 1]
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    volumes = ohlcv[:, 4]
    
    n = len(closes)
    
    # Compute indicators
    rsi_vals = _rsi(closes, 14)
    cci_vals = _cci(highs, lows, closes, 20)
    ma20 = _sma(closes, 20)
    ma60 = _sma(closes, 60)
    
    # Bollinger bands
    rolling_std = np.zeros(n)
    for i in range(20, n):
        rolling_std[i] = np.std(closes[i-19:i+1])
    boll_upper = ma20 + 2 * rolling_std
    boll_lower = ma20 - 2 * rolling_std
    
    # Generate entry signals
    entries = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        if signal_name == "rsi_oversold":
            # P2-Q9-fix (Q9-M556): 去掉 RSI>0 守卫——RSI=0（窗口内全跌，最强
            # 超卖）被排除导致最强信号永不触发；预热由循环起点 lookback 保证。
            if rsi_vals[i] < 35:
                entries[i] = True
        elif signal_name == "rsi_oversold_10d":
            if rsi_vals[i] < 35:
                entries[i] = True
        elif signal_name == "cci_oversold":
            if cci_vals[i] < -100:
                entries[i] = True
        elif signal_name == "boll_lower":
            if boll_lower[i] > 0 and closes[i] <= boll_lower[i]:
                entries[i] = True
        elif signal_name == "ma60_bounce":
            if ma60[i] > 0 and abs(closes[i] - ma60[i]) / ma60[i] < 0.01 and closes[i] >= opens[i]:
                entries[i] = True
    
    # P2-Q9-fix (Q9-M557): 计入 A股交易成本（佣金/印花税/过户费/滑点），
    # 见 _cost_rates()。buy_cost/sell_cost 为占成交金额的比例成本率。
    buy_cost, sell_cost = _cost_rates()

    # Simulate trades
    trade_returns = []
    # P2-Q9-fix (Q9-L559): trade_dates 收集后从未返回/使用（死变量），删除。

    i = lookback
    while i < n:
        if entries[i]:
            # 信号产生于收盘后，因此以次日开盘价入场
            entry_idx = i + 1
            if entry_idx >= n:
                i += 1
                continue
            entry_price = opens[entry_idx]

            # 持有 hold_days 个交易日，以收盘价出场
            exit_idx = min(entry_idx + hold_days, n - 1)
            exit_price = closes[exit_idx]

            # 买入按 entry_price*(1+buy_cost) 成交，卖出按 exit_price*(1-sell_cost)
            # 成交；成本均摊到收益率。
            trade_return = (exit_price * (1 - sell_cost)) / (entry_price * (1 + buy_cost)) - 1
            trade_returns.append(trade_return)

            i = exit_idx + 1  # Skip forward to avoid overlapping trades
        else:
            i += 1
    
    # Compute metrics
    if not trade_returns:
        return {
            "signal": signal_name,
            "total_trades": 0,
            "error": "No trades generated",
        }
    
    returns_arr = np.array(trade_returns)
    wins = returns_arr > 0
    losses = returns_arr <= 0
    
    win_rate = np.sum(wins) / len(returns_arr) * 100
    avg_return = np.mean(returns_arr) * 100
    total_return = (np.prod(1 + returns_arr) - 1) * 100
    
    # Sharpe (V12.3 审计 P1-4): 统一口径——扣无风险利率(rf=0.02)、样本标准差
    # ddof=1，年化 sqrt(252/hold_days)。原实现 rf=0 与其它引擎(扣0.02)不可比；
    # 现通过 metrics_calculator.MetricsCalculator.sharpe 复用同一代码路径，
    # trading_days=252/hold_days 保持本引擎"按持有期年化"的既有缩放。
    if np.std(returns_arr) > 0:
        from quant_system.metrics_calculator import MetricsCalculator
        sharpe = MetricsCalculator.sharpe(
            returns_arr, rf=0.02, ddof=1,
            trading_days=252.0 / hold_days, annualize=True,
        )
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = np.cumprod(1 + returns_arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = np.min(drawdown) * 100
    
    # Profit factor
    gross_profit = np.sum(returns_arr[returns_arr > 0])
    gross_loss = abs(np.sum(returns_arr[returns_arr < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Avg win / avg loss
    avg_win = np.mean(returns_arr[wins]) * 100 if np.any(wins) else 0
    avg_loss = np.mean(returns_arr[losses]) * 100 if np.any(losses) else 0
    
    return {
        "signal": signal_name,
        "symbol": "AGGREGATE",
        "total_trades": len(returns_arr),
        "win_rate": round(win_rate, 1),
        "avg_return_pct": round(avg_return, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "hold_days": hold_days,
    }


def backtest_stock(symbol: str, signal_name: str = "all", hold_days: int = 5) -> list[dict]:
    """
    对一只股票的一个或多个信号做回测。
    
    Args:
        symbol: 股票代码
        signal_name: "all"对所有信号回测, 或具体信号名
        hold_days: 持有天数
    
    Returns:
        每个信号的回测结果
    """
    ohlcv = fetch_history(symbol, days=500)
    if ohlcv is None or len(ohlcv) < 100:
        return [{"error": f"Cannot fetch data for {symbol}", "symbol": symbol}]
    
    results = []
    if signal_name == "all":
        for sname in _SIGNAL_DEFS:
            info = _SIGNAL_DEFS[sname]
            result = backtest_signal(ohlcv, sname, info["hold_days"])
            result["symbol"] = symbol
            result["signal_label"] = info["name"]
            results.append(result)
    else:
        info = _SIGNAL_DEFS.get(signal_name)
        if not info:
            return [{"error": f"Unknown signal: {signal_name}", "symbol": symbol}]
        result = backtest_signal(ohlcv, signal_name, hold_days)
        result["symbol"] = symbol
        result["signal_label"] = info["name"]
        results.append(result)
    
    return results


# ════════════════════════════════════════════════════════════════
# 4. 格式化 & CLI
# ════════════════════════════════════════════════════════════════

def format_result(results: list[dict]) -> str:
    """Format backtest results."""
    lines = []
    for r in results:
        if "error" in r:
            lines.append(f"⚠️ {r.get('error', 'Unknown error')}")
            continue
        
        signal_label = r.get("signal_label", r["signal"])
        lines.append(f"\n{'='*50}")
        lines.append(f"🔬 **{signal_label}** — {r['symbol']}")
        lines.append(f"{'='*50}")
        lines.append(f"  交易次数: {r['total_trades']}")
        lines.append(f"  🏆 **胜率: {r['win_rate']}%**" if r.get("win_rate", 0) > 50
                    else f"  📉 胜率: {r['win_rate']}%")
        lines.append(f"  平均收益: {r['avg_return_pct']:+.2f}%")
        lines.append(f"  总收益: {r['total_return_pct']:+.2f}%")
        lines.append(f"  夏普比率: {r['sharpe']}")
        lines.append(f"  最大回撤: {r['max_drawdown_pct']:.1f}%")
        lines.append(f"  盈亏比: {r['profit_factor']}")
        lines.append(f"  平均盈利: {r['avg_win_pct']:+.2f}% | 平均亏损: {r['avg_loss_pct']:+.2f}%")
        
        # Assessment
        wr = r.get("win_rate", 0)
        sharpe = r.get("sharpe", 0)
        if wr > 60 and sharpe > 1:
            lines.append(f"  ✅ **有效信号** — 胜率>60% 夏普>1.0")
        elif wr > 50 and sharpe > 0.5:
            lines.append(f"  ✅ 可用信号 — 略优于随机")
        else:
            lines.append(f"  ⚠️ 信号效果一般,需配合其他条件使用")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="信号回测验证")
    parser.add_argument("--signal", type=str, default="all",
                        help=f"信号名称 ({'|'.join(_SIGNAL_DEFS.keys())}|all)")
    parser.add_argument("--symbol", type=str, default="600519", help="股票代码")
    parser.add_argument("--hold", type=int, default=5, help="持有天数")
    parser.add_argument("--list", action="store_true", help="列出所有可回测信号")
    args = parser.parse_args()
    
    if args.list:
        print("📚 **可回测信号**\n")
        for key, info in _SIGNAL_DEFS.items():
            print(f"  {key}: {info['name']} — {info['desc']}")
        sys.exit(0)
    
    t0 = _time.time()
    results = backtest_stock(args.symbol, args.signal, args.hold)
    elapsed = _time.time() - t0
    print(format_result(results))
    print(f"\n⏱ 回测耗时: {elapsed:.1f}s")
    # P2-Q9-fix (Q9-M557): 成本已建模；幸存者偏差与不构成未来收益的提示保留
    print("⚠️ 回测已计入佣金/印花税/过户费/滑点（见 config.PortfolioConfig）")
    print("⚠️ 样本仅含现存标的（幸存者偏差），实际胜率通常低于回测值；不代表未来收益")
