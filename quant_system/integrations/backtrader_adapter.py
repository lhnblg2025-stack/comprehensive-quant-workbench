"""
backtrader_adapter — 独立回测验证适配层 (V5)

用途：用 backtrader 独立运行策略，验证 backtest_engine.py 结果的准确性。
  - 相同的策略逻辑 → 不同引擎执行 → 对比绩效
  - 差异分析：成交价/滑点/佣金哪个环节不对
  - 经典策略模板：SMA交叉/海龟/波动率突破

降级策略: backtrader不可用时，仅输出提示。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    _HAS_BT = True
except ImportError:
    _HAS_BT = False


class BacktraderValidator:
    """回测结果验证器（封装backtrader）。"""

    def __init__(self) -> None:
        self.available = _HAS_BT

    def run_sma_cross(self, data: pd.DataFrame, fast: int = 5,
                      slow: int = 20, commission: float = 0.00085,
                      stake: int = 100) -> dict:
        """运行双均线交叉策略并返回绩效。

        Args:
            data: OHLCV数据 (columns: open,high,low,close,volume)
            fast: 快线周期
            slow: 慢线周期
            commission: 佣金率
            stake: 每笔股数

        Returns:
            {total_return, sharpe, max_drawdown, total_trades,
             win_rate, avg_win, avg_loss}
        """
        if not self.available:
            return self._no_bt_result()

        try:
            import backtrader as bt

            cerebro = bt.Cerebro()
            cerebro.addstrategy(SmaCrossStrategy, fast=fast, slow=slow)

            # 转换数据格式
            data_feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(data_feed)

            # 佣金
            # P2-Q26-fix: 显式 stocklike=True + commtype=COMM_PERC——原未设 stocklike
            # 使 comminfo 按期货处理（stocklike=False, commtype=None），mult=1.0 下
            # 佣金数额恰好等价，但 mult 一旦被改动即算错。
            cerebro.broker.setcommission(
                commission=commission,
                margin=None,
                mult=1.0,
                stocklike=True,
                commtype=bt.CommInfoBase.COMM_PERC,
            )
            # P2-Q26-fix: 使用 stake 参数（原声明但未使用，backtrader 默认每笔1股，
            # 回测结果与 A 股 100 股整手严重失真）。
            cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

            # 分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

            results = cerebro.run()
            strat = results[0]

            return self._extract_stats(strat)

        except Exception as e:
            return {"error": str(e), "available": False}

    def run_custom_strategy(self, strategy_class, data: pd.DataFrame,
                            **kwargs) -> dict:
        """运行自定义策略。"""
        if not self.available:
            return self._no_bt_result()

        try:
            import backtrader as bt
            cerebro = bt.Cerebro()
            cerebro.addstrategy(strategy_class, **kwargs)
            data_feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(data_feed)
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            # P2-Q26-fix: 补加 Returns analyzer——_extract_stats 读取
            # strat.analyzers.returns，缺失时自定义策略 total_return 恒为 0，
            # compare_with_our_result 对比失真。
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

            results = cerebro.run()
            return self._extract_stats(results[0])
        except Exception as e:
            return {"error": str(e)}

    def _extract_stats(self, strat) -> dict:
        """从策略结果提取统计。"""
        result = {}

        # Sharpe
        try:
            sharpe = strat.analyzers.sharpe.get_analysis()
            result["sharpe"] = round(float(sharpe.get("sharperatio", 0) or 0), 2)
        except Exception:
            result["sharpe"] = 0

        # Drawdown
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            result["max_drawdown"] = round(float(dd.get("max", {}).get("drawdown", 0) or 0) / 100, 4)
        except Exception:
            result["max_drawdown"] = 0

        # Trades
        try:
            t = strat.analyzers.trades.get_analysis()
            total = t.get("total", {})
            won = t.get("won", {})
            lost = t.get("lost", {})

            result["total_trades"] = total.get("total", 0)
            result["won"] = won.get("total", 0)
            result["lost"] = lost.get("total", 0)
            total_closed = result["won"] + result["lost"]
            result["win_rate"] = round(result["won"] / max(total_closed, 1), 4)
            result["avg_win"] = round(float(won.get("pnl", {}).get("average", 0) or 0), 2)
            result["avg_loss"] = round(float(lost.get("pnl", {}).get("average", 0) or 0), 2)
        except Exception:
            result["total_trades"] = 0

        # Returns
        try:
            r = strat.analyzers.returns.get_analysis()
            result["total_return"] = round(float(r.get("rtot", 0) or 0), 4)
            result["annual_return"] = round(float(r.get("rnorm100", 0) or 0) / 100, 4)
        except Exception:
            result["total_return"] = 0

        return result

    def compare_with_our_result(self, bt_result: dict,
                                 our_result: dict) -> dict:
        """对比 BT 与自有引擎的结果。

        Args:
            bt_result: run_sma_cross 的输出
            our_result: backtest_engine 的输出

        Returns:
            {match: bool, differences: [{metric, bt, ours, diff_pct}]}
        """
        metrics = ["total_return", "sharpe", "max_drawdown",
                   "total_trades", "win_rate"]

        differences = []
        all_match = True

        for metric in metrics:
            bt_val = bt_result.get(metric, 0)
            our_val = our_result.get(metric, 0)

            if bt_val == 0 and our_val == 0:
                continue

            diff_pct = abs(bt_val - our_val) / max(abs(bt_val), 1e-8)
            match = diff_pct < 0.05  # 5%误差容忍

            differences.append({
                "metric": metric,
                "bt": bt_val,
                "ours": our_val,
                "diff_pct": round(float(diff_pct), 4),
                "match": match,
            })
            if not match:
                all_match = False

        return {
            "match": all_match,
            "match_count": sum(1 for d in differences if d["match"]),
            "total_metrics": len(differences),
            "differences": differences,
        }

    def _no_bt_result(self) -> dict:
        return {
            "available": False,
            "error": "backtrader not installed (pip install backtrader)",
        }


# ── 内置策略定义 ──

# P2-Q26-fix: 删除恒被覆盖的死代码 SmaCrossStrategy（该首个类定义无论是否安装
# backtrader，都会被下方真实 bt.Strategy 版本（:206 起）或 else 降级桩类覆盖，
# 从未被实际使用）。

if _HAS_BT:
    import backtrader as _bt_inner

    class SmaCrossStrategy(_bt_inner.Strategy):
        """backtrader SMA交叉策略。"""
        params = (("fast", 5), ("slow", 20))

        def __init__(self):
            self.sma_fast = _bt_inner.indicators.SMA(self.data.close, period=self.params.fast)
            self.sma_slow = _bt_inner.indicators.SMA(self.data.close, period=self.params.slow)
            self.crossover = _bt_inner.indicators.CrossOver(self.sma_fast, self.sma_slow)

        def next(self):
            if not self.position and self.crossover > 0:
                self.buy()
            elif self.position and self.crossover < 0:
                self.close()
else:
    class SmaCrossStrategy:
        """降级桩类。"""
        pass


def main() -> None:
    """测试: SMA交叉 + 结果对比。"""
    np.random.seed(42)

    # 生成模拟OHLCV
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    close = 100 + np.cumsum(np.random.randn(252) * 0.5)
    data = pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1e6, 1e7, 252),
    }, index=dates)

    bv = BacktraderValidator()
    print(f"Backtrader available: {bv.available}")

    if bv.available:
        result = bv.run_sma_cross(data, fast=5, slow=20)
        print(f"\nSMA Cross results:")
        print(f"  Return: {result.get('total_return', 'N/A'):.2%}")
        print(f"  Sharpe: {result.get('sharpe', 'N/A')}")
        print(f"  Max DD: {result.get('max_drawdown', 'N/A'):.2%}")
        print(f"  Trades: {result.get('total_trades', 'N/A')}")
        print(f"  Win rate: {result.get('win_rate', 'N/A'):.0%}")
    else:
        print("Backtrader not available, install with: pip install backtrader")


if __name__ == "__main__":
    main()
