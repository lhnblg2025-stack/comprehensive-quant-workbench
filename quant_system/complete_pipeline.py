"""
complete_pipeline.py — 端到端量化管道
V4.1 feature

一站式：数据获取 → 因子计算 → 信号生成 → 回测 → 分析 → 报告
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

try:
    from .config import DEFAULT_PORTFOLIO, PortfolioConfig
except ImportError:  # 允许以独立脚本方式运行
    from config import DEFAULT_PORTFOLIO, PortfolioConfig

logger = __import__('logging').getLogger(__name__)


class CompletePipeline:
    """端到端量化管道"""

    def __init__(self, data_dir: str = "~/.quant_system/"):
        self.data_dir = data_dir
        self._prices: Optional[pd.DataFrame] = None
        self._factors: Optional[pd.DataFrame] = None
        self._signals: Optional[pd.Series] = None
        self._returns: Optional[pd.Series] = None
        self._report: Optional[str] = None

    def step1_load_data(self, symbols: Optional[list] = None,
                        start: str = "20200101",
                        end: Optional[str] = None) -> pd.DataFrame:
        """步骤1: 加载价格数据"""
        import akshare as ak
        end = end or datetime.now().strftime("%Y%m%d")

        if symbols is None:
            try:
                spot = ak.stock_zh_a_spot_em()
                symbols = spot["代码"].astype(str).str.zfill(6).tolist()[:100]
            except Exception:
                symbols = ["000001", "000002", "600519", "000858", "002415"]

        prices = pd.DataFrame()
        for sym in symbols:
            try:
                df = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                         start_date=start, end_date=end)
                if df is not None and not df.empty:
                    df.columns = [c.strip() for c in df.columns]
                    prices[sym] = df.set_index("日期")["收盘"]
            except Exception as e:
                logger.warning(f"加载{sym}失败: {e}")

        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        self._prices = prices
        logger.info(f"[Pipeline] 加载 {prices.shape[1]} 只股票, {prices.shape[0]} 天")
        return prices

    def step2_compute_factors(self, method: str = "all") -> pd.DataFrame:
        """步骤2: 计算因子（多标的面板）

        修复 Q25: 原实现 `factors["mom_20d"] = close.pct_change(20)` 把
        多列 DataFrame 赋给单列 → ValueError 必然崩溃。现按因子×标的组织为
        MultiIndex 列 (factor, symbol)，保留多标的截面，与 step4 的
        `_prices.mean(axis=1)` 设计一致。
        """
        if self._prices is None:
            raise ValueError("请先运行 step1_load_data()")

        close = self._prices

        factor_data: dict[str, pd.DataFrame] = {}

        # 动量因子
        factor_data["mom_20d"] = close.pct_change(20)
        factor_data["mom_60d"] = close.pct_change(60)
        factor_data["mom_120d"] = close.pct_change(120)

        # 反转因子 (V4.1 fix: use shift(1) to avoid lookahead — only use prior day return)
        factor_data["rev_1d"] = -close.pct_change(1).shift(1).fillna(0)
        factor_data["rev_5d"] = -close.pct_change(5).shift(1).fillna(0)

        # 波动率因子
        ret = close.pct_change()
        factor_data["vol_20d"] = ret.rolling(20).std()
        factor_data["vol_60d"] = ret.rolling(60).std()
        factor_data["max_dd_20d"] = ret.rolling(20).apply(
            lambda x: (1 + x).cumprod().div((1 + x).cumprod().expanding().max()).min() - 1
        )

        # 技术因子
        factor_data["ma5_pct"] = close / close.rolling(5).mean() - 1
        factor_data["ma20_pct"] = close / close.rolling(20).mean() - 1
        factor_data["ma60_pct"] = close / close.rolling(60).mean() - 1

        # 成交量因子
        if hasattr(self, '_volume') and self._volume is not None:
            vol = self._volume
            factor_data["vol_ratio"] = vol / vol.rolling(20).mean()

        # 合并为 MultiIndex 列 (factor, symbol)
        factors = pd.concat(factor_data, axis=1)
        factors.columns = pd.MultiIndex.from_tuples(factors.columns)

        self._factors = factors
        logger.info(
            f"[Pipeline] 计算 {len(factor_data)} 个因子 × {close.shape[1]} 只股票"
        )
        return factors

    def step3_generate_signals(self, method: str = "rank") -> pd.Series:
        """步骤3: 生成交易信号"""
        if self._factors is None:
            raise ValueError("请先运行 step2_compute_factors()")

        # 因子合成 (V4.1 fix: use expanding window instead of full-dataset mean/std)
        zs = self._factors.apply(
            lambda x: (x - x.expanding().mean()) / x.expanding().std().clip(lower=1e-12)
        )
        zs = zs.clip(-3, 3)

        # 简单等权合成
        if method == "rank":
            ranks = self._factors.rank(pct=True)
            composite = ranks.mean(axis=1)
        else:
            composite = zs.mean(axis=1)

        # 转化为信号 (修复 Q25: 阈值改用 expanding 分位数, 且 shift(1) 只用过去数据,
        # 避免全样本分位数造成的前视泄漏, 早期样本不再按未来分布切分)
        min_periods = min(20, len(composite) // 2) if len(composite) > 20 else max(2, len(composite) // 4)
        long_threshold = composite.expanding(min_periods=min_periods).quantile(0.7).shift(1)
        short_threshold = composite.expanding(min_periods=min_periods).quantile(0.3).shift(1)
        signals = pd.Series(0, index=composite.index, dtype=int)
        signals[composite > long_threshold] = 1
        signals[composite < short_threshold] = -1

        self._signals = signals
        logger.info(f"[Pipeline] 信号: 多头{sum(signals==1)}, 空头{sum(signals==-1)}, 空仓{sum(signals==0)}")
        return signals

    def step4_backtest(self, initial_capital: float = 1_000_000,
                       commission: Optional[float] = None,
                       portfolio: Optional[PortfolioConfig] = None) -> pd.Series:
        """步骤4: 执行回测

        # P2-Q25-fix(M293): 成本模型由 `tc = turnover * commission`(单边 0.0003)
        升级为 config.PortfolioConfig 的 A股 契约模型: 佣金(最低 5 元/笔) +
        印花税(卖出 0.05%) + 过户费(0.001% 双边) + 滑点。旧 `commission` 参数
        保留为可选覆盖值(显式传入时覆盖配置佣金率), 不破坏既有调用方。
        """
        if self._signals is None or self._prices is None:
            raise ValueError("请先运行 step3_generate_signals()")

        if portfolio is None:
            portfolio = DEFAULT_PORTFOLIO
        if commission is not None:
            from dataclasses import replace
            portfolio = replace(portfolio, commission_pct=commission)

        close = self._prices.mean(axis=1)
        ret = close.pct_change().fillna(0)

        pos = self._signals.shift(1).fillna(0)
        strategy_ret = pos * ret

        # 持仓变动(单位: 满仓倍数, 取值 -2/-1/0/1/2; 例如 -1→1 计 2 笔)
        delta = pos.diff().fillna(0)
        # 单仓名义金额: 满仓时占用资金
        unit_notional = initial_capital * portfolio.max_position_pct
        # 单边成本率: 佣金+过户费+滑点(买卖同收); 卖出另加印花税 0.05%
        buy_rate = portfolio.commission_pct + portfolio.transfer_fee_pct + portfolio.slippage_pct
        sell_rate = (portfolio.commission_pct + portfolio.stamp_tax_pct
                     + portfolio.transfer_fee_pct + portfolio.slippage_pct)

        def _leg_cost(legs: float, rate: float) -> float:
            """按腿计费: 每腿 max(名义*费率, 最低佣金 5 元)。"""
            if legs <= 0:
                return 0.0
            return legs * max(unit_notional * rate, portfolio.min_commission)

        buy_legs = delta.clip(lower=0)
        sell_legs = (-delta).clip(lower=0)
        cost_yuan = (
            buy_legs.apply(lambda x: _leg_cost(x, buy_rate))
            + sell_legs.apply(lambda x: _leg_cost(x, sell_rate))
        )
        # 成本按初始资金归一化为日收益拖累
        strategy_ret = strategy_ret - cost_yuan / initial_capital

        self._returns = strategy_ret
        cum = (1 + strategy_ret).cumprod()

        ann_ret = strategy_ret.mean() * 252
        ann_vol = strategy_ret.std() * np.sqrt(252)
        sharpe = ann_ret / max(ann_vol, 1e-12)

        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = dd.min()

        logger.info(
            f"[Pipeline] 回测: 年化{ann_ret*100:.1f}% 波动{ann_vol*100:.1f}% "
            f"夏普{sharpe:.2f} 回撤{max_dd*100:.1f}%"
        )
        return strategy_ret

    def step5_analyze(self) -> dict:
        """步骤5: 绩效分析"""
        if self._returns is None:
            raise ValueError("请先运行 step4_backtest()")

        ret = self._returns
        cum = (1 + ret).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak

        analysis = {
            "total_return": float(cum.iloc[-1] - 1),
            "annual_return": float(ret.mean() * 252),
            "annual_vol": float(ret.std() * np.sqrt(252)),
            "sharpe": float(ret.mean() / max(ret.std(), 1e-12) * np.sqrt(252)),
            "sortino": float(ret.mean() / max(ret[ret < 0].std(), 1e-12) * np.sqrt(252)),
            "max_drawdown": float(dd.min()),
            "calmar": float(ret.mean() * 252 / max(abs(dd.min()), 1e-12)),
            "win_rate": float((ret > 0).mean()),
            # P2-Q25-fix(L308): 盈亏比改用业界标准定义 总盈利/总亏损,
            # 而非平均盈利/平均亏损。
            "profit_factor": float(
                ret[ret > 0].sum() / max(abs(ret[ret < 0].sum()), 1e-12)
            ),
            "var_95": float(ret.quantile(0.05)),
            "cvar_95": float(ret[ret <= ret.quantile(0.05)].mean()),
            "positive_months": float(
                ret.resample("ME").apply(lambda x: (1 + x).prod() - 1 > 0).mean()
            ),
            "n_days": len(ret),
            "start_date": str(ret.index[0])[:10] if len(ret) > 0 else "",
            "end_date": str(ret.index[-1])[:10] if len(ret) > 0 else "",
        }
        return analysis

    def step6_report(self) -> str:
        """步骤6: 生成综合报告"""
        if self._returns is None:
            return "请先完成所有步骤"

        analysis = self.step5_analyze()
        lines = [
            "=" * 55,
            "量化管道综合报告 (V4.1 feature)",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
            "",
            "【绩效指标】",
            f"  总收益:     {analysis['total_return']*100:.2f}%",
            f"  年化收益:   {analysis['annual_return']*100:.2f}%",
            f"  年化波动:   {analysis['annual_vol']*100:.2f}%",
            f"  夏普比率:   {analysis['sharpe']:.3f}",
            f"  Sortino:    {analysis['sortino']:.3f}",
            f"  Calmar:     {analysis['calmar']:.3f}",
            f"  最大回撤:   {analysis['max_drawdown']*100:.2f}%",
            f"  胜率:       {analysis['win_rate']*100:.1f}%",
            f"  盈亏比:     {analysis['profit_factor']:.2f}",
            f"  VaR(95%):   {analysis['var_95']*100:.2f}%",
            f"  CVaR(95%):  {analysis['cvar_95']*100:.2f}%",
            "",
            "【回测信息】",
            f"  数据范围:   {analysis['start_date']} ~ {analysis['end_date']}",
            f"  交易天数:   {analysis['n_days']}",
            f"  正收益月:   {analysis['positive_months']*100:.1f}%",
            "",
            "【因子表现】",
        ]
        if self._factors is not None:
            ic_series = self._factors.corrwith(
                self._returns.shift(-1).fillna(0),
                axis=0
            ).dropna().sort_values(ascending=False)
            for name, val in ic_series.head(5).items():
                lines.append(f"  {name}: IC={val:.4f}")
            lines.append("")
            lines.append(f"  因子总数: {self._factors.shape[1]}")
            lines.append(f"  Best IC: {ic_series.iloc[0]:.4f} ({ic_series.index[0]})")
            lines.append(f"  Worst IC: {ic_series.iloc[-1]:.4f} ({ic_series.index[-1]})")

        lines.append("")
        lines.append("=" * 55)
        self._report = "\n".join(lines)
        return self._report

    def run_all(self, symbols: Optional[list] = None,
                start: str = "20200101") -> dict:
        """一键运行全流程"""
        logger.info("[Pipeline] 开始全流程...")
        self.step1_load_data(symbols, start)
        self.step2_compute_factors()
        self.step3_generate_signals()
        self.step4_backtest()
        analysis = self.step5_analyze()
        report = self.step6_report()
        return {"analysis": analysis, "report": report, "status": "success"}

    def market_forecast(self) -> dict:
        """指数级市场预测（market_forecast 引擎，原 QuantV6 预测层移植）。

        懒加载：预测引擎缺失/异常时返回 empty 结果，不影响原管道。
        输出 SignalCenter 综合预测（regime 状态机 + 情绪周期 + k-NN 相似日
        + 次日 ML 概率 + 尾部风险 + 动量衰竭 + 波动率预测）。
        """
        try:
            from quant_system.market_forecast import SignalCenter
        except Exception as e:
            logger.warning(f"market_forecast 不可用: {e}")
            return {"ok": False, "error": str(e)}
        try:
            sc = SignalCenter()
            signal = sc.generate()
            md = sc.to_markdown(signal) if signal is not None else ""
            return {"ok": True, "signal": signal, "markdown": md}
        except Exception as e:
            logger.warning(f"market_forecast 运行失败: {e}")
            return {"ok": False, "error": str(e)}



class BatchRunner:
    """批量策略运行器"""

    def __init__(self):
        self._results: list[dict] = []

    def run_multiple(self, configs: list[dict]) -> pd.DataFrame:
        """运行多个配置"""
        from quant_system.public_strategies import list_public_strategies as get_all_strategies
        all_strats = get_all_strategies()

        for cfg in configs:
            try:
                prices = pd.DataFrame()
                import akshare as ak
                symbols = cfg.get("symbols", ["000001", "600519"])
                start = cfg.get("start", "20200101")
                for sym in symbols:
                    df = ak.stock_zh_a_hist(symbol=sym, start_date=start)
                    if df is not None:
                        df.columns = [c.strip() for c in df.columns]
                        prices[sym] = df.set_index("日期")["收盘"]
                prices.index = pd.to_datetime(prices.index)

                # 运行每种策略
                strategy_names = cfg.get("strategies", ["double_ma", "bollinger", "rsi"])
                for sname in strategy_names:
                    if sname not in all_strats:
                        continue
                    strat = all_strats[sname]()
                    sig = strat.generate(prices)
                    ret = prices.pct_change().fillna(0).mean(axis=1)
                    strategy_ret = sig.shift(1).fillna(0) * ret
                    sharpe = strategy_ret.mean() / max(strategy_ret.std(), 1e-12) * np.sqrt(252)
                    self._results.append({
                        "strategy": sname,
                        "sharpe": sharpe,
                        "total_return": float((1 + strategy_ret).prod() - 1),
                        "max_dd": float((1 + strategy_ret).cumprod().div(
                            (1 + strategy_ret).cumprod().expanding().max()).min() - 1),
                    })
            except Exception as e:
                logger.error(f"配置运行失败: {e}")

        return pd.DataFrame(self._results)


def _board_limit_pct(code, name: str = "") -> float:
    """按板块/ST 返回 A股 涨跌停阈值(百分比数值, 如 10.0)。

    主板 10% / 创业板·科创板 20% / 北交所 30% / ST 5%。
    # P2-Q25-fix(M296): 原固定 ±9.5% 会漏计创业板/科创板 20cm 涨停与 ST 5% 涨停。
    """
    code = str(code).zfill(6)
    if "ST" in str(name).upper():
        return 5.0
    if code.startswith(("688", "689", "300", "301", "302")):
        return 20.0
    if code.startswith(("4", "8", "92")):
        return 30.0
    return 10.0


class MarketScanner:
    """市场扫描器"""

    def __init__(self):
        self._alerts: list[dict] = []

    def scan_all(self, top_n: int = 100) -> pd.DataFrame:
        """扫描全市场"""
        import akshare as ak
        try:
            spot = ak.stock_zh_a_spot_em()
            if spot is None or spot.empty:
                return pd.DataFrame()

            spot.columns = [c.strip() for c in spot.columns]
            result = spot.rename(columns={
                "代码": "code", "名称": "name", "最新价": "price",
                "涨跌幅": "pct_chg", "成交量": "volume", "成交额": "amount",
                "换手率": "turnover", "市盈率-动态": "pe",
                "市净率": "pb",
            })

            # 筛选条件
            result = result[result["pct_chg"].notna()]
            result["pct_chg"] = result["pct_chg"].astype(float)

            # 异常波动: 按板块/ST 动态涨跌停阈值(保留 0.5% 缓冲, 与 10% 档口径一致)
            # P2-Q25-fix(M296): 创业板/科创板 20cm、北交所 30%、ST 5% 分别按各自阈值识别。
            limit_pct = result.apply(
                lambda r: _board_limit_pct(r.get("code", ""), r.get("name", "")),
                axis=1,
            )
            up_limit = result[result["pct_chg"] > limit_pct - 0.5]
            down_limit = result[result["pct_chg"] < -(limit_pct - 0.5)]
            high_vol = result[result["pct_chg"].abs() > 7]

            self._alerts.append({
                "type": "涨停", "count": len(up_limit),
                "top": up_limit.head(5).get("name", "").tolist(),
            })
            self._alerts.append({
                "type": "跌停", "count": len(down_limit),
                "top": down_limit.head(5).get("name", "").tolist(),
            })
            self._alerts.append({
                "type": "异动", "count": len(high_vol),
            })

            # P2-Q25-fix(M296): 按 |涨跌幅| 降序排序后再取 top_n, 避免未排序随机截取。
            result = result.loc[result["pct_chg"].abs().sort_values(ascending=False).index]
            return result.head(top_n)
        except Exception as e:
            logger.error(f"市场扫描失败: {e}")
            return pd.DataFrame()

    def scan_report(self) -> str:
        """扫描报告"""
        self.scan_all()
        lines = [
            "=" * 55,
            f"市场扫描报告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            "=" * 55,
        ]
        for a in self._alerts:
            lines.append(f"  {a['type']}: {a['count']}只")
            if "top" in a and a["top"]:
                lines.append(f"    例: {', '.join(str(x) for x in a['top'][:3])}")
        return "\n".join(lines)


__all__ = ["CompletePipeline", "BatchRunner", "MarketScanner"]
