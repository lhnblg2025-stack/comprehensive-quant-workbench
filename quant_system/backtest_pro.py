"""backtest_pro.py — 高级回测引擎（多资产/稳健性/归因）
V4.1 feature

扩展 backtest_engine.py 能力至专业级回测平台。

D2收敛登记 (2026-08-11): 与 backtest_engine.py 能力重叠的部分
（参数扫描/Walk-Forward/绩效指标/回测报告/多资产）签名不兼容，
按保守策略保留独立实现，标注 'D1/D2 收敛: 保留独立实现（能力未合并）'；
engine 未覆盖的能力（公司行为/存活偏差/子周期/归因/MC/基准统计/优化器）
标注 'D2收敛登记: 独立能力保留'。文件与公开 API 均保留，不转发、不强迁。

幸存者偏差接入 (2026-08-13): ``load_delisted_prices`` 读取
``data_warehouse/delisted.parquet`` 并从 ``data_warehouse/kline`` 提取退市前
有效收盘价；``integrate_delisted`` 将退市股列注入回测价格面板并生成死亡占比/
风险等级报告。退市日后无交易，缺失即 NaN，由引擎按无行情处理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from itertools import product

from quant_system import execution_broker as _execution_broker
from quant_system.metrics_calculator import MetricsCalculator

logger = __import__('logging').getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DELISTED_PATH = _REPO_ROOT / "data_warehouse" / "delisted.parquet"
_DEFAULT_KLINE_DIR = _REPO_ROOT / "data_warehouse" / "kline"


@dataclass
class BacktestResult:
    """回测结果统一结构"""
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=pd.Series)


@dataclass
class Trade:
    """单笔交易"""
    symbol: str = ""
    side: str = ""
    open_date: str = ""
    close_date: str = ""
    open_price: float = 0.0
    close_price: float = 0.0
    volume: int = 0
    pnl: float = 0.0
    return_pct: float = 0.0
    n_holding_days: int = 0


# ══════════════════════════════════════
# 1. MultiAssetBacktest
# ══════════════════════════════════════

class MultiAssetBacktest:
    """多资产回测引擎

    D1/D2 收敛: 保留独立实现（能力未合并）——多资产回测能力
    backtest_engine.BacktestEngine/run_multiple_strategies 已覆盖，
    但本类为价格×信号矩阵权重式 API，签名不兼容，保留独立实现。
    """

    def __init__(self, initial_capital: float = 1_000_000.0,
                 commission: float = _execution_broker.COMMISSION_RATE,
                 slippage: float = 0.001,
                 stamp_tax_pct: float = _execution_broker.STAMP_TAX_RATE,
                 transfer_fee_pct: float = _execution_broker.TRANSFER_FEE_RATE,
                 min_commission: float = _execution_broker.MIN_COMMISSION):
        # P1-Q14-fix(H06): 补齐 A股 成本模型（原仅佣金+slippage 从未生效）
        # D3/D组收敛: 费率默认值引用 execution_broker 唯一真源（同数值同方向），签名保留。
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax_pct = stamp_tax_pct          # 印花税 0.05% 仅卖出
        self.transfer_fee_pct = transfer_fee_pct    # 过户费 0.001% 双边
        self.min_commission = min_commission        # 最低佣金 5 元/笔（A股契约）
        # P2-Q14-fix(M090): 做空政策——A股 融券受限于券源/费率（algo-trading-pitfalls
        # #7），默认 long-only（负信号权重清零）。如需融券回测可显式置 True。
        self.allow_short = False
        self._trades: list[Trade] = []
        self._equity = []

    def run(self, prices: pd.DataFrame, signals: pd.DataFrame,
            weights: Optional[pd.DataFrame] = None,
            max_position: float = 0.2) -> BacktestResult:
        """运行多资产回测

        Parameters
        ----------
        prices : pd.DataFrame
            (date x symbol) 价格矩阵
        signals : pd.DataFrame
            (date x symbol) 信号矩阵 [-1, 0, 1]
        weights : pd.DataFrame, optional
            (date x symbol) 权重矩阵，None=等权所有带信号标的
        max_position : float
            单标的最大仓位比例（硬约束：clip 后整体缩放使总仓位 ≤ 100%）

        Returns
        -------
        BacktestResult
        """
        if weights is None:
            weights = self._equal_weight_from_signals(signals)

        # P2-Q14-fix(M090): 执行 max_position 截断 + 明确做空政策。
        # 原实现 max_position 参数从未被使用：仅 1 个信号时等权权重可达 100%，
        # 且信号为负时直接做空。A股 融券受限（algo-trading-pitfalls #7），
        # 默认 long-only；max_position 作为单标的上限，先 clip 再整体缩放
        # 使总仓位 ≤ 100%（超出部分保留现金）。
        if not self.allow_short:
            weights = weights.clip(lower=0.0)
        if max_position is not None and max_position > 0:
            weights = weights.clip(upper=max_position)
            gross = weights.sum(axis=1).clip(lower=1.0)
            weights = weights.div(gross, axis=0)

        positions = weights.shift(1).fillna(0)  # T+1 执行
        # 审计 2026-08-16：缺失行情不再填 0%（避免把停牌/未上市/退市缺失当“不涨不跌”）。
        # NaN 资产在该日不参与组合收益（相当于现金/不可交易），全部缺失时收益为 0。
        returns = prices.pct_change()
        gross_ret = (positions * returns).sum(axis=1, min_count=1).fillna(0.0)

        # P1-Q14-fix(H06): 成本按 Q13 契约（execution_broker）计费：
        #   佣金 万0.85 + 最低 5 元/笔（每标每日一笔）
        #   印花税 卖出 0.05%
        #   过户费 0.001% 双边
        #   滑点 self.slippage（原实现仅存储从不使用）
        # 原实现只收 0.000085×turnover，无印花/过户/最低佣金/滑点。
        # P2-Q14-fix(M089): 换手成本用 positions.diff()（= weights.shift(1) 之差，
        # 即 t-1→t 实际持仓变动），与当日组合收益（基于 w[t-1] 持仓）同口径。
        # 原实现 delta = weights - weights.shift(1) 含当日未来权重 w[t]，
        # 成本记账日期偏移 1 天且用到未来信息。
        delta = positions.diff().fillna(0)
        equity = self.initial_capital
        cost_drag = pd.Series(0.0, index=gross_ret.index)
        for date in gross_ret.index:
            if date not in delta.index:
                continue
            d = delta.loc[date]
            traded = d[d.abs() > 1e-12]
            if not traded.empty:
                buy_value = float(traded.clip(lower=0).sum()) * equity
                sell_value = float((-traded.clip(upper=0)).sum()) * equity
                # 每标的按当日成交金额计佣金，最低 5 元/笔
                commissions = traded.map(
                    lambda dw: max(abs(dw) * equity * self.commission, self.min_commission)
                )
                total_commission = float(commissions.sum())
                stamp_tax = sell_value * self.stamp_tax_pct
                transfer_fee = (buy_value + sell_value) * self.transfer_fee_pct
                slippage_cost = (buy_value + sell_value) * self.slippage
                total_cost = total_commission + stamp_tax + transfer_fee + slippage_cost
                cost_drag[date] = total_cost / equity if equity > 0 else 0.0
            equity *= (1 + gross_ret.loc[date] - cost_drag.loc[date])

        portfolio_ret = gross_ret - cost_drag

        # 净值曲线
        equity_curve = (1 + portfolio_ret).cumprod() * self.initial_capital
        self._equity = equity_curve

        # P2-Q14-fix(M102): 交易记录改用当前组合净值（equity_curve 逐日更新）核算，
        # 不再使用冻结的 self.capital；交易级 PnL 按实际持有期（FIFO 批次）计算，
        # n_holding_days 反映真实持仓天数而非恒为 1。首日建仓也纳入交易日志，
        # 使卖出批次能正确配对。
        prev_w = weights.shift(1).fillna(0)
        open_lots: dict[str, list[dict]] = {}
        for date in weights.index:
            nav = float(equity_curve.loc[date]) if date in equity_curve.index else self.initial_capital
            for sym in weights.columns:
                delta_w = float(weights.loc[date, sym] - prev_w.loc[date, sym])
                if abs(delta_w) <= 0.01:
                    continue
                price = float(prices.loc[date, sym]) if sym in prices.columns else np.nan
                if not np.isfinite(price) or price <= 0:
                    continue
                delta_notional = abs(delta_w) * nav
                if delta_w > 0:
                    qty = int(delta_notional / price / 100) * 100
                    if qty <= 0:
                        continue
                    open_lots.setdefault(sym, []).append({
                        "open_date": str(date), "open_price": price, "qty": qty,
                    })
                    self._trades.append(Trade(
                        symbol=sym, side="buy", open_date=str(date),
                        open_price=price, close_price=price,
                        volume=qty, pnl=0.0, return_pct=0.0, n_holding_days=1,
                    ))
                else:
                    remaining = delta_notional
                    lots = open_lots.get(sym, [])
                    while remaining > 0 and lots:
                        lot = lots[0]
                        close_qty = int(min(remaining, lot["qty"] * price) / price / 100) * 100
                        if close_qty <= 0:
                            break
                        ret = price / lot["open_price"] - 1 if lot["open_price"] > 0 else 0.0
                        pnl = (price - lot["open_price"]) * close_qty
                        hold_days = int((pd.Timestamp(date) - pd.Timestamp(lot["open_date"])).days) + 1
                        self._trades.append(Trade(
                            symbol=sym, side="sell", open_date=lot["open_date"],
                            close_date=str(date), open_price=lot["open_price"],
                            close_price=price, volume=close_qty, pnl=pnl,
                            return_pct=ret, n_holding_days=hold_days,
                        ))
                        remaining -= close_qty * price
                        lot["qty"] -= close_qty
                        if lot["qty"] <= 0:
                            lots.pop(0)

        # 计算指标
        metrics = self._compute_metrics(portfolio_ret, equity_curve)

        return BacktestResult(
            returns=portfolio_ret,
            positions=positions,
            signals=signals,
            trades=self._trades,
            metrics=metrics,
            equity_curve=equity_curve,
        )

    def run_with_benchmark(self, prices: pd.DataFrame, signals: pd.DataFrame,
                            benchmark_returns: pd.Series) -> dict:
        """带基准的回测"""
        result = self.run(prices, signals)
        common = result.returns.index.intersection(benchmark_returns.index)
        result.metrics["benchmark_sharpe"] = (
            benchmark_returns.loc[common].mean() /
            max(benchmark_returns.loc[common].std(), 1e-12) * np.sqrt(252)
        )
        result.metrics["excess_return"] = (
            result.returns.loc[common].mean() -
            benchmark_returns.loc[common].mean()
        ) * 252
        return result

    def _equal_weight_from_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """信号驱动等权"""
        n = signals.abs().sum(axis=1).clip(lower=1)
        return signals.div(n, axis=0).fillna(0)

    def _compute_metrics(self, returns: pd.Series,
                         equity: pd.Series) -> dict:
        """综合绩效指标"""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak

        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        # V12.x 审计 P1-4: 统一 Sharpe 无风险利率口径——复用 MetricsCalculator.sharpe
        # （扣 rf=0.02、ddof=1、年化 sqrt(252)），不再 ann_ret/ann_vol 不减 rf，
        # 与 backtest.py/backtest_engine/signal_backtest 跨引擎可比。
        sharpe = MetricsCalculator.sharpe(returns, rf=0.02, ddof=1)
        max_dd = dd.min()
        calmar = ann_ret / max(abs(max_dd), 1e-12)
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        profit_factor = avg_win / max(avg_loss, 1e-12)

        return {
            "total_return": float(cum.iloc[-1] - 1),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "n_trades": len(self._trades),
            "final_capital": float(equity.iloc[-1]) if len(equity) > 0 else 0,
        }


# ══════════════════════════════════════
# 2. CorporateActionsHandler
# ══════════════════════════════════════

class CorporateActionsHandler:
    """公司行为处理（分红、送股、配股、拆合）

    D2收敛登记: 独立能力保留——公司行为调整 backtest_engine 未覆盖。
    """

    @staticmethod
    def adjust_for_dividend(close: pd.Series, dividend_per_share: float,
                             ex_date: str) -> pd.Series:
        """分红除权调整（价格口径）

        仅对除权日后价格做减法平滑，得到的是**价格口径**序列：除权日
        价格跳空不补回，除权日之后的收益率会被系统性压低。要得到
        total-return 口径，需与 dividend_income() 配合：
            total_ret = price_ret + dividend_yield
        """
        adjusted = close.copy()
        if ex_date in adjusted.index:
            idx = adjusted.index.get_loc(ex_date)
            adjusted.iloc[idx:] = adjusted.iloc[idx:] - dividend_per_share
        return adjusted

    @staticmethod
    def dividend_income(close: pd.Series, dividend_per_share: float,
                         ex_date: str) -> pd.Series:
        """股息收益序列（total-return 口径的"股息收益"轨）

        在除权日返回 dividend_per_share / 前一交易日收盘价，其余日期为 0。
        与 adjust_for_dividend 的输出相加即得含股息的总收益序列，
        避免除权日之后收益率被压低。
        """
        income = pd.Series(0.0, index=close.index)
        if ex_date in close.index:
            idx = close.index.get_loc(ex_date)
            if idx > 0:
                prev_close = float(close.iloc[idx - 1])
                if prev_close > 0:
                    income.iloc[idx] = dividend_per_share / prev_close
        return income

    @staticmethod
    def adjust_for_split(close: pd.Series, split_ratio: float,
                          ex_date: str) -> pd.Series:
        """拆股/合股调整

        P2-Q14-fix(M096): 明确 split_ratio 语义 = **除权后股数 / 除权前股数**
        （1→n 拆股时 split_ratio=n）。拆 1→n 后每股价格为原价的 1/n，
        为保证除权日前后的价格连续，历史价格必须 ÷n（= ×1/n），因此这里
        做除法而非原实现的乘法。原实现方向取决于调用方约定，存在反向风险。
        """
        adjusted = close.copy()
        if ex_date in adjusted.index:
            idx = adjusted.index.get_loc(ex_date)
            if split_ratio <= 0:
                raise ValueError(f"split_ratio 必须为正: {split_ratio}")
            adjusted.iloc[idx:] = adjusted.iloc[idx:] / split_ratio
        return adjusted

    @staticmethod
    def detect_splits(close: pd.Series, volume: pd.Series) -> list[dict]:
        """自动检测拆股事件"""
        if len(close) < 5:
            return []
        pct_c = close.pct_change().abs()
        vol_r = volume / volume.shift(1).clip(lower=1)
        events = []
        for i in range(1, len(close)):
            if pct_c.iloc[i] > 0.05 and vol_r.iloc[i] > 5:
                events.append({
                    "date": str(close.index[i])[:10],
                    "price_change": round(pct_c.iloc[i] * 100, 2),
                    "vol_ratio": round(vol_r.iloc[i], 2),
                })
        return events

    @staticmethod
    def adjust_for_rights_issue(close: pd.Series, rights_price: float,
                                  rights_ratio: float, ex_date: str) -> pd.Series:
        """配股调整"""
        adjusted = close.copy()
        if ex_date in adjusted.index:
            idx = adjusted.index.get_loc(ex_date)
            old_price = adjusted.iloc[idx]
            new_price = (old_price + rights_price * rights_ratio) / (1 + rights_ratio)
            factor = new_price / max(old_price, 1e-12)
            adjusted.iloc[idx:] = adjusted.iloc[idx:] * factor
        return adjusted

    # ── P1-2 (审计回测层) 动态复权事件记账适配 ──────────────────────────────
    # 目标：用"原始未复权价 + 公司行为事件记账"替代 qfq/hfq 全序列缩放——分红作
    # 为现金入账、送转为股数变动并重摊成本，而非把历史价整体缩放。

    @staticmethod
    def apply_dividend_to_position(shares: int, cash: float, dividend_per_share: float,
                                   shares_dividend: bool = False) -> tuple[int, float]:
        """一次现金分红（或送股式红利）的账户记账。

        返回 (调整后股数, 调整后现金)。现金分红入现金；若 dividend_per_share 为
        '送股股利'（送股按面值计现金）则两可，此处统一按现金入账，不改变股数。
        """
        if shares <= 0:
            return shares, cash
        cash += shares * dividend_per_share
        return shares, cash

    @staticmethod
    def apply_split_to_position(shares: int, cash: float, cost_basis: float,
                                split_ratio: float) -> tuple[int, float, float]:
        """一次送转/拆股的股数变动记账（split_ratio = 除权后股数/除权前股数）。

        股数 ×split_ratio；总成本不变，故每股成本重摊为 cost_basis/split_ratio，
        现金不变。返回 (调整后股数, 现金, 重摊后的每股成本)。
        """
        if shares <= 0:
            return shares, cash, cost_basis
        if split_ratio is None or split_ratio <= 0:
            return shares, cash, cost_basis
        new_shares = int(round(shares * split_ratio))
        new_cost = cost_basis / split_ratio if split_ratio > 0 else cost_basis
        return new_shares, cash, new_cost

    @staticmethod
    def events_from_price_columns(df: pd.DataFrame) -> list[dict]:
        """轻量适配：从日线 DataFrame 的显式事件列推导公司行为事件列表。

        数据层可能在任何一项提供除权除息事件信息（原始价模式）：
          - 'dividend' / 'dividend_per_share' / 'cash_dividend'：当日每股现金分红（元）
          - 'split' / 'split_ratio' / 'adjustment'：送转/拆股比例（除权后/除权前）
        若整列缺失或全为 0/NaN 则返回空列表（qfq/hfq 复权模式无显式事件 → 无事发生）。

        返回示例：
          [{"date": ..., "type": "dividend", "amount": 0.5},
           {"date": ..., "type": "split", "ratio": 2.0}]
        """
        if df is None or df.empty:
            return []
        events: list[dict] = []
        date_col = "date" if "date" in df.columns else df.index.name
        for idx, row in df.iterrows():
            date_val = str(row.get("date", idx))[:10] if isinstance(row, pd.Series) else str(idx)[:10]
            div_val = None
            for k in ("dividend", "dividend_per_share", "cash_dividend"):
                if k in row and pd.notna(row[k]):
                    try:
                        v = float(row[k])
                    except (TypeError, ValueError):
                        v = 0.0
                    if v > 0:
                        div_val = v
                        break
            split_val = None
            for k in ("split", "split_ratio", "adjustment"):
                if k in row and pd.notna(row[k]):
                    try:
                        v = float(row[k])
                    except (TypeError, ValueError):
                        v = 0.0
                    if v and v > 1e-9:
                        split_val = v
                        break
            if div_val is not None:
                events.append({"date": date_val, "type": "dividend", "amount": div_val})
            if split_val is not None:
                events.append({"date": date_val, "type": "split", "ratio": split_val})
        return events


# ══════════════════════════════════════
# 3. SurvivalBiasCorrector
# ══════════════════════════════════════

class SurvivalBiasCorrector:
    """存活偏差纠正

    D2收敛登记: 独立能力保留——存活偏差纠正 backtest_engine 未覆盖。
    """

    @staticmethod
    def check_coverage(backtest_symbols: list, all_symbols_by_date: dict) -> dict:
        """检测每期存活偏差覆盖率"""
        coverage = {}
        for date, symbols in all_symbols_by_date.items():
            common = len(set(backtest_symbols) & set(symbols))
            coverage[date] = {
                "total": len(symbols),
                "covered": common,
                "missing": len(symbols) - common,
                "pct": common / max(len(symbols), 1),
            }
        return coverage

    @staticmethod
    def add_delisted_stocks(backtest_df: pd.DataFrame,
                            delisted_prices: dict) -> pd.DataFrame:
        """补充已退市股票的行情"""
        for sym, prices in delisted_prices.items():
            if sym not in backtest_df.columns:
                backtest_df[sym] = np.nan
            backtest_df[sym] = backtest_df[sym].fillna(prices)
        return backtest_df

    @staticmethod
    def flag_survivorship_risk(symbols: list, current_alive: list) -> dict:
        """标记存活偏差风险"""
        dead = [s for s in symbols if s not in current_alive]
        return {
            "dead_count": len(dead),
            "dead_symbols": dead[:20],
            "death_rate": len(dead) / max(len(symbols), 1),
            "risk_level": "high" if len(dead) > len(symbols) * 0.3 else "medium" if len(dead) > len(symbols) * 0.1 else "low",
        }


def _normalize_symbol(value) -> str:
    """把退市表里的股票代码统一成 6 位字符串。"""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    if "." in text:
        text = text.split(".")[0]
    return text.zfill(6)


def _parse_delist_date(value) -> Optional[pd.Timestamp]:
    """解析退市日期；无效/缺失返回 None。"""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.normalize()


def _read_delisted_prices_for_symbol(kline_path: Path,
                                     delist_date: pd.Timestamp,
                                     max_days_after_delist: int,
                                     liquidation_price: float = 0.0) -> dict:
    """读取单只退市股行情。

    退市前（≤ 退市日）保留可交易 close；退市日后无交易，按清算价
    ``liquidation_price``（默认 0.0 = close→0 的保守清算假设；数据层 dcf 无
    清算价字段，可显式传入）填充，使回测引擎在 `pct_change().fillna(0)` 时
    实现退市损失而非把后续 NaN 抹成 0% 收益。
    """
    if not kline_path.is_file():
        return {}
    try:
        kline = pd.read_parquet(kline_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取退市股K线失败 %s: %s", kline_path, exc)
        return {}
    if kline is None or kline.empty:
        return {}
    if "close" not in kline.columns:
        return {}

    if "date" in kline.columns:
        date_values = kline["date"]
    else:
        date_values = kline.index

    frame = pd.DataFrame({
        "date": pd.to_datetime(date_values, errors="coerce"),
        "close": pd.to_numeric(kline["close"], errors="coerce"),
    })
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")

    try:
        grace_days = max(int(max_days_after_delist), 0)
    except (TypeError, ValueError):
        grace_days = 5
    cutoff = delist_date + pd.Timedelta(days=grace_days)
    frame = frame[frame["date"] <= cutoff]

    prices: dict[str, float] = {}
    has_tradable_close = False
    for _, row in frame.iterrows():
        trade_date = row["date"]
        date_str = trade_date.strftime("%Y-%m-%d")
        if trade_date <= delist_date:
            close = row["close"]
            if pd.isna(close) or close <= 0:
                continue
            prices[date_str] = float(close)
            has_tradable_close = True
        else:
            # P2-3 (审计回测层): 退市日实现资本损失——把退市日后价格填为清算价
            # （默认 0.0，即 close→0 假设；数据层无清算价字段，可显式传入），
            # 使回测引擎 pct_change 在退市次日记 -100% 而非 fillna(0) 返回 0%，
            # 避免存活偏差被平台抹零（原实现置 NaN 被 fillna(0) 永不实现损失）。
            prices[date_str] = float(liquidation_price)

    # 退市股 K 线通常止步于退市日及其前后，未必有退市后的 bar——即便没有明确
    # 的退市后 K 线，也把 guard 窗口 (delist_date, cutoff] 内的交易日填为清算价，
    # 使退市日之后面板上的 NaN 不再被 fillna(0) 抹成 0% 收益，而是实现资本损失。
    if has_tradable_close and cutoff > delist_date:
        # 审计 2026-08-16：用真实交易日历填充清算价，避免 pd.bdate_range 把
        # 节假日也填成清算日导致退市损失日期偏斜
        try:
            from quant_system.market_clock import is_trading_day
            dates = [d for d in pd.bdate_range(delist_date + pd.Timedelta(days=1), cutoff)
                     if is_trading_day(d)]
        except Exception:
            dates = list(pd.bdate_range(delist_date + pd.Timedelta(days=1), cutoff))
        for d in dates:
            if d.strftime("%Y-%m-%d") not in prices:
                prices[d.strftime("%Y-%m-%d")] = float(liquidation_price)
    return prices if has_tradable_close else {}


def load_delisted_prices(delisted_path=None,
                         kline_dir=None,
                         max_days_after_delist: int = 5,
                         liquidation_price: float = 0.0) -> dict:
    """从退市表与本地 K 线构造 ``{code: {date: close}}`` 价格映射。

    默认读取 ``data_warehouse/delisted.parquet`` 和
    ``data_warehouse/kline/{code}.parquet``。退市日（含）之前保留可交易
    close；退市日后按清算价 ``liquidation_price``（默认 0.0=close→0）填充，
    使回测在退市日实现资本损失而非把无行情 NaN 抹成 0% 收益
    （P2-3 审计回测层）。文件不存在、为空或 K 线缺失时返回空 dict，不中断
    回测流程。

    Parameters
    ----------
    liquidation_price : float
        退市后清算价。数据层 dcf 无清算价字段，默认 0.0（close→0 保守清算
        假设）；若上游提供退市清算价可显式传入。
    """
    delisted_file = (
        Path(delisted_path) if delisted_path is not None
        else _DEFAULT_DELISTED_PATH
    )
    kline_dir_path = (
        Path(kline_dir) if kline_dir is not None
        else _DEFAULT_KLINE_DIR
    )
    if not delisted_file.is_file():
        return {}
    try:
        table = pd.read_parquet(delisted_file)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取退市股表失败 %s: %s", delisted_file, exc)
        return {}
    if table is None or table.empty:
        return {}
    if not {"code", "delist_date"}.issubset(table.columns):
        return {}

    result: dict[str, dict[str, float]] = {}
    for _, row in table.iterrows():
        code = _normalize_symbol(row.get("code"))
        if not code:
            continue
        delist_date = _parse_delist_date(row.get("delist_date"))
        if delist_date is None:
            continue
        prices = _read_delisted_prices_for_symbol(
            kline_dir_path / f"{code}.parquet",
            delist_date,
            max_days_after_delist,
            liquidation_price,
        )
        if prices:
            result[code] = prices
    return result


def _align_delisted_series(prices: dict, target_index: pd.Index) -> pd.Series:
    """把字符串日期键映射对齐到回测价格面板的索引类型。"""
    series = pd.Series(prices, dtype="float64")
    if isinstance(target_index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, errors="coerce")
    return series


def integrate_delisted(backtest_df: pd.DataFrame,
                       delisted_path=None,
                       kline_dir=None,
                       max_days_after_delist: int = 5,
                       liquidation_price: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """把退市股注入回测价格面板，并返回存活偏差风险报告。

    ``backtest_df`` 形如日期索引 × 股票代码列。注入仅在目标列不存在时
    新增；若列已存在，``add_delisted_stocks`` 只 fillna，不覆盖真实值。

    P2-3 (审计回测层): 退市后（> 退市日）的面板单元格填为清算价
    ``liquidation_price``（默认 0.0=close→0），从而在回测 ``pct_change`` 中
    实现退市损失，而不是把退市后的 NaN 经 ``fillna(0)`` 抹成 0% 收益。
    """
    if not isinstance(backtest_df, pd.DataFrame):
        raise TypeError("backtest_df 必须是 pandas DataFrame")

    prices = load_delisted_prices(
        delisted_path=delisted_path,
        kline_dir=kline_dir,
        max_days_after_delist=max_days_after_delist,
        liquidation_price=liquidation_price,
    )
    aligned_prices = {
        sym: _align_delisted_series(value, backtest_df.index)
        for sym, value in prices.items()
    }

    out_df = SurvivalBiasCorrector.add_delisted_stocks(
        backtest_df.copy(), aligned_prices
    )
    original_symbols = set(backtest_df.columns)
    delisted_in_universe = [sym for sym in out_df.columns if sym in aligned_prices]
    injected_symbols = [sym for sym in aligned_prices if sym not in original_symbols]
    alive_symbols = [sym for sym in out_df.columns if sym not in aligned_prices]
    risk = SurvivalBiasCorrector.flag_survivorship_risk(
        list(out_df.columns), alive_symbols
    )

    report = {
        "delisted_in_universe": delisted_in_universe,
        "injected_count": len(injected_symbols),
        "death_rate": risk["death_rate"],
        "risk_level": risk["risk_level"],
    }
    return out_df, report


# ══════════════════════════════════════
# 4. SubPeriodAnalyzer
# ══════════════════════════════════════

class SubPeriodAnalyzer:
    """子周期绩效分析

    D2收敛登记: 独立能力保留——子周期稳定性检验 backtest_engine 未覆盖。
    """

    def __init__(self, n_periods: int = 4):
        self.n_periods = n_periods

    def analyze(self, returns: pd.Series) -> pd.DataFrame:
        """等分周期分析"""
        n = len(returns)
        chunk = n // self.n_periods
        results = []
        for i in range(self.n_periods):
            start = i * chunk
            end = start + chunk if i < self.n_periods - 1 else n
            sub = returns.iloc[start:end]
            if len(sub) < 5:
                continue
            results.append({
                "period": i + 1,
                "start": str(returns.index[start])[:10],
                "end": str(returns.index[min(end, n) - 1])[:10],
                "return": sub.sum(),
                "vol": sub.std() * np.sqrt(252),
                "sharpe": sub.mean() / max(sub.std(), 1e-12) * np.sqrt(252),
                "max_dd": self._max_dd(sub),
                "win_rate": (sub > 0).mean(),
            })
        return pd.DataFrame(results)

    def stability_score(self, results: pd.DataFrame) -> dict:
        """稳定性评分"""
        if results.empty or "sharpe" not in results.columns:
            return {"score": 0, "std": 0}
        sharpe_std = results["sharpe"].std()
        mean_sharpe = results["sharpe"].mean()
        return {
            "mean_sharpe": mean_sharpe,
            "sharpe_std": sharpe_std,
            "stability": mean_sharpe / max(sharpe_std, 1e-12),
            "positive_periods": (results["sharpe"] > 0).mean(),
        }

    def _max_dd(self, returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        return (cum - peak).min()


# ══════════════════════════════════════
# 5. WalkForwardBacktest
# ══════════════════════════════════════

class WalkForwardBacktest:
    """Walk-Forward 回测

    D1/D2 收敛: 保留独立实现（能力未合并）——能力与
    backtest_engine.walk_forward_backtest 重叠，但本类为
    prices+strategy_fn 回调式 API（engine 为 strategy_class+data），
    签名不兼容，保留独立实现。
    """

    def __init__(self, train_window: int = 504, test_window: int = 126,
                 step: int = 63):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self._results: list[dict] = []

    def run(self, prices: pd.DataFrame, strategy_fn: Callable,
            param_grid: dict, **kwargs) -> pd.DataFrame:
        """执行 Walk-Forward"""
        n = len(prices)
        windows = []
        train_start = 0
        # Q14 修复：使用实例属性（self.*），原代码引用裸变量 train_window/
        # test_window/step → 首次迭代必然 NameError。
        while train_start + self.train_window + self.test_window <= n:
            train_end = train_start + self.train_window
            test_end = train_end + self.test_window
            windows.append((train_start, train_end, test_end))
            train_start += self.step

        # P2-Q14-fix(M087/M088): 全量收益提前算一次（pct_change 在窗口内切片会
        # 把各窗口首日算成 0，同一日期不同窗口收益不一致），OOS 收益 = 信号
        # shift(1) × 全局收益，保证同一日期跨窗口值一致。
        all_rets = prices.pct_change().fillna(0)
        oos_returns = []
        for ts, te, tste in windows:
            train = prices.iloc[ts:te]
            test = prices.iloc[te:tste]

            # 网格搜索最佳参数
            best_param, best_score = None, -np.inf
            for values in product(*param_grid.values()):
                params = dict(zip(param_grid.keys(), values))
                try:
                    signals = strategy_fn(train, **params)
                    score = self._score_signals(signals)
                    if score is not None and score > best_score:
                        best_score = score
                        best_param = params
                except Exception as exc:
                    # P2-Q14-fix(M088): 不再静默 continue——原实现当 strategy_fn
                    # 返回 DataFrame 时 signals.mean() 是 Series，与标量比较抛
                    # TypeError 被 except 吞掉，导致所有参数组合被跳过、best_param
                    # 恒为 None 而以默认参数跑测试集。现记录失败原因，可见降级。
                    logger.warning(f"WalkForward 参数组合 {params} 评分失败: {exc}")

            # 在验证集上测试（输出按信号换算的收益，而非原始信号）
            oos_signals = strategy_fn(test, **(best_param or {}))
            if oos_signals is not None and len(oos_signals) > 0:
                oos_returns.append(self._signals_to_returns(oos_signals, all_rets.iloc[te:tste]))
                self._results.append({
                    "train_start": str(train.index[0])[:10],
                    "train_end": str(train.index[-1])[:10],
                    "test_start": str(test.index[0])[:10],
                    "test_end": str(test.index[-1])[:10],
                    "best_param": best_param,
                    "train_score": best_score,
                })

        if not oos_returns:
            return pd.DataFrame()

        combined = pd.concat(oos_returns).sort_index()
        # Q14 #13 同根修复（与 Q17 walk_forward 一致）：当 step < test_window 时
        # 相邻窗口测试期重叠，直接 concat 会产生重复日期索引，cumprod 重复计息。
        # 这里按日期去重（重叠日取均值）。【该部分已在前序 Q14 修复中处理】
        if combined.index.has_duplicates:
            combined = combined.groupby(level=0).mean().sort_index()
        return pd.DataFrame({"walk_forward_return": combined})

    @staticmethod
    def _score_signals(signals) -> float | None:
        """把策略输出统一压成单一评分（均值/标准差），兼容 Series/DataFrame。

        原实现 signals.mean()/max(signals.std(),1e-12)：strategy_fn 返回
        DataFrame 时 mean()/std() 均为 Series，score 与标量比较抛 TypeError，
        被 except 静默吞掉 → 所有参数组合跳过、以默认参数跑测试集。现先展平
        为数值数组再评分；失败返回 None（调用方记录为候选跳过）。
        """
        if signals is None:
            return None
        if isinstance(signals, pd.DataFrame):
            vals = signals.to_numpy(dtype=float).ravel()
        elif isinstance(signals, pd.Series):
            vals = signals.to_numpy(dtype=float)
        else:
            vals = np.asarray(signals, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return None
        return float(np.mean(vals)) / max(float(np.std(vals)), 1e-12)

    @staticmethod
    def _net_after_turnover_cost(positions: pd.DataFrame | pd.Series,
                                 gross_ret: pd.Series) -> pd.Series:
        """按换手计提 A股 交易成本，返回扣除成本后的净收益。

        P2-2 (审计回测层): walk-forward/参数扫描路径原本把
        ``signal.shift(1) × 日收益`` 直接当策略收益，零成本评估，换手越夸张
        高估越狠。此处套用与 ``MultiAssetBacktest.run``（L151-169）一致的成本
        模型：佣金(万0.85，最低 5 元/笔/标的/日) + 印花税(卖出 0.05%) +
        过户费(0.001% 双边) + 滑点(0.1%)，按 ``positions.diff()``（t-1→t 实际
        持仓变动）计提。

        Parameters
        ----------
        positions : pd.DataFrame | pd.Series
            T+1 已 shifted 的持仓矩阵（index=日期，values=持仓比例/方向仓位）
        gross_ret : pd.Series
            与 positions 对齐的每日组合毛收益

        Returns
        -------
        pd.Series
            扣除换手成本后的净收益（与 gross_ret 同索引）
        """
        pos_df = positions.to_frame() if isinstance(positions, pd.Series) else positions.copy()
        delta = pos_df.diff().fillna(0)

        commission = _execution_broker.COMMISSION_RATE          # 万0.85
        min_commission = _execution_broker.MIN_COMMISSION       # 5 元/笔
        stamp_tax_pct = _execution_broker.STAMP_TAX_RATE         # 卖出 0.05%
        transfer_fee_pct = _execution_broker.TRANSFER_FEE_RATE   # 过户 0.001% 双边
        slippage = _execution_broker.DEFAULT_SLIPPAGE_RATE       # 0.1%

        # 归一化本金：把信号持仓当作从一份默认规模（1_000_000）资金池按权重建仓，
        # 保证最低佣金(5元)在真实金额量级上起效、cost_drag(=总成本/权益)为分数。
        notional = 1_000_000.0
        equity = notional
        cost_drag = pd.Series(0.0, index=pos_df.index)
        nav = notional
        for date in pos_df.index:
            d = delta.loc[date]
            traded = d[d.abs() > 1e-12]
            if not traded.empty:
                buy_value = float(traded.clip(lower=0).sum()) * equity
                sell_value = float((-traded.clip(upper=0)).sum()) * equity
                commissions = traded.map(
                    lambda dw: max(abs(dw) * equity * commission, min_commission)
                )
                total_commission = float(commissions.sum())
                stamp_tax = sell_value * stamp_tax_pct
                transfer_fee = (buy_value + sell_value) * transfer_fee_pct
                slippage_cost = (buy_value + sell_value) * slippage
                total_cost = total_commission + stamp_tax + transfer_fee + slippage_cost
                cost_drag[date] = total_cost / equity if equity > 0 else 0.0
            nav *= 1 + float(gross_ret.loc[date]) - float(cost_drag.loc[date])
            equity = nav
        return gross_ret - cost_drag

    @staticmethod
    def _signals_to_returns(signals, rets: pd.DataFrame) -> pd.Series:
        """把策略信号（-1/0/1 持仓指示）换算为 OOS 收益序列。

        信号在 t 日收盘产生，t+1 日按 t 日信号持仓（避免同日成交前视），
        收益 = signal.shift(1) × 次日收益。支持单资产 Series 与多资产 DataFrame。
        原实现直接把原始信号 concat 到 walk_forward_return 列（实测值域即
        信号而非收益），导致下游 cumprod/夏普全部失真。

        P2-2 (审计回测层): 在毛收益之上按换手计提交易成本（佣金/印花税/过户费/
        滑点），换手越高的信号策略不再被零成本高估。
        """
        if isinstance(signals, pd.DataFrame):
            common = signals.columns.intersection(rets.columns)
            if len(common) == 0:
                return pd.Series(dtype=float)
            positions = signals[common].shift(1).fillna(0)
            gross_ret = (positions * rets[common]).sum(axis=1).fillna(0.0)
            out = WalkForwardBacktest._net_after_turnover_cost(positions, gross_ret)
        elif isinstance(signals, pd.Series):
            r = rets.iloc[:, 0] if isinstance(rets, pd.DataFrame) else rets
            r = r.reindex(signals.index).fillna(0.0)
            positions = signals.shift(1).fillna(0.0)
            gross_ret = (positions * r).fillna(0.0)
            out = WalkForwardBacktest._net_after_turnover_cost(positions, gross_ret)
        else:
            out = pd.Series(np.asarray(signals, dtype=float), index=rets.index)
        out.name = "walk_forward_return"
        return out


# ══════════════════════════════════════
# 6. ComprehensiveMetrics
# ══════════════════════════════════════

class ComprehensiveMetrics:
    """全面绩效指标体系

    D1/D2 收敛: 保留独立实现（能力未合并）——指标计算能力与
    backtest_engine.compute_sharpe/compute_max_drawdown 等重叠，
    但本类为批量聚合 dict API（engine 为单指标标量函数），签名不兼容。
    """

    @staticmethod
    def compute_all(returns: pd.Series, rf: float = 0.025) -> dict:
        """计算全套绩效指标"""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak

        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        rf_daily = rf / 252

        # 夏普比率
        excess = returns - rf_daily
        sharpe = excess.mean() / max(excess.std(), 1e-12) * np.sqrt(252)

        # Sortino
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = ann_ret / max(downside, 1e-12)

        # 最大回撤
        max_dd = dd.min()

        # Calmar
        calmar = ann_ret / max(abs(max_dd), 1e-12)

        # 滚动指标
        rolling_sharpe_60d = returns.rolling(60).apply(
            lambda x: x.mean() / max(x.std(), 1e-12) * np.sqrt(252))

        # 偏度峰度
        skew = returns.skew()
        kurt = returns.kurtosis()

        # VaR/CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # 胜率/盈亏比
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        profit_factor = avg_win / max(avg_loss, 1e-12)

        # 资金曲线指标
        ulcer = np.sqrt((dd ** 2).mean())
        martin_ratio = ann_ret / max(ulcer, 1e-12)

        return {
            "annualized_return": round(ann_ret, 4),
            "annualized_vol": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "martin_ratio": round(martin_ratio, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(cum.iloc[-1] - 1, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4) if not np.isnan(cvar_95) else 0,
            "rolling_sharpe_mean": round(rolling_sharpe_60d.mean(), 4),
            "rolling_sharpe_std": round(rolling_sharpe_60d.std(), 4),
            "n_observations": len(returns),
        }

    @staticmethod
    def generate_report(returns: pd.Series, name: str = "策略") -> str:
        """生成绩效报告"""
        m = ComprehensiveMetrics.compute_all(returns)
        lines = [
            "=" * 55,
            f"绩效报告: {name} (V4.1 feature)",
            "=" * 55,
            f"  年化收益: {m['annualized_return']*100:.2f}%",
            f"  年化波动: {m['annualized_vol']*100:.2f}%",
            f"  夏普比率: {m['sharpe_ratio']:.3f}",
            f"  Sortino:  {m['sortino_ratio']:.3f}",
            f"  Calmar:   {m['calmar_ratio']:.3f}",
            f"  最大回撤: {m['max_drawdown']*100:.2f}%",
            f"  总收益:   {m['total_return']*100:.2f}%",
            f"  胜率:     {m['win_rate']*100:.1f}%",
            f"  盈亏比:   {m['profit_factor']:.2f}",
            f"  VaR(95%): {m['var_95']*100:.2f}%",
            f"  CVaR(95%):{m['cvar_95']*100:.2f}%",
            f"  偏度:     {m['skewness']:.3f}",
            f"  峰度:     {m['kurtosis']:.3f}",
            f"  观测数:   {m['n_observations']}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════
# 7. PerformanceAttribution  (V4.1 feature)
# ══════════════════════════════════════

class PerformanceAttribution:
    """绩效归因分析（Brinson / 因子 / 滚动）V4.1 feature

    D2收敛登记: 独立能力保留——Brinson/因子/滚动归因 backtest_engine 未覆盖。
    """

    @staticmethod
    def brinson_attribution(returns: pd.DataFrame,
                            weights: pd.DataFrame,
                            benchmark_weights: pd.DataFrame) -> pd.DataFrame:
        """Brinson归因：选股效应 + 行业配置效应 + 交叉效应

        Parameters
        ----------
        returns : pd.DataFrame
            (date x sector) 各行业/板块收益率
        weights : pd.DataFrame
            (date x sector) 组合权重
        benchmark_weights : pd.DataFrame
            (date x sector) 基准权重

        Returns
        -------
        pd.DataFrame  columns=[allocation, selection, cross, total]
        """
        common_idx = returns.index.intersection(weights.index)
        common_idx = common_idx.intersection(benchmark_weights.index)
        r = returns.loc[common_idx]
        w = weights.loc[common_idx]
        b = benchmark_weights.loc[common_idx]

        # 基准收益率（指数收益）
        benchmark_ret = (b * r).sum(axis=1)

        # 行业配置效应： (w - b) * benchmark_ret
        allocation = ((w - b) * benchmark_ret.values.reshape(-1, 1)).sum(axis=1)

        # 选股效应： (r - benchmark_ret.values.reshape(-1, 1)) * b
        selection = ((r.subtract(benchmark_ret, axis=0)) * b).sum(axis=1)

        # 交叉效应： (w - b) * (r - benchmark_ret.values.reshape(-1, 1))
        cross = ((w - b) * r.subtract(benchmark_ret, axis=0)).sum(axis=1)

        # 总超额收益
        port_ret = (w * r).sum(axis=1)
        total = port_ret - benchmark_ret

        df = pd.DataFrame({
            "allocation": allocation,
            "selection": selection,
            "cross": cross,
            "total": total,
        }, index=common_idx)
        df.index.name = "date"
        return df

    @staticmethod
    def factor_attribution(returns: pd.Series,
                           factor_returns: pd.DataFrame) -> pd.DataFrame:
        """因子归因（多因子时间序列回归）

        Parameters
        ----------
        returns : pd.Series
            组合日收益率序列
        factor_returns : pd.DataFrame
            (date x factor) 各因子日收益率

        Returns
        -------
        pd.DataFrame  columns=[factor, beta, t_stat, p_value, r_squared]
        每行一个因子+截距项
        """
        import statsmodels.api as sm

        common = returns.index.intersection(factor_returns.index)
        y = returns.loc[common]
        X = factor_returns.loc[common].copy()
        X = sm.add_constant(X)

        model = sm.OLS(y, X.astype(float).dropna(), missing="drop").fit()

        rows = []
        for col in X.columns:
            idx = list(X.columns).index(col)
            rows.append({
                "factor": "Alpha" if col == "const" else col,
                "beta": round(model.params.iloc[idx], 6),
                "t_stat": round(model.tvalues.iloc[idx], 4),
                "p_value": round(model.pvalues.iloc[idx], 6),
                "r_squared": round(model.rsquared, 4),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def rolling_attribution(returns: pd.Series,
                            factor_returns: pd.DataFrame,
                            window: int = 252) -> pd.DataFrame:
        """滚动因子归因（固定窗口滚动回归）

        Returns
        -------
        pd.DataFrame  (date x factor_beta) 每期滚动因子暴露
        """
        import statsmodels.api as sm

        common = returns.index.intersection(factor_returns.index)
        y = returns.loc[common]
        X = factor_returns.loc[common].copy()
        X = sm.add_constant(X)
        cols = X.columns.tolist()

        results: list[pd.Series] = []
        for i in range(window, len(y)):
            yw = y.iloc[i - window:i]
            Xw = X.iloc[i - window:i].astype(float).dropna()
            yw = yw.loc[Xw.index]
            if len(yw) < window * 0.5:
                continue
            try:
                mod = sm.OLS(yw, Xw, missing="drop").fit()
                s = pd.Series(mod.params.values, index=cols, name=y.index[i])
                s["_rsquared"] = mod.rsquared
                results.append(s)
            except Exception as e:
                logger.error(f"[backtest_pro] 操作失败: {e}", exc_info=True)
                continue

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).dropna(how="all")


# ══════════════════════════════════════
# 8. MonteCarloBacktest  (V4.1 feature)
# ══════════════════════════════════════

class MonteCarloBacktest:
    """蒙特卡洛回测模拟 V4.1 feature

    D2收敛登记: 独立能力保留——Monte Carlo 模拟 backtest_engine 未覆盖。
    """

    def __init__(self, random_seed: Optional[int] = 42):
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=random_seed)

    def run(self, returns: pd.Series,
            n_simulations: int = 1000,
            n_days: int = 252) -> dict:
        """蒙特卡洛模拟

        Parameters
        ----------
        returns : pd.Series
            历史收益率序列（用于估计均值和波动率）
        n_simulations : int
            模拟路径数量
        n_days : int
            单路径模拟天数

        Returns
        -------
        dict 包含:
            - paths: pd.DataFrame (n_simulations x n_days) 净值路径
            - final_values: np.ndarray 最终净值
            - stats: dict 汇总统计
        """
        mu = returns.mean()
        sigma = returns.std()
        dt = 1.0 / 252  # 日单位

        # P1-Q14-fix(H03): mu/sigma 为日频估计，须年化后再乘 dt。
        # 原实现对日频 mu/sigma 直接乘 dt=1/252，252 步模拟的终值分布≈单日
        # 分布（std=sigma 而非 sigma×√252），p5/p95/prob_positive_return、
        # 置信区间全部失真。
        mu_ann = mu * 252
        sigma_ann = sigma * np.sqrt(252)

        # 几何布朗运动模拟
        Z = self._rng.standard_normal((n_simulations, n_days))
        drift = (mu_ann - 0.5 * sigma_ann ** 2) * dt
        diffusion = sigma_ann * np.sqrt(dt) * Z
        log_returns = drift + diffusion
        log_returns[:, 0] = 0.0  # 初始收益为0

        # 累乘得到净值路径
        paths = np.exp(np.cumsum(log_returns, axis=1))
        final_values = paths[:, -1]

        paths_df = pd.DataFrame(paths.T)
        paths_df.index.name = "day"

        stats = {
            "mean_final": float(np.mean(final_values)),
            "median_final": float(np.median(final_values)),
            "std_final": float(np.std(final_values)),
            "min_final": float(np.min(final_values)),
            "max_final": float(np.max(final_values)),
            "p5": float(np.percentile(final_values, 5)),
            "p95": float(np.percentile(final_values, 95)),
            "prob_positive_return": float((final_values > 1).mean()),
        }

        return {
            "paths": paths_df,
            "final_values": final_values,
            "stats": stats,
        }

    def confidence_intervals(self, results: dict,
                              confidence: float = 0.95) -> dict:
        """计算置信区间

        Parameters
        ----------
        results : dict
            MonteCarloBacktest.run() 返回结果
        confidence : float
            置信度（默认0.95）

        Returns
        -------
        dict 各时间点的置信区间
        """
        paths = results["paths"]
        alpha = 1.0 - confidence
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        lower = paths.quantile(lower_pct / 100, axis=1).values
        upper = paths.quantile(upper_pct / 100, axis=1).values
        median = paths.median(axis=1).values

        return {
            "lower": lower,
            "median": median,
            "upper": upper,
            "confidence": confidence,
            "days": paths.index.tolist(),
        }

    @staticmethod
    def probability_of_loss(results: dict, threshold: float = 0.0) -> float:
        """亏损概率：最终净值低于threshold的概率

        threshold : float
            亏损阈值（默认0，即最终净值<1）
        """
        final = results.get("final_values", np.array([]))
        if len(final) == 0:
            return float("nan")
        return float((final < (1 + threshold)).mean())


# ══════════════════════════════════════
# 9. ParameterSweeper  (V4.1 feature)
# ══════════════════════════════════════

class ParameterSweeper:
    """全参数网格扫描 V4.1 feature

    D1/D2 收敛: 保留独立实现（能力未合并）——能力与
    backtest_engine.parameter_scan 重叠，但本类为 strategy_fn 回调式
    API（engine 为 strategy_class+data+param_grid），签名不兼容。
    """

    def __init__(self, progress_bar: bool = False):
        self.progress_bar = progress_bar

    def sweep(self, strategy_fn: Callable, param_grid: dict,
              prices: pd.DataFrame) -> pd.DataFrame:
        """全参数网格扫描

        Parameters
        ----------
        strategy_fn : Callable
            策略函数 fn(prices, **params) -> pd.Series 收益率序列
        param_grid : dict
            参数网格 {param_name: [values]}
        prices : pd.DataFrame
            价格数据

        Notes
        -----
        P2-2 (审计回测层): 本方法直接评分 ``strategy_fn`` 返回的收益序列，
        不做额外的信号→收益转换，因此**必须由调用方保证 strategy_fn 返回
        的是已扣交易成本（佣金/印花税/过户费/滑点）的净收益**；若传入毛收益，
        换手越高的策略越会被零成本高估。如需按信号矩阵 + 换手自动计提成本，
        请走 ``WalkForwardBacktest._signals_to_returns``。
        
        Returns
        -------
        pd.DataFrame  行列: 参数组合 + 绩效指标
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        rows = []
        total = np.prod([len(v) for v in value_lists])
        for i, values in enumerate(product(*value_lists)):
            params = dict(zip(keys, values))
            try:
                ret = strategy_fn(prices, **params)
                if ret is None or len(ret) == 0:
                    continue
                metrics = ComprehensiveMetrics.compute_all(ret)
                row = {**params, **metrics}
                rows.append(row)
            except Exception as e:
                logger.warning(f"参数组合 {params} 扫描失败: {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 将参数列移到前面
        param_cols = [k for k in keys if k in df.columns]
        metric_cols = [c for c in df.columns if c not in param_cols]
        return df[param_cols + metric_cols]

    @staticmethod
    def heatmap_data(results: pd.DataFrame, x_param: str, y_param: str,
                      metric: str = "sharpe_ratio") -> pd.DataFrame:
        """生成热力图数据（pivot table）

        Parameters
        ----------
        results : pd.DataFrame
            sweep() 返回的结果表
        x_param : str
            热力图X轴参数名
        y_param : str
            热力图Y轴参数名
        metric : str
            指标列名

        Returns
        -------
        pd.DataFrame  透视表，适合直接传入 seaborn.heatmap
        """
        if x_param not in results.columns or y_param not in results.columns:
            raise ValueError(f"参数 {x_param}/{y_param} 不在结果中")
        if metric not in results.columns:
            raise ValueError(f"指标 {metric} 不在结果中")
        return results.pivot_table(
            index=y_param, columns=x_param, values=metric, aggfunc="first"
        )

    @staticmethod
    def optimal_region(results: pd.DataFrame, metric: str = "sharpe_ratio",
                        threshold: float = 0.8) -> dict:
        """最优参数区域：找出绩效在top threshold内的所有参数组合

        Returns
        -------
        dict 包含:
            - best_params: dict 全局最优参数
            - best_metric: float 最优值
            - region: pd.DataFrame 所有在阈值内的参数
            - n_best: int 参数组合数
        """
        if results.empty or metric not in results.columns:
            return {"best_params": {}, "best_metric": 0, "region": pd.DataFrame(), "n_best": 0}

        best_val = results[metric].max()
        cutoff = best_val * threshold
        region = results[results[metric] >= cutoff].copy()

        # 排除指标列，仅返回参数列
        param_cols = [c for c in results.columns if c not in [
            "annualized_return", "annualized_vol", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "martin_ratio", "max_drawdown",
            "total_return", "win_rate", "profit_factor", "skewness",
            "kurtosis", "var_95", "cvar_95", "rolling_sharpe_mean",
            "rolling_sharpe_std", "n_observations",
        ]]

        best_row = results.loc[results[metric].idxmax()]
        best_params = {c: best_row[c] for c in param_cols if c in best_row.index}

        return {
            "best_params": best_params,
            "best_metric": float(best_val),
            "region": region[param_cols].drop_duplicates() if len(param_cols) > 0 else region,
            "n_best": len(region),
        }


# ══════════════════════════════════════
# 10. BenchmarkComparison  (V4.1 feature)
# ══════════════════════════════════════

class BenchmarkComparison:
    """多基准对比分析 V4.1 feature

    D2收敛登记: 独立能力保留——alpha/beta/IR/上下行捕获/滚动alpha
    backtest_engine.BacktestComparison（仅策略间摘要对比）未覆盖。
    """

    @staticmethod
    def compare_to_benchmark(strategy_returns: pd.Series,
                              benchmarks: pd.DataFrame) -> pd.DataFrame:
        """多基准对比

        Parameters
        ----------
        strategy_returns : pd.Series
            策略日收益率
        benchmarks : pd.DataFrame
            (date x benchmark_name) 各基准日收益率

        Returns
        -------
        pd.DataFrame  每行为一个基准的对比指标
        """
        common = strategy_returns.index.intersection(benchmarks.index)
        sr = strategy_returns.loc[common]

        rows = []
        for col in benchmarks.columns:
            br = benchmarks.loc[common, col]
            if br.std() < 1e-12:
                continue

            # Alpha / Beta （滚动252日均值）
            ab = BenchmarkComparison.alpha_beta(sr, br)

            # IR
            ir = BenchmarkComparison.information_ratio(sr, br)

            # 上下行捕获
            ud = BenchmarkComparison.up_down_capture(sr, br)

            rows.append({
                "benchmark": col,
                "alpha": ab["alpha"],
                "beta": ab["beta"],
                "tracking_error": ab["tracking_error"],
                "information_ratio": ir,
                "up_capture": ud["up_capture"],
                "down_capture": ud["down_capture"],
                "up_down_ratio": ud["up_down_ratio"],
                "correlation": sr.corr(br),
                "excess_return": (sr.mean() - br.mean()) * 252,
                "excess_vol": (sr.std() - br.std()) * np.sqrt(252),
            })

        return pd.DataFrame(rows).set_index("benchmark")

    @staticmethod
    def alpha_beta(strategy_returns: pd.Series,
                    benchmark_returns: pd.Series) -> dict:
        """计算Alpha/Beta/跟踪误差"""
        import statsmodels.api as sm

        common = strategy_returns.index.intersection(benchmark_returns.index)
        y = strategy_returns.loc[common].values
        X = sm.add_constant(benchmark_returns.loc[common].values)

        model = sm.OLS(y, X, missing="drop").fit()
        alpha = model.params[0] * 252  # 年化Alpha
        beta = model.params[1]

        # 跟踪误差
        residual = y - model.fittedvalues
        tracking_error = np.std(residual) * np.sqrt(252)

        return {
            "alpha": round(alpha, 6),
            "beta": round(beta, 4),
            "tracking_error": round(tracking_error, 6),
            "t_stat_alpha": round(model.tvalues[0], 4),
            "p_value_alpha": round(model.pvalues[0], 6),
        }

    @staticmethod
    def information_ratio(strategy_returns: pd.Series,
                           benchmark_returns: pd.Series) -> float:
        """信息比率 IR = 年化超额收益 / 跟踪误差"""
        common = strategy_returns.index.intersection(benchmark_returns.index)
        excess = strategy_returns.loc[common] - benchmark_returns.loc[common]
        ann_excess = excess.mean() * 252
        tracking_err = excess.std() * np.sqrt(252)
        if tracking_err < 1e-12:
            return 0.0
        return round(ann_excess / tracking_err, 4)

    @staticmethod
    def up_down_capture(strategy_returns: pd.Series,
                         benchmark: pd.Series) -> dict:
        """上下行捕获率

        up_capture: 基准上涨时策略的平均收益 / 基准上涨时的平均收益
        down_capture: 基准下跌时策略的平均收益 / 基准下跌时的平均收益
        """
        common = strategy_returns.index.intersection(benchmark.index)
        sr = strategy_returns.loc[common]
        br = benchmark.loc[common]

        up_mask = br > 0
        down_mask = br < 0

        up_bench = br[up_mask].mean()
        down_bench = abs(br[down_mask].mean())

        up_strat = sr[up_mask].mean() if up_mask.any() else 0.0
        down_strat = abs(sr[down_mask].mean()) if down_mask.any() else 0.0

        up_cap = up_strat / up_bench if abs(up_bench) > 1e-12 else 1.0
        down_cap = down_strat / down_bench if abs(down_bench) > 1e-12 else 1.0

        return {
            "up_capture": round(up_cap, 4),
            "down_capture": round(down_cap, 4),
            "up_down_ratio": round(up_cap / max(down_cap, 1e-12), 4),
        }

    @staticmethod
    def rolling_alpha(returns: pd.Series, benchmark: pd.Series,
                       window: int = 252) -> pd.Series:
        """滚动Alpha（固定窗口回归）"""
        import statsmodels.api as sm

        common = returns.index.intersection(benchmark.index)
        y = returns.loc[common]
        x = benchmark.loc[common]

        alphas: list[float] = []
        idx: list = []
        for i in range(window, len(y)):
            yw = y.iloc[i - window:i].values
            xw = sm.add_constant(x.iloc[i - window:i].values)
            try:
                mod = sm.OLS(yw, xw, missing="drop").fit()
                alphas.append(mod.params[0] * 252)  # 年化
                idx.append(y.index[i])
            except Exception as e:
                logger.error(f"[backtest_pro] 操作失败: {e}", exc_info=True)
                continue

        return pd.Series(alphas, index=idx, name="rolling_alpha")


# ══════════════════════════════════════
# 11. MultiPeriodOptimizer  (V4.1 feature)
# ══════════════════════════════════════

class MultiPeriodOptimizer:
    """多期组合优化 V4.1 feature

    D2收敛登记: 独立能力保留——组合权重优化 backtest_engine 未覆盖。
    """

    def __init__(self, turnover_penalty: float = 0.001):
        self.turnover_penalty = turnover_penalty

    def optimize(self, returns: pd.DataFrame, n_periods: int = 4,
                  method: str = "equal_risk") -> pd.DataFrame:
        """多期组合优化

        Parameters
        ----------
        returns : pd.DataFrame
            (date x asset) 各资产日收益率
        n_periods : int
            将数据等分为n期，逐期滚动优化
        method : str
            "equal_risk" = 等风险贡献（ERC）
            "min_variance" = 最小方差
            "max_sharpe" = 最大夏普

        Returns
        -------
        pd.DataFrame  (period x asset) 每期最优权重
        """
        n = len(returns)
        chunk = n // n_periods
        weights_list: list[pd.Series] = []
        period_labels: list[str] = []

        for i in range(n_periods):
            start = i * chunk
            end = start + chunk if i < n_periods - 1 else n
            sub = returns.iloc[start:end]
            if sub.empty or len(sub.columns) < 2:
                continue

            cov = sub.cov() * 252  # 年化协方差
            mu = sub.mean() * 252  # 年化收益

            if method == "equal_risk":
                w = self._erc_weights(cov)
            elif method == "min_variance":
                w = self._min_variance_weights(cov)
            elif method == "max_sharpe":
                w = self._max_sharpe_weights(cov, mu)
            else:
                raise ValueError(f"不支持的方法: {method}")

            weights_list.append(w)
            period_labels.append(f"P{i+1}")

        if not weights_list:
            return pd.DataFrame()

        df = pd.DataFrame(weights_list, index=period_labels)
        df.index.name = "period"
        return df

    def _erc_weights(self, cov: pd.DataFrame, max_iter: int = 100,
                     tol: float = 1e-8) -> pd.Series:
        """等风险贡献（ERC）权重 - 数值优化"""
        n = len(cov)
        w = np.ones(n) / n

        for _ in range(max_iter):
            sigma = np.sqrt(np.diag(cov @ np.outer(w, w) @ cov.T))
            # 风险贡献
            rc = w * (cov @ w) / max(np.sqrt(w @ cov @ w), 1e-12)
            target_rc = np.ones(n) / n
            # 梯度调整
            grad = rc - target_rc
            w = w - 0.1 * grad
            w = np.clip(w, 0, 1)
            w = w / w.sum()
            if np.linalg.norm(grad) < tol:
                break

        return pd.Series(w, index=cov.columns, name="erc")

    def _min_variance_weights(self, cov: pd.DataFrame) -> pd.Series:
        """最小方差组合"""
        n = len(cov)
        inv_cov = np.linalg.inv(cov.values + np.eye(n) * 1e-8)
        ones = np.ones(n)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        return pd.Series(w, index=cov.columns, name="min_var")

    def _max_sharpe_weights(self, cov: pd.DataFrame,
                             mu: pd.Series) -> pd.Series:
        """最大夏普组合"""
        n = len(cov)
        inv_cov = np.linalg.inv(cov.values + np.eye(n) * 1e-8)
        w = inv_cov @ mu.values
        # 只允许做多
        w = np.clip(w, 0, 1)
        if w.sum() < 1e-12:
            w = np.ones(n) / n
        else:
            w = w / w.sum()
        return pd.Series(w, index=cov.index, name="max_sharpe")

    @staticmethod
    def rebalance_schedule(dates: pd.DatetimeIndex,
                            frequency: str = "monthly") -> list:
        """再平衡计划

        Parameters
        ----------
        dates : pd.DatetimeIndex
            交易日历
        frequency : str
            "monthly" / "quarterly" / "weekly" / "yearly"

        Returns
        -------
        list[dict]  每期 rebalance 的日期和索引
        """
        series = pd.Series(index=dates, data=range(len(dates)))

        if frequency == "monthly":
            groups = series.groupby([series.index.year, series.index.month])
        elif frequency == "quarterly":
            groups = series.groupby([series.index.year, series.index.quarter])
        elif frequency == "weekly":
            # P2-Q14-fix(L097): 原按 isocalendar().week 分组不含年份 → 跨年同周
            # 合并、年初周与上年末周混组。改为 (year, week) 分组。
            iso = series.index.isocalendar()
            groups = series.groupby([iso.year, iso.week])
        elif frequency == "yearly":
            groups = series.groupby(series.index.year)
        else:
            raise ValueError(f"不支持频率: {frequency}")

        schedule = []
        for (key, grp) in groups:
            idx = grp.index
            schedule.append({
                "period": key,
                "start_date": idx[0],
                "end_date": idx[-1],
                "start_idx": series.loc[idx[0]],
                "end_idx": series.loc[idx[-1]],
                "n_days": len(idx),
            })
        return schedule


# ══════════════════════════════════════
# 12. BacktestReport  (V4.1 feature)
# ══════════════════════════════════════

class BacktestReport:
    """回测报告生成与导出 V4.1 feature

    D1/D2 收敛: 保留独立实现（能力未合并）——报告生成能力与
    backtest_engine.BacktestReporter 重叠，但本类为静态方法直接消费
    pro 版 BacktestResult（engine 版为实例方法消费 engine 版结果），
    签名/数据模型不兼容，保留独立实现。
    """

    @staticmethod
    def generate_full_report(backtest_result) -> str:
        """生成完整图文报告（文本版）

        Parameters
        ----------
        backtest_result : BacktestResult or dict
            BacktestResult 对象或包含 returns/metrics 的 dict

        Returns
        -------
        str  格式化的文本报告
        """
        # 兼容 BacktestResult 和 dict
        if hasattr(backtest_result, "returns"):
            returns = backtest_result.returns
            metrics = backtest_result.metrics if hasattr(backtest_result, "metrics") else {}
            trades = backtest_result.trades if hasattr(backtest_result, "trades") else []
            equity = backtest_result.equity_curve if hasattr(backtest_result, "equity_curve") else None
        else:
            returns = backtest_result.get("returns", pd.Series())
            metrics = backtest_result.get("metrics", {})
            trades = backtest_result.get("trades", [])
            equity = backtest_result.get("equity_curve", None)

        lines = []
        sep = "═" * 60

        # ── 头部 ──
        lines.append(sep)
        lines.append(f"  回测完整报告 (V4.1 feature)")
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(sep)

        # ── 核心绩效 ──
        if metrics:
            lines.append("")
            lines.append("  📊 核心绩效指标")
            lines.append("  " + "─" * 40)
            fmt_list = [
                ("累计收益率", "total_return", "{:.2%}"),
                ("年化收益率", "annualized_return", "{:.2%}"),
                ("年化波动率", "annualized_vol", "{:.2%}"),
                ("夏普比率", "sharpe_ratio", "{:.3f}"),
                ("Sortino比率", "sortino_ratio", "{:.3f}"),
                ("Calmar比率", "calmar_ratio", "{:.3f}"),
                ("最大回撤", "max_drawdown", "{:.2%}"),
                ("胜率", "win_rate", "{:.1%}"),
                ("盈亏比", "profit_factor", "{:.2f}"),
                ("VaR(95%)", "var_95", "{:.2%}"),
                ("CVaR(95%)", "cvar_95", "{:.2%}"),
            ]
            for label, key, fmt in fmt_list:
                if key in metrics:
                    val = metrics[key]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            lines.append(f"    {label:>12}: {fmt.format(val)}")
                        except Exception:
                            lines.append(f"    {label:>12}: {val}")
            if "n_trades" in metrics:
                lines.append(f"    {'交易次数':>12}: {metrics['n_trades']}")
            if "final_capital" in metrics:
                lines.append(f"    {'最终资金':>12}: {metrics['final_capital']:,.2f}")

        # ── 净值曲线摘要 ──
        if equity is not None and len(equity) > 0:
            lines.append("")
            lines.append("  📈 净值曲线摘要")
            lines.append("  " + "─" * 40)
            lines.append(f"    起始净值: {equity.iloc[0]:,.2f}")
            lines.append(f"    最终净值: {equity.iloc[-1]:,.2f}")
            lines.append(f"    最高净值: {equity.max():,.2f}")
            lines.append(f"    最低净值: {equity.min():,.2f}")

        # ── 交易统计 ──
        if trades:
            lines.append("")
            lines.append("  💹 交易统计")
            lines.append("  " + "─" * 40)
            lines.append(f"    总交易次数: {len(trades)}")
            avg_hold = np.mean([getattr(t, "n_holding_days", 1) if hasattr(t, "n_holding_days") else 1 for t in trades])
            lines.append(f"    平均持仓天数: {avg_hold:.1f}")

            # 分多空统计
            buys = [t for t in trades if (hasattr(t, "side") and t.side == "buy") or (isinstance(t, dict) and t.get("side") == "buy")]
            sells = [t for t in trades if (hasattr(t, "side") and t.side == "sell") or (isinstance(t, dict) and t.get("side") == "sell")]
            lines.append(f"    买入次数: {len(buys)}")
            lines.append(f"    卖出次数: {len(sells)}")

            # 收益率分布
            pnls = [getattr(t, "return_pct", 0) if hasattr(t, "return_pct") else (isinstance(t, dict) and t.get("return_pct", 0)) or 0 for t in trades]
            pnl_arr = np.array(pnls).flatten()
            if len(pnl_arr) > 0:
                lines.append(f"    单笔平均收益: {np.mean(pnl_arr):.4%}")
                lines.append(f"    单笔收益中位数: {np.median(pnl_arr):.4%}")
                lines.append(f"    单笔收益标准差: {np.std(pnl_arr):.4%}")
                lines.append(f"    最大单笔收益: {np.max(pnl_arr):.4%}")
                lines.append(f"    最大单笔亏损: {np.min(pnl_arr):.4%}")

        # ── 尾部风险 ──
        if returns is not None and len(returns) > 0:
            lines.append("")
            lines.append("  ⚠️ 尾部风险分析")
            lines.append("  " + "─" * 40)
            lines.append(f"    偏度: {returns.skew():.4f}")
            lines.append(f"    峰度: {returns.kurtosis():.4f}")
            lines.append(f"    1%分位: {returns.quantile(0.01):.4%}")
            lines.append(f"    5%分位: {returns.quantile(0.05):.4%}")
            lines.append(f"    最大单日跌幅: {returns.min():.4%}")

            # 连续亏损
            neg_streaks = BacktestReport._consecutive_negatives(returns)
            if neg_streaks:
                lines.append(f"    最长连续亏损天数: {max(neg_streaks)}")
                lines.append(f"    平均连续亏损天数: {np.mean(neg_streaks):.1f}")

        # ── 月度分析 ──
        if returns is not None and len(returns) > 0:
            lines.append("")
            lines.append("  📅 月度收益分析")
            lines.append("  " + "─" * 40)
            monthly = returns.groupby([returns.index.year, returns.index.month]).sum()
            positive_months = (monthly > 0).mean()
            lines.append(f"    月胜率: {positive_months:.1%}")
            lines.append(f"    月均收益: {monthly.mean():.4%}")
            lines.append(f"    月收益波动: {monthly.std():.4%}")
            lines.append(f"    最佳月份: {monthly.max():.4%}")
            lines.append(f"    最差月份: {monthly.min():.4%}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    @staticmethod
    def _consecutive_negatives(returns: pd.Series) -> list[int]:
        """统计连续亏损天数"""
        streaks = []
        cur = 0
        for val in returns.values:
            if val < 0:
                cur += 1
            else:
                if cur > 0:
                    streaks.append(cur)
                    cur = 0
        if cur > 0:
            streaks.append(cur)
        return streaks

    @staticmethod
    def export_to_excel(result, path: str) -> None:
        """导出回测结果到Excel

        工作表:
        - 绩效指标
        - 净值曲线
        - 交易记录
        - 月度收益率
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            logger.warning("openpyxl 未安装，使用 ExcelWriter 替代")

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # 兼容 BacktestResult 和 dict
            if hasattr(result, "returns"):
                returns = result.returns
                metrics = result.metrics if hasattr(result, "metrics") else {}
                trades = result.trades if hasattr(result, "trades") else []
                equity = result.equity_curve if hasattr(result, "equity_curve") else None
            else:
                returns = result.get("returns", pd.Series())
                metrics = result.get("metrics", {})
                trades = result.get("trades", [])
                equity = result.get("equity_curve", None)

            # ── 绩效指标表 ──
            metrics_df = pd.DataFrame([metrics]) if metrics else pd.DataFrame()
            if not metrics_df.empty:
                metrics_df.T.reset_index().rename(
                    columns={"index": "指标", 0: "值"}
                ).to_excel(writer, sheet_name="绩效指标", index=False)

            # ── 净值曲线 ──
            if equity is not None and len(equity) > 0:
                equity.to_frame("净值").to_excel(writer, sheet_name="净值曲线")
            elif returns is not None and len(returns) > 0:
                cum = (1 + returns).cumprod()
                cum.to_frame("累计净值").to_excel(writer, sheet_name="净值曲线")

            # ── 交易记录 ──
            if trades:
                trade_records = []
                for t in trades:
                    if hasattr(t, "__dict__"):
                        trade_records.append(t.__dict__)
                    elif isinstance(t, dict):
                        trade_records.append(t)
                    else:
                        trade_records.append({"trade": str(t)})
                pd.DataFrame(trade_records).to_excel(
                    writer, sheet_name="交易记录", index=False
                )

            # ── 月度收益率 ──
            if returns is not None and len(returns) > 0:
                monthly = returns.groupby(
                    [returns.index.year, returns.index.month]
                ).agg(["sum", "mean", "std", "count"])
                monthly.columns = ["月收益", "日均收益", "日均波动", "交易日数"]
                monthly.to_excel(writer, sheet_name="月度收益率")

        logger.info(f"回测报告已导出到 {path}")

    @staticmethod
    def summary_table(results: list) -> pd.DataFrame:
        """多回测汇总表

        Parameters
        ----------
        results : list[BacktestResult | dict]
            多个回测结果

        Returns
        -------
        pd.DataFrame  每行一个回测的汇总指标
        """
        rows = []
        for i, r in enumerate(results):
            if hasattr(r, "metrics"):
                m = r.metrics
            elif isinstance(r, dict):
                m = r.get("metrics", {})
            else:
                m = {}
            row = {
                "回测#": i + 1,
                "累计收益": m.get("total_return"),
                "年化收益": m.get("annualized_return"),
                "年化波动": m.get("annualized_vol"),
                "夏普": m.get("sharpe_ratio"),
                "Calmar": m.get("calmar_ratio"),
                "最大回撤": m.get("max_drawdown"),
                "胜率": m.get("win_rate"),
                "盈亏比": m.get("profit_factor"),
                "交易次数": m.get("n_trades"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if "回测#" in df.columns:
            df = df.set_index("回测#")
        return df


# ══════════════════════════════════════
# Module exports
# ══════════════════════════════════════

__all__ = [
    "BacktestResult", "Trade",
    "MultiAssetBacktest",
    "CorporateActionsHandler",
    "SurvivalBiasCorrector",
    "SubPeriodAnalyzer",
    "WalkForwardBacktest",
    "ComprehensiveMetrics",
    # V4.1 new classes
    "PerformanceAttribution",
    "MonteCarloBacktest",
    "ParameterSweeper",
    "BenchmarkComparison",
    "MultiPeriodOptimizer",
    "BacktestReport",
    # Survivorship integration
    "load_delisted_prices",
    "integrate_delisted",
]
