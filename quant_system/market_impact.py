"""
market_impact.py — # V4.1 feature: Almgren-Chriss 市场冲击模型

A股市场冲击成本模型，基于 Almgren & Chriss (2001) 最优执行框架。

理论基础:
  - Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-39.
  - 永久冲击 (Permanent Impact): 持久性的价格影响，与成交量线性相关
  - 临时冲击 (Temporary Impact): 短暂的反向冲击，与交易速率线性相关
  - 波动成本 (Timing Risk): 交易期间由波动带来的不确定性成本

核心功能:
  - permanent_impact()   — 计算永久性市场冲击成本
  - temporary_impact()   — 计算临时性市场冲击成本
  - total_cost()         — 交易总成本（冲击 + 价差 + 波动 + A股显性费用）
  - optimal_trajectory() — 均匀 TWAP 交易路径（VWAP 变体，非严格 AC 最优）
  - optimal_tau()        — 给定风险厌恶下的最优执行时长
  - cost_report()        — 格式化输出冲击成本分析报告
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_system import execution_broker as _execution_broker


# ---------------------------------------------------------------------------
# Constants — 默认参数（A股经验值，可调）
# ---------------------------------------------------------------------------
DEFAULT_VOLATILITY: float = 0.30            # 年化波动率 30%
DEFAULT_SPREAD: float = 0.001               # 相对买卖价差 0.1%
DEFAULT_DAILY_VOLUME: float = 1e8           # 日均成交额 1 亿（占流动性比值时忽略）
DEFAULT_TRADING_DAYS: int = 242             # A股年化交易日
DEFAULT_PARTICIPATION_RATE: float = 0.1     # 默认参与率 10%

# 冲击系数（A股经验校准，可后续替换为回测估计值）
DEFAULT_PERMANENT_ETA: float = 2.5e-7       # 永久冲击系数
DEFAULT_TEMPORARY_ETA: float = 1.0e-6       # 临时冲击系数
DEFAULT_TEMPORARY_GAMMA: float = 0.5        # 临时冲击指数（~sqrt 规律）
DEFAULT_SPREAD_COST_FRACTION: float = 0.5   # 单边冲击中价差占比

# D3/D组收敛: 费率唯一真源为 execution_broker 常量（同数值同方向），此处改为引用。
# P2-Q27-fix(M336): A股显性费用常量（Grinold-Kahn 成本要素框架）
DEFAULT_COMMISSION_RATE: float = _execution_broker.COMMISSION_RATE      # 佣金率 万0.85（双边收取）
DEFAULT_MIN_COMMISSION: float = _execution_broker.MIN_COMMISSION        # 单笔最低佣金 5 元
DEFAULT_STAMP_TAX_RATE: float = _execution_broker.STAMP_TAX_RATE        # 印花税 0.05%（仅卖出）
DEFAULT_TRANSFER_FEE_RATE: float = _execution_broker.TRANSFER_FEE_RATE  # 过户费 0.001%（双边收取）


# ---------------------------------------------------------------------------
# Almgren-Chriss 参数容器
# ---------------------------------------------------------------------------
@dataclass
class ACParams:
    """
    Almgren-Chriss 模型参数。

    Attributes:
        sigma:          年化波动率 (decimal)
        spread:         相对买卖价差 (decimal)
        daily_volume:   标的日均成交额（元）
        permanent_eta:  永久冲击系数  η_p
        temporary_eta:  临时冲击系数  η_t
        temporary_gamma: 临时冲击指数 γ (通常 0.3~0.6)
        risk_aversion:  风险厌恶系数 λ (仅用于最优执行)
        trading_days:   年化交易日数
    """
    sigma: float = DEFAULT_VOLATILITY
    spread: float = DEFAULT_SPREAD
    daily_volume: float = DEFAULT_DAILY_VOLUME
    permanent_eta: float = DEFAULT_PERMANENT_ETA
    temporary_eta: float = DEFAULT_TEMPORARY_ETA
    temporary_gamma: float = DEFAULT_TEMPORARY_GAMMA
    risk_aversion: float = 1e-6
    trading_days: int = DEFAULT_TRADING_DAYS
    # P2-Q27-fix(M336): A股显性费用参数
    commission_rate: float = DEFAULT_COMMISSION_RATE
    min_commission: float = DEFAULT_MIN_COMMISSION
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE
    transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE

    @classmethod
    def a_share_defaults(cls) -> "ACParams":
        """返回 A 股默认参数（经验值，请根据回测校准）。"""
        return cls()

    @classmethod
    def liquid_stock(cls) -> "ACParams":
        """流动性好的大盘股参数：冲击小、深度高。"""
        return cls(
            sigma=0.25,
            spread=0.0005,
            daily_volume=5e9,
            permanent_eta=1.0e-7,
            temporary_eta=5.0e-7,
        )

    @classmethod
    def illiquid_stock(cls) -> "ACParams":
        """流动性差的小盘股参数：冲击大、深度低。"""
        return cls(
            sigma=0.45,
            spread=0.003,
            daily_volume=5e7,
            permanent_eta=5.0e-7,
            temporary_eta=3.0e-6,
        )


# ---------------------------------------------------------------------------
# 单笔交易冲击结果
# ---------------------------------------------------------------------------
@dataclass
class ImpactResult:
    """
    单笔交易的冲击成本分解。

    Attributes:
        total_shares:       交易股数
        total_value:        交易金额（本币）
        avg_price:          交易均价（本币）
        arrival_price:      决策时点参考价
        permanent_impact:   永久冲击成本（基点 bps）
        temporary_impact:   临时冲击成本（基点 bps）
        spread_cost:        价差成本（基点 bps）
        timing_risk:        波动风险成本（基点 bps）
        total_cost_bps:     总成本（基点）
        total_cost_value:   总成本（本币）
        participation_rate: 参与率（占日均成交量比例）
        execution_time:     执行时长（交易日数）
    """
    total_shares: float
    total_value: float
    avg_price: float
    arrival_price: float
    permanent_impact: float
    temporary_impact: float
    spread_cost: float
    timing_risk: float
    explicit_cost_bps: float = 0.0   # P2-Q27-fix(M336): A股显性费用（佣金/印花税/过户费）
    total_cost_bps: float = 0.0
    total_cost_value: float = 0.0
    participation_rate: float = 0.0
    execution_time: float = 1.0

    # P1-Q27-fix: 旧调用方 (execution_broker/tests) 以 dict 方式读取字段，
    # 提供 __getitem__/__contains__ 保持向后兼容，新调用方仍用属性访问。
    _LEGACY_KEYS = {
        "total_slippage_bp": "total_cost_bps",
        "total_cost_bps": "total_cost_bps",
        "participation_rate": "participation_rate",
        "total_value": "total_value",
        "avg_price": "avg_price",
        "arrival_price": "arrival_price",
        "permanent_impact": "permanent_impact",
        "temporary_impact": "temporary_impact",
        "spread_cost": "spread_cost",
        "timing_risk": "timing_risk",
        "explicit_cost_bps": "explicit_cost_bps",
        "execution_time": "execution_time",
    }

    def __getitem__(self, key: str) -> float:
        if key == "cost_rate":
            # P2-Q27-fix(M336): 滑点小数口径仅含冲击+价差（不含显性费用），
            # execution_broker 用该键计算成交价，其自身再按佣金/印花税/过户费
            # 做现金结算，若此处包含显性费用会与 execution_broker 重复计费。
            return (self.total_cost_bps - self.explicit_cost_bps) / 1e4
        if key in self._LEGACY_KEYS:
            return getattr(self, self._LEGACY_KEYS[key])
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key == "cost_rate" or key in self._LEGACY_KEYS or hasattr(self, key)


# ---------------------------------------------------------------------------
# MarketImpact —— 核心类
# ---------------------------------------------------------------------------
class MarketImpact:
    """
    Almgren-Chriss 市场冲击模型。

    用法:
        mi = MarketImpact(params=ACParams.a_share_defaults())
        result = mi.total_cost(shares=100000, arrival_price=10.0,
                                daily_vol=1e8, horizon_days=1)
        print(result)

    也支持 pandas Series/DataFrame 批量计算:
        result_df = mi.cost_dataframe(trade_records_df)
    """

    def __init__(self, params: ACParams | None = None):
        """
        Args:
            params: Almgren-Chriss 参数。默认 AShareDefaults。
        """
        self.p = params or ACParams.a_share_defaults()

    # ------------------------------------------------------------------
    # 核心冲击公式
    # ------------------------------------------------------------------

    def permanent_impact(
        self,
        shares: float | np.ndarray,
        total_shares_outstanding: float | None = None,
        arrival_price: float | None = None,
        daily_vol: float | None = None,
    ) -> float | np.ndarray:
        """
        永久性市场冲击（基点 bps）。

        Almgren-Chriss 永久冲击与成交量成正比:
            I_perm = eta_p * (X / V)
        其中 X 为交易量, V 为日均成交量。

        Args:
            shares:             交易股数
            total_shares_outstanding: 总流通股数（如提供则额外计算持仓比例指标）
            arrival_price:      决策时参考价，用于把日均成交额(元)换算为股数
            daily_vol:          标的日均成交额(元)；None 时回退 self.p.daily_volume

        Returns:
            永久冲击 cost (bps, 单边)
        """
        daily_vol_shares = self._daily_vol_in_shares(arrival_price, daily_vol)
        participation = np.asarray(shares, dtype=float) / daily_vol_shares
        impact = self.p.permanent_eta * participation * 1e4  # 转为 bps

        if total_shares_outstanding is not None:
            # 附加告警: 持仓占比超 5% 时永久冲击显著放大
            pct_outstanding = np.asarray(shares, dtype=float) / total_shares_outstanding
            if np.any(pct_outstanding > 0.05):
                # 超过流通盘 5% 时线性放大永久冲击系数
                multiplier = 1.0 + 10.0 * (pct_outstanding - 0.05)
                impact = impact * multiplier

        return impact

    def temporary_impact(
        self,
        shares: float | np.ndarray,
        horizon_days: float | np.ndarray = 1.0,
        arrival_price: float | None = None,
        daily_vol: float | None = None,
    ) -> float | np.ndarray:
        """
        临时性市场冲击（基点 bps）。

        Almgren-Chriss 临时冲击与交易速率有关:
            I_temp = eta_t * sigma * (X / (V * T))^gamma
        其中 T 为执行天数。

        Args:
            shares:       交易股数
            horizon_days: 执行天数。默认 1 个交易日。
            arrival_price:决策时参考价，用于把日均成交额(元)换算为股数
            daily_vol:    标的日均成交额(元)；None 时回退 self.p.daily_volume

        Returns:
            临时冲击 cost (bps, 单边)
        """
        daily_vol_shares = self._daily_vol_in_shares(arrival_price, daily_vol)
        trade_rate = np.asarray(shares, dtype=float) / (daily_vol_shares * horizon_days)
        impact = (
            self.p.temporary_eta
            * self.p.sigma
            * (trade_rate ** self.p.temporary_gamma)
            * 1e4  # 转为 bps
        )
        return impact

    def spread_cost_component(self) -> float:
        """
        价差成本分量（基点 bps）。

        按单边冲击假设计算，实际成交时一半价差成本计入买入/卖出:
            c_spread = spread / 2
        """
        return self.p.spread * DEFAULT_SPREAD_COST_FRACTION * 1e4

    def timing_risk(
        self,
        shares: float | np.ndarray,
        arrival_price: float | np.ndarray,
        horizon_days: float | np.ndarray = 1.0,
        daily_vol: float | None = None,
    ) -> float | np.ndarray:
        """
        波动风险成本（基点 bps）。

        表示执行期间由于价格波动导致的不确定性:
            risk = sigma * sqrt(T / trading_days) * (X / V)
        即在执行期间内预期波动带来的额外冲击。

        Args:
            shares:       交易股数
            arrival_price:决策时参考价（本币）
            horizon_days: 执行天数
            daily_vol:    标的日均成交额(元)；None 时回退 self.p.daily_volume

        Returns:
            波动风险成本 (bps)
        """
        daily_vol_shares = self._daily_vol_in_shares(arrival_price, daily_vol)
        participation = np.asarray(shares, dtype=float) / daily_vol_shares
        sigma_daily = self.p.sigma / math.sqrt(self.p.trading_days)
        risk = sigma_daily * math.sqrt(horizon_days) * participation * 1e4
        return risk

    def explicit_cost(
        self,
        shares: float,
        arrival_price: float,
        side: str = "buy",
    ) -> float:
        """A股显性费用（基点 bps）。

        P2-Q27-fix(M336): 原成本模型仅含冲击+价差，未覆盖 A 股显性费用。
        此处按现行规则补全（Grinold-Kahn 成本要素框架）:
          - 佣金:  成交金额 × 万0.85，单笔最低 5 元（双边收取）
          - 印花税: 成交金额 × 0.05%，仅卖出方向征收
          - 过户费: 成交金额 × 0.001%（双边收取）

        Args:
            shares:       交易股数
            arrival_price:参考价（元）
            side:         "buy" | "sell"

        Returns:
            显性费用 (bps)
        """
        if arrival_price is None or arrival_price <= 0 or shares <= 0:
            return 0.0
        value = float(shares) * float(arrival_price)
        commission = max(value * self.p.commission_rate, self.p.min_commission)
        stamp_tax = value * self.p.stamp_tax_rate if side == "sell" else 0.0
        transfer_fee = value * self.p.transfer_fee_rate
        total = commission + stamp_tax + transfer_fee
        return total / value * 1e4

    # ------------------------------------------------------------------
    # 综合费用计算
    # ------------------------------------------------------------------

    def total_cost(
        self,
        shares: float = 0.0,
        arrival_price: float = 0.0,
        daily_vol: float | None = None,
        horizon_days: float = 1.0,
        total_shares_outstanding: float | None = None,
        side: str = "buy",
        # P1-Q27-fix: 兼容旧签名 (execution_broker.py / tests.py 用 trade_value 等)
        trade_value: float | None = None,
        stock_price: float | None = None,
        adv_shares: float | None = None,
        sigma: float | None = None,
    ) -> ImpactResult:
        """
        计算 Almgren-Chriss 总交易成本。

        公式（单边交易）:
            TotalCost = I_perm + I_temp + C_spread + timing_risk + explicit_cost
        其中 explicit_cost 为 A股显性费用（佣金最低5元/笔 + 印花税卖出0.05% + 过户费0.001%双边）。
        不同 side 下符号含义:
            buy  : 冲击成本抬高了买入均价
            sell : 冲击成本压低了卖出均价

        Args:
            shares:                    交易股数
            arrival_price:            决策时点参考价格（元）
            daily_vol:                标的日均成交额（元），为 None 则使用参数默认值
            horizon_days:             执行天数（默认 1 天）
            total_shares_outstanding: 总流通股数（可选，影响永久冲击）
            side:                     交易方向 "buy" | "sell"
            trade_value/stock_price/adv_shares/sigma:
                P1-Q27-fix 兼容旧调用方的参数；传入时自动换算为
                shares = trade_value / stock_price, daily_vol = adv_shares × stock_price。

        Returns:
            ImpactResult — 详细成本分解（同时支持属性访问与旧 dict 键访问）
        """
        # V11 审计修复（Medium）: 原实现 shares=0/arrival_price=0 默认值时
        # _daily_vol_in_shares 抛 ValueError（必崩路径，默认值掩盖调用错误）。
        # 修正: 零股/零价直接返回零成本结果（调用方未传参时的安全兜底）。
        if (shares <= 0 or arrival_price <= 0) and trade_value is None:
            return ImpactResult(
                total_shares=0, total_value=0.0, avg_price=0.0, arrival_price=0.0,
                permanent_impact=0.0, temporary_impact=0.0, spread_cost=0.0,
                timing_risk=0.0, total_cost_bps=0.0, total_cost_value=0.0,
                participation_rate=0.0,
            )

        # P1-Q27-fix: 旧签名兼容换算
        if trade_value is not None or stock_price is not None or adv_shares is not None:
            price = stock_price if (stock_price is not None and stock_price > 0) else arrival_price
            value = trade_value if trade_value is not None else (shares * price if shares else 0.0)
            if price <= 0:
                raise ValueError(
                    "legacy 参数 (trade_value/stock_price/adv_shares) 需要有效的 stock_price>0 才能换算 shares"
                )
            shares = value / price
            arrival_price = price
            if daily_vol is None and adv_shares:
                daily_vol = adv_shares * price  # 日均成交额(元)

        # P2-Q27-fix(M334): 使用本次调用的局部 daily_vol，不再写回 self.p.daily_volume。
        # 长生命周期实例（execution_broker 持有同一 MarketImpact）在带 daily_vol 的调用后，
        # 若被后续不带 daily_vol 的调用读取 self.p.daily_volume 会拿到被污染的值。
        _daily_vol = daily_vol if daily_vol is not None else self.p.daily_volume

        # sigma 覆盖参数中的波动率（仅本次调用有效）
        _old_sigma = None
        if sigma is not None:
            _old_sigma = self.p.sigma
            self.p.sigma = float(sigma)

        try:
            daily_vol_shares = self._daily_vol_in_shares(arrival_price, _daily_vol)
            total_value = shares * arrival_price

            # 参与率
            participation_rate = shares / daily_vol_shares if daily_vol_shares > 0 else 0.0

            # 计算冲击分量（P1-Q27-fix: 传入 arrival_price 以正确换算股数）
            perm = self.permanent_impact(shares, total_shares_outstanding, arrival_price, _daily_vol)
            temp = self.temporary_impact(shares, horizon_days, arrival_price, _daily_vol)
            spr  = self.spread_cost_component()
            risk = self.timing_risk(shares, arrival_price, horizon_days, _daily_vol)
            # P2-Q27-fix(M336): A股显性费用（佣金最低5元/笔、印花税卖出0.05%、过户费0.001%双边）
            explicit = self.explicit_cost(shares, arrival_price, side)

            # 方向性：买方向上的冲击是正的（买入价格更高）
            # 卖方向上的冲击符号上取绝对值，显示为成本
            sign = 1.0
            total_cost_bps = sign * (perm + temp + spr + risk + explicit)

            # 总成本金额 = 总交易额 × 总成本（bps）/ 10000
            total_cost_value = total_value * abs(total_cost_bps) / 1e4

            # 实际成交均价
            #   买入: 均价 = arrival_price * (1 + total_cost_bps / 1e4)
            #   卖出: 均价 = arrival_price * (1 - total_cost_bps / 1e4)
            if side == "buy":
                avg_price = arrival_price * (1.0 + total_cost_bps / 1e4)
            else:
                avg_price = arrival_price * (1.0 - total_cost_bps / 1e4)

            return ImpactResult(
                total_shares=shares,
                total_value=total_value,
                avg_price=avg_price,
                arrival_price=arrival_price,
                permanent_impact=float(perm),
                temporary_impact=float(temp),
                spread_cost=float(spr),
                timing_risk=float(risk),
                explicit_cost_bps=float(explicit),
                total_cost_bps=float(total_cost_bps),
                total_cost_value=float(total_cost_value),
                participation_rate=float(participation_rate),
                execution_time=horizon_days,
            )
        finally:
            if _old_sigma is not None:
                self.p.sigma = _old_sigma

    def total_cost_with_twap(
        self,
        shares: float = 0.0,
        arrival_price: float = 0.0,
        daily_vol: float | None = None,
        minutes: int = 30,
        side: str = "buy",
    ) -> ImpactResult:
        """
        按 TWAP 分拆的冲击成本（分钟级切片后汇总）。

        TWAP 将大单均匀拆分到 N 个切片上，每片冲击远小于整单，
        但执行时间更长，波动风险上升。这是 Almgren-Chriss 的核心
        权衡: 冲击 vs 波动。

        Args:
            shares:       总交易股数
            arrival_price: 参考价格
            daily_vol:    日均成交额（元）
            minutes:      TWAP 执行总时长（分钟）
            side:         "buy" | "sell"

        Returns:
            ImpactResult — 切片汇总后的综合成本
        """
        # 240 分钟 ≈ 1 个交易日，做比例换算
        horizon_days = minutes / 240.0
        slice_count = max(1, int(minutes / 5))  # 每 5 分钟一个切片
        shares_per_slice = shares / slice_count

        # 对单一片算冲击（horizon 同步 ÷slice_count → 速率与整单相同）
        single = self.total_cost(
            shares=shares_per_slice,
            arrival_price=arrival_price,
            daily_vol=daily_vol,
            horizon_days=horizon_days / slice_count,
            side=side,
        )
        # P1-Q27-fix: 汇总各组件
        #   - 临时冲击按交易速率计算：单切片与整单速率相同(horizon 同步÷slice_count)，
        #     汇总 bps 应取单片 temp(≈整单 temp)，原实现 ×slice_count 高估 slice_count 倍。
        #   - 永久冲击线性于成交量：Σ N 片 = 整单，须 ×slice_count 还原整单 perm。
        #   - 价差成本与切片数无关，取单片即可。
        #   - 波动风险按整单全时长重算。
        #   - P2-Q27-fix(M336): 显性费用按整单计算（单笔订单佣金/印花税/过户费）。
        perm_bps = float(single.permanent_impact) * slice_count
        temp_bps = float(single.temporary_impact)
        spread_bps = float(single.spread_cost)
        risk = self.timing_risk(shares, arrival_price, horizon_days, daily_vol=daily_vol)
        explicit = self.explicit_cost(shares, arrival_price, side)
        total_bps = perm_bps + temp_bps + spread_bps + risk + explicit

        # 汇总成本金额 = 总交易额 × 总成本（bps）/ 10000
        total_val = shares * arrival_price * total_bps / 1e4

        return ImpactResult(
            total_shares=shares,
            total_value=shares * arrival_price,
            avg_price=arrival_price * (1 + total_bps / 1e4) if side == "buy"
                      else arrival_price * (1 - total_bps / 1e4),
            arrival_price=arrival_price,
            permanent_impact=perm_bps,
            temporary_impact=temp_bps,
            spread_cost=spread_bps,
            timing_risk=float(risk),
            explicit_cost_bps=float(explicit),
            total_cost_bps=float(total_bps),
            total_cost_value=float(total_val),
            participation_rate=shares / max(self._daily_vol_in_shares(arrival_price, daily_vol), 1),
            execution_time=horizon_days,
        )

    # ------------------------------------------------------------------
    # 最优执行路径
    # ------------------------------------------------------------------

    def optimal_trajectory(
        self,
        shares: float,
        arrival_price: float,
        daily_vol: float | None = None,
        horizon_days: float = 1.0,
        n_steps: int = 10,
        lambda_: float | None = None,
        side: str = "buy",
    ) -> pd.DataFrame:
        # V11 审计修复（Medium）: 新增 side 参数（默认 buy 兼容旧调用），
        # 卖出路径正确计印花税。
        """
        均匀 TWAP 交易路径（VWAP 变体）。

        P2-Q27-fix(M335): 原 docstring 声称"AC 最优路径/VWAP 变体"，但实现恒为
        均匀 TWAP（每期权重 1/N），并非严格 AC 最优——本模型的临时冲击为幂律形式
        (γ≈0.5)，AC 闭式最优解（sinh 型）仅适用于线性冲击，此处不做数值优化，
        故如实描述为均匀 TWAP 路径。

        每期交易量 x_j = X / N，成本 = 永久冲击 + 临时冲击 + 价差 + 显性费用。
        风险项按 AC 离散公式使用本期交易后的期末持仓 x_j（原实现误用期初 remaining）。

        Args:
            shares:       总交易股数
            arrival_price: 参考价
            daily_vol:    日均成交额
            horizon_days: 执行天数
            n_steps:      分割期数
            lambda_:      风险厌恶系数，默认使用 params.risk_aversion

        Returns:
            DataFrame 每期交易计划
                columns: period, trade_shares, remaining, price_impact,
                         cost_bps, cost_value, risk_term
        """
        # P2-Q27-fix(M334): 局部 daily_vol，不写回 self.p.daily_volume
        _daily_vol = daily_vol if daily_vol is not None else self.p.daily_volume
        lam = lambda_ if lambda_ is not None else self.p.risk_aversion

        daily_vol_shares = self._daily_vol_in_shares(arrival_price, _daily_vol)

        # 每期成交量权重（按均匀分布 = TWAP）
        weights = np.ones(n_steps) / n_steps
        # 也可以用 VWAP 权重: 按日内成交量曲线
        # weights = _vwap_weights(n_steps)

        dt = horizon_days / n_steps
        trade_per_step = shares * weights

        remaining = shares
        records: list[dict[str, Any]] = []

        for j in range(n_steps):
            tj = (j + 1) * dt  # 该期结束时刻
            x_j = trade_per_step[j]

            perm = self.permanent_impact(x_j, arrival_price=arrival_price, daily_vol=_daily_vol)
            temp = self.temporary_impact(x_j, dt, arrival_price=arrival_price, daily_vol=_daily_vol)
            spr  = self.spread_cost_component()
            # P2-Q27-fix(M336): 每期显性费用（切片独立成单，佣金单笔最低5元）
            explicit = self.explicit_cost(x_j, arrival_price, side=side)
            cost_bps = perm + temp + spr + explicit
            cost_val = x_j * arrival_price * cost_bps / 1e4

            remaining -= x_j
            # P2-Q27-fix(M335): 风险项改用本期交易后的期末持仓（AC 离散公式为 x_j^2）
            hold_j = max(remaining, 0.0)
            sigma_daily = self.p.sigma / math.sqrt(self.p.trading_days)
            risk_term = lam * (sigma_daily ** 2) * (hold_j ** 2) * dt

            records.append({
                "period": j + 1,
                "trade_shares": round(x_j, 0),
                "remaining": round(hold_j, 0),
                "price_impact_bps": round(float(cost_bps), 2),
                "cost_bps": round(float(cost_bps), 2),
                "cost_value": round(float(cost_val), 2),
                "risk_term": round(float(risk_term), 2),
            })
            if remaining <= 0:
                break

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 批量计算
    # ------------------------------------------------------------------

    def cost_dataframe(
        self,
        trades: pd.DataFrame,
        price_col: str = "price",
        shares_col: str = "shares",
        daily_vol_col: str | None = None,
        horizon_col: str | None = None,
        side_col: str | None = None,
    ) -> pd.DataFrame:
        """
        对 DataFrame 批量计算每笔交易的冲击成本。

        Args:
            trades:        包含交易记录的 DataFrame
            price_col:     参考价格列名
            shares_col:    交易股数列名
            daily_vol_col: 日均成交额列名（可选）
            horizon_col:   执行天数列名（可选）
            side_col:      方向列名（可选，"buy"/"sell"）

        Returns:
            原 DataFrame 附加冲击成本列
        """
        df = trades.copy()

        df["_daily_vol"] = (
            df[daily_vol_col] if daily_vol_col and daily_vol_col in df.columns
            else self.p.daily_volume
        )
        df["_horizon"] = (
            df[horizon_col] if horizon_col and horizon_col in df.columns
            else 1.0
        )
        df["_side"] = (
            df[side_col] if side_col and side_col in df.columns
            else "buy"
        )

        results = df.apply(
            lambda row: self.total_cost(
                shares=row[shares_col],
                arrival_price=row[price_col],
                daily_vol=row["_daily_vol"],
                horizon_days=row["_horizon"],
                side=row["_side"],
            ),
            axis=1,
        )

        df["perm_impact_bps"] = results.apply(lambda r: r.permanent_impact)
        df["temp_impact_bps"] = results.apply(lambda r: r.temporary_impact)
        df["spread_cost_bps"] = results.apply(lambda r: r.spread_cost)
        df["total_cost_bps"] = results.apply(lambda r: r.total_cost_bps)
        df["total_cost_value"] = results.apply(lambda r: r.total_cost_value)
        df["avg_exec_price"] = results.apply(lambda r: r.avg_price)
        df["participation_rate"] = results.apply(lambda r: r.participation_rate)

        return df

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def cost_report(self, result: ImpactResult) -> str:
        """
        格式化输出冲击成本报告。

        Args:
            result: ImpactResult 对象

        Returns:
            格式化的文本报告
        """
        lines = [
            "=" * 60,
            "Almgren-Chriss 市场冲击成本报告  # V4.1",
            "=" * 60,
            f"  交易金额:           {result.total_value:>14,.2f} 元",
            f"  交易股数:           {result.total_shares:>14,.0f} 股",
            f"  参考均价:           {result.arrival_price:>14.4f} 元",
            f"  实际均价:           {result.avg_price:>14.4f} 元",
            f"  执行天数:           {result.execution_time:>14.2f} 天",
            f"  参与率:             {result.participation_rate:>13.2%}",
            "",
            f"  永久冲击成本:       {result.permanent_impact:>14.2f} bps  "
            f"({result.permanent_impact * result.total_value / 1e4:>,.2f} 元)",
            f"  临时冲击成本:       {result.temporary_impact:>14.2f} bps  "
            f"({result.temporary_impact * result.total_value / 1e4:>,.2f} 元)",
            f"  价差成本:           {result.spread_cost:>14.2f} bps  "
            f"({result.spread_cost * result.total_value / 1e4:>,.2f} 元)",
            f"  波动风险成本:       {result.timing_risk:>14.2f} bps  "
            f"({result.timing_risk * result.total_value / 1e4:>,.2f} 元)",
            f"  显性费用:           {result.explicit_cost_bps:>14.2f} bps  "
            f"({result.explicit_cost_bps * result.total_value / 1e4:>,.2f} 元)",
            "",
            f"  总成本:             {result.total_cost_bps:>14.2f} bps",
            f"  总成本金额:         {result.total_cost_value:>14,.2f} 元",
            "=" * 60,
        ]
        return "\n".join(lines)

    def optimal_tau(
        self,
        shares: float,
        arrival_price: float,
        daily_vol: float | None = None,
        risk_aversion: float | None = None,
    ) -> float:
        """
        给定风险厌恶系数下的最优执行时长（交易日）。

        当临时冲击和波动风险平衡时:
            tau_opt = ( (eta_t * sigma * X) / (lambda * sigma^2 * V) ) ^ (2/(2+gamma))
        此处仅作示意，完整推导参见 Almgren & Chriss (2001) Sec 3.3。

        Args:
            shares:        交易股数
            arrival_price: 参考价
            daily_vol:     日均成交额
            risk_aversion: 风险厌恶系数 lambda

        Returns:
            最优执行天数
        """
        # P2-Q27-fix(M334): 局部 daily_vol，不写回 self.p.daily_volume
        _daily_vol = daily_vol if daily_vol is not None else self.p.daily_volume
        lam = risk_aversion if risk_aversion is not None else self.p.risk_aversion

        daily_vol_shares = self._daily_vol_in_shares(arrival_price, _daily_vol)
        if daily_vol_shares <= 0 or lam <= 0:
            return 1.0

        ratio = shares / daily_vol_shares
        sigma_daily = self.p.sigma / math.sqrt(self.p.trading_days)

        # 简化版最优时长
        numerator = self.p.temporary_eta * self.p.sigma * ratio
        denominator = lam * (sigma_daily ** 2)
        if denominator <= 0:
            return 1.0

        base = numerator / denominator
        exponent = 2.0 / (2.0 + self.p.temporary_gamma)
        tau = math.pow(base, exponent)
        # 限制取值范围
        return max(0.01, min(tau, 20.0))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _daily_vol_in_shares(self, price: float | None = None,
                             daily_vol: float | None = None) -> float:
        """将日均成交额(元)转为股数。

        P1-Q27-fix: 原实现当 price 为 None 时直接把 daily_volume(元) 当作股数，
        导致永久/临时冲击的参与率被低估约 price 倍。此处强制要求提供有效价格，
        无法换算时显式抛错（禁止静默把金额当股数）。

        P2-Q27-fix(M334): 允许通过 daily_vol 参数显式传入本次调用的日均成交额，
        不再依赖/修改 self.p.daily_volume，避免长生命周期实例状态被污染。
        """
        dv = daily_vol if daily_vol is not None else self.p.daily_volume
        if price is None or price <= 0:
            raise ValueError(
                "必须提供有效的 arrival_price>0 才能把日均成交额(元)换算为股数；"
                f"当前 price={price!r}, daily_volume={dv!r}"
            )
        return dv / price

    def __repr__(self) -> str:
        return (
            f"MarketImpact(sigma={self.p.sigma:.2%}, "
            f"spread={self.p.spread:.2%}, "
            f"daily_vol={self.p.daily_volume:.2e})"
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _vwap_weights(n: int) -> np.ndarray:
    """
    模拟 A 股日内成交量分布权重。

    参考: A 股日内成交量呈 U 型分布，开盘和收盘各约 15%。
    典型权重:
        开盘 (前30min): ~20%
        午盘前 (10:30-11:30): ~15%
        午盘后 (13:00-14:00): ~30%
        尾盘 (14:00-15:00): ~35%
    """
    x = np.linspace(0, 1, n)
    # Beta 分布模拟 U 型
    weights = np.exp(-((x - 0.5) ** 2) / 0.1) + 0.5
    weights = weights * (1.0 + 0.8 * np.sin(np.pi * x))  # 突出开盘尾盘
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# __main__ 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ---- 演示 1: 大盘股 100 万买入 ----
    print("\n>>> 大盘股买入 100 万元")
    mi = MarketImpact(ACParams.liquid_stock())
    r1 = mi.total_cost(
        shares=50000,        # 50,000 股
        arrival_price=20.0,  # 价格 20 元 => 100 万
        daily_vol=5e9,
        horizon_days=1.0,
        side="buy",
    )
    print(mi.cost_report(r1))

    # ---- 演示 2: 小盘股 10 万买入 ----
    print("\n>>> 小盘股买入 10 万元")
    mi2 = MarketImpact(ACParams.illiquid_stock())
    r2 = mi2.total_cost(
        shares=50000,         # 50,000 股
        arrival_price=2.0,    # 价格 2 元 => 10 万
        daily_vol=5e7,
        horizon_days=1.0,
        side="buy",
    )
    print(mi2.cost_report(r2))

    # ---- 演示 3: TWAP 分拆对比 ----
    print("\n>>> 整单 vs TWAP 分拆 100 万买入")
    r_whole = mi.total_cost(
        shares=50000, arrival_price=20.0, daily_vol=5e9,
        horizon_days=1.0, side="buy",
    )
    r_twap = mi.total_cost_with_twap(
        shares=50000, arrival_price=20.0, daily_vol=5e9,
        minutes=30, side="buy",
    )
    print(f"  整单冲击: {r_whole.total_cost_bps:.2f} bps  ({r_whole.total_cost_value:,.2f} 元)")
    print(f"  30min TWAP: {r_twap.total_cost_bps:.2f} bps  ({r_twap.total_cost_value:,.2f} 元)")

    # ---- 演示 4: 最优交易路径 ----
    print("\n>>> 最优交易路径 (10 期)")
    traj = mi.optimal_trajectory(
        shares=50000, arrival_price=20.0, daily_vol=5e9,
        horizon_days=1.0, n_steps=10,
    )
    print(traj.to_string(index=False))
