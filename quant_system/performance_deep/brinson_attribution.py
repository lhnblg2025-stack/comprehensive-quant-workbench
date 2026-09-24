"""
brinson_attribution — Brinson 归因分析 (V5)

核心问题：组合跑赢/跑输基准的超额收益，有多少来自"选对了行业"，
有多少来自"在行业内选对了股票"？

Brinson-Fachler 模型将主动收益拆解为三部分：
  1. 配置效应 (Allocation)  — 行业超/低配贡献
  2. 选择效应 (Selection)   — 行业内个股选择贡献
  3. 交互效应 (Interaction) — 配置与选择的交叉贡献

公式（以行业 i 为例）：
  Allocation_i  = (w_pf_i - w_bm_i) * (R_bm_i - R_bm)
  Selection_i   = w_bm_i * (R_pf_i - R_bm_i)
  Interaction_i = (w_pf_i - w_bm_i) * (R_pf_i - R_bm_i)

  总主动收益 = Σ(Allocation_i + Selection_i + Interaction_i) = R_pf - R_bm

对标：Brinson, Hood & Beebower (1986) / Brinson & Fachler (1985)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


class BrinsonAttribution:
    """Brinson 归因分析引擎。

    将组合相对基准的主动收益拆解为行业配置效应、个股选择效应与交互效应。

    核心假设：
      总主动收益 = Σ_行业(配置效应 + 选择效应 + 交互效应)
      配置效应回答："行业超低配对不对？"
      选择效应回答："行业内选的股票好不好？"
    """

    def __init__(self) -> None:
        self._last_result: dict[str, Any] | None = None

    @staticmethod
    def _portfolio_total_return(returns: pd.Series | list[float] | np.ndarray) -> float:
        """将日频（或周期）收益率序列复合为区间总收益。

        Args:
            returns: 收益率序列（每期收益率，非累计）

        Returns:
            区间累计收益率（复合）
        """
        arr = pd.Series(returns).dropna().astype(float).values
        if arr.size == 0:
            return 0.0
        return float(np.prod(1.0 + arr) - 1.0)

    def _sector_period_return(
        self,
        sector: str,
        stock_returns_by_sector: dict[str, dict[str, list[float]]],
        stock_weights: dict[str, float] | None = None,
    ) -> float:
        """计算某行业内个股按权重加权的区间收益。

        Args:
            sector: 行业名称
            stock_returns_by_sector: {行业: {股票代码: [日收益率, ...]}}
            stock_weights: 可选，股票代码 -> 权重（行业内相对权重）；
                           为空时按等权处理（近似代表基准在该行业内的分散持仓）

        Returns:
            行业加权区间收益率
        """
        codes_ret = stock_returns_by_sector.get(sector, {})
        if not codes_ret:
            return 0.0

        if stock_weights is None:
            # 等权：先复合每只股票区间收益，再等权平均
            period_rets = [self._portfolio_total_return(r) for r in codes_ret.values()]
            return float(np.mean(period_rets)) if period_rets else 0.0

        total_w = sum(stock_weights.get(c, 0.0) for c in codes_ret)
        if total_w <= 0:
            period_rets = [self._portfolio_total_return(r) for r in codes_ret.values()]
            return float(np.mean(period_rets)) if period_rets else 0.0

        weighted = 0.0
        for code, rets in codes_ret.items():
            w = stock_weights.get(code, 0.0) / total_w
            weighted += w * self._portfolio_total_return(rets)
        return weighted

    def compute(
        self,
        portfolio_returns: pd.Series | list[float] | np.ndarray,
        benchmark_returns: pd.Series | list[float] | np.ndarray,
        sector_weights_pf: dict[str, float],
        sector_weights_bm: dict[str, float],
        stock_returns_by_sector: dict[str, dict[str, list[float]]],
        pf_stock_weights_by_sector: dict[str, dict[str, float]] | None = None,
        bm_stock_weights_by_sector: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """执行 Brinson-Fachler 归因分解。

        Args:
            portfolio_returns: 组合区间收益率序列（日频/周期），用于校验总收益
            benchmark_returns: 基准区间收益率序列，用于校验总收益及作为 R_bm
            sector_weights_pf: 组合行业权重 {行业: 权重}，权重和应为1
            sector_weights_bm: 基准行业权重 {行业: 权重}
            stock_returns_by_sector: 分行业个股收益率序列
                {行业: {股票代码: [收益率, ...]}}，用于计算行业内组合/基准收益
            pf_stock_weights_by_sector: 可选，组合在各行业内的个股相对权重
                {行业: {股票代码: 权重}}。缺省时行业内按等权近似（此时
                selection_effect 会退化为0，因为组合与基准的行业内收益
                无法区分——建议尽量提供此参数以获得有意义的选股效应）
            bm_stock_weights_by_sector: 可选，基准在各行业内的个股相对权重，
                结构同上；缺省时按等权近似

        Returns:
            dict:
                - total_active_return: 总主动收益
                - allocation_effect: 总配置效应
                - selection_effect: 总选择效应
                - interaction_effect: 总交互效应
                - sector_details: 各行业明细列表
        """
        try:
            R_bm = self._portfolio_total_return(benchmark_returns)
            R_pf = self._portfolio_total_return(portfolio_returns)

            all_sectors = sorted(set(sector_weights_pf) | set(sector_weights_bm))

            # 归一化权重，避免权重和不为1导致的偏差
            total_pf_w = sum(sector_weights_pf.values()) or 1.0
            total_bm_w = sum(sector_weights_bm.values()) or 1.0
            w_pf = {s: sector_weights_pf.get(s, 0.0) / total_pf_w for s in all_sectors}
            w_bm = {s: sector_weights_bm.get(s, 0.0) / total_bm_w for s in all_sectors}

            sector_details: list[dict[str, Any]] = []
            total_alloc = 0.0
            total_sel = 0.0
            total_inter = 0.0

            for sector in all_sectors:
                wp = w_pf.get(sector, 0.0)
                wb = w_bm.get(sector, 0.0)

                # 行业内收益：组合侧用组合个股权重加权，基准侧用基准个股权重加权；
                # 未提供个股权重时退化为等权近似
                pf_weights = (
                    pf_stock_weights_by_sector.get(sector) if pf_stock_weights_by_sector else None
                )
                bm_weights = (
                    bm_stock_weights_by_sector.get(sector) if bm_stock_weights_by_sector else None
                )
                R_pf_sector = self._sector_period_return(
                    sector, stock_returns_by_sector, stock_weights=pf_weights
                )
                R_bm_sector = self._sector_period_return(
                    sector, stock_returns_by_sector, stock_weights=bm_weights
                )

                alloc = (wp - wb) * (R_bm_sector - R_bm)
                sel = wb * (R_pf_sector - R_bm_sector)
                inter = (wp - wb) * (R_pf_sector - R_bm_sector)

                total_alloc += alloc
                total_sel += sel
                total_inter += inter

                sector_details.append({
                    "sector": sector,
                    "weight_pf": round(wp, 4),
                    "weight_bm": round(wb, 4),
                    "return_pf": round(R_pf_sector, 4),
                    "return_bm": round(R_bm_sector, 4),
                    "allocation_effect": round(alloc, 4),
                    "selection_effect": round(sel, 4),
                    "interaction_effect": round(inter, 4),
                })

            total_active = total_alloc + total_sel + total_inter

            # P2-Q24-fix (M279): 恒等式校验——总主动收益应等于 R_pf - R_bm。
            # 两条计算路径（整段收益序列复合 vs 行业个股加权聚合）口径不同，
            # 允许 1e-3 量级的复合/聚合误差，超差时显式告警并写入 notes。
            expected_active = R_pf - R_bm
            identity_gap = total_active - expected_active

            notes: list[str] = []
            if abs(identity_gap) > 1e-3:
                _msg = (
                    f"总主动收益({total_active:.4f})与 R_pf-R_bm({expected_active:.4f}) "
                    f"偏差 {identity_gap:.4f}，可能因行业收益/个股权重口径不一致"
                )
                notes.append(_msg)
                print(f"[brinson] 警告: {_msg}", file=sys.stderr)

            # P2-Q24-fix (M279): 未提供行业内个股相对权重时退化为等权，
            # 组合与基准的行业内收益无法区分 → selection_effect 会被系统性低估/失真。
            # 文档虽有说明，但仍需显式告警让调用方知情。
            if pf_stock_weights_by_sector is None:
                _msg = "未提供组合行业内个股权重(pf_stock_weights_by_sector)，行业内按等权近似，选股效应失真"
                notes.append(_msg)
                print(f"[brinson] 警告: {_msg}", file=sys.stderr)
            if bm_stock_weights_by_sector is None:
                _msg = "未提供基准行业内个股权重(bm_stock_weights_by_sector)，行业内按等权近似"
                notes.append(_msg)
                print(f"[brinson] 警告: {_msg}", file=sys.stderr)

            result = {
                "total_active_return": round(total_active, 4),
                "allocation_effect": round(total_alloc, 4),
                "selection_effect": round(total_sel, 4),
                "interaction_effect": round(total_inter, 4),
                "sector_details": sorted(
                    sector_details,
                    key=lambda d: abs(d["allocation_effect"]) + abs(d["selection_effect"]),
                    reverse=True,
                ),
                "portfolio_return": round(R_pf, 4),
                "benchmark_return": round(R_bm, 4),
                "identity_gap": round(identity_gap, 6),
                "expected_active_return": round(expected_active, 4),
                "notes": notes if notes else None,
            }
            self._last_result = result
            return result
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"Brinson归因计算失败: {exc}",
                "total_active_return": 0.0,
                "allocation_effect": 0.0,
                "selection_effect": 0.0,
                "interaction_effect": 0.0,
                "sector_details": [],
            }


def main() -> None:
    """使用模拟数据自测 BrinsonAttribution。"""
    try:
        rng = np.random.default_rng(42)

        sectors = ["食品饮料", "医药生物", "电子", "银行", "新能源"]
        sector_weights_pf = {
            "食品饮料": 0.25, "医药生物": 0.20, "电子": 0.20,
            "银行": 0.15, "新能源": 0.20,
        }
        sector_weights_bm = {
            "食品饮料": 0.15, "医药生物": 0.18, "电子": 0.22,
            "银行": 0.30, "新能源": 0.15,
        }

        stock_returns_by_sector: dict[str, dict[str, list[float]]] = {}
        base_drift = {
            "食品饮料": 0.0008, "医药生物": 0.0003, "电子": 0.0002,
            "银行": 0.0001, "新能源": -0.0002,
        }
        pf_stock_weights_by_sector: dict[str, dict[str, float]] = {}
        bm_stock_weights_by_sector: dict[str, dict[str, float]] = {}
        for sector in sectors:
            n_stocks = 3
            stock_returns_by_sector[sector] = {
                f"{sector}_{i}": (
                    base_drift[sector] + rng.normal(0, 0.015, 120)
                ).tolist()
                for i in range(n_stocks)
            }
            # 组合超配第一只个股（模拟选股偏好），基准等权持有
            pf_stock_weights_by_sector[sector] = {
                f"{sector}_0": 0.6, f"{sector}_1": 0.25, f"{sector}_2": 0.15,
            }
            bm_stock_weights_by_sector[sector] = {
                f"{sector}_0": 1 / 3, f"{sector}_1": 1 / 3, f"{sector}_2": 1 / 3,
            }

        portfolio_returns = (0.0004 + rng.normal(0, 0.01, 120)).tolist()
        benchmark_returns = (0.0002 + rng.normal(0, 0.009, 120)).tolist()

        engine = BrinsonAttribution()
        result = engine.compute(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            sector_weights_pf=sector_weights_pf,
            sector_weights_bm=sector_weights_bm,
            stock_returns_by_sector=stock_returns_by_sector,
            pf_stock_weights_by_sector=pf_stock_weights_by_sector,
            bm_stock_weights_by_sector=bm_stock_weights_by_sector,
        )

        print("=== Brinson 归因结果 ===")
        print(f"总主动收益: {result['total_active_return']:.4f}")
        print(f"配置效应: {result['allocation_effect']:.4f}")
        print(f"选择效应: {result['selection_effect']:.4f}")
        print(f"交互效应: {result['interaction_effect']:.4f}")
        print("行业明细:")
        for d in result["sector_details"]:
            print(
                f"  {d['sector']}: alloc={d['allocation_effect']:.4f} "
                f"sel={d['selection_effect']:.4f}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 测试运行失败: {exc}")


if __name__ == "__main__":
    main()
