"""
rolling_attribution — 滚动窗口 Brinson 归因 (V5)

核心问题：Brinson 归因结果是否随时间稳定，还是集中在某几个月爆发？
用滚动窗口反复计算 Brinson 归因，观察配置/选择效应随时间的演变。

对标：机构归因报告中的"月度/滚动归因走势图"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .brinson_attribution import BrinsonAttribution

BASE_DIR = Path(__file__).resolve().parent.parent


class RollingAttribution:
    """滚动窗口归因引擎。

    在给定窗口长度下，逐窗口滑动计算 Brinson 归因，输出归因结果的时间序列。

    核心假设：
      单次归因容易被短期极端行情主导，滚动窗口能揭示归因结果的持续性/稳定性。
    """

    def __init__(self, brinson_engine: BrinsonAttribution | None = None) -> None:
        self.brinson = brinson_engine or BrinsonAttribution()

    @staticmethod
    def _sector_weights_for_window(
        sector_map: dict[str, str],
        weights_df_row: pd.Series,
    ) -> dict[str, float]:
        """将个股权重按行业映射汇总为行业权重。

        Args:
            sector_map: {股票代码: 行业名称}
            weights_df_row: 该窗口的个股权重（Series，index=代码），
                由 _window_avg_weights 计算期均得到

        Returns:
            {行业: 汇总权重}
        """
        sector_w: dict[str, float] = {}
        for code, w in weights_df_row.items():
            sector = sector_map.get(code, "其他")
            sector_w[sector] = sector_w.get(sector, 0.0) + float(w)
        return sector_w

    @staticmethod
    def _window_avg_weights(win: pd.DataFrame) -> dict[str, float]:
        """计算窗口内个股权重的期均权重。

        P1-Q24-fix (H10): 原实现用窗口末行权重代表整个窗口，存在时点错配/
        前视（期末才知道的权重被用来解释窗口内全部收益）。改为对窗口内逐日
        权重取平均，权重剧烈变动时归因更稳健。

        Returns:
            {股票代码: 窗口期均权重}；窗口内无任何权重明细时返回空 dict。
        """
        acc: dict[str, float] = {}
        count = 0
        for _, row in win.iterrows():
            w = row.get("weights", {}) or {}
            if not isinstance(w, dict) or not w:
                continue
            for code, val in w.items():
                acc[code] = acc.get(code, 0.0) + float(val)
            count += 1
        if count == 0:
            return {}
        return {code: v / count for code, v in acc.items()}

    def compute(
        self,
        portfolio_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        window: int = 30,
        sector_map: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """滚动计算 Brinson 归因。

        Args:
            portfolio_df: 组合数据，需含列：
                - 'date': 日期
                - 'return': 组合当日收益率
                - 个股权重列或 'weights' 列（dict 每行为 {代码: 权重}）
                - 'stock_returns' 列（dict 每行为 {代码: 当日收益率}），
                  用于聚合行业内个股收益
            benchmark_df: 基准数据，需含列 'date', 'return'
            window: 滚动窗口长度（期数）
            sector_map: {股票代码: 行业名称}，用于将个股权重汇总为行业权重

        Returns:
            list[dict]: 每个窗口的 Brinson 归因结果，附加 'window_start' 和
                        'window_end' 字段
        """
        try:
            sector_map = sector_map or {}
            if "date" not in portfolio_df.columns or "return" not in portfolio_df.columns:
                return []
            if "date" not in benchmark_df.columns or "return" not in benchmark_df.columns:
                return []

            pf = portfolio_df.reset_index(drop=True)
            bm = benchmark_df.reset_index(drop=True)
            n = min(len(pf), len(bm))
            if n < window:
                return []

            results: list[dict[str, Any]] = []

            for end in range(window, n + 1):
                start = end - window
                pf_win = pf.iloc[start:end]
                bm_win = bm.iloc[start:end]

                pf_returns = pf_win["return"].astype(float).tolist()
                bm_returns = bm_win["return"].astype(float).tolist()

                # P1-Q24-fix (H10): 改用窗口期均权重（_window_avg_weights），
                # 不再用窗口末期权重解释窗口内全部收益，避免时点错配/前视。
                weights_pf_stock = self._window_avg_weights(pf_win)
                weights_bm_stock = self._window_avg_weights(bm_win)

                sector_weights_pf = (
                    self._sector_weights_for_window(sector_map, pd.Series(weights_pf_stock))
                    if weights_pf_stock else {}
                )
                sector_weights_bm = (
                    self._sector_weights_for_window(sector_map, pd.Series(weights_bm_stock))
                    if weights_bm_stock else {}
                )

                if not sector_weights_pf or not sector_weights_bm:
                    # 无个股权重明细时跳过该窗口
                    continue

                # 聚合窗口内的个股收益，按行业分组
                stock_returns_by_sector: dict[str, dict[str, list[float]]] = {}
                for _, row in pf_win.iterrows():
                    stock_rets = row.get("stock_returns", {}) or {}
                    if not isinstance(stock_rets, dict):
                        continue
                    for code, ret in stock_rets.items():
                        sector = sector_map.get(code, "其他")
                        stock_returns_by_sector.setdefault(sector, {}).setdefault(code, []).append(
                            float(ret)
                        )

                # 行业内个股相对权重（用于区分组合/基准在同行业内的不同持股结构，
                # 从而产生有意义的 selection_effect）
                pf_stock_weights_by_sector: dict[str, dict[str, float]] = {}
                for code, w in weights_pf_stock.items():
                    sector = sector_map.get(code, "其他")
                    pf_stock_weights_by_sector.setdefault(sector, {})[code] = float(w)

                bm_stock_weights_by_sector: dict[str, dict[str, float]] = {}
                for code, w in weights_bm_stock.items():
                    sector = sector_map.get(code, "其他")
                    bm_stock_weights_by_sector.setdefault(sector, {})[code] = float(w)

                brinson_result = self.brinson.compute(
                    portfolio_returns=pf_returns,
                    benchmark_returns=bm_returns,
                    sector_weights_pf=sector_weights_pf,
                    sector_weights_bm=sector_weights_bm,
                    stock_returns_by_sector=stock_returns_by_sector,
                    pf_stock_weights_by_sector=pf_stock_weights_by_sector,
                    bm_stock_weights_by_sector=bm_stock_weights_by_sector,
                )

                brinson_result["window_start"] = str(pf_win.iloc[0]["date"])
                brinson_result["window_end"] = str(pf_win.iloc[-1]["date"])
                results.append(brinson_result)

            return results
        except Exception as exc:  # noqa: BLE001
            return [{"error": f"滚动归因计算失败: {exc}"}]


def main() -> None:
    """使用模拟数据自测 RollingAttribution。"""
    try:
        rng = np.random.default_rng(11)
        n_days = 90
        dates = pd.date_range("2026-01-01", periods=n_days, freq="D")

        codes = [f"60000{i}" for i in range(5)]
        sector_map = {
            "600000": "银行", "600001": "食品饮料", "600002": "电子",
            "600003": "医药生物", "600004": "新能源",
        }

        # 组合超配前2只股票，基准等权持有，以便产生非零的选择/配置效应
        weights = {"600000": 0.10, "600001": 0.35, "600002": 0.30, "600003": 0.15, "600004": 0.10}
        bm_weights = {c: round(1.0 / len(codes), 4) for c in codes}

        pf_rows = []
        bm_rows = []
        for d in dates:
            stock_rets = {c: float(rng.normal(0.0005, 0.02)) for c in codes}
            pf_ret = sum(weights[c] * stock_rets[c] for c in codes)
            bm_ret = sum(bm_weights[c] * stock_rets[c] for c in codes)
            pf_rows.append({
                "date": d, "return": pf_ret,
                "weights": dict(weights), "stock_returns": stock_rets,
            })
            bm_rows.append({
                "date": d, "return": bm_ret,
                "weights": dict(bm_weights), "stock_returns": stock_rets,
            })

        portfolio_df = pd.DataFrame(pf_rows)
        benchmark_df = pd.DataFrame(bm_rows)

        engine = RollingAttribution()
        results = engine.compute(portfolio_df, benchmark_df, window=30, sector_map=sector_map)

        print(f"=== 滚动归因结果 (共{len(results)}个窗口) ===")
        for r in results[:3]:
            print(
                f"  {r.get('window_start')} ~ {r.get('window_end')}: "
                f"active={r.get('total_active_return')}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 测试运行失败: {exc}")


if __name__ == "__main__":
    main()
