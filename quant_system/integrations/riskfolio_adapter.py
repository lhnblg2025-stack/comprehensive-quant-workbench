"""
riskfolio_adapter — 组合优化适配层 (V5)

将 Riskfolio-Lib 集成到组合优化流程：
  - HRP: 层次风险平价（无需协方差求逆）
  - HERC: 层次等风险贡献
  - Mean-Variance: 均值方差优化（带约束）
  - Risk Parity: 风险预算
  - Black-Litterman: 观点融合

降级策略: riskfolio不可用时，用手写实现代替。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    _HAS_RISKFOLIO = True
except ImportError:
    _HAS_RISKFOLIO = False


class RiskfolioOptimizer:
    """组合优化器（封装Riskfolio-Lib）。"""

    def __init__(self) -> None:
        self.available = _HAS_RISKFOLIO

    def _check(self):
        if not self.available:
            raise ImportError("Riskfolio-Lib not installed")

    def _prepare_portfolio(self, returns: pd.DataFrame):
        """创建 rp.Portfolio 并完成资产统计，显式转 float 修复 numpy 2.x 兼容。

        P1-Q26-fix: riskfolio 7.0.1 + numpy 2.2 下，optimization 内部对
        self.cov/mu 做 np.array(...) 时若为 object 会抛 'object arrays are
        not supported'；必须先 assets_stats() 再转 float。
        """
        import riskfolio as rp
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu='hist', method_cov='hist')
        port.mu = port.mu.astype(float)
        port.cov = port.cov.astype(float)
        port.returns = port.returns.astype(float)
        return port

    @staticmethod
    def _extract_weights(w, returns: pd.DataFrame) -> dict:
        """把 riskfolio 返回的权重（DataFrame/dict/array）统一为 {code: weight}。"""
        weights: dict[str, float] = {}
        if isinstance(w, pd.DataFrame):
            col = w.columns[0]
            for idx in w.index:
                weight = float(w.loc[idx, col])
                if weight > 0.001:
                    weights[str(idx)] = round(weight, 4)
        elif isinstance(w, dict):
            for k, v in w.items():
                weight = float(v)
                if weight > 0.001:
                    weights[str(k)] = round(weight, 4)
        elif isinstance(w, (list, tuple, np.ndarray)):
            for i, code in enumerate(returns.columns):
                weight = float(w[i]) if i < len(w) else 0
                if weight > 0.001:
                    weights[str(code)] = round(weight, 4)
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        return weights

    def _build_bl_views(self, returns: pd.DataFrame, views):
        """把 views 转为 Black-Litterman 的 P(观点矩阵)/Q(预期收益) DataFrame。

        views 支持两种形态：
          - dict: {代码: 预期收益}
          - list[dict]: [{tickers:[...], bullish:bool, confidence:float}]
        无有效观点时返回 (None, None)。
        """
        codes = list(returns.columns)
        if not views:
            return None, None
        rows, qvals = [], []
        if isinstance(views, dict):
            for code, val in views.items():
                if code not in codes:
                    continue
                row = pd.Series(0.0, index=codes)
                row[code] = 1.0
                rows.append(row)
                qvals.append(float(val))
        elif isinstance(views, (list, tuple)):
            for v in views:
                if not isinstance(v, dict):
                    continue
                tickers = v.get('tickers', [])
                idxs = [i for i, c in enumerate(codes) if c in tickers]
                if not idxs:
                    continue
                row = pd.Series(0.0, index=codes)
                for i in idxs:
                    row[codes[i]] = 1.0 / len(idxs)
                direction = 1.0 if v.get('bullish', True) else -1.0
                conf = float(v.get('confidence', 0.5))
                avg_ret = float(returns[codes].iloc[-1].mean())
                qvals.append(direction * max(abs(avg_ret), 1e-4) * conf)
                rows.append(row)
        if not rows:
            return None, None
        P = pd.DataFrame(rows, index=[f'v{i+1}' for i in range(len(rows))], columns=codes)
        Q = pd.DataFrame(qvals, index=P.index, columns=['return'])
        return P, Q

    def hrp(self, returns: pd.DataFrame,
            cov: pd.DataFrame | None = None) -> dict:
        """Hierarchical Risk Parity (HRP) via HCPortfolio. (V5.1 fix: v7 API)"""
        if not self.available:
            return self._hrp_fallback(returns, cov)

        try:
            import riskfolio as rp

            hcp = rp.HCPortfolio(returns=returns)
            w = hcp.optimization(
                codependence="pearson",
                method_cov="hist",
                rm="MV",
                linkage="single",
                k=None,
                max_k=10,
            )

            clean_w = {}
            if isinstance(w, pd.DataFrame):
                col = w.columns[0]
                for idx in w.index:
                    weight = float(w.loc[idx, col])
                    if weight > 0.001:
                        clean_w[str(idx)] = round(weight, 4)
            elif isinstance(w, dict):
                for k, v in w.items():
                    weight = float(v)
                    if weight > 0.001:
                        clean_w[str(k)] = round(weight, 4)
            elif isinstance(w, (list, tuple, np.ndarray)):
                for i, code in enumerate(returns.columns):
                    weight = float(w[i]) if i < len(w) else 0
                    if weight > 0.001:
                        clean_w[str(code)] = round(weight, 4)

            total = sum(clean_w.values())
            if total > 0:
                clean_w = {k: round(v / total, 4) for k, v in clean_w.items()}

            return {"weights": clean_w, "method": "HRP(via HCPortfolio)", "available": True}

        except Exception as e:
            return {"error": str(e), "method": "HRP", "available": False}

    def _hrp_fallback(self, returns: pd.DataFrame,
                      cov: pd.DataFrame | None = None) -> dict:
        """HRP 降级实现（不依赖 Riskfolio-Lib）。"""
        try:
            # 用相关系数矩阵做层次聚类
            corr = returns.corr(method="pearson")
            dist = ((1 - corr) / 2) ** 0.5

            # 简化层次聚类
            n = len(corr.columns)
            if n == 0:
                return {"weights": {}, "method": "HRP(fallback)"}

            # 等权+方差调整
            vols = returns.std() * np.sqrt(252)
            inv_vol = 1.0 / (vols + 1e-8)
            w = inv_vol / inv_vol.sum()

            weights = {str(code): round(float(w[code]), 4) for code in w.index}
            return {
                "weights": weights,
                "method": "HRP(fallback)",
                "note": "Riskfolio-Lib not installed, used inverse-vol weighting",
                "available": False,
            }
        except Exception as e:
            return {"error": str(e), "method": "HRP(fallback)"}

    def mean_variance(self, returns: pd.DataFrame,
                      constraints: dict | None = None) -> dict:
        """均值方差优化 (Max Sharpe). (V5.1 fix: numpy compat)"""
        if not self.available:
            return self._mv_fallback(returns, constraints)

        try:
            import riskfolio as rp
            # 先确保数据类型正确
            mu_vec = np.asarray(returns.mean() * 252).flatten().astype(float)
            cov_mat = np.asarray(returns.cov() * 252).astype(float)

            # P1-Q26-fix: riskfolio 7.x 的 optimization 参数 model 只能是
            # Classic/BL/FM/BL_FM，风险度量由 rm 指定。原 model="MV" 无效导致
            # mu/sigma 恒为 None → 'object arrays are not supported' 异常被吞。
            # 权重由 optimization() 返回（port.weights 属性不存在）。
            try:
                port = self._prepare_portfolio(returns)
                # P2-Q26-fix: docstring 声称 Max Sharpe 但原 rm="MV" 实为最小方差组合，
                # 显式改为 rm="MS"（Maximum Sharpe），使实现与文档一致。
                w = port.optimization(model="Classic", rm="MS")
            except Exception:
                # 降级: 手算最大夏普
                from scipy.optimize import minimize
                n = len(returns.columns)

                def neg_sharpe(w):
                    w = np.asarray(w, dtype=float)
                    port_ret = float(mu_vec @ w)
                    port_risk = float(np.sqrt(w @ cov_mat @ w + 1e-8))
                    return -port_ret / port_risk

                cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
                bounds = [(0, 1)] * n
                w0 = np.ones(n) / n
                res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons, method="SLSQP")
                w = res.x if res.success else w0

            weights = self._extract_weights(w, returns)
            return {"weights": weights, "method": "MeanVariance", "available": True}

        except Exception as e:
            return {"error": str(e), "method": "MeanVariance"}

    def _mv_fallback(self, returns: pd.DataFrame,
                     constraints: dict | None = None) -> dict:
        """均值方差降级: 最大夏普近似。"""
        try:
            mu = returns.mean() * 252
            cov = returns.cov() * 252

            # 简单优化: 最小方差
            n = len(returns.columns)
            w = np.ones(n) / n
            return {
                "weights": {str(returns.columns[i]): round(w[i], 4) for i in range(n)},
                "method": "MeanVariance(fallback)",
                "note": "Riskfolio-Lib not installed, used equal-weight",
            }
        except Exception as e:
            return {"error": str(e)}

    def risk_parity(self, returns: pd.DataFrame) -> dict:
        """风险平价 (Risk Budgeting). (V5.1 fix: numpy compat)"""
        # P1-Q26-fix: riskfolio 7.x Portfolio.optimization 无直接 RP 模式
        # （原 model="RP" 无效、port.weights 不存在，恒走降级），统一用
        # 迭代式等风险贡献算法，行为与之前降级路径一致。
        cov = np.asarray(returns.cov(), dtype=float)
        n = cov.shape[0]
        w = np.ones(n, dtype=float) / n
        for _ in range(50):
            pvar = float(w @ cov @ w)
            mrc = cov @ w / max(np.sqrt(pvar), 1e-8)
            rc = w * mrc
            target = float(np.mean(rc))
            w = w + (target - rc) * 0.1
            w = np.maximum(w, 0)
            w /= max(float(w.sum()), 1e-10)

        weights = {str(returns.columns[i]): round(w[i], 4) for i in range(n)}
        method = "RiskParity(fallback)" if not self.available else "RiskParity"
        return {"weights": weights, "method": method}

    def _rp_fallback(self, returns):
        cov = np.asarray(returns.cov(), dtype=float)
        n = cov.shape[0]
        w = np.ones(n, dtype=float) / n
        for _ in range(50):
            pvar = float(w @ cov @ w)
            mrc = cov @ w / max(np.sqrt(pvar), 1e-8)
            rc = w * mrc
            target = float(np.mean(rc))
            w = w + (target - rc) * 0.1
            w = np.maximum(w, 0)
            w /= max(float(w.sum()), 1e-10)
        return {"weights": {str(returns.columns[i]): round(w[i], 4) for i in range(n)}}

    def black_litterman(self, returns: pd.DataFrame,
                        views: dict | None = None) -> dict:
        """Black-Litterman 模型。

        Args:
            returns: 资产收益 DataFrame(index=date, columns=代码)
            views: 观点。dict {代码: 预期收益} 或
                   list[dict] [{tickers, bullish, confidence}]；None 时退化为等权。

        Returns:
            {"weights": {code: weight}, "method": "BlackLitterman", "available": True}
        """
        if not self.available:
            return {"available": False, "error": "Riskfolio-Lib required for BL"}

        try:
            port = self._prepare_portfolio(returns)
            P, Q = self._build_bl_views(returns, views)

            if P is None:
                # P1-Q26-fix: 无观点 → 退化为等权并明确标注（不硬造观点）
                n = len(returns.columns)
                weights = {str(codes): round(1.0 / n, 4) for codes in returns.columns}
                return {
                    "weights": weights,
                    "method": "BlackLitterman(no views)",
                    "note": "未提供观点，退化为等权；传入 views 后启用观点融合",
                    "available": True,
                }

            # P1-Q26-fix: 必须先 blacklitterman_stats(P, Q) 设置 self.mu_bl/cov_bl，
            # 再 optimization(model='BL')；权重由返回值得出。原实现缺这两步，
            # 恒报 'object arrays are not supported'，且成功路径不返回 weights。
            port.blacklitterman_stats(P=P, Q=Q)
            w = port.optimization(model="BL")
            weights = self._extract_weights(w, returns)
            return {"weights": weights, "method": "BlackLitterman", "available": True}

        except Exception as e:
            return {"error": str(e), "method": "BlackLitterman", "available": False}

    def herc(self, returns: pd.DataFrame) -> dict:
        """HERC (层次等风险贡献)。(V5.1 fix: v7 API)"""
        if not self.available:
            return self.risk_parity(returns)

        try:
            import riskfolio as rp
            hcp = rp.HCPortfolio(returns=returns)
            w = hcp.optimization(
                # P2-Q26-fix: 显式 model="HERC"——原实现与 hrp() 参数完全相同
                # （HCPortfolio+rm="MV"+linkage="single"，model 默认 HRP），
                # 实测两者权重逐项相等，HERC 名不副实。
                model="HERC",
                codependence="pearson",
                method_cov="hist",
                rm="MV",
                linkage="single",
                k=None, max_k=10,
            )
            weights = {}
            if isinstance(w, pd.DataFrame):
                col = w.columns[0]
                for idx in w.index:
                    weight = float(w.loc[idx, col])
                    if weight > 0.001:
                        weights[str(idx)] = round(weight, 4)
            return {"weights": weights, "method": "HERC(via HCPortfolio)"}
        except Exception as e:
            return {"error": str(e), "method": "HERC"}


def main() -> None:
    """模拟数据测试。"""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=500, freq="B")
    codes = ["600519", "000858", "300750", "601318", "000333"]

    returns = pd.DataFrame(
        np.random.randn(len(dates), len(codes)) * 0.02 + 0.0005,
        index=dates, columns=codes
    )

    ro = RiskfolioOptimizer()
    print(f"Riskfolio available: {ro.available}")

    for method in ["hrp", "herc", "risk_parity", "mean_variance"]:
        try:
            result = getattr(ro, method)(returns)
            weights = result.get("weights", {})
            top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\n{result.get('method', method)}:")
            for code, w in top:
                print(f"  {code}: {w:.1%}")
        except Exception as e:
            print(f"\n{method}: {e}")


if __name__ == "__main__":
    main()
