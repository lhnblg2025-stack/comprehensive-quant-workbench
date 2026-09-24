"""
barra_risk.py — Barra CNE6 风格多因子风险模型
V4.1 feature

实现行业-风格因子协方差矩阵分解，提供风险归因能力：
1. 行业因子暴露（申万一级行业）
2. 风格因子暴露（Barra CNE6 标准风格因子）
3. 因子协方差矩阵估计
4. 风险归因：总风险 → 行业风险 + 风格风险 + 特异风险

D6收敛登记 (2026-08-11): 风控域收敛（保守策略）——Barra CNE6 多因子风险为独立能力：
  BarraExposure(行业+风格暴露)/FactorCovariance(EWMA+Newey-West+RMT去噪)/
  RiskDecomposition(风险归因)/BarraModel(特异风险) 无等价实现，不强迁；
  唯一引用方为 tests/integration_tests.py（测试），无生产引用。
"""

import numpy as np
import pandas as pd
from typing import Optional

logger = __import__('logging').getLogger(__name__)


# ══════════════════════════════════════
# 风格因子定义
# ══════════════════════════════════════

# P2-Q6-fix (Q6-L475): 本表为 **CNE6 风格近似**，非 Barra 标准 10 风格：
#   - 混入 dividend_yield / short_term_rev / sentiment（标准 CNE6 无此三因子）；
#   - volatility 用原始 60d 波动率而非残差波动率；
#   - momentum 用 240d-20d 收益近似（标准 CNE6 为剔除最近 1 个月的 12 个月动量）。
#   保留全部 13 列以兼容既有调用方；如需严格 CNE6，请按上述标准替换/补充。
STYLE_FACTORS = [
    "beta",           # 市场 Beta（60日回归）
    "momentum",       # 动量（12个月剔除最近1个月）
    "size",           # 规模（对数流通市值）
    "earning_yield",  # 盈利收益率（E/P、PEG）
    "volatility",     # 波动率（60日收益标准差）
    "value",          # 价值（B/P、C/P、S/P）
    "growth",         # 增长（营收增长、利润增长）
    "leverage",       # 杠杆（D/E、资产负债率）
    "liquidity",      # 流动性（换手率）
    "dividend_yield", # 股息率
    "short_term_rev", # 短期反转（最近5日收益）
    "non_linear_size",# 非线性规模（Size^3）
    "sentiment",      # 情绪因子
]


# ══════════════════════════════════════
# 工具函数
# ══════════════════════════════════════

def _winsorize_zscores(z: np.ndarray, limits: float = 5.0) -> np.ndarray:
    """将 Z-score 限制在 [-limits, limits] 内"""
    return np.clip(z, -limits, limits)


def _mad_filter(x: np.ndarray, n_mad: float = 5.0) -> np.ndarray:
    """基于中位数绝对偏差的异常值过滤（NaN 感知）。

    P2-Q6-fix (Q6-M470): 原实现对含 NaN 的列 np.clip(NaN 边界) 会把整列清成 NaN；
    现只对有限值截断，NaN 保留原样交由 standardize_exposures 统一 fillna(0)。
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return x
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad < 1e-12:
        return x
    lo, hi = median - n_mad * mad, median + n_mad * mad
    clipped = x.copy()
    clipped[finite] = np.clip(clipped[finite], lo, hi)
    return clipped


# ══════════════════════════════════════
# 因子暴露计算
# ══════════════════════════════════════

class BarraExposure:
    """Barra 因子暴露计算器。

    根据日线数据计算每只股票的各风格因子暴露值。
    """

    def __init__(self, industry_type: str = "sw"):
        """
        Parameters
        ----------
        industry_type : str
            "sw" — 申万行业（SW 一级）
            "zjh" — 证监会行业
        """
        self.industry_type = industry_type

    def compute(self, df: pd.DataFrame, market_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """计算截面因子暴露。

        Parameters
        ----------
        df : pd.DataFrame
            包含以下列的截面数据：
            symbol, close, volume, amount, high, low, returns, pct_chg,
            industry, total_mv, float_mv, pe, pb, turnover,
            returns_60d, returns_240d, returns_5d
        market_returns : pd.Series, optional
            市场指数收益率系列（用于 Beta 计算）

        Returns
        -------
        pd.DataFrame
            (symbol x style_factors + industry_dummies) 的因子暴露矩阵
        """
        records = []

        for _, row in df.iterrows():
            exp = {"symbol": row.get("symbol")}
            # P2-Q6-fix (Q6-L474): industry_type 参数此前声明但从未使用（行业一律
            #   row.get("industry")）；现按语义选择行业列：zjh 优先 industry_zjh，
            #   sw 优先 industry_sw，均缺省时回退 industry
            if self.industry_type == "zjh":
                industry = row.get("industry_zjh", row.get("industry", "未知"))
            elif self.industry_type == "sw":
                industry = row.get("industry_sw", row.get("industry", "未知"))
            else:
                industry = row.get("industry", "未知")

            # 行业暴露（one-hot）
            exp["industry"] = industry

            # 风格因子暴露
            # P2-Q6-fix (Q6-L474): market_returns 参数保留兼容，但本模块仅持截面 df、
            #   单行无时间序列无法重算 beta；使用预处理好的 beta_60d，缺失回退 returns_60d
            beta_val = row.get("beta_60d", row.get("returns_60d", 0))
            exp["beta"] = beta_val if (beta_val is not None and np.isfinite(beta_val)) else np.nan
            exp["momentum"] = row.get("returns_240d", 0) - row.get("returns_20d", 0)
            # P2-Q6-fix (Q6-M463): float_mv 为 NaN 时 max(NaN,1e6) 返回 NaN（实测）会传播
            #   到整个风险分解；显式 NaN 处理，缺失市值暴露置 NaN 由 standardize 统一 fillna(0)
            mv = row.get("float_mv", None)
            if mv is None or not np.isfinite(mv):
                mv = row.get("total_mv", None)
            exp["size"] = np.log(mv) if (mv is not None and np.isfinite(mv) and mv > 1e6) else np.nan
            # P2-Q6-fix (Q6-M464): 亏损股（pe<0）时 1/max(pe,1)=1.0 被标成最佳盈利收益率，
            #   方向性失真；改为 pe<=0 置 NaN（缺失不参与暴露，标准化时按 0 处理）
            pe_val = row.get("pe", np.nan)
            exp["earning_yield"] = (1.0 / pe_val) if (pe_val is not None and np.isfinite(pe_val) and pe_val > 0) else np.nan

            # P2-Q6-fix (Q6-M466): 缺失 volatility_60d 时原实现回退到当日收益 returns
            #   （~0.01），与 60 日波动（~0.3）量纲不同 → 暴露尺度混乱；改为缺失置 NaN
            vol = row.get("volatility_60d", np.nan)
            exp["volatility"] = vol if (vol is not None and np.isfinite(vol) and abs(vol) > 1e-8) else np.nan

            # P2-Q6-fix (Q6-M465): 原 value=pb/pe=E/B 非 Barra 标准价值因子，且负 PE 时
            #   退化为 pb（被高估）；按 CNE6 用 B/P=1/pb，pb<=0/缺失置 NaN
            pb_val = row.get("pb", np.nan)
            exp["value"] = (1.0 / pb_val) if (pb_val is not None and np.isfinite(pb_val) and pb_val > 0) else np.nan
            exp["growth"] = row.get("revenue_growth", 0)
            exp["leverage"] = row.get("debt_ratio", 0.5)
            turnover = row.get("turnover", 0.01)
            exp["liquidity"] = np.log(max(turnover, 0.001))
            exp["dividend_yield"] = row.get("dividend_yield", 0)
            exp["short_term_rev"] = row.get("returns_5d", 0)
            # V4.1 fix: non_linear_size 原为 size ** 3，对数市值（~15-25）的三次方
            # 会产生天文数字（3375~15625），使该因子主导所有其他因子。
            # 改为 np.sign(size) * abs(size)**2 / 100 做标准化，
            # 使非线性规模因子与 size 因子量级相当。
            exp["non_linear_size"] = np.sign(exp["size"]) * (exp["size"] ** 2) / 100.0
            exp["sentiment"] = row.get("sentiment", 0)

            records.append(exp)

        exposures = pd.DataFrame(records)
        return exposures

    def standardize_exposures(self, exposures: pd.DataFrame) -> pd.DataFrame:
        """标准化因子暴露：Z-score + MAD 过滤 + 市值加权归一化

        P2-Q6-fix (Q6-M470): 标准化后对缺失暴露 fillna(0)（视为中性），并记录缺失率；
        原实现只 winsorize 不填充，NaN 保留进最终暴露矩阵（与 size 的 NaN 联动）。
        """
        result = exposures.copy()
        numeric_cols = [c for c in STYLE_FACTORS if c in result.columns]

        missing_rates: dict[str, float] = {}
        for col in numeric_cols:
            values = result[col].values.copy()
            n_missing = int(np.isnan(values).sum())
            missing_rates[col] = n_missing / max(len(values), 1)
            # MAD 过滤（NaN 感知，见 _mad_filter）
            values = _mad_filter(values)
            # 全 NaN 列：直接置 0（中性），避免 nanmean/nanstd 空切片告警
            if not np.isfinite(values).any():
                result[col] = 0.0
                continue
            # Z-score
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 1e-12:
                result[col] = (values - mean) / std
            else:
                result[col] = 0.0
            # 截断
            result[col] = _winsorize_zscores(result[col].values)
            # P2-Q6-fix (Q6-M470): 缺失暴露标准化后 fillna(0)（中性），不再泄漏 NaN
            result[col] = result[col].fillna(0.0)

        high_missing = {c: round(r, 4) for c, r in missing_rates.items() if r > 0.3}
        if high_missing:
            logger.warning(
                "[BarraExposure] 因子缺失率 >30%%: %s（缺失暴露已按 0 处理）", high_missing
            )
        return result


# ══════════════════════════════════════
# 因子协方差矩阵估计
# ══════════════════════════════════════

class FactorCovariance:
    """因子协方差矩阵（含行业哑变量）。

    使用指数加权移动平均（EWMA）或简单滚动窗口估计。
    支持 Newey-West 调整和特征值去噪。
    """

    def __init__(self, halflife: int = 60, newey_lags: int = 2,
                 eig_cutoff: float = 0.95):
        """
        Parameters
        ----------
        halflife : int
            EWMA 半衰期（交易日），默认 60
        newey_lags : int
            Newey-West 自相关修正滞后阶数，默认 2
        eig_cutoff : float
            特征值去噪：保留前 k 个特征值占比达到 cutoff，默认 0.95
        """
        self.halflife = halflife
        self.newey_lags = newey_lags
        self.eig_cutoff = eig_cutoff

    def estimate(self, factor_returns: pd.DataFrame) -> np.ndarray:
        """估计因子协方差矩阵。

        Parameters
        ----------
        factor_returns : pd.DataFrame
            (date x factor) 格式的因子收益序列

        Returns
        -------
        np.ndarray
            (n_factors x n_factors) 协方差矩阵
        """
        # EWMA 协方差
        T, N = factor_returns.shape
        if T < 2 or N == 0:
            logger.warning(
                "FactorCovariance.estimate: 样本不足（T=%d, N=%d），返回零协方差", T, N
            )
            return np.zeros((N, N))
        decay = 2 ** (-1 / self.halflife)
        weights = np.array([decay ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()

        mean = np.average(factor_returns.values, axis=0, weights=weights)
        demeaned = factor_returns.values - mean
        weighted = demeaned * np.sqrt(weights[:, np.newaxis])
        cov = weighted.T @ weighted

        # P2-Q6-fix (Q6-M467): NW 滞后自协方差项也用相同 EWMA 权重（滞后对取较晚一期的
        #   权重），避免混合「非加权」自协方差与「EWMA 加权」协方差两种估计器；
        #   缩放因子 T/(T-1) 只在最后统一应用一次（原实现 EWMA 部分被缩放两次、
        #   NW 部分仅一次）。
        if self.newey_lags > 0:
            for lag in range(1, self.newey_lags + 1):
                w_lag = 1 - lag / (self.newey_lags + 1)
                auto_cov = (demeaned[lag:] * weights[lag:, np.newaxis]).T @ demeaned[:-lag]
                cov += w_lag * (auto_cov + auto_cov.T)

        cov *= T / max(T - 1, 1)

        # 特征值去噪（RMT 去噪）
        cov = self._denoise_eigenvalues(cov, T, N)

        return cov

    def _denoise_eigenvalues(self, cov: np.ndarray, T: int, N: int) -> np.ndarray:
        """特征值去噪：保留主导特征值，其余压缩到噪声区均值。

        P1-Q6-fix:
        - 原实现用**全体特征值均值**替换噪声特征值（被最大特征值主导），
          把约一半噪声方向方差抬高到均值之上，噪声被大幅放大；
          现改为用阈值以下特征值自身的均值替换（无噪声时置 0）。
        - q 原硬编码 2.0，现用实际 T/N 比（Marchenko-Pastur）。
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        n = len(eigenvalues)
        if n == 0:
            return cov

        # 计算 Marchenko-Pastur 阈值
        q = T / max(N, 1)  # 实际 T/N 比
        q = max(q, 1e-6)
        sigma2 = np.median(eigenvalues) / (1 + np.sqrt(1 / q)) ** 2
        threshold = sigma2 * (1 + np.sqrt(1 / q)) ** 2

        # 压缩噪声特征值：用噪声特征值自身均值替换（而非全体均值）
        noise = eigenvalues[eigenvalues <= threshold]
        noise_mean = float(np.mean(noise)) if len(noise) else 0.0
        cleaned = np.where(eigenvalues > threshold, eigenvalues, noise_mean)

        # P2-Q6-fix (Q6-L468): eig_cutoff 参数此前为死代码（pass，实际无效）。现实现
        #   真实截断：按特征值降序累计贡献达到 eig_cutoff 之后的尾部特征值压缩到噪声
        #   均值，避免高维尾部噪声进入协方差矩阵。
        asc = np.sort(cleaned)[::-1]  # 降序
        total_v = max(float(asc.sum()), 1e-12)
        cum_ratio = np.cumsum(asc) / total_v
        n_keep = int(np.searchsorted(cum_ratio, self.eig_cutoff)) + 1  # 达到 cutoff 所需最少个数
        n_keep = max(1, min(n_keep, len(cleaned)))
        if n_keep < len(cleaned):
            tail = np.argsort(cleaned)[:len(cleaned) - n_keep]  # 最小的 (n-n_keep) 个下标
            cleaned[tail] = noise_mean

        return eigenvectors @ np.diag(cleaned) @ eigenvectors.T

    def update(self, old_cov: np.ndarray, new_factor_return: pd.Series,
               old_mean: Optional[np.ndarray] = None) -> tuple:
        """增量更新协方差矩阵（递归 EWMA）。(V5.2 fix)

        Args:
            old_cov: 上一期协方差矩阵 (n_factors, n_factors)
            new_factor_return: 新一期因子收益 Series (index=factor_names)
            old_mean: 上一期因子收益均值向量 (n_factors,)

        Returns:
            (new_cov, new_mean)
        """
        new_arr = np.asarray(new_factor_return.values, dtype=float)
        # P2-Q6-fix (Q6-M469): 原硬编码 λ=0.94（RiskMetrics 半衰期≈11天），与类声明
        #   halflife=60（λ≈0.9885）不一致 → 增量更新与全量估计口径分裂；
        #   改为从 self.halflife 推导 λ = 2^(-1/halflife)
        lambda_ = 2 ** (-1.0 / max(int(self.halflife), 1))

        if old_mean is None:
            old_mean = np.zeros(len(new_arr))
        old_mean = np.asarray(old_mean, dtype=float)

        new_mean = lambda_ * old_mean + (1 - lambda_) * new_arr
        dev = new_arr - new_mean
        new_cov = lambda_ * np.asarray(old_cov, dtype=float) + (1 - lambda_) * np.outer(dev, dev)

        return (new_cov, new_mean)


# ══════════════════════════════════════
# 风险归因
# ══════════════════════════════════════

class RiskDecomposition:
    """风险归因引擎。

    将组合风险分解为：
    - 行业因子风险
    - 风格因子风险
    - 个股特异风险
    """

    def __init__(self, factor_cov: np.ndarray, specific_risk: np.ndarray,
                 industry_map: Optional[dict] = None):
        """
        Parameters
        ----------
        factor_cov : np.ndarray
            (K x K) 因子协方差矩阵
        specific_risk : np.ndarray
            (N x N) 对角阵，个股特异风险方差
        industry_map : dict, optional
            symbol -> industry 的映射
        """
        self.factor_cov = factor_cov
        self.specific_risk = specific_risk
        self.industry_map = industry_map or {}

    def decompose(self, weights: np.ndarray, exposures: np.ndarray,
                  factor_types: Optional[dict] = None) -> dict:
        """分解组合风险。

        Parameters
        ----------
        weights : np.ndarray
            (N,) 组合权重向量
        exposures : np.ndarray
            (N x K) 因子暴露矩阵
        factor_types : dict, optional
            {factor_name: "style"/"industry"} 因子分类标记

        Returns
        -------
        dict
            - total_var: 组合总方差
            - factor_var: 因子部分方差
            - specific_var: 特异部分方差
            - factor_contrib: 各因子贡献度
            - industry_contrib: 行业因子贡献
            - style_contrib: 风格因子贡献
        """
        # 组合方差 = w' (B @ F @ B' + S) w
        factor_part = exposures @ self.factor_cov @ exposures.T
        total_var = weights @ factor_part @ weights + weights @ self.specific_risk @ weights

        # 各因子边际贡献
        # 因子贡献度 = w' @ B_i * (F @ B' @ w)_i / total_var
        Bw = exposures.T @ weights  # (K,)
        FBw = self.factor_cov @ Bw    # (K,)

        factor_contrib = {}
        industry_contrib = None
        style_contrib = None
        for i in range(len(Bw)):
            factor_contrib[f"factor_{i}"] = Bw[i] * FBw[i] / max(total_var, 1e-12)

        # P2-Q6-fix (Q6-L471): docstring 承诺的 industry_contrib/style_contrib 此前
        #   从未计算。现按 factor_types（{factor名/index: "style"/"industry"}）聚合；
        #   未传 factor_types 时返回 None（文档化的诚实缺省）。
        if factor_types:
            ind_sum = 0.0
            sty_sum = 0.0
            for i in range(len(Bw)):
                label = factor_types.get(i, factor_types.get(f"factor_{i}", "style"))
                c = Bw[i] * FBw[i] / max(total_var, 1e-12)
                if str(label) == "industry":
                    ind_sum += c
                else:
                    sty_sum += c
            industry_contrib = float(ind_sum)
            style_contrib = float(sty_sum)

        return {
            "total_volatility": np.sqrt(total_var),
            "total_var": total_var,
            "factor_var": weights @ factor_part @ weights,
            "specific_var": weights @ self.specific_risk @ weights,
            "factor_contrib": factor_contrib,
            "industry_contrib": industry_contrib,
            "style_contrib": style_contrib,
            "n_assets": len(weights),
            "n_factors": exposures.shape[1],
        }

    def risk_budget(self, weights: np.ndarray, exposures: np.ndarray) -> pd.DataFrame:
        """计算每个资产的风险预算"""
        total = self.decompose(weights, exposures)
        specific_part = weights ** 2 * np.diag(self.specific_risk)
        specific_contrib = specific_part / max(total["total_var"], 1e-12)
        return pd.DataFrame({
            "weight": weights,
            "specific_risk_contrib": specific_contrib,
        })


# ══════════════════════════════════════
# 完整 Barra CNE6 风险模型
# ══════════════════════════════════════

class BarraModel:
    """Barra CNE6 风格风险模型。

    D6收敛登记: 独立能力保留（Barra CNE6 因子风险, 无等价实现）。

    整合因子暴露计算、因子协方差估计、风险归因于一体。

    Usage
    -----
    model = BarraModel()
    exposures = model.fit_exposures(data)
    cov = model.fit_covariance(exposures)
    result = model.risk_decompose(weights, exposures)
    """

    def __init__(self, halflife: int = 60, newey_lags: int = 2):
        self.exposure_calc = BarraExposure()
        self.cov_estimator = FactorCovariance(halflife=halflife, newey_lags=newey_lags)
        self._last_exposures: Optional[pd.DataFrame] = None
        self._last_cov: Optional[np.ndarray] = None
        self._specific_risk: Optional[np.ndarray] = None

    def fit_exposures(self, df: pd.DataFrame,
                      market_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """计算并标准化因子暴露"""
        raw = self.exposure_calc.compute(df, market_returns)
        self._last_exposures = self.exposure_calc.standardize_exposures(raw)

        # 构建行业哑变量
        industry_dummies = pd.get_dummies(self._last_exposures["industry"],
                                          prefix="ind")
        result = pd.concat([
            self._last_exposures[STYLE_FACTORS],
            industry_dummies,
        ], axis=1)

        return result

    def fit_covariance(self, factor_returns: pd.DataFrame) -> np.ndarray:
        """估计因子协方差矩阵"""
        self._last_cov = self.cov_estimator.estimate(factor_returns)
        return self._last_cov

    def estimate_specific_risk(self, returns: pd.DataFrame,
                               exposures: pd.DataFrame) -> np.ndarray:
        """估计个股特异风险（EWMA 残差方差）。

        P1-Q6-fix: 原实现 np.eye(N)*0.01 注释称“各股1%特异波动”，但该矩阵在
        decompose() 中被当**方差**用（w @ S @ w）→ 隐含 10%/日的波动，比重注释
        高 100 倍（方差量级），组合风险归因被特异风险主导。
        这里改用真实估计：逐期截面回归 r_{i,t}=Σ_k X_{i,k}f_{k,t}+ε_{i,t}，
        对残差序列做 EWMA 方差（半衰期沿用 cov_estimator.halflife），
        并以 0.01**2（1% 日波动方差）作为保底下限。
        """
        arr = np.asarray(returns, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        X = np.asarray(exposures, dtype=np.float64)
        T, N = arr.shape
        fallback = np.eye(N) * (0.01 ** 2)

        if T < 3 or N == 0 or X.ndim != 2 or X.shape[0] != N:
            # 样本或维度不足：回退到 1% 日波动方差（0.01**2）
            self._specific_risk = fallback
            return self._specific_risk

        K = X.shape[1]
        if K == 0 or N <= K:
            self._specific_risk = fallback
            return self._specific_risk

        try:
            # 逐期截面回归求残差（hat matrix 的逐期 lstsq 版本，数值更稳）
            resid = np.empty_like(arr)
            for t in range(T):
                coef, *_ = np.linalg.lstsq(X, arr[t], rcond=None)
                resid[t] = arr[t] - X @ coef
            resid -= resid.mean(axis=0)

            # EWMA 残差方差
            halflife = max(int(self.cov_estimator.halflife), 1)
            decay = 2 ** (-1 / halflife)
            weights = np.array([decay ** (T - 1 - t) for t in range(T)])
            weights /= weights.sum()
            var = np.einsum("i,ij->j", weights, resid ** 2) * (T / max(T - 1, 1))
            var = np.maximum(var, 0.01 ** 2)  # 保底 1% 日波动
            self._specific_risk = np.diag(var)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("特异风险 EWMA 估计失败（%s），回退 0.01**2 对角阵", exc)
            self._specific_risk = fallback
        return self._specific_risk

    def risk_decompose(self, weights: np.ndarray,
                       exposures: pd.DataFrame) -> dict:
        """风险归因"""
        if self._last_cov is None:
            raise ValueError("请先调用 fit_covariance()")

        K = self._last_cov.shape[0]
        N = exposures.shape[0]
        # P2-Q6-fix (Q6-M472): 校验暴露矩阵列数与协方差维度一致（行业哑变量数与
        #   协方差列数不一致时原实现静默算错或崩溃）；同时校验权重长度与暴露行数
        if exposures.shape[1] != K:
            raise ValueError(
                f"因子暴露列数 {exposures.shape[1]} 与协方差矩阵维度 {K} 不一致："
                f"请用同一批次 fit_exposures/fit_covariance 的结果"
            )
        if len(weights) != N:
            raise ValueError(f"权重长度 {len(weights)} 与暴露矩阵行数 {N} 不一致")

        if self._specific_risk is None:
            # P1-Q6-fix: 0.01 是”波动”不是”方差”，此处按方差语义用 0.01**2
            self._specific_risk = np.eye(N) * (0.01 ** 2)

        decomp = RiskDecomposition(self._last_cov, self._specific_risk)
        return decomp.decompose(weights, exposures.values)

    def risk_report(self, weights: np.ndarray, symbols: list[str]) -> str:
        """生成可读的风险归因报告"""
        if self._last_exposures is None or self._last_cov is None:
            return "风险模型未就绪（需先 fit）"

        # P1-Q6-fix: 原实现把已标准化的暴露矩阵 self._last_exposures 当原始行情
        # 再传给 fit_exposures() 二次加工，compute() 里所有 row.get() 全部落到
        # 默认值（beta=0、size=log(1e6)…），风险报告数字全是垃圾。
        # 改为直接用已标准化的缓存：风格因子 + 行业哑变量 拼出完整暴露矩阵。
        exposures = pd.concat([
            self._last_exposures[STYLE_FACTORS],
            pd.get_dummies(self._last_exposures["industry"], prefix="ind"),
        ], axis=1)
        result = self.risk_decompose(weights, exposures)

        lines = [
            "=" * 50,
            "Barra CNE6 风险归因报告 (V4.1 feature)",
            "=" * 50,
            f"组合总波动: {result['total_volatility']*100:.2f}%",
            f"  ├ 因子风险: {np.sqrt(result['factor_var'])*100:.2f}%",
            f"  └ 特异风险: {np.sqrt(result['specific_var'])*100:.2f}%",
            f"资产数量: {result['n_assets']}",
            f"因子数量: {result['n_factors']}",
        ]
        return "\n".join(lines)
