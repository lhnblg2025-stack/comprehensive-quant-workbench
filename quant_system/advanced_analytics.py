"""
advanced_analytics.py — 高级统计分析工具
V4.1 feature

提供：滚动回归、卡尔曼滤波、Bootstrap、因子诊断、状态空间模型
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

logger = __import__('logging').getLogger(__name__)

try:
    from scipy import stats, linalg
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ══════════════════════════════════════
# 1. 滚动回归
# ══════════════════════════════════════

class RollingRegression:
    """滚动回归分析"""

    def __init__(self, window: int = 60):
        self.window = window

    def ols(self, y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
        """滚动OLS回归"""
        if not _HAS_STATSMODELS:
            raise ImportError("需要statsmodels")
        results = []
        for i in range(self.window, len(y)):
            yw = y.iloc[i - self.window:i]
            Xw = X.iloc[i - self.window:i]
            Xw = sm.add_constant(Xw)
            model = sm.OLS(yw, Xw).fit()
            results.append({
                "date": y.index[i],
                "alpha": model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0],
                "alpha_tstat": model.tvalues.iloc[0] if hasattr(model.tvalues, 'iloc') else model.tvalues[0],
                "rsquared": model.rsquared,
                "nobs": model.nobs,
            })
            for j, col in enumerate(X.columns):
                results[-1][f"beta_{col}"] = (
                    model.params.iloc[j + 1] if hasattr(model.params, 'iloc') else model.params[j + 1]
                )
        return pd.DataFrame(results)

    def linear_trend(self, series: pd.Series) -> pd.Series:
        """滚动线性趋势斜率"""
        slopes = series.rolling(self.window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 10 else 0,
            raw=True
        )
        return slopes

    def rolling_corr(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """滚动相关系数"""
        return a.rolling(self.window).corr(b)

    def rolling_beta(self, stock_ret: pd.Series, market_ret: pd.Series) -> pd.Series:
        """滚动Beta"""
        cov = stock_ret.rolling(self.window).cov(market_ret)
        var = market_ret.rolling(self.window).var().clip(lower=1e-12)
        return cov / var

    def rolling_alpha(self, stock_ret: pd.Series, market_ret: pd.Series,
                       rf: float = 0.025) -> pd.Series:
        """滚动Alpha (Jensen)"""
        rf_daily = rf / 252
        excess_stock = stock_ret - rf_daily
        excess_market = market_ret - rf_daily
        beta = self.rolling_beta(stock_ret, market_ret)
        alpha = excess_stock - beta * excess_market
        return alpha

    def rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """滚动夏普比率"""
        mean = returns.rolling(self.window).mean() * 252
        std = returns.rolling(self.window).std() * np.sqrt(252)
        return mean / std.clip(lower=1e-12)


# ══════════════════════════════════════
# 2. 卡尔曼滤波
# ══════════════════════════════════════

class KalmanFilterAnalyzer:
    """卡尔曼滤波分析"""

    def __init__(self, vt: float = 1e-3, wt: float = 1e-5):
        # P2-Q27-fix(L351): 移除从未使用的 delta 参数（原 self.delta 为死存储，
        # 全仓库无调用方使用 delta，直接删除不再保存）。
        self.vt = vt
        self.wt = wt

    def estimate_beta(self, y: np.ndarray, x: np.ndarray) -> dict:
        """卡尔曼滤波估计时变Beta
        
        状态空间: Beta(t) = Beta(t-1) + w(t)
        观测方程: y(t) = x(t) * Beta(t) + v(t)
        """
        n = len(y)
        beta = np.zeros(n)
        beta_cov = np.zeros(n)

        # 初始状态
        b = 1.0
        P = 100.0

        for t in range(n):
            # 预测步
            b_pred = b
            P_pred = P + self.wt

            # 更新步
            K = P_pred * x[t] / (x[t] ** 2 * P_pred + self.vt)
            b = b_pred + K * (y[t] - x[t] * b_pred)
            P = (1 - K * x[t]) * P_pred

            beta[t] = b
            beta_cov[t] = P

        return {"beta": beta, "beta_std": np.sqrt(beta_cov)}

    def estimate_state(self, observations: np.ndarray) -> dict:
        """简单状态估计"""
        n = len(observations)
        state = np.zeros(n)
        state_cov = np.zeros(n)

        s = observations[0]
        P = 10.0

        for t in range(n):
            s_pred = s
            P_pred = P + self.wt
            K = P_pred / (P_pred + self.vt)
            s = s_pred + K * (observations[t] - s_pred)
            P = (1 - K) * P_pred
            state[t] = s
            state_cov[t] = P

        return {"state": state, "state_std": np.sqrt(state_cov)}

    def smoother(self, observations: np.ndarray) -> np.ndarray:
        """卡尔曼平滑（前向后向）"""
        n = len(observations)
        # 前向滤波
        s_fwd = np.zeros(n)
        s = observations[0]
        P = 10.0
        P_fwd = np.zeros(n)

        for t in range(n):
            # P2-Q27-fix(M350): 前向滤波预测步补状态噪声 P_pred = P + self.wt，
            # 与 estimate_state 一致；原实现未加 wt，而后向平滑却按含 wt 的公式
            # C = P_fwd/(P_fwd+wt) 计算，前后不一致导致平滑权重错误。
            P_pred = P + self.wt
            K = P_pred / (P_pred + self.vt)
            s = s + K * (observations[t] - s)
            P = (1 - K) * P_pred
            s_fwd[t] = s
            P_fwd[t] = P

        # 后向平滑
        s_smooth = np.zeros(n)
        s_smooth[-1] = s_fwd[-1]

        for t in range(n - 2, -1, -1):
            C = P_fwd[t] / (P_fwd[t] + self.wt)
            s_smooth[t] = s_fwd[t] + C * (s_smooth[t + 1] - s_fwd[t])

        return s_smooth


# ══════════════════════════════════════
# 3. Bootstrap 分析
# ══════════════════════════════════════

class BootstrapAnalyzer:
    """Bootstrap 重抽样分析"""

    def __init__(self, n_bootstrap: int = 1000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    def sharpe_confidence(self, returns: pd.Series) -> dict:
        """夏普比率的Bootstrap置信区间"""
        n = len(returns)
        sharpes = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = returns.sample(n=n, replace=True, random_state=self.rng)
            sr = sample.mean() / max(sample.std(), 1e-12) * np.sqrt(252)
            sharpes[i] = sr

        return {
            "sharpe_obs": returns.mean() / max(returns.std(), 1e-12) * np.sqrt(252),
            "sharpe_mean": sharpes.mean(),
            "sharpe_std": sharpes.std(),
            "ci_95": (np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)),
            "ci_90": (np.percentile(sharpes, 5), np.percentile(sharpes, 95)),
            "prob_positive": (sharpes > 0).mean(),
        }

    def max_drawdown_confidence(self, returns: pd.Series) -> dict:
        """最大回撤置信区间"""
        mdd_values = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = returns.sample(n=len(returns), replace=True, random_state=self.rng)
            cum = (1 + sample).cumprod()
            peak = cum.expanding().max()
            dd = (cum - peak) / peak
            mdd_values[i] = dd.min()

        return {
            "mdd_obs": float((1 + returns).cumprod().div(
                (1 + returns).cumprod().expanding().max()).min() - 1),
            "mdd_mean": mdd_values.mean(),
            "mdd_median": np.median(mdd_values),
            "ci_95": (np.percentile(mdd_values, 2.5), np.percentile(mdd_values, 97.5)),
        }

    def correlation_test(self, a: pd.Series, b: pd.Series) -> dict:
        """Bootstrap相关系数检验"""
        n = len(a)
        corrs = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self.rng.randint(0, n, n)
            corrs[i] = a.iloc[idx].corr(b.iloc[idx])
        return {
            "corr_obs": a.corr(b),
            "corr_mean": corrs.mean(),
            "corr_std": corrs.std(),
            "ci_95": (np.percentile(corrs, 2.5), np.percentile(corrs, 97.5)),
            "p_value": (np.abs(corrs) >= abs(a.corr(b))).mean(),
        }

    def strategy_significance(self, strategy_ret: pd.Series,
                               benchmark_ret: pd.Series) -> dict:
        """策略显著性Bootstrap检验

        P1-Q27-fix: 原实现直接从 excess 重抽样（分布中心=观测均值），
        未按 H0(均值=0) 重中心化 → 任何策略都判"不显著"。
        现改为对 (excess − mean) 重抽样，再计算双尾 p 值。
        """
        n = len(strategy_ret)
        excess = strategy_ret - benchmark_ret
        actual_mean = excess.mean()
        # 在 H0 下重抽样：均值归零
        centered = excess - actual_mean

        bootstrap_means = np.zeros(self.n_bootstrap)   # H0 中心化样本 → p 值
        bootstrap_null = np.zeros(self.n_bootstrap)    # 原始样本 → CI
        for i in range(self.n_bootstrap):
            s0 = centered.sample(n=n, replace=True, random_state=self.rng)
            bootstrap_means[i] = s0.mean()
            s1 = excess.sample(n=n, replace=True, random_state=self.rng)
            bootstrap_null[i] = s1.mean()

        # 双尾检验: |H0重抽样均值| >= |观测均值| 的比例
        p_value = float((np.abs(bootstrap_means) >= abs(actual_mean)).mean())
        return {
            "excess_return": actual_mean * 252,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "ci_95": (np.percentile(bootstrap_null, 2.5) * 252,
                      np.percentile(bootstrap_null, 97.5) * 252),
        }


# ══════════════════════════════════════
# 4. 因子模型诊断
# ══════════════════════════════════════

class FactorDiagnostics:
    """因子模型诊断工具"""

    @staticmethod
    def ic_test(ic_series: pd.Series) -> dict:
        """IC显著性检验"""
        if not _HAS_SCIPY:
            return {"error": "需要scipy"}
        t_stat = ic_series.mean() / max(ic_series.std() / np.sqrt(len(ic_series)), 1e-12)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ic_series) - 1))
        return {
            "mean_ic": ic_series.mean(),
            "ic_std": ic_series.std(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_1pct": p_value < 0.01,
            "significant_5pct": p_value < 0.05,
            "positive_ratio": (ic_series > 0).mean(),
        }

    @staticmethod
    def factor_collinearity(factor_df: pd.DataFrame, threshold: float = 0.8) -> dict:
        """因子共线性检测"""
        corr = factor_df.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [(col, row, upper.loc[row, col])
                     for col in upper.columns
                     for row in upper.index
                     if abs(upper.loc[row, col]) > threshold]
        return {
            "n_high_corr": len(high_corr),
            "high_corr_pairs": high_corr[:10],
            "avg_abs_corr": corr.abs().values[np.triu_indices_from(corr.values, k=1)].mean(),
            "condition_number": np.linalg.cond(factor_df.fillna(0).values),
        }

    @staticmethod
    def vif_test(factor_df: pd.DataFrame) -> pd.Series:
        """方差膨胀因子 (VIF)"""
        if not _HAS_STATSMODELS:
            return pd.Series(dtype=float)
        vifs = {}
        for col in factor_df.columns:
            others = factor_df.drop(columns=[col]).fillna(0)
            X = sm.add_constant(others)
            try:
                model = sm.OLS(factor_df[col].fillna(0), X).fit()
                rsq = model.rsquared
                vifs[col] = 1 / (1 - rsq) if rsq < 0.999 else 999
            except Exception:
                vifs[col] = 999
        return pd.Series(vifs)

    @staticmethod
    def factor_breakdown(returns: pd.Series, factor_exposures: pd.DataFrame,
                          factor_returns: pd.DataFrame) -> dict:
        """因子收益分解"""
        if not _HAS_STATSMODELS:
            return {"error": "需要statsmodels"}
        X = sm.add_constant(factor_exposures.fillna(0))
        try:
            model = sm.OLS(returns, X).fit()
            return {
                "factor_loadings": dict(zip(["const"] + list(factor_exposures.columns),
                                            model.params)),
                "t_values": dict(zip(["const"] + list(factor_exposures.columns),
                                    model.tvalues)),
                "rsquared": model.rsquared,
                "adj_rsquared": model.rsquared_adj,
                "f_stat": model.fvalue,
                "f_pvalue": model.f_pvalue,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def spanning_test(test_returns: pd.Series,
                       benchmark_returns: pd.DataFrame) -> dict:
        """Spanning Test：检验策略是否提供了基准之外的新信息"""
        if not _HAS_STATSMODELS:
            return {"error": "需要statsmodels"}
        X = sm.add_constant(benchmark_returns.fillna(0))
        model = sm.OLS(test_returns, X).fit()
        # Wald test: alpha = 0
        f_test = model.f_test("const = 0")
        return {
            "alpha": model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0],
            "alpha_pvalue": f_test.pvalue,
            # P2-Q27-fix(M349): 原实现 `pvalue > 0.05` 表示 alpha 不显著（未超越基准）
            # 却返回 True，语义与命名颠倒。改为 pvalue < 0.05：alpha 显著 ≠ 0 才认为
            # 策略提供了基准之外的新信息（spanning）。另增 has_alpha 别名便于理解。
            "spanning_alpha": f_test.pvalue < 0.05,
            "has_alpha": f_test.pvalue < 0.05,
            "rsquared": model.rsquared,
        }


# ══════════════════════════════════════
# 5. 波动率建模
# ══════════════════════════════════════

class VolatilityModeling:
    """波动率建模工具"""

    @staticmethod
    def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
        """EWMA波动率（RiskMetrics）"""
        var = returns.ewm(alpha=1 - lambda_, adjust=False).var()
        return np.sqrt(var)

    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series,
                              window: int = 20) -> pd.Series:
        """Parkinson 极值波动率"""
        ratio = (high / low).clip(lower=1e-12)
        log_ratio = np.log(ratio)
        vol = np.sqrt(log_ratio.pow(2).rolling(window).mean() / (4 * np.log(2)))
        return vol

    @staticmethod
    def garman_klass_volatility(open_: pd.Series, high: pd.Series,
                                 low: pd.Series, close: pd.Series,
                                 window: int = 20) -> pd.Series:
        """Garman-Klass 波动率"""
        log_hl = np.log(high / low.clip(lower=1e-12))
        log_co = np.log(close / open_.clip(lower=1e-12))
        var = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
        return np.sqrt(var.rolling(window).mean())

    @staticmethod
    def yang_zhang_volatility(open_: pd.Series, high: pd.Series,
                               low: pd.Series, close: pd.Series,
                               window: int = 20) -> pd.Series:
        """Yang-Zhang 波动率（标准公式，修复 Q27）

        σ_yz² = σ_o² + k·σ_c² + (1−k)·σ_rs²
          - overnight:   σ_o² = E[ln(O_t / C_{t-1})²]   （隔夜收益）
          - open-close:  σ_c² = E[ln(C_t / O_t)²]       （日内开收收益）
          - Rogers-Satchell: σ_rs² = E[ln(H_t/C_t)·ln(H_t/O_t) + ln(L_t/C_t)·ln(L_t/O_t)]
          - k = 0.34 / (1.34 + (n+1)/(n−1))

        原实现 :404 引用了未定义的 close_（NameError 必然崩溃），
        overnight/open-close 项亦不符合标准定义。
        """
        # P1-Q27-fix: 按标准 Yang-Zhang 重写。
        #   σ_yz² = σ_o² + k·σ_c² + (1−k)·σ_rs²
        #   - overnight   o_t = ln(O_t / C_{t-1})          → σ_o² = Var(o)
        #   - open-close  c_t = ln(C_t / O_t)              → σ_c² = Var(c)
        #   - Rogers-Satchell σ_rs² = E[ ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ]
        # 原实现把 σ_o/σ_c 用均方而非方差，且 RS 项误用 ln(C/O)（应为 ln(H/C)、ln(L/C)），
        # 导致波动率系统性偏高（实测均值 0.494 vs 参考 0.289）。
        o_t = np.log(open_ / close.shift(1).clip(lower=1e-12))
        c_t = np.log(close / open_.clip(lower=1e-12))
        log_hc = np.log(high / close.clip(lower=1e-12))
        log_ho = np.log(high / open_.clip(lower=1e-12))
        log_lc = np.log(low / close.clip(lower=1e-12))
        log_lo = np.log(low / open_.clip(lower=1e-12))

        sigma_o2 = o_t.rolling(window).var()
        sigma_c2 = c_t.rolling(window).var()
        sigma_rs = (log_hc * log_ho + log_lc * log_lo).rolling(window).mean()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        return np.sqrt(sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs)

    @staticmethod
    def volatility_term_structure(returns: pd.Series,
                                   horizons: list = None) -> pd.Series:
        """波动率期限结构"""
        if horizons is None:
            horizons = [5, 10, 20, 60, 120, 252]
        vols = {}
        for h in horizons:
            vol = returns.rolling(h).std() * np.sqrt(252)
            vols[f"{h}d"] = vol.iloc[-1] if len(vol) > 0 else 0
        return pd.Series(vols)


# ══════════════════════════════════════
# 6. 统计检验
# ══════════════════════════════════════

class StatisticalTests:
    """统计检验工具包"""

    @staticmethod
    def normality_test(series: pd.Series) -> dict:
        """正态性检验"""
        if not _HAS_SCIPY:
            return {"error": "需要scipy"}
        stat, p_value = stats.jarque_bera(series.dropna())
        return {
            "test": "Jarque-Bera",
            "statistic": stat,
            "p_value": p_value,
            "is_normal": p_value > 0.05,
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
        }

    @staticmethod
    def stationarity_test(series: pd.Series) -> dict:
        """平稳性检验 (ADF)"""
        if not _HAS_STATSMODELS:
            return {"error": "需要statsmodels"}
        result = sm.tsa.adfuller(series.dropna(), maxlag=20)
        return {
            "test": "ADF",
            "adf_stat": result[0],
            "p_value": result[1],
            "critical_1pct": result[4]["1%"],
            "critical_5pct": result[4]["5%"],
            "critical_10pct": result[4]["10%"],
            "is_stationary": result[1] < 0.05,
        }

    @staticmethod
    def autocorrelation_test(series: pd.Series, lags: int = 20) -> dict:
        """自相关检验 (Ljung-Box)"""
        if not _HAS_STATSMODELS:
            return {"error": "需要statsmodels"}
        result = sm.stats.acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
        return {
            "test": "Ljung-Box",
            "lag": lags,
            "statistic": result["lb_stat"].iloc[0],
            "p_value": result["lb_pvalue"].iloc[0],
            "has_autocorr": result["lb_pvalue"].iloc[0] < 0.05,
        }

    @staticmethod
    def cointegration_test(a: pd.Series, b: pd.Series) -> dict:
        """协整检验 (Engle-Granger)"""
        if not _HAS_STATSMODELS:
            return {"error": "需要statsmodels"}
        from statsmodels.tsa.stattools import coint
        stat, p_value, _ = coint(a.dropna(), b.dropna())
        return {
            "test": "Engle-Granger",
            "coint_stat": stat,
            "p_value": p_value,
            "is_cointegrated": p_value < 0.05,
        }

    @staticmethod
    def structural_break_test(series: pd.Series) -> dict:
        """结构突变检验 (Chow)"""
        n = len(series)
        if n < 30:
            return {"error": "样本不足"}
        mid = n // 2
        if not _HAS_STATSMODELS:
            return {"error": "需要statsmodels"}
        try:
            X_full = sm.add_constant(np.arange(n))
            X_1 = sm.add_constant(np.arange(mid))
            X_2 = sm.add_constant(np.arange(mid, n)) - mid
            model_full = sm.OLS(series.values, X_full).fit()
            model_1 = sm.OLS(series.values[:mid], X_1).fit()
            model_2 = sm.OLS(series.values[mid:], X_2).fit()
            rss_full = model_full.ssr
            rss_1 = model_1.ssr
            rss_2 = model_2.ssr
            f_stat = ((rss_full - (rss_1 + rss_2)) / 2) / ((rss_1 + rss_2) / (n - 4))
            p_value = 1 - stats.f.cdf(f_stat, 2, n - 4)
            return {
                "test": "Chow",
                "f_stat": f_stat,
                "p_value": p_value,
                "break_point": mid,
                "has_break": p_value < 0.05,
            }
        except Exception as e:
            return {"error": str(e)}


# ══════════════════════════════════════
# 7. 高级报告生成
# ══════════════════════════════════════

class AnalyticsReport:
    """统计分析报告"""

    def __init__(self):
        self.rolling = RollingRegression()
        self.kalman = KalmanFilterAnalyzer()
        self.bootstrap = BootstrapAnalyzer()
        self.factor_diag = FactorDiagnostics()
        self.vol = VolatilityModeling()
        self.stats = StatisticalTests()

    def full_report(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> str:
        """综合统计分析报告"""
        sections = [
            "=" * 55,
            "高级统计分析报告 (V4.1 feature)",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
        ]

        # 描述性统计
        desc = returns.describe()
        sections.extend([
            "",
            "【描述性统计】",
            f"  均值: {returns.mean()*100:.4f}%  中位数: {returns.median()*100:.4f}%",
            f"  标准差: {returns.std()*100:.4f}%  偏度: {returns.skew():.3f}",
            f"  峰度: {returns.kurtosis():.3f}  最小值: {returns.min()*100:.4f}%",
            f"  最大值: {returns.max()*100:.4f}%",
        ])

        # 正态性检验
        norm = self.stats.normality_test(returns)
        if "p_value" in norm:
            sections.append(f"  正态性: JB={norm['statistic']:.2f} p={norm['p_value']:.4f} {'正态' if norm['is_normal'] else '非正态'}")

        # 平稳性
        station = self.stats.stationarity_test(returns)
        if "p_value" in station:
            sections.append(f"  平稳性: ADF={station['adf_stat']:.3f} p={station['p_value']:.4f} {'平稳' if station['is_stationary'] else '非平稳'}")

        # 自相关
        ac = self.stats.autocorrelation_test(returns, lags=10)
        if "p_value" in ac:
            sections.append(f"  自相关: LB={ac['statistic']:.2f} p={ac['p_value']:.4f} {'有自相关' if ac['has_autocorr'] else '无自相关'}")

        # Bootstrap 夏普
        sections.extend(["", "【Bootstrap 夏普置信区间】"])
        bs = self.bootstrap.sharpe_confidence(returns)
        sections.append(f"  观测夏普: {bs['sharpe_obs']:.3f}")
        sections.append(f"  均值夏普: {bs['sharpe_mean']:.3f}")
        sections.append(f"  95% CI: ({bs['ci_95'][0]:.3f}, {bs['ci_95'][1]:.3f})")
        sections.append(f"  夏普为正概率: {bs['prob_positive']*100:.1f}%")

        # 滚动夏普
        sections.extend(["", "【滚动夏普 (60d)】"])
        roll_sharpe = self.rolling.rolling_sharpe(returns)
        sections.append(f"  均值: {roll_sharpe.mean():.3f}")
        sections.append(f"  标准差: {roll_sharpe.std():.3f}")
        sections.append(f"  最小值: {roll_sharpe.min():.3f}")

        # 波动率
        sections.extend(["", "【波动率分析】"])
        ewma_vol = self.vol.ewma_volatility(returns)
        sections.append(f"  EWMA波动率: {ewma_vol.iloc[-1]*100:.2f}%" if len(ewma_vol) > 0 else "")
        term = self.vol.volatility_term_structure(returns)
        for h, v in term.items():
            sections.append(f"  {h}: {v*100:.2f}%")

        # 如果提供基准
        if benchmark is not None:
            sections.extend(["", "【与基准对比】"])
            common = returns.index.intersection(benchmark.index)
            if len(common) > 0:
                rolling_beta = self.rolling.rolling_beta(
                    returns.loc[common], benchmark.loc[common])
                rolling_alpha = self.rolling.rolling_alpha(
                    returns.loc[common], benchmark.loc[common])
                sections.append(f"  滚动Beta均值: {rolling_beta.mean():.3f}")
                sections.append(f"  滚动Alpha均值: {rolling_alpha.mean()*100:.2f}%")

        sections.append("")
        sections.append("=" * 55)
        return "\n".join(sections)


__all__ = [
    "RollingRegression", "KalmanFilterAnalyzer",
    "BootstrapAnalyzer", "FactorDiagnostics",
    "VolatilityModeling", "StatisticalTests",
    "AnalyticsReport",
]
