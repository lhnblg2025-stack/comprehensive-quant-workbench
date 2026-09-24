"""
factor_model.py — V8 多因子量化模型

A股多因子模型，覆盖六大类因子 + IC 跟踪 + 因子组合。

因子分类:
  1. 价值 (Value)        — PE, PB, PS, PC, 股息率
  2. 动量 (Momentum)     — 1M/3M/6M/12M 动量, 近N日超额
  3. 质量 (Quality)      — ROE, ROA, 毛利率, 资产负债率, 现金流
  4. 规模 (Size)         — 总市值, 流通市值, 对数市值
  5. 波动 (Volatility)   — 日波动率, 特异波动率, Beta, 最大回撤
  6. 成长 (Growth)       — 营收增长, 利润增长, 超预期

核心功能:
  - init_universe()    — 初始化全A股票池, 过滤ST/*ST/新股
  - compute_factors()   — 计算指定日期的因子暴露
  - compute_ic()       — 计算因子 IC (信息系数)
  - factor_report()    — 因子表现报告 (IC/IR/分组收益)
  - risk_model()       — Barra-style 风险模型协方差矩阵
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 本模块 _compute_single_factor 产出38个因子名，其中15个与 factor_zoo 同名异口径(不强迁)，其余独特。

因子自动退役制度 (P2a, 2026-08-13):
  - factor_retirement_scheduler() 按自然周（W-SUN）计算因子 IC，滚动检查
    连续 RETIRE_CONSEC_WEEKS 周的 ICIR 与胜率；双低时标 RETIRED。
  - save_retirement_state() 将扫描结果幂等合并到
    generated/factor_retirement.json（独立状态文件）。
  - factor_zoo.active 是静态配置（人工登记/未实现标记），与本调度器生成的
    factor_retirement.json 两处并存；调度器输出供人工复核，不直接改写
    factor_zoo.active。
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime  # P2-Q5-fix (M436): timedelta 不再用于 IC 序列步进，移除未使用导入
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_system.data_store import get_store

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ANNUAL_TRADING_DAYS = 242
KLINE_DIR = ROOT / "data_warehouse" / "kline"

# 因子分组
FACTOR_GROUPS = {
    "value":     ["pe_ttm", "pb", "ps_ttm", "pcf_ttm", "div_yield"],
    "momentum":  ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "excess_ret_20d"],
    "quality":   ["roe", "roa", "gross_margin", "debt_ratio", "ocf_to_sales"],
    "size":      ["ln_cap", "ln_float_cap"],
    "volatility": ["daily_vol_20d", "idio_vol_20d", "beta_60d", "max_dd_60d", "har_rv_pred", "rv_daily", "rv_weekly", "rv_monthly"],
    "growth":    ["sales_growth_yoy", "profit_growth_yoy", "surprise"],
    "microstructure": ["amihud_illiq"],
}

ALL_FACTOR_NAMES = [
    # Value
    "pe_ttm", "pb", "ps_ttm", "pcf_ttm", "div_yield",
    # Momentum
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "excess_ret_20d",
    # Quality
    "roe", "roa", "gross_margin", "debt_ratio", "ocf_to_sales",
    # Size
    "ln_cap", "ln_float_cap",
    # Volatility
    "daily_vol_20d", "idio_vol_20d", "beta_60d", "max_dd_60d",
    # Growth
    "sales_growth_yoy", "profit_growth_yoy", "surprise",
    # Microstructure (V8⁺)
    "amihud_illiq", "har_rv_pred", "rv_daily", "rv_weekly", "rv_monthly",
]

# 因子去极值和标准化方法
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# V6 fix (Q5 HIGH: factor_model.py:335-341,604-628): 风险因子方向语义
# 这些因子取正值（越大越危险），选股端按「越小越好」取多头，避免方向性反转。
RISK_FACTORS_LOWER_IS_BETTER = frozenset({
    "daily_vol_20d", "idio_vol_20d", "beta_60d", "max_dd_60d",
    "har_rv_pred", "rv_daily", "rv_weekly", "rv_monthly",
    "amihud_illiq", "debt_ratio",
})

# 因子自动退役调度器阈值（P2a）
RETIRE_ICIR_MIN = 0.1          # 周 ICIR 下限：连续低于该值且胜率不达标才退役
RETIRE_WIN_MIN = 0.45          # 周胜率下限：IC>0 的周占比低于该值才退役
RETIRE_CONSEC_WEEKS = 4        # 连续观察周数
RETIRE_WINDOW_DAYS = 60        # IC 计算回看窗口（自然日）

FACTOR_RETIREMENT_PATH = ROOT / "generated" / "factor_retirement.json"


# ---------------------------------------------------------------------------
# Helpers: neutralization
# ---------------------------------------------------------------------------

def _scan_kline_row_counts(kline_dir: str | Path) -> dict[str, int]:
    """Scan kline parquet row counts without loading full dataframes.

    Uses ``pyarrow.parquet.ParquetFile.metadata.num_rows`` when available; falls
    back to reading only the date column for any unreadable/corrupt file.
    """
    directory = Path(kline_dir)
    if not directory.is_dir():
        return {}

    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None

    row_counts: dict[str, int] = {}
    for path in directory.glob("*.parquet"):
        code = path.stem.zfill(6)
        row_count: int | None = None
        if pq is not None:
            try:
                parquet_file = pq.ParquetFile(path)
                if parquet_file.metadata is not None:
                    row_count = parquet_file.metadata.num_rows
            except Exception:
                row_count = None
        if row_count is None:
            try:
                row_count = len(pd.read_parquet(path, columns=["date"]))
            except Exception:
                continue
        if isinstance(row_count, int) and row_count >= 0:
            row_counts[code] = row_count
    return row_counts


@lru_cache(maxsize=1)
def _get_kline_row_counts(kline_dir: str, dir_mtime_ns: int) -> dict[str, int]:
    """Cached row-count map; directory mtime participates in the cache key."""
    return _scan_kline_row_counts(kline_dir)


def _winsorize(s: pd.Series, lower: float = WINSORIZE_LOWER,
               upper: float = WINSORIZE_UPPER) -> pd.Series:
    """Winsorize at quantiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def _standardize(s: pd.Series) -> pd.Series:
    """Z-score standardization (cross-sectional)."""
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _neutralize(s: pd.Series, industry_dummies: pd.DataFrame) -> pd.Series:
    """Cross-sectional industry neutralization via robust regression residual."""
    valid = s.notna()
    if valid.sum() < 5:
        return pd.Series(0.0, index=s.index)
    X = industry_dummies.loc[valid].values
    y = s[valid].values
    if X.shape[1] < 1 or X.shape[0] < X.shape[1] + 5:
        return s
    try:
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor(epsilon=1.35).fit(X, y)
    except Exception:
        try:
            from sklearn.linear_model import LinearRegression
            y = _winsorize(pd.Series(y, index=s[valid].index)).values
            model = LinearRegression().fit(X, y)
        except Exception:
            return s
    residual = pd.Series(y - model.predict(X), index=s[valid].index)
    result = pd.Series(np.nan, index=s.index)
    result[valid] = residual
    return result


def _gfv_nan(symbol: str, key: str) -> float:
    """P1-Q5-fix: financial_data.get_latest_value 缺失时返回 0.0。

    若把 0.0 当作真实值参与截面 z-score，无财务数据的股票会被误判为
    "极便宜/零增长"。这里把缺失（0.0）转为 NaN——缺失即不进截面，
    由 compute_factors 的标准化作中性 0 处理；也不许用价格代理冒充。
    """
    try:
        from quant_system.financial_data import get_latest_value
        v = get_latest_value(symbol, key)
        if v is None or v == 0.0 or not np.isfinite(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

class FactorModel:
    """A股多因子量化模型"""

    def __init__(self, universe_size: int = 500):
        self.universe_size = universe_size
        self._factor_cache: dict[str, pd.DataFrame] = {}  # date -> factor DataFrame
        self._ic_cache: dict[str, pd.Series] = {}  # factor_name -> IC series
        # V4.1 arch fix: 单因子缓存 {(date_str, symbol): factor_dict}，避免重复计算
        self._single_factor_cache: dict[tuple[str, str], dict[str, float]] = {}
        # V4.1 feature: Parquet持久化因子存储
        self.factor_store = FactorStore()

    # ── 股票池 ──────────────────────────────────────────────────────────

    def get_universe(self, n: int | None = None, min_trading_days: int | None = 60) -> list[str]:
        """Return factor-universe symbols, optionally applying an IPO cool-down.

        ``n <= 30`` uses the hardcoded large-cap fallback and is never filtered.
        Larger universes fetch the akshare spot list, then apply
        ``_filter_ipo_lockout`` when ``min_trading_days`` is non-zero.

        Because ``listing_dates.parquet`` does not provide reliable listing dates,
        the lockout infers tradable history from the row count of each
        ``data_warehouse/kline/{code}.parquet`` file. ``None`` or ``0`` disables
        the lockout.
        """
        n = n or self.universe_size
        fallback = ["600519","000858","002714","601899","002594","300750","600036","601318","000333","600276",
                    "000568","002415","000001","601166","600900","600887","601398","601939","601288","601988",
                    "600030","601211","000651","000725","002304","600585","601088","600028","601857","600031"]
        # Try to fetch A-share stock list dynamically for larger universe
        if n > 30:
            try:
                import akshare as ak
                stock_info = ak.stock_zh_a_spot_em()
                if stock_info is not None and not stock_info.empty:
                    codes = stock_info["代码"].astype(str).str.zfill(6).tolist()
                    # Remove ST / *ST / 北交所
                    if "名称" in stock_info.columns:
                        mask = ~stock_info["名称"].str.contains(r"ST|退|北交|B股", na=False)
                        codes = [c for c, m in zip(codes, mask) if m]
                    if min_trading_days:
                        codes = self._filter_ipo_lockout(codes, min_trading_days)
                    return codes[:min(n, 3000)]
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
        return fallback[:min(n, 30)]

    def _filter_ipo_lockout(self, codes: list[str], min_trading_days: int) -> list[str]:
        """Remove codes with too few kline rows (recent-IPO cool-down).

        Row counts are read once from ``data_warehouse/kline/*.parquet`` via a
        small cache. Codes with fewer than ``min_trading_days`` rows are excluded;
        codes whose kline file is missing or unreadable are kept so the filter
        never silently removes a stock it could not inspect.
        """
        if not codes:
            return []

        kline_dir = KLINE_DIR
        try:
            dir_mtime_ns = kline_dir.stat().st_mtime_ns if kline_dir.exists() else 0
        except OSError:
            dir_mtime_ns = 0
        row_counts = _get_kline_row_counts(str(kline_dir), dir_mtime_ns)
        return [code for code in codes if row_counts.get(code, min_trading_days) >= min_trading_days]

    # ── 因子计算 ────────────────────────────────────────────────────────

    def compute_factors(
        self,
        date_str: str | None = None,
        symbols: list[str] | None = None,
        neutralize: bool = True,
    ) -> pd.DataFrame:
        """Compute all factors for a universe on a given date.

        Returns DataFrame indexed by symbol, columns = factor names.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        if date_str in self._factor_cache:
            return self._factor_cache[date_str]

        # V4.1 feature: 检查Parquet持久缓存，避免重复计算
        # V6 fix (Q5 CRITICAL 前视偏差): 校验缓存数据截止日。
        #   V5.4 之前缓存的因子暴露是用「截至今天」的数据算的（对历史 date_str
        #   含未来信息）；is_cache_safe 仅当元数据存在且 data_through ≤ date_str
        #   才放行，否则拒绝复用并重算。
        stored = self.factor_store.load(date_str, date_str)
        if not stored.empty and self.factor_store.is_cache_safe(date_str):
            try:
                cached = stored.reset_index(level="date", drop=True)
                if isinstance(cached, pd.DataFrame) and len(cached) > 5:
                    self._factor_cache[date_str] = cached
                    return cached
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)

        if symbols is None:
            # P1-Q5-fix: V5.4 硬编码 n=200，universe_size（默认500）形同虚设。
            #   改为按 self.universe_size 取数，调用方可通过构造参数控制截面规模。
            symbols = self.get_universe(n=self.universe_size)

        if not symbols:
            return pd.DataFrame()

        store = get_store()
        try:
            # V6 fix (Q5 CRITICAL 前视偏差: factor_model.py:203,210):
            #   传入 end=date_str 严格截取 ≤ date_str 的行情。
            #   V5.4 版无结束日期，get() 数据截至今天 → 对任意历史 date_str
            #   返回的都是「截至今天」的因子暴露，含未来信息。
            raw = store.get_many(symbols, days=260, end=date_str)  # 260 days for 12-month momentum
        except RuntimeError:
            raw = {}
        for sym in symbols:
            if sym not in raw:
                try:
                    # V6 fix: 同 get_many，end=date_str 防前视
                    df = store.get(sym, days=260, end=date_str)
                    if not df.empty:
                        raw[sym] = df
                except Exception as e:
                    logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                    continue
        dfs = {k: v for k, v in raw.items() if not v.empty}
        if not dfs:
            return pd.DataFrame()
        # 记录底层行情数据截至日（写 Parquet 缓存元数据，供 is_cache_safe 校验）
        try:
            data_through = max(
                pd.to_datetime(v["date"]).max() for v in dfs.values()
            ).strftime("%Y-%m-%d")
        except Exception:
            data_through = date_str
        # P1-Q5-fix (Q5 HIGH: factor_model.py:219-222): V5.4 用 dict 前 100 个键截断——
        # 键序任意 → 截面不稳定、universe_size 形同虚设、结果不可复现。
        # 改为：按流通市值代理（最后收盘价 × 最后成交量 × 100）确定性降序取前
        # self.universe_size 只，并告警说明实际截面规模（调用方知晓，不静默）。
        max_stocks = self.universe_size
        if len(dfs) > max_stocks:
            def _size_proxy(sym_df: pd.DataFrame) -> float:
                try:
                    c = float(sym_df["close"].iloc[-1])
                    if "volume" in sym_df.columns:
                        v = float(sym_df["volume"].iloc[-1])
                    else:
                        v = 0.0
                    return c * v * 100
                except Exception:
                    return 0.0
            keys = sorted(
                dfs, key=lambda s: _size_proxy(dfs[s]), reverse=True
            )[:max_stocks]
            dfs = {k: dfs[k] for k in keys}
            _log.warning(
                "[FactorModel] compute_factors(%s): 截面截断到 %d 只（共 %d），"
                "已按市值/流动性代理确定性排序选取",
                date_str, max_stocks, len(raw),
            )

        factors: list[dict[str, float]] = []
        factor_index: list[str] = []

        for sym, df in dfs.items():
            if df.empty or len(df) < 20:
                continue
            f = self._compute_single_factor(sym, df, date_str)
            if f:
                factors.append(f)
                factor_index.append(sym)

        if not factors:
            return pd.DataFrame()

        result = pd.DataFrame(factors, index=factor_index)

        # Remove infinite/NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        # Winsorize + standardize per factor
        # P1-Q5-fix: 常数列/全缺失列直接剔除，不再填 0.0 —— 恒 0 因子留在模型里会让
        #   截面排序失真（无财务数据的股票被误判为"极便宜/零增长"）。
        drop_cols = []
        for col in result.columns:
            valid = result[col].notna()
            if valid.sum() < 3 or result[col].nunique() <= 1:
                drop_cols.append(col)
                continue
            result.loc[valid, col] = _winsorize(result.loc[valid, col])
            result.loc[valid, col] = _standardize(result.loc[valid, col])
            result[col] = result[col].fillna(0.0)
        if drop_cols:
            result = result.drop(columns=drop_cols)

        # Industry neutralization (if we had industry data)
        # V6 fix (Q5 HIGH: factor_model.py:96-115,244-246): 中性化从未生效——
        #   neutralize=True 之前被静默忽略（注释 "For now, just standardize"）。
        #   V6: 标准化后对 log(市值)+行业哑变量回归取残差（A股必做市值/行业中性化）。
        if neutralize and len(result) >= 10:
            result = self._apply_neutralization(result, list(result.index))

        # V4.1 feature: 持久化因子暴露到Parquet
        # V6 fix: 同时写入 data_through 元数据，杜绝旧缓存（含未来数据）被复用
        try:
            self.factor_store.save(date_str, result, data_through=data_through)
        except Exception as e:
            logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)

        self._factor_cache[date_str] = result
        return result

    def _compute_single_factor(
        self,
        symbol: str,
        df: pd.DataFrame,
        date_str: str,
    ) -> dict[str, float] | None:
        """Compute factors for one symbol.
        D4收敛登记: 与factor_zoo同名异口径(mom_*/beta_60d/roe/roa/gross_margin/net_margin/current_ratio/debt_equity/asset_turn/earnings_growth_qoq/roe_change/margin_change/surprise/ep/sales_growth_yoy)-不强迁保留；其余因子独特保留
        """
        # V4.1 arch fix: 检查单因子缓存
        cache_key = (date_str, symbol)
        if cache_key in self._single_factor_cache:
            return dict(self._single_factor_cache[cache_key])
        if "close" not in df.columns or len(df) < 60:
            return None

        close = df["close"].values
        volume = df.get("volume", pd.Series([0] * len(df))).values if "volume" in df else None
        outstanding_share = df["outstanding_share"].values if "outstanding_share" in df.columns else None
        if "pct_chg" in df.columns:
            pct_chg = df["pct_chg"].values
        else:
            pct_chg = pd.Series(close).pct_change().fillna(0).values

        result: dict[str, float] = {}

        # ── Value factors ──
        # P1-Q5-fix: 全部经 _gfv_nan 取数——缺失→NaN（不进截面），不再用 0.0 冒充；
        #   ps_ttm/pcf_ttm 不再重复调用 get_latest_value（每因子只查一次 DB）。
        result["pe_ttm"] = _gfv_nan(symbol, "pe_ttm")
        result["pb"] = _gfv_nan(symbol, "pb")
        result["ps_ttm"] = _gfv_nan(symbol, "ps_ttm")
        result["pcf_ttm"] = _gfv_nan(symbol, "pcf_ttm")
        result["div_yield"] = _gfv_nan(symbol, "div_yield")

        # ── Momentum ──
        result["mom_1m"] = self._calc_ret(close, 21)
        result["mom_3m"] = self._calc_ret(close, 63)
        result["mom_6m"] = self._calc_ret(close, 126)
        result["mom_12m"] = self._calc_ret(close, 252)
        result["excess_ret_20d"] = self._calc_excess_ret(pct_chg, 20)

        # ── Quality ──
        # P1-Q5-fix: 缺失 → NaN（不进截面），禁止用 12 个月价格涨幅冒充 ROE
        #   （财务缺失时质量因子沦为动量因子，含义错乱）。
        result["roe"] = _gfv_nan(symbol, "roe")
        result["roa"] = _gfv_nan(symbol, "roa")
        result["gross_margin"] = _gfv_nan(symbol, "gross_margin")
        debt_ratio_raw = _gfv_nan(symbol, "debt_ratio")
        result["debt_ratio"] = debt_ratio_raw * 0.01 if not np.isnan(debt_ratio_raw) else float("nan")  # % → ratio
        result["ocf_to_sales"] = _gfv_nan(symbol, "ocf_to_sales")

        # ── Size ──
        # V4.1 arch fix: 使用真实流通市值，回退到 volume*close*100 近似
        # 2026-08-10 审计: 传入本地当日流通股本（outstanding_share），消除历史截面前视
        cap_yuan, ln_cap = self._get_market_cap(symbol, close, volume, outstanding_share, as_of=date_str)
        result["ln_cap"] = ln_cap
        result["ln_float_cap"] = ln_cap  # 流通市值本身已经是流通口径，ln_float_cap=ln_cap


        # ── Volatility ──
        result["daily_vol_20d"] = float(np.std(pct_chg[-20:])) * 100
        result["idio_vol_20d"] = result["daily_vol_20d"] * 0.7
        # Fetch market index returns for proper beta calculation
        # Stock beta should measure systematic risk vs the market, not vs itself.
        market_rets = self._get_market_returns(len(pct_chg), date_str=date_str)
        result["beta_60d"] = self._calc_beta(pct_chg[-60:], market_rets[-60:]) if len(pct_chg) >= 60 and len(market_rets) >= 60 else 1.0
        result["max_dd_60d"] = self._calc_max_dd(close[-60:]) * 100 if len(close) >= 60 else 0.0

        # ── Growth ──
        # P1-Q5-fix: 财务键名对齐（debt_equity_ratio/asset_turnover 才是 DB 中的键，
        #   V5.4 的 debt_equity/asset_turn 恒为 0.0）；缺失 → NaN（不进截面）；
        #   禁止用价格涨幅代理营收/利润增速。
        result["sales_growth_yoy"] = _gfv_nan(symbol, "sales_growth_yoy")
        result["profit_growth_yoy"] = _gfv_nan(symbol, "profit_growth_yoy")
        result["eps"] = _gfv_nan(symbol, "eps")
        result["ep"] = _gfv_nan(symbol, "ep")
        result["net_margin"] = _gfv_nan(symbol, "net_margin")
        result["current_ratio"] = _gfv_nan(symbol, "current_ratio")
        result["debt_equity"] = _gfv_nan(symbol, "debt_equity_ratio")
        result["asset_turn"] = _gfv_nan(symbol, "asset_turnover")
        result["earnings_growth_qoq"] = _gfv_nan(symbol, "earnings_growth_qoq")
        result["roe_change"] = _gfv_nan(symbol, "roe_change")
        result["margin_change"] = _gfv_nan(symbol, "margin_change")
        # Surprise: 净利润增速vs营收增速偏差（任一缺失 → NaN，禁止 0 冒充）
        sg = result.get("sales_growth_yoy")
        pg = result.get("profit_growth_yoy")
        if pd.notna(sg) and pd.notna(pg) and abs(sg) > 0:
            result["surprise"] = pg - sg
        else:
            result["surprise"] = float("nan")

        # ── Microstructure: Amihud 非流动性 (V8⁺) ──
        # Amihud = |R| / 成交额(元) * 1e6, 越大越不流动
        if volume is not None and len(close) >= 20 and len(volume) >= 20:
            c20 = close[-20:]
            v20 = volume[-20:]
            ret_abs = np.abs(np.diff(c20) / c20[:-1])
            amt = c20[:-1] * v20[:-1] * 100  # 元
            amt[amt < 1] = 1
            illq = np.mean(ret_abs / amt) * 1e9
            result["amihud_illiq"] = float(np.clip(illq, 0, 100))
        else:
            result["amihud_illiq"] = 0.0

        # ── Microstructure: HAR-RV 波动率 (V8⁺) ──
        if len(pct_chg) >= 60:
            rv_d = pct_chg[-1] ** 2 * 10000  # daily realized var
            rv_w = np.mean(pct_chg[-5:] ** 2) * 10000 if len(pct_chg) >= 5 else 0
            rv_m = np.mean(pct_chg[-22:] ** 2) * 10000 if len(pct_chg) >= 22 else 0
            # HAR-RV regression coefficient (simplified: OLS on RV_d ~ RV_w + RV_m)
            # HAR-RV prediction using fixed Corsi (2009) weights
            # har_rv = 0.2*RV_d + 0.3*RV_w + 0.5*RV_m (typical parameters)
            # Avoids unstable 22-observation OLS with only 60 data points
            har_rv = 0.2 * rv_d + 0.3 * rv_w + 0.5 * rv_m
            result["har_rv_pred"] = float(np.clip(har_rv, 0, 100))
            result["rv_daily"] = float(rv_d)
            result["rv_weekly"] = float(rv_w)
            result["rv_monthly"] = float(rv_m)
        else:
            result["har_rv_pred"] = 0.0
            result["rv_daily"] = 0.0
            result["rv_weekly"] = 0.0
            result["rv_monthly"] = 0.0

        # V4.1 arch fix: 存储单因子缓存
        self._single_factor_cache[cache_key] = result
        return result

    @staticmethod
    def _calc_ret(close: np.ndarray, n: int) -> float:
        if len(close) < n + 1:
            return 0.0
        return float(close[-1] / close[-n - 1] - 1) * 100

    @staticmethod
    def _calc_excess_ret(pct_chg: np.ndarray, n: int) -> float:
        if len(pct_chg) < n:
            return 0.0
        return float(np.mean(pct_chg[-n:]) - np.mean(pct_chg[-252:])) * 100

    def _get_market_returns(self, n_days: int, date_str: str | None = None) -> np.ndarray:
        """Fetch market index (沪深300) daily returns for beta calculation.

        审计 2026-08-16：历史 date_str 必须传 end=date_str，禁止把未来行情纳入 Beta。
        """
        try:
            from quant_system.data_store import get_store
            store = get_store()
            # Use CSI 300 (000300) as market benchmark
            if date_str:
                df = store.get("000300", days=n_days + 10, end=date_str)
            else:
                df = store.get("000300", days=n_days + 10)
            if df is not None and not df.empty and "pct_chg" in df.columns:
                return df["pct_chg"].dropna().values.astype(np.float64)
        except Exception as e:
            logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
        # Fallback: return zeros (beta will default to 0 / 0 → 1.0)
        return np.zeros(max(n_days, 1), dtype=np.float64)

    @staticmethod
    def _calc_beta(stock_ret: np.ndarray, market_ret: np.ndarray) -> float:
        n = min(len(stock_ret), len(market_ret))
        if n < 5:
            return 1.0
        s = stock_ret[-n:]
        m = market_ret[-n:]
        cov = np.cov(s, m)[0, 1]
        var_m = np.var(m)
        return float(cov / var_m) if var_m > 0 else 1.0

    @staticmethod
    def _calc_max_dd(close: np.ndarray) -> float:
        peak = np.maximum.accumulate(close)
        dd = (close - peak) / peak
        return float(np.min(dd))

    @staticmethod
    def _calc_growth(close: np.ndarray, n: int) -> float:
        if len(close) < n + n // 2:
            return 0.0
        recent = np.mean(close[-n // 2:])
        prev = np.mean(close[-n:-n // 2])
        if prev <= 0:
            return 0.0
        return float(recent / prev - 1) * 100

    # V4.1 arch fix: 真实流通市值获取，从akshare获取，失败时回退到volume*close*100近似
    @staticmethod
    def _get_market_cap(symbol: str, close: np.ndarray, volume: np.ndarray | None,
                        outstanding_share: np.ndarray | None = None,
                        as_of: str | None = None) -> tuple[float, float]:
        """获取流通市值（元）和对数市值。

        优先级（2026-08-10 审计修复 Critical 前视）:
          1. 本地当日流通股本 outstanding_share（kline parquet 列）→ close×股本（逐日精确、无前视）
          2. akshare 当前流通市值 —— 仅当无本地股本时使用（历史截面回填今日市值 = 前视，已废弃优先）
          3. volume * close * 100 近似

        Returns:
            (circulating_market_cap_yuan, ln_cap)
        """
        import logging
        _log = logging.getLogger(__name__)
        # 1) 本地当日流通股本（无前视，历史截面正确）
        if outstanding_share is not None and len(outstanding_share):
            os_arr = np.asarray(outstanding_share, dtype=float)
            os_arr = os_arr[~np.isnan(os_arr)]
            if len(os_arr):
                avg_price = float(np.mean(close[-20:])) if len(close) else 0.0
                cap_yuan = float(np.mean(os_arr[-20:])) * avg_price
                if cap_yuan > 0:
                    _log.debug("_get_market_cap %s: 本地股本市值=%.2f亿", symbol, cap_yuan / 1e8)
                    return cap_yuan, math.log(max(cap_yuan, 1))
        try:
            # 审计 2026-08-16：历史截面禁止用“当前”流通市值快照
            if as_of is not None and as_of < datetime.now().strftime("%Y-%m-%d"):
                raise RuntimeError("历史截面禁用当前市值快照")
            import akshare as ak
            info = ak.stock_individual_info_em(symbol=symbol)
            if info is not None and not info.empty:
                # info 返回格式为 DataFrame 两列: item / value
                # '流通市值' 字段单位为亿元
                cap_row = info[info["item"] == "流通市值"]
                if not cap_row.empty:
                    cap_yuan = float(cap_row.iloc[0]["value"]) * 1e8  # 亿→元
                    _log.debug("_get_market_cap %s: akshare流通市值=%.2f亿元", symbol, cap_yuan / 1e8)
                    return cap_yuan, math.log(max(cap_yuan, 1))
        except Exception as e:
            logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
        # 3) Fallback: use volume * close * 100 as rough proxy
        avg_price = np.mean(close[-20:])
        avg_vol = np.mean(volume[-20:]) if volume is not None else 1e6
        proxy_cap = avg_price * avg_vol * 100
        _log.debug("_get_market_cap %s: fallback proxy_cap=%.2e", symbol, proxy_cap)
        return proxy_cap, math.log(max(proxy_cap, 1))

    # ── 因子IC ──────────────────────────────────────────────────────────

    def _apply_neutralization(
        self, factor_df: pd.DataFrame, symbols: list[str]
    ) -> pd.DataFrame:
        """截面中性化：因子对 log(市值) + 行业哑变量回归取残差，再标准化。

        V6 fix (Q5 HIGH: factor_model.py:96-115,244-246): 修复 neutralize=True
        被静默忽略的问题（V5.4 注释 "For now, just standardize"）。
        - 市值：用因子面板内已计算的 ln_cap 列（_compute_single_factor 产出）；
        - 行业：优先 sector_rotation.get_stock_industry_map()（网络，失败/为空时
          仅做市值中性化）；
        - sklearn 不可用时回退为原样返回（告警，不静默）。
        """
        if "ln_cap" not in factor_df.columns:
            return factor_df
        try:
            import sklearn  # noqa: F401
        except Exception:
            _log.warning("[FactorModel] sklearn 不可用，跳过中性化（仅标准化）")
            return factor_df

        ind_map: dict = {}
        try:
            from quant_system.sector_rotation import get_stock_industry_map
            ind_map = get_stock_industry_map() or {}
        except Exception:
            ind_map = {}

        X_df = pd.DataFrame(index=factor_df.index)
        X_df["ln_cap"] = factor_df["ln_cap"]
        if ind_map:
            ind_series = pd.Series(ind_map).reindex(factor_df.index)
            dummies = pd.get_dummies(ind_series.astype(str), prefix="ind", dummy_na=False)
            for c in dummies.columns:
                X_df[c] = dummies[c].astype(float)

        out = factor_df.copy()
        for col in factor_df.columns:
            if col in ("ln_cap", "ln_float_cap"):
                continue  # 规模因子本身不中性化
            s = factor_df[col]
            valid = s.notna()
            if valid.sum() < 10:
                continue
            X = X_df.loc[valid].values
            y = s[valid].values
            if X.shape[1] < 1 or X.shape[0] < X.shape[1] + 5:
                continue
            try:
                from sklearn.linear_model import HuberRegressor
                model = HuberRegressor(epsilon=1.35, max_iter=2000).fit(X, y)
                resid = pd.Series(y - model.predict(X), index=factor_df.index[valid])
                std = resid.std()
                if std is None or pd.isna(std) or std == 0:
                    continue
                out.loc[valid, col] = (resid - resid.mean()) / std
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                continue
        return out

    def compute_ic(
        self,
        date_str: str,
        forward_days: int = 3,  # 3 days forward (enough for recent data)
        symbols: list[str] | None = None,
    ) -> pd.Series:
        """Compute factor IC for a single date.

        IC = cross-sectional Spearman rank correlation between
        factor exposure and forward return.
        """
        # Compute factors (V6 fix: 因子暴露用 ≤ date_str 的数据，无前视)
        factor_df = self.compute_factors(date_str, symbols)
        if factor_df.empty:
            return pd.Series()

        # Forward returns
        store = get_store()
        fwd_returns: list[float] = []
        valid_symbols: list[str] = []
        n_insufficient = 0

        for sym in factor_df.index:
            try:
                # V6 fix (Q5 CRITICAL: factor_model.py:504-538):
                #   V5.4 用 store.get(sym, days=forward_days+5)（截至今天、窗口起点
                #   由 now() 决定）：① 历史 date_str 不在窗口内 → 全部被跳过；
                #   ② 默认 date_str=今天 时 idx+forward_days >= len(dates) 恒成立
                #   → IC 恒为空 Series。
                #   V6: 从 date_str 起取数；基准日 = 首个 ≥ date_str 的交易日；
                #   前向收益 = 基准日之后第 forward_days 个交易日的收益
                #   （date_str 之后无足够数据时显式告警「数据不足」，而非静默空）。
                df = store.get(sym, start=date_str.replace("-", ""), days=forward_days + 5)
                if df.empty or len(df) < forward_days + 1:
                    continue
                base_idx = None
                for i, d in enumerate(pd.to_datetime(df["date"])):
                    if d.strftime("%Y-%m-%d") >= date_str:
                        base_idx = i
                        break
                if base_idx is None or base_idx + forward_days >= len(df):
                    n_insufficient += 1
                    continue
                base_close = float(df.iloc[base_idx]["close"])
                fwd_close = float(df.iloc[base_idx + forward_days]["close"])
                if base_close <= 0 or fwd_close <= 0:
                    continue
                fwd_returns.append(fwd_close / base_close - 1.0)
                valid_symbols.append(sym)
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                continue

        if len(valid_symbols) < 10:
            if n_insufficient:
                _log.warning(
                    "[FactorModel] compute_ic(%s, forward_days=%d): %d 只股票在 date_str "
                    "之后无足够数据（数据不足），返回空 IC —— date_str 过新或窗口不足",
                    date_str, forward_days, n_insufficient,
                )
            else:
                _log.warning(
                    "[FactorModel] compute_ic(%s): 有效股票不足 10 只，返回空 IC", date_str
                )
            return pd.Series()

        fwd_series = pd.Series(fwd_returns, index=valid_symbols)
        factor_df = factor_df.loc[valid_symbols]

        # IC per factor
        ic_values: dict[str, float] = {}
        for col in factor_df.columns:
            valid = factor_df[col].notna() & fwd_series.notna()
            if valid.sum() < 10:
                ic_values[col] = 0.0
                continue
            from scipy.stats import spearmanr
            rho, _ = spearmanr(factor_df.loc[valid, col], fwd_series[valid])
            ic_values[col] = round(float(rho), 4) if not pd.isna(rho) else 0.0

        result = pd.Series(ic_values)
        self._ic_cache[date_str] = result
        return result

    def compute_ic_series(
        self,
        start_date: str,
        end_date: str | None = None,
        step_days: int = 20,
        forward_days: int = 5,
    ) -> pd.DataFrame:
        """Compute IC over a time series.

        Returns DataFrame: index = dates, columns = factor names.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        all_ics: list[pd.Series] = []
        all_dates: list[str] = []
        # P2-Q5-fix (M436): 用真实交易日历步进（market_clock.get_trade_calendar，
        #   全局唯一交易日历源），替代 V5.4 的 timedelta(days=step_days) 日历步进。
        #   V5.4 注释自称"交易日历步进"，实际周末/节假日日期 IC 恒空被跳过，序列
        #   稀疏且与注释不符。无交易日历时降级工作日粗筛并告警（可见降级）。
        trading_dates = self._get_trading_dates(start_date[:10], end_date[:10])
        if not trading_dates:
            _log.warning(
                "[FactorModel] compute_ic_series(%s~%s): 无法获取交易日历，返回空序列",
                start_date, end_date,
            )
            return pd.DataFrame()
        for i in range(0, len(trading_dates), max(1, step_days)):
            d = trading_dates[i]
            try:
                ic = self.compute_ic(d, forward_days=forward_days)
                if not ic.empty:
                    all_ics.append(ic)
                    all_dates.append(d)
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)

        if not all_ics:
            return pd.DataFrame()
        return pd.DataFrame(all_ics, index=all_dates)

    @classmethod
    def factor_retirement_scheduler(
        cls,
        factor_matrix: pd.DataFrame,
        return_matrix: pd.DataFrame,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        """因子自动退役调度器（P2a）。

        对每个因子按自然周（W-SUN）聚合 IC，滚动检查连续
        ``RETIRE_CONSEC_WEEKS`` 周的 ICIR 与胜率。两者均低于阈值时，
        将因子标为 RETIRED。

        输入：
          factor_matrix: index=日期、columns=因子名；
          return_matrix: index=日期；若与 factor_matrix 存在同名列则逐因子
            配对，否则使用唯一 return 列与所有因子配对。
          as_of: 可选扫描截止日，缺省取 factor_matrix 最大日期。

        返回：
          {"retired": {factor: {...}}, "active": [...], "scan_date": ...}
        """
        try:
            from quant_system.ic_factors.evaluator import (
                ic_summary,
                information_coefficient,
            )
        except Exception:  # pragma: no cover - 仅在依赖导入失败时降级
            _log.warning("[FactorModel] factor_retirement_scheduler: evaluator 不可用")
            return {
                "retired": {},
                "active": list(factor_matrix.columns) if isinstance(factor_matrix, pd.DataFrame) else [],
                "scan_date": (as_of or ""),
            }

        factor_df = cls._prepare_retirement_frame(factor_matrix)
        return_df = cls._prepare_retirement_frame(return_matrix)
        if factor_df.empty or factor_df.shape[1] == 0:
            return {
                "retired": {},
                "active": [],
                "scan_date": cls._retirement_scan_date(factor_df, as_of),
            }

        scan_ts = pd.Timestamp(cls._retirement_scan_date(factor_df, as_of))
        window_start = scan_ts - pd.Timedelta(days=RETIRE_WINDOW_DAYS)

        factor_df = factor_df.loc[(factor_df.index <= scan_ts)]
        factor_df = factor_df.loc[(factor_df.index >= window_start)]
        if factor_df.empty:
            return {
                "retired": {},
                "active": [],
                "scan_date": scan_ts.strftime("%Y-%m-%d"),
            }

        common_dates = factor_df.index.intersection(return_df.index)
        if len(common_dates) == 0:
            return {
                "retired": {},
                "active": list(factor_df.columns),
                "scan_date": scan_ts.strftime("%Y-%m-%d"),
            }
        factor_df = factor_df.loc[common_dates]
        return_df = return_df.loc[common_dates]

        retired: dict[str, dict[str, Any]] = {}
        active: list[str] = []
        insufficient: dict[str, dict[str, Any]] = {}

        for factor_name in factor_df.columns:
            return_col = cls._select_retirement_return_column(
                factor_name, factor_df, return_df
            )
            if return_col is None:
                insufficient[factor_name] = {
                    "reason": "insufficient: no aligned return series",
                    "last_active_date": None,
                    "icir": None,
                    "win_rate": None,
                }
                active.append(factor_name)
                continue

            weekly_ics = cls._weekly_ic_series(
                factor_df[factor_name],
                return_df[return_col],
                information_coefficient,
            )
            valid_weekly_ics = weekly_ics.dropna()
            if len(valid_weekly_ics) < RETIRE_CONSEC_WEEKS:
                insufficient[factor_name] = {
                    "reason": (
                        f"insufficient: {len(valid_weekly_ics)} valid weeks < "
                        f"{RETIRE_CONSEC_WEEKS}"
                    ),
                    "last_active_date": None,
                    "icir": None,
                    "win_rate": None,
                }
                active.append(factor_name)
                continue

            weak = cls._find_retirement_weak_window(
                weekly_ics, ic_summary
            )
            if weak is None:
                active.append(factor_name)
                continue

            start_pos, summary = weak
            weak_start = weekly_ics.index[start_pos]
            weak_week_start = weak_start - pd.Timedelta(days=6)
            prior_dates = factor_df.index[factor_df.index < weak_week_start]
            last_active_date = (
                prior_dates.max().strftime("%Y-%m-%d")
                if len(prior_dates)
                else None
            )
            retired[factor_name] = {
                "status": "RETIRED",
                "reason": (
                    f"ICIR={summary.get('ic_ir', 0.0):.4f} < {RETIRE_ICIR_MIN} "
                    f"and win_rate={summary.get('hit_rate', 0.0):.4f} < {RETIRE_WIN_MIN} "
                    f"for {RETIRE_CONSEC_WEEKS} consecutive weeks"
                ),
                "last_active_date": last_active_date,
                "icir": round(float(summary.get("ic_ir", 0.0)), 4),
                "win_rate": round(float(summary.get("hit_rate", 0.0)), 4),
            }

        return {
            "retired": retired,
            "active": active,
            "scan_date": scan_ts.strftime("%Y-%m-%d"),
            "insufficient": insufficient,
        }

    @staticmethod
    def _prepare_retirement_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """将输入转换为 DatetimeIndex、按日期排序并去重的 DataFrame。"""
        if not isinstance(frame, pd.DataFrame):
            return pd.DataFrame()
        if frame.empty:
            return frame.copy()
        df = frame.copy()
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return pd.DataFrame()
        valid = idx.notna()
        df = df.loc[valid]
        if df.empty:
            return df
        df.index = idx[valid]
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    @staticmethod
    def _retirement_scan_date(factor_df: pd.DataFrame, as_of: str | None) -> str:
        if as_of is not None:
            ts = pd.Timestamp(as_of)
            if pd.notna(ts):
                return ts.strftime("%Y-%m-%d")
        if factor_df.empty:
            return datetime.now().strftime("%Y-%m-%d")
        return factor_df.index.max().strftime("%Y-%m-%d")

    @staticmethod
    def _select_retirement_return_column(
        factor_name: str,
        factor_df: pd.DataFrame,
        return_df: pd.DataFrame,
    ) -> str | None:
        """为因子选择配对收益列。

        优先同名列；若没有同名列但收益表只有一列，则所有因子共用该列。
        """
        if factor_name in return_df.columns:
            return factor_name
        if return_df.shape[1] == 1:
            return str(return_df.columns[0])
        return None

    @staticmethod
    def _weekly_ic_series(
        factor_series: pd.Series,
        return_series: pd.Series,
        information_coefficient,
    ) -> pd.Series:
        """按 W-SUN 聚合每周 IC。

        每周至少需要 3 个有效配对样本才计算；样本不足的周记为 NaN。
        """
        paired = pd.DataFrame(
            {"factor": factor_series, "forward_return": return_series}
        ).replace([np.inf, -np.inf], np.nan)
        paired = paired.dropna()
        if paired.empty:
            return pd.Series(dtype=float)

        weekly_ics: dict[pd.Timestamp, float] = {}
        for week_end, group in paired.groupby(pd.Grouper(freq="W-SUN")):
            if len(group) < 3:
                weekly_ics[pd.Timestamp(week_end)] = np.nan
                continue
            if len(group) < 5:
                # evaluator.information_coefficient 在样本<5 时固定返回 0.0；
                # 这里按 prompt 的 3 样本下限直接计算秩相关，保留小周的真实 IC。
                f_rank = group["factor"].rank()
                r_rank = group["forward_return"].rank()
                ic = f_rank.corr(r_rank)
            else:
                ic = information_coefficient(
                    group["factor"], group["forward_return"], method="spearman"
                )
            weekly_ics[pd.Timestamp(week_end)] = (
                float(ic) if pd.notna(ic) else np.nan
            )
        if not weekly_ics:
            return pd.Series(dtype=float)
        result = pd.Series(weekly_ics, dtype=float).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        return result

    @staticmethod
    def _find_retirement_weak_window(
        weekly_ics: pd.Series,
        ic_summary,
    ) -> tuple[int, dict[str, Any]] | None:
        """查找首个连续 RETIRE_CONSEC_WEEKS 周弱窗口。

        ``ic_summary`` 由 evaluator 提供；窗口内 ICIR 与胜率均低于阈值
        时视为弱窗口。
        """
        window = RETIRE_CONSEC_WEEKS
        for end_pos in range(window - 1, len(weekly_ics)):
            start_pos = end_pos - window + 1
            window_ics = weekly_ics.iloc[start_pos:end_pos + 1]
            if window_ics.notna().sum() < window:
                continue
            summary = ic_summary(window_ics)
            if summary.get("ic_ir", 0.0) < RETIRE_ICIR_MIN and \
                    summary.get("hit_rate", 0.0) < RETIRE_WIN_MIN:
                return start_pos, summary
        return None

    @classmethod
    def save_retirement_state(
        cls,
        retired_info: dict[str, Any],
        out_path: str | Path | None = None,
    ) -> None:
        """把本次退役扫描结果幂等合并到状态文件。

        规则：
          - 新退役因子追加并写入 retired_date=scan_date；
          - 已退役因子保留原有 retired_date；
          - 本次扫描标记为 active 的因子从 retired 状态移除。

        默认路径为 ``ROOT/generated/factor_retirement.json``。
        """
        path = Path(out_path) if out_path is not None else FACTOR_RETIREMENT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    raw_retired = payload.get("retired", {})
                    if isinstance(raw_retired, dict):
                        existing = raw_retired
            except Exception:
                existing = {}

        scan_date = str(
            retired_info.get("scan_date")
            or datetime.now().strftime("%Y-%m-%d")
        )[:10]
        current_retired = retired_info.get("retired", {}) or {}
        current_active = set(retired_info.get("active", []) or [])
        current_active.difference_update(current_retired)

        merged: dict[str, Any] = {}
        for name, info in current_retired.items():
            old = existing.get(name) if isinstance(existing.get(name), dict) else {}
            entry = cls._json_safe(info)
            entry["retired_date"] = (
                old.get("retired_date") or old.get("date") or scan_date
            )
            if old:
                entry["last_active_date"] = old.get(
                    "last_active_date", entry.get("last_active_date")
                )
            merged[name] = entry

        # 已存在但本次未明确退役的因子：默认保留历史退役状态；
        # 只有本次扫描 active 的因子会被恢复移除。
        for name, info in existing.items():
            if name in current_active:
                continue
            if name not in merged:
                merged[name] = cls._json_safe(info)

        payload = {
            "retired": merged,
            "active": sorted(current_active),
            "scan_date": scan_date,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """递归转换 numpy 标量，保证可 JSON 序列化。"""
        if isinstance(value, dict):
            return {str(k): FactorModel._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [FactorModel._json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if value is pd.NaT:
            return None
        if pd.isna(value):
            return None
        return value

    def _get_trading_dates(self, start_date: str, end_date: str) -> list[str]:
        """返回 [start_date, end_date] 内的 A 股交易日列表（YYYY-MM-DD）。

        优先用 market_clock.get_trade_calendar()（真实交易日，含节假日剔除）；
        失败时回退沪深300指数行情日期（data_store 落库，无网络）；仍失败则用
        工作日粗筛（周一到周五）并告警。
        """
        try:
            from quant_system.market_clock import get_trade_calendar
        except Exception:
            get_trade_calendar = None
        if get_trade_calendar is not None:
            try:
                cal = get_trade_calendar()
                if cal:
                    return sorted(d for d in cal if start_date <= d <= end_date)
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
        try:
            store = get_store()
            df = store.get("000300", start=start_date.replace("-", ""), days=900)
            if df is not None and not df.empty and "date" in df.columns:
                dates = sorted({str(d)[:10] for d in pd.to_datetime(df["date"])})
                out = [d for d in dates if start_date <= d <= end_date]
                if out:
                    return out
        except Exception as e:
            logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
        _log.warning(
            "[FactorModel] _get_trading_dates: 无真实交易日历，降级为工作日粗筛（含节假日误差）"
        )
        return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start_date, end_date)]

    # ── 因子组合 ────────────────────────────────────────────────────────

    def factor_portfolio_returns(
        self,
        date_str: str,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
    ) -> dict[str, float]:
        """Compute long-short portfolio return for each factor."""
        factor_df = self.compute_factors(date_str)
        if factor_df.empty:
            return {}

        store = get_store()
        result: dict[str, float] = {}
        for col in factor_df.columns:
            # V6 fix (Q5 HIGH: factor_model.py:335-341,604-628): 方向语义统一。
            #   daily_vol_20d/beta_60d/max_dd_60d 等风险因子均为正值（越大越危险），
            #   V5.4 对所有因子 sort_values(ascending=False) 取多头 → 买的是最高波动/
            #   最高Beta的股票，方向性反转。V6: 风险因子按「越小越好」取多头端。
            ascending = col in RISK_FACTORS_LOWER_IS_BETTER
            sorted_idx = factor_df[col].sort_values(ascending=ascending).index
            n = len(sorted_idx)
            n_long = max(1, int(n * long_pct))
            n_short = max(1, int(n * short_pct))
            long_symbols = sorted_idx[:n_long]
            short_symbols = sorted_idx[-n_short:]

            # Get next-day returns
            def _next_day_ret(syms: list[str]) -> float:
                rets = []
                for sym in syms:
                    try:
                        # P2-Q5-fix (M439): 按 date_str 起取次日收益（与 compute_ic 同源）。
                        #   V5.4 用 store.get(sym, days=5) 截至今天 → 对历史 date_str，
                        #   窗口内没有 date_str 之后的行情 → 多空收益恒 0.0，报表
                        #   long_short_returns 在回看日无意义。
                        df = store.get(sym, start=date_str.replace("-", ""), days=10)
                        if df.empty:
                            continue
                        # Find first date > date_str (next day return)
                        for _, row in df.iterrows():
                            if str(row["date"])[:10] > date_str:
                                ret_fwd = row.get("pct_chg", 0)
                                if pd.notna(ret_fwd):
                                    rets.append(float(ret_fwd))
                                break
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                        continue
                return np.mean(rets) if rets else 0.0

            long_ret = _next_day_ret(long_symbols.tolist())
            short_ret = _next_day_ret(short_symbols.tolist())
            result[col] = round(long_ret - short_ret, 4)

        return result

    # ── 风险模型 ────────────────────────────────────────────────────────

    def risk_model(
        self,
        date_str: str,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Barra-style risk model with Newey-West HAC covariance.

        Returns:
        - factor_cov: full covariance matrix (list of lists)
        - specific_risk: list of idiosyncratic vol per stock
        - factor_exposure: dict[symbol][factor] loading matrix
        """
        factor_df = self.compute_factors(date_str, symbols)
        if factor_df.empty or len(factor_df) < 5:
            return {"status": "insufficient_data"}

        # 1. Factor covariance (Newey-West HAC)
        nw_cov = self._newey_west_cov(factor_df, lags=5)

        # 2. Specific risk per stock — residual vol from historical returns
        # P2-Q5-fix (M438): 特异风险改为「对因子暴露回归后的残差 std」。
        #   V5.4 用 demean 后的总波动冒充特异波动——未剥离因子暴露，风险模型的特异
        #   风险实际是总波动。做法（Barra 风格）：对窗口内每个交易日做截面回归
        #   r_i(t) = α_t + Σ_f β_f(t)·x_{i,f} + ε_i(t)，取每只股票残差序列的 std。
        store = get_store()
        exposure_df = factor_df.copy()
        exposure_cols = [c for c in exposure_df.columns if c not in ("ln_cap", "ln_float_cap")]
        if not exposure_cols:
            exposure_cols = list(exposure_df.columns)
        # 暴露矩阵 + 截距列（截距吸收市场因子）
        X_all = np.hstack([
            np.ones((len(exposure_df), 1)),
            exposure_df[exposure_cols].fillna(0.0).values.astype(np.float64),
        ])
        syms = list(exposure_df.index)
        sym_pos = {s: i for i, s in enumerate(syms)}

        # 对齐窗口内各股票的日收益（不同股票交易日可能错位，用 NaN 补）
        raw_rets: dict[str, pd.DataFrame] = {}
        for sym in syms:
            try:
                df = store.get(sym, days=180)  # V4.1 fix: 增加窗口天数，降低停牌股数据不足概率
                if df is None or df.empty or "pct_chg" not in df.columns:
                    continue
                sub = df[["date", "pct_chg"]].dropna().tail(60)
                if len(sub) >= 10:
                    raw_rets[sym] = sub
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                continue
        dates_all = sorted(set().union(*[set(sub["date"]) for sub in raw_rets.values()]))
        date_idx = {d: i for i, d in enumerate(dates_all)}
        Y_mat = np.full((len(syms), len(dates_all)), np.nan)
        for sym, sub in raw_rets.items():
            i = sym_pos[sym]
            for d, r in zip(sub["date"], sub["pct_chg"]):
                if d in date_idx:
                    Y_mat[i, date_idx[d]] = float(r)

        resid_by_sym: dict[str, list[float]] = {s: [] for s in syms}
        for k in range(len(dates_all)):
            col = Y_mat[:, k]
            valid = np.isfinite(col)
            if valid.sum() < max(10, X_all.shape[1] + 5):
                continue
            Xv, yv = X_all[valid], col[valid]
            try:
                beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
                resid = yv - Xv @ beta
                for pos, s in enumerate([s for s, ok in zip(syms, valid) if ok]):
                    resid_by_sym[s].append(float(resid[pos]))
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                continue

        specific_risk_list: list[float] = []
        for sym in syms:
            rs = resid_by_sym.get(sym)
            if rs and len(rs) >= 5:
                specific_risk_list.append(float(np.std(rs, ddof=1)))
            else:
                # 数据不足 → 可见降级（沿用历史默认 0.02），不静默冒充真实残差波动
                specific_risk_list.append(0.02)

        # 3. Factor exposure matrix
        exposure_dict: dict[str, dict[str, float]] = {}
        for sym in factor_df.index:
            row = factor_df.loc[sym]
            exposure_dict[sym] = {col: float(row[col]) if not pd.isna(row[col]) else 0.0
                                  for col in factor_df.columns}

        return {
            "status": "ok",
            "n_stocks": len(factor_df),
            "n_factors": len(factor_df.columns),
            "factor_names": list(factor_df.columns),
            "factor_cov": nw_cov.values.tolist(),
            "factor_exposure": exposure_dict,
            "specific_risk": specific_risk_list,
            "date": date_str,
        }

    def _newey_west_cov(self, factor_df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
        """Newey-West HAC covariance with Bartlett kernel.

        P1-Q5-fix: V5.4 对单期截面（rows=股票）做 NW，T 取的是股票数——自协方差项
        对任意顺序排列的股票做"滞后相关"，无时间含义，lags 参数失效。修复：
        - 输入为时间序列（DatetimeIndex 或 date 列，rows=期数）时，NW-HAC 正确作用于
          因子收益率的时间序列，T=期数；
        - 输入为单期截面（rows=股票）时，NW 不适用，改用 Ledoit-Wolf 收缩协方差
          （sklearn 已用，无新依赖），并告警说明（可见降级，不静默）。
        """
        base_cov = factor_df.cov().fillna(0)
        n_f = len(base_cov)
        if n_f < 2:
            return base_cov

        # 判断是否时间序列布局
        is_timeseries = isinstance(factor_df.index, pd.DatetimeIndex) or (
            "date" in factor_df.columns
            and factor_df["date"].nunique() == len(factor_df)
        )

        if not is_timeseries:
            _log.warning(
                "[FactorModel] _newey_west_cov: 输入为单期截面（rows=股票），"
                "Newey-West 自协方差项对股票顺序无时间含义，改用 Ledoit-Wolf 收缩协方差"
            )
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(np.asarray(factor_df.fillna(0.0), dtype=np.float64))
                return pd.DataFrame(lw.covariance_,
                                    index=base_cov.index, columns=base_cov.columns)
            except Exception:
                return base_cov

        T = min(60, len(factor_df))
        if T < 10:
            return base_cov
        sample = factor_df.iloc[-T:]
        if T < lags + 3:
            return base_cov
        try:
            X = sample.values.astype(np.float64) - np.mean(sample.values, axis=0)
            nw = base_cov.values.copy()
            for k in range(1, lags + 1):
                w = 1.0 - k / (lags + 1)
                X_t, X_tk = X[k:], X[:-k]
                if len(X_t) == 0:
                    continue
                g = (X_t.T @ X_tk) / (len(X_t) - 1)
                nw += w * (g + g.T)
            from numpy.linalg import eigh
            evals, evecs = eigh((nw + nw.T) / 2)
            evals = np.maximum(evals, 1e-8)
            return pd.DataFrame(evecs @ np.diag(evals) @ evecs.T,
                                index=base_cov.index, columns=base_cov.columns)
        except Exception:
            return base_cov

    # ── 报告 ────────────────────────────────────────────────────────────

    def factor_report(self, date_str: str) -> dict[str, Any]:
        """Comprehensive factor report for a given date."""
        factor_df = self.compute_factors(date_str)
        if factor_df.empty:
            return {"date": date_str, "status": "no_data"}

        # Factor statistics
        stats = {}
        for col in factor_df.columns:
            vals = factor_df[col].dropna()
            stats[col] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "skew": round(float(vals.skew()), 4) if len(vals) > 2 else 0,
                "pct_pos": round(float((vals > 0).mean() * 100), 1),
                "n_valid": int(vals.count()),
            }

        # IC
        ic = self.compute_ic(date_str)
        ic_dict = ic.to_dict() if not ic.empty else {}

        # Group summary
        group_stats = {}
        for gname, factors in FACTOR_GROUPS.items():
            present = [f for f in factors if f in factor_df.columns]
            if not present:
                group_stats[gname] = {}
                continue
            group_df = factor_df[present]
            group_stats[gname] = {
                "n_factors": len(present),
                "mean_exposure": round(float(group_df.mean().mean()), 4),
                "mean_ic": round(
                    float(np.mean([ic_dict.get(f, 0) for f in present])), 4
                ) if ic_dict else 0,
                "factors": present,
            }

        # Portfolio returns
        ls_returns = self.factor_portfolio_returns(date_str)

        return {
            "date": date_str,
            "n_stocks": len(factor_df),
            "n_factors": len(factor_df.columns),
            "factor_stats": stats,
            "factor_ic": ic_dict,
            "group_stats": group_stats,
            "long_short_returns": ls_returns,
        }


class FactorTiming:
    """Factor timing helper for regime-specific weights, IC records, and correlations."""

    def __init__(self) -> None:
        """Initialize in-memory state and the shared SQLite cache path."""
        self.cache: dict[str, dict[str, Any]] = {}
        self.db_path = Path.home() / ".quant_system" / "factor_ic_cache.sqlite3"
        self.regime_adjustments: dict[str, dict[str, float]] = {
            "bull": {
                "mom": 1.25, "momentum": 1.25, "growth": 1.20,
                "sales_growth": 1.20, "profit_growth": 1.20,
            },
            "bear": {
                "vol": 1.25, "low_vol": 1.25, "quality": 1.20,
                "roe": 1.15, "roa": 1.15, "gross_margin": 1.15,
            },
            "range": {
                "reversal": 1.25, "rsi": 1.20, "cci": 1.20,
                "boll": 1.15, "value": 1.10, "bp": 1.10, "ep": 1.10,
            },
            "sideways": {
                "reversal": 1.25, "rsi": 1.20, "cci": 1.20,
                "boll": 1.15, "value": 1.10, "bp": 1.10, "ep": 1.10,
            },
        }

    def _ensure_db(self) -> None:
        """Create SQLite tables used by factor timing if they do not exist."""
        try:
            import sqlite3
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_ic_regime (
                    factor_name TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    date TEXT NOT NULL,
                    ic REAL,
                    PRIMARY KEY (factor_name, regime, date)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_corr_snapshot (
                    factor_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (factor_name, date)
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception:
            return

    def get_regime_weights(self, base_weights: dict[str, float], regime: str) -> dict[str, float]:
        """Adjust factor weights by market regime and normalize absolute exposure."""
        if not base_weights:
            return {}
        regime_key = regime.lower()
        if regime_key in {"neutral", "oscillation", "choppy"}:
            regime_key = "range"
        adjustments = self.regime_adjustments.get(regime_key, {})
        weights = dict(base_weights)
        for factor_name, base_weight in base_weights.items():
            factor_l = factor_name.lower()
            multiplier = 1.0
            for token, adj in adjustments.items():
                if token in factor_l:
                    multiplier = max(multiplier, adj)
            weights[factor_name] = float(base_weight) * multiplier
        denom = sum(abs(v) for v in weights.values())
        if denom <= 0:
            return weights
        return {k: float(v) / denom for k, v in weights.items()}

    def update_factor_ic_regime(self, factor_name: str, regime: str, ic: float, date: str) -> None:
        """Persist one factor IC observation under a market regime."""
        self.cache[factor_name] = {"regime": regime, "ic": float(ic), "date": date}
        try:
            import sqlite3
            self._ensure_db()
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT OR REPLACE INTO factor_ic_regime(factor_name, regime, date, ic)
                VALUES (?, ?, ?, ?)
                """,
                (factor_name, regime, date, float(ic)),
            )
            conn.commit()
            conn.close()
        except Exception:
            return

    def compute_factor_corr_matrix(self, factor_names: list[str], lookback: int = 60) -> pd.DataFrame:
        """Compute a factor correlation matrix from cached timing snapshots or IC history."""
        if not factor_names:
            return pd.DataFrame()
        try:
            import sqlite3
            self._ensure_db()
            conn = sqlite3.connect(str(self.db_path))
            placeholders = ",".join("?" for _ in factor_names)
            df = pd.read_sql_query(
                f"""
                SELECT factor_name, date, value
                FROM factor_corr_snapshot
                WHERE factor_name IN ({placeholders})
                ORDER BY date DESC
                """,
                conn,
                params=tuple(factor_names),
            )
            if df.empty:
                df = pd.read_sql_query(
                    f"""
                    SELECT factor_name, date, rank_ic AS value
                    FROM factor_ic
                    WHERE factor_name IN ({placeholders})
                    ORDER BY date DESC
                    """,
                    conn,
                    params=tuple(factor_names),
                )
            conn.close()
            if df.empty:
                return pd.DataFrame(np.eye(len(factor_names)), index=factor_names, columns=factor_names)
            wide = df.dropna().pivot_table(index="date", columns="factor_name", values="value", aggfunc="mean")
            wide = wide.sort_index().tail(int(lookback))
            corr = wide.reindex(columns=factor_names).corr().fillna(0.0)
            for name in factor_names:
                corr.loc[name, name] = 1.0
            return corr.reindex(index=factor_names, columns=factor_names).fillna(0.0)
        except Exception:
            return pd.DataFrame(np.eye(len(factor_names)), index=factor_names, columns=factor_names)


# ---------------------------------------------------------------------------
# FactorStore — V4.1 feature: Parquet-based 因子持久化存储
# ---------------------------------------------------------------------------

# P2-Q5-fix (M437): 缓存 schema 版本。计算逻辑/数据口径变化时 +1，旧缓存即失效
#   （is_cache_safe 校验 schema_version + data_through + data_hash），杜绝
#   "修好数据层后旧缓存仍按 date_str 返回错误旧值"。
_FACTOR_CACHE_SCHEMA_VERSION = 2


def _factor_df_hash(factors_df: pd.DataFrame) -> str:
    """对因子面板做内容哈希（缓存失效校验用）。"""
    try:
        vals = pd.util.hash_pandas_object(factors_df, index=True)
        return str(hash(tuple(sorted(vals.values.tolist()))))[:16]
    except Exception:
        return "unknown"


class FactorStore:
    """Parquet因子持久化存储

    V4.1 feature: 将计算完成的因子暴露数据以Parquet格式持久化到磁盘，
    避免在回测和报告中重复计算。
    存储路径: ~/.quant_system/factors/{year}/{month}/{date}.parquet
    """

    def __init__(self, base_dir: str | Path | None = None):
        """初始化存储目录

        Args:
            base_dir: 存储根目录，默认 ~/.quant_system/factors/
        """
        if base_dir is None:
            base_dir = Path.home() / ".quant_system" / "factors"
        self.base_dir = Path(base_dir)

    def _date_to_path(self, date_str: str) -> Path:
        """根据日期字符串返回Parquet文件完整路径"""
        dt = pd.Timestamp(date_str)
        return self.base_dir / str(dt.year) / f"{dt.month:02d}" / f"{date_str}.parquet"

    def save(self, date: str, factors_df: pd.DataFrame,
             data_through: str | None = None) -> None:
        """保存因子DataFrame到Parquet文件

        Args:
            date: 日期字符串 YYYY-MM-DD
            factors_df: 因子DataFrame (index=symbol, columns=factor_names)
            data_through: 底层行情数据截至日 YYYY-MM-DD（V6 fix: 写入元数据，
                          供 is_cache_safe 校验，防止含未来数据的旧缓存被复用）
        """
        if factors_df is None or factors_df.empty:
            return
        path = self._date_to_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 保存股票代码为索引，因子列名为列
        factors_df.to_parquet(path, engine="pyarrow", index=True)
        # V6 fix: 元数据侧车文件
        try:
            meta = {
                "date": date,
                "data_through": data_through or date,
                "saved_at": datetime.now().isoformat(),
                # P2-Q5-fix (M437): 缓存带内容哈希 + schema 版本——数据截止日 data_through
                #   只防"前视"；schema_version/data_hash 防"修好数据层后旧值仍被复用"。
                "schema_version": _FACTOR_CACHE_SCHEMA_VERSION,
                "data_hash": _factor_df_hash(factors_df),
            }
            Path(str(path) + ".meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)

    def is_cache_safe(self, date_str: str) -> bool:
        """V6 fix (Q5 CRITICAL 前视偏差): 校验缓存是否可用。

        仅当元数据存在、schema_version 匹配且 data_through ≤ date_str 时返回 True。
        - V5.4 之前（无元数据）或 data_through > date_str（因子暴露含未来数据）拒绝复用；
        - P2-Q5-fix (M437): schema_version 不匹配（计算口径已变）也拒绝复用，
          避免"修好数据层后旧缓存仍按 date_str 返回错误旧值"。
        """
        meta_path = Path(str(self._date_to_path(date_str)) + ".meta.json")
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            through = str(meta.get("data_through", "9999-99-99"))[:10]
            ver = int(meta.get("schema_version", 0) or 0)
            if ver != _FACTOR_CACHE_SCHEMA_VERSION:
                return False
            return through <= date_str[:10]
        except Exception:
            return False

    def load(self, start_date: str, end_date: str,
             factors: list[str] | None = None) -> pd.DataFrame:
        """加载日期范围内的因子数据

        Args:
            start_date: 起始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            factors: 可选，仅加载指定因子列的子集

        Returns:
            MultiIndex DataFrame: (date, symbol) 双层索引，columns = factor_names
            若未找到数据返回空 DataFrame
        """
        available = self.list_dates()
        if not available:
            return pd.DataFrame()

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        selected = sorted(d for d in available if start <= pd.Timestamp(d) <= end)

        frames: list[pd.DataFrame] = []
        for d in selected:
            path = self._date_to_path(d)
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path, engine="pyarrow")
                if factors:
                    existing = [c for c in factors if c in df.columns]
                    if not existing:
                        continue
                    df = df[existing]
                if df.empty:
                    continue
                df.index.name = "symbol"
                df["date"] = d
                frames.append(df)
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_model] 操作失败: {e}", exc_info=True)
                continue

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=0)
        result = result.set_index("date", append=True).reorder_levels(["date", "symbol"])
        return result

    def list_dates(self) -> list[str]:
        """列出所有已存储的因子日期（按字典序排序）"""
        if not self.base_dir.exists():
            return []
        dates: list[str] = []
        for year_dir in sorted(self.base_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dates.append(f.stem)
        return dates


# ---------------------------------------------------------------------------
# Singleton convenience
# ---------------------------------------------------------------------------

_model: FactorModel | None = None


def get_model() -> FactorModel:
    global _model
    if _model is None:
        _model = FactorModel()
    return _model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    model = get_model()

    if cmd == "report":
        date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y-%m-%d")
        print(f"Factor report for {date}...")
        r = model.factor_report(date)
        print(f"  Stocks: {r.get('n_stocks', 0)}, Factors: {r.get('n_factors', 0)}")
        print(f"  分组IC:")
        for g, s in r.get("group_stats", {}).items():
            print(f"    {g:>12}: {s.get('mean_ic', 0):+.4f}")
        print(f"  Top IC factors:")
        ic = r.get("factor_ic", {})
        for name, val in sorted(ic.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"    {name:>20}: {val:+.4f}")

    elif cmd == "ic":
        start = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
        ic_df = model.compute_ic_series(start)
        print(f"IC series: {len(ic_df)} dates, {len(ic_df.columns)} factors")
        if not ic_df.empty:
            print(ic_df.mean().sort_values(ascending=False).head(10).to_string())

    elif cmd == "stats":
        date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y-%m-%d")
        r = model.factor_report(date)
        for name, s in r.get("factor_stats", {}).items():
            print(f"{name:>20}: mean={s['mean']:>8.4f}  std={s['std']:>8.4f}  skew={s['skew']:>8.4f}  pos={s['pct_pos']:>5.1f}%")

    else:
        print(f"Usage: {sys.argv[0]} [report|ic|stats] [date]")
