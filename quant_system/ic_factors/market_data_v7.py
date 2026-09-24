"""
market_data_v7.py — V7.0 市场级数据抓取层（衍生品/宏观/指数/另类）
=====================================================================
- 全部走 cache + usage + 熔断（DataLoaderV7._fetch 模式）
- 返回 dict[str, pd.DataFrame]，供衍生品/宏观/指数/另类因子模块消费
- 每个接口独立 try/except，单点失败不影响整体
- 数据源: akshare（国内无限量）
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 市场数据加载(衍生品/宏观/指数)独特保留。
"""
from __future__ import annotations


import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.data_loader_v7 import DataLoaderV7

log = get_logger("qv6.market_data_v7")

# 股指期货主力合约（新浪）
FUT_IF = "IF0"   # 沪深300
FUT_IC = "IC0"   # 中证500
FUT_IM = "IM0"   # 中证1000
FUT_IH = "IH0"   # 上证50
FUT_T = "T0"     # 10年期国债期货
FUT_LH = "LH0"   # 生猪期货

# 现货指数代码（新浪）
SPOT_IF = "sh000300"
SPOT_IC = "sh000905"
SPOT_IM = "sh000852"
SPOT_IH = "sh000016"

# 乐咕指数 PE 的 symbol 名
PE_SYMBOLS = ["上证50", "沪深300", "中证500", "中证1000", "创业板指"]


class MarketDataV7(DataLoaderV7):
    """市场级数据装配器（衍生品/宏观/指数/另类）。"""

    def __init__(self):
        super().__init__()

    # ── 通用抓取 ──────────────────────────────────────────
    def _fetch(self, source: str, key: str, ttl_hours: float, fetcher,
               *args, **kwargs) -> pd.DataFrame:
        return super()._fetch(source, key, ttl_hours, fetcher, *args, **kwargs)

    # ── 工具：日期解析 ────────────────────────────────────
    @staticmethod
    def _norm_date(idx) -> pd.DatetimeIndex:
        try:
            return pd.to_datetime(idx, errors="coerce")
        except Exception:  # noqa: BLE001
            return pd.DatetimeIndex([])

    # ── 1. 股指期货基差（现货-期货）/现货指数 ─────────────
    def _load_futures_basis(self, n_days: int = 400) -> pd.DataFrame:
        import akshare as ak
        rows = []
        pairs = [(FUT_IF, SPOT_IF, "IF"), (FUT_IC, SPOT_IC, "IC"),
                 (FUT_IM, SPOT_IM, "IM"), (FUT_IH, SPOT_IH, "IH")]
        for fut, spot, tag in pairs:
            try:
                df_f = self._fetch("akshare", f"fut_main_{fut}", 24,
                                   ak.futures_main_sina, symbol=fut)
                df_s = self._fetch("akshare", f"index_daily_{spot}", 24,
                                   ak.stock_zh_index_daily, symbol=spot)
                if df_f is None or df_s is None or df_f.empty or df_s.empty:
                    continue
                df_f = df_f.copy()
                df_f["日期"] = pd.to_datetime(df_f["日期"])
                df_s = df_s.copy()
                df_s["date"] = pd.to_datetime(df_s["date"])
                m = df_f.merge(df_s[["date", "close"]], left_on="日期", right_on="date")
                m = m.tail(n_days)
                for _, r in m.iterrows():
                    spot_px, fut_px = float(r["close"]), float(r["收盘价"])
                    if spot_px <= 0:
                        continue
                    rows.append({
                        "date": r["日期"], "contract": tag,
                        "spot": spot_px, "fut": fut_px,
                        "basis_rate": (spot_px - fut_px) / spot_px,
                    })
            except Exception as e:  # noqa: BLE001
                log.warning(f"期货基差 {tag} 失败: {e}")
        return pd.DataFrame(rows)

    # ── 2. 国债期货主力 ───────────────────────────────────
    def _load_bond_futures(self, n_days: int = 400) -> pd.DataFrame:
        import akshare as ak
        try:
            df = self._fetch("akshare", "fut_main_T0", 24,
                             ak.futures_main_sina, symbol=FUT_T)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            df["ret"] = df["收盘价"].pct_change()
            return df.tail(n_days).reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"国债期货失败: {e}")
            return pd.DataFrame()

    # ── 3. 期权波动率 QVIX（中国波指）────────────────────
    def _load_qvix(self) -> pd.DataFrame:
        import akshare as ak
        out = pd.DataFrame()
        for tag, fn in [("50etf", ak.index_option_50etf_qvix),
                        ("300etf", ak.index_option_300etf_qvix)]:
            try:
                df = self._fetch("akshare", f"qvix_{tag}", 24, fn)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                df["symbol"] = tag
                out = pd.concat([out, df], ignore_index=True)
            except Exception as e:  # noqa: BLE001
                log.warning(f"QVIX {tag} 失败: {e}")
        return out

    # ── 4. 可转债快照（双低）──────────────────────────────
    def _load_cb_spot(self) -> pd.DataFrame:
        import akshare as ak
        try:
            # 转债比价表（含转股溢价率）
            df = self._fetch("akshare_cb", "cb_comparison", 2, ak.bond_cov_comparison)
            if df is not None and not df.empty:
                df = df.copy()
                rename = {}
                for c in df.columns:
                    if "转股溢价率" in c:
                        rename[c] = "premium_rt"
                    elif "转债价格" in c or c == "trade":
                        rename[c] = "trade"
                df = df.rename(columns=rename)
                for c in ["trade", "premium_rt"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                if "trade" in df.columns and "premium_rt" in df.columns:
                    df["double_low"] = df["trade"] + df["premium_rt"]
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"转债比价表失败，退回行情快照: {e}")
        try:
            df = self._fetch("akshare_cb", "cb_spot", 2, ak.bond_zh_hs_cov_spot)
            if df is not None and not df.empty:
                df = df.copy()
                if "trade" in df.columns:
                    df["trade"] = pd.to_numeric(df["trade"], errors="coerce")
                # 无溢价率时双低退化为价格（仅价格维度）
                if "double_low" not in df.columns and "trade" in df.columns:
                    df["double_low"] = df["trade"]
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"可转债快照失败: {e}")
        return pd.DataFrame()

    # ── 5. 回购利率（FR007/FDR007，Shibor 1W 兜底）────────
    def _load_repo_rate(self) -> pd.DataFrame:
        import akshare as ak
        import datetime as _dt
        try:
            # V10 审计 M1：end/start 动态化（原硬编码 20261231/20250101）
            end = _dt.date.today().strftime("%Y%m%d")
            start = (_dt.date.today() - _dt.timedelta(days=365 * 5)).strftime("%Y%m%d")
            df = self._fetch("akshare_repo", "repo_rate_hist", 24,
                             ak.repo_rate_hist, start_date=start,
                             end_date=end)
            if df is not None and not df.empty:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date")
        except Exception as e:  # noqa: BLE001
            log.warning(f"回购利率失败，转 Shibor 1W: {e}")
        try:
            df = self._fetch("akshare", "shibor_1w", 24, ak.rate_interbank,
                             market="上海银行同业拆借市场", symbol="Shibor人民币",
                             indicator="1周")
            if df is not None and not df.empty:
                df = df.copy()
                df.columns = ["date", "FR007", "chg"]
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Shibor 1W 兜底失败: {e}")
        return pd.DataFrame()

    # ── 6. Shibor ─────────────────────────────────────────
    def _load_shibor(self) -> pd.DataFrame:
        import akshare as ak
        try:
            df = self._fetch("akshare", "shibor_3m", 24, ak.rate_interbank,
                             market="上海银行同业拆借市场", symbol="Shibor人民币",
                             indicator="3月")
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df.columns = ["date", "rate", "chg"]
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Shibor 失败: {e}")
            return pd.DataFrame()

    # ── 7. 宏观月度（PMI/CPI/PPI/M2/社融）────────────────
    def _load_macro_monthly(self) -> pd.DataFrame:
        import akshare as ak
        out: dict[str, pd.DataFrame] = {}
        specs = [
            ("pmi", ak.macro_china_pmi),
            ("cpi", ak.macro_china_cpi),
            ("ppi", ak.macro_china_ppi),
            ("m2", ak.macro_china_m2_yearly),
            ("shrzgm", ak.macro_china_shrzgm),
            ("exports", ak.macro_china_exports_yoy),
            ("imports", ak.macro_china_imports_yoy),
        ]
        for tag, fn in specs:
            try:
                df = self._fetch("akshare", f"macro_{tag}", 72, fn)
                if df is not None and not df.empty:
                    out[tag] = df
            except Exception as e:  # noqa: BLE001
                log.warning(f"宏观 {tag} 失败: {e}")
        return out

    # ── 8. LPR / 美债 / 汇率 ─────────────────────────────
    def _load_rates(self) -> pd.DataFrame:
        import akshare as ak
        out = {}
        try:
            df = self._fetch("akshare", "lpr", 72, ak.macro_china_lpr)
            if df is not None and not df.empty:
                out["lpr"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"LPR 失败: {e}")
        try:
            df = self._fetch("akshare", "us_rate", 24, ak.bond_zh_us_rate)
            if df is not None and not df.empty:
                out["us_rate"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"美债失败: {e}")
        try:
            df = self._fetch("akshare", "fx_usdcny", 24,
                             ak.currency_boc_sina, symbol="美元")
            if df is not None and not df.empty:
                out["fx"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"汇率失败: {e}")
        return out

    # ── 9. 商品（原油/生猪期货）与 BDI ───────────────────
    def _load_commodity(self) -> pd.DataFrame:
        import akshare as ak
        out = {}
        for tag, fut in [("crude", "SC0"), ("lh", FUT_LH)]:
            try:
                df = self._fetch("akshare", f"fut_main_{fut}", 24,
                                 ak.futures_main_sina, symbol=fut)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["日期"] = pd.to_datetime(df["日期"])
                    out[tag] = df
            except Exception as e:  # noqa: BLE001
                log.warning(f"商品期货 {fut} 失败: {e}")
        try:
            df = self._fetch("akshare", "bdi", 24, ak.macro_shipping_bdi)
            if df is not None and not df.empty:
                out["bdi"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"BDI 失败: {e}")
        return out

    # ── 10. 指数估值分位（乐咕 PE）───────────────────────
    def _load_index_pe(self) -> pd.DataFrame:
        import akshare as ak
        out = pd.DataFrame()
        for sym in PE_SYMBOLS:
            try:
                df = self._fetch("akshare", f"index_pe_{sym}", 24,
                                 ak.stock_index_pe_lg, symbol=sym)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["日期"] = pd.to_datetime(df["日期"])
                df["symbol"] = sym
                out = pd.concat([out, df], ignore_index=True)
            except Exception as e:  # noqa: BLE001
                log.warning(f"指数PE {sym} 失败: {e}")
        return out

    # ── 11. 市场 PE / 拥挤度 / 巴菲特指标 ────────────────
    def _load_market_heat(self) -> pd.DataFrame:
        import akshare as ak
        out = {}
        try:
            df = self._fetch("akshare", "market_pe", 24, ak.stock_market_pe_lg)
            if df is not None and not df.empty:
                out["market_pe"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"市场PE失败: {e}")
        try:
            df = self._fetch("akshare", "congestion", 24, ak.stock_a_congestion_lg)
            if df is not None and not df.empty:
                out["congestion"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"拥挤度失败: {e}")
        try:
            df = self._fetch("akshare", "buffett", 72, ak.stock_buffett_index_lg)
            if df is not None and not df.empty:
                out["buffett"] = df
        except Exception as e:  # noqa: BLE001
            log.warning(f"巴菲特指标失败: {e}")
        return out

    # ── 12. 申万一级行业估值 ─────────────────────────────
    def _load_sw_industry(self) -> pd.DataFrame:
        import akshare as ak
        try:
            df = self._fetch("akshare", "sw_first_info", 24, ak.sw_index_first_info)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            for c in ["静态市盈率", "TTM(滚动)市盈率", "市净率"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"申万行业失败: {e}")
            return pd.DataFrame()

    # ── 13. 另类：票房（实时→日榜双保险）────────────────
    def _load_boxoffice(self) -> pd.DataFrame:
        import akshare as ak
        import datetime as dt
        try:
            df = self._fetch("akshare_movie", "boxoffice_realtime", 24,
                             ak.movie_boxoffice_realtime)
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"票房实时失败: {e}")
        try:
            y = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
            df = self._fetch("akshare_movie", f"boxoffice_daily_{y}", 24,
                             ak.movie_boxoffice_daily, date=y)
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"票房日榜失败: {e}")
        return pd.DataFrame()

    # ── 总入口 ────────────────────────────────────────────
    def load(self, keys: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """keys: futures_basis/bond_futures/qvix/cb_spot/repo_rate/shibor/
                 macro_monthly/rates/commodity/index_pe/market_heat/
                 sw_industry/boxoffice
        """
        all_keys = keys or ["futures_basis", "bond_futures", "qvix", "cb_spot",
                            "repo_rate", "shibor", "macro_monthly", "rates",
                            "commodity", "index_pe", "market_heat",
                            "sw_industry", "boxoffice"]
        data: dict[str, pd.DataFrame | dict] = {}
        for k in all_keys:
            fn = getattr(self, f"_load_{k}", None)
            if fn is None:
                log.warning(f"未知市场数据域: {k}")
                continue
            try:
                data[k] = fn()
            except Exception as e:  # noqa: BLE001
                log.warning(f"市场数据域 {k} 加载失败: {e}")
                data[k] = pd.DataFrame()
        return data


def get_market_data(keys: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """便捷入口。"""
    return MarketDataV7().load(keys)
