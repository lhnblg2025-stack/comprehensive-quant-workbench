"""
data_loader_v7.py — V7.0 因子数据加载层
========================================
把 warehouse/akshare/baostock 数据组装成因子 handler 需要的 data dict。
全部走缓存层（TTL: 日频24h / 盘中15min / 财报季1周），用量记账。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 数据加载(仓库/baostock)独特保留。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_system.market_forecast._support.common.cache import get_cache
from quant_system.market_forecast._support.common.usage import get_tracker
from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.data_loader")


@dataclass
class DataLoaderV7:
    """因子数据装配器。用法:
    loader = DataLoaderV7()
    data = loader.load({"profit", "growth", "north", "margin"})
    """

    cache = None
    tracker = None

    def __post_init__(self):
        self.cache = get_cache()
        self.tracker = get_tracker()

    # ── 通用抓取封装（缓存 + 用量 + 熔断）────────────────

    def _fetch(self, source: str, key: str, ttl_hours: float, fetcher,
               *args, **kwargs) -> pd.DataFrame:
        if not self.tracker.is_allowed(source):
            log.info(f"[{source}] 预算/熔断拦截，走缓存")
            hit = self.cache.get(key)
            return hit if hit is not None else pd.DataFrame()
        try:
            df = self.cache.cached_fetch(key, ttl_hours, fetcher, *args,
                                         source=source, **kwargs)
            self.tracker.record(source)
            self.tracker.record_success(source)
            return df
        except Exception as e:  # noqa: BLE001
            self.tracker.record_failure(source, str(e))
            log.warning(f"[{source}] 抓取失败 {e}")
            hit = self.cache.get(key)
            return hit if hit is not None else pd.DataFrame()

    # ── 各数据域装配 ──────────────────────────────────────

    def load(self, keys: list[str], codes: list[str] | None = None,
             years: list[int] | None = None) -> dict[str, pd.DataFrame]:
        """keys: profit/growth/balance/cashflow/financial/kline/valuation/north/margin/block/lhb/
                holders/pledge/unlock/etf/mainflow/industry
        """
        data: dict[str, pd.DataFrame] = {}
        for k in keys:
            fn = getattr(self, f"_load_{k}", None)
            if fn is None:
                log.warning(f"未知数据域: {k}")
                continue
            try:
                data[k] = fn(codes or [], years or list(range(2023, 2027)))
            except Exception as e:  # noqa: BLE001
                log.warning(f"数据域 {k} 加载失败: {e}")
                data[k] = pd.DataFrame()
        return data

    # 基本面（baostock，季频）
    def _load_profit(self, codes, years):
        from quant_system.ic_factors.fundamental_v7 import fetch_quarterly_profit
        key = self.cache.make_key("bs_profit", "|".join(codes[:50]), max(years))
        return self._fetch("baostock", key, 24 * 7, fetch_quarterly_profit, codes, years)

    def _load_growth(self, codes, years):
        from quant_system.ic_factors.fundamental_v7 import fetch_quarterly_growth
        key = self.cache.make_key("bs_growth", "|".join(codes[:50]), max(years))
        return self._fetch("baostock", key, 24 * 7, fetch_quarterly_growth, codes, years)

    def _load_balance(self, codes, years):
        from quant_system.ic_factors.fundamental_v7 import fetch_quarterly_balance
        key = self.cache.make_key("bs_balance", "|".join(codes[:50]), max(years))
        return self._fetch("baostock", key, 24 * 7, fetch_quarterly_balance, codes, years)

    def _load_cashflow(self, codes, years):
        from quant_system.ic_factors.fundamental_v7 import fetch_quarterly_cashflow
        key = self.cache.make_key("bs_cashflow", "|".join(codes[:50]), max(years))
        return self._fetch("baostock", key, 24 * 7, fetch_quarterly_cashflow, codes, years)

    # 财务分析指标（新浪，季频，86列）
    def _load_financial(self, codes, years):
        import akshare as ak
        frames = []
        for code in codes[:60]:
            try:
                key = self.cache.make_key("fin_ind", code, max(years))
                df = self._fetch("akshare", key, 24 * 7,
                                 ak.stock_financial_analysis_indicator,
                                 symbol=code, start_year=str(max(years) - 3))
                if df is not None and not df.empty:
                    df = df.copy()
                    df["code"] = code
                    frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning(f"财务指标 {code} 失败: {e}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ── K线（东财日K → 新浪日K 回退）────────────────────
    def _load_kline(self, codes, years):
        """返回 {code: DataFrame(date,open,close,high,low,volume,amount,turnover)}。"""
        import akshare as ak
        import datetime as _dt
        out: dict[str, pd.DataFrame] = {}
        # V10 审计 M1 修复：结束日期动态化（原硬编码 20261231，跨年即断）
        end = _dt.date.today().strftime("%Y%m%d")
        start = f"{max(years) - 2}0101"
        for code in codes[:120]:
            df = None
            try:
                key = self.cache.make_key("kline", code, start)
                df = self._fetch("kline_em", key, 24,
                                 ak.stock_zh_a_hist, symbol=code,
                                 period="daily", start_date=start,
                                 end_date=end, adjust="qfq")
            except Exception as e:  # noqa: BLE001
                log.warning(f"K线东财 {code} 失败: {e}")
            if df is None or df.empty:
                try:
                    # 回退：新浪日K（需交易所前缀）
                    prefix = "sh" if code.startswith(("6", "9")) else \
                             "bj" if code.startswith(("4", "8")) else "sz"
                    key2 = self.cache.make_key("kline_sina", code, start)
                    df = self._fetch("kline_sina", key2, 24,
                                     ak.stock_zh_a_daily,
                                     symbol=f"{prefix}{code}",
                                     start_date=start, end_date=end,
                                     adjust="qfq")
                except Exception as e:  # noqa: BLE001
                    log.warning(f"K线新浪 {code} 失败: {e}")
            if df is None or df.empty:
                continue
            df = df.copy()
            # 东财中文列 → 英文标准列
            rename = {"日期": "date", "开盘": "open", "收盘": "close",
                      "最高": "high", "最低": "low", "成交量": "volume",
                      "成交额": "amount", "换手率": "turnover"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            for c in ["date", "open", "close", "high", "low",
                      "volume", "amount", "turnover"]:
                if c in df.columns:
                    if c != "date":
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    else:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.sort_values("date").dropna(subset=["date"])
            if not df.empty:
                out[code] = df
        return out

    # ── 估值历史（baostock，日频）────────────────────────
    def _load_valuation(self, codes, years):
        import baostock as bs
        import datetime as _dt
        frames = []
        fields = "date,close,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
        start = f"{max(years) - 3}-01-01"
        end = _dt.date.today().strftime("%Y-%m-%d")  # V10 审计 M1：动态化

        def _fetch_bs(bscode: str) -> pd.DataFrame:
            rs = bs.query_history_k_data_plus(
                bscode, fields, start_date=start, end_date=end,
                frequency="d", adjustflag="2")
            rows = []
            while (rs.error_code == "0") and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=rs.fields)

        bs.login()
        try:
            for code in codes[:120]:
                bscode = "sh." + code if code.startswith(("6", "9")) else \
                         "bj." + code if code.startswith(("4", "8")) else "sz." + code
                try:
                    key = self.cache.make_key("val_bs", code, start)
                    df = self._fetch("baostock", key, 24 * 7,
                                     _fetch_bs, bscode)
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["code"] = code
                        frames.append(df)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"估值 {code} 失败: {e}")
        finally:
            bs.logout()
        if not frames:
            return pd.DataFrame()
        # 统一列名层：旧格式(peTTM/pbMRQ/psTTM/pcfNcfTTM) → 标准(pe_ttm/pb/ps/pcf)
        from quant_system.ic_factors.valuation_v7 import normalize_valuation_columns
        return normalize_valuation_columns(pd.concat(frames, ignore_index=True))

    # 资金行为（akshare，日频）
    def _load_north(self, codes, years):
        import akshare as ak
        key = self.cache.make_key("north_hsgt")
        def _f():
            return ak.stock_hsgt_hist_em(symbol="北向资金")
        return self._fetch("akshare", key, 24, _f)

    def _load_margin(self, codes, years):
        import akshare as ak
        import datetime as _dt
        key = self.cache.make_key("margin", "|".join(codes[:50]))
        def _f():
            # V10 审计 M1：动态化（原硬编码 20260701，数据永远陈旧）
            d = _dt.date.today().strftime("%Y%m%d")
            df = ak.stock_margin_detail_szse(date=d)
            return df
        return self._fetch("akshare", key, 24, _f)

    def _load_block(self, codes, years):
        import akshare as ak
        key = self.cache.make_key("block_trade")
        def _f():
            return ak.stock_dzjy_mrmx(symbol="全部")
        return self._fetch("akshare", key, 24, _f)

    def _load_lhb(self, codes, years):
        import akshare as ak
        import datetime as _dt
        key = self.cache.make_key("lhb")
        def _f():
            # V10 审计 M1：动态化（原硬编码 20260701~20260804）
            today = _dt.date.today()
            start = (today - _dt.timedelta(days=60)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            return ak.stock_lhb_detail_em(start_date=start, end_date=end)
        return self._fetch("akshare", key, 24, _f)

    def _load_holders(self, codes, years):
        import akshare as ak
        key = self.cache.make_key("holders", "|".join(codes[:20]))
        def _f():
            out = []
            for c in codes[:20]:
                try:
                    out.append(ak.stock_zh_a_gdhs_detail_em(symbol=c))
                except Exception as e:  # noqa: BLE001
                    log.error(f"[data_loader_v7] 操作失败: {e}", exc_info=True)
                    continue
            return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        return self._fetch("akshare", key, 24 * 7, _f)

    def _load_pledge(self, codes, years):
        import akshare as ak
        key = self.cache.make_key("pledge")
        def _f():
            return ak.stock_pledge_ratio_detail_em()
        return self._fetch("akshare", key, 24, _f)

    def _load_unlock(self, codes, years):
        import akshare as ak
        key = self.cache.make_key("unlock")
        def _f():
            return ak.stock_restricted_release_queue_em(symbol="全部")
        return self._fetch("akshare", key, 24, _f)

    def _load_etf(self, codes, years):
        import akshare as ak
        import datetime as _dt
        key = self.cache.make_key("etf_510300")
        def _f():
            # V10 审计 M1：动态化
            today = _dt.date.today()
            start = (today - _dt.timedelta(days=30)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            return ak.fund_etf_hist_em(symbol="510300", period="daily",
                                       start_date=start, end_date=end,
                                       adjust="")
        return self._fetch("akshare", key, 24, _f)

    def _load_mainflow(self, codes, years):
        import akshare as ak
        key = self.cache.make_key("mainflow", "|".join(codes[:20]))
        def _f():
            out = []
            for c in codes[:20]:
                try:
                    out.append(ak.stock_individual_fund_flow(stock=c, market="sh" if c.startswith("6") else "sz"))
                except Exception as e:  # noqa: BLE001
                    log.error(f"[data_loader_v7] 操作失败: {e}", exc_info=True)
                    continue
            return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        return self._fetch("akshare", key, 24, _f)

    def _load_industry(self, codes, years):
        import akshare as ak
        import datetime as _dt
        key = self.cache.make_key("industry_sw")
        def _f():
            # V10 审计 M1：动态化
            today = _dt.date.today()
            start = (today - _dt.timedelta(days=60)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            return ak.stock_board_industry_hist_em(symbol="小金属",
                                                   start_date=start,
                                                   end_date=end,
                                                   period="日k", adjust="")
        return self._fetch("akshare", key, 24, _f)
