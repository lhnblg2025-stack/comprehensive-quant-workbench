from __future__ import annotations

import importlib
import sqlite3
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd


class TestQ1DataQualityIntegration(unittest.TestCase):
    """Q1-H04/H05: data_quality import path and singleton DB lifetime."""

    def test_data_quality_imports_from_package_context(self):
        module = importlib.import_module("quant_system.data_quality")
        self.assertTrue(hasattr(module, "get_trade_calendar"))

    def test_get_delisted_stocks_keeps_singleton_connection_open(self):
        import quant_system.data_quality as dq

        old_conn = dq._QUALITY_DB_CONN
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE quality_cache (key TEXT PRIMARY KEY, data TEXT, updated_at TEXT)"
        )
        dq._QUALITY_DB_CONN = conn
        try:
            with patch.dict("sys.modules", {"akshare": None}):
                with patch("quant_system.data_quality.requests.get", side_effect=RuntimeError("offline")):
                    self.assertEqual(dq.get_delisted_stocks(), [])
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
            dq._QUALITY_DB_CONN = old_conn


class TestQ2DataPipelineHighRisk(unittest.TestCase):
    """Q2-H01..H05: 数据管线高危缺口闭环。"""

    def test_daily_kline_write_includes_pct_chg_column(self):
        import quant_system.data_pipeline as dp

        # V3: _KLINE_DB_CONN 已重构为 _KLINE_DB_LOCAL（threading.local + conns dict 按路径独立连接）
        old_conns = getattr(dp._KLINE_DB_LOCAL, "conns", None)
        dp._KLINE_DB_LOCAL.conns = {}
        try:
            df = dp._standardize_daily_kline(
                pd.DataFrame(
                    {
                        "日期": ["2026-08-01"],
                        "开盘": [10],
                        "收盘": [11],
                        "最高": [12],
                        "最低": [9],
                        "成交量": [10000],
                        "成交额": [110000],
                        "涨跌幅": [1.23],
                    }
                ),
                "600519",
            )
            self.assertEqual(dp._write_daily_kline(df, db_path=":memory:"), 1)
            conn = dp._get_kline_db(":memory:")
            cols = [row[1] for row in conn.execute("PRAGMA table_info(daily_kline)").fetchall()]
            self.assertIn("pct_chg", cols)
            row = conn.execute("SELECT pct_change, pct_chg FROM daily_kline").fetchone()
            self.assertEqual(row, (1.23, 1.23))
        finally:
            dp._KLINE_DB_LOCAL.conns = old_conns

    def test_sources_all_akshare_tx_amount_lots_becomes_volume_not_amount(self):
        from quant_system.sources_all import _normalize

        out = _normalize(
            pd.DataFrame(
                {
                    "date": ["2026-08-01"],
                    "open": [10],
                    "close": [11],
                    "high": [12],
                    "low": [9],
                    "amount": [71873],
                }
            ),
            "akshare",
        )
        self.assertIn("volume", out.columns)
        self.assertNotIn("amount", out.columns)
        self.assertEqual(float(out.loc[0, "volume"]), 7187300.0)

    def test_data_fallback_does_not_call_ambiguous_akshare_tx(self):
        import quant_system.data as data

        class FakeAk:
            def stock_zh_a_hist_tx(self, **kwargs):
                raise AssertionError("stock_zh_a_hist_tx should not be used")

            def stock_zh_a_hist(self, **kwargs):
                return pd.DataFrame(
                    {
                        "日期": ["2026-08-01"],
                        "开盘": [10],
                        "收盘": [11],
                        "最高": [12],
                        "最低": [9],
                        "成交量": [100],
                        "成交额": [100000],
                    }
                )

        with patch("quant_system.sources_all.fetch_daily_unified", side_effect=RuntimeError("offline")):
            with patch.dict("sys.modules", {"akshare": FakeAk()}):
                df = data._fetch_daily_with_fallbacks("600519", "20260801", "20260802", "qfq", retries=1)
        self.assertIn("成交量", df.columns)

    def test_fetch_daily_unified_receives_adjust_parameter(self):
        import quant_system.data as data

        calls = []

        def fake_unified(symbol, start, end, adjust, timeout):
            calls.append(adjust)
            return pd.DataFrame(
                {
                    "日期": ["2026-08-01"],
                    "开盘": [10],
                    "收盘": [11],
                    "最高": [12],
                    "最低": [9],
                    "成交量": [100],
                    "成交额": [100000],
                }
            )

        with patch("quant_system.sources_all.fetch_daily_unified", side_effect=fake_unified):
            data._fetch_daily_with_fallbacks("600519", "20260801", "20260802", "hfq", retries=1)
        self.assertEqual(calls, ["hfq"])

    def test_tushare_qfq_uses_pro_bar_not_unadjusted_daily(self):
        from quant_system import sources_all

        calls = []

        class FakePro:
            def daily(self, **kwargs):
                raise AssertionError("unadjusted pro.daily should not be used for qfq")

        def fake_pro_api(token):
            return FakePro()

        def fake_pro_bar(**kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {
                    "trade_date": ["20260801"],
                    "open": [10],
                    "high": [12],
                    "low": [9],
                    "close": [11],
                    "vol": [100],
                    "amount": [123],
                }
            )

        fake_ts = types.SimpleNamespace(pro_api=fake_pro_api, pro_bar=fake_pro_bar)
        fake_open = mock_open(read_data='{"tushare_pro":{"api_key":"token"}}')
        with patch.dict("sys.modules", {"tushare": fake_ts}):
            with patch("builtins.open", fake_open):
                out = sources_all._fetch_tushare("600519", "20260801", "20260802", adjust="qfq")
        self.assertEqual(calls[0]["adj"], "qfq")
        self.assertEqual(calls[0]["ts_code"], "600519.SH")
        self.assertEqual(str(out.loc[0, "date"])[:10], "2026-08-01")

    def test_etf_tencent_fields_are_turnover_volume_ratio_and_amount_yuan(self):
        import quant_system.data_pipeline as dp

        parts = [""] * 50
        parts[1] = "50ETF"
        parts[3] = "3.03"
        parts[4] = "3.00"
        parts[6] = "12345"
        parts[32] = "1.00"
        parts[37] = "456.7"
        parts[38] = "9.32"
        parts[49] = "0.82"

        class Resp:
            text = "v_sh510050=\"" + "~".join(parts) + "\";"

        with patch("quant_system.data_pipeline.requests.get", return_value=Resp()):
            with patch("quant_system.data_pipeline.log_data_fetch"):
                data = dp._fetch_etf()["data"]["上证50ETF"]
        self.assertEqual(data["amount"], 4567000.0)
        self.assertEqual(data["turnover"], 9.32)
        self.assertEqual(data["volume_ratio"], 0.82)
        self.assertNotIn("nav", data)
        self.assertNotIn("total_shares_wan", data)

    def test_fetch_incremental_qfq_refetches_full_history(self):
        import quant_system.data_pipeline as dp

        calls = []

        def fake_fetch(symbol, start_date=None):
            calls.append(start_date)
            return pd.DataFrame(
                {
                    "symbol": [symbol],
                    "date": ["2026-08-01"],
                    "open": [10],
                    "close": [11],
                    "high": [12],
                    "low": [9],
                    "volume": [10000],
                    "amount": [110000],
                }
            )

        with patch("quant_system.data_pipeline.get_last_kline_date", return_value="2026-07-31"):
            with patch("quant_system.data_pipeline._fetch_akshare_daily", side_effect=fake_fetch):
                with patch("quant_system.data_pipeline._write_daily_kline", return_value=1):
                    with patch("quant_system.data_pipeline.log_data_fetch"):
                        dp.fetch_incremental("600519")
        self.assertEqual(calls, [None])

    def test_sina_daily_uses_valid_marketdata_service_name(self):
        import quant_system.data_pipeline as dp

        seen = []

        class Resp:
            text = 'var _data=([{"day":"2026-08-01","open":"10","close":"11","high":"12","low":"9","volume":"100"}])'

            def raise_for_status(self):
                return None

        def fake_get(url, **kwargs):
            seen.append(url)
            return Resp()

        with patch("quant_system.data_pipeline.requests.get", side_effect=fake_get):
            dp._fetch_sina_daily("600519")
        self.assertIn("CN_MarketData.getKLineData", seen[0])
        self.assertNotIn("CN_MarketDataService.getKLineData", seen[0])

    def test_market_money_flow_single_total_column_is_not_doubled(self):
        import quant_system.data_pipeline as dp

        # V3: _MONEY_FLOW_DB_CONN 已重构为 _MONEY_FLOW_DB_LOCAL（threading.local）
        old_conn = getattr(dp._MONEY_FLOW_DB_LOCAL, "conn", None)
        dp._MONEY_FLOW_DB_LOCAL.conn = sqlite3.connect(":memory:")
        dp._MONEY_FLOW_DB_LOCAL.conn.executescript(
            """
            CREATE TABLE market_money_flow (
                date TEXT PRIMARY KEY,
                sh_main_net REAL,
                sz_main_net REAL,
                total_main_net REAL,
                updated_at TEXT
            );
            """
        )

        class FakeAk:
            def stock_market_fund_flow(self):
                return pd.DataFrame({"日期": ["2026-08-01"], "主力净流入-净额": [100.0]})

        try:
            with patch.dict("sys.modules", {"akshare": FakeAk()}):
                out = dp.download_market_money_flow(days=5)
            self.assertEqual(float(out.loc[0, "total_main_net"]), 100.0)
            self.assertTrue(pd.isna(out.loc[0, "sh_main_net"]))
            self.assertTrue(pd.isna(out.loc[0, "sz_main_net"]))
        finally:
            if dp._MONEY_FLOW_DB_LOCAL.conn is not None:
                dp._MONEY_FLOW_DB_LOCAL.conn.close()
            dp._MONEY_FLOW_DB_LOCAL.conn = old_conn

    def test_stock_money_flow_ratio_uses_net_components_when_buy_sell_missing(self):
        import quant_system.data_pipeline as dp

        raw = pd.DataFrame(
            {
                "日期": ["2026-08-01", "2026-08-02"],
                "主力净流入-净额": [100.0, 50.0],
                "超大单净流入-净额": [60.0, 20.0],
                "大单净流入-净额": [40.0, 30.0],
                "中单净流入-净额": [-20.0, -10.0],
                "小单净流入-净额": [-80.0, -40.0],
            }
        )
        out = dp._standardize_stock_money_flow(raw, "600519", days=20)
        self.assertEqual(float(out["super_large_net"].sum()), 80.0)
        self.assertEqual(float(out["large_net"].sum()), 70.0)

        ratio = float(out["main_net"].sum()) / dp._money_flow_abs_base(out) * 100.0
        self.assertAlmostEqual(ratio, 150.0 / 300.0 * 100.0)
        self.assertLess(ratio, 100.0)


class TestQ3MacroAndFinancialHighRisk(unittest.TestCase):
    """Q3-H01..H04: macro/financial data contract fixes."""

    def test_macro_all_uses_real_single_metric_keys_and_visible_missing_api_error(self):
        import quant_system.macro_calendar as mc

        class FakeAk:
            def macro_china_pmi(self):
                return pd.DataFrame(
                    {
                        "月份": ["2026年07月份", "2026年06月份"],
                        "制造业-指数": [50.3, 49.7],
                    }
                )

        with patch.dict("sys.modules", {"akshare": FakeAk()}):
            all_macro = mc.get_all_macro()

        self.assertIn("PMI", all_macro)
        self.assertEqual(float(all_macro["PMI"]["latest_value"]), 50.3)
        self.assertIn("PPI", all_macro)
        self.assertIn("无接口 macro_china_ppi", all_macro["PPI"]["error"])
        self.assertNotIn("CPI/PPI", all_macro)
        self.assertNotIn("社融/M2", all_macro)

    def test_financial_derived_indicators_are_persisted_and_hot_path_stable(self):
        import quant_system.financial_data as fd

        class FakeAk:
            def stock_financial_abstract(self, symbol):
                return pd.DataFrame(
                    {
                        "指标": ["基本每股收益", "每股净资产", "每股营业收入", "每股经营现金流", "净利润", "净资产收益率(ROE)", "毛利率"],
                        "20260331": [-0.5, 5.0, 3.0, 2.0, 120.0, 11.0, 32.0],
                        "20251231": [1.2, 4.5, 12.0, 8.0, 500.0, 10.0, 30.0],
                        "20250331": [0.4, 4.0, 2.0, 1.0, 100.0, 9.0, 29.0],
                    }
                )

            def stock_fhps_detail_em(self, symbol):
                return pd.DataFrame(
                    {
                        "除权除息日": ["2026-07-01"],
                        "方案进度": ["实施"],
                        "现金分红-现金分红比例描述": ["10派2.5元"],
                        "现金分红-现金分红比例": [2.5],
                    }
                )

        class FakeStore:
            def get(self, symbol, days=5):
                return pd.DataFrame({"close": [10.0]})

        with tempfile.TemporaryDirectory() as td:
            old_path = fd.DB_PATH
            fd.DB_PATH = Path(td) / "financial.db"
            try:
                with patch.dict("sys.modules", {"akshare": FakeAk()}):
                    with patch("quant_system.data_store.get_store", return_value=FakeStore()):
                        cold = fd.fetch_financial_indicators("600519", force=True)
                self.assertNotIn("pe_ttm", cold)
                self.assertNotIn("ep", cold)
                self.assertEqual(float(cold["pb"]), 2.0)
                self.assertAlmostEqual(float(cold["earnings_growth_qoq"]), 20.0)
                self.assertAlmostEqual(float(cold["roe_change"]), 2.0)

                self.assertEqual(fd.get_latest_value("600519", "pb"), 2.0)
                # W2.5 修复：缺失因子返回 None（与"真实 0"区分），不再返回 0.0
                self.assertIsNone(fd.get_latest_value("600519", "pe_ttm"))

                hot = fd.fetch_financial_indicators("600519", force=False, max_age_days=9999)
                self.assertIn("pb", hot)
                self.assertIn("div_yield", hot)
                self.assertIn("earnings_growth_qoq", hot)
                self.assertNotIn("pe_ttm", hot)

                matrix = fd.get_all_for_date(["600519"])
                self.assertIn("pb", matrix.columns)
                self.assertIn("earnings_growth_qoq", matrix.columns)
                self.assertNotIn("pe_ttm", matrix.columns)

                empty_matrix = fd.get_all_for_date([])
                self.assertTrue(empty_matrix.empty)
                self.assertEqual(list(empty_matrix.index), [])
            finally:
                fd.DB_PATH = old_path

    def test_trade_refresh_prices_handles_sqlite_row_highest_price(self):
        import quant_system.trade_db as trade_db

        with tempfile.TemporaryDirectory() as td:
            old_path = trade_db._DB_PATH
            trade_db._DB_PATH = Path(td) / "trade_log.db"
            try:
                trade_db.init_db()
                trade_db.add_buy("600519", "贵州茅台", 100, 10.0)

                def fake_fetch_quotes(symbols):
                    return [{"symbol": "600519", "price": 12.0}]

                with patch("quant_system.watchlist.fetch_quotes", side_effect=fake_fetch_quotes):
                    self.assertEqual(trade_db.refresh_prices(), 1)

                pos = trade_db.get_position("600519")
                self.assertEqual(float(pos["current_price"]), 12.0)
                self.assertEqual(float(pos["highest_price"]), 12.0)
                self.assertEqual(float(pos["pnl_pct"]), 20.0)
            finally:
                trade_db._DB_PATH = old_path

    def test_task_timeout_does_not_auto_retry_while_thread_is_alive(self):
        import quant_system.task_queue as tq

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tasks.db"
            old_db = tq._DB
            old_timeout = tq.TASK_TIMEOUT
            old_handlers = tq._HANDLERS.copy()
            release = threading.Event()
            started = threading.Event()

            def slow_handler(params):
                started.set()
                release.wait(2)
                return {"ok": True}

            try:
                tq._DB = db_path
                tq.TASK_TIMEOUT = 0.05
                tq._HANDLERS.clear()
                tq.register_handler("slow", slow_handler)
                tq.init(db_path)
                with sqlite3.connect(str(db_path)) as conn:
                    conn.execute(
                        "INSERT INTO tasks (id, kind, params, status, created, max_retries) VALUES (?, ?, ?, 'running', ?, ?)",
                        ("task-timeout", "slow", "{}", 1.0, 3),
                    )

                tq._execute_with_timeout("task-timeout", "slow", {})
                self.assertTrue(started.is_set())

                with sqlite3.connect(str(db_path)) as conn:
                    status, retry_count, next_retry_at, error = conn.execute(
                        "SELECT status, retry_count, next_retry_at, error FROM tasks WHERE id=?",
                        ("task-timeout",),
                    ).fetchone()
                self.assertEqual(status, "timed_out")
                self.assertEqual(retry_count, 1)
                self.assertIsNone(next_retry_at)
                self.assertIn("automatic retry disabled", error)
                self.assertEqual(tq.resubmit_failed(), 0)
            finally:
                release.set()
                tq._RUNNING = False
                tq._DB = old_db
                tq.TASK_TIMEOUT = old_timeout
                tq._HANDLERS.clear()
                tq._HANDLERS.update(old_handlers)

    def test_fundamental_safe_float_blocks_unit_strings_and_false_pollution(self):
        import quant_system.fundamental as fundamental

        class FakeAk:
            def stock_financial_abstract_ths(self, symbol):
                return pd.DataFrame(
                    {
                        "报告期": ["2026-06-30"],
                        "营业总收入": ["823.20亿"],
                        "营业总收入同比增长率": ["12.5%"],
                        "净利润": [False],
                        "净利润同比增长率": ["False"],
                        "净资产收益率": ["15.2%"],
                        "销售毛利率": ["--"],
                    }
                )

        with patch.dict("sys.modules", {"akshare": FakeAk()}):
            out = fundamental.fetch_financials("600519")

        self.assertEqual(out["source"], "akshare_ths")
        self.assertEqual(out["revenue"], 82320000000.0)
        self.assertEqual(out["revenue_yoy"], 12.5)
        self.assertEqual(out["roe"], 15.2)
        self.assertNotIn("profit", out)
        self.assertNotIn("profit_yoy", out)
        self.assertNotIn("gross_margin", out)

        text = fundamental.format_fundamentals(
            {
                "symbol": "600519",
                "name": "贵州茅台",
                "revenue_yoy": "False",
                "profit_yoy": "1720.54亿",
                "roe": "15.2%",
                "gross_margin": False,
            }
        )
        self.assertIn("ROE: 15.20%", text)
        self.assertNotIn("营收增速", text)

        with patch("quant_system.watchlist.get_watchlist", return_value=[{"symbol": "600519", "name": "贵州茅台", "market_cap_yi": 1000}]):
            with patch("quant_system.fundamental.fetch_financials", return_value={"symbol": "600519", "roe": "15.2%", "revenue_yoy": "False", "profit_yoy": "8%"}):
                self.assertEqual(fundamental.screener(min_roe=10, min_revenue_growth=5, min_profit_growth=5), [])

    def test_margin_szse_uses_compact_yyyymmdd_dates(self):
        import quant_system.margin as margin

        seen_sz_dates = []
        today_compact = margin.date.today().strftime("%Y%m%d")

        class FakeAk:
            def stock_margin_sse(self, start_date=None, end_date=None):
                return pd.DataFrame(
                    {
                        "信用交易日期": [today_compact, "20260731"],
                        "融资余额": [100000000000.0, 99000000000.0],
                    }
                )

            def stock_margin_szse(self, date=None):
                seen_sz_dates.append(date)
                # 真实接口返回多列（日期+融资余额+融券余额等）；实现要求 len(columns)>=2
                return pd.DataFrame({"日期": [date], "融资余额": [5000.0], "融券余额": [100.0]})

        with patch.dict("sys.modules", {"akshare": FakeAk()}):
            out = margin.fetch_margin_summary()

        self.assertTrue(out["ok"])
        self.assertGreaterEqual(len(seen_sz_dates), 1)
        self.assertTrue(all(isinstance(d, str) and len(d) == 8 and d.isdigit() for d in seen_sz_dates))
        self.assertTrue(all("-" not in d for d in seen_sz_dates))

    def test_margin_sse_short_balance_uses_share_unit_not_amount_unit(self):
        import quant_system.margin as margin

        query_date = margin.date.today().strftime("%Y%m%d")

        class FakeAk:
            def stock_margin_detail_sse(self, date=None):
                return pd.DataFrame(
                    {
                        "日期": [query_date],
                        "标的证券代码": ["600519"],
                        "融资余额": [1000000000.0],
                        "融券余量": [2300000.0],
                    }
                )

            def stock_margin_detail_szse(self, date=None):
                return pd.DataFrame()

        with patch.dict("sys.modules", {"akshare": FakeAk()}):
            out = margin.fetch_margin_individual("600519")

        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "sse")
        self.assertEqual(out["rqye_unit"], "亿股")
        self.assertAlmostEqual(out["rqye"], 0.02)

    def test_financial_report_growth_score_uses_real_growth_inputs_and_skips_missing(self):
        from quant_system.fundamental_analysis import FinancialReportGenerator

        base_financials = {
            "ratios": {
                "roe": 12.0,
                "roa": 6.0,
                "gross_margin": 0.35,
                "net_margin": 0.12,
                "current_ratio": 2.2,
                "debt_to_equity": 0.4,
                "operating_cf_margin": 0.15,
            },
            "income": {"net_profit": 100.0},
            "balance": {"total_liabilities": 50.0, "cash": 20.0},
            "net_profit": 100.0,
            "revenue": 1000.0,
            "equity": 500.0,
            "total_assets": 800.0,
        }

        generator = FinancialReportGenerator()
        generator.generate("600519", {**base_financials, "revenue_growth": 0.35, "profit_growth": 0.40}, price=10.0, shares=100.0)
        self.assertEqual(generator.health.scores["growth"], 50)
        high_growth_total = generator.health.total_score()

        generator.generate("600519", {**base_financials, "revenue_growth": -0.05, "profit_growth": -0.02}, price=10.0, shares=100.0)
        self.assertEqual(generator.health.scores["growth"], 0)
        self.assertLess(generator.health.total_score(), high_growth_total)

        report = generator.generate("600519", base_financials, price=10.0, shares=100.0)
        self.assertNotIn("growth", generator.health.scores)
        self.assertNotIn("growth", generator.health.max_scores)
        self.assertIn("成长: 跳过", report)

    def test_comparable_multiples_drop_non_positive_denominators(self):
        from quant_system.fundamental_analysis import ComparableValuation, FinancialReportGenerator

        multiples = ComparableValuation.compute_multiples(
            {
                "net_profit": -5.0,
                "equity": -20.0,
                "revenue": 0.0,
                "total_assets": 100.0,
            },
            price=10.0,
            shares=100.0,
        )
        self.assertEqual(multiples["market_cap"], 1000.0)
        self.assertIsNone(multiples["pe"])
        self.assertIsNone(multiples["pb"])
        self.assertIsNone(multiples["ps"])
        self.assertEqual(multiples["market_cap_to_assets"], 10.0)

        peers = [
            {"pe": None, "pb": None, "ps": None},
            {"pe": -4.0, "pb": -2.0, "ps": 0.0},
            {"pe": 8.0, "pb": 1.2, "ps": 2.0},
            {"pe": 12.0, "pb": 1.8, "ps": 3.0},
        ]
        comparison = ComparableValuation.peer_comparison({"pe": 10.0, "pb": None, "ps": 2.5}, peers)
        self.assertEqual(comparison["metric"].tolist(), ["pe", "ps"])
        self.assertAlmostEqual(comparison.loc[comparison["metric"] == "pe", "peer_mean"].iloc[0], 10.0)
        self.assertAlmostEqual(comparison.loc[comparison["metric"] == "ps", "peer_median"].iloc[0], 2.5)

        report = FinancialReportGenerator().generate(
            "600519",
            {
                "ratios": {"roe": 1.0, "roa": 1.0, "gross_margin": 0.2, "net_margin": 0.01},
                "income": {"net_profit": -5.0},
                "balance": {"total_liabilities": 0.0, "cash": 0.0},
                "net_profit": -5.0,
                "equity": -20.0,
                "revenue": 0.0,
                "total_assets": 100.0,
            },
            price=10.0,
            shares=100.0,
        )
        self.assertIn("PE: N/A", report)
        self.assertIn("PB: N/A", report)
        self.assertIn("PS: N/A", report)

    def test_earnings_forecast_maps_current_akshare_columns_and_requires_schema(self):
        from quant_system import earnings_calendar

        current_columns = pd.DataFrame(
            [
                {
                    "序号": 1,
                    "股票代码": "600519",
                    "股票简称": "贵州茅台",
                    "预测指标": "净利润",
                    "业绩变动": "增长",
                    "预测数值": "100亿",
                    "业绩变动幅度": "50%~70%",
                    "业绩变动原因": "主营增长",
                    "预告类型": "预增",
                    "上年同期值": "60亿",
                    "公告日期": "2026-07-15",
                },
                {
                    "序号": 2,
                    "股票代码": "000001",
                    "股票简称": "平安银行",
                    "预测指标": "净利润",
                    "业绩变动": "下降",
                    "预测数值": "80亿",
                    "业绩变动幅度": "-30%~-10%",
                    "业绩变动原因": "利润下降",
                    "预告类型": "预减",
                    "上年同期值": "100亿",
                    "公告日期": "2026-07-16",
                },
            ]
        )

        fake_ak = types.SimpleNamespace(stock_yjyg_em=lambda date: current_columns.copy())
        with patch.dict("sys.modules", {"akshare": fake_ak}):
            forecast = earnings_calendar.get_earnings_forecast("20260630")

        self.assertIn("forecast_type", forecast.columns)
        self.assertIn("change_range", forecast.columns)
        self.assertIn("announce_date", forecast.columns)
        self.assertEqual(forecast.loc[0, "forecast_type"], "预增")
        self.assertEqual(forecast.loc[0, "change_range"], "50%~70%")

        with patch("quant_system.earnings_calendar.get_earnings_forecast", return_value=forecast):
            surprises = earnings_calendar.get_surprises(top_n=20)
        self.assertEqual(surprises["name"].tolist(), ["贵州茅台"])
        text = earnings_calendar.format_surprises(surprises)
        self.assertIn("贵州茅台: 预增 50%~70%", text)
        self.assertNotIn("平安银行", text)

        missing_columns = current_columns.drop(columns=["预告类型"])
        fake_bad_ak = types.SimpleNamespace(stock_yjyg_em=lambda date: missing_columns.copy())
        with patch.dict("sys.modules", {"akshare": fake_bad_ak}):
            with self.assertRaisesRegex(ValueError, "forecast_type"):
                earnings_calendar.get_earnings_forecast("20260630")

        with patch("quant_system.earnings_calendar.get_earnings_forecast", return_value=forecast.drop(columns=["forecast_type"])):
            with self.assertRaisesRegex(ValueError, "forecast_type"):
                earnings_calendar.get_surprises(top_n=20)

    def test_market_clock_shifts_from_non_trading_days(self):
        import quant_system.market_clock as market_clock

        old_cache = market_clock._TRADE_CALENDAR_CACHE
        market_clock._TRADE_CALENDAR_CACHE = {
            "2024-06-07",
            "2024-06-11",
            "2024-06-12",
        }
        try:
            self.assertEqual(market_clock.prev_trading_day("2024-06-08").isoformat(), "2024-06-07")
            self.assertEqual(market_clock.next_trading_day("2024-06-08").isoformat(), "2024-06-11")
            self.assertEqual(market_clock.latest_trading_day("2024-06-10").isoformat(), "2024-06-07")
            self.assertEqual(market_clock.prev_trading_day("2024-06-11").isoformat(), "2024-06-07")
            self.assertEqual(market_clock.next_trading_day("2024-06-07").isoformat(), "2024-06-11")
        finally:
            market_clock._TRADE_CALENDAR_CACHE = old_cache

    def test_market_clock_trading_session_respects_real_calendar(self):
        from datetime import datetime
        import quant_system.market_clock as market_clock

        old_cache = market_clock._TRADE_CALENDAR_CACHE
        market_clock._TRADE_CALENDAR_CACHE = {"2024-10-08"}
        try:
            holiday_morning = datetime(2024, 10, 1, 9, 30, tzinfo=market_clock.CST)
            trading_morning = datetime(2024, 10, 8, 9, 30, tzinfo=market_clock.CST)
            self.assertFalse(market_clock.is_trading_session(holiday_morning))
            self.assertTrue(market_clock.is_trading_session(trading_morning))
        finally:
            market_clock._TRADE_CALENDAR_CACHE = old_cache

    def test_ashare_special_uses_real_lhb_apis_not_limit_pool_fallback(self):
        import quant_system.ashare_special as ashare_special

        lhb = pd.DataFrame(
            {
                "代码": ["600519", "000001"],
                "名称": ["贵州茅台", "平安银行"],
                "净买入额": [1200.0, 300.0],
            }
        )

        class FakeAk:
            def stock_lhb_detail_daily_sina(self, date):
                self.daily_date = date
                return lhb.copy()

            def stock_zt_pool_em(self, date):
                raise AssertionError("limit-up pool must not be used as dragon-tiger fallback")

            def stock_dt_pool_em(self, date):
                raise AssertionError("limit-down pool must not be used as dragon-tiger fallback")

        fake_ak = FakeAk()
        with patch.dict("sys.modules", {"akshare": fake_ak}):
            daily = ashare_special.get_dragon_tiger_daily("20260731")
            self.assertEqual(fake_ak.daily_date, "20260731")
            self.assertEqual(daily["source"].unique().tolist(), ["stock_lhb_detail_daily_sina"])
            self.assertEqual(daily["代码"].tolist(), ["600519", "000001"])

            with patch("quant_system.ashare_special.get_dragon_tiger_daily", return_value=daily):
                top = ashare_special.get_top_dragon_tiger(top_n=1)
            self.assertEqual(top["代码"].tolist(), ["600519"])

        class EmptyAk:
            def stock_lhb_detail_daily_sina(self, date):
                return pd.DataFrame()

            def stock_lhb_detail_em(self, start_date, end_date):
                return pd.DataFrame()

            def stock_zt_pool_em(self, date):
                raise AssertionError("limit-up pool must not be used when LHB is empty")

        with patch.dict("sys.modules", {"akshare": EmptyAk()}):
            self.assertTrue(ashare_special.get_dragon_tiger_daily("20260731").empty)

    def test_ashare_special_block_trades_use_current_akshare_signature(self):
        import quant_system.ashare_special as ashare_special

        block_trades = pd.DataFrame(
            {
                "代码": ["600519"],
                "名称": ["贵州茅台"],
                "折溢率": [-3.5],
            }
        )

        class FakeAk:
            def stock_dzjy_mrmx(self, symbol, start_date, end_date):
                self.mrmx_args = (symbol, start_date, end_date)
                return block_trades.copy()

            def stock_dzjy_mrtj(self, start_date, end_date):
                raise AssertionError("daily summary fallback should not run after detail data succeeds")

            def stock_block_trade_em(self, date):
                raise AssertionError("removed nonexistent block trade fallback must not be called")

        fake_ak = FakeAk()
        with patch.dict("sys.modules", {"akshare": fake_ak}):
            df = ashare_special.get_block_trades("2026-07-31")

        self.assertEqual(fake_ak.mrmx_args, ("A股", "20260731", "20260731"))
        self.assertEqual(df["source"].tolist(), ["stock_dzjy_mrmx"])
        self.assertEqual(df["代码"].tolist(), ["600519"])

        class FallbackAk:
            def stock_dzjy_mrmx(self, symbol, start_date, end_date):
                return pd.DataFrame()

            def stock_dzjy_mrtj(self, start_date, end_date):
                self.mrtj_args = (start_date, end_date)
                return pd.DataFrame({"成交额": [100.0]})

        fallback_ak = FallbackAk()
        with patch.dict("sys.modules", {"akshare": fallback_ak}):
            fallback = ashare_special.get_block_trades("20260731")

        self.assertEqual(fallback_ak.mrtj_args, ("20260731", "20260731"))
        self.assertEqual(fallback["source"].tolist(), ["stock_dzjy_mrtj"])

    def test_ashare_special_shareholder_changes_do_not_use_wrong_domain_fallbacks(self):
        import quant_system.ashare_special as ashare_special

        changes = pd.DataFrame({"代码": ["600519"], "变动人": ["高管A"], "变动数量": [1000]})

        class FakeAk:
            def stock_shareholder_change_ths(self, symbol):
                self.symbol = symbol
                return changes.copy()

            def stock_ggt_net_buy_detail(self):
                raise AssertionError("HK connect net-buy data must not be used for shareholder changes")

            def stock_shareholder_hold_change(self):
                raise AssertionError("nonexistent shareholder fallback must not be called")

        fake_ak = FakeAk()
        with patch.dict("sys.modules", {"akshare": fake_ak}):
            stock_df = ashare_special.get_shareholder_changes(stock="600519", top_n=1)
            all_df = ashare_special.get_shareholder_changes(stock=None, top_n=20)

        self.assertEqual(fake_ak.symbol, "600519")
        self.assertEqual(stock_df["source"].tolist(), ["stock_shareholder_change_ths"])
        self.assertEqual(stock_df["代码"].tolist(), ["600519"])
        self.assertEqual(all_df["source"].tolist(), ["stock_shareholder_change_ths"])
        self.assertIn("请传入stock参数", all_df.loc[0, "error"])

    def test_north_flow_summary_compat_and_market_pulse_keeps_north_component(self):
        import quant_system.market_pulse as market_pulse
        import quant_system.north_flow as north_flow

        north_summary = {
            "date": "2026-07-31",
            "total_net_yi": 12.34,
            "sh_net_yi": 5.0,
            "sz_net_yi": 7.34,
            "net_buy_disclosed": True,
            "trading_status": "非交易时段",
        }

        with patch("quant_system.north_flow.fetch_north_summary", return_value=north_summary.copy()):
            compat = north_flow.get_north_flow_summary()

        self.assertTrue(compat["ok"])
        self.assertEqual(compat["net_amount"], 1234000000.0)
        self.assertEqual(compat["sh_net"], 500000000.0)
        self.assertEqual(compat["sz_net"], 734000000.0)

        class FakeAk:
            def stock_zt_pool_em(self, date):
                return pd.DataFrame({"代码": ["600519"]})

            def stock_zt_pool_dtgc_em(self, date):
                return pd.DataFrame({"代码": ["000001"]})

            def stock_sse_summary(self):
                return pd.DataFrame({"成交额": [7500 * 1e8]})

        with patch.dict("sys.modules", {"akshare": FakeAk()}):
            with patch("quant_system.market_pulse._fetch_adv_dec", return_value={"up": 60, "down": 40}):
                with patch("quant_system.north_flow.fetch_north_summary", return_value=north_summary.copy()):
                    pulse = market_pulse.get_fear_greed_index()

        self.assertIn("北向资金", pulse["components"])
        self.assertGreater(pulse["components"]["北向资金"], 50.0)

    def test_funding_sentiment_uses_existing_margin_and_north_summary_functions(self):
        from quant_system.sentiment_factory.funding_sentiment import FundingSentiment

        margin_summary = {
            "total_margin_balance": 21000.0,
            "total_margin_change": 80.0,
        }
        north_summary = {
            "total_net_yi": 65.0,
        }

        with patch("quant_system.margin.fetch_margin_summary", return_value=margin_summary):
            with patch("quant_system.north_flow.fetch_north_summary", return_value=north_summary):
                result = FundingSentiment(cache_ttl=0).compute()

        self.assertIn("margin", result)
        self.assertIn("north", result)
        self.assertEqual(result["margin"]["margin_direction"], "add")
        self.assertEqual(result["north"]["north_direction"], "inflow")
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["direction"], "bullish")
        self.assertTrue(any("两融加仓" in signal for signal in result["sub_signals"]))
        self.assertTrue(any("北向净流入" in signal for signal in result["sub_signals"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
