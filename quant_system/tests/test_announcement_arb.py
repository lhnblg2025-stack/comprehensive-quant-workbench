"""analysis_core/announcement_arbitrage 公告事件深度排雷/套利单元测试。

覆盖:
  - 正则解析: _cn_to_int / _parse_date_token / _norm_date / _extract_dated /
    _extract_price / _extract_pct / 五类事件参数提取
  - analyze: 事件类型优先级（强赎最先）、各类型结论、unknown 降级
  - 本地行情读取: _load_cb_price / _load_cb_convert_price / _load_stock_price /
    _bond_name（缺失/损坏/空/命中）
  - _days_to 日期语义 + 防前视: 公告事件日期 > 分析日(now) 时不触发当日紧迫；
    _cb_recommendation 全分支
  - convertible_bond_arb: 亏损比例/转股价值/安全边际/紧迫度（构造数据 + loader 降级）
  - announcement_risk: 强赎致命/重大/警示、要约溢价/折价、配股/减持/回购、unknown
  - self_test 验收自测（CB_SPOT/KLINE_DIR mock 到 tmp，零真实行情依赖）

无网络：行情全部走 mock.patch.object 路径常量到 tmp；防前视用显式 now。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.announcement_arbitrage as aa
    return aa


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="ann_arb_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


class TestParseUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_cn_to_int(self):
        f = self.aa._cn_to_int
        self.assertEqual(f("12"), 12)
        self.assertEqual(f("八"), 8)
        self.assertEqual(f("十二"), 12)
        self.assertEqual(f("二十"), 20)
        self.assertEqual(f("三十一"), 31)
        self.assertEqual(f("两"), 2)
        self.assertEqual(f("十五"), 15)
        self.assertIsNone(f(""))
        self.assertIsNone(f("abc"))
        self.assertIsNone(f("一二三a"))

    def test_parse_date_token(self):
        f = self.aa._parse_date_token
        self.assertEqual(f("2026-08-12"), (2026, 8, 12))
        self.assertEqual(f("2026/8/12"), (2026, 8, 12))
        self.assertEqual(f("2026.08.12"), (2026, 8, 12))
        self.assertEqual(f("8/12"), (None, 8, 12))
        self.assertEqual(f("8-12"), (None, 8, 12))
        self.assertEqual(f("8月12日"), (None, 8, 12))
        self.assertEqual(f("八月十二日"), (None, 8, 12))
        self.assertIsNone(f(""))
        self.assertIsNone(f("随便写"))

    def test_norm_date(self):
        n = self.aa._norm_date
        self.assertEqual(n((2026, 8, 12), "2026-08-12"), ("2026-08-12", "2026-08-12"))
        self.assertEqual(n((None, 8, 12), "8/12"), ("08-12", "8/12"))
        self.assertEqual(n(None, "raw"), (None, "raw"))

    def test_extract_dated(self):
        f = self.aa._extract_dated
        self.assertEqual(
            f("最后转股日2026-08-12，赎回价格100.16元", "最后转股日"),
            ("2026-08-12", "2026-08-12"))
        self.assertEqual(f("最后转股日8月12日", "最后转股日"), ("08-12", "8月12日"))
        self.assertEqual(f("无日期", "最后转股日"), (None, None))
        # 日期在 label 后 16 字符之外 → 不提取
        self.assertEqual(f("最后转股日" + "很长的填充" * 4 + "2026-08-12", "最后转股日"),
                         (None, None))

    def test_extract_price_and_pct(self):
        self.assertEqual(self.aa._extract_price("赎回价格100.16元", r"赎回价(?:格)?(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)"), 100.16)
        self.assertEqual(self.aa._extract_price("无价格", r"(\d+(?:\.\d+)?)"), None)
        self.assertEqual(self.aa._extract_pct("减持比例18.05%", r"(\d+(?:\.\d+)?)"), 18.05)


class TestParamExtractors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_redemption_params(self):
        p = self.aa._extract_redemption_params(
            "最后交易日8月7日，最后转股日8月12日，赎回登记日8月11日，赎回价格100.16元，转股价50元")
        self.assertEqual(p["redeem_price"], 100.16)
        self.assertEqual(p["last_trade_date"], "08-07")
        self.assertEqual(p["last_convert_date"], "08-12")
        self.assertEqual(p["record_date"], "08-11")
        self.assertEqual(p["convert_price"], 50.0)
        self.assertIn("8月7日", p["last_trade_date_raw"])

    def test_offer_params(self):
        p = self.aa._extract_offer_params("要约收购价格28.5元，要约收购比例18.05%")
        self.assertEqual(p["offer_price"], 28.5)
        self.assertEqual(p["offer_ratio"], 18.05)
        p2 = self.aa._extract_offer_params("收购股份比例5%")
        self.assertEqual(p2["offer_ratio"], 5.0)

    def test_rights_params(self):
        p = self.aa._extract_rights_params("每10股配3股，配股价6.8元，股权登记日2026-08-15")
        self.assertEqual(p["rights_price"], 6.8)
        self.assertEqual(p["rights_ratio"], 0.3)
        self.assertEqual(p["rights_ratio_raw"], "每10股配3股")
        self.assertEqual(p["record_date"], "2026-08-15")
        p2 = self.aa._extract_rights_params("配股比例10%")
        self.assertEqual(p2["rights_ratio"], 10.0)
        self.assertEqual(p2["rights_ratio_raw"], "10.0%")

    def test_reduce_params(self):
        p = self.aa._extract_reduce_params("股东拟减持不超过1.5%")
        self.assertEqual(p["reduce_ratio"], 1.5)
        p2 = self.aa._extract_reduce_params("减持比例2%")
        self.assertEqual(p2["reduce_ratio"], 2.0)
        p3 = self.aa._extract_reduce_params("拟减持1.2%股份")
        self.assertEqual(p3["reduce_ratio"], 1.2)

    def test_buyback_params(self):
        p = self.aa._extract_buyback_params("回购金额为2亿元，回购价格上限15元")
        self.assertEqual(p["buyback_amount"], 2.0)
        self.assertEqual(p["buyback_amount_unit"], "亿")
        self.assertEqual(p["buyback_price"], 15.0)
        p2 = self.aa._extract_buyback_params("回购资金总额为5万")
        self.assertEqual(p2["buyback_amount"], 5.0)
        self.assertEqual(p2["buyback_amount_unit"], "万")


class TestAnalyze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_redemption(self):
        r = self.aa.analyze("正帆转债赎回实施的提示性公告",
                            "最后交易日8月7日，最后转股日8月12日，赎回价格100.16元")
        self.assertEqual(r["event_type"], "强赎")
        self.assertEqual(r["params"]["redeem_price"], 100.16)
        self.assertEqual(r["params"]["last_convert_date"], "08-12")
        self.assertIn("已触发强赎", r["conclusion"])
        self.assertIn("最后转股日08-12", r["conclusion"])

    def test_redemption_missing_params(self):
        r = self.aa.analyze("可转债赎回公告", "公司决定行使赎回权")
        self.assertEqual(r["event_type"], "强赎")
        self.assertIn("关键参数缺失", r["conclusion"])

    def test_event_priority_redemption_first(self):
        r = self.aa.analyze("关于强制赎回与要约收购的公告", "要约收购价格10元，赎回价格100元")
        self.assertEqual(r["event_type"], "强赎")

    def test_offer(self):
        r = self.aa.analyze("关于要约收购的提示性公告", "要约收购价格28.5元，要约收购比例18.05%")
        self.assertEqual(r["event_type"], "要约收购")
        self.assertEqual(r["params"]["offer_price"], 28.5)
        self.assertIn("要约收购", r["conclusion"])

    def test_offer_missing_params(self):
        r = self.aa.analyze("要约收购公告", "公司收到要约收购通知")
        self.assertEqual(r["event_type"], "要约收购")
        self.assertIn("缺少要约价", r["conclusion"])

    def test_rights(self):
        r = self.aa.analyze("配股发行公告", "每10股配3股，配股价6.8元，股权登记日2026-08-15")
        self.assertEqual(r["event_type"], "配股")
        self.assertEqual(r["params"]["rights_ratio"], 0.3)
        self.assertIn("配股方案", r["conclusion"])

    def test_rights_missing_params(self):
        r = self.aa.analyze("配股公告", "公司拟实施配股")
        self.assertEqual(r["event_type"], "配股")
        self.assertIn("缺少配股价", r["conclusion"])

    def test_reduce(self):
        r = self.aa.analyze("股东减持计划公告", "拟减持不超过1.5%")
        self.assertEqual(r["event_type"], "减持")
        self.assertEqual(r["params"]["reduce_ratio"], 1.5)
        self.assertIn("短期或有抛压", r["conclusion"])

    def test_buyback(self):
        r = self.aa.analyze("回购股份方案公告", "回购金额为2亿元，回购价格上限15元")
        self.assertEqual(r["event_type"], "回购")
        self.assertEqual(r["params"]["buyback_price"], 15.0)
        self.assertIn("中性偏多", r["conclusion"])

    def test_buyback_missing_params(self):
        r = self.aa.analyze("回购公告", "公司计划回购股份")
        self.assertEqual(r["event_type"], "回购")
        self.assertIsNone(r["params"].get("buyback_amount"))
        self.assertEqual(r["conclusion"], "公司回购，中性偏多")

    def test_unknown(self):
        r = self.aa.analyze("日常经营公告", "公司召开董事会会议")
        self.assertEqual(r["event_type"], "unknown")
        self.assertEqual(r["conclusion"], "无量化结论")


class TestLocalMarketLoaders(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def _cb_df(self):
        return pd.DataFrame({
            "code": ["118053", "113050"],
            "symbol": ["sh118053", "sh113050"],
            "name": ["正帆转债", "南银转债"],
            "trade": [168.9, 102.0],
            "转股价": [50.0, 10.0],
        })

    def test_load_cb_price(self):
        tmp = self._make_tmp()
        f = tmp / "cb.parquet"
        self._cb_df().to_parquet(f)
        with mock.patch.object(self.aa, "CB_SPOT", f):
            self.assertEqual(self.aa._load_cb_price("118053"), 168.9)
            # symbol 后缀匹配
            self.assertEqual(self.aa._load_cb_price("113050"), 102.0)
            self.assertIsNone(self.aa._load_cb_price("999999"))

    def test_load_cb_price_missing_corrupt_empty(self):
        tmp = self._make_tmp()
        with mock.patch.object(self.aa, "CB_SPOT", tmp / "nope.parquet"):
            self.assertIsNone(self.aa._load_cb_price("118053"))
        bad = tmp / "bad.parquet"
        bad.write_bytes(b"not-parquet")
        with mock.patch.object(self.aa, "CB_SPOT", bad):
            self.assertIsNone(self.aa._load_cb_price("118053"))
        empty = tmp / "empty.parquet"
        pd.DataFrame(columns=["code", "symbol"]).to_parquet(empty)
        with mock.patch.object(self.aa, "CB_SPOT", empty):
            self.assertIsNone(self.aa._load_cb_price("118053"))

    def test_load_cb_convert_price(self):
        tmp = self._make_tmp()
        f = tmp / "cb.parquet"
        self._cb_df().to_parquet(f)
        with mock.patch.object(self.aa, "CB_SPOT", f):
            self.assertEqual(self.aa._load_cb_convert_price("118053"), 50.0)
            self.assertIsNone(self.aa._load_cb_convert_price("999999"))

    def test_load_stock_price(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        pd.DataFrame({"date": ["2026-08-10", "2026-08-11"], "close": [64.0, 65.98]}).to_parquet(
            kdir / "600001.parquet")
        with mock.patch.object(self.aa, "KLINE_DIR", kdir):
            self.assertEqual(self.aa._load_stock_price("600001"), 65.98)
            self.assertIsNone(self.aa._load_stock_price("999999"))
        bad = kdir / "bad.parquet"
        bad.write_bytes(b"x")
        with mock.patch.object(self.aa, "KLINE_DIR", kdir):
            self.assertIsNone(self.aa._load_stock_price("bad"))

    def test_bond_name(self):
        tmp = self._make_tmp()
        f = tmp / "cb.parquet"
        self._cb_df().to_parquet(f)
        self.aa._bond_name_cache.clear()
        with mock.patch.object(self.aa, "CB_SPOT", f):
            self.assertEqual(self.aa._bond_name("118053"), "正帆转债")
            # 命中缓存
            self.assertEqual(self.aa._bond_name("118053"), "正帆转债")
            self.assertIsNone(self.aa._bond_name("999999"))
        self.aa._bond_name_cache.clear()
        with mock.patch.object(self.aa, "CB_SPOT", tmp / "nope.parquet"):
            self.assertIsNone(self.aa._bond_name("118053"))
        self.aa._bond_name_cache.clear()


class TestDaysTo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_absolute_dates(self):
        f = self.aa._days_to
        self.assertEqual(f("2026-08-20", now=datetime(2026, 8, 12)), 8)
        self.assertEqual(f("2026-08-12", now=datetime(2026, 8, 12)), 0)
        self.assertEqual(f("2026-08-10", now=datetime(2026, 8, 12)), -2)

    def test_no_year_date_resolves_this_year(self):
        f = self.aa._days_to
        self.assertEqual(f("08-12", now=datetime(2026, 8, 12)), 0)
        # 已过期 → 顺延一年（不把过去日期当作未来事件）
        self.assertEqual(f("01-01", now=datetime(2026, 12, 31)), 1)

    def test_invalid(self):
        f = self.aa._days_to
        self.assertIsNone(f(""))
        self.assertIsNone(f("不是日期"))

    def test_anti_lookahead_future_event_not_urgent(self):
        """防前视硬断言: 公告事件日期(2026-08-20) > 分析日(2026-08-12) →
        urgency_days=8 → 不触发'务必今日'。"""
        arb = self.aa.convertible_bond_arb(
            "118053", 100.16, "2026-08-20",
            bond_price=102.0, stock_price=65.98, convert_price=50.0,
            now=datetime(2026, 8, 12))
        self.assertEqual(arb["urgency_days"], 8)
        self.assertIn("尽快", arb["recommendation"])
        self.assertNotIn("务必今日", arb["recommendation"])
        # 同日/已到期 → 务必今日
        arb2 = self.aa.convertible_bond_arb(
            "118053", 100.16, "2026-08-20",
            bond_price=102.0, stock_price=65.98, convert_price=50.0,
            now=datetime(2026, 8, 20))
        self.assertEqual(arb2["urgency_days"], 0)
        self.assertIn("务必今日", arb2["recommendation"])


class TestCbRecommendation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_missing_market(self):
        r = self.aa._cb_recommendation(None, 100.16, None, None, None)
        self.assertIn("观望", r)

    def test_below_redeem_price(self):
        r = self.aa._cb_recommendation(90.0, 100.16, None, None, None)
        self.assertIn("可持有等待赎回", r)

    def test_urgent_convert(self):
        r = self.aa._cb_recommendation(168.9, 100.16, 180.0, 0.4, 0)
        self.assertIn("务必今日转股", r)

    def test_urgent_sell(self):
        r = self.aa._cb_recommendation(168.9, 100.16, 90.0, 0.4, 0)
        self.assertIn("务必今日卖出", r)

    def test_loss_high_convert_value_missing(self):
        r = self.aa._cb_recommendation(168.9, 100.16, None, 0.4, 5)
        self.assertIn("务必今日卖出或转股", r)

    def test_soon(self):
        r = self.aa._cb_recommendation(168.9, 100.16, 180.0, 0.05, 5)
        self.assertIn("尽快转股", r)


class TestConvertibleBondArb(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_full_params(self):
        arb = self.aa.convertible_bond_arb(
            "118053", 100.16, "2026-08-12",
            bond_price=168.9, stock_price=65.98, convert_price=50.0,
            now=datetime(2026, 8, 12))
        self.assertEqual(arb["loss_if_not_convert"], round((168.9 - 100.16) / 168.9, 4))
        self.assertEqual(arb["convert_value"], 131.96)
        self.assertEqual(arb["margin"], round((131.96 - 100.16) / 100.16, 4))
        self.assertEqual(arb["urgency_days"], 0)
        self.assertIn("务必今日", arb["recommendation"])

    def test_loader_fallback(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        pd.DataFrame({"date": ["2026-08-11"], "close": [65.98]}).to_parquet(
            kdir / "118053.parquet")
        f = tmp / "cb.parquet"
        pd.DataFrame({
            "code": ["118053"], "symbol": ["sh118053"], "name": ["正帆转债"],
            "trade": [168.9], "转股价": [50.0],
        }).to_parquet(f)
        with mock.patch.object(self.aa, "CB_SPOT", f), \
                mock.patch.object(self.aa, "KLINE_DIR", kdir):
            arb = self.aa.convertible_bond_arb("118053", 100.16, "2026-08-20",
                                               now=datetime(2026, 8, 12))
        self.assertEqual(arb["bond_price"], 168.9)
        self.assertEqual(arb["convert_value"], round(65.98 * 2.0, 2))
        self.assertIsNotNone(arb["convert_value"])

    def test_market_missing_degrades(self):
        tmp = self._make_tmp()
        with mock.patch.object(self.aa, "CB_SPOT", tmp / "nope.parquet"), \
                mock.patch.object(self.aa, "KLINE_DIR", tmp / "kline"):
            arb = self.aa.convertible_bond_arb("118053", 100.16, "2026-08-20",
                                               now=datetime(2026, 8, 12))
        self.assertIsNone(arb["bond_price"])
        self.assertIsNone(arb["loss_if_not_convert"])
        self.assertIsNone(arb["convert_value"])
        self.assertIn("观望", arb["recommendation"])

    def test_convert_ratio_passed_directly(self):
        arb = self.aa.convertible_bond_arb(
            "118053", 100.16, "2026-08-20", bond_price=168.9, stock_price=65.98,
            convert_ratio=2.5, now=datetime(2026, 8, 12))
        self.assertEqual(arb["convert_value"], round(65.98 * 2.5, 2))


class TestShortDateAndDesc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_short_date(self):
        self.assertEqual(self.aa._short_date("2026-08-12"), "2026/8/12")
        self.assertEqual(self.aa._short_date("08-12"), "8/12")
        self.assertEqual(self.aa._short_date("乱"), "乱")

    def test_fmt_cb_desc(self):
        arb = {"redeem_price": 100.16, "bond_price": 168.9,
               "loss_if_not_convert": 0.4, "last_convert_date": "2026-08-12",
               "recommendation": "务必今日卖出"}
        with mock.patch.object(self.aa, "_bond_name", return_value="正帆转债"):
            d = self.aa._fmt_cb_desc("118053", arb)
        self.assertIn("正帆转债", d)
        self.assertIn("现价168.9 vs 强赎价100.16", d)
        self.assertIn("亏损40.0%", d)
        self.assertIn("2026/8/12", d)
        # 行情缺失
        arb2 = {"redeem_price": 100.16, "bond_price": None, "loss_if_not_convert": None,
                "last_convert_date": "", "recommendation": "观望"}
        with mock.patch.object(self.aa, "_bond_name", return_value=None):
            d2 = self.aa._fmt_cb_desc("118053", arb2)
        self.assertIn("强赎公告（行情缺失）", d2)


class TestAnnouncementRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_unknown(self):
        r = self.aa.announcement_risk("600001", "日常公告", "董事会召开")
        self.assertEqual(r["risk_level"], "未知")
        self.assertEqual(r["event_type"], "unknown")

    def test_redemption_fatal(self):
        r = self.aa.announcement_risk(
            "118053", "转债赎回实施公告", "最后转股日2026-08-12，赎回价格100.16元",
            bond_price=168.9, stock_price=65.98, convert_price=50.0,
            now=datetime(2026, 8, 12))
        self.assertEqual(r["event_type"], "强赎")
        self.assertEqual(r["risk_level"], "致命")
        self.assertIn("loss_if_not_convert", r["numbers"])
        self.assertIn("118053", r["description"])

    def test_redemption_major(self):
        r = self.aa.announcement_risk(
            "118053", "转债赎回实施公告", "最后转股日2026-08-20，赎回价格100.16元",
            bond_price=105.0, stock_price=65.98, convert_price=50.0,
            now=datetime(2026, 8, 12))
        self.assertEqual(r["risk_level"], "重大")

    def test_redemption_warning_loss_zero(self):
        r = self.aa.announcement_risk(
            "118053", "转债赎回实施公告", "最后转股日2026-08-12，赎回价格100.16元",
            bond_price=95.0, now=datetime(2026, 8, 12))
        self.assertEqual(r["risk_level"], "警示")
        self.assertIsNotNone(r["numbers"])

    def test_redemption_missing_market(self):
        tmp = Path(tempfile.mkdtemp(prefix="ann_arb_risk_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch.object(self.aa, "CB_SPOT", tmp / "nope.parquet"), \
                mock.patch.object(self.aa, "KLINE_DIR", tmp / "kline"):
            r = self.aa.announcement_risk(
                "118053", "转债赎回实施公告", "最后转股日2026-08-12，赎回价格100.16元",
                now=datetime(2026, 8, 12))
        self.assertEqual(r["risk_level"], "警示")
        self.assertIsNone(r["numbers"])

    def test_redemption_missing_redeem_price(self):
        r = self.aa.announcement_risk(
            "118053", "转债赎回实施公告", "最后转股日2026-08-12",
            bond_price=168.9, now=datetime(2026, 8, 12))
        self.assertEqual(r["risk_level"], "重大")
        self.assertIn("未提取到赎回价", r["actionable"])

    def test_offer_premium(self):
        r = self.aa.announcement_risk(
            "600001", "要约收购公告", "要约收购价格28.5元",
            stock_price=25.0)
        self.assertEqual(r["event_type"], "要约收购")
        self.assertEqual(r["risk_level"], "重大")
        self.assertEqual(r["numbers"]["gap_pct"], round((28.5 - 25.0) / 25.0, 4))
        self.assertIn("溢价", r["actionable"])

    def test_offer_discount(self):
        r = self.aa.announcement_risk(
            "600001", "要约收购公告", "要约收购价格28.5元",
            stock_price=30.0)
        self.assertEqual(r["risk_level"], "警示")
        self.assertIn("折价", r["actionable"])

    def test_offer_missing_price(self):
        r = self.aa.announcement_risk("600001", "要约收购公告", "收到要约通知")
        self.assertEqual(r["risk_level"], "警示")
        self.assertIsNone(r["numbers"])

    def test_rights_major(self):
        r = self.aa.announcement_risk("600001", "配股公告", "每10股配3股，配股价6.8元，股权登记日2026-08-15")
        self.assertEqual(r["event_type"], "配股")
        self.assertEqual(r["risk_level"], "重大")
        self.assertEqual(r["numbers"]["rights_ratio"], 0.3)

    def test_rights_warning(self):
        r = self.aa.announcement_risk("600001", "配股公告", "拟实施配股")
        self.assertEqual(r["risk_level"], "警示")

    def test_reduce(self):
        r = self.aa.announcement_risk("600001", "减持公告", "拟减持不超过1.5%")
        self.assertEqual(r["risk_level"], "警示")
        self.assertEqual(r["numbers"]["reduce_ratio"], 1.5)

    def test_buyback(self):
        r = self.aa.announcement_risk("600001", "回购公告", "回购金额2亿元，回购价格上限15元")
        self.assertEqual(r["risk_level"], "警示")
        self.assertEqual(r["numbers"]["buyback_amount"], 2.0)
        self.assertEqual(r["numbers"]["buyback_amount_unit"], "亿")


class TestSelfTest(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aa = _import()

    def test_self_test_returns_true_offline(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        with mock.patch.object(self.aa, "CB_SPOT", tmp / "nope.parquet"), \
                mock.patch.object(self.aa, "KLINE_DIR", kdir):
            self.assertTrue(self.aa.self_test())


if __name__ == "__main__":
    unittest.main()
