#!/usr/bin/env python3
"""V3 新增模块测试：openclaw_api / audit / data_store V3 登记（2026-08-07）

覆盖：
- DataStore: 40 数据集登记完整性 / get_dataset files 参数 / freshness 状态
- openclaw_api: 个股总览 / 龙虎榜 / 市场全景（mock 仓库）
- audit: 数据合理性 / 输出内容 / 维护自检（语法扫描）
"""
import sys
import pathlib
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestDataStoreV3(unittest.TestCase):
    """DataStore V3 登记扩展（40 数据集）。"""

    def setUp(self):
        from quant_system.data_store import DataStore
        self.ds = DataStore()

    def test_datasets_registered_count(self):
        """登记数据集 ≥ 30（V3 扩展后 40 个）。"""
        dsets = self.ds.list_datasets()
        self.assertGreaterEqual(len(dsets), 30)

    def test_new_datasets_present(self):
        """关键非量化指标数据集已登记。"""
        dsets = set(self.ds.list_datasets())
        for name in ("lhb_institution", "lhb_broker", "lhb_rank", "gdhs",
                     "fund_hold", "market_fund_flow", "zlkp_jgcyd",
                     "sw_industry_map", "concept_board", "theme_board",
                     "margin_summary", "commodity", "futures", "rates"):
            self.assertIn(name, dsets, f"缺少数据集登记: {name}")

    def test_schema_of_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.ds.schema_of("不存在的数据集")

    def test_get_dataset_files_param(self):
        """files 参数只读最近 N 文件（lhb 30 文件 → files=2 应 ≤2 文件数据）。"""
        df = self.ds.get_dataset("lhb", files=2, limit=10)
        # 不抛错即可（数据可能为空但不应异常）
        self.assertIsNotNone(df)

    def test_get_dataset_date_filter(self):
        """V11 突变测试补盲: get_dataset 日期过滤必须生效（防突变漏网）。

        用 zlkp_jgcyd（日序列数据集）验证 date 参数过滤：指定历史日期
        返回的行必须全部等于该日期；同时验证 start/end 区间过滤。
        """
        import pandas as pd
        df = self.ds.get_dataset("zlkp_jgcyd")
        # 动态探测日期列（英文 date 或中文 交易日/日期）
        dcol = next((c for c in ("date", "trade_date", "交易日", "日期") if c in df.columns), None)
        if df.empty or dcol is None:
            self.skipTest("zlkp_jgcyd 无数据/无日期列，跳过")
        dvals = pd.to_datetime(df[dcol], errors="coerce").dropna()
        dmax = dvals.max()
        # 取历史某一天（非最后一天），过滤后必须只含该日期
        mid_date = dvals.min() + (dmax - dvals.min()) / 2
        day = mid_date.strftime("%Y-%m-%d")
        sub = self.ds.get_dataset("zlkp_jgcyd", date=day)
        self.assertFalse(sub.empty, f"date={day} 过滤后为空——日期过滤可能失效")
        vals = pd.to_datetime(sub[dcol]).dt.date.astype(str)
        self.assertTrue((vals == day).all(), f"date={day} 过滤后出现其他日期: {vals.unique()[:5]}")
        # start/end 区间过滤
        start = mid_date.strftime("%Y-%m-%d")
        sub2 = self.ds.get_dataset("zlkp_jgcyd", start=start)
        self.assertFalse(sub2.empty, f"start={start} 过滤后为空")
        vals2 = pd.to_datetime(sub2[dcol])
        self.assertTrue((vals2 >= pd.Timestamp(start)).all())

    def test_freshness_structure(self):
        """freshness 返回结构完整。"""
        rows = self.ds.freshness()
        self.assertGreater(len(rows), 30)
        for r in rows:
            self.assertIn("dataset", r)
            self.assertIn("status", r)
            self.assertIn("files", r)


class TestOpenclawApi(unittest.TestCase):
    """openclaw_api 内置接口（用户 #9：非量化指标供 openclaw 调用）。"""

    def test_stock_overview_structure(self):
        from quant_platform.openclaw_api import stock_overview
        out = stock_overview("600519")
        self.assertEqual(out["code"], "600519")
        self.assertIn("kline", out)
        self.assertIn("industry", out)
        self.assertIn("lhb_recent", out)
        self.assertIn("margin_recent", out)

    def test_norm_code_variants(self):
        """代码归一化：带后缀/前缀统一为 6 位。"""
        from quant_platform.openclaw_api import _norm_code
        self.assertEqual(_norm_code("600519.SH"), "600519")
        self.assertEqual(_norm_code("sh600519"), "600519")
        self.assertEqual(_norm_code("000001"), "000001")
        self.assertEqual(_norm_code("SZ000001"), "000001")

    def test_lhb_recent_no_crash(self):
        from quant_platform.openclaw_api import lhb_recent
        r = lhb_recent(5)
        # 返回 dict，不抛错（数据可能有/可能空）
        self.assertIsInstance(r, dict)

    def test_lhb_recent_column_detection(self):
        """龙虎榜列名探测：净买额列存在时应返回 top_net_buy 而非 note。

        保护突变：buy_col 探测逻辑被移除时（buy_col=None）本测试必须抓到。
        """
        import pandas as pd
        from quant_platform import openclaw_api as api
        # 构造含净买额列的数据，替换真实数据读取
        fake = pd.DataFrame({
            "代码": ["600519", "000001"],
            "名称": ["贵州茅台", "平安银行"],
            "龙虎榜净买额": [100.0, 50.0],
            "日期": ["2026-08-07", "2026-08-07"],
        })
        with patch.object(api._ds, "get_dataset", return_value=fake) as m:
            r = api.lhb_recent(5)
        m.assert_called()
        self.assertIn("top_net_buy", r, f"列探测失效: {r}")
        self.assertEqual(r["top_net_buy"][0]["代码"], "600519")

    def test_broker_activity(self):
        from quant_platform.openclaw_api import broker_activity
        r = broker_activity(5)
        self.assertIsInstance(r, dict)

    def test_market_panorama(self):
        from quant_platform.openclaw_api import market_panorama
        r = market_panorama()
        self.assertIsInstance(r, dict)

    def test_commodity_price(self):
        from quant_platform.openclaw_api import commodity_price
        r = commodity_price()
        self.assertIsInstance(r, dict)

    def test_all_datasets_status(self):
        from quant_platform.openclaw_api import all_datasets_status
        rows = all_datasets_status()
        self.assertGreater(len(rows), 30)


class TestAudit(unittest.TestCase):
    """审计维护模块（用户 #3 #6）。"""

    def test_data_audit_structure(self):
        from quant_platform.audit import _data_audit
        r = _data_audit()
        self.assertIn("findings", r)
        self.assertIn("count", r)
        for f in r["findings"]:
            self.assertIn("level", f)
            self.assertIn("msg", f)

    def test_maintenance_audit_finds_no_syntax_errors(self):
        """全仓语法检查应通过（刚修复的 2 个双 docstring bug 不再复发）。"""
        from quant_platform.audit import _maintenance_audit
        r = _maintenance_audit()
        syntax = [f for f in r["findings"] if "语法" in f.get("msg", "")]
        for f in syntax:
            self.assertNotEqual(f["level"], "high", f"存在语法错误: {f['msg']}")

    def test_py_compile_all_empty(self):
        from quant_platform.audit import _py_compile_all
        errs = _py_compile_all()
        self.assertEqual(errs, [], f"语法错误: {errs}")

    def test_output_audit_no_crash(self):
        from quant_platform.audit import _output_audit
        r = _output_audit()
        self.assertIn("findings", r)

    def test_run_full_audit_score_range(self):
        from quant_platform.audit import run_full_audit
        r = run_full_audit()
        self.assertGreaterEqual(r["score"], 0)
        self.assertLessEqual(r["score"], 100)
        self.assertIn("all_findings", r)
        self.assertIn("high_count", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMarketSegments(unittest.TestCase):
    """市场口径分层（用户 #11：主板/非ST/北交双创/科创创业）。"""

    def test_classify_board(self):
        from quant_platform.market_segments import classify_board
        self.assertEqual(classify_board("600519"), "main")
        self.assertEqual(classify_board("000001"), "main")
        self.assertEqual(classify_board("sz000001"), "main")
        self.assertEqual(classify_board("300750"), "gem")
        self.assertEqual(classify_board("301001"), "gem")
        self.assertEqual(classify_board("688981"), "star")
        self.assertEqual(classify_board("920982"), "bj")
        self.assertEqual(classify_board("830001"), "bj")
        self.assertEqual(classify_board("510300"), "other")

    def test_is_st(self):
        from quant_platform.market_segments import is_st
        self.assertTrue(is_st("*ST康美"))
        self.assertTrue(is_st("ST中安"))
        self.assertFalse(is_st("贵州茅台"))

    def test_segment_market_english_schema(self):
        """快照英文列（code/name/pct_chg/amount_wan）兼容。"""
        import pandas as pd
        from quant_platform.market_segments import segment_market
        df = pd.DataFrame({
            "code": ["600519", "sz000001", "300750", "688981", "920982"],
            "name": ["贵州茅台", "平安银行", "宁德时代", "中芯国际", "贝特瑞"],
            "pct_chg": [2.0, 1.0, 3.0, 4.0, 5.0],
            "amount_wan": [1000, 500, 300, 200, 100],
        })
        seg = segment_market(df)
        self.assertEqual(seg["主板"]["股票数"], 2)
        self.assertEqual(seg["创业板"]["股票数"], 1)
        self.assertEqual(seg["科创板"]["股票数"], 1)
        self.assertEqual(seg["北交所"]["股票数"], 1)
        self.assertAlmostEqual(seg["全A"]["等权涨跌幅%"], 3.0, places=1)

    def test_segment_market_filters_st(self):
        """ST 从主板/双创剔除，但保留在全A原始口径。"""
        import pandas as pd
        from quant_platform.market_segments import segment_market
        df = pd.DataFrame({
            "代码": ["600519", "000001", "600100"],
            "名称": ["贵州茅台", "平安银行", "ST中安"],
            "涨跌幅": [2.0, 1.0, -3.0],
        })
        seg = segment_market(df)
        self.assertEqual(seg["全A"]["股票数"], 3)
        self.assertEqual(seg["主板"]["股票数"], 2)  # ST被剔除

    def test_segment_industry(self):
        import pandas as pd
        from quant_platform.market_segments import segment_industry
        spot = pd.DataFrame({"代码": ["600519", "000001", "300750"],
                             "涨跌幅": [2.0, 1.0, 3.0]})
        ind = pd.DataFrame({"代码": ["600519", "000001", "300750"],
                            "industry": ["食品饮料", "银行", "电力设备"]})
        out = segment_industry(spot, ind)
        self.assertEqual(out["行业数"], 3)
        self.assertIn("Top5", out)


class TestSegmentsApi(unittest.TestCase):
    """/api/v1/segments 数据源逻辑（用户 #11）。"""

    def test_style_segments_english_schema(self):
        import pandas as pd
        from quant_platform.market_segments import style_segments
        df = pd.DataFrame({
            "code": ["600519", "000001", "300750", "688981", "002415"],
            "name": ["茅台", "平安", "宁德", "中芯", "海康"],
            "pct_chg": [2.0, 1.0, 3.0, 4.0, -1.0],
            "amount_wan": [9000, 7000, 5000, 3000, 1000],
        })
        out = style_segments(df)
        keys = [k for k in out if k != "meta"]
        self.assertEqual(len(keys), 3)
        self.assertIn("大票(成交额top33%)", keys)

    def test_short_medium_long(self):
        import pandas as pd
        from quant_platform.market_segments import short_medium_long
        spot = pd.DataFrame({"代码": ["600519", "000001"], "涨跌幅": [2.0, -1.0]})
        out = short_medium_long(spot)
        self.assertIn("短线_涨停梯队", out)
        self.assertEqual(out["短线_涨停梯队"]["涨停数(≥9.8%)"], 0)

    def test_valuation_temperature(self):
        import tempfile, os
        import pandas as pd
        from quant_platform.market_segments import valuation_temperature
        # 无文件时返回 error 不抛异常
        out = valuation_temperature([])
        self.assertIn("error", out)


class TestConceptFusion(unittest.TestCase):
    """概念/题材数据融合测试（V3 用户 #10：概念板块题材板块分类存储+复盘融合）。"""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from pathlib import Path
        # 直接用本地真实数据（已落地）
        root = Path(__file__).resolve().parent.parent
        cls.cb_path = root / "data_warehouse" / "classification" / "concept_board.parquet"
        cls.cm_path = root / "data_warehouse" / "classification" / "concept_member.parquet"
        cls.has_data = cls.cb_path.exists() and cls.cm_path.exists()

    def test_concept_board_present(self):
        if not self.has_data:
            self.skipTest("classification 数据未落地")
        import pandas as pd
        cb = pd.read_parquet(self.cb_path)
        self.assertGreaterEqual(len(cb), 100)  # 至少 100 个概念板块
        self.assertIn("board_name", cb.columns)

    def test_concept_member_mapping(self):
        if not self.has_data:
            self.skipTest("classification 数据未落地")
        import pandas as pd
        cm = pd.read_parquet(self.cm_path)
        self.assertGreaterEqual(len(cm), 1000)  # 至少 1000 行成分
        self.assertIn("concept", cm.columns)
        self.assertIn("code", cm.columns)

    def test_stock_concepts_returns_names(self):
        from quant_platform.openclaw_api import stock_concepts
        sc = stock_concepts("600519")
        # 概念应为中文名（BK 代码映射后），不应是 BK 开头
        self.assertIn("count", sc)
        for c in sc.get("concepts", [])[:3]:
            self.assertFalse(str(c).startswith("BK"))

    def test_concept_heatmap_structure(self):
        from quant_platform.openclaw_api import concept_heatmap
        h = concept_heatmap(5)
        self.assertIn("涨幅榜", h)
        self.assertIn("跌幅榜", h)
        self.assertTrue(len(h.get("涨幅榜", [])) <= 5)

    def test_concept_members_exact_match(self):
        from quant_platform.openclaw_api import concept_members
        rows = concept_members("CRO", 3)
        if rows and "error" not in rows[0]:
            # 精确匹配不应命中 MicroLED（子串误配）
            names = {r.get("concept_name") for r in rows}
            self.assertNotIn("MicroLED", names)

    def test_hot_theme_from_limit_up(self):
        """涨停股概念聚合（短线热点题材）：用真实快照验证不报错且结构正确。"""
        import glob
        import pandas as pd
        from quant_platform.market_segments import short_medium_long
        snaps = sorted(glob.glob("data_warehouse/realtime_snapshot/*/*.parquet"))
        if not snaps:
            self.skipTest("无实时快照")
        spot = pd.read_parquet(snaps[-1])
        spot = spot.rename(columns={"code": "代码", "pct_chg": "涨跌幅"})
        spot["代码"] = spot["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
        out = short_medium_long(spot)
        hot = out.get("短线_热点题材(涨停概念)")
        if hot:
            self.assertIn("Top概念", hot)
            for r in hot["Top概念"]:
                self.assertNotIn(r.get("concept", ""), {"融资融券", "沪股通"})
