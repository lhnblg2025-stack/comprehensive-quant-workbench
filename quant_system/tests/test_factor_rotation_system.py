"""analysis_core/factor_rotation_system 因子轮动系统单元测试。

覆盖:
  - 原始因子计算: 动量(20/60日) / 低波(20日std, 负值) / 规模(-ln流通市值)
  - 截面因子: z-score winsorize、动量=可用分量均值、价值EP/代理切换、质量缺失保留NaN
  - 因子收益: 截面分位多空组合(前20%多/后20%空)近20日收益
  - 截面IC: factor_zoo 复用 + 兜底
  - 拥挤度: 截面相关度 / 换手代理 / 数据不足标注
  - 综合: 占优因子组合 → 多/空/震荡 + 风格映射
  - TrendSystem 式接口: view 结构 / detect 异常降级 / report 写 md
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module("quant_system.analysis_core.factor_rotation_system")


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="factor_rotation_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


def _synth_close(n=100, trend=0.001, vol=0.01, start=10.0, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (start * np.cumprod(1 + trend + rng.normal(0, vol, n))).astype(float)


class TestRawFactors(unittest.TestCase):
    """原始因子: 动量/低波/规模。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_momentum_uptrend_positive(self):
        r = self.m._raw_factors_from_close(_synth_close(trend=0.003, n=80), share=1e8)
        self.assertIsNotNone(r["mom20"])
        self.assertIsNotNone(r["mom60"])
        self.assertGreater(r["mom20"], 0)
        self.assertGreater(r["mom60"], 0)

    def test_downtrend_negative(self):
        r = self.m._raw_factors_from_close(_synth_close(trend=-0.003, n=80), share=1e8)
        self.assertLess(r["mom60"], 0)

    def test_low_vol_negative_std(self):
        r_hi = self.m._raw_factors_from_close(_synth_close(vol=0.002, n=80), share=1e8)
        r_lo = self.m._raw_factors_from_close(_synth_close(vol=0.05, n=80), share=1e8)
        self.assertLess(r_hi["vol20"], 0)
        self.assertGreater(r_hi["vol20"], r_lo["vol20"])  # 低波股票 vol20 值更高(更接近0)

    def test_size_small_cap_higher(self):
        big = self.m._raw_factors_from_close(_synth_close(n=60), share=1e11)
        small = self.m._raw_factors_from_close(_synth_close(n=60), share=1e7)
        self.assertGreater(small["size"], big["size"])  # -ln(mcap): 小盘更高

    def test_insufficient_bars(self):
        r = self.m._raw_factors_from_close(_synth_close(n=10), share=1e8)
        self.assertIsNone(r["mom20"])
        self.assertIsNone(r["vol20"])


class TestWinsorizeZscore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_normalized_clipped(self):
        s = pd.Series(np.arange(100.0))
        z = self.m._winsorize_zscore(s)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=9)
        self.assertLessEqual(float(z.max()), 3.0)

    def test_constant(self):
        z = self.m._winsorize_zscore(pd.Series([1.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(z, 0.0))


class TestFactorCrossSection(unittest.TestCase):
    """截面因子: 动量合成 / 价值EP / 质量缺失 / 覆盖计数。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _stocks(self):
        return {
            "000001": {"close": _synth_close(trend=0.002, n=80, seed=1),
                       "share": 1e8, "fin": {"roe": 12.0, "eps": 0.8}},
            "000002": {"close": _synth_close(trend=-0.001, n=80, seed=2),
                       "share": 1e9, "fin": {"roe": 8.0, "eps": 0.3}},
            "000003": {"close": _synth_close(trend=0.0, n=80, seed=3),
                       "share": 1e7, "fin": {}},  # 质量/价值财务缺失
        }

    def test_factor_columns_present(self):
        stocks = self._stocks()
        f, cov = self.m._factor_cross_section(stocks, at_offset=0)
        for col in ("momentum", "value", "quality", "size", "low_vol"):
            self.assertIn(col, f.columns)
        self.assertEqual(cov["roe"], 2)  # 3 只中 2 只有 ROE

    def test_momentum_higher_for_uptrend(self):
        stocks = self._stocks()
        f, _ = self.m._factor_cross_section(stocks, at_offset=0)
        self.assertGreater(f.loc["000001", "momentum"], f.loc["000002", "momentum"])

    def test_quality_nan_preserved(self):
        stocks = self._stocks()
        f, _ = self.m._factor_cross_section(stocks, at_offset=0)
        self.assertTrue(pd.isna(f.loc["000003", "quality"]))


class TestFactorLSReturn(unittest.TestCase):
    """因子收益: 截面分位多空组合近20日收益。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_top_quantile_long_bottom_short(self):
        n = 100
        rng = np.random.default_rng(7)
        values = pd.Series(rng.normal(size=n), index=range(n))
        # 高因子值股票近20日收益更高 → 多空收益为正
        ret20 = pd.Series(values.values * 0.01 + rng.normal(0, 0.001, n), index=range(n))
        ls = self.m._factor_ls_return(values, ret20)
        self.assertGreater(ls, 0.0)

    def test_insufficient_sample_zero(self):
        values = pd.Series([1.0, 2.0, 3.0])
        ret20 = pd.Series([0.01, 0.02, 0.03])
        self.assertEqual(self.m._factor_ls_return(values, ret20, min_n=30), 0.0)


class TestCrowding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_crowding_available(self):
        rng = np.random.default_rng(3)
        n = 60
        f = pd.DataFrame({"momentum": rng.normal(size=n),
                          "value": rng.normal(size=n),
                          "low_vol": rng.normal(size=n)})
        c = self.m._crowding(f, f, {})
        self.assertTrue(c["available"])
        self.assertGreaterEqual(c["cross_corr"], 0.0)
        self.assertIsNotNone(c["turnover_proxy"])

    def test_insufficient_sample_labeled(self):
        f = pd.DataFrame({"momentum": [1.0, 2.0]})
        c = self.m._crowding(f, None, {})
        self.assertFalse(c["available"])
        self.assertIn("不足", c["note"])


class TestCompositeAndStyle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_bullish(self):
        top = [{"factor": "momentum", "ls_return": 0.05},
               {"factor": "size", "ls_return": 0.03},
               {"factor": "value", "ls_return": 0.02}]
        c = self.m._composite_signal(top)
        self.assertEqual(c["signal"], "多")
        s = self.m._style_suggestion(top, c)
        self.assertEqual(s["primary"]["style"], "强势题材/趋势成长")

    def test_bearish(self):
        top = [{"factor": "low_vol", "ls_return": -0.04},
               {"factor": "momentum", "ls_return": -0.02},
               {"factor": "quality", "ls_return": -0.01}]
        c = self.m._composite_signal(top)
        self.assertEqual(c["signal"], "空")
        s = self.m._style_suggestion(top, c)
        self.assertIn("防御", s["primary"]["style"])

    def test_mixed_neutral(self):
        top = [{"factor": "momentum", "ls_return": 0.03},
               {"factor": "value", "ls_return": -0.05},
               {"factor": "size", "ls_return": -0.02}]
        self.assertEqual(self.m._composite_signal(top)["signal"], "震荡")

    def test_style_map_all_factors(self):
        for factor in self.m.FACTOR_NAMES:
            self.assertIn(factor, self.m.STYLE_MAP)


class TestReadFinancial(unittest.TestCase):
    """财务读取: 截止日防前视。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_as_of_filters_recent(self):
        d = Path(tempfile.mkdtemp(prefix="fin_test_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        df = pd.DataFrame({"日期": pd.to_datetime(["2026-03-31", "2026-06-30"]),
                           "净资产收益率(%)": [10.0, 15.0],
                           "摊薄每股收益(元)": [0.5, 0.7]})
        path = d / "000001.parquet"
        df.to_parquet(path)
        # as_of 2026-05-20: 6-30 未披露(滞后45天截止04-05)，只能用 3-31
        out = self.m.read_financial(path, as_of=pd.Timestamp("2026-05-20"))
        self.assertEqual(out["roe"], 10.0)
        self.assertEqual(out["eps"], 0.5)


class TestViewInterface(_TmpDirMixin, unittest.TestCase):
    """view() multi_agent 兼容结构 / 异常降级 / report 写 md。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _mock_res(self):
        return {
            "date": "2026-08-11", "confidence": 0.7,
            "composite": {"signal": "多", "view": "看多", "weighted_ret": 0.05,
                          "note": "占优因子组合盈利"},
            "top_factors": [{"factor": "momentum", "name": "动量", "ls_return": 0.05,
                             "ic": 0.3, "rank": 1},
                            {"factor": "size", "name": "规模", "ls_return": 0.03,
                             "ic": 0.2, "rank": 2}],
            "factors": [{"factor": "momentum", "name": "动量", "ls_return": 0.05,
                         "ic": 0.3, "n_valid": 500, "data_ok": True, "zoo_factors": []},
                        {"factor": "size", "name": "规模", "ls_return": 0.03,
                         "ic": 0.2, "n_valid": 500, "data_ok": True, "zoo_factors": []}],
            "style": {"primary": {"style": "强势题材/趋势成长", "mode": "进攻",
                                  "desc": "追强势题材"},
                      "secondary": {"style": "小盘", "mode": "进攻", "desc": ""},
                      "note": "占优因子组合盈利"},
            "crowding": {"available": True, "cross_corr": 0.3, "turnover_proxy": 0.4,
                         "note": ""},
            "meta": {"status": "ok", "universe": "fullA_sample", "parsed": 500,
                     "sample_target": 500, "value_source": "财务EP(PE倒数)",
                     "elapsed_sec": 8.0, "factor_coverage": {}},
            "rag": {"query": "因子轮动 动量 价值 风格", "basis": "检索可用", "hits": []},
        }

    def test_view_structure(self):
        frs = self.m.FactorRotationSystem()
        with mock.patch.object(frs, "detect", return_value=self._mock_res()):
            v = frs.view()
        self.assertEqual(v["agent"], "因子轮动")
        for k in ("signal", "view", "confidence", "evidence", "weight", "status", "detail"):
            self.assertIn(k, v)
        self.assertEqual(v["signal"], "多")
        self.assertEqual(v["view"], "多")  # 词表映射: 看多 → multi_agent VIEW_SCORE 的 多
        self.assertIn(v["view"], {"多", "空", "震荡", "防守"})
        self.assertGreaterEqual(v["weight"], 0)
        self.assertEqual(v["detail"]["top_factors"][0], "momentum")

    def test_view_vocabulary_from_zh(self):
        frs = self.m.FactorRotationSystem()
        for zh, std in (("看多", "多"), ("看空", "空"), ("震荡", "震荡"), ("中性", "震荡"),
                        ("未知", "震荡")):
            self.assertEqual(self.m._signal_view(zh), std)
        for sig in ("多", "空"):
            res = self._mock_res()
            res["composite"] = {"signal": sig, "view": {"多": "看多", "空": "看空"}[sig]}
            with mock.patch.object(frs, "detect", return_value=res):
                v = frs.view()
            self.assertEqual(v["view"], sig)
            self.assertIn(v["view"], {"多", "空", "震荡", "防守"})

    def test_view_degrades_on_exception(self):
        frs = self.m.FactorRotationSystem()
        with mock.patch.object(frs, "detect", side_effect=RuntimeError("boom")):
            v = frs.view()
        self.assertEqual(v["status"], "degraded")
        self.assertEqual(v["signal"], "震荡")
        self.assertEqual(v["confidence"], 0.0)

    def test_report_writes_md(self):
        tmp = self._make_tmp()
        frs = self.m.FactorRotationSystem(out_dir=tmp)
        with mock.patch.object(frs, "detect", return_value=self._mock_res()):
            path = frs.report(out_dir=tmp)
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("# 因子轮动报告", text)
        self.assertIn("占优因子", text)

    def test_render_markdown_sections(self):
        md = self.m.render_markdown(self._mock_res())
        for sec in ("综合结论", "因子收益", "占优因子", "拥挤度", "风格映射", "RAG"):
            self.assertIn(sec, md)


if __name__ == "__main__":
    unittest.main()
