"""analysis_core/macro_learner 宏观规律自主学习器单元测试。

覆盖:
  - discover() 全链路（CCI/阈值/传导链/权重建议），全 mock parquet 数据驱动
  - 防前视硬断言：as_of ≤ 分析月-1；阈值训练先截断再 shift（不含未走完的 2026-08）
  - 样本不足降级（<24 个月 / 全文件缺失）
  - _welch_t 与 scipy 对比 / _betainc_approx / _pearson / 月份解析
  - load_latest_learner_result / format_ai_rules / _render_md / 落盘与历史去重
"""

from __future__ import annotations

import calendar
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

CONSUMER = ["食品饮料", "家用电器", "商贸零售", "医药生物", "社会服务", "美容护理"]
OTHER = ["有色金属", "电力设备", "汽车", "电子", "计算机", "基础化工", "银行",
         "非银金融", "房地产", "建筑装饰", "建筑材料", "钢铁", "煤炭", "石油石化",
         "公用事业", "交通运输", "国防军工", "通信", "传媒", "农林牧渔", "纺织服饰",
         "轻工制造", "机械设备", "环保", "综合"]
ALL_SECTORS = CONSUMER + OTHER  # 31 个申万一级


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.macro_learner as ml
    return ml


def _month_end(year: int, month: int) -> str:
    return f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _months(start=(2021, 1), end=(2026, 7)) -> list[tuple[int, int]]:
    out = []
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


class _TmpMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="macro_learner_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp

    def _patch_dirs(self, data_dir=None, out_dir=None):
        self.ml = _import()
        patchers = []
        if data_dir is not None:
            patchers.append(mock.patch.object(self.ml, "DATA_DIR", data_dir))
        if out_dir is not None:
            patchers.append(mock.patch.object(self.ml, "OUT_DIR", out_dir))
        for p in patchers:
            p.start()
        self.addCleanup(mock.patch.stopall)


def _build_dataset(tmp: Path, *, with_spike: bool = False,
                   cci_start=(2021, 1), cci_end=(2026, 7),
                   cci_high_from=(2023, 10)) -> dict:
    """构造完整 macro + 31 申万一级行业数据集（返回 CCI 月份→值映射便于断言）。"""
    months = _months(cci_start, cci_end)
    cci_map: dict[str, float] = {}
    rows = []
    for y, m in months:
        ym = f"{y}-{m:02d}"
        val = 100.0 if (y, m) >= cci_high_from else 80.0
        cci_map[ym] = val
        rows.append({"月份": ym, "消费者信心指数": val})
    # 防前视陷阱：分析月（2026-08）及其后数据必须被截断
    rows.append({"月份": "2026-08", "消费者信心指数": 130.0})
    rows.append({"月份": "2026-09", "消费者信心指数": 150.0})
    pd.DataFrame(rows).to_parquet(tmp / "macro" / "consumer_confidence.parquet", index=False)

    retail = [{"月份": ym, "同比增长": 2.0 + (cci_map[ym] - 80) * 0.1}
              for ym in cci_map]
    retail += [{"月份": "2026-08", "同比增长": 99.0}]
    pd.DataFrame(retail).to_parquet(tmp / "macro" / "retail_sales_yoy.parquet", index=False)

    hp = []
    for ym in cci_map:
        for city in ("北京", "上海"):
            hp.append({"日期": _month_end(*map(int, ym.split("-"))),
                       "城市": city,
                       "新建商品住宅价格指数-同比": 0.5 + (cci_map[ym] - 80) * 0.05})
    hp.append({"日期": "2026-08-10", "城市": "北京",
               "新建商品住宅价格指数-同比": 20.0})
    pd.DataFrame(hp).to_parquet(tmp / "macro" / "new_house_price.parquet", index=False)

    # 行业：31 个申万一级；食品饮料月收益随 CCI 高低分组 ±3%（加微小噪声保证组内方差>0）
    closes: dict[str, list] = {"代码": [], "日期": [], "收盘": []}
    rng = np.random.default_rng(42)
    food_close = 100.0
    for y, m in months:
        ym = f"{y}-{m:02d}"
        date = _month_end(y, m)
        hi = (y, m) >= cci_high_from
        food_ret = (0.03 if hi else -0.03) + rng.uniform(-0.002, 0.002)
        food_close *= 1.0 + food_ret
        for i, sec in enumerate(ALL_SECTORS):
            closes["代码"].append(f"{801000 + i}")
            closes["日期"].append(date)
            closes["收盘"].append(round(food_close, 4) if sec == "食品饮料" else 100.0)
    if with_spike:
        # 2026-08 不完整月：食品饮料 +1000%（若泄漏进阈值训练会严重抬高高信心组均值）
        spike_close = food_close * 11.0
        for i, sec in enumerate(ALL_SECTORS):
            closes["代码"].append(f"{801000 + i}")
            closes["日期"].append("2026-08-10")
            closes["收盘"].append(spike_close if sec == "食品饮料" else 100.0)
    pd.DataFrame(closes).to_parquet(tmp / "industry" / "sw_first_hist.parquet", index=False)

    sw = [{"行业代码": f"{801000 + i}.SI", "行业名称": sec}
          for i, sec in enumerate(ALL_SECTORS)]
    pd.DataFrame(sw).to_parquet(tmp / "industry" / "sw_first.parquet", index=False)
    return cci_map


class TestDiscoverFullChain(_TmpMixin, unittest.TestCase):
    """discover() 全链路：2026-08-11 分析日 → as_of=2026-07，阈值/传导/权重建议齐备。"""

    def _run(self, with_spike=False):
        tmp = self._make_tmp()
        (tmp / "macro").mkdir(parents=True)
        (tmp / "industry").mkdir(parents=True)
        cci_map = _build_dataset(tmp, with_spike=with_spike)
        self._patch_dirs(data_dir=tmp)
        res = self.ml.discover("2026-08-11", write=False)
        return res, cci_map

    def test_as_of_no_lookahead(self):
        res, _ = self._run()
        self.assertEqual(res["date"], "2026-08-11")
        self.assertEqual(res["analysis_month"], "2026-08")
        # 硬断言：月度宏观滞后 1 个月 → as_of 必须为分析月 - 1 个月
        self.assertEqual(res["as_of_month"], "2026-07")
        # CCI 只取 ≤2026-07：即使 parquet 里有 2026-08/2026-09 陷阱行
        self.assertEqual(res["cci"]["latest_month"], "2026-07")
        self.assertEqual(res["cci"]["latest_cci"], 100.0)
        self.assertNotIn("CCI 数据不足", res["degraded"])

    def test_thresholds_and_consumer_adjustments(self):
        res, _ = self._run()
        self.assertTrue(res["thresholds"])
        food = [t for t in res["thresholds"] if t["sector"] == "食品饮料"]
        self.assertTrue(food, "食品饮料应触发阈值规律")
        th = food[0]
        self.assertLess(th["p_value"], 0.10)
        self.assertGreaterEqual(th["n_pos"], 24)
        self.assertGreaterEqual(th["n_neg"], 24)
        self.assertEqual(th["direction"], "高信心利好")
        # 当前 CCI=100 ≥ 阈值 → 高信心侧 → 增配
        adj = [a for a in res["sector_adjustments"] if a["sector"] == "食品饮料"]
        self.assertTrue(adj)
        a = adj[0]
        self.assertEqual(a["trigger_side"], "high")
        self.assertEqual(a["direction"], "高信心利好")
        self.assertGreater(a["factor"], 1.0)
        self.assertLessEqual(a["factor"], 1.15)
        self.assertIn("增配", a["reason"])

    def test_transmission_chain_present(self):
        res, _ = self._run()
        tr = res["transmission"]
        self.assertIn("same_month", tr["house_price_to_cci"])
        self.assertIn("same_month", tr["cci_to_retail"])
        self.assertIn("lead_1m", tr["cci_to_retail"])
        r2c = tr["retail_to_consumer"]
        self.assertIn("食品饮料", r2c)
        self.assertTrue(all(sec in r2c for sec in CONSUMER))
        # 数据同源构造 → 相关性应显著为正（弱断言：r>0 且 n 足够）
        self.assertGreater(tr["cci_to_retail"]["same_month"]["r"], 0.5)
        self.assertGreaterEqual(tr["cci_to_retail"]["same_month"]["n"], 36)
        self.assertGreater(tr["house_price_to_cci"]["same_month"]["r"], 0.5)

    def test_current_point_observation(self):
        res, _ = self._run()
        self.assertEqual(res["cci"]["latest_cci"], 100.0)
        self.assertEqual(res["cci"]["mom_chg"], 0.0)  # 2026-06 与 2026-07 均为 100
        self.assertEqual(res["cci"]["pct_rank"], round(33 / 67, 3))
        self.assertAlmostEqual(res["cci"]["vs_hist_mean"], 100.0 - 108.47, places=2)

    def test_incomplete_month_spike_excluded_from_training(self):
        """2026-08 未走完月份出现 +1000% 异常 → 不得进入阈值训练（先截断再 shift）。"""
        base, _ = self._run(with_spike=False)
        spk, _ = self._run(with_spike=True)
        self.assertEqual(spk["as_of_month"], "2026-07")
        base_th = next(t for t in base["thresholds"] if t["sector"] == "食品饮料")
        spk_th = next(t for t in spk["thresholds"] if t["sector"] == "食品饮料")
        # 若 2026-08 的 +1000% 泄漏，pos_excess 会从 ~2.9% 跳升到 ~31%
        self.assertLess(abs(spk_th["pos_excess"] - base_th["pos_excess"]), 0.005)
        # 加入 spike 只新增一条合法历史对（CCI 2026-06 → 超额 2026-07）：n_pos 恰 +1
        self.assertEqual(spk_th["n_pos"], base_th["n_pos"] + 1)
        self.assertEqual(spk_th["n_neg"], base_th["n_neg"])
        self.assertLess(spk_th["p_value"], 0.10)

    def test_write_outputs_and_history_dedup(self):
        tmp = self._make_tmp()
        out = self._make_tmp()
        (tmp / "macro").mkdir(parents=True)
        (tmp / "industry").mkdir(parents=True)
        _build_dataset(tmp)
        self._patch_dirs(data_dir=tmp, out_dir=out)
        self.ml.discover("2026-08-11", write=True)
        self.ml.discover("2026-08-11", write=True)  # 同日重跑 → 历史去重
        self.assertTrue((out / "macro_learner_2026-08-11.json").exists())
        self.assertTrue((out / "macro_learner_2026-08-11.md").exists())
        hist = pd.read_parquet(out / "macro_learner_history.parquet")
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist.iloc[0]["date"], "2026-08-11")
        saved = json.loads((out / "macro_learner_2026-08-11.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["analysis_month"], "2026-08")
        self.assertEqual(saved["as_of_month"], "2026-07")


class TestDiscoverDegraded(_TmpMixin, unittest.TestCase):
    def test_insufficient_cci_samples(self):
        tmp = self._make_tmp()
        (tmp / "macro").mkdir(parents=True)
        (tmp / "industry").mkdir(parents=True)
        _build_dataset(tmp)  # 先建全量行业 + 房价
        # 再用仅 12 个月 CCI/社零 覆盖（2025-08..2026-07），行业数据保持完整
        months = _months((2025, 8), (2026, 7))
        cci = [{"月份": f"{y}-{m:02d}", "消费者信心指数": 100.0} for y, m in months]
        pd.DataFrame(cci).to_parquet(tmp / "macro" / "consumer_confidence.parquet", index=False)
        pd.DataFrame({"月份": [f"{y}-{m:02d}" for y, m in months],
                      "同比增长": [5.0] * len(months)}).to_parquet(
            tmp / "macro" / "retail_sales_yoy.parquet", index=False)
        self._patch_dirs(data_dir=tmp)
        res = self.ml.discover("2026-08-11", write=False)
        self.assertTrue(any("CCI 样本 < 24 个月" in d for d in res["degraded"]))
        self.assertEqual(res["thresholds"], [])
        # 样本不足仍出点位观察
        self.assertEqual(res["cci"]["latest_cci"], 100.0)
        self.assertEqual(res["cci"]["latest_month"], "2026-07")

    def test_all_sources_missing_degrades_gracefully(self):
        tmp = self._make_tmp()
        self._patch_dirs(data_dir=tmp)
        res = self.ml.discover("2026-08-11", write=False)
        for key in ("CCI 数据不足", "社零数据不足", "房价数据不足", "行业收益不足"):
            self.assertTrue(any(key in d for d in res["degraded"]),
                            f"degraded 应包含: {key}")
        self.assertEqual(res["thresholds"], [])
        self.assertEqual(res["correlations"], [])
        self.assertEqual(res["sector_adjustments"], [])
        self.assertIsNone(res["cci"]["latest_cci"])
        self.assertIn("degraded", res["transmission"]["cci_to_retail"])
        self.assertIn("degraded", res["transmission"]["retail_to_consumer"])


class TestWelchT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml = _import()

    def test_matches_scipy(self):
        from scipy import stats
        x = np.array([1.0, 2.0, 3.5, 4.0, 5.2, 6.0])
        y = np.array([10.0, 11.5, 12.0, 13.0, 14.5])
        t, p = self.ml._welch_t(x, y)
        s_t, s_p = stats.ttest_ind(x, y, equal_var=False)
        self.assertAlmostEqual(t, float(s_t), places=10)
        self.assertAlmostEqual(p, float(s_p), places=10)

    def test_identical_samples(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t, p = self.ml._welch_t(x, x.copy())
        self.assertAlmostEqual(t, 0.0, places=10)
        self.assertAlmostEqual(p, 1.0, places=10)

    def test_insufficient_samples(self):
        t, p = self.ml._welch_t(np.array([1.0]), np.array([2.0, 3.0]))
        self.assertTrue(np.isnan(t))
        self.assertEqual(p, 1.0)

    def test_betainc_approx_sanity(self):
        """回退路径不精确（见 test_betainc_approx_known_bug），
        仅断言按构造成立的结构性质：端点与对称性。"""
        self.assertEqual(self.ml._betainc_approx(2.0, 0.5, 0.0), 0.0)
        self.assertEqual(self.ml._betainc_approx(2.0, 0.5, 1.0), 1.0)
        self.assertEqual(self.ml._betainc_approx(0.5, 2.0, 1.0), 1.0)
        # 对称性：I_x(a,b) = 1 - I_(1-x)(b,a)
        for a, b in [(2.0, 0.5), (5.0, 5.0), (10.0, 0.5), (3.0, 2.0), (0.5, 0.5)]:
            self.assertAlmostEqual(self.ml._betainc_approx(a, b, 0.3),
                                   1.0 - self.ml._betainc_approx(b, a, 0.7),
                                   places=6)

    def test_betainc_approx_matches_scipy(self):
        """域G 修复后：回退路径与 scipy 一致（相对误差 <1e-3），不再偏大 a 倍。"""
        from scipy import special
        for a, b, x in [(5.0, 5.0, 0.3), (3.0, 2.0, 0.5), (2.0, 0.5, 0.3),
                        (10.0, 0.5, 0.01), (0.5, 2.0, 0.99), (5.0, 0.5, 0.5)]:
            got = self.ml._betainc_approx(a, b, x)
            exp = float(special.betainc(a, b, x))
            self.assertGreaterEqual(got, 0.0)
            self.assertLessEqual(got, 1.0)
            rel = abs(got - exp) / max(exp, 1e-12)
            self.assertLess(rel, 1e-3, f"(a={a},b={b},x={x}) rel={rel}")


class TestParsersAndStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml = _import()

    def test_ts_month_formats(self):
        self.assertEqual(self.ml._ts_month("2026年08月"), pd.Timestamp("2026-08-01"))
        self.assertEqual(self.ml._ts_month("2026-8月"), pd.Timestamp("2026-08-01"))
        self.assertEqual(self.ml._ts_month("2026-08"), pd.Timestamp("2026-08-01"))
        self.assertEqual(self.ml._ts_month("202608"), pd.Timestamp("2026-08-01"))
        self.assertEqual(self.ml._ts_month(pd.Timestamp("2026-08-15")),
                         pd.Timestamp("2026-08-01"))
        self.assertIsNone(self.ml._ts_month(None))
        self.assertIsNone(self.ml._ts_month("garbage"))
        self.assertEqual(self.ml._ym("2026年08月"), "2026-08")
        self.assertIsNone(self.ml._ym(None))

    def test_month_series_truncation_and_dedup(self):
        df = pd.DataFrame({"月份": ["2026-06", "2026-07", "2026-08", "2026-07"],
                           "数值": [1.0, 2.0, 3.0, 99.0]})
        s = self.ml._month_series(df, "月份", "数值", max_ym="2026-07")
        self.assertEqual(list(s.index), ["2026-06", "2026-07"])
        self.assertEqual(s["2026-07"], 99.0)  # 重复月保留最后
        self.assertEqual(self.ml._month_series(None, "月份", "数值").empty, True)
        self.assertEqual(self.ml._month_series(
            pd.DataFrame(), "月份", "数值").empty, True)

    def test_pearson(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"])
        b = pd.Series([2.0, 4.0, 6.0, 8.0], index=["a", "b", "c", "d"])
        r, n = self.ml._pearson(a, b)
        self.assertAlmostEqual(r, 1.0, places=10)
        self.assertEqual(n, 4)
        r, n = self.ml._pearson(a[:1], b[:1])
        self.assertIsNone(r)
        self.assertEqual(n, 0)

    def test_format_ai_rules(self):
        res = {"thresholds": [
            {"sector": "食品饮料", "threshold": 90, "direction": "高信心利好",
             "pos_excess": 0.03, "neg_excess": -0.03, "n_pos": 30, "n_neg": 30,
             "p_value": 0.01},
            {"sector": "家用电器", "threshold": 85, "direction": "低信心利空",
             "pos_excess": -0.02, "neg_excess": 0.02, "n_pos": 31, "n_neg": 29,
             "p_value": 0.02}],
            "sector_adjustments": [
                {"sector": "食品饮料", "factor": 1.05, "reason": "增配"}]}
        rules = self.ml.format_ai_rules(res, max_rules=2)
        self.assertEqual(len(rules), 2)
        self.assertIn("CCI≥90", rules[0])
        self.assertIn("下月超额+3.0%", rules[0])
        self.assertIn("CCI<85", rules[1])
        self.assertEqual(self.ml.format_ai_rules(None), [])
        self.assertEqual(self.ml.format_ai_rules({}), [])

    def test_load_latest_learner_result(self):
        out = Path(tempfile.mkdtemp(prefix="ml_result_"))
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        (out / "macro_learner_2026-08-01.json").write_text(
            json.dumps({"date": "2026-08-01", "k": "old"}), encoding="utf-8")
        (out / "macro_learner_2026-08-05.json").write_text(
            json.dumps({"date": "2026-08-05", "k": "new"}), encoding="utf-8")
        (out / "macro_learner_bad.json").write_text("{corrupt", encoding="utf-8")
        with mock.patch.object(self.ml, "OUT_DIR", out):
            self.assertEqual(self.ml.load_latest_learner_result("2026-08-03")["date"],
                             "2026-08-01")
            self.assertEqual(self.ml.load_latest_learner_result(None)["date"],
                             "2026-08-05")
            self.assertIsNone(self.ml.load_latest_learner_result("2026-07-31"))
        # 目录不可读/损坏 → None（异常吞掉）
        with mock.patch.object(self.ml, "OUT_DIR", out / "nope"):
            self.assertIsNone(self.ml.load_latest_learner_result("2026-08-10"))

    def test_render_md(self):
        result = {
            "date": "2026-08-11", "analysis_month": "2026-08",
            "as_of_month": "2026-07", "generated_at": "2026-08-11T10:00:00+08:00",
            "cci": {"latest_cci": 100.0, "latest_month": "2026-07", "mom_chg": 1.2,
                    "pct_rank": 0.49, "vs_hist_mean": -8.47, "hist_mean": 108.47},
            "thresholds": [{"sector": "食品饮料", "threshold": 90, "direction": "高信心利好",
                            "pos_excess": 0.03, "neg_excess": -0.03, "n_pos": 30,
                            "n_neg": 30, "p_value": 0.01}],
            "sector_adjustments": [{"sector": "食品饮料", "factor": 1.05,
                                    "reason": "高信心侧增配"}],
            "transmission": {
                "house_price_to_cci": {"same_month": {"r": 0.8, "n": 60},
                                       "lead_1m": {"r": 0.5, "n": 59}},
                "cci_to_retail": {"same_month": {"r": 0.9, "n": 60},
                                  "lead_1m": {"r": 0.6, "n": 59}},
                "retail_to_consumer": {"食品饮料": {"r": 0.7, "n": 59}}},
            "correlations": [{"sector": "食品饮料", "same_month_r": 0.8,
                              "same_month_n": 60, "next_month_r": 0.5,
                              "next_month_n": 59}],
            "degraded": ["测试降级说明"],
        }
        md = self.ml._render_md(result)
        for key in ("分析月 2026-08", "as_of（宏观滞后1月）2026-07",
                    "CCI≥90", "板块权重建议", "传导链", "降级说明", "测试降级说明"):
            self.assertIn(key, md)


if __name__ == "__main__":
    unittest.main()

