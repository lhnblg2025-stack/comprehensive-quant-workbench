"""analysis_core/chart_pattern_system 图表形态系统单元测试。

覆盖:
  - _norm_kline 规整（补 OHLC/去重/排序/≤target 截断）
  - resolve_target: --date 指定 / 共同日期 / 无文件兜底
  - pivots fractal 摆动点 + 同类型合并、vol_ratio_last 量比（截断/clip/缺样本）
  - 形态识别: 头肩顶/底、双顶/底、上升/下降三角形、旗形 + 各排除分支
  - detect_patterns: 样本<60日标注、seq<3 空、60日新鲜度过滤、同形态去重
  - detect_breakouts: Murphy 3% 确认/待确认/量比不足/无效颈线
  - 斐波那契: _major_swing up/down/兜底、fib_analysis 全位置分支
  - volume_profile_note 密集区降级链（缺数据/异常→标注不抛）
  - compose_view 多/空/防守/震荡 + 置信钳制、detect_symbol 单标的
  - ChartPatternSystem: _name_map 缓存、_rag 检索可用/无命中/异常降级、
    detect 全跳过分支（缺文件/空/无数据/失败）、report 落盘、view 单标的/多数票/降级
  - render_markdown / _fmt_levels / main CLI

无网络: 全部本地 mock；KLINE_DIR 等路径常量 mock.patch.object 到 tmp。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.chart_pattern_system as cp
    return cp


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="chart_pattern_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


def _kline(n, start=100.0, end=None, volumes=None, cols=("date", "open", "high", "low", "close", "volume")):
    """日K构造: 默认收盘从 start 线性到 end(=start 时持平)。"""
    end = start if end is None else end
    closes = list(np.linspace(start, end, n))
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "open": c, "high": c * 1.01, "low": c * 0.99,
            "close": c,
            "volume": 100.0 if volumes is None else volumes[i],
        })
    return pd.DataFrame(rows)[list(cols)]


class TestNormKline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_none_and_empty(self):
        self.assertIsNone(self.cp._norm_kline(None, pd.Timestamp("2026-08-01")))
        self.assertIsNone(self.cp._norm_kline(pd.DataFrame(), pd.Timestamp("2026-08-01")))

    def test_fills_ohlc_and_truncates(self):
        df = pd.DataFrame({
            "date": ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-03", "2026-08-10"],
            "close": [10, 11, 12, 13, 99],
            "volume": [100, 100, 100, 100, 100],
        })
        out = self.cp._norm_kline(df, pd.Timestamp("2026-07-03"))
        self.assertLessEqual(len(out), 3)
        self.assertTrue((out["date"] <= pd.Timestamp("2026-07-03")).all())
        # 去重保留最后一条
        self.assertEqual(len(out), 3)
        for c in ("open", "high", "low"):
            self.assertTrue(out[c].fillna(-1).eq(out["close"]).all())
        # NaN date/close 被剔除
        df2 = df.copy()
        df2.loc[0, "date"] = None
        df2.loc[1, "close"] = None
        out2 = self.cp._norm_kline(df2, pd.Timestamp("2026-08-10"))
        self.assertEqual(len(out2), 2)  # 2 行 NaN 剔除 + 同日去重

    def test_all_dates_beyond_target_returns_none(self):
        df = pd.DataFrame({"date": ["2026-08-10"], "close": [1]})
        self.assertIsNone(self.cp._norm_kline(df, pd.Timestamp("2026-07-01")))


class TestResolveTarget(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_explicit_date(self):
        self.assertEqual(self.cp.resolve_target(["600001"], "2026-08-10"),
                         pd.Timestamp("2026-08-10").normalize())

    def test_common_date_from_files(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        pd.DataFrame({"date": ["2026-08-01", "2026-08-08"]}).to_parquet(kdir / "600001.parquet")
        pd.DataFrame({"date": ["2026-08-01", "2026-08-10"]}).to_parquet(kdir / "600002.parquet")
        with mock.patch.object(self.cp, "KLINE_DIR", kdir):
            target = self.cp.resolve_target(["600001", "600002"])
        self.assertEqual(target, pd.Timestamp("2026-08-08").normalize())

    def test_no_files_falls_back_today(self):
        tmp = self._make_tmp()
        with mock.patch.object(self.cp, "KLINE_DIR", tmp / "nope"):
            target = self.cp.resolve_target(["600001"])
        self.assertEqual(target, pd.Timestamp.now().normalize())


class TestPivotsAndVolume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_pivots_detects_peak_valley(self):
        highs = np.array([3, 4, 5, 6, 7, 8, 7, 6, 5, 4], dtype=float)
        lows = np.array([7, 6, 5, 4, 3, 2, 3, 4, 5, 6], dtype=float)
        pts = self.cp.pivots(highs, lows, k=2)
        self.assertIn((5, 8.0, "H"), pts)
        self.assertIn((5, 2.0, "L"), pts)

    def test_pivots_merge_same_type(self):
        highs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)  # 严格上升 → 无 H 摆动
        lows = np.array([9, 8, 7, 5, 5, 7, 8, 9], dtype=float)   # 相邻双谷 → 合并取后者
        pts = self.cp.pivots(highs, lows, k=2)
        self.assertEqual(pts, [(4, 5.0, "L")])

    def test_pivots_too_short(self):
        self.assertEqual(self.cp.pivots(np.array([1.0, 2.0]), np.array([1.0, 2.0]), k=2), [])

    def test_vol_ratio_too_short(self):
        df = _kline(3, volumes=[100, 100, 100])
        self.assertEqual(self.cp.vol_ratio_last(df), 1.0)

    def test_vol_ratio_non_finite_last(self):
        df = _kline(6, volumes=[100, 100, 100, 100, 100, np.nan])
        self.assertEqual(self.cp.vol_ratio_last(df), 1.0)

    def test_vol_ratio_unit_switch_break(self):
        df = _kline(4, volumes=[100, 100, 2500, 10])
        self.assertEqual(self.cp.vol_ratio_last(df), 1.0)  # 尾段 <3 样本

    def test_vol_ratio_normal_clipped(self):
        df = _kline(6, volumes=[100, 90, 80, 70, 60, 120])
        r = self.cp.vol_ratio_last(df)
        self.assertAlmostEqual(r, 120 / 80.0, places=3)
        df2 = _kline(4, volumes=[1000, 1000, 1000, 10])
        self.assertEqual(self.cp.vol_ratio_last(df2), self.cp.VOL_RATIO_FLOOR)

    def test_vol_at_and_median(self):
        df = _kline(6, volumes=[10, 20, 30, 40, 50, 60])
        self.assertEqual(self.cp._vol_at(df, 3), 40.0)
        self.assertTrue(np.isnan(self.cp._vol_at(pd.DataFrame(
            {"volume": [10, 0]}, index=[0, 1]), 1)))
        self.assertEqual(self.cp._median_vol(df, 0, 3), 20.0)
        self.assertTrue(np.isnan(self.cp._median_vol(df, 0, 0)))


def _seq_top_head_shoulder():
    return [(0, 10.0, "H"), (3, 8.5, "L"), (6, 12.0, "H"), (9, 8.6, "L"), (12, 11.0, "H")]


class TestPatternDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def _df(self, n=13, volumes=None):
        vols = [100.0] * n if volumes is None else volumes
        return _kline(n, volumes=vols)

    def test_head_shoulders_top(self):
        vols = [100.0] * 13
        vols[6], vols[12] = 100.0, 50.0
        out = self.cp._detect_head_shoulders(_seq_top_head_shoulder(), self._df(volumes=vols))
        self.assertEqual(len(out), 1)
        p = out[0]
        self.assertEqual(p["name"], "头肩顶")
        self.assertEqual(p["direction"], "空")
        self.assertEqual(p["neckline"], 8.55)
        self.assertIn("右肩量缩", p["note"])
        self.assertEqual(p["pivots"], [0, 3, 6, 9, 12])

    def test_head_shoulders_bottom(self):
        seq = [(0, 10.0, "L"), (3, 12.0, "H"), (6, 8.6, "L"), (9, 12.0, "H"), (12, 10.0, "L")]
        out = self.cp._detect_head_shoulders(seq, self._df())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "头肩底")
        self.assertEqual(out[0]["direction"], "多")
        self.assertEqual(out[0]["neckline"], 12.0)

    def test_head_shoulders_shoulder_too_low(self):
        seq = [(0, 5.0, "H"), (3, 8.5, "L"), (6, 12.0, "H"), (9, 8.6, "L"), (12, 11.0, "H")]
        self.assertEqual(self.cp._detect_head_shoulders(seq, self._df()), [])

    def test_head_shoulders_wrong_alternation(self):
        seq = [(0, 10.0, "H"), (3, 9.0, "H"), (6, 12.0, "H"), (9, 8.6, "L"), (12, 11.0, "H")]
        self.assertEqual(self.cp._detect_head_shoulders(seq, self._df()), [])

    def test_double_bottom(self):
        seq = [(0, 10.0, "L"), (5, 12.0, "H"), (10, 10.4, "L")]
        vols = [100.0] * 11
        vols[0], vols[10] = 100.0, 60.0
        out = self.cp._detect_double(seq, _kline(11, volumes=vols))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "双底")
        self.assertIn("第二底量能萎缩", out[0]["note"])

    def test_double_top(self):
        seq = [(0, 10.0, "H"), (5, 8.0, "L"), (10, 9.8, "H")]
        out = self.cp._detect_double(seq, self._df(11))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "双顶")
        self.assertEqual(out[0]["direction"], "空")

    def test_double_neck_too_low(self):
        seq = [(0, 10.0, "L"), (5, 10.2, "H"), (10, 10.4, "L")]
        self.assertEqual(self.cp._detect_double(seq, self._df(11)), [])

    def test_triangles(self):
        seq_up = [(0, 10.0, "H"), (2, 8.0, "L"), (5, 10.2, "H"), (8, 9.0, "L")]
        out = self.cp._detect_triangles(seq_up)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "上升三角形")
        self.assertEqual(out[0]["direction"], "多")
        seq_dn = [(0, 10.0, "L"), (2, 12.0, "H"), (5, 9.9, "L"), (8, 11.0, "H")]
        out2 = self.cp._detect_triangles(seq_dn)
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0]["name"], "下降三角形")
        self.assertEqual(out2[0]["direction"], "空")

    def test_triangle_rejections(self):
        # 上边不水平
        seq = [(0, 10.0, "H"), (2, 8.0, "L"), (5, 11.0, "H"), (8, 9.0, "L")]
        self.assertEqual(self.cp._detect_triangles(seq), [])
        # 下边不抬高
        seq2 = [(0, 10.0, "H"), (2, 8.0, "L"), (5, 10.1, "H"), (8, 8.0, "L")]
        self.assertEqual(self.cp._detect_triangles(seq2), [])
        # 跨度 <8
        seq3 = [(0, 10.0, "H"), (2, 8.0, "L"), (3, 10.1, "H"), (4, 9.0, "L")]
        self.assertEqual(self.cp._detect_triangles(seq3), [])

    def test_flag_bull(self):
        closes = [100, 101, 102, 104, 106, 108, 110, 110, 110.5, 111, 111, 110.8]
        vols = [100.0] * 12
        for i in range(7, 12):
            vols[i] = 30.0
        df = _kline(12, start=100, end=110, volumes=vols)
        df["close"] = closes
        df["high"] = [c * 1.005 for c in closes]
        df["low"] = [c * 0.995 for c in closes]
        out = self.cp._detect_flag(df)
        self.assertIsNotNone(out)
        self.assertEqual(out["name"], "旗形")
        self.assertEqual(out["direction"], "多")
        self.assertIn("整理缩量", out["note"])

    def test_flag_bear(self):
        closes = [110, 109, 107, 105, 103, 101, 100, 100.5, 100, 100.8, 100.2, 100.5]
        vols = [100.0] * 12
        for i in range(7, 12):
            vols[i] = 30.0
        df = _kline(12, start=100, end=110, volumes=vols)
        df["close"] = closes
        df["high"] = [c * 1.005 for c in closes]
        df["low"] = [c * 0.995 for c in closes]
        out = self.cp._detect_flag(df)
        self.assertIsNotNone(out)
        self.assertEqual(out["direction"], "空")

    def test_flag_flat_returns_none(self):
        df = _kline(20)
        self.assertIsNone(self.cp._detect_flag(df))


_HS_PIVOTS = [
    # (idx, high, low, close, volume)
    (50, 10.0, 8.5, 9.0, 100.0),   # 左肩
    (51, 9.0, 8.8, 8.9, 100.0),
    (52, 9.5, 9.2, 9.3, 100.0),
    (53, 10.0, 9.5, 9.8, 100.0),
    (54, 11.0, 9.8, 10.5, 100.0),
    (55, 11.5, 10.0, 11.0, 100.0),
    (56, 12.0, 10.0, 11.5, 100.0),  # 头
    (57, 11.0, 10.0, 10.8, 100.0),
    (58, 10.5, 9.5, 10.2, 100.0),
    (59, 10.0, 8.6, 9.2, 100.0),
    (60, 9.5, 9.0, 9.2, 100.0),
    (61, 10.5, 10.0, 10.2, 100.0),
    (62, 11.0, 10.0, 10.6, 50.0),   # 右肩（量缩）
    (63, 10.0, 9.5, 9.8, 100.0),
    (64, 9.5, 9.0, 9.3, 100.0),
]


def _pattern_df(n=80):
    rows = []
    for i in range(n):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0})
    for idx, h, lo, cl, v in _HS_PIVOTS:
        rows[idx]["high"] = h
        rows[idx]["low"] = lo
        rows[idx]["close"] = cl
        rows[idx]["volume"] = v
    return pd.DataFrame(rows)


class TestDetectPatternsAndBreakouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_short_sample_note(self):
        df = _kline(10)
        patterns, note = self.cp.detect_patterns(df)
        self.assertIn("形态样本<60日", note)
        self.assertIn("仅10日", note)

    def test_flat_no_pivots(self):
        df = _kline(70)
        patterns, note = self.cp.detect_patterns(df)
        self.assertEqual(patterns, [])
        self.assertEqual(note, "")

    def test_detect_head_shoulders_via_main(self):
        df = _pattern_df()
        patterns, note = self.cp.detect_patterns(df)
        self.assertEqual(note, "")
        names = [p["name"] for p in patterns]
        self.assertIn("头肩顶", names)
        # 新鲜度过滤 + 去重: 同形态名+方向只保留一条
        self.assertEqual(sum(1 for p in patterns if p["name"] == "头肩顶"), 1)

    def test_stale_pattern_filtered(self):
        df = _kline(80)
        # 把完整形态块平移到 idx 0..14（pivot 最远 idx 12, last_idx 79 → 67 > 60）→ 被过滤
        for idx, h, lo, cl, v in _HS_PIVOTS:
            j = idx - 50
            df.loc[j, "high"] = h
            df.loc[j, "low"] = lo
            df.loc[j, "close"] = cl
            df.loc[j, "volume"] = v
        patterns, _ = self.cp.detect_patterns(df)
        self.assertNotIn("头肩顶", [p["name"] for p in patterns])

    def test_breakouts_bull(self):
        df = _kline(5)
        df["close"] = [100.0] * 4 + [110.0]
        patterns = [{"name": "双底", "direction": "多", "neckline": 100.0, "pivots": [0, 1, 2]}]
        out = self.cp.detect_breakouts(df, patterns, 2.0)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["confirmed"])
        self.assertEqual(out[0]["name"], "双底突破")
        out2 = self.cp.detect_breakouts(df, patterns, 1.0)
        self.assertFalse(out2[0]["confirmed"])
        self.assertIn("放量不足", out2[0]["note"])

    def test_breakouts_pending_and_invalid(self):
        df = _kline(5)
        df["close"] = [100.0] * 4 + [101.5]
        patterns = [{"name": "双底", "direction": "多", "neckline": 100.0, "pivots": [0]}]
        out = self.cp.detect_breakouts(df, patterns, 1.0)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["confirmed"])
        self.assertIn("待 3% 确认", out[0]["note"])
        # gap 过小 → 无输出
        df["close"] = [100.0] * 4 + [100.2]
        self.assertEqual(self.cp.detect_breakouts(df, patterns, 1.0), [])
        # 无效颈线 → 跳过
        patterns2 = [{"name": "x", "direction": "多", "neckline": 0.0, "pivots": []}]
        self.assertEqual(self.cp.detect_breakouts(df, patterns2, 1.0), [])

    def test_breakouts_bear(self):
        df = _kline(5)
        df["close"] = [100.0] * 4 + [90.0]
        patterns = [{"name": "双顶", "direction": "空", "neckline": 100.0, "pivots": [0]}]
        out = self.cp.detect_breakouts(df, patterns, 2.0)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["confirmed"])
        self.assertEqual(out[0]["name"], "双顶跌破")
        df["close"] = [100.0] * 4 + [98.0]
        out2 = self.cp.detect_breakouts(df, patterns, 1.0)
        self.assertFalse(out2[0]["confirmed"])


def _swing_up_df(n=30):
    """尾段 up swing: 低点 idx n-6(5.0) → 高点 idx n-3(11.0)，中间严格上升防伪摆动。"""
    rows = []
    for i in range(n):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0})
    lo_i, hi_i = n - 8, n - 5  # 22 / 25（pivot 扫描区间排除最后 k 个索引）
    highs = [9.5, 10.0, 10.5, 10.8, 10.9, 11.0, 10.0, 9.5, 9.0, 8.5]
    for k, h in enumerate(highs):
        rows[lo_i - 2 + k]["high"] = h
    rows[lo_i]["low"] = 5.0
    rows[lo_i]["close"] = 5.0
    rows[hi_i]["close"] = 11.0
    for j, (h, l) in ((n - 2, (9.5, 8.5)), (n - 1, (9.0, 8.0))):
        rows[j]["high"] = h
        rows[j]["low"] = l
        rows[j]["close"] = (h + l) / 2
    return pd.DataFrame(rows)


class TestFib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_major_swing_up(self):
        df = _swing_up_df()
        w = df.tail(self.cp.FIB_WINDOW)
        swing = self.cp._major_swing(w)
        self.assertEqual(swing["swing"], "up")
        self.assertEqual(swing["a"], 5.0)
        self.assertEqual(swing["b"], 11.0)

    def _swing_down_df(self, n=30):
        rows = []
        for i in range(n):
            rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0})
        hi_i, lo_i = n - 8, n - 5  # 22 / 25（pivot 扫描区间排除最后 k 个索引）
        highs = [9.5, 10.0, 12.0, 10.9, 10.8, 10.0, 9.5, 9.0, 8.5, 8.0]
        for k, h in enumerate(highs):
            rows[hi_i - 2 + k]["high"] = h
        rows[hi_i]["close"] = 12.0
        rows[lo_i]["low"] = 5.0
        rows[lo_i]["close"] = 5.0
        for j, (h, l) in ((n - 2, (9.5, 8.5)), (n - 1, (9.0, 8.0))):
            rows[j]["high"] = h
            rows[j]["low"] = l
            rows[j]["close"] = (h + l) / 2
        return pd.DataFrame(rows)

    def test_major_swing_down(self):
        df = self._swing_down_df(25)
        w = df.tail(self.cp.FIB_WINDOW)
        swing = self.cp._major_swing(w)
        self.assertEqual(swing["swing"], "down")
        self.assertEqual(swing["a"], 12.0)
        self.assertEqual(swing["b"], 5.0)

    def test_major_swing_fallback(self):
        n = 25
        rows = []
        for i in range(n):
            rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                         "open": float(i), "high": float(i), "low": float(i),
                         "close": float(i), "volume": 100.0})
        df = pd.DataFrame(rows)
        swing = self.cp._major_swing(df)
        # 单调序列无 fractal 摆动 → 兜底按高低点顺序
        self.assertIn(swing["swing"], ("up", "down"))

    def test_fib_short_sample(self):
        df = _kline(5)
        self.assertIsNone(self.cp.fib_analysis(df))

    def test_fib_flat_returns_none(self):
        rows = [{"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                 "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                 "volume": 100.0} for i in range(30)]
        self.assertIsNone(self.cp.fib_analysis(pd.DataFrame(rows)))

    def test_fib_up_positions(self):
        base = _swing_up_df(30)
        cases = [
            (15.0, "扩展1.618上方(强势突破)", 0.8),
            (13.0, "扩展1.27-1.618(强势区)", 0.6),
            (11.5, "突破前高(回撤上方)", 0.6),
            (8.8, "回撤0-0.382(强势回撤)", 0.3),
            (8.0, "回撤0.382-0.5", 0.1),
            (7.5, "回撤0.5-0.618(黄金区)", -0.2),
            (6.5, "回撤0.618-0.786(深回撤)", -0.6),
            (5.5, "跌破0.786(趋势转弱)", -1.0),
        ]
        for close, pos, bias in cases:
            with self.subTest(close=close):
                df = _swing_up_df(30).copy()
                df.loc[df.index[-1], "close"] = close
                r = self.cp.fib_analysis(df)
                self.assertIsNotNone(r)
                self.assertEqual(r["position"], pos)
                self.assertEqual(r["bias"], bias)
                self.assertIn("span_days", r)

    def test_fib_down_positions(self):
        base = self._swing_down_df(30)
        cases = [
            (-8.0, "扩展1.618下方(弱势破位)", -0.8),
            (-4.0, "扩展1.27-1.618(弱势区)", -0.6),
            (3.0, "跌破前低(回撤下方)", -0.6),
            (7.0, "回撤0-0.382(弱势反抽)", -0.3),
            (8.0, "回撤0.382-0.5", -0.1),
            (9.0, "回撤0.5-0.618(黄金区)", 0.2),
            (10.0, "回撤0.618-0.786(深回撤)", 0.6),
            (11.0, "突破0.786(趋势转强)", 1.0),
        ]
        for close, pos, bias in cases:
            with self.subTest(close=close):
                df = base.copy()
                df.loc[df.index[-1], "close"] = close
                r = self.cp.fib_analysis(df)
                self.assertIsNotNone(r)
                self.assertEqual(r["position"], pos)
                self.assertEqual(r["bias"], bias)


class TestVolumeProfileNote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def _patch_vp(self, df60, vp):
        from quant_system.analysis_core import volume_profile
        return [
            mock.patch.object(volume_profile, "load_kline_60d", return_value=(df60, "x")),
            mock.patch.object(volume_profile, "compute_volume_profile", return_value=vp),
        ]

    def test_no_data(self):
        for p in self._patch_vp(None, {}):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r = self.cp.volume_profile_note("600001", pd.Timestamp("2026-08-10"))
        self.assertFalse(r["ok"])
        self.assertIn("密集区不可用", r["note"])

    def test_vp_not_ok(self):
        df60 = _kline(30)
        for p in self._patch_vp(df60, {"ok": False, "note": "样本不足"}):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r = self.cp.volume_profile_note("600001", pd.Timestamp("2026-08-10"))
        self.assertFalse(r["ok"])
        self.assertIn("样本不足", r["note"])

    def test_above(self):
        df60 = _kline(30)
        vp = {"ok": True, "close": 11.0, "upper": 10.0, "lower": 9.0,
              "pos_label": "上方", "pos_pct": 0.9, "poc": 9.5}
        for p in self._patch_vp(df60, vp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r = self.cp.volume_profile_note("600001", pd.Timestamp("2026-08-10"))
        self.assertTrue(r["ok"])
        self.assertIn("高于上沿", r["pos_txt"])

    def test_below_and_pct(self):
        df60 = _kline(30)
        for p in self._patch_vp(df60, {"ok": True, "close": 8.0, "upper": 10.0,
                                       "lower": 9.0, "pos_label": "下方",
                                       "pos_pct": 0.5, "poc": 9.5}):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r = self.cp.volume_profile_note("600001", pd.Timestamp("2026-08-10"))
        self.assertIn("低于下沿", r["pos_txt"])
        for p in self._patch_vp(df60, {"ok": True, "close": 9.5, "upper": 10.0,
                                       "lower": 9.0, "pos_label": "区间内",
                                       "pos_pct": 50.0, "poc": 9.5}):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r2 = self.cp.volume_profile_note("600001", pd.Timestamp("2026-08-10"))
        self.assertIn("区间内 50%", r2["pos_txt"])

    def test_exception_degrade(self):
        from quant_system.analysis_core import volume_profile
        with mock.patch.object(volume_profile, "load_kline_60d",
                               side_effect=RuntimeError("boom")):
            r = self.cp.volume_profile_note("600001", pd.Timestamp("2026-08-10"))
        self.assertFalse(r["ok"])
        self.assertIn("密集区不可用", r["note"])


class TestComposeAndSymbol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_bull(self):
        row = {"patterns": [{"name": "双底", "direction": "多", "pivots": [10, 20, 30]}],
               "breakouts": [], "fib": {"bias": 0.3}, "vp": {"ok": True, "pos_label": "上方"}}
        r = self.cp.compose_view(row)
        self.assertEqual(r["view"], "多")
        self.assertGreaterEqual(r["confidence"], 0.15)

    def test_defense(self):
        row = {"patterns": [{"name": "双顶", "direction": "空", "pivots": [10, 20, 30]}],
               "breakouts": [{"name": "双顶跌破", "direction": "空", "confirmed": True}],
               "fib": {"bias": -0.6}, "vp": {}}
        r = self.cp.compose_view(row)
        self.assertEqual(r["view"], "防守")

    def test_bear_and_neutral(self):
        row = {"patterns": [{"name": "双顶", "direction": "空", "pivots": [10, 20, 30]}],
               "breakouts": [], "fib": {"bias": -0.3}, "vp": {}}
        r = self.cp.compose_view(row)
        self.assertEqual(r["view"], "空")
        neutral = {"patterns": [], "breakouts": [], "fib": None, "vp": {}}
        r2 = self.cp.compose_view(neutral)
        self.assertEqual(r2["view"], "震荡")
        self.assertEqual(r2["confidence"], round(0.35 - 0.05, 2))  # 无 fib -0.05

    def test_confidence_sample_note_cap(self):
        row = {"patterns": [{"name": "双底", "direction": "多", "pivots": [10, 20, 30]}],
               "breakouts": [{"name": "双底突破", "direction": "多", "confirmed": True}],
               "fib": {"bias": 0.8}, "vp": {}, "sample_note": "形态样本<60日"}
        r = self.cp.compose_view(row)
        self.assertLessEqual(r["confidence"], 0.40)

    def test_recency_weighting(self):
        row = {"patterns": [
            {"name": "双底", "direction": "多", "pivots": [0, 1, 2]},
            {"name": "双顶", "direction": "空", "pivots": [10, 11, 12]},
            {"name": "上升三角形", "direction": "多", "pivots": [20, 21, 22, 23]},
        ], "breakouts": [], "fib": None, "vp": {}}
        r = self.cp.compose_view(row)
        self.assertIsInstance(r["score"], float)

    def test_symbol_evidence(self):
        row = {"close": 10.5, "date": "2026-08-10", "samples": 65, "sample_note": "",
               "patterns": [{"name": "双底", "direction": "多", "neckline": 9.0, "note": "n"}],
               "breakouts": [{"name": "双底突破", "direction": "多", "gap_pct": 5.0,
                              "vol_ratio": 2.0, "confirmed": True}],
               "fib": {"swing": "up", "a": 5.0, "b": 11.0, "r382": 8.7, "r500": 8.0,
                       "r618": 7.3, "r786": 6.3, "x127": 12.6, "x1618": 14.7,
                       "position": "突破前高(回撤上方)"},
               "vp": {"ok": True, "poc": 9.5, "lower": 9.0, "upper": 10.0,
                      "pos_label": "上方", "pos_txt": "高于上沿 5.0%"}}
        ev = self.cp._symbol_evidence("600001", "测试", row)
        self.assertTrue(any("600001 测试" in e for e in ev))
        self.assertTrue(any("形态[双底·多]" in e for e in ev))
        self.assertTrue(any("信号[双底突破·多]" in e for e in ev))
        self.assertTrue(any("斐波那契(主要摆动 up" in e for e in ev))
        self.assertTrue(any("密集区 POC" in e for e in ev))
        # 降级分支
        row2 = {"close": 10.5, "date": "2026-08-10", "samples": 10, "sample_note": "形态样本<60日",
                "patterns": [], "breakouts": [], "fib": None,
                "vp": {"ok": False, "note": "密集区不可用(x)"}}
        ev2 = self.cp._symbol_evidence("600001", "测试", row2)
        self.assertTrue(any("⚠形态样本<60日" in e for e in ev2))
        self.assertTrue(any("密集区不可用" in e for e in ev2))

    def test_detect_symbol(self):
        df = _pattern_df()
        with mock.patch.object(self.cp, "volume_profile_note",
                               return_value={"ok": True, "poc": 9.5, "lower": 9.0,
                                             "upper": 10.0, "pos_label": "上方",
                                             "pos_pct": 0.5, "pos_txt": "高于上沿 5.0%"}):
            row = self.cp.detect_symbol(df, "600001", "测试")
        self.assertTrue(row["ok"])
        self.assertEqual(row["code"], "600001")
        self.assertIn("view", row)
        self.assertIn("confidence", row)
        self.assertTrue(row["evidence"])


class TestChartPatternSystem(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def _cps(self, tmp):
        return self.cp.ChartPatternSystem(out_dir=tmp / "generated")

    def test_name_map_cached(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        with mock.patch.object(self.cp, "load_names",
                               return_value=({"600001": "紫金矿业"}, {})) as m:
            self.assertEqual(cps._name_map()["600001"], "紫金矿业")
            cps._name_map()
        self.assertEqual(m.call_count, 1)

    def test_rag_hits(self):
        from quant_system.analysis_core import knowledge_rag
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        hits = [{"file": "a.md", "cat": "形态", "score": 0.9, "text": "头肩顶规则"}]
        with mock.patch.object(knowledge_rag, "search", return_value=hits) as m:
            rag = cps._rag()
        self.assertTrue(rag["available"])
        self.assertEqual(rag["basis"], "检索可用")
        self.assertEqual(rag["hits"][0]["file"], "a.md")
        # 缓存
        cps._rag()
        self.assertEqual(m.call_count, 1)

    def test_rag_no_hits(self):
        from quant_system.analysis_core import knowledge_rag
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        with mock.patch.object(knowledge_rag, "search", return_value=[]):
            rag = cps._rag()
        self.assertFalse(rag["available"])
        self.assertEqual(rag["basis"], "检索无命中")

    def test_rag_exception(self):
        from quant_system.analysis_core import knowledge_rag
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        with mock.patch.object(knowledge_rag, "search", side_effect=RuntimeError("boom")):
            rag = cps._rag()
        self.assertFalse(rag["available"])
        self.assertEqual(rag["basis"], "检索不可用")
        self.assertIn("boom", rag.get("error", ""))

    def _write_kline(self, kdir, code, dates):
        pd.DataFrame({
            "date": pd.to_datetime(dates),
            "open": [10.0] * len(dates), "high": [10.0] * len(dates),
            "low": [9.0] * len(dates), "close": [9.5] * len(dates),
            "volume": [100.0] * len(dates),
        }).to_parquet(kdir / f"{code}.parquet")

    def _ok_row(self, code, name="测试"):
        return {"code": code, "name": name, "ok": True, "date": "2026-08-10",
                "close": 10.0, "samples": 65, "sample_note": "",
                "patterns": [], "breakouts": [], "fib": None, "vp": {"ok": False},
                "view": "震荡", "confidence": 0.35, "score": 0.0,
                "evidence": [f"{code} 无形态"]}

    def test_detect_watch_all_skip_branches(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        self._write_kline(kdir, "600001", ["2026-08-07", "2026-08-10"])
        self._write_kline(kdir, "600003", ["2026-08-07", "2026-08-08"])  # 目标日无交易
        self._write_kline(kdir, "600004", [])  # 空
        self._write_kline(kdir, "600005", ["2026-08-10"])  # detect_symbol 抛异常
        cps = self._cps(tmp)

        def fake_read(path, cols, ws):
            if path.stem == "600004":
                return pd.DataFrame()
            return pd.read_parquet(path)

        def fake_detect_symbol(df, code, name):
            if code == "600005":
                raise RuntimeError("boom")
            return self._ok_row(code)

        ok_row = self._ok_row("600001")
        with mock.patch.object(self.cp, "KLINE_DIR", kdir), \
                mock.patch.object(self.cp, "read_kline_window", side_effect=fake_read), \
                mock.patch.object(self.cp, "load_names",
                                  return_value=({"600001": "A", "600002": "B",
                                                 "600003": "C", "600004": "D",
                                                 "600005": "E"}, {})), \
                mock.patch.object(self.cp, "detect_symbol", side_effect=fake_detect_symbol), \
                mock.patch.object(self.cp.ChartPatternSystem, "_rag",
                                  return_value={"basis": "检索可用", "hits": []}):
            res = cps.detect(date="2026-08-10", watch=["600001", "600002", "600003", "600004", "600005"])
        meta = res["meta"]
        self.assertEqual(meta["parsed"], 1)
        self.assertEqual(meta["skip_read_error"], 1)   # 600002 文件缺失
        self.assertEqual(meta["skip_no_data"], 1)      # 600003
        self.assertEqual(meta["skip_empty"], 1)        # 600004
        self.assertEqual(meta["skip_failed"], 1)       # 600005 由 mock 抛异常
        codes = {r["code"] for r in res["rows"] if not r.get("ok")}
        self.assertEqual(codes, {"600002", "600003", "600004", "600005"})

    def test_detect_fulla_limit(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        self._write_kline(kdir, "600001", ["2026-08-10"])
        cps = self._cps(tmp)
        with mock.patch.object(self.cp, "KLINE_DIR", kdir), \
                mock.patch.object(self.cp, "kline_files",
                                  return_value=[kdir / "600001.parquet"]), \
                mock.patch.object(self.cp, "read_kline_window",
                                  side_effect=lambda f, c, w: pd.read_parquet(f)), \
                mock.patch.object(self.cp, "load_names",
                                  return_value=({"600001": "A"}, {})), \
                mock.patch.object(self.cp, "detect_symbol",
                                  return_value=self._ok_row("600001")), \
                mock.patch.object(self.cp.ChartPatternSystem, "_rag",
                                  return_value={"basis": "检索可用", "hits": []}):
            res1 = cps.detect(date="2026-08-10", limit=1)
            res2 = cps.detect(date="2026-08-10")
        self.assertEqual(res1["meta"]["universe"], "fullA_sample")
        self.assertEqual(res2["meta"]["universe"], "fullA")
        self.assertEqual(res2["meta"]["parsed"], 1)

    def test_report_writes_md(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        res = {"date": "2026-08-10", "rows": [self._ok_row("600001")],
               "meta": {"universe": "watch", "parsed": 1},
               "rag": {"basis": "检索可用", "hits": []}}
        with mock.patch.object(self.cp.ChartPatternSystem, "detect", return_value=res):
            path = cps.report(date="2026-08-10", watch=["600001"])
        self.assertEqual(path.name, "chart_report_2026-08-10.md")
        text = path.read_text(encoding="utf-8")
        self.assertIn("## 明细", text)
        self.assertIn("600001", text)

    def test_view_single_code(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        res = {"date": "2026-08-10", "rows": [self._ok_row("600001")],
               "meta": {}, "rag": {"basis": "检索可用"}}
        with mock.patch.object(self.cp.ChartPatternSystem, "detect", return_value=res):
            v = cps.view(date="2026-08-10", code="600001")
        self.assertEqual(v["status"], "ok")
        self.assertEqual(v["object_code"], "600001")
        self.assertIn("RAG", v["evidence"][-1])

    def test_view_code_missing_degraded(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        res = {"date": "2026-08-10", "rows": [], "meta": {}, "rag": {"basis": "x"}}
        with mock.patch.object(self.cp.ChartPatternSystem, "detect", return_value=res):
            v = cps.view(date="2026-08-10", code="999999")
        self.assertEqual(v["status"], "degraded")
        self.assertEqual(v["signal"], "震荡")
        self.assertIn("无有效图表形态结果", v["evidence"][0])

    def test_view_majority_vote(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        rows = [
            dict(self._ok_row("600001"), view="多", confidence=0.8),
            dict(self._ok_row("600002"), view="多", confidence=0.7),
            dict(self._ok_row("600003"), view="震荡", confidence=0.5),
        ]
        res = {"date": "2026-08-10", "rows": rows,
               "meta": {"universe": "watch", "insufficient": 0},
               "rag": {"basis": "检索可用"}}
        with mock.patch.object(self.cp.ChartPatternSystem, "detect", return_value=res):
            v = cps.view(date="2026-08-10", watch=["600001"])
        self.assertEqual(v["signal"], "多")
        self.assertEqual(v["detail"]["bull"], 2)
        self.assertEqual(v["detail"]["neutral"], 1)

    def test_view_no_rows_degraded(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        res = {"date": "2026-08-10", "rows": [{"code": "600001", "ok": False,
                                               "error": "kline文件缺失"}],
               "meta": {}, "rag": {"basis": "x"}}
        with mock.patch.object(self.cp.ChartPatternSystem, "detect", return_value=res):
            v = cps.view(date="2026-08-10")
        self.assertEqual(v["status"], "degraded")
        self.assertIn("无可用图表形态样本", v["evidence"][0])

    def test_view_detect_exception(self):
        tmp = self._make_tmp()
        cps = self._cps(tmp)
        with mock.patch.object(self.cp.ChartPatternSystem, "detect",
                               side_effect=RuntimeError("x")):
            v = cps.view(date="2026-08-10")
        self.assertEqual(v["status"], "degraded")
        self.assertIn("检测异常", v["evidence"][0])

    def test_render_markdown_and_fmt_levels(self):
        res = {
            "date": "2026-08-10",
            "rows": [
                dict(self._ok_row("600001"), view="多", confidence=0.8, score=1.5,
                     patterns=[{"name": "双底", "direction": "多", "neckline": 9.0,
                                "note": "n", "pivots": [1]}],
                     breakouts=[{"name": "双底突破", "direction": "多", "confirmed": True}],
                     fib={"swing": "up", "a": 5.0, "b": 11.0, "r382": 8.7, "r500": 8.0,
                          "r618": 7.3, "r786": 6.3, "x127": 12.6, "x1618": 14.7,
                          "position": "突破前高(回撤上方)"},
                     evidence=["证据一"]),
                {"code": "600009", "name": "失败", "ok": False, "error": "检测失败"},
            ],
            "meta": {},
            "rag": {"basis": "检索可用", "query": "q", "hits": [{"score": 0.9, "cat": "c",
                                                                  "file": "f.md"}]},
        }
        md = self.cp.render_markdown(res)
        self.assertIn("# 图表形态", md)
        self.assertIn("## 标的全览", md)
        self.assertIn("## 明细", md)
        self.assertIn("600009", md)
        self.assertIn("跳过（检测失败）", md)
        self.assertEqual(self.cp._fmt_levels(res["rows"][0]["fib"]).count("回撤"), 1)
        self.assertEqual(self.cp._fmt_levels(None), "—")


class TestMain(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _import()

    def test_main_report_and_view(self):
        tmp = self._make_tmp()
        kdir = tmp / "kline"
        kdir.mkdir()
        pd.DataFrame({
            "date": pd.to_datetime(["2026-08-10"]),
            "open": [10.0], "high": [10.0], "low": [9.0], "close": [9.5],
            "volume": [100.0],
        }).to_parquet(kdir / "600001.parquet")
        ok_row = {"code": "600001", "name": "测试", "ok": True, "date": "2026-08-10",
                  "close": 9.5, "samples": 65, "sample_note": "",
                  "patterns": [], "breakouts": [], "fib": None,
                  "view": "震荡", "confidence": 0.35, "score": 0.0, "evidence": []}
        from quant_system.analysis_core import knowledge_rag
        argv = ["chart_pattern_system", "--watch", "600001,999999",
                "--date", "2026-08-10", "--report", "--view",
                "--out-dir", str(tmp / "generated")]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.cp, "KLINE_DIR", kdir), \
                mock.patch.object(self.cp, "read_kline_window",
                                  side_effect=lambda f, c, w: pd.read_parquet(f)), \
                mock.patch.object(self.cp, "load_names",
                                  return_value=({"600001": "测试"}, {})), \
                mock.patch.object(self.cp, "detect_symbol",
                                  return_value=ok_row), \
                mock.patch.object(knowledge_rag, "search", return_value=[]):
            self.cp.main()
        report = tmp / "generated" / "chart_report_2026-08-10.md"
        self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
