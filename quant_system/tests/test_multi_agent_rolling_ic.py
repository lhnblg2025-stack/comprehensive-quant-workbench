"""multi_agent 20 日滚动 IC 动态专家权重测试。

覆盖:
  - _view_history 日期过滤 / days 截断 / degraded 跳过
  - _rolling_ic_weights 窗口不足回退 None / 充足窗口正负 IC 排序 / 归一化
  - 0.7*IC + 0.3*static 混合系数
  - degraded 与历史缺失专家不抛异常
  - ret_next 缺失日剔除后仍可计算
  - arbitrate 集成滚动 IC 权重与 weights_note
  - 原 _effective_weights 单返回值接口兼容
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.multi_agent as ma
    return ma


def _dates(n: int, start: date = date(2026, 1, 1)) -> list[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _make_views(ma, degraded: set[str] | None = None, asof: str = "2026-01-20") -> list[dict]:
    views = []
    for agent in ma.BASE_WEIGHTS:
        views.append({"agent": agent, "view": "震荡", "confidence": 0.3,
                      "evidence": [f"{agent}-ev"], "weight": 0.0,
                      "status": "ok", "detail": {"date": asof}})
    for i, v in enumerate(views):
        if v["agent"] in (degraded or set()):
            v["status"] = "degraded"
            v["confidence"] = 0.0
            v["detail"] = {"error": "boom"}
    return views


def _vote(agent: str, view: str, status: str = "ok") -> dict:
    return {"agent": agent, "view": view, "confidence": 0.7,
            "value": 0, "weight": 0.1, "status": status}


def _write_parquet(out_dir: Path, date_rets: list[tuple[str, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": pd.to_datetime([d for d, _ in date_rets]),
                       "ret_next": [r for _, r in date_rets]})
    df.to_parquet(out_dir / "temperature_history.parquet", index=False)


def _write_multi_agent_files(out_dir: Path, date_votes: list[tuple[str, list[dict]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for day, votes in date_votes:
        (out_dir / f"multi_agent_{day}.json").write_text(
            json.dumps({"date": day, "votes": votes}, ensure_ascii=False), encoding="utf-8")


def _paired_history(dates: list[str], agent: str, views: list[str]) -> list[dict]:
    return [{"date": d, "agent": agent, "view": v, "status": "ok"}
            for d, v in zip(dates, views)]


class _TmpOutMixin:
    def _tmp_out(self) -> Path:
        out = Path(tempfile.mkdtemp(prefix="multi_agent_ic_test_"))
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        return out


class TestRollingIcWeights(_TmpOutMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_window_insufficient_returns_none(self):
        dates = _dates(10)
        out = self._tmp_out()
        _write_multi_agent_files(out, [(d, [_vote("情绪面", "多")]) for d in dates])
        _write_parquet(out, [(d, i / 1000.0) for i, d in enumerate(dates)])
        with mock.patch.object(self.ma, "OUT_DIR", out):
            history = self.ma._view_history(dates[-1], days=20)
            self.assertEqual(len(history), 10)
            self.assertIsNone(self.ma._rolling_ic_weights(history, dates[-1]))

    def test_target_day_opinion_is_excluded(self):
        dates = _dates(20)
        history = _paired_history(dates, "情绪面", ["多"] * len(dates))
        out = self._tmp_out()
        _write_parquet(out, [(d, i / 1000.0) for i, d in enumerate(dates)])
        with mock.patch.object(self.ma, "OUT_DIR", out):
            weights = self.ma._rolling_ic_weights(history, dates[-1])
        # 排除 target 当日观点后仅剩 19 个配对，达不到 20 日窗口。
        self.assertIsNone(weights)

    def test_window_sufficient_positive_beats_negative(self):
        dates = _dates(21)
        pos = ["空"] * 7 + ["震荡"] * 7 + ["多"] * 7
        neg = ["多"] * 7 + ["震荡"] * 7 + ["空"] * 7
        history = _paired_history(dates, "情绪面", pos) + _paired_history(dates, "资金面", neg)
        out = self._tmp_out()
        _write_parquet(out, [(d, i / 1000.0) for i, d in enumerate(dates)])
        with mock.patch.object(self.ma, "OUT_DIR", out):
            weights = self.ma._rolling_ic_weights(history, dates[-1])
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertGreater(weights["情绪面"], weights["资金面"])

    def test_blend_coefficient_07_ic_03_static(self):
        views = _make_views(self.ma)
        rolling = {a: 0.0 for a in self.ma.BASE_WEIGHTS}
        rolling["情绪面"] = 1.0
        with mock.patch.object(self.ma, "_view_history", return_value=[]), \
                mock.patch.object(self.ma, "_rolling_ic_weights", return_value=rolling):
            weights, _ = self.ma._effective_weights_ic(views, "2026-01-20")

        static = self.ma.BASE_WEIGHTS
        raw_final = {a: self.ma.ROLLING_IC_IC_BLEND * rolling[a] +
                     self.ma.ROLLING_IC_STATIC_BLEND * static[a] for a in static}
        total_final = sum(raw_final.values())
        expected = {a: raw_final[a] / total_final for a in static}
        for agent in static:
            self.assertAlmostEqual(weights[agent], expected[agent], places=10)
        self.assertGreater(weights["情绪面"], static["情绪面"])

    def test_missing_agent_keeps_static_share_no_raise(self):
        dates = _dates(21)
        views = ["空"] * 7 + ["震荡"] * 7 + ["多"] * 7
        history = _paired_history(dates, "情绪面", views)
        out = self._tmp_out()
        _write_parquet(out, [(d, i / 1000.0) for i, d in enumerate(dates)])
        with mock.patch.object(self.ma, "OUT_DIR", out):
            weights = self.ma._rolling_ic_weights(history, dates[-1])
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertGreater(weights["资金面"], 0.0)

    def test_ret_next_missing_days_are_dropped(self):
        dates = _dates(25)
        views = (["空"] * 7 + ["震荡"] * 7 + ["多"] * 6) * 2
        history = _paired_history(dates, "情绪面", views[:len(dates)])
        out = self._tmp_out()
        rets = [i / 1000.0 if i < 20 else float("nan") for i in range(len(dates))]
        _write_parquet(out, list(zip(dates, rets)))
        with mock.patch.object(self.ma, "OUT_DIR", out):
            weights = self.ma._rolling_ic_weights(history, dates[-1])
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)

    def test_view_history_filters_future_days_and_degraded(self):
        dates = _dates(6)
        out = self._tmp_out()
        _write_multi_agent_files(out, [
            (dates[0], [_vote("情绪面", "多")]),
            (dates[1], [_vote("情绪面", "多")]),
            (dates[2], [_vote("情绪面", "多")]),
            (dates[3], [_vote("情绪面", "多", status="degraded")]),
            (dates[4], [_vote("情绪面", "空")]),
            (dates[5], [_vote("情绪面", "空")]),  # 目标日之外
        ])
        with mock.patch.object(self.ma, "OUT_DIR", out):
            history = self.ma._view_history(dates[4], days=3)
        self.assertEqual({h["date"] for h in history}, {dates[2], dates[4]})
        self.assertTrue(all(h["status"] == "ok" for h in history))


class TestVoteHitRatesAndDirectionBlend(_TmpOutMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_hit_rate_breaks_ic_tie(self):
        dates = _dates(25)
        history = (_paired_history(dates, "情绪面", ["多"] * len(dates)) +
                   _paired_history(dates, "资金面", ["空"] * len(dates)))
        out = self._tmp_out()
        _write_parquet(out, [(d, 0.02) for d in dates])
        sig = lambda stat: mock.Mock(statistic=stat)  # noqa: E731
        with mock.patch.object(self.ma, "OUT_DIR", out), \
                mock.patch.object(self.ma, "spearmanr",
                                  side_effect=[sig(0.4), sig(0.4)]):
            weights = self.ma._rolling_ic_weights(history, dates[-1])

        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertGreater(weights["情绪面"], weights["资金面"])
        rates = self.ma._vote_hit_rates(history, {d: 0.02 for d in dates})
        self.assertEqual(rates["情绪面"], 1.0)
        self.assertEqual(rates["资金面"], 0.0)

    def test_ic_hit_conflict_blends_nearly_equal(self):
        dates = _dates(25)
        history = (_paired_history(dates, "情绪面", ["空"] * len(dates)) +
                   _paired_history(dates, "资金面", ["多"] * len(dates)))
        out = self._tmp_out()
        _write_parquet(out, [(d, 0.02) for d in dates])
        sig = lambda stat: mock.Mock(statistic=stat)  # noqa: E731
        with mock.patch.object(self.ma, "OUT_DIR", out), \
                mock.patch.object(self.ma, "spearmanr",
                                  side_effect=[sig(0.8), sig(-0.4)]):
            weights = self.ma._rolling_ic_weights(history, dates[-1])

        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertAlmostEqual(weights["情绪面"], weights["资金面"], places=10)
        self.assertLess(abs(weights["情绪面"] - weights["资金面"]), 0.1)

    def test_missing_hit_rate_uses_neutral_rank(self):
        dates = _dates(25)
        history = (_paired_history(dates, "情绪面", ["多"] * len(dates)) +
                   _paired_history(dates[:5], "规律面", ["多"] * 5))
        out = self._tmp_out()
        _write_parquet(out, [(d, 0.02) for d in dates])
        sig = mock.Mock(statistic=0.3)
        with mock.patch.object(self.ma, "OUT_DIR", out), \
                mock.patch.object(self.ma, "spearmanr", return_value=sig):
            weights = self.ma._rolling_ic_weights(history, dates[-1])

        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertGreater(weights["规律面"], 0.0)
        rates = self.ma._vote_hit_rates(history, {d: 0.02 for d in dates})
        self.assertIsNone(rates["规律面"])

    def test_defense_view_counts_as_short_hit(self):
        dates = _dates(20)
        history = _paired_history(dates, "风险面", ["防守"] * len(dates))
        rates = self.ma._vote_hit_rates(history, {d: -0.02 for d in dates})
        self.assertEqual(rates["风险面"], 1.0)

    def test_flat_ret_days_are_excluded_from_hit_rate(self):
        dates = _dates(20)
        history = _paired_history(dates, "情绪面", ["多"] * len(dates))
        rets = [0.02 if i < 8 else 0.0 for i in range(len(dates))]
        rates = self.ma._vote_hit_rates(history, dict(zip(dates, rets)))
        self.assertEqual(rates["情绪面"], 1.0)


class TestEffectiveWeightsIc(_TmpOutMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_degraded_view_redistributes_rolling_ic_weights(self):
        views = _make_views(self.ma, degraded={"情绪面"})
        with mock.patch.object(self.ma, "_view_history", return_value=[]), \
                mock.patch.object(self.ma, "_rolling_ic_weights",
                                  return_value=dict(self.ma.BASE_WEIGHTS)):
            weights, note = self.ma._effective_weights_ic(views, "2026-01-20")
        self.assertEqual(weights["情绪面"], 0.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertIn("降级重分配[情绪面 失效]", note)

    def test_original_effective_weights_single_return_compat(self):
        views = _make_views(self.ma)
        weights = self.ma._effective_weights(views)
        self.assertIsInstance(weights, dict)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)


class TestArbitrateRollingIcIntegration(_TmpOutMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_arbitrate_falls_back_to_static_on_short_window(self):
        dates = _dates(5)
        out = self._tmp_out()
        _write_multi_agent_files(out, [(d, [_vote("情绪面", "多")]) for d in dates])
        _write_parquet(out, [(d, i / 1000.0) for i, d in enumerate(dates)])
        views = _make_views(self.ma, asof=dates[-1])
        with mock.patch.object(self.ma, "OUT_DIR", out), \
                mock.patch.object(self.ma, "agent_views", return_value=views):
            res = self.ma.arbitrate(dates[-1], devil_advocate=False)
        self.assertIn("回退静态", res["weights_note"])
        self.assertAlmostEqual(sum(v["weight"] for v in res["votes"]), 1.0, places=10)

    def test_arbitrate_uses_rolling_ic_weights(self):
        dates = _dates(21)
        pos = ["空"] * 7 + ["震荡"] * 7 + ["多"] * 7
        neg = ["多"] * 7 + ["震荡"] * 7 + ["空"] * 7
        out = self._tmp_out()
        date_votes = []
        for i, d in enumerate(dates):
            date_votes.append((d, [_vote("情绪面", pos[i]), _vote("资金面", neg[i])]))
        _write_multi_agent_files(out, date_votes)
        _write_parquet(out, [(d, i / 1000.0) for i, d in enumerate(dates)])
        views = _make_views(self.ma, asof=dates[-1])
        with mock.patch.object(self.ma, "OUT_DIR", out), \
                mock.patch.object(self.ma, "agent_views", return_value=views):
            res = self.ma.arbitrate(dates[-1], devil_advocate=False)

        vote_weights = {v["agent"]: v["weight"] for v in res["votes"]}
        self.assertIn("滚动IC权重", res["weights_note"])
        self.assertGreater(vote_weights["情绪面"], self.ma.BASE_WEIGHTS["情绪面"])
        self.assertLess(vote_weights["资金面"], self.ma.BASE_WEIGHTS["资金面"])
        self.assertAlmostEqual(sum(vote_weights.values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
