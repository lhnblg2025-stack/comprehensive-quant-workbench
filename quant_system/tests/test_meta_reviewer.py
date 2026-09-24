"""analysis_core/meta_reviewer 元审查器（多空矛盾焦点图）单元测试。

覆盖:
  - 各报告解析: battle_map(json/md)/emotion/risk/valuation/vpa/trend/factor/cycle/
    chart/behavior/macro/daily + 通用 md/json 兜底
  - 焦点图结构: 3 做多 + 3 做空（按置信度排序）、矛盾焦点、冲突强度排序
  - 数据时效性: 依赖数据陈旧 > 3 天 → 置信度 × 0.7；≤ 3 天不降权
  - 降级: 缺失报告跳过不阻塞 / 空目录输出空焦点图 + note / 无结论文件跳过
  - 防前视: 文件名日期 > 目标日期的报告不读
  - CLI 输出: json + md 双格式落盘 generated/meta_review/YYYYMMDD/

无网络: 全部 mock；REPORT_DIR 覆盖到 tmp。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.meta_reviewer as mr
    return mr


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="meta_review_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def _write(self, tmp: Path, name: str, content: str):
        p = tmp / name
        p.write_text(content, encoding="utf-8")
        return p


D = "2026-08-10"


def _valuation_md(as_of: str = "2026-08-10", n_long: int = 2, n_neutral: int = 1) -> str:
    """基本面估值报告: 多数看多 + 个别中性, 财务口径 as_of 显式标注。"""
    rows = ["| 代码 | 名称 | 估值分位 | 信号 | 置信 |",
            "|---|---|---|---|---|"]
    body = []
    for i in range(n_long):
        code = f"6005{10 + i}"
        rows.append(f"| {code} | 标的{i} | 低估 | 看多 | 0.81 |")
        body.append(f"### 标的{i}({code}) — 看多\n"
                    f"- 现价: 100（{D}） | 财务口径 as_of: {as_of}\n"
                    f"- **综合**: **看多**（置信 81%）— 低估+优质")
    for i in range(n_neutral):
        code = f"6019{i:02d}"
        rows.append(f"| {code} | 中性标的{i} | 未知 | 中性 | 0.30 |")
        body.append(f"### 中性标的{i}({code}) — 中性\n"
                    f"- 现价: 50（{D}） | 财务口径 as_of: {as_of}\n"
                    f"- **综合**: **中性**（置信 30%）— 数据不足")
    return (f"# 基本面估值系统 — {D}\n\n- 信号: {{'看多': {n_long}, '看空': 0, '中性': {n_neutral}}}\n\n"
            f"## 汇总\n\n" + "\n".join(rows) + "\n\n## 明细\n\n" + "\n\n".join(body) + "\n")


def _trend_md(level: str = "多", conf: str = "0.65") -> str:
    return (f"# 三屏趋势系统 — {D}\n\n"
            f"> 综合评级: **{level}**（置信 {conf}）\n\n- 其他内容\n")


def _factor_md(stance: str = "多", conf: str = "0.72") -> str:
    return (f"# 因子轮动报告 — {D}\n\n## 一、综合结论\n\n"
            f"- 信号: **{stance}**（看多） | 置信度 {conf}\n")


def _emotion_md(stance: str = "看空", conf: str = "82") -> str:
    return (f"# 情绪周期系统报告 — {D}\n\n"
            f"- 结论: **{stance}** | 置信 {conf}% | 六阶段: **下跌**（阶段置信 82%）\n")


def _risk_md(action: str = "减仓", conf: str = "0.57") -> str:
    return (f"# 风险纪律报告 {D}\n\n## 综合结论\n"
            f"- 风险等级: **中** | 操作建议: **{action}** | 置信度 {conf}\n")


def _vpa_md(stance: str = "看空", conf: float = 0.47) -> str:
    return (f"# 量价筹码系统 VPA — {D}\n\n### 贵州茅台(600519) — {stance}\n"
            f"- 结论: **{stance}**（置信 {int(conf * 100)}%）\n")


class TestReportParsers(_TmpMixin, unittest.TestCase):
    """各报告类型解析: 立场/置信度/as_of。"""

    def test_battle_map_json_parsed(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "battle_map_2026-08-10.json",
                    json.dumps({"date": D, "confidence": 0.61, "recommended": "防守"}))
        concl = mr.scan_reports(target=date(2026, 8, 10), report_dir=tmp)["conclusions"]
        self.assertEqual(len(concl), 1)
        self.assertEqual(concl[0]["stance"], "short")
        self.assertEqual(concl[0]["confidence"], 0.61)

    def test_emotion_report_parsed(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "emotion_report_2026-08-10.md", _emotion_md())
        concl = mr.scan_reports(target=date(2026, 8, 10), report_dir=tmp)["conclusions"][0]
        self.assertEqual(concl["stance"], "short")
        self.assertEqual(concl["confidence"], 0.82)

    def test_risk_report_parsed(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "risk_report_2026-08-10.md", _risk_md())
        concl = mr.scan_reports(target=date(2026, 8, 10), report_dir=tmp)["conclusions"][0]
        self.assertEqual(concl["stance"], "short")
        self.assertEqual(concl["confidence"], 0.57)

    def test_trend_report_parsed_neutral(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "trend_report_2026-08-10.md", _trend_md(level="震荡", conf="0.40"))
        concl = mr.scan_reports(target=date(2026, 8, 10), report_dir=tmp)["conclusions"][0]
        self.assertEqual(concl["stance"], "neutral")
        self.assertEqual(concl["confidence"], 0.40)

    def test_valuation_aggregation_and_as_of(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md",
                    _valuation_md(as_of="2026-08-08", n_long=2, n_neutral=1))
        concl = mr.scan_reports(target=date(2026, 8, 10), report_dir=tmp)["conclusions"][0]
        self.assertEqual(concl["report_type"], "valuation")
        self.assertEqual(concl["stance"], "long")
        self.assertEqual(concl["confidence"], 0.81)
        self.assertEqual(concl["as_of"], date(2026, 8, 8))

    def test_generic_md_fallback(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "fusion_2026-08-10.md", f"- 结论: **看多** | 置信 66%\n")
        concl = mr.scan_reports(target=date(2026, 8, 10), report_dir=tmp)["conclusions"][0]
        self.assertEqual(concl["report_type"], "generic")
        self.assertEqual(concl["stance"], "long")
        self.assertEqual(concl["confidence"], 0.66)


class TestFocusMap(_TmpMixin, unittest.TestCase):
    """焦点图组装: 3 多 3 空、排序、矛盾、降级、防前视。"""

    def _seed_6(self, tmp: Path):
        """3 做多 + 3 做空（2026-08-10）。"""
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "trend_report_2026-08-10.md", _trend_md())
        self._write(tmp, "factor_report_2026-08-10.md", _factor_md())
        self._write(tmp, "emotion_report_2026-08-10.md", _emotion_md())
        self._write(tmp, "risk_report_2026-08-10.md", _risk_md())
        self._write(tmp, "vpa_report_2026-08-10.md", _vpa_md())

    def test_focus_map_3long_3short_structure(self):
        mr = _import()
        tmp = self._make_tmp()
        self._seed_6(tmp)
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        self.assertEqual(len(f["long_logics"]), 3)
        self.assertEqual(len(f["short_logics"]), 3)
        for it in f["long_logics"] + f["short_logics"]:
            self.assertEqual(set(it), {"rank", "stance", "report_type", "report_type_label",
                                       "report", "confidence", "confidence_raw",
                                       "as_of", "stale", "stale_days", "summary"})
        self.assertEqual([it["stance"] for it in f["long_logics"]], ["看多"] * 3)
        self.assertEqual([it["stance"] for it in f["short_logics"]], ["看空"] * 3)
        self.assertEqual([it["rank"] for it in f["long_logics"]], [1, 2, 3])

    def test_logics_sorted_by_confidence_desc(self):
        mr = _import()
        tmp = self._make_tmp()
        self._seed_6(tmp)
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        lc = [it["confidence"] for it in f["long_logics"]]
        sc = [it["confidence"] for it in f["short_logics"]]
        self.assertEqual(lc, sorted(lc, reverse=True))
        self.assertEqual(sc, sorted(sc, reverse=True))
        self.assertEqual(f["long_logics"][0]["report"], "valuation_report_2026-08-10.md")
        self.assertEqual(f["short_logics"][0]["report"], "emotion_report_2026-08-10.md")

    def test_conflict_detection(self):
        mr = _import()
        tmp = self._make_tmp()
        self._seed_6(tmp)
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        pairs = {(c["long_report"], c["short_report"]) for c in f["conflicts"]}
        self.assertIn(("valuation_report_2026-08-10.md", "risk_report_2026-08-10.md"), pairs)
        self.assertEqual(len(f["conflicts"]), 3 * 3)
        vc = next(c for c in f["conflicts"]
                  if c["long_report"] == "valuation_report_2026-08-10.md"
                  and c["short_report"] == "risk_report_2026-08-10.md")
        self.assertEqual(vc["severity"], 0.57)
        self.assertIn("基本面估值看多", vc["description"])
        self.assertIn("风险纪律看空", vc["description"])

    def test_conflicts_sorted_by_severity(self):
        mr = _import()
        tmp = self._make_tmp()
        self._seed_6(tmp)
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        sev = [c["severity"] for c in f["conflicts"]]
        self.assertEqual(sev, sorted(sev, reverse=True))
        self.assertEqual(f["conflicts"][0]["severity"], 0.81)  # 0.81 多 vs 0.82 空

    def test_stale_confidence_penalty_x0_7(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md",
                    _valuation_md(as_of="2026-08-06", n_long=1, n_neutral=0))
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        it = f["long_logics"][0]
        self.assertTrue(it["stale"])
        self.assertEqual(it["stale_days"], 4)
        self.assertEqual(it["confidence_raw"], 0.81)
        self.assertAlmostEqual(it["confidence"], round(0.81 * 0.7, 3), places=3)

    def test_no_penalty_within_3_days(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md",
                    _valuation_md(as_of="2026-08-07", n_long=1, n_neutral=0))
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        it = f["long_logics"][0]
        self.assertFalse(it["stale"])
        self.assertEqual(it["stale_days"], 3)
        self.assertEqual(it["confidence"], 0.81)

    def test_missing_report_skipped_not_blocking(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "emotion_report_2026-08-10.md", _emotion_md())
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        self.assertEqual(len(f["long_logics"]), 1)
        self.assertEqual(len(f["short_logics"]), 1)
        self.assertIn("风险纪律", f["inputs"]["missing_reports"])
        self.assertIsNone(f["note"])

    def test_empty_dir_degrade(self):
        mr = _import()
        tmp = self._make_tmp()
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        self.assertEqual(f["long_logics"], [])
        self.assertEqual(f["short_logics"], [])
        self.assertEqual(f["conflicts"], [])
        self.assertEqual(f["inputs"]["scanned"], 0)
        self.assertIsNotNone(f["note"])
        self.assertIn("空焦点图", f["note"])

    def test_all_neutral_no_conflicts(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "trend_report_2026-08-10.md", _trend_md(level="震荡", conf="0.40"))
        self._write(tmp, "behavior_report_2026-08-10.md",
                    f"- 结论: **中性** | 置信 40% | 行为状态: **理性**\n")
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        self.assertEqual(f["long_logics"], [])
        self.assertEqual(f["short_logics"], [])
        self.assertEqual(f["conflicts"], [])
        self.assertIsNotNone(f["note"])

    def test_unparsed_files_skipped(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "rag_behavior_cache.json", json.dumps({"k": 1}))
        self._write(tmp, "rs_strength_2026-08-07.md", "# RS\n| 代码 | 评级 |\n| 1 | 强 |\n")
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        self.assertEqual(len(f["long_logics"]), 1)
        self.assertEqual(f["inputs"]["no_date_skipped"], 1)  # 缓存 json 无日期信息
        self.assertTrue(any("rs_strength" in u for u in f["inputs"]["unparsed"]))


class TestLookAheadAndCLI(_TmpMixin, unittest.TestCase):
    """防前视 + 最近交易日 + 落盘。"""

    def test_future_report_not_read(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "emotion_report_2026-08-11.md", _emotion_md())  # 未来
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        self.assertEqual(f["inputs"]["future_skipped"], 1)
        names = [it["report"] for it in f["short_logics"]]
        self.assertNotIn("emotion_report_2026-08-11.md", names)

    def test_latest_report_date(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "emotion_report_2026-08-08.md", _emotion_md())
        self._write(tmp, "risk_report_2026-08-09.md", _risk_md())
        self._write(tmp, "trend_report_2026-08-10.md", _trend_md())
        self.assertEqual(mr.latest_report_date(tmp), date(2026, 8, 10))

    def test_latest_report_date_empty_dir_returns_today(self):
        mr = _import()
        tmp = self._make_tmp()
        self.assertEqual(mr.latest_report_date(tmp), datetime.now().date())

    def test_output_written_json_and_md(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "emotion_report_2026-08-10.md", _emotion_md())
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp, write=True)
        out_dir = tmp / "meta_review" / "20260810"
        jp = out_dir / "meta_review_2026-08-10.json"
        mp = out_dir / "meta_review_2026-08-10.md"
        self.assertTrue(jp.is_file())
        self.assertTrue(mp.is_file())
        data = json.loads(jp.read_text(encoding="utf-8"))
        self.assertEqual(data["date"], "2026-08-10")
        self.assertEqual(data["long_logics"], f["long_logics"])
        md = mp.read_text(encoding="utf-8")
        self.assertIn("多空矛盾焦点", md)
        self.assertIn("做多逻辑", md)

    def test_render_md_contains_sections(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "emotion_report_2026-08-10.md", _emotion_md())
        f = mr.build_focus_map(target=date(2026, 8, 10), report_dir=tmp)
        md = mr.render_md(f)
        for sec in ("做多逻辑", "做空逻辑", "多空矛盾焦点", "数据时效性", "降级说明"):
            self.assertIn(sec, md)
        self.assertIn("valuation_report_2026-08-10.md", md)

    def test_cli_main_writes_output(self):
        mr = _import()
        tmp = self._make_tmp()
        self._write(tmp, "valuation_report_2026-08-10.md", _valuation_md())
        self._write(tmp, "emotion_report_2026-08-10.md", _emotion_md())
        old_dir, old_dirs = mr.REPORT_DIR, mr.REPORT_DIRS
        try:
            mr.REPORT_DIR = tmp
            mr.REPORT_DIRS = [tmp]
            rc = mr.main(["--date", "2026-08-10"])
        finally:
            mr.REPORT_DIR, mr.REPORT_DIRS = old_dir, old_dirs
        self.assertEqual(rc, 0)
        self.assertTrue((tmp / "meta_review" / "20260810" / "meta_review_2026-08-10.json").is_file())


if __name__ == "__main__":
    unittest.main()
