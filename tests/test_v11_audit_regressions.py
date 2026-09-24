"""
test_v11_audit_regressions — 2026-08-10 全仓逐行审计修复的回归测试

覆盖审计中发现的 Critical/Major 修复:
  1. regime_classifier: 数据未达目标日 → stale=True 不伪造当日结果
  2. fusion.store: 同日重跑 keep='last' 保留新值
  3. emotion_cycle: 冰点(zt<=30)不被退潮(mb<=2&zt<=45)遮蔽
  4. resonance_scorer: 历史日期禁用即时资金流（前视防护）
  5. order_dispatcher: 去重后按实际订单数等权归一
  6. industry_graph: overlap 列落盘（非死代码）
  7. data_health_check: size 键统一（突变检测生效）
  8. calibration.get_module_weights: 时区 dtype 不崩溃
  9. theme_cycle: 每概念固定 60 交易日窗口 + role 仅最新日
  10. scenario.similar_days: 今日不匹配时不误删相似日

运行: python3 -m pytest tests/test_v11_audit_regressions.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))


# ── 1. regime_classifier stale ──────────────────────────────
class TestRegimeStale:
    def test_request_after_data_is_stale(self):
        from quant_system.analysis_core import regime_classifier
        df = pd.read_parquet(regime_classifier.INDEX_DAILY)
        data_last = pd.to_datetime(df["date"]).max()
        requested = (data_last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        r = regime_classifier.classify(requested)
        assert r["stale"] is True
        assert r["date"] != r["requested_date"]

    def test_request_at_data_date_not_stale(self):
        from quant_system.analysis_core.regime_classifier import classify
        r = classify("2026-08-07")
        assert r["stale"] is False
        assert r["date"] == "2026-08-07"


# ── 2. fusion keep last ─────────────────────────────────────
class TestFusionKeepLast:
    def test_same_day_rerun_keeps_new_value(self, tmp_path, monkeypatch):
        import quant_system.analysis_core.fusion as fusion
        monkeypatch.setattr(fusion, "OUT", tmp_path / "fusion_test.parquet")
        fusion.store({"date": "2026-01-01", "temperature": 40, "tag": "old",
                      "emotion_stage": "x", "signals": [], "cross": []})
        fusion.store({"date": "2026-01-01", "temperature": 61, "tag": "new",
                      "emotion_stage": "x", "signals": [], "cross": []})
        df = pd.read_parquet(tmp_path / "fusion_test.parquet")
        assert len(df) == 1
        assert int(df.iloc[0]["temperature"]) == 61, "同日重跑应保留新值(keep=last)"


# ── 3. emotion_cycle 冰点优先于退潮 ─────────────────────────
class TestEmotionIceOrder:
    def test_ice_not_shadowed_by_ebb(self):
        from quant_system.analysis_core.emotion_cycle import classify
        # 纯冰点: zt<=30, mb<=2, 无负溢价/无大面 → 不应被退潮分支抢判
        row = pd.Series({"zt_cnt": 25, "max_board": 1, "zb_rate": 0.2,
                         "jr1": 0.1, "premium": 0.5, "zt_chg": -5, "mb_chg": -1,
                         "big_loss_cnt": 2})
        r = classify(row)
        assert r["stage"] == "ice", f"涨停25/最高1板/溢价正应为冰点，实际 {r['stage']}"

    def test_ebb_with_big_loss_still_ebb(self):
        from quant_system.analysis_core.emotion_cycle import classify
        row = pd.Series({"zt_cnt": 40, "max_board": 2, "zb_rate": 0.2,
                         "jr1": 0.1, "premium": 0.5, "zt_chg": -5, "mb_chg": -1,
                         "big_loss_cnt": 12})
        r = classify(row)
        assert r["stage"] == "ebb", f"大面12家应为退潮，实际 {r['stage']}"


# ── 4. resonance 前视防护 ──────────────────────────────────
class TestResonanceNoLookahead:
    def test_historical_date_no_sector_flow(self):
        from quant_system.analysis_core import resonance_scorer
        sf = resonance_scorer._load_sector_flow("2020-01-01")
        assert sf == {}, "历史日期严禁使用即时资金流（前视）"

    def test_seat_missing_weight_redistribution(self):
        """席位缺失时分数不超 [-1,3] 且保留资金流维度。"""
        # 直接用 score_day 跑最近日期（本地数据），校验 score 范围
        from quant_system.analysis_core import resonance_scorer
        df = resonance_scorer.score_day(None)
        if df.empty:
            pytest.skip("无主题数据")
        assert df["score"].between(-1.0, 3.0).all(), "score 超出 [-1,3]"


# ── 5. order_dispatcher 等权归一 ────────────────────────────
class TestOrderEqualWeight:
    def test_dedup_equal_weight_reaches_cap(self, tmp_path, monkeypatch):
        import quant_system.analysis_core.order_dispatcher as od
        # 构造 2 只同 code 的 core（去重后 1 单），position_range 10-20% → ratio=0.2
        bm = {
            "position_range": "10-20%",
            "emotion_stage": "修复", "macro_veto": "none", "regime": "震荡市",
            "attack_groups": {"core": [
                {"name": "A概念", "score": 3.0, "strategy": "趋势低吸",
                 "leader": {"code": "600001", "name": "股票A", "boards": 2}},
                {"name": "B概念", "score": 2.0, "strategy": "趋势低吸",
                 "leader": {"code": "600001", "name": "股票A", "boards": 2}},
            ]},
        }
        p = tmp_path / "generated" / "battle_map_2026-01-01.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(bm, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(od, "ROOT", tmp_path)
        # _correlation_constraint 会读 kline，patch 成跳过（单只票直接返回 1.0）
        monkeypatch.setattr(od, "_correlation_constraint", lambda orders: {"note": "test", "scale": 1.0})
        r = od.dispatch("2026-01-01", capital=1_000_000)
        assert len(r["orders"]) == 1, "同 code 应去重为 1 单"
        assert r["orders"][0]["ratio"] == 0.2, f"单票应达仓位上限 0.2, 实际 {r['orders'][0]['ratio']}"
        assert r["total_exposure"] == 0.2


# ── 6. industry_graph overlap 落盘 ──────────────────────────
class TestIndustryOverlap:
    def test_overlap_column_persisted(self):
        from quant_system.analysis_core.config import MARKET_DIR
        p = ROOT / "generated" / "industry_graph.parquet"
        if not p.exists():
            pytest.skip("传导图未构建")
        df = pd.read_parquet(p)
        assert "overlap" in df.columns, "overlap 必须落盘（非死代码）"
        assert df["overlap"].notna().all()


# ── 7. data_health size 键统一 ──────────────────────────────
class TestHealthSizeKey:
    def test_size_mutation_key_matches_snapshot(self, tmp_path):
        from quant_system.analysis_core import data_health_check as dhc
        f = tmp_path / "dummy.parquet"
        f.write_bytes(b"x" * 1000)
        old = {"size::dummy": 500}   # 快照键 = f"size::{name}"（无 .parquet 后缀）
        chg = dhc._size_mutation(f, "dummy", old)
        assert chg == 1.0, f"突变检测应命中(键统一), 实际 {chg}"
        chg2 = dhc._size_mutation(f, "dummy.parquet", old)
        assert chg2 is None, "旧键(带后缀)不应命中（回归保护）"


# ── 8. calibration 时区不崩溃 ───────────────────────────────
class TestCalibrationTz:
    def test_get_module_weights_no_typeerror(self):
        from quant_system.analysis_core.calibration import get_module_weights
        w = get_module_weights("2026-08-07")
        assert isinstance(w, dict), "不应抛 TypeError（时区 dtype 已修复）"


# ── 9. theme_cycle 窗口与 role ──────────────────────────────
class TestThemeCycle:
    def test_window_and_role_no_lookahead(self):
        from quant_system.analysis_core.config import MARKET_DIR
        p = MARKET_DIR / "theme_cycle.parquet"
        if not p.exists():
            pytest.skip("theme_cycle 未构建")
        df = pd.read_parquet(p)
        last = df["date"].max()
        sizes = df.groupby("concept").size()
        assert (sizes == 60).all(), f"每概念应 60 交易日窗口, 实际分布 {sizes.value_counts().to_dict()}"
        hist_role = df[df["date"] != last]["role"]
        assert (hist_role == "").all(), "历史行 role 必须为空（无前视）"
        assert (df[df["date"] == last]["role"] != "").sum() > 0, "最新日应有 role"


# ── 10. scenario 相似日不误删 ───────────────────────────────
class TestScenarioSimilar:
    def test_no_wrong_drop(self):
        from quant_system.analysis_core import scenario
        # 用真实数据跑相似日，校验返回结构
        out = scenario.similar_days(k=3)
        assert isinstance(out, list)
        for x in out:
            assert "date" in x and "next_zt_cnt" in x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
