"""
test_fuzzing_chain — 融合链暴力/模糊测试（2026-08-22 稳定化专项）

目标: 任何畸形/缺失/边界输入都不允许炸穿决策链或 scorers:
  1. 随机变异 blocks（删key/None/类型错/超长串/负值/NaN/inf/嵌套错误）→ make_decision 不炸、输出结构完整
  2. 数值极值(±1e308/inf/-inf) → 各 scorer 分数有限且 available 为 bool
  3. 畸形预测库(空/缺列/时间乱序) → verify_pending 不炸
  4. 恶毒 blocks → engine_fused_signals 不炸

运行: python3 -m pytest tests/test_fuzzing_chain.py -q --no-header
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fusion_decision import make_decision  # noqa: E402
import fusion_decision as fd  # noqa: E402

random.seed(42)

_MUTATORS = [
    lambda: None,
    lambda: "垃圾字符串" * 50,
    lambda: [],
    lambda: {},
    lambda: 12345,
    lambda: -999,
    lambda: float("nan"),
    lambda: float("inf"),
    lambda: -float("inf"),
    lambda: {"嵌套": [1, {"更深": None}]},
    lambda: 1e308,
]


def _random_mutate(base: dict, depth: int = 2) -> dict:
    """对 blocks 各 key 随机: 删除 / 替换为毒值 / 保留 / 注入毒 key。"""
    out = {}
    for k, v in base.items():
        r = random.random()
        if r < 0.15:
            continue  # 删 key
        if r < 0.30:
            out[k] = random.choice(_MUTATORS)()
            continue
        if isinstance(v, dict) and depth > 0 and r < 0.55:
            out[k] = _random_mutate(v, depth - 1)
        elif isinstance(v, list) and r < 0.50:
            out[k] = [random.choice(_MUTATORS)() for _ in range(random.randint(1, 3))]
        else:
            out[k] = v
    # 注入几个陌生 key（模拟未来新增块）
    for _ in range(random.randint(1, 3)):
        out[f"weird_key_{random.randint(0, 999)}"] = random.choice(_MUTATORS)()
    return out


def _base() -> dict:
    from extreme_market_drill import _base_blocks  # noqa: PLC0415
    return _base_blocks()


def test_fuzz_make_decision_never_crashes():
    base = _base()
    for i in range(40):
        blocks = _random_mutate(base)
        try:
            dec = make_decision(blocks)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"round{i} 炸链: {type(e).__name__}: {str(e)[:120]}")
        assert isinstance(dec, dict) and dec.get("定调"), f"round{i} 缺定调"
        assert dec["定调"].get("posture"), f"round{i} 缺 posture"
        assert dec["定调"].get("position"), f"round{i} 缺 position"


@pytest.mark.parametrize("dim", ["technical", "emotion", "fund", "factor",
                                 "overseas", "mainline", "engine"])
def test_scorer_extreme_values_bounded(dim):
    """毒值/极值下每个 scorer 分数有限且返回 3 元组。"""
    fn = getattr(fd, f"_score_{dim}")
    for blocks in ({}, _random_mutate(_base()), _random_mutate(_base())):
        s, note, avail = fn(blocks)
        assert isinstance(avail, bool)
        assert math.isfinite(s), f"{dim} 分数非有限: {s} note={note}"


def test_scorer_poison_values_per_key():
    """逐个 key 毒化: 每维在单 key 毒化下不炸。"""
    base = _base()
    dims = {"technical": "technical", "emotion": "market_temperature",
            "fund": "market_temperature", "factor": "factor_signal",
            "overseas": "overseas", "mainline": "strong_direction",
            "engine": "engine_fusion"}
    for dim, key in dims.items():
        fn = getattr(fd, f"_score_{dim}")
        for poison in _MUTATORS:
            blocks = dict(base)
            blocks[key] = poison
            try:
                fn(blocks)
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"{dim} 毒化 {key}={type(poison).__name__} 炸: {e}")


def test_verify_malformed_predictions(tmp_path, monkeypatch):
    """畸形预测库: 空df/缺目标列/时间乱序 → verify_pending 降级不炸。"""
    import quant_system.analysis_core.predictions as predictions
    import prediction_verify as pv

    p = tmp_path / "preds.parquet"
    monkeypatch.setattr(predictions, "PREDICTIONS", p)
    # 指数可用但为空映射（不早退），让结构检查成为主要判定路径
    monkeypatch.setattr(pv, "_load_index_returns", lambda: ({}, {}))
    monkeypatch.setattr(pv, "_INDEX_PATH", tmp_path / "no_index.parquet")

    # 空库
    r = pv.verify_pending("2026-08-22", void_stale=False)
    assert r.get("note") == "预测库为空"

    # 缺关键列: 直接 monkeypatch _load 返回畸形 df（不经 _save 的 COLS 约束）
    malformed = pd.DataFrame({"pred_id": ["P1"], "status": ["pending"]})  # 缺 target_type/target
    monkeypatch.setattr(predictions, "_load", lambda: malformed)
    r3 = pv.verify_pending("2026-08-22", void_stale=False)
    assert isinstance(r3, dict) and "结构异常" in r3.get("note", "")


def test_engine_fusion_malicious_blocks(monkeypatch):
    import engine_fusion as ef

    def boom(engine, entry, kwargs=None, timeout=18):
        return {"ok": False, "note": f"{engine} 全挂"}
    monkeypatch.setattr(ef, "_call_engine", boom)

    for blocks in ({}, {"engines": {"x": {"status": "真产"}}},
                   {"engines": "string"}, {"engines": None},
                   _random_mutate(_base())):
        r = ef.engine_fused_signals(blocks, date="2026-08-21")
        assert r["ok_n"] < 3
        assert r["consensus"] in ("样本不足", "中性", "共振", "分歧")
        assert 0 <= r["score"] <= 100


def test_render_decision_md_with_poison_decision():
    """决策渲染对畸形 dec 也不炸（渲染层健壮性）。"""
    from fusion_decision import render_decision_md
    for dec in ({}, {"error": "x"}, {"定调": None}, {"定调": {"posture": None}},
                _random_mutate(_base())):
        md = render_decision_md(dec)
        assert isinstance(md, str)

def test_verify_date_rollback_and_chaos_index(tmp_path, monkeypatch):
    """日期回拨/乱序/损坏指数 → verify_pending 与 engine_factory 日期推导降级不炸。"""
    import quant_system.analysis_core.predictions as predictions
    import prediction_verify as pv

    p = tmp_path / "preds.parquet"
    monkeypatch.setattr(predictions, "PREDICTIONS", p)
    import pandas as _pd

    # 指数日期乱序 + 重复 + 回拨（2026-08-21 之后插回 2026-08-15）
    chaos = _pd.DataFrame({
        "date": _pd.to_datetime(["2026-08-19", "2026-08-21", "2026-08-15",
                                 "2026-08-20", "2026-08-21", "2026-08-14"]),
        "close": [100.0, 103.0, 99.0, 101.0, 103.5, 98.0],
    })
    ip = tmp_path / "index_chaos.parquet"
    chaos.to_parquet(ip, index=False)
    monkeypatch.setattr(pv, "_INDEX_PATH", ip)

    # 一条 target=08-20 的 pending（乱序索引应仍能正确找到 08-20 的下一交易日 08-21）
    pid = predictions.add_prediction(target_type="market_trend", target="2026-08-20",
                                     direction="up", source_module="chaos_test")
    r = pv.verify_pending("2026-08-22", void_stale=False)
    # 乱序索引排序后 08-20 的下一交易日是 08-21: close 103.5/101 - 1 = +2.47% → up 判对
    assert r["verified_n"] == 1 and r["correct_n"] == 1, r
    df = predictions._load()
    assert df.loc[df["pred_id"] == pid, "status"].iloc[0] == "correct"

    # 指数损坏(空文件) → 降级不炸
    bad = tmp_path / "index_bad.parquet"
    bad.write_bytes(b"not a parquet")
    monkeypatch.setattr(pv, "_INDEX_PATH", bad)
    r2 = pv.verify_pending("2026-08-22", void_stale=False)
    assert isinstance(r2, dict) and ("不可用" in r2.get("note", "") or "损坏" in r2.get("note", ""))

    # engine_factory 日期推导: 指数缺失 → 回退 (今天,今天) 不炸
    monkeypatch.setattr(pv, "_INDEX_PATH", tmp_path / "no_such.parquet")
    import engine_factory as efmod
    from pathlib import Path as _P
    monkeypatch.setattr(efmod, "ROOT", tmp_path)  # ROOT 无 index_daily → 回退
    d1, d2 = efmod._latest_trade_dates()
    assert isinstance(d1, str) and isinstance(d2, str)
