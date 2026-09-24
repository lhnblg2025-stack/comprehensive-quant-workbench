"""test_extreme_drill — 极端行情压力测试回归化（W1.2，2026-08-23）

把 `scripts/extreme_market_drill.py` 的 8 内置场景 + `scripts/scenario_library.py` 的
21 个知识库场景固化为 pytest 回归测试，与脚本共享同一套 `_check_all` 断言，避免
「脚本能过、测试没覆盖」的断言漂移。断言语义（安全约束）:
  - 极端场景不得 fail-open（禁止"积极进攻"、仓位上限受控）
  - 输出结构完整（必含"定调"/"决策依据"）
  - 高风险场景（expect_risk）必出 ≥1 条风险预案
  - 畸形/恶毒输入（None/字符串/NaN/嵌套错误）不炸链

运行: python3 -m pytest tests/test_extreme_drill.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extreme_market_drill import SCENES, _check_all  # noqa: E402
from scenario_library import EXTRA_SCENES  # noqa: E402

ALL_SCENES: list[dict] = list(SCENES) + list(EXTRA_SCENES)


def test_scene_coverage_floor() -> None:
    """回归护栏：内置 ≥8 场景、知识库扩展 ≥20 场景，防止场景被误删而不自知。"""
    assert len(SCENES) >= 8, f"内置场景少于 8（现 {len(SCENES)}）"
    assert len(EXTRA_SCENES) >= 20, f"知识库扩展场景少于 20（现 {len(EXTRA_SCENES)}）"
    assert len(ALL_SCENES) == len(SCENES) + len(EXTRA_SCENES)


def test_scene_ids_unique() -> None:
    """场景名唯一，避免 pytest ids 冲突与同名覆盖。"""
    names = [s["name"] for s in ALL_SCENES]
    assert len(names) == len(set(names)), f"存在重名场景: {sorted({n for n in names if names.count(n) > 1})}"


@pytest.mark.parametrize("sc", ALL_SCENES, ids=[s["name"] for s in ALL_SCENES])
def test_extreme_scene_safe(sc: dict) -> None:
    """每个极端场景都必须满足安全约束（复用脚本同款 _check_all，零漂移）。"""
    errs = _check_all(
        sc["blocks"],
        sc.get("exp_posture"),
        sc.get("exp_pos_max"),
        sc.get("exp_posture_banned") or (),
        sc.get("expect_risk", False),
    )
    assert not errs, f"场景[{sc['name']}]违反安全约束: {'; '.join(errs[:5])}"
