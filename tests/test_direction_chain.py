#!/usr/bin/env python3
"""test_direction_chain.py — 因子方向校准链测试（2026-08-14 修复后锁定）

覆盖:
  1. 校准脚本（validate_calibrate_direction）在 fixture CSV 上正确计算
     direction = sign(ic_mean) 并落盘 direction_calibration.json
  2. 覆盖表生成（validate_apply_direction.generate_overrides）剔除孤儿
  3. registry.ensure_loaded 在注册表已非空时也应用覆盖（旧版早退 bug 回归）
  4. zoo register_defaults 后覆盖生效
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALIB = ROOT / "quant_system" / "validate_calibrate_direction.py"
APPLY = ROOT / "quant_system" / "validate_apply_direction.py"


class TestDirectionChain(unittest.TestCase):

    def setUp(self):
        # 2026-08-14 review M3: 校准/落盘脚本会真实改写 tracked 文件
        # （IC CSV 的 direction 列、direction_calibration.json、overrides.json），
        # 测试前后快照恢复，避免污染 git 状态。
        self._snap = {}
        for rel in ("generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.csv",
                    "generated/ic_report/direction_calibration.json",
                    "config/factor_direction_overrides.json"):
            p = ROOT / rel
            self._snap[p] = p.read_bytes() if p.exists() else None

    def tearDown(self):
        for p, content in self._snap.items():
            if content is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(content)

    def test_calibrate_script_compiles_and_runs(self):
        """校准脚本在真实 IC CSV 上运行成功，且 not_in_registry 为空。"""
        csv_path = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"
        if not csv_path.exists():
            self.skipTest("IC CSV 不存在（先跑 scripts/ic_vectorized.py）")
        cp = subprocess.run([sys.executable, str(CALIB)], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=600)
        self.assertEqual(cp.returncode, 0, cp.stderr[-800:])
        out = ROOT / "generated" / "ic_report" / "direction_calibration.json"
        self.assertTrue(out.exists())
        d = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("details", d)
        # 规则: 每个校准项 new_direction == sign(ic_mean)
        for c in d["details"]:
            expected = 1 if c["ic_mean"] > 0 else -1
            self.assertEqual(c["new_direction"], expected, c["factor"])
        # M2 回归: 校准必须覆盖全部 CSV 因子（zoo 40 因子不得丢失）
        self.assertEqual(d["not_in_registry"], [], "zoo/registry 因子丢失")

    def test_calibrate_converges(self):
        """M2 回归: 连续两次校准结果必须一致（基线=base_direction，不振荡）。"""
        csv_path = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"
        if not csv_path.exists():
            self.skipTest("IC CSV 不存在")
        results = []
        for _ in range(2):
            cp = subprocess.run([sys.executable, str(CALIB)], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=600)
            self.assertEqual(cp.returncode, 0, cp.stderr[-800:])
            d = json.loads((ROOT / "generated" / "ic_report"
                            / "direction_calibration.json").read_text(encoding="utf-8"))
            results.append([c["factor"] for c in d["details"]])
        self.assertEqual(results[0], results[1], "校准不收敛（基线被覆盖污染）")

    def test_generate_overrides_no_orphan(self):
        """覆盖表生成: 所有 key 都必须在 registry 中存在（孤儿=0）。"""
        cp = subprocess.run([sys.executable, str(APPLY)], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=600)
        self.assertEqual(cp.returncode, 0, cp.stderr[-800:])
        ov = ROOT / "config" / "factor_direction_overrides.json"
        d = json.loads(ov.read_text(encoding="utf-8"))
        from quant_system.ic_factors import registry
        registry.ensure_loaded()
        orphans = [k for k in d["overrides"] if registry.get_factor(k) is None]
        self.assertEqual(orphans, [])

    def test_zoo_register_defaults_not_skipped(self):
        """M2 回归: zoo 内置因子必须完整注册（旧版 _REGISTRY 早退只剩 4 个）。"""
        from quant_system.ic_factors import zoo
        zoo._REGISTRY.clear()          # 模拟 autodiscover 已预注册少量因子的场景
        zoo._DEFAULTS_REGISTERED = False
        zoo.register_defaults()
        self.assertGreaterEqual(len(zoo._REGISTRY), 40,
                                "内置因子被早退跳过（预期 ≥40）")

    def test_ensure_loaded_applies_overrides_when_prepopulated(self):
        """回归: 注册表已非空时 ensure_loaded 也必须应用方向覆盖（旧版早退 bug）。"""
        from quant_system.ic_factors import registry
        registry.autodiscover()
        self.assertGreater(len(registry._REGISTRY), 0)
        registry._OVERRIDES_APPLIED = False
        registry.ensure_loaded()
        self.assertTrue(registry._OVERRIDES_APPLIED)

    def test_zoo_direction_override_applied(self):
        """zoo 因子注册后覆盖生效（direction 与覆盖表一致）。"""
        ov = json.loads((ROOT / "config" / "factor_direction_overrides.json")
                        .read_text(encoding="utf-8"))["overrides"]
        from quant_system.ic_factors import zoo
        zoo.register_defaults()
        for name, d in ov.items():
            f = zoo.get_factor(name)
            if f is not None:
                self.assertEqual(f.direction, d, f"zoo[{name}]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
