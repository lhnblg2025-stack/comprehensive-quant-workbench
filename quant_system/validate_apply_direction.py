#!/usr/bin/env python3
"""
apply_direction_overrides.py — 方向校准持久化（遗留任务①的落盘层）
====================================================================
校准结果（direction_calibration.json）在进程内生效，重启即失。
本模块把校准结果转成持久化覆盖表 config/factor_direction_overrides.json，
并在 registry.ensure_loaded() 末尾应用，保证：
  - zoo 因子（Factor.direction）被覆盖
  - registry 因子（FactorMeta.direction）被覆盖
  - IC 合成/组合层（combination._apply_direction）同口径生效

用法:
  python3 quant_system/validate_apply_direction.py        # 生成覆盖表
  python3 scripts/refresh_factor_direction.py --full   # 全链刷新(周任务)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # quant_system/ → workspace
sys.path.insert(0, str(ROOT))

CALIB_JSON = ROOT / "generated" / "ic_report" / "direction_calibration.json"
OVERRIDES_JSON = ROOT / "config" / "factor_direction_overrides.json"


def generate_overrides() -> int:
    """从校准结果生成持久化覆盖表（JSON: {factor: direction}）。"""
    source = CALIB_JSON
    if CALIB_JSON.exists():
        with open(CALIB_JSON, encoding="utf-8") as f:
            calib = json.load(f)
        overrides = {c["factor"]: c["new_direction"] for c in calib.get("details", [])}
    elif OVERRIDES_JSON.exists():
        with open(OVERRIDES_JSON, encoding="utf-8") as f:
            existing = json.load(f)
        overrides = dict(existing.get("overrides") or {})
        source = OVERRIDES_JSON
    else:
        print(f"[失败] 校准结果不存在: {CALIB_JSON}（先跑 calibrate_direction.py）")
        return 1
    # 排除不在注册表的孤儿（如 boll_width_zoo 无对应 FactorMeta）
    # 注册表: quant_system.ic_factors.registry（2026-08-14 链修复）
    import importlib
    reg = importlib.import_module("quant_system.ic_factors.registry")
    reg.autodiscover()
    reg.import_from_zoo()
    orphan = [k for k in overrides if reg.get_factor(k) is None]
    for k in orphan:
        overrides.pop(k, None)

    payload = {
        "version": 1,
        "rule": "direction = sign(ic_mean)，IC=0 保持原方向；由 calibrate_direction.py 生成",
        "source": str(source),
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "overrides": dict(sorted(overrides.items())),
    }
    OVERRIDES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERRIDES_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"覆盖表已生成: {OVERRIDES_JSON}")
    print(f"  覆盖因子 {len(overrides)} 个（剔除孤儿 {len(orphan)}: {orphan}）")
    return 0


def apply_overrides(registry_module=None, zoo_module=None) -> int:
    """
    应用方向覆盖（幂等）。供 registry.ensure_loaded() 末尾调用。
    - registry_module: quant_system.ic_factors.registry（FactorMeta.direction）
    - zoo_module: quant_system.ic_factors.zoo（Factor.direction）
    返回应用数。
    """
    if not OVERRIDES_JSON.exists():
        return 0
    with open(OVERRIDES_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    overrides = payload.get("overrides", {})
    if not overrides:
        return 0

    n = 0
    if registry_module is not None:
        for name, d in overrides.items():
            meta = registry_module.get_factor(name)
            if meta is not None:
                meta.direction = d
                n += 1
    if zoo_module is not None:
        for name, d in overrides.items():
            fac = zoo_module.get_factor(name)
            if fac is not None:
                fac.direction = d
                n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(generate_overrides())
