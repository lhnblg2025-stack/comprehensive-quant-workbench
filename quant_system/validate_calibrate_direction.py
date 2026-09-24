#!/usr/bin/env python3
"""
calibrate_direction.py — direction 字段校准（遗留任务①）
=========================================================
问题：ic_full 中 direction 与 IC 实测符号 89/121 不一致（以 IC 实测符号为准）。
做法：
1. 读 generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.csv（121 因子实测 IC）
2. 对每个因子：IC 实测符号 vs registry.direction（1=越大越看多 / -1=越小越看多）
   - direction=1 且 IC>0 → 一致
   - direction=1 且 IC<0 → 不一致（该因子实际是越小越好）
   - direction=-1 且 IC<0 → 一致
   - direction=-1 且 IC>0 → 不一致
3. 校准方向 = sign(ic_mean)（IC 为 0 时保持原 direction 不动）
4. 输出校准报告 + 写回 registry（通过 FactorMeta.direction 原地更新）
5. 校准后 IC 报告 direction 列同步更新
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # V11 审计修复(P0-A): quant_v6/validate 迁移时层级差一级, parents[2]=~主目录
sys.path.insert(0, str(ROOT))

IC_CSV = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"
REPORT_MD = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.md"
OUT_JSON = ROOT / "generated" / "ic_report" / "direction_calibration.json"
# 审计 2026-08-16：OOS 符号保持率门槛——OOS 翻转的因子 IC 符号不稳定，不自动翻转方向
OOS_REPORT = ROOT / "generated" / "ic_report" / "IC_OOS_REPORT.json"
MIN_ABS_IC_FOR_FLIP = 0.005  # IC 绝对值最小门槛，低于此视为噪声不翻转


def _load_oos_flipped() -> set[str]:
    """读取 OOS 报告中的 flipped_factors（OOS 方向翻转、不稳定因子）。"""
    import json
    try:
        if OOS_REPORT.exists():
            d = json.loads(OOS_REPORT.read_text(encoding="utf-8"))
            return set(d.get("flipped_factors", []) or [])
    except Exception:
        pass
    return set()


def main() -> int:
    from quant_system.ic_factors import registry as _reg
    _ensure = _reg.ensure_loaded

    _ensure()

    # 1. 读 IC 报告（UTF-8 BOM + CRLF 兼容）
    if not IC_CSV.exists():
        print(f"[失败] IC 报告不存在: {IC_CSV}")
        return 1
    with open(IC_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"IC 报告共 {len(rows)} 个因子")

    # 2. 逐个比对（基线=base_direction，即定义时方向，不叠加历史覆盖）
    # 2026-08-14 修复: 旧版以 meta.direction（已被覆盖污染）为基线 →
    # 覆盖表在多次校准间振荡（13↔48 条横跳），永不收敛。
    # 审计 2026-08-16：加入 OOS 门槛——OOS 翻转的因子不自动翻转方向（噪声保护）
    changed: list[dict] = []
    consistent = 0
    ic_zero = 0
    skipped_oos_flip = 0
    not_in_registry = []
    oos_flipped = _load_oos_flipped()
    for r in rows:
        name = r["factor"]
        try:
            ic = float(r["ic_mean"])
        except (ValueError, TypeError):
            ic = 0.0
        meta = _reg.get_factor(name)
        if meta is None:
            not_in_registry.append(name)
            continue
        base_dir = getattr(meta, "base_direction", None) or meta.direction
        if ic == 0:
            ic_zero += 1
            continue
        # IC 符号确定方向：IC>0 → direction=1；IC<0 → direction=-1
        new_dir = 1 if ic > 0 else -1
        if new_dir == base_dir:
            consistent += 1
            continue
        # 审计 2026-08-16：两条保护门槛
        # ① |IC| 过小（<阈值）视为噪声，不翻转
        # ② OOS 报告确认该因子方向翻转（不稳定）→ 不自动翻转，避免噪声固化
        if abs(ic) < MIN_ABS_IC_FOR_FLIP:
            ic_zero += 1
            continue
        if name in oos_flipped:
            skipped_oos_flip += 1
            continue
        changed.append({
            "factor": name, "category": r.get("category", ""),
            "ic_mean": ic, "old_direction": base_dir, "new_direction": new_dir,
        })
        meta.direction = new_dir

    # 3. 写校准报告
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total": len(rows),
        "consistent": consistent,
        "changed": len(changed),
        "ic_zero": ic_zero,
        "skipped_oos_flip": skipped_oos_flip,
        "not_in_registry": not_in_registry,
        "details": changed,
        "rule": "direction = sign(ic_mean)；|IC|<{} 保持原方向；OOS翻转因子不自动翻转；以 IC 实测符号为准".format(MIN_ABS_IC_FOR_FLIP),
        "calibrated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 4. 同步更新 IC CSV 的 direction 列（保持 UTF-8 无 BOM + LF）
    with open(IC_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))
    header = reader[0]
    di = header.index("direction")
    for row in reader[1:]:
        name = row[0]
        meta = _reg.get_factor(name)
        if meta is not None and len(row) > di:
            row[di] = str(meta.direction)
    with open(IC_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(reader)

    # 5. 汇总
    print(f"\n=== direction 校准结果 ===")
    print(f"总因子: {len(rows)} | 一致: {consistent} | 已校准: {len(changed)} | IC=0 跳过: {ic_zero}")
    if not_in_registry:
        print(f"不在注册表: {len(not_in_registry)} -> {not_in_registry[:10]}")
    if changed:
        print("\n校准明细（前 20）:")
        for c in changed[:20]:
            print(f"  {c['factor']:<28} IC={c['ic_mean']:+.4f}  "
                  f"direction {c['old_direction']} -> {c['new_direction']}")
        print(f"\n共 {len(changed)} 个因子方向已更新（详见 {OUT_JSON.name}）")
    print(f"报告: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
