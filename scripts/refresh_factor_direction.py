#!/usr/bin/env python3
"""
refresh_factor_direction.py — 因子方向校准一键刷新（2026-08-14）
================================================================
方向链（修复后完整链路）:
  1. ic_vectorized   → generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.csv
                       全市场向量化 IC 实测（121 因子）
  2. validate_calibrate_direction → direction_calibration.json
                       direction = sign(ic_mean)，写回 registry 内存态
  3. validate_apply_direction      → config/factor_direction_overrides.json
                       持久化覆盖表（zoo 40 因子 + registry 194 因子）
  4. 运行期: zoo.register_defaults / registry.ensure_loaded 自动应用覆盖

用法:
  python3 scripts/refresh_factor_direction.py --help
  python3 scripts/refresh_factor_direction.py --skip-ic      # 复用已有 CSV，只跑 2+3
  python3 scripts/refresh_factor_direction.py --full         # 全量 IC 重算（重，周日跑）

背景: 2026-08-13 归档 68 个废弃脚本时误删本链两个源文件（calibrate/ic_vectorized），
方向校准断链。本脚本重建端到端入口并登记为周任务。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IC_CSV = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"


def run_step(args: list[str], timeout_s: int) -> bool:
    t0 = time.time()
    print(f"\n[step] {' '.join(args)}", flush=True)
    cp = subprocess.run([sys.executable, *args], cwd=str(ROOT), timeout=timeout_s)
    print(f"[step] 耗时 {time.time()-t0:.0f}s, exit={cp.returncode}", flush=True)
    return cp.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="因子方向校准一键刷新")
    ap.add_argument("--skip-ic", action="store_true",
                    help="复用已有 IC CSV，只跑校准+落盘（默认）")
    ap.add_argument("--full", action="store_true",
                    help="全量重算 IC（--forward 5,10,20，重）")
    ap.add_argument("--n", type=int, default=0,
                    help="IC 重算股票数（0=全部；--full 时默认全部）")
    args = ap.parse_args()

    ic_args = [
        "scripts/ic_vectorized.py",
        "--source", "warehouse", "--n", str(args.n), "--forward", "5,10,20",
    ]
    if args.full:
        if not run_step(ic_args, timeout_s=6 * 3600):
            print("[失败] IC 重算失败，方向校准中止", file=sys.stderr)
            return 1
    elif not args.skip_ic and not IC_CSV.exists():
        print("[提示] IC CSV 不存在，自动执行 --full 逻辑", flush=True)
        if not run_step(ic_args, timeout_s=6 * 3600):
            return 1

    if not IC_CSV.exists():
        print("[失败] IC CSV 不存在，请先 --full", file=sys.stderr)
        return 1

    if not run_step(["quant_system/validate_calibrate_direction.py"], timeout_s=900):
        return 1
    if not run_step(["quant_system/validate_apply_direction.py"], timeout_s=900):
        return 1

    print("\n[完成] 方向校准链刷新成功: IC → 校准 → 覆盖表（运行期自动应用）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
