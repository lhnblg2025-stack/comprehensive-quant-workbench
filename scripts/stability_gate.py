#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定化门禁（2026-08-22 长期运维建议②落地）

"新引擎/采集器必须过 fuzz 毒化矩阵" —— 提交/上线前一键跑关键稳定化测试集。
失败即拒绝合入（退出码 1），防"能跑但一遇畸形输入就炸"的回归。

覆盖（全部 pytest，确定性）:
  1. tests/test_fuzzing_chain.py        —— 融合链暴力/模糊（40轮变异+逐key毒化+极值+日期回拨）
  2. tests/test_prediction_verify.py    —— 预测验证闭环
  3. tests/test_engine_fusion.py        —— 引擎→融合适配器(含降级)
  4. quant_system/tests/test_daily_review_chain.py —— 主链回归

用法:
  python3 scripts/stability_gate.py            # 全量门禁
  python3 scripts/stability_gate.py --quick    # 只跑 fuzz + 适配器(新增代码必跑)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FULL = [
    "tests/test_fuzzing_chain.py",
    "tests/test_prediction_verify.py",
    "tests/test_engine_fusion.py",
    "quant_system/tests/test_daily_review_chain.py",
]
QUICK = ["tests/test_fuzzing_chain.py", "tests/test_engine_fusion.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    suites = QUICK if args.quick else FULL
    all_ok = True
    print(f"🔒 稳定化门禁（{'quick' if args.quick else 'full'}）")
    for suite in suites:
        print(f"  · pytest {suite} ...", flush=True)
        pr = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-q", "--no-header"],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
        ok = pr.returncode == 0
        all_ok = all_ok and ok
        tail = (pr.stdout or "")[-300:] + (pr.stderr or "")[-200:]
        print(f"    {'✅' if ok else '❌'} rc={pr.returncode}")
        if not ok:
            print(tail[-400:])
    print("\n🔒 门禁结论:", "✅ 通过, 可合入" if all_ok else "❌ 未通过, 禁止合入")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())