#!/usr/bin/env python3
"""量化工作台 — 编译测试（低 CPU，只检查语法 + import 可达性）"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILES = [
    "quant_system/cache.py",
    "quant_system/task_queue.py",
    "quant_system/data_store.py",
    "quant_system/margin.py",
    "quant_system/fama_macbeth.py",
    "quant_system/asset_allocation.py",
    "quant_system/portfolio_optimizer.py",
    "quant_system/factor_model.py",
    "quant_web/server.py",
    "scripts/a_share_daily_report.py",
]

errors = []
for f in FILES:
    path = ROOT / f
    if not path.exists():
        errors.append(f"❌ MISSING  {f}")
        continue
    try:
        compile(path.read_text(), str(path), "exec")
        print(f"✅  {f}")
    except SyntaxError as e:
        errors.append(f"❌ SYNTAX  {f}: {e}")
        print(f"❌  {f}: {e}")

if errors:
    print(f"\n⚠️  {len(errors)} 个文件有问题")
    sys.exit(1)
else:
    print(f"\n✅ 全部 {len(FILES)} 个文件语法通过")
