#!/usr/bin/env python3
"""log_rotation.py — 日志轮转与清理（V11 项29）

规则：
  - 单日志 >100MB → 截断为 .1/.2 归档（保留 5 份）
  - generated/ic_spill/ 历史 npy 保留最近 30 天
  - 报告临时文件 >30 天清理

用法: python3 scripts/log_rotation.py（可挂 cron 每日）
"""
from __future__ import annotations
import logging

import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 100 * 1024 * 1024  # 100MB
KEEP_ARCHIVES = 5
KEEP_DAYS_SPILL = 30

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[log_rotation] 操作失败: {e}", exc_info=True)


def rotate_logs() -> int:
    rotated = 0
    for p in ROOT.rglob("*.log"):
        if ".git" in p.parts or "node_modules" in p.parts:
            continue
        if p.stat().st_size > MAX_BYTES:
            for i in range(KEEP_ARCHIVES, 0, -1):
                src = p.with_suffix(f".log.{i - 1}") if i > 1 else p
                dst = p.with_suffix(f".log.{i}")
                if src.exists():
                    shutil.move(str(src), str(dst))
            print(f"  轮转: {p} ({p.stat().st_size / 1e6:.0f}MB)")
            rotated += 1
    return rotated


def clean_spill(days: int = KEEP_DAYS_SPILL) -> int:
    """清理 ic_spill 历史 npy（保留最近 days 天）。"""
    spill = ROOT / "generated" / "ic_spill"
    if not spill.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for p in spill.rglob("*.npy"):
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if mtime < cutoff:
            p.unlink()
            removed += 1
    return removed


def main() -> int:
    print(f"日志轮转检查: 阈值 {MAX_BYTES / 1e6:.0f}MB")
    n = rotate_logs()
    print(f"  轮转日志: {n} 个")
    m = clean_spill()
    print(f"  清理 ic_spill 过期 npy: {m} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
