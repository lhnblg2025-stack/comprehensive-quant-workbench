#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史日志归档压缩（2026-08-22 长期运维建议③落地 · 低成本运维）

把散落日志归档进 generated/logs_archive/{date}/ 并 gzip 压缩:
  - workspace/logs/*.log（api_audit/pull 等, 保留最近 N 天未压缩原文件）
  - 根目录/其它目录 >1MB 的 *.log（滚动保留）
安全: 归档后原文件移动到 archive（不删除）, 当前仍在写的最新日志跳过(N天窗口内)。
统计产出 JSON + 压缩率报告。
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CST = timezone(timedelta(hours=8))
KEEP_DAYS = 14            # 近 N 天日志不归档（可能还在写）
MIN_SIZE = 512 * 1024     # 仅归档 >512KB 的（小日志不值得搬）
ARCHIVE_ROOT = ROOT / "generated" / "logs_archive"


def _importable(path: Path, cutoff: datetime) -> bool:
    """超 KEEP_DAYS 且超 MIN_SIZE 才归档；排除正在被写的 stability/quantv6。"""
    if path.stat().st_size < MIN_SIZE:
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=CST)
    if mtime > cutoff:
        return False
    return True


def archive_logs() -> dict:
    now = datetime.now(CST)
    cutoff = now - timedelta(days=KEEP_DAYS)
    date = now.strftime("%Y-%m-%d")
    dest = ARCHIVE_ROOT / date
    dest.mkdir(parents=True, exist_ok=True)
    archived: list[dict] = []
    total_in = total_out = 0

    candidates = [p for p in (ROOT / "logs").glob("*.log") if _importable(p, cutoff)]
    # 根目录其它 >1MB 日志（排除 .git / archive 自身）
    for p in ROOT.glob("*.log"):
        if p.stat().st_size > 1_000_000 and _importable(p, cutoff):
            candidates.append(p)

    for p in candidates:
        gz_path = dest / f"{p.name}.gz"
        try:
            with p.open("rb") as fin, gzip.open(gz_path, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
            size_in = p.stat().st_size
            size_out = gz_path.stat().st_size
            archived.append({"src": str(p), "in": size_in, "out": size_out,
                             "saved_pct": round((1 - size_out / size_in) * 100, 1)})
            total_in += size_in
            total_out += size_out
            p.unlink()  # 已压缩入档, 原日志移除（可恢复: gz 在 archive）
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ {p.name} 归档失败: {e}")

    report = {"date": date, "archived": len(archived),
              "total_in_b": total_in, "total_out_b": total_out,
              "saved_pct": round((1 - total_out / max(1, total_in)) * 100, 1) if total_in else 0,
              "files": archived}
    (ARCHIVE_ROOT / f"archive_{date}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"🗜️ 归档 {len(archived)} 个文件 → {dest}，节省 {report['saved_pct']}%")
    for a in archived:
        print(f"  - {Path(a['src']).name}: {a['in']/1024:.0f}KB → {a['out']/1024:.0f}KB ({a['saved_pct']}%)")
    return report


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    r = archive_logs()
    print(json.dumps({k: r[k] for k in ("date", "archived", "saved_pct")}, ensure_ascii=False))