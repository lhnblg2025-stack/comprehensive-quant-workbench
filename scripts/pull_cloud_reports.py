#!/usr/bin/env python3
"""
pull_cloud_reports.py — 腾讯云日报/快照拉回本机（2026-08-14）
============================================================
背景: 日报/快照/数据健康已在腾讯云 schtasks 生成并推飞书，
本机 quant_web(8600) 的报告页读的是本机 generated/（REPORT_DIRS）。
原由 openclaw agentTurn（pull-daily-report/pull-snapshot, main session
systemEvent）承担，因 session 不可唤醒而被跳过（err=disabled）。
本脚本 headless 增量拉回，替代 agentTurn。

用法: python3 scripts/pull_cloud_reports.py [--all]
增量: 按 mtime+size 判断，只拉云端更新的文件。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 审计 2026-08-16：基础设施信息允许从环境变量注入；未设置时保留本地兼容默认
import os as _os
PEM = _os.environ.get("QUANT_CLOUD_PEM", "/mnt/hgfs/share/openclaw.pem")
HOST = _os.environ.get("QUANT_CLOUD_HOST_TX", "Administrator@124.223.219.237")
REMOTE = "C:/quant/generated"
LOCAL = ROOT / "generated"

# 拉回本机的云端产物（子目录通配，用 find 语义在云端枚举）
PATTERNS = (
    "*.md", "*.html", "*.json",
)


def _ssh(cmd: str, timeout: int = 120) -> str:
    known_hosts = _os.environ.get("QUANT_CLOUD_KNOWN_HOSTS", "").strip()
    options = ["-i", PEM, "-o", "ConnectTimeout=15"]
    if known_hosts:
        options += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}"]
    else:
        options += ["-o", "StrictHostKeyChecking=accept-new"]
    r = subprocess.run(
        ["ssh", *options, HOST, cmd],
        capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh 失败: {r.stderr.decode('gbk', errors='replace')[:200]}")
    return r.stdout.decode("gbk", errors="replace")


def _list_remote() -> list[tuple[str, int]]:
    """云端 generated 顶层文件 (name, size)。GBK 解码 + 正则解析 dir /-c 输出。

    dir /-c 文件行格式: `2026/08/13  20:30  12,345 name with spaces.md`
    （目录行、汇总行、表头行由正则自然排除。）
    """
    out = _ssh(f'dir "{REMOTE}" /-c', timeout=120)
    items = []
    for line in out.splitlines():
        m = re.match(r"^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}\s+([\d,]+)\s+(.+)$", line.strip())
        if not m:
            continue
        name = m.group(2).strip()
        if not name or name in (".", ".."):
            continue
        size = int(m.group(1).replace(",", ""))
        items.append((name, size))
    return items


def pull(force: bool = False) -> int:
    LOCAL.mkdir(parents=True, exist_ok=True)
    got = 0
    try:
        remote_items = _list_remote()
    except Exception as e:
        print(f"[pull_reports] 云端不可达: {e}", file=sys.stderr)
        return -1  # 审计 2026-08-16：不可达不得伪装成功（0）
    for name, size in remote_items:
        # 只拉关键产物: 日报/数据健康/作战地图/复盘（避免拉回全部大数据文件）
        # 只拉关键产物: 日报/数据健康/作战地图/复盘 + 关键分析 json（避免拉回全部大数据文件）
        KEY_JSON = ("battle_map_", "multi_agent_", "stock_lens_", "after_close_extra_",
                    "research_flow_", "alt_data_report_", "emotion_", "leader_follower_",
                    "review_", "delivery_receipts_", "daily_run_manifest_")
        if not (name.endswith(".md") or name.endswith(".html")
                or name in ("data_health_state.json", "alert_state.json")
                or any(name.startswith(k) and name.endswith(".json") for k in KEY_JSON)):
            continue
        local = LOCAL / name
        if local.exists() and not force:
            ls = local.stat()
            if abs(ls.st_size - size) < 4 and local.is_file():
                # review_/receipt 类产物以内容为准：大小一致即视为未变（mtime 跨平台不可靠）
                continue  # 大小一致视为未变（mtime 跨平台不可靠）
        known_hosts = _os.environ.get("QUANT_CLOUD_KNOWN_HOSTS", "").strip()
        scp_options = ["-i", PEM, "-C"]
        if known_hosts:
            scp_options += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}"]
        else:
            scp_options += ["-o", "StrictHostKeyChecking=accept-new"]
        r = subprocess.run(
            ["scp", *scp_options, f"{HOST}:{REMOTE}/{name}", str(local)],
            capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            got += 1
        else:
            print(f"[pull_reports] {name} 拉取失败: {r.stderr[:120]}", file=sys.stderr)
    print(f"[pull_reports] 拉回 {got} 个文件")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="腾讯云日报/快照拉回本机")
    ap.add_argument("--all", action="store_true", help="覆盖全部（默认增量按大小）")
    args = ap.parse_args()
    raise SystemExit(0 if pull(force=args.all) >= 0 else 1)
