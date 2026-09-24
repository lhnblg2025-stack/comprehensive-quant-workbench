#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""three_endpoint_panel — 三端统一巡检面板（W3.5，2026-08-23）

一张图看三端：
  - 本机   : 关键服务(8600/8767/3080)存活 + 数据仓库新鲜度
  - 腾讯云 : SSH 连通 + quant schtasks 数 + 关键数据目录
  - Vultr  : SSH 连通 + 磁盘使用率 + 日志规模

用法:
  python3 scripts/three_endpoint_panel.py          # 全量巡检
  python3 scripts/three_endpoint_panel.py --json   # JSON 输出(供 cron/看板)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
# 安全加固后 PEM 优先取受保护副本，兼容回退原共享路径
_SECURE_PEM = Path.home() / ".openclaw" / "credentials" / "quant_cloud.pem"
PEM = str(_SECURE_PEM) if _SECURE_PEM.exists() else "/mnt/hgfs/share/openclaw.pem"
TX = "Administrator@quant-cloud.example.com"
VULTR = "root@quant-relay.example.com"


def _ssh(host: str, cmd: str, timeout: int = 25) -> tuple[bool, str]:
    known_hosts = os.environ.get("QUANT_CLOUD_KNOWN_HOSTS", "").strip()
    host_key_opts = (["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}"]
                     if known_hosts else ["-o", "StrictHostKeyChecking=accept-new"])
    args = ["ssh", "-i", PEM, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            *host_key_opts, host, cmd]
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout)
        out = r.stdout.decode("utf-8", errors="replace")
        return r.returncode == 0, out.strip()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _local_service(port: int) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _state(required: list[bool | None]) -> str:
    """四态：全真=healthy，部分真=degraded，全不可知=unknown，明确全假=failed。"""
    known = [v for v in required if v is not None]
    if not known:
        return "unknown"
    if all(known) and len(known) == len(required):
        return "healthy"
    if any(known):
        return "degraded"
    return "failed"


def check_local() -> dict:
    svc = {p: _local_service(p) for p in (8600, 8767, 3080)}
    kline = sorted((ROOT / "data_warehouse" / "kline").glob("*.parquet")) if (ROOT / "data_warehouse" / "kline").exists() else []
    kline_age_h = None
    if kline:
        import time
        kline_age_h = round((time.time() - max(p.stat().st_mtime for p in kline)) / 3600, 1)
    state = _state(list(svc.values()) + [kline_age_h is not None and kline_age_h <= 72])
    return {"state": state, "services": svc, "kline_files": len(kline), "kline_age_h": kline_age_h}


def check_tx() -> dict:
    ok, out = _ssh(TX, "schtasks /query /fo csv | find /c \"\\quant_\"")
    tasks = out.strip() if ok else None
    ok2, _ = _ssh(TX, "dir C:\\quant\\data_warehouse\\financial /-c | find /c \".parquet\"")
    return {"state": _state([ok, ok2]), "ssh": ok, "quant_tasks": tasks, "financial_parquet_ok": ok2}


def check_vultr() -> dict:
    ok, out = _ssh(VULTR, "df -h / | tail -1 | awk '{print $5}'")
    disk = out.strip() if ok else None
    ok2, out2 = _ssh(VULTR, "du -sm /var/log 2>/dev/null | awk '{print $1}'")
    log_mb = out2.strip() if ok2 else None
    disk_pct = None
    try:
        disk_pct = int(str(disk or "").rstrip("%"))
    except ValueError:
        pass
    return {"state": _state([ok, ok2, disk_pct is not None and disk_pct < 80]),
            "ssh": ok, "disk_use": disk, "disk_use_pct": disk_pct, "var_log_mb": log_mb}


def render(results: dict) -> str:
    L = [f"# 🖥️ 三端统一巡检（{datetime.now(CST):%Y-%m-%d %H:%M}）", ""]
    lo = results["local"]
    L.append("## 本机")
    L.append(f"- 服务 8600(quant_web)/8767(claude_proxy)/3080(DSH): "
             f"{lo['services'][8600]}/{lo['services'][8767]}/{lo['services'][3080]}")
    L.append(f"- K线: {lo['kline_files']} 文件, 最新 {lo['kline_age_h']}h 前")
    tx = results["tx"]
    L.append("## 腾讯云 " + ("✅" if tx["ssh"] else "❌"))
    if tx["ssh"]:
        L.append(f"- quant schtasks: {tx['quant_tasks']} · financial 产物: {'✅' if tx['financial_parquet_ok'] else '❌'}")
    vu = results["vultr"]
    L.append("## Vultr " + ("✅" if vu["ssh"] else "❌"))
    if vu["ssh"]:
        L.append(f"- 磁盘: {vu['disk_use']} · /var/log: {vu['var_log_mb']}MB")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = {
        "local": check_local(),
        "tx": check_tx(),
        "vultr": check_vultr(),
        "checked_at": datetime.now(CST).isoformat(timespec="seconds"),
    }
    states = [results[k]["state"] for k in ("local", "tx", "vultr")]
    if all(s == "healthy" for s in states):
        results["overall_state"] = "healthy"
    elif any(s == "failed" for s in states):
        results["overall_state"] = "failed"
    elif any(s == "unknown" for s in states):
        results["overall_state"] = "unknown"
    else:
        results["overall_state"] = "degraded"
    stamp = datetime.now(CST).strftime("%Y%m%dT%H%M%S")
    panel_dir = ROOT / "generated" / "endpoint_health"
    panel_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_dir / f"endpoint_health_{stamp}.json"
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    manifest_path.write_text(payload, encoding="utf-8")
    (panel_dir / "latest.json").write_text(payload, encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(render(results))
    # unknown/degraded 需要告警但不冒充成功；脚本仅对明确 failed 返回非零，便于网络隔离环境采集证据。
    return 0 if results["overall_state"] in ("healthy", "degraded", "unknown") else 1


if __name__ == "__main__":
    sys.exit(main())
