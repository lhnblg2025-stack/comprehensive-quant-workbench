#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日全量系统汇报 → 飞书（2026-08-21）

用户需求: "全量跑一遍要到飞书汇报"。每天盘后一条汇总消息:
  📊 今日复盘决策卡（结论）
  ✅ 正常任务清单
  ❌ 失败任务清单（含原因）
  ⚠️ 数据源状态

用法:
  python3 scripts/daily_delivery_report.py  # 读云端任务状态+复盘 → 推飞书
"""
from __future__ import annotations

import csv
import fcntl
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(os.environ.get("QUANT_ROOT", str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("daily_delivery")
CST = timezone(timedelta(hours=8))
RUN_DIR = Path(os.environ.get("QUANT_RUN_DIR", str(ROOT / "generated" / "run"))).expanduser()
LOCK_FILE = Path(os.environ.get("DAILY_DELIVERY_LOCK", str(RUN_DIR / "daily_delivery_report.lock"))).expanduser()

PEM = os.environ.get("QUANT_CLOUD_KEY", "").strip()
HOST_TX = os.environ.get("QUANT_CLOUD_HOST", "").strip()
KNOWN_HOSTS = Path(os.environ.get("QUANT_KNOWN_HOSTS", str(Path.home() / ".ssh" / "known_hosts"))).expanduser()


def _ssh_tx(cmd: str, timeout: int = 40) -> str:
    if not HOST_TX or not PEM:
        raise RuntimeError("QUANT_CLOUD_HOST and QUANT_CLOUD_KEY are required")
    if not KNOWN_HOSTS.is_file():
        raise RuntimeError(f"known_hosts not found: {KNOWN_HOSTS}")
    r = subprocess.run([
        "ssh", "-i", PEM, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o", "ConnectTimeout=12", HOST_TX, cmd,
    ], capture_output=True, timeout=timeout)
    out = (r.stdout or b"") + (r.stderr or b"")
    text = out.decode("gbk", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"cloud ssh failed rc={r.returncode}: {text[:200]}")
    return text


def _parse_task_status(output: str) -> dict[str, list[str]]:
    """解析 schtasks CSV；空结果与连接错误均显式标为未知。"""
    result = {"ok": [], "fail": [], "running": [], "unknown": []}
    text = (output or "").strip()
    if not text:
        result["unknown"].append("云端任务查询无输出")
        return result
    if any(token in text.lower() for token in ("connection timed out", "connection refused", "permission denied", "连接失败")):
        result["unknown"].append(f"云端任务查询失败：{text[:120]}")
        return result
    try:
        rows = csv.reader(io.StringIO(text))
        for parts in rows:
            parts = [part.strip() for part in parts]
            if len(parts) < 2 or not parts[0] or parts[0].lower() in {"taskname", "任务名"}:
                continue
            name = parts[0]
            if not name.lower().lstrip("\\").startswith("quant_"):
                continue
            # schtasks /fo csv /v columns: TaskName, Next Run Time, Status,
            # Logon Mode, Last Run Time, Last Result, ... . Never infer a result
            # from the shorter non-/v format: its final field is often "Ready".
            status = parts[2] if len(parts) > 2 else ""
            code = parts[5] if len(parts) > 5 else ""
            normalized = code.lower()
            if "运行中" in status or status.lower() == "running":
                result["running"].append(name)
            elif normalized in {"0", "0x0"}:
                result["ok"].append(name)
            elif normalized in {"", "n/a", "未知", "unknown", "就绪", "ready"}:
                result["unknown"].append(f"{name}（无最近结果）")
            else:
                result["fail"].append(f"{name}（最近结果 {code}）")
    except csv.Error as exc:
        result["unknown"].append(f"云端任务表解析失败：{exc}")
    if not any(result.values()):
        result["unknown"].append("未发现 quant_ 任务")
    return result


def _cloud_task_status() -> dict[str, list[str]]:
    """扫云端 quant_* 计划任务，返回成功、失败、运行中和未知四类。"""
    try:
        out = _ssh_tx('cmd /c schtasks /query /v /fo csv 2>nul | findstr /i "quant_"')
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": [], "fail": [], "running": [], "unknown": [f"云端任务查询异常：{exc}"]}
    return _parse_task_status(out)


def _task_summary(status: dict[str, list[str]]) -> dict[str, int | str]:
    """成功率只使用有确定结果的任务，未知状态不计入成功。"""
    ok, fail = len(status["ok"]), len(status["fail"])
    determined = ok + fail
    rate = f"{ok / determined:.1%}" if determined else "N/A"
    return {"ok": ok, "fail": fail, "running": len(status["running"]),
            "unknown": len(status["unknown"]), "determined": determined, "rate": rate}


def _review_card(expected_date: str | None = None) -> str:
    """复盘决策卡；指定业务日期时只读同日 review，防止标题/决策串线。"""
    gen = ROOT / "generated"
    files = sorted(gen.glob("review_*.json"))
    if expected_date:
        files = [p for p in files if p.name == f"review_{expected_date}.json"]
    if not files:
        return "（复盘未生成）"
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    dec = d.get("decision") or {}
    dz = dec.get("定调", {})
    mz = dec.get("主线研判", {})
    op = dec.get("操作清单", {})
    rp = dec.get("风险预案", [])
    date = d.get("date", "")
    L = [f"📅 {date} 决策卡："]
    L.append(f"- 定调: **{dz.get('posture','?')}** | 仓位 {dz.get('position','?')}"
             + (f" | {dz.get('fund_adj','')}" if dz.get("fund_adj") else ""))
    L.append(f"- 主线: {mz.get('main','?')}（{mz.get('quality','?')}）")
    for a in (op.get("attack") or [])[:4]:
        L.append(f"- ✅ {a.get('name')} [{a.get('type')}] {a.get('action')} {a.get('why','')}")
    for a in (op.get("avoid") or [])[:2]:
        L.append(f"- ⛔ {a.get('name')} {a.get('action')}")
    for r in rp[:3]:
        L.append(f"- 🚨 {r.get('trigger')} → {r.get('action')}")
    return "\n".join(L)


def _latest_decision_snapshot(expected_date: str | None = None) -> dict:
    files = sorted((ROOT / "generated").glob("decision_snapshot_after_close_*.json"))
    if expected_date:
        files = [p for p in files if p.name == f"decision_snapshot_after_close_{expected_date}.json"]
    if not files:
        return {}
    try:
        value = json.loads(files[-1].read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        if expected_date and value.get("as_of") != expected_date:
            return {}
        return value
    except (OSError, json.JSONDecodeError):
        return {}


def _data_summary(snapshot: dict) -> list[str]:
    if not snapshot:
        return ["- 决策快照：未生成，不能确认盘后扫描、研究证据与门控是否完整。"]
    market = snapshot.get("market") or {}
    counts = snapshot.get("counts") or {}
    coverage = snapshot.get("intraday_coverage") or {}
    evidence = snapshot.get("evidence") or {}
    research = evidence.get("research") or {}
    regulatory = evidence.get("regulatory") or {}
    return [
        f"- 数据日：{snapshot.get('as_of', '-')}｜情绪 {market.get('emotion_stage', '-')}｜温度 {market.get('temperature', '-')}｜资金合力 {market.get('force_index', '-')}｜宽度 {float(market.get('breadth') or 0) * 100:.1f}%",
        f"- 扫描：全市场 {counts.get('scanned', 0)}｜候选 {counts.get('candidates', 0)}｜可执行初筛 {counts.get('execution_candidates', 0)}｜硬风险拦截 {counts.get('hard_risk_candidates', 0)}｜覆盖 {coverage.get('signal_evaluated', 0)}/{coverage.get('snapshot_rows', 0)} ({coverage.get('status', 'unknown')})",
        f"- 证据：IMA/研报 {research.get('reports', 0)} 篇｜覆盖代码 {research.get('all_codes', 0)}｜监管索引 {regulatory.get('total_hits', 0)} 条／代码 {regulatory.get('codes', 0)} 个",
    ]


def _gate_and_risks(snapshot: dict) -> tuple[list[str], list[str]]:
    if not snapshot:
        return (["- BLOCKED：未找到统一决策快照，禁止将日报视作完成的交易日决策交付。"],
                ["- 数据风险：快照缺失，日期与证据链不可审计。"])
    market = snapshot.get("market") or {}
    counts = snapshot.get("counts") or {}
    coverage = snapshot.get("intraday_coverage") or {}
    evidence = snapshot.get("evidence") or {}
    decay = evidence.get("factor_decay") or {}
    gates, risks = [], []
    force = float(market.get("force_index") or 0)
    if force < 35:
        gates.append(f"- CAUTION：资金合力 {force:.1f} < 35，候选仅观察，等待竞价和成交二次确认。")
    if coverage.get("status") != "complete":
        gates.append(f"- BLOCKED：盘中覆盖 {coverage.get('status', 'unknown')}，禁止升格可执行候选。")
    if decay.get("status") not in ("ok", "available", "ready"):
        gates.append(f"- CAUTION：因子衰减 {decay.get('status', 'unavailable')}，不纳入权重。")
    if not gates:
        gates.append("- OPEN：未触发额外数据或市场门控，仍需执行单票风控和次日确认。")
    for flag in market.get("risk_flags") or []:
        risks.append(f"- 市场风险：{flag}")
    if counts.get("hard_risk_candidates"):
        risks.append(f"- 监管风险：已拦截 {counts.get('hard_risk_candidates')} 个候选，不因涨停、低估值或净买放行。")
    if not risks:
        risks.append("- 暂无额外风险标记；不代表没有市场波动或个股公告风险。")
    return gates, risks


def build_report() -> str:
    task_status = _cloud_task_status()
    summary = _task_summary(task_status)
    snapshot = _latest_decision_snapshot()
    expected_date = str(snapshot.get("as_of") or "") or None
    gates, risks = _gate_and_risks(snapshot)
    today = datetime.now(CST).strftime("%m-%d %H:%M")
    L = [f"# 📊 每日系统运行汇报 {today}", "", "## 数据摘要", *_data_summary(snapshot), "", "## 任务交付成功率"]
    L.append(f"- 已判定 {summary['determined']} 个｜成功 {summary['ok']}｜失败 {summary['fail']}｜成功率 {summary['rate']}｜运行中 {summary['running']}｜未知 {summary['unknown']}")
    L.append("- 口径：成功率 = 成功 /（成功 + 失败）；运行中与未知不计入成功，也不掩盖为正常。")
    L.append("")
    L.append("## 任务状态")
    L.append("- 成功：" + ("、".join(task_status["ok"][:16]) if task_status["ok"] else "无"))
    L.append("- 失败：" + ("；".join(task_status["fail"][:8]) if task_status["fail"] else "无"))
    L.append("- 运行中：" + ("、".join(task_status["running"][:8]) if task_status["running"] else "无"))
    L.append("- 未知：" + ("；".join(task_status["unknown"][:6]) if task_status["unknown"] else "无"))
    L.extend(["", "## 决策门控", *gates, "", "## 风险结构", *risks, "", "## 复盘行动卡", _review_card(expected_date)])
    return "\n".join(L)


def deliver() -> bool:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.error("已有日报投递实例运行: %s", LOCK_FILE)
            return False
        report = build_report()
        from feishu_sender import send_markdown  # noqa: PLC0415
        ok = send_markdown(report, title="📊 每日系统运行汇报")
        logger.info("每日汇报飞书推送: %s", ok)
        return ok


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if deliver() else 1)