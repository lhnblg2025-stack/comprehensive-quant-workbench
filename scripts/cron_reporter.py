#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 cron 当日汇总汇报（2026-08-22 —— 用户"所有云端cron跑了都要汇报,当天汇报一次")

扫描全部 quant_* 计划任务的上次运行时间+结果码 → 汇总当日执行情况 → 飞书推送一次。
用 schtasks /query /v /fo csv 解析(Windows)。

用法:
  python3 scripts/cron_reporter.py --report   # 扫描+汇总+推送飞书
  python3 scripts/cron_reporter.py --dry      # 只打印不推送
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fetch_tasks() -> list[dict]:
    """schtasks /query /v /fo csv 解析出全部任务(名称/上次运行/上次结果/状态/时间)."""
    r = subprocess.run(
        ["schtasks", "/query", "/v", "/fo", "csv"],
        capture_output=True, timeout=60)
    txt = r.stdout.decode("gbk", errors="replace")
    rows = list(csv.reader(io.StringIO(txt)))
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    tasks = []
    for row in rows[1:]:
        if len(row) < len(header):
            continue
        d = dict(zip(header, [c.strip() for c in row]))
        tasks.append(d)
    return tasks


def quant_tasks(tasks: list[dict]) -> list[dict]:
    """过滤 quant_* 任务, 补关键字段。"""
    out = []
    for t in tasks:
        name = t.get("任务名") or t.get("Task Name") or ""
        if not name.lower().startswith("quant_") and "quant_" not in name.lower():
            continue
        last_run = t.get("上次运行时间") or t.get("Last Run Time") or ""
        result = t.get("上次结果") or t.get("Last Result") or ""
        status = t.get("状态") or t.get("Status") or ""
        sch = t.get("计划类型") or t.get("Schedule Type") or ""
        out.append({"name": name, "last": last_run, "result": result,
                    "status": status, "schedule": sch})
    return out


def parse_run_time(s: str):
    """解析'2026/8/22 18:30:00' → datetime(含中文日期变形容错)."""
    if not s or s in ("N/A", "从未运行", "-"):
        return None
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None


_EMOJI = "✅❌☁️⚠️🔴🟢📊📈🔥💬📐🗄️🗃️🌍📚🔬💧🌡️🚀💰🎯🤖🧬⛔🗓📅"


def _clean(s: str) -> str:
    """去 emoji(飞书GBK安全)."""
    for ch in _EMOJI:
        s = s.replace(ch, "")
    return s


def build_report(today: date | None = None) -> str:
    """当日 cron 执行汇总 → 报告文本。"""
    today = today or date.today()
    tasks = quant_tasks(fetch_tasks())
    if not tasks:
        return "## 云端任务今日汇总\n- 未找到 quant_* 任务(可能schtasks不可用)"
    # 当日运行过的(上次运行日期==今天)
    ran, not_ran = [], []
    for t in tasks:
        dt = parse_run_time(t["last"])
        if dt and dt.date() == today:
            ok = t["result"] == "0"
            ran.append({**t, "ok": ok, "run_dt": dt})
        else:
            not_ran.append(t)
    ok_n = sum(1 for r in ran if r["ok"])
    fail_n = len(ran) - ok_n
    dry_n = len(not_ran)
    L = [f"## 云端任务今日汇总（{today}）",
         f"当日执行 <b>{len(ran)}</b> 项 · 成功 <b>{ok_n}</b> · 失败 <b style='color:#f23645'>{fail_n}</b> · 未运行 {dry_n}",
         ""]
    if ran:
        L.append("### ✅ 执行明细（按时间）")
        for r in sorted(ran, key=lambda x: x["run_dt"]):
            st = "[OK]" if r["ok"] else "[FAIL]"
            L.append(f"{st} {r['name']}  {r['run_dt'].strftime('%H:%M')}  结果{r['result']}")
    if fail_n:
        L.append("")
        L.append("### ⚠️ 失败项（需处理）")
        for r in ran:
            if not r["ok"]:
                L.append(f"- [FAIL] <b>{r['name']}</b> 结果 {r['result']}（上次{r['last']}）")
    return _clean("\n".join(L))


def push_report() -> bool:
    """汇总+推送飞书。"""
    rep = build_report()
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from feishu_sender import send_markdown
        ok = send_markdown(rep, title="云端任务日报")
        return ok
    except Exception as e:  # noqa: BLE001
        print("推送失败:", str(e)[:100])
        return False


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if "--dry" in sys.argv:
        print(build_report())
    else:
        ok = push_report()
        print("推送:", ok)
        if "--print" in sys.argv:
            print(build_report())