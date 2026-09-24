#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""周日稳定化批次（2026-08-22 长期运维建议①落地）

低波动时间段(默认周日早)串行跑三个稳定性资产，任一失败 → 飞书告警:
  1. scripts/extreme_market_drill.py    —— 极端行情注入压力测试(8场景)
  2. scripts/chaos_drill.py             —— 三断混沌演练(断源/断库/断网)
  3. quant_system/stress_baseline.py --check —— 性能/内存基线退化检查

安全: 全部为 dry-run/内存注入, 不真实断网/删文件; 输出日志与 md 汇总。

用法:
  python3 scripts/weekly_stability_drill.py            # 三件套连跑 + 失败告警
  python3 scripts/weekly_stability_drill.py --skip-baseline   # 跳过基线(无基线时)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CST = timezone(timedelta(hours=8))
LOG_DIR = ROOT / "generated" / "logs"
OUT = ROOT / "generated" / "stability_drill"

STEPS = [
    {"name": "extreme_market_drill", "cmd": [sys.executable, "scripts/extreme_market_drill.py"], "timeout": 400},
    {"name": "chaos_drill", "cmd": [sys.executable, "scripts/chaos_drill.py"], "timeout": 900},
]


def _run_step(step: dict, log_f) -> dict:
    t0 = time.time()
    try:
        pr = subprocess.run(step["cmd"], cwd=str(ROOT), capture_output=True,
                            timeout=step["timeout"], text=True, encoding="utf-8",
                            errors="replace")
        ok = pr.returncode == 0
        tail = (pr.stdout or "")[-1200:] + (pr.stderr or "")[-600:]
        return {"name": step["name"], "ok": ok, "rc": pr.returncode,
                "secs": round(time.time() - t0, 1), "tail": tail}
    except subprocess.TimeoutExpired:
        return {"name": step["name"], "ok": False, "rc": "timeout",
                "secs": round(time.time() - t0, 1), "tail": f"超时>{step['timeout']}s"}
    except Exception as e:  # noqa: BLE001
        return {"name": step["name"], "ok": False, "rc": "err",
                "secs": round(time.time() - t0, 1), "tail": str(e)[:300]}


def run(skip_baseline: bool = False, push: bool = True) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    date = datetime.now(CST).strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"stability_drill_{date}.log"
    results: list[dict] = []
    with log_path.open("w", encoding="utf-8") as log_f:
        steps = list(STEPS)
        if not skip_baseline:
            bl = ROOT / "generated" / "stress_baseline.json"
            steps.append({"name": "stress_baseline_check",
                          "cmd": [sys.executable, "quant_system/stress_baseline.py", "--check"],
                          "timeout": 900})
        for i, step in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {step['name']} ...", flush=True)
            r = _run_step(step, log_f)
            results.append(r)
            log_f.write(f"[{datetime.now(CST).strftime('%H:%M:%S')}] {r['name']}: "
                        f"{'✅' if r['ok'] else '❌'} rc={r['rc']} {r['secs']}s\n{r['tail']}\n")
            log_f.flush()
            print(f"  {'✅' if r['ok'] else '❌'} {r['name']} rc={r['rc']} {r['secs']}s", flush=True)

    all_ok = all(r["ok"] for r in results)
    md = [f"# 🛡️ 周日稳定化批次（{date}）",
          f"- 结果: **{'✅ 全部通过' if all_ok else '❌ 存在失败'}**",
          ""]
    for r in results:
        md.append(f"### {'✅' if r['ok'] else '❌'} {r['name']}（rc={r['rc']} · {r['secs']}s）")
        if not r["ok"]:
            md.append(f"```\n{r['tail'][-800:]}\n```")
        md.append("")
    md_txt = "\n".join(md)
    (OUT / f"stability_{date}.md").write_text(md_txt, encoding="utf-8")

    if push and not all_ok:
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from feishu_sender import send_markdown  # noqa: PLC0415
            send_markdown(f"🚨 周日稳定化批次发现异常\n\n{md_txt}", title=f"稳定化批次 {date}")
            print("⚠️ 已推送飞书告警(失败)")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 飞书告警失败: {e}")
    return {"date": date, "all_ok": all_ok, "results": results,
            "log": str(log_path), "md": str(OUT / f"stability_{date}.md")}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    r = run(skip_baseline=args.skip_baseline, push=not args.no_push)
    print(f"\n批次完成: {'✅ 全过' if r['all_ok'] else '❌ 有失败'} | 日志 {r['log']}")
    sys.exit(0 if r["all_ok"] else 1)