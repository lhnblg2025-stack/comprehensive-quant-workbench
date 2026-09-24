#!/usr/bin/env python3
"""数据健康看板 digest —— 汇总 data_freshness.json + 云快照自检 → generated/data_health_{YYYYMMDD}.md

退出码（供 cron 判断告警）:
  0  全部数据集新鲜
  1  存在 stale 数据集（非 critical）
  2  存在 stale 的 critical 数据集（宏观核心 CPI/PPI/M2/PMI 等）
     critical 清单可用环境变量 DATA_HEALTH_CRITICAL 覆盖（逗号分隔的数据集文件名）

用法:
  python3 scripts/data_health_digest.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRESHNESS_FILE = ROOT / "data_warehouse" / "data_freshness.json"
OUT_DIR = ROOT / "generated"
CST = timezone(timedelta(hours=8))

DEFAULT_CRITICAL = [
    "cpi_yearly.parquet",
    "ppi_yearly.parquet",
    "m2_yearly.parquet",
    "pmi_yearly.parquet",
    "cx_pmi_yearly.parquet",
]


def critical_set() -> set[str]:
    raw = os.environ.get("DATA_HEALTH_CRITICAL", "").strip()
    if not raw:
        return set(DEFAULT_CRITICAL)
    return {x.strip() for x in raw.split(",") if x.strip()}


def load_freshness() -> dict:
    if not FRESHNESS_FILE.exists():
        raise FileNotFoundError(f"缺少 {FRESHNESS_FILE}（先跑 stamp_data_freshness.py）")
    return json.loads(FRESHNESS_FILE.read_text(encoding="utf-8"))


def load_cloud_status() -> dict:
    """云快照自检 {源: {ok, note, as_of, n_files}}。优先直接 import，失败退回 subprocess 解析。"""
    try:
        from quant_system.analysis_core.data_sources import _cloud_snapshots_status
        return _cloud_snapshots_status()
    except Exception as exc:
        try:
            out = subprocess.run(
                [sys.executable, "-m", "quant_system.analysis_core.data_sources", "--cloud-status"],
                capture_output=True, text=True, timeout=120, cwd=str(ROOT),
            )
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or f"exit {out.returncode}") from exc
            return json.loads(out.stdout)
        except Exception as e:
            print(f"  [digest] 云快照状态获取失败: {type(e).__name__}: {e}", file=sys.stderr)
            return {}


def fmt_as_of(as_of: str | None) -> str:
    if not as_of:
        return "-"
    s = str(as_of)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def main() -> int:
    stamp = datetime.now(CST)
    today = stamp.strftime("%Y%m%d")
    try:
        freshness = load_freshness()
    except FileNotFoundError as e:
        print(f"[data-health] {e}", file=sys.stderr)
        return 1

    crit = critical_set()
    entries = {k: v for k, v in freshness.items() if isinstance(v, dict) and "stale" in v}
    stale_by_key = {k: v for k, v in entries.items() if v.get("stale")}
    crit_stale = [
        (k, v) for k, v in stale_by_key.items()
        if k in crit or Path(str(v.get("file", ""))).name in crit
    ]

    total = len(entries)
    n_stale = len(stale_by_key)
    n_fresh = total - n_stale
    non_tracked = len(freshness) - total
    cloud = load_cloud_status()

    if crit_stale:
        status_cn = f"🔴 critical 陈旧 {len(crit_stale)} 个（需告警）"
    elif stale_by_key:
        status_cn = f"⚠️ 陈旧 {n_stale} 个"
    else:
        status_cn = "✅ 全部新鲜"

    stale_sorted = sorted(stale_by_key.items(), key=lambda kv: kv[1].get("stale_days") or 0, reverse=True)

    md = [
        f"# 数据健康看板 {today}",
        "",
        f"- 生成时间: {stamp.strftime('%Y-%m-%d %H:%M')} (+08:00)",
        f"- 数据源: {FRESHNESS_FILE.name} + 云快照自检",
        f"- 数据集总数: {total} | 新鲜: {n_fresh} | 陈旧: {n_stale}"
        f"（另有 {non_tracked} 项静态快照/元数据无 stale 追踪，未计入）",
        f"- 状态: {status_cn}",
        "",
        "## 陈旧清单",
        "",
        f"共 {n_stale} 个陈旧数据集（按 stale_days 降序）:",
        "",
        "| 数据集 | 源 | as_of | stale_days | note |",
        "|---|---|---|---|---|",
    ]
    for k, v in stale_sorted:
        md.append(
            f"| {k} | {v.get('source') or '-'} | {fmt_as_of(v.get('as_of'))} | "
            f"{v.get('stale_days', '-')} | {v.get('note') or ''} |"
        )
    if not stale_sorted:
        md.append("| （无） | - | - | - | - |")

    md += ["", "## 关键数据集状态（critical，stale 时退出码=2）", "",
           "| 数据集 | 状态 | as_of | stale_days |", "|---|---|---|---|"]
    for k in sorted(crit):
        if k not in entries:
            md.append(f"| {k} | ⚠️ 未收录 | - | - |")
            continue
        v = entries[k]
        st = "🔴 stale" if v.get("stale") else "✅ 新鲜"
        md.append(f"| {k} | {st} | {fmt_as_of(v.get('as_of'))} | {v.get('stale_days', '-')} |")

    md += ["", "## 云源快照状态", "",
           "| 源 | ok | as_of | 文件数 | 说明 |", "|---|---|---|---|---|"]
    if cloud:
        for name, st in cloud.items():
            md.append(
                f"| {name} | {'✅ ok' if st.get('ok') else '❌ false'} | "
                f"{fmt_as_of(st.get('as_of'))} | {st.get('n_files', '-')} | {st.get('note') or ''} |"
            )
    else:
        md.append("| （获取失败） | - | - | - | 见运行 stderr |")

    md += ["", "## TOP 陈旧 TOP10", ""]
    for i, (k, v) in enumerate(stale_sorted[:10], 1):
        md.append(f"{i}. **{k}** — {v.get('stale_days', '-')} 天（as_of={fmt_as_of(v.get('as_of'))}）")
    if not stale_sorted:
        md.append("（无陈旧数据集）")

    md += ["", "---", "", "退出码语义: 0=全新鲜 / 1=存在陈旧 / 2=critical 陈旧（cron 告警依据）", ""]

    out_path = OUT_DIR / f"data_health_{today}.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import os as _os
    _tmp = out_path.with_suffix(".md.tmp")
    _tmp.write_text("\n".join(md), encoding="utf-8")
    _os.replace(_tmp, out_path)

    exit_code = 2 if crit_stale else (1 if stale_by_key else 0)
    print(f"[data-health] 看板 → {out_path}")
    print(f"[data-health] 数据集 {total} | 新鲜 {n_fresh} | 陈旧 {n_stale} | "
          f"critical陈旧 {len(crit_stale)} → exit {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
