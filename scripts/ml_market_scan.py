#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML 全市场扫描（2026-08-22 —— 方案v8: 全池预测不留2只）

扫 data_warehouse/kline 全部个股 → predict_ensemble 批量预测
→ 输出 Top强势(up概率最高) / Top回避(down概率最高) 榜单。
带缓存(当日重复跑秒回) + 限时(每只超时跳)。

用法:
  python3 scripts/ml_market_scan.py           # 全量扫描(缓存1h)
  python3 scripts/ml_market_scan.py --limit 200  # 限样本(测试)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "generated" / "ml_scan_cache.json"
KL = ROOT / "data_warehouse" / "kline"


def _cached():
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 3600:
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def scan(limit: int = 0, core_only: bool = False) -> dict:
    """扫 kline 全池预测 → 强势/回避榜(core_only=只扫50核心池快测)."""
    sys.path.insert(0, str(ROOT / "quant_system"))
    from quant_system.ml_signals import predict_ensemble  # noqa: PLC0415
    from quant_system.ml_signals import DEFAULT_TRAIN_SYMBOLS  # noqa: PLC0415
    cached = _cached()
    if cached:
        return cached
    files = sorted(KL.glob("*.parquet"))
    if core_only:
        files = [KL / f"{s}.parquet" for s in DEFAULT_TRAIN_SYMBOLS]
    if limit:
        files = files[:limit]
    results = []
    t0 = time.time()
    for i, f in enumerate(files):
        sym = f.stem
        try:
            r = predict_ensemble(sym, horizon_days=5)
            if r and isinstance(r, dict) and r.get("prediction_label"):
                probs = r.get("probs") or {}
                results.append({"symbol": sym, "label": r["prediction_label"],
                                "conf": round(float(r.get("confidence", 0) or 0), 3),
                                "up": round(float(probs.get("up", 0) or 0), 3),
                                "down": round(float(probs.get("down", 0) or 0), 3)})
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    # 分类
    up_list = sorted([r for r in results if r["up"] > 0.4], key=lambda x: -x["up"])
    down_list = sorted([r for r in results if r["down"] > 0.4], key=lambda x: -x["down"])
    out = {"scanned": len(results), "elapsed": round(time.time() - t0, 1),
           "top_up": up_list[:15], "top_down": down_list[:15],
           "up_n": len(up_list), "down_n": len(down_list)}
    CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def scan_md() -> str:
    """全池扫描 → Markdown(研报/复盘用)。"""
    r = scan()
    if not r.get("top_up") and not r.get("top_down"):
        return "## 🤖 ML全市场扫描\n- 无结果(模型或数据不足)"
    L = ["## 🤖 ML全市场扫描（5204股）"]
    L.append(f"- 扫描 {r.get('scanned')} 只({r.get('elapsed')}s) · 强势信号{r.get('up_n')} · 回避信号{r.get('down_n')}")
    if r.get("top_up"):
        L.append("### 🟢 强势候选(up概率最高)")
        L.append(" | ".join(f"{x['symbol']} up{x['up']:.2f}" for x in r["top_up"][:8]))
    if r.get("top_down"):
        L.append("### 🔴 回避预警(down概率最高)")
        L.append(" | ".join(f"{x['symbol']} dn{x['down']:.2f}" for x in r["top_down"][:8]))
    return "\n".join(L)


def scan_data() -> dict:
    return scan()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    args = sys.argv[1:]
    limit = 200 if "--limit" in args else 0
    core = "--core" in args
    if "--limit" in args:
        i = args.index("--limit")
        if i + 1 < len(args):
            limit = int(args[i + 1])
    r = scan(limit, core_only=core)
    if limit:
        print(json.dumps(r, ensure_ascii=False)[:500])
    else:
        print(scan_md())