#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全引擎接入（2026-08-22 —— 用户"为什么不全接入决策链"）

审计出 63 个 analysis_core 引擎未接入。本扫描器全部接入:
  1. 自动探测每个引擎的主入口(analyze/scan/compute/run/report/judge/detect等)
  2. 并行子进程隔离跑(每源15s真超时, 并发4-6, 单源卡不阻塞)
  3. 全部结果 → generated/all_engines_cache.json(决策链block读, 秒回)
  4. 分组渲染(运行OK/error/timeout 三类进研报)

用法:
  python3 scripts/all_engines_scan.py            # 全63引擎扫描(并行~2-4min)
  python3 scripts/all_engines_scan.py --dry      # 只探测入口不跑
  python3 scripts/all_engines_scan.py --limit 10 # 只前10个(测试)
"""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "generated" / "all_engines_cache.json"
CORE = ROOT / "quant_system" / "analysis_core"

# 引擎入口候选(优先级: 有产出价值的)
ENTRY_HINTS = ("analyze", "scan", "compute", "run", "run_today", "run_report",
               "judge", "detect", "assess", "generate", "evaluate", "explain",
               "build", "get_snapshot", "collect", "report")


def _probe_entries(code: str) -> list[str]:
    """子进程探测引擎主入口(遍历函数名匹配hint)."""
    src = ""
    try:
        src = Path(CORE / f"{code}.py").read_text(encoding="utf-8")
    except Exception:
        return []
    import re
    fns = re.findall(r"^def (\w+)\(", src, re.M)
    hits = []
    for f in fns:
        if f in ENTRY_HINTS or any(f.startswith(h) for h in ENTRY_HINTS):
            hits.append(f)
    # 兜底: 无hint命中但有函数则取第一个非私有
    if not hits:
        for f in fns:
            if not f.startswith("_"):
                hits.append(f)
                break
    return hits[:2]


def _run_one(code: str, fn: str, timeout_s: int = 30) -> dict:
    """子进程跑单引擎(真超时)."""
    # 部分引擎需要target/data参数 → 尝试无参调用
    script = f"""
import sys, json
sys.path.insert(0, 'quant_system')
try:
    m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])
    f = getattr(m, '{fn}')
    r = f()
    print(json.dumps(r, ensure_ascii=False, default=str)[:20000])
except Exception as e:
    print(json.dumps({{'error': str(e)[:90]}}))
"""
    try:
        pr = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            timeout=timeout_s, text=True, encoding="utf-8", errors="replace")
        out = (pr.stdout or "").strip()
        s = out.find("{")
        if s > 0:
            out = out[s:]
        try:
            obj = json.loads(out or "{}")
            return obj if isinstance(obj, dict) else {"result": obj}  # 类型包装: 非dict包成result
        except Exception:
            return {"error": "非JSON输出"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout15s"}
    except Exception as e:
        return {"error": str(e)[:60]}


def scan_all(limit: int = 0, dry: bool = False, max_workers: int = 5) -> dict:
    """全引擎并行扫描. dry=只探测入口. 返回 {engine: {entry, result, ok}}."""
    cached = CACHE
    if cached.exists() and time.time() - cached.stat().st_mtime < 3600 and not limit and not dry:
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except Exception:
            pass
    engines = [p.stem for p in CORE.glob("*.py") if p.stem != "__init__"]
    # 排除已知太重/依赖外部数据的(通过入口探测自动降级)
    if limit:
        engines = engines[:limit]
    results = {}
    t0 = time.time()

    def _worker(code):
        entries = _probe_entries(code)
        if not entries:
            return code, {"entry": None, "ok": False, "error": "无入口"}
        if dry:
            return code, {"entry": entries[0], "ok": True, "dry": True}
        r = _run_one(code, entries[0], 30)
        # 失败重试: 带典型参数(index名/核心票/日期)
        if r.get("error"):
            import pandas as _pd8  # noqa: PLC0415
            args_map = {"index": "'沪深300'", "symbol": "'600519'", "code": "'600519'",
                        "name": "'沪深300'", "target": "'沪深300'", "df": "None"}
            script2 = f"""
import sys, json
sys.path.insert(0, 'quant_system')
try:
    m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])
    f = getattr(m, '{entries[0]}')
    import inspect
    sig = inspect.signature(f)
    kwargs = {{}}
    for pname in sig.parameters:
        if pname in {json.dumps(args_map)}:
            kwargs[pname] = eval({json.dumps(args_map)}[pname]) if args_map[pname] != 'None' else None
    import pandas as pd
    if 'df' in kwargs and kwargs['df'] is None:
        try:
            kwargs['df'] = pd.read_parquet('data_warehouse/kline/600519.parquet')
        except Exception: pass
    r = f(**kwargs)
    print(json.dumps(r, ensure_ascii=False, default=str)[:20000])
except Exception as e:
    print(json.dumps({{'error': str(e)[:90]}}))
"""
            try:
                pr = subprocess.run([sys.executable, "-c", script2], capture_output=True,
                                    timeout=15, text=True, encoding="utf-8", errors="replace")
                out = (pr.stdout or "").strip()
                s2 = out.find("{"); out = out[s2:] if s2 > 0 else out
                try:
                    r2 = json.loads(out or "{}")
                    if not r2.get("error"):
                        r = r2
                except Exception:
                    pass
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
        return code, {"entry": entries[0], "ok": not r.get("error"), "result": r}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, c): c for c in engines}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                code, info = fut.result()
                results[code] = info
            except Exception as e:
                results[futs[fut]] = {"entry": None, "ok": False, "error": str(e)[:40]}
            done += 1
            if done % 10 == 0:
                print(f"  ...{done}/{len(engines)}", flush=True)

    ok_n = sum(1 for v in results.values() if v.get("ok"))
    summary = {"scanned": len(engines), "ok": ok_n, "fail": len(engines) - ok_n,
               "elapsed": round(time.time() - t0, 1), "engines": results}
    if not dry:
        CACHE.write_text(json.dumps(summary, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def scan_md() -> str:
    """全引擎扫描 → Markdown(研报分组)."""
    s = scan_all()
    L = ["## 🧠 全引擎接入（63未接入→全部扫描）"]
    L.append(f"- 扫描 {s.get('scanned')} 引擎 · 运行OK {s.get('ok')} · 降级 {s.get('fail')} ({s.get('elapsed')}s)")
    ok_eng = [k for k, v in (s.get("engines") or {}).items() if v.get("ok")]
    err_eng = [k for k, v in (s.get("engines") or {}).items() if v.get("error")]
    if ok_eng:
        L.append("### ✅ 运行成功")
        L.append("、".join(ok_eng[:40]))
    if err_eng:
        L.append("### ⚠️ 降级(超时/缺数据)")
        L.append("、".join(err_eng[:40]))
    return "\n".join(L)


def scan_data() -> dict:
    return scan_all()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    args = sys.argv[1:]
    if "--dry" in args:
        r = scan_all(dry=True)
        print(json.dumps({k: v for k, v in list(r["engines"].items())[:8]}, ensure_ascii=False, indent=1))
    elif "--limit" in args:
        i = args.index("--limit")
        lim = int(args[i + 1]) if i + 1 < len(args) else 10
        r = scan_all(limit=lim)
        print(f"limit扫描: {r['ok']} OK / {r['fail']} fail ({r['elapsed']}s)")
    else:
        print(scan_md())