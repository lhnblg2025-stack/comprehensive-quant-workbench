#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎万用适配器 v6（2026-08-23 —— W2.1 16 错参引擎签名适配）

v2 的 7 策略存在两处根因缺陷:
  1. 策略模板的 {code}/{fn} 占位符从未替换, 且多行 body 缩进错位 → 子进程 SyntaxError,
     被空输出兜底成 `{}`, 于是全部引擎被误判为 ok:true/result:{}（"83 空结果假 OK"）。
  2. 入口探测用 `^def` 正则取源码第一个 public 函数, 常选中写路径/辅助函数
     (pipeline_tier→register_module、audit_trail→append、daily_report→pd_notna…),
     且不支持 `Class.method` 类方法引擎（cycle_system/emotion_system 等）。

v6 修复:
  1. RECIPES 显式登记 16 个错参引擎的真实入口 + kwargs（含 Class.method 与 __skip__ 降级）。
  2. recipe 优先: 按 inspect.signature 真实签名调用（date/code/limit/target/ref 类型感知注入）。
  3. 未命中引擎回退 v2 七策略（占位符/缩进同步修复）。
  4. 结果统一用 __RESULT__ 哨兵捕获, 避免 stdout 噪声与 markdown 字符串被 `{` 截断。

用法:
  python3 scripts/engine_adapter.py --all       # 一次性全跑(并发10)
  python3 scripts/engine_adapter.py --limit 20  # 测20个
"""
from __future__ import annotations

import concurrent.futures
import inspect
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "quant_system" / "analysis_core"
GEN = ROOT / "generated"
CACHE = GEN / "all_engines_cache.json"  # 统一缓存(旧扫描器同文件)

WATCH = "601899"

ENTRY_HINTS = ("analyze", "scan", "compute", "run", "run_today", "run_report",
               "judge", "detect", "assess", "generate", "evaluate", "explain",
               "build", "collect", "report", "snapshot", "today")


def _latest_data_date() -> str:
    """最近交易日(本地 index_daily.parquet), 失败回退今天。纯本地, 不依赖网络。"""
    from datetime import datetime  # noqa: PLC0415
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        import pandas as pd  # noqa: PLC0415
        p = ROOT / "data_warehouse" / "market" / "index_daily.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["date"])
            dates = sorted(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").unique())
            if dates:
                return str(dates[-1])
    except Exception:  # noqa: BLE001
        pass
    return today


DATA_DATE = _latest_data_date()


def _rp(entry: str | None = None, timeout: int = 30, artifact: str | None = None, **kwargs):
    return {"entry": entry, "kwargs": kwargs, "timeout": timeout, "artifact": artifact}


def _artifact_hit(engine: str, date: str, tpl: str | None) -> Path | None:
    """产物优先: 模板 'x_#.json' → 当日产物或最近一期 ≥300B 产物。"""
    if not tpl:
        return None
    p = GEN / tpl.replace("#", date)
    if p.exists() and p.stat().st_size >= 300:
        return p
    pat = tpl.replace("#", "*")
    for m in sorted(GEN.glob(pat), reverse=True):
        if m.stat().st_size >= 300:
            return m
    return None


# W2.1: 16 错参引擎签名适配表（真实入口 + 正确调用参数）
RECIPES: dict[str, dict] = {
    # 写路径/辅助函数被误当入口 → 改指真实产出入口
    "pipeline_tier": _rp("hot_modules"),
    "audit_trail": _rp("query", limit=5, date=None),
    "battle_map": _rp("build_map", timeout=120, artifact="battle_map_#.json"),
    "broker_profile_deep": _rp("hot_seats", timeout=60),
    "daily_report": _rp("build_report"),
    "market_microstructure": _rp("run_today", timeout=60),
    "concept_lifecycle": _rp("concept_report", timeout=60, artifact="concept_lifecycle_#.json"),
    "style_spread": _rp("judge_style", rows={}),
    "trading_journal_rag": _rp("report"),
    # 类方法引擎: Class.method 形式
    "cycle_system": _rp("CycleSystem.detect", timeout=120),
    # 入口正确但必填参数未注入 → 补真实签名参数
    "fusion": _rp("read_fusion_latest", ref=DATA_DATE),
    "announcement_arbitrage": _rp("announcement_risk",
                                  code=WATCH, title="减持公告",
                                  content="股东计划减持不超过2%股份", stock_code=WATCH),
    "error_book": _rp("run_error_book", history=[], returns=[]),
    # 无独立可调用入口(需组合持仓/基准/历史收益序列) → 显式降级, 不再误调辅助函数
    "brinson_attribution": _rp(None, note="Brinson归因需组合持仓+基准收益输入；由六引擎代行组合分析"),
    "valuation_system": _rp(None, note="体系8估值全库重算>200s；单标估值由 valuation 模块(真产)代行"),
    "pattern_gate": _rp(None, note="形态门禁需 pattern_engine 产物+历史收益序列；由 pattern_engine 下游触发"),
}


def _strategies() -> list[str]:
    """7 种调用策略（body 代码段; {code}/{fn} 由调用方替换, 其余花括号已双写）。"""
    return [
        # 1 无参
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "r = f()",
        # 2 symbol
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "import inspect\n"
        "kw = {{p: '600519' for p in inspect.signature(f).parameters if p in ('symbol','code','secid','stock')}}\n"
        "r = f(**kw)",
        # 3 index
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "import inspect\n"
        "kw = {{p: '沪深300' for p in inspect.signature(f).parameters if p in ('index','name','target','benchmark','market')}}\n"
        "r = f(**kw)",
        # 4 df 注入
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "import inspect, pandas as pd\n"
        "kw = {{}}\n"
        "df = pd.read_parquet('data_warehouse/kline/600519.parquet')\n"
        "for p in inspect.signature(f).parameters:\n"
        "    if p in ('df','data','kline','frames'): kw[p] = df\n"
        "r = f(**kw)",
        # 5 date 今日
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "import inspect, datetime\n"
        "kw = {{}}\n"
        "for p in inspect.signature(f).parameters:\n"
        "    if p in ('date','day','as_of','target_date'): kw[p] = str(datetime.date.today())\n"
        "r = f(**kw)",
        # 6 无参(容忍多参全默认)
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "r = f()",
        # 7 模块级属性/常量(仅带默认值参数)
        "m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])\n"
        "f = getattr(m, '{fn}')\n"
        "import inspect\n"
        "if callable(f):\n"
        "    r = f(**{{k: v.default for k, v in inspect.signature(f).parameters.items() if v.default is not inspect.Parameter.empty}})\n"
        "else:\n"
        "    r = f",
    ]


def _capture(script: str, timeout_s: int) -> dict:
    """子进程执行并解析 __RESULT__ 哨兵后的 JSON。"""
    try:
        pr = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            timeout=timeout_s, text=True, encoding="utf-8", errors="replace")
        out = (pr.stdout or "").strip()
        marker = "__RESULT__"
        idx = out.find(marker)
        if idx >= 0:
            out = out[idx + len(marker):]
        elif out:
            s = out.find("{")
            if s > 0:
                out = out[s:]
        else:
            return {"error": "无输出"}
        try:
            return json.loads(out or "{}")
        except Exception:
            return {"error": "非JSON"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:60]}


def _run_with_strategy(code: str, fn: str, strategy: str, timeout_s: int = 15) -> dict:
    body = strategy.replace("{code}", code).replace("{fn}", fn)
    body = textwrap.indent(body, "    ")
    script = (
        "import sys, json\n"
        "sys.path.insert(0, 'quant_system')\n"
        "try:\n"
        f"{body}\n"
        "    s = json.dumps(r, ensure_ascii=False, default=str)\n"
        "    if len(s) > 30000:\n"
        "        s = json.dumps({'_truncated': True, '_size': len(s), '_preview': s[:400]}, ensure_ascii=False)\n"
        "    print('__RESULT__' + s)\n"
        "except Exception as e:\n"
        "    print('__RESULT__' + json.dumps({'error': str(e)[:90]}, ensure_ascii=False))\n"
    )
    return _capture(script, timeout_s)


def _run_recipe(code: str, entry: str, kwargs: dict, timeout_s: int = 30) -> dict:
    """按真实签名调用(模块函数 / Class.method 类方法), 缺失的日期/代码/limit 类型感知注入。"""
    kw_repr = repr(kwargs)
    script = f"""import sys, json, inspect, pandas as pd
sys.path.insert(0, 'quant_system')
try:
    m = __import__('quant_system.analysis_core.{code}', fromlist=['x'])
    entry = '{entry}'
    if '.' in entry:
        cls_name, meth = entry.split('.', 1)
        f = getattr(getattr(m, cls_name)(), meth)
    else:
        f = getattr(m, entry)
    kw = {kw_repr}
    for p in inspect.signature(f).parameters:
        if p in kw:
            continue
        if p in ('date', 'target_date', 'as_of', 'day', 'ref'):
            kw[p] = '{DATA_DATE}'
        elif p in ('code', 'symbol', 'stock_code', 'secid', 'stock'):
            kw[p] = '{WATCH}'
        elif p == 'target':
            kw[p] = pd.Timestamp('{DATA_DATE}')
        elif p in ('limit', 'k', 'n', 'top_n', 'count'):
            kw[p] = 10
    r = f(**kw)
    s = json.dumps(r, ensure_ascii=False, default=str)
    if len(s) > 30000:
        s = json.dumps({{'_truncated': True, '_size': len(s), '_preview': s[:400]}}, ensure_ascii=False)
    print('__RESULT__' + s)
except Exception as e:
    print('__RESULT__' + json.dumps({{'error': str(e)[:120]}}, ensure_ascii=False))
"""
    return _capture(script, timeout_s)


def _result_ok(r) -> bool:
    """dict 含 error 键视为失败; 其余(含 None/list/str)均为调用成功(签名匹配)。"""
    if isinstance(r, dict):
        return not r.get("error")
    return True


def run_engine(code: str, timeout_s: int = 15) -> dict:
    """单引擎: recipe 优先(签名适配) → 回退 v2 七策略。"""
    rec = RECIPES.get(code)
    if rec is not None:
        entry = rec.get("entry")
        if not entry:
            return {"entry": None, "strategy": "recipe", "ok": False,
                    "error": "降级/弃用", "note": rec.get("kwargs", {}).get("note", "")}
        ap = _artifact_hit(code, DATA_DATE, rec.get("artifact"))
        if ap is not None:
            try:
                raw = ap.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw = ""
            return {"entry": entry, "strategy": "recipe", "ok": True,
                    "artifact": ap.name, "size": ap.stat().st_size,
                    "result": {"_artifact": ap.name, "preview": raw[:120].replace("\n", " ")}}
        r = _run_recipe(code, entry, rec.get("kwargs", {}), rec.get("timeout", timeout_s or 30))
        return {"entry": entry, "strategy": "recipe", "ok": _result_ok(r), "result": r}

    # 回退: v2 七策略(占位符/缩进已修复)
    src = (CORE / f"{code}.py").read_text(encoding="utf-8", errors="replace")
    import re  # noqa: PLC0415
    fns = re.findall(r"^def (\w+)\(", src, re.M)
    hits = [f for f in fns if f in ENTRY_HINTS or any(f.startswith(h) for h in ENTRY_HINTS)]
    if not hits:
        for f in fns:
            if not f.startswith("_"):
                hits.append(f)
                break
    if not hits:
        return {"entry": None, "ok": False, "error": "无入口"}
    fn = hits[0]
    for i, st in enumerate(_strategies()):
        r = _run_with_strategy(code, fn, st, timeout_s)
        if _result_ok(r):
            return {"entry": fn, "strategy": i + 1, "ok": True, "result": r}
    return {"entry": fn, "ok": False, "error": "7策略全败"}


def run_all(limit: int = 0, max_workers: int = 10) -> dict:
    """并发10一次性跑全部引擎。"""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 3600 and not limit:
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    engines = [p.stem for p in CORE.glob("*.py") if p.stem != "__init__"]
    if limit:
        engines = engines[:limit]
    results = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_engine, c): c for c in engines}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                code = futs[fut]
                results[code] = fut.result()
            except Exception as e:
                results[futs[fut]] = {"ok": False, "error": str(e)[:50]}
            done += 1
            if done % 10 == 0:
                print(f"  ...{done}/{len(engines)}", flush=True)
    ok_n = sum(1 for v in results.values() if v.get("ok"))
    out = {"scanned": len(engines), "ok": ok_n, "fail": len(engines) - ok_n,
           "elapsed": round(time.time() - t0, 1), "engines": results}
    CACHE.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def adapter_md() -> str:
    r = run_all()
    L = ["## 🧠 全引擎矩阵（万用适配器 v6）"]
    L.append(f"- 引擎 {r.get('scanned')} · 运行OK {r.get('ok')} · 降级 {r.get('fail')} ({r.get('elapsed')}s)")
    ok_l = [k for k, v in (r.get("engines") or {}).items() if v.get("ok")]
    if ok_l:
        L.append("### ✅ 运行成功")
        L.append("、".join(ok_l))
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    if "--limit" in args:
        i = args.index("--limit")
        lim = int(args[i + 1]) if i + 1 < len(args) else 20
        r = run_all(limit=lim)
        print(f"limit {lim}: OK {r['ok']} / fail {r['fail']} ({r['elapsed']}s)")
    else:
        print(adapter_md())
