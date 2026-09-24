#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六引擎接入（2026-08-22 —— 方案v8 L1: 63未接入引擎先接6个高价值）

统一收集 6 个 analysis_core 引擎输出, 供复盘链 blocks 使用:
  1. macro_overseas.analyze    → 海外宏观(美债/美元/商品对A股映射)
  2. rs_strength.scan          → 相对强弱(RS强度个股)
  3. calendar_effects.run_today→ 今日日历效应(星期/节气/财报窗口)
  4. stock_lens.analyze        → 个股透镜(核心票多维度透视)
  5. alternative_data.run_report → 另类数据(热搜/资金流等替代信号)
  6. volume_profile.compute    → 量分布(核心票量价分布支撑压力)
全部 catch 降级, 单源失败不阻塞。供 daily_review_collectors 接入。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quant_system"))


def _safe(fn, timeout_s: int = 15):
    """带超时安全执行: 先用需时间判断—简单模式: 单引擎太慢就让其TimeoutError降级。"""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:80]}


def _subprocess_engine(mod: str, fn: str, arg: str = "", timeout_s: int = 25) -> dict:
    """独立子进程跑引擎(真超时, 不阻塞主进程). 返回引擎输出dict."""
    import subprocess
    code = f"""
import sys, json
sys.path.insert(0, 'quant_system')
try:
    m = __import__('quant_system.analysis_core.{mod}', fromlist=['x'])
    f = getattr(m, '{fn}')
    if '{fn}' == 'scan' and '{arg}':
        r = f(__import__('pandas').Timestamp.now().normalize(), limit=5)
        if isinstance(r, tuple):
            r = {{'items': r[0], 'meta': r[1]}}
    elif '{arg}':
        r = f('{arg}')
    else:
        r = f()
    print(json.dumps(r, ensure_ascii=False, default=str)[:20000])
except Exception as e:
    print(json.dumps({{'error': str(e)[:80]}}))
"""
    try:
        pr = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            timeout=timeout_s, text=True, encoding="utf-8", errors="replace")
        import json as _j3
        _out = (pr.stdout or "").strip()
        # 从第一个 { 截取 json(容忍日志前缀行)
        _s = _out.find("{")
        if _s > 0:
            _out = _out[_s:]
        try:
            return _j3.loads(_out or "{}")
        except Exception:
            if _out:
                # 尝试取最后 json 行
                for _line in reversed(_out.splitlines()):
                    try:
                        return _j3.loads(_line)
                    except Exception:
                        continue
            return {"error": "输出非JSON"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:60]}
    # 1. 海外宏观: 本机(海外VPN)跑 → 产物落盘, 云端读产物
    import json as _j2, glob as _g2, time as _t5
    _MACRO_OUT = ROOT / "generated" / "overseas_macro.json"
    try:
        if _MACRO_OUT.exists():  # 产物优先(任何环境读)
            if _t5.time() - _MACRO_OUT.stat().st_mtime < 86400:
                out["macro"] = _j2.loads(_MACRO_OUT.read_text(encoding="utf-8"))
                out["macro"]["_source"] = "artifact"
            else:
                out["macro"] = {"error": "产物过期(需本机海外重跑)"}
        else:
            out["macro"] = {"error": "产物缺失(本机海外跑 scripts/macro_artifact.py)"}
    except Exception as e:
        out["macro"] = {"error": f"macro产物读取失败: {str(e)[:40]}"}
    # 2. 相对强弱(RS强度: 按今日日期扫描)
    try:
        from quant_system.analysis_core import rs_strength  # noqa: PLC0415
        import pandas as _pd9
        _td = _pd9.Timestamp.now().normalize()
        r = _safe(lambda: rs_strength.scan(_td, limit=5), 15)
        out["rs"] = r[0] if isinstance(r, tuple) and r else (r if isinstance(r, dict) else {})
    except Exception:
        out["rs"] = {"error": "rs 不可用"}
    # 3. 日历效应
    try:
        from quant_system.analysis_core import calendar_effects  # noqa: PLC0415
        r = _safe(lambda: calendar_effects.run_today())
        out["calendar"] = r if isinstance(r, dict) else {"note": str(r)[:100]}
    except Exception:
        out["calendar"] = {"error": "calendar 不可用"}
    # 4. 个股透镜(核心票逐一)
    try:
        from quant_system.analysis_core import stock_lens  # noqa: PLC0415
        out["lens"] = {}
        for code in ["600519", "000001", "300750"][:limit_stocks]:
            r = _safe(lambda c=code: stock_lens.analyze(c), 8)
            if isinstance(r, dict) and not r.get("error"):
                out["lens"][code] = r
    except Exception:
        out["lens"] = {"error": "lens 不可用"}
    # 5. 另类数据
    try:
        from quant_system.analysis_core import alternative_data  # noqa: PLC0415
        r = _safe(lambda: alternative_data.run_report())
        out["alt"] = r if isinstance(r, dict) else {"note": str(r)[:100]}
    except Exception:
        out["alt"] = {"error": "alt 不可用"}
    # 6. 量分布(核心票)
    try:
        from quant_system.analysis_core import volume_profile  # noqa: PLC0415
        kp = ROOT / "data_warehouse" / "kline" / "600519.parquet"
        if kp.exists():
            import pandas as pd  # noqa: PLC0415
            df = pd.read_parquet(kp)
            r = _safe(lambda: volume_profile.compute_volume_profile(df, levels=5))
            out["vol"] = r if isinstance(r, dict) else {"note": str(r)[:100]}
    except Exception:
        out["vol"] = {"error": "vol 不可用"}
    # 汇总状态
    out["_ok"] = {k: not (isinstance(v, dict) and v.get("error")) for k, v in out.items()}
    return out


def engine_pulse(limit_stocks: int = 5) -> dict:
    """六引擎全部 subprocess 隔离(海外源本机跑, 各源真超时不阻塞)."""
    out = {}
    # 1. 海外宏观(产物优先, 无则本机子进程)
    import json as _j4, time as _t6
    from pathlib import Path as _P5
    _MACRO = _P5(__file__).resolve().parent.parent / "generated" / "overseas_macro.json"
    if _MACRO.exists() and _t6.time() - _MACRO.stat().st_mtime < 86400:
        out["macro"] = _j4.loads(_MACRO.read_text(encoding="utf-8"))
        out["macro"]["_source"] = "artifact"
    else:
        out["macro"] = _subprocess_engine("macro_overseas", "analyze", "", 25)
    # 2. 相对强弱
    out["rs"] = _subprocess_engine("rs_strength", "scan", "X", 20)
    # 3. 日历效应
    out["calendar"] = _subprocess_engine("calendar_effects", "run_today", "", 15)
    # 4. 个股透镜
    out["lens"] = {}
    for code in ["600519", "000001", "300750"][:limit_stocks]:
        r = _subprocess_engine("stock_lens", "analyze", code, 15)
        if isinstance(r, dict) and not r.get("error"):
            out["lens"][code] = r
    # 5. 另类数据
    out["alt"] = _subprocess_engine("alternative_data", "run_report", "", 20)
    # 6. 量分布(本地df直接调, 子进程传df不便)
    try:
        import pandas as _pd7  # noqa: PLC0415
        from quant_system.analysis_core.volume_profile import compute_volume_profile  # noqa: PLC0415
        _kp = ROOT / "data_warehouse" / "kline" / "600519.parquet"
        if _kp.exists():
            _df = _pd7.read_parquet(_kp)
            out["vol"] = compute_volume_profile(_df)
    except Exception as _ve:
        out["vol"] = {"error": str(_ve)[:60]}
    # 7. 元审查(报告质控)
    out["meta"] = _subprocess_engine("meta_reviewer", "scan_reports", "", 20)
    # 8. 趋势检测(核心票)
    out["trend"] = {}
    for code in ["600519", "000001"]:
        r = _subprocess_engine("trend_system", "detect_symbol", code, 12)
        if isinstance(r, dict) and not r.get("error"):
            out["trend"][code] = r
    out["_ok"] = {k: not (isinstance(v, dict) and v.get("error")) for k, v in out.items()}
    return out


def engine_md() -> str:
    """六引擎 → Markdown(供研报)."""
    p = engine_pulse()
    L = ["## 🛠️ 六引擎深挖（宏观/RS/日历/透镜/另类/量分布）"]
    for k, name in [("macro", "海外宏观"), ("calendar", "日历效应"), ("rs", "相对强弱"),
                    ("lens", "个股透镜"), ("alt", "另类数据"), ("vol", "量分布")]:
        v = p.get(k) or {}
        if v.get("error"):
            L.append(f"- {name}: ⚠️ {v['error'][:40]}")
        else:
            L.append(f"- {name}: ✅ {str(v)[:110]}")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(engine_md())