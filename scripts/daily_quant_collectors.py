#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日复盘融合链 · 量化分析层采集器（2026-08-21 v3 深化）

把因子研究层/回测/ML 融合进复盘链：
- 因子强度: ic_spill 因子面板 → 近5日截面强度 → 技术/价值/质量/量价风格占优
- 因子短板: 连续走弱因子（动量衰减/拥挤）
- 机会池:   opportunity 扫描（实时，超时降级）或读云端产物
- ML 预测:  ml_signals（对核心标的预测，超时降级）
- 回测验证: 对选股池代表标的做短期持有收益回测（用 ic_spill 历史面板）

每个采集器返回 SignalBlock（与复盘链一致）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "quant_system"))

# QV6 因子层需要可写路径（默认 /root/quant 只读环境）
for _k, _v in (("QV6_LOG_FILE", "generated/qv6.log"),
               ("QV6_STATE_DIR", "generated/qv6_state"),
               ("QV6_DB", "generated/qv6.db"),
               ("QUANT_LOG_DIR", "generated")):
    if not os.environ.get(_k):
        os.environ[_k] = str(ROOT / _v)

logger = logging.getLogger("daily_quant_collectors")

from daily_review_collectors import SignalBlock, _safe_collect  # noqa: E402

# 核心因子名单（技术动量/价值/质量/量价 四组）
FACTOR_GROUPS = {
    "技术动量": ["tech_mom_10", "tech_mom_20", "tech_mom_60", "ma_cross", "tech_ma_bull", "tech_rsi14"],
    "价值": ["val_pb_pct", "val_pe_ttm_pct", "val_ps_pct", "val_pe_pb_ratio"],
    "质量": ["fa_roe_adj", "fa_net_margin", "fa_retained_eps", "turnover_stability"],
    "量价": ["tech_vol_ratio", "tech_breakout_20", "volume_surge", "tech_vol_contract"],
    "波动": ["realized_vol", "vol_20d", "downside_vol", "tech_sharpe_60"],
}


_WANTED = {fn for fl in FACTOR_GROUPS.values() for fn in fl}


def _load_factor_panels() -> dict:
    """只读 FACTOR_GROUPS 涉及的因子面板（80个全部读会 4GB 内存/超时）。"""
    import numpy as np
    d = ROOT / "generated" / "ic_spill"
    if not d.exists():
        return {}
    panels = {}
    for name in sorted(_WANTED):
        f = d / f"{name}.npy"
        if not f.exists():
            continue
        try:
            arr = np.load(f, allow_pickle=True, mmap_mode="r")
            if arr.ndim == 2:
                panels[name] = arr
        except Exception:  # noqa: BLE001
            continue
    return panels


def _panel_codes_dates() -> tuple[list, list]:
    d = ROOT / "generated" / "ic_spill"
    import numpy as np
    codes_p = d / "codes.npy"
    dates_p = d / "dates.npy"
    codes = np.load(codes_p, allow_pickle=True).tolist() if codes_p.exists() else []
    dates = np.load(dates_p, allow_pickle=True).tolist() if dates_p.exists() else []
    return codes, dates


# ═══════════════════════════════════════════════════════════
# 1. 因子强度维度
# ═══════════════════════════════════════════════════════════
def collect_factor_signal(gateway, date: str | None = None) -> SignalBlock:
    """因子强度：近5日截面 z-score 均值的组内均值 → 风格占优/走弱。"""
    def _run() -> dict:
        out: dict = {"groups": {}, "top_factors": [], "weak_factors": []}
        import numpy as np
        panels = _load_factor_panels()
        if not panels:
            out["error_hint"] = "ic_spill 面板缺失（需跑 ic_vectorized 落盘）"
            return out
        codes, dates = _panel_codes_dates()
        # 面板 shape=(日期数, 股票数)；dates=日期列表(行)
        nd = len(panels[next(iter(panels))]) if panels else 0
        start = max(0, nd - 5)  # 最近5个交易日
        group_scores = {}
        for gname, flist in FACTOR_GROUPS.items():
            vals = []
            for fn in flist:
                arr = panels.get(fn)
                if arr is None:
                    continue
                rows = arr[start:nd, :]  # (5, 股票数)
                if rows.shape[0] == 0:
                    continue
                # 因子强度 = 每日截面高分位占比（>60分位的股票占比），再取5日均
                # 比 z-score 组均更直观：某风格因子强 ↔ 多数股票处高分位
                try:
                    q60 = np.nanpercentile(rows, 60, axis=1, keepdims=True)
                    high_ratio = float(np.nanmean(rows > q60))
                    vals.append((fn, high_ratio))
                except Exception:  # noqa: BLE001
                    continue
            if vals:
                vals = [(f, v) for f, v in vals if v == v]  # 去 nan
                if vals:
                    avg = sum(v for _, v in vals) / len(vals)
                    group_scores[gname] = {"score": round(avg, 3), "n": len(vals)}
        out["groups"] = group_scores
        # top/weak 因子
        all_f = []
        for gname, flist in FACTOR_GROUPS.items():
            for fn in flist:
                arr = panels.get(fn)
                if arr is None:
                    continue
                v = float(np.nanmean(arr[start:nd, :]))
                if v == v:
                    all_f.append((fn, v))
        if all_f:
            all_f.sort(key=lambda x: -x[1])
            out["top_factors"] = [f for f, _ in all_f[:5]]
            out["weak_factors"] = [f for f, _ in all_f[-5:]]
        _d = date
        if not _d and dates:
            _raw = dates[-1]
            # 支持 int 秒/纳秒时间戳 与 ISO 字符串
            if isinstance(_raw, (int, float)):
                _ts = _raw / 1e9 if _raw > 1e12 else _raw
                from datetime import datetime, timezone
                _d = datetime.fromtimestamp(_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                _s = str(_raw)
                _d = _s[:10] if len(_s) >= 10 else _s
        out["date"] = _d or ""
        out["confidence"] = 0.6 if group_scores else 0.2
        return out
    return _safe_collect(_run, "factor_signal", "ic_spill因子面板", {}, timeout_seconds=20)


# ═══════════════════════════════════════════════════════════
# 2. 机会扫描（读产物优先，实时降级）
# ═══════════════════════════════════════════════════════════
def collect_opportunity(gateway, date: str | None = None) -> SignalBlock:
    """机会池：读生成的 opportunity 产物，无则实时扫描（限时）。"""
    def _run() -> dict:
        out: dict = {"opportunities": []}
        # 1) 读产物优先（云端 opp_* cron 生成）
        import glob
        cands = sorted(glob.glob(str(ROOT / "generated" / "opportunity_*.json")))
        if cands:
            try:
                ops = json.loads(Path(cands[-1]).read_text(encoding="utf-8"))
                lst = ops.get("opportunities") or ops.get("data") or (ops if isinstance(ops, list) else [])
                out["opportunities"] = (lst if isinstance(lst, list) else [])[:5]
                out["_source"] = "artifact"
                out["confidence"] = 0.6
                return out
            except Exception as e:  # noqa: BLE001
                # 产物存在但损坏：不得谎称"无产物"，否则下游把"损坏"当"今日无机会"→假数据
                logger.warning(f"[collect_opportunity] opportunity 产物损坏 {Path(cands[-1]).name}: {type(e).__name__}: {e}")
                out["error_hint"] = f"opportunity 产物损坏: {type(e).__name__}: {str(e)[:100]}"
                out["confidence"] = 0.2
                return out
        # 2) 无产物 → 快速降级标注（实时扫描30s+太重，云端 opp_* cron 专门生成；
        #    避免复盘链被拖慢）
        out["error_hint"] = "无 opportunity 产物(云端 opp_* cron 生成后自动接入)"
        out["confidence"] = 0.2
        return out
    return _safe_collect(_run, "opportunity", "opportunity扫描", {}, timeout_seconds=25)


# ═══════════════════════════════════════════════════════════
# 3. ML 预测（对核心标的）
# ═══════════════════════════════════════════════════════════
def collect_ml_verdict(gateway, symbols: list[str] | None = None) -> SignalBlock:
    """ML 模型预测：对核心标的 predict_ensemble（短中预测），限时降级。"""
    def _run() -> dict:
        out: dict = {"predictions": []}
        targets = (symbols or ["600519", "000001"])[:3]
        try:
            from quant_system import ml_signals  # noqa: PLC0415
            preds = []
            for sym in targets:
                try:
                    r = ml_signals.predict_ensemble(sym, horizon_days=5)
                    if r:
                        preds.append({"symbol": sym, "pred": r if isinstance(r, dict) else str(r)[:60]})
                except Exception:  # noqa: BLE001
                    continue
            out["predictions"] = preds
            out["confidence"] = 0.5 if preds else 0.2
        except Exception as e:  # noqa: BLE001
            out["error_hint"] = "ml_signals 未就绪(需模型文件)"
            out["confidence"] = 0.2
        return out
    return _safe_collect(_run, "ml_verdict", "ml_signals", {}, timeout_seconds=25)


# ═══════════════════════════════════════════════════════════
# 4. 回测验证（对选股池代表标的近20日持有收益）
# ═══════════════════════════════════════════════════════════
def collect_backtest_check(gateway, symbols: list[str] | None = None) -> SignalBlock:
    """回测验证：对代表标的用 ic_spill 面板算近20日收益/波动（轻量，不跑完整回测引擎）。"""
    def _run() -> dict:
        out: dict = {"checks": []}
        targets = (symbols or ["600519", "000001"])[:3]
        try:
            import numpy as np
            panels = _load_factor_panels()
            codes, dates = _panel_codes_dates()
            if not codes or not panels:
                out["error_hint"] = "面板缺失"
                return out
            # 用 vol_20d 或 realied_vol 作为风险代理；用 tech_mom_20 看动量
            mom = panels.get("tech_mom_20")
            vol = panels.get("vol_20d")
            for sym in targets:
                try:
                    idx = codes.index(str(sym)) if str(sym) in codes else None
                    if idx is None:
                        continue
                    m = float(mom[idx, -1]) if mom is not None else None
                    v = float(vol[idx, -1]) if vol is not None else None
                    out["checks"].append({"symbol": sym, "mom20": round(m, 3) if m else None,
                                          "vol20": round(v, 3) if v else None})
                except Exception:  # noqa: BLE001
                    continue
            out["confidence"] = 0.5 if out["checks"] else 0.2
        except Exception as e:  # noqa: BLE001
            out["error_hint"] = str(e)[:80]
            out["confidence"] = 0.2
        return out
    return _safe_collect(_run, "backtest_check", "ic_spill回测", {}, timeout_seconds=15)


def collect_quant_all(gateway, symbols: list[str] | None = None) -> dict[str, SignalBlock]:
    """量化层四维度采集。"""
    return {
        "factor_signal": collect_factor_signal(gateway),
        "opportunity": collect_opportunity(gateway),
        "ml_verdict": collect_ml_verdict(gateway, symbols),
        "backtest_check": collect_backtest_check(gateway, symbols),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from daily_fusion_base import get_gateway
    gw = get_gateway()
    blocks = collect_quant_all(gw)
    for k, b in blocks.items():
        err = b.error or (b.value.get("error_hint") if isinstance(b.value, dict) else "")
        print(f"  {k:18s} conf={b.confidence:.2f} {err or 'OK'}")