"""
margin_deep — 两融深度分析 (V5)

比简单"两融余额"更深入：
  1. 两融余额的分板块拆解（上交所/深交所）
  2. 各板块融资净买入/偿还
  3. 两融余额历史分位（真实历史序列）
  4. 风险预警（杠杆过热/冰点）

数据源: akshare.stock_margin_sse(带end_date) + stock_margin_szse(带date)

Q23-fix（2026-08-02，V6 P0 修复 B 组）：
  - CRITICAL: 原调 ak.stock_margin_sh()/stock_margin_sz() 在 akshare 1.18.64
    不存在（AttributeError 被 except: pass 吞掉）→ 全零数据被渲染成
    "🟢 两融分位0% — 杠杆冰点"底部买入信号。现改调 stock_margin_sse/szse
    并适配字段；数据缺失时置 error，禁止输出"冰点/过热"预警。
  - MEDIUM: "历史分位"原为 1万~2.5万亿线性插值假分位 → 改真实历史序列分位，
    样本不足时返回 None 而非假值；segment 标注"主板/中小板/创业板/科创板"
    实为上交所/深交所合计 → 改名为 上交所/深交所；方向仅用融资余额变化
    → 补融券口径（用融资融券余额变动）。
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


def _percentile_of_series(value: float | None,
                          series: list[float]) -> float | None:
    """真实历史分位：value 在 series 中的百分位（0~100）。

    样本不足（<20）或 value 缺失时返回 None，由调用方决定不下结论。
    """
    if value is None or not series or len(series) < 20:
        return None
    arr = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    pct = float((arr <= value).mean()) * 100
    return round(max(0.0, min(100.0, pct)), 1)


class MarginDeep:
    """两融深度分析。"""

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """计算两融深度指标。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "total_margin": None,
            "margin_change": None,
            "margin_percentile": None,
            "segment": {},
            "top_sectors": [],
            "risk_warnings": [],
            "direction": "数据缺失",
            "ok": False,
        }

        sh_history: list[float] = []   # SSE 融资融券余额历史序列（真实分位用）
        sh_last: float | None = None
        sh_prev: float | None = None
        sh_short: float | None = None
        sz_last: float | None = None
        sz_prev: float | None = None
        sz_short: float | None = None
        errors: list[str] = []

        def _norm(vals) -> list[float]:
            """数值列统一换算为元（SSE=元；SZSE 汇总=亿元，按量级探测）。"""
            out: list[float] = []
            for v in vals:
                try:
                    out.append(float(v))
                except (ValueError, TypeError):
                    continue
            if out and max(abs(x) for x in out) < 1e9:
                out = [x * 1e8 for x in out]
            return out

        try:
            import akshare as ak
            from datetime import date as _date

            # ── 上交所两融（历史序列 → 真实分位）──
            try:
                end = _date.today().strftime("%Y%m%d")
                start = (_date.today() - timedelta(days=400)).strftime("%Y%m%d")
                sh = ak.stock_margin_sse(start_date=start, end_date=end)
                if sh is not None and not sh.empty:
                    # 列：信用交易日期/融资余额/融资买入额/融券余量/融券余量金额/
                    #     融券卖出量/融资融券余额
                    balance_col = next((c for c in sh.columns
                                        if "融资融券余额" in str(c)), None)
                    if balance_col is None:
                        balance_col = next((c for c in sh.columns
                                            if "融资余额" in str(c)), None)
                    if balance_col:
                        # SSE 返回按日期降序（最新在前）→ 反转后取最新/前值
                        vals = _norm(sh[balance_col].dropna().values)
                        vals = list(reversed(vals))  # 升序
                        sh_history = vals
                        if sh_history:
                            sh_last = float(sh_history[-1])
                            sh_prev = float(sh_history[-2]) if len(sh_history) >= 2 else None
                    short_col = next((c for c in sh.columns
                                      if "融券余量金额" in str(c)), None)
                    if short_col and len(sh) > 0:
                        try:
                            sh_short = float(_norm(sh[short_col].dropna().values)[0])
                        except Exception:
                            sh_short = None
                    result["segment"]["上交所"] = {
                        "balance": round(sh_last, 1) if sh_last else None,
                        "short_balance": round(sh_short, 1) if sh_short else None,
                        "change": (round(sh_last - sh_prev, 1)
                                   if sh_last is not None and sh_prev is not None else None),
                        "source": "stock_margin_sse",
                    }
            except Exception as e:
                errors.append(f"上交所两融获取失败: {repr(e)[:80]}")

            # ── 深交所两融（当日；单行汇总，单位亿元→元）──
            try:
                from datetime import date as _date

                def _sz_on(d: str):
                    try:
                        sz = ak.stock_margin_szse(date=d)
                        if sz is None or sz.empty:
                            return None, None, None
                        bal_col = next((c for c in sz.columns
                                        if "融资融券余额" in str(c)), None)
                        if bal_col is None:
                            bal_col = next((c for c in sz.columns
                                            if "融资余额" in str(c)), None)
                        short_col = next((c for c in sz.columns if "融券余额" in str(c)), None)
                        bal = None
                        short = None
                        if bal_col:
                            bv = _norm(sz[bal_col].dropna().values)
                            if bv:
                                bal = float(bv[0])
                        if short_col:
                            sv = _norm(sz[short_col].dropna().values)
                            if sv:
                                short = float(sv[0])
                        return bal, short, d
                    except Exception:
                        return None, None, None

                bal, short, d0 = _sz_on(_date.today().strftime("%Y%m%d"))
                if bal is None:
                    for back in range(1, 8):
                        dd = (_date.today() - timedelta(days=back)).strftime("%Y%m%d")
                        bal, short, d0 = _sz_on(dd)
                        if bal is not None:
                            break
                if bal is not None:
                    sz_last = bal
                    sz_short = short
                    # 上一交易日 prev
                    for back in range(1, 9):
                        dd = (pd.to_datetime(d0).date() - timedelta(days=back)).strftime("%Y%m%d")
                        pbal, _, _ = _sz_on(dd)
                        if pbal is not None:
                            sz_prev = pbal
                            break
                    result["segment"]["深交所"] = {
                        "balance": round(sz_last, 1),
                        "short_balance": round(sz_short, 1) if sz_short else None,
                        "change": (round(sz_last - sz_prev, 1)
                                   if sz_prev is not None else None),
                        "source": "stock_margin_szse",
                        "data_date": d0,
                    }
            except Exception as e:
                errors.append(f"深交所两融获取失败: {repr(e)[:80]}")

            # ── 总量与变动（融资融券余额口径，含融券）──
            if sh_last is not None or sz_last is not None:
                total = (sh_last or 0.0) + (sz_last or 0.0)
                total_prev = (sh_prev or 0.0) + (sz_prev or 0.0)
                result["total_margin"] = round(total, 1)
                if total_prev > 0:
                    result["margin_change"] = round(total - total_prev, 1)
                result["ok"] = True

                # ── 真实历史分位（基于上交所历史序列）──
                if sh_last is not None and sh_history:
                    # 用上交所融资融券余额历史序列的分位近似全市场
                    pct = _percentile_of_series(sh_last, sh_history)
                    result["margin_percentile"] = pct

                # ── 方向判断（融资融券余额变动，含融券口径）──
                change = result["margin_change"]
                if change is not None:
                    if change > 50:
                        result["direction"] = "加杠杆"
                    elif change < -50:
                        result["direction"] = "去杠杆"
                    else:
                        result["direction"] = "中性"

                # ── 风险预警（仅在数据真实且分位有效时输出，Q23-fix：
                #    数据缺失/分位无效时禁止输出"冰点/过热"）──
                m_pct = result["margin_percentile"]
                if m_pct is not None and result["ok"]:
                    if m_pct > 90:
                        result["risk_warnings"].append(
                            f"🔴 两融分位{m_pct:.0f}% — 杠杆过热,警惕回调风险"
                        )
                    elif m_pct > 80:
                        result["risk_warnings"].append(
                            f"⚠️ 两融分位{m_pct:.0f}% — 杠杆偏高"
                        )
                    elif m_pct < 10:
                        result["risk_warnings"].append(
                            f"🟢 两融分位{m_pct:.0f}% — 杠杆冰点,可能为底部区域"
                        )
                elif result["ok"]:
                    result["risk_warnings"].append(
                        "ℹ️ 两融历史分位样本不足,暂不下杠杆位置结论"
                    )
            else:
                errors.append("沪/深两融数据均缺失")

        except Exception as e:
            errors.append(repr(e)[:200])

        if errors:
            result["error"] = "; ".join(errors)
        result["data_errors"] = errors

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    md = MarginDeep()
    r = md.compute()
    print("═" * 55)
    print("  两融深度分析")
    print("═" * 55)
    total = r.get("total_margin")
    print(f"  两融余额: {total if total is not None else 'N/A'}")
    print(f"  当日变动: {r.get('margin_change', 'N/A')}")
    pct = r.get("margin_percentile")
    print(f"  历史分位: {pct if pct is not None else '样本不足/N/A'}%")
    print(f"  方向: {r.get('direction', 'N/A')}")
    print()
    for name, seg in r.get("segment", {}).items():
        bal = seg.get("balance")
        print(f"  {name}: 余额{bal if bal is not None else 'N/A'}, "
              f"变化{seg.get('change', 'N/A')}")
    for w in r.get("risk_warnings", []):
        print(f"  {w}")
    if r.get("error"):
        print(f"  ⚠️ {r['error']}")


if __name__ == "__main__":
    main()
