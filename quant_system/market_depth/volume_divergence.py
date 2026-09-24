"""
volume_divergence — 量价背离检测 (V5)

识别关键的背离模式：
1. 缩量新高（量萎缩但价格创新高）= 上涨动能衰竭，即将回调
2. 放量下跌（量暴增但价格大跌）= 恐慌出逃，可能见底
3. 缩量回踩（量萎缩但价格未破前低）= 洗盘结束
4. 放量滞涨（量放大但价格不动）= 顶部出货
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

CST = timezone(timedelta(hours=8))


class VolumeDivergence:
    """量价背离检测。"""

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self, symbol: str = "sh000300") -> dict[str, Any]:
        """检测指定指数的量价背离。"""
        now = _time.time()
        cache_key = f"div_{symbol}"
        if now - self.last_fetch < self.cache_ttl and cache_key in self.cache:
            return self.cache[cache_key]

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "symbol": symbol,
            "patterns": [],
            "alerts": [],
            "score": 0,
        }

        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or len(df) < 120:
                return result

            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            dates = df["date"].values

            # 计算均量
            vol_ma20 = np.convolve(volume, np.ones(20) / 20, mode="valid")
            vol_ma60 = np.convolve(volume, np.ones(60) / 60, mode="valid")

            # 补齐长度
            pad_20 = np.full(19, np.nan)
            pad_60 = np.full(59, np.nan)
            vol_ma20_full = np.concatenate([pad_20, vol_ma20])
            vol_ma60_full = np.concatenate([pad_60, vol_ma60])

            # 最近 60 日分析
            recent_close = close[-60:]
            recent_vol = volume[-60:]
            recent_ma20 = vol_ma20_full[-60:]
            recent_ma60 = vol_ma60_full[-60:]
            recent_dates = dates[-60:]

            # 60日新高/新低
            max_60 = np.max(recent_close)
            min_60 = np.min(recent_close)
            current_close = recent_close[-1]
            current_vol = recent_vol[-1]
            current_vol_ma20 = recent_ma20[-1] if not np.isnan(recent_ma20[-1]) else np.mean(recent_vol)
            vol_ratio = current_vol / max(current_vol_ma20, 1)

            result["current_close"] = round(float(current_close), 2)
            # P2-Q22-fix(L254): 原 int(current_vol) 遇 NaN 抛 ValueError，被外层
            # except 吞掉导致整个分析失败。现先判 np.isnan，NaN 时输出 None 并降级继续。
            result["current_vol"] = int(current_vol) if not np.isnan(current_vol) else None
            result["vol_ratio_vs_ma20"] = round(float(vol_ratio), 2)

            # ── 模式检测 ──

            # Pattern 1: 缩量新高
            if current_close >= max_60 * 0.98:  # 接近60日新高
                prev_peak_vol = np.max(recent_vol[:-5])  # 前5天外的最大量
                vol_shrink = current_vol / max(prev_peak_vol, 1)
                if vol_shrink < 0.7:
                    result["patterns"].append({
                        "type": "缩量新高",
                        "detail": (f"价格接近60日新高({current_close:.0f})但量仅为前峰"
                                   f"{vol_shrink:.0%}"),
                        "severity": "warning",
                        "implication": "上涨动能减弱,警惕回调"
                    })
                    result["alerts"].append(
                        f"⚠️ 缩量新高: {symbol} — 量仅为前峰{vol_shrink:.0%}"
                    )
                    result["score"] -= 2

            # Pattern 2: 放量下跌
            ret_5d = (current_close / recent_close[-6] - 1) if len(recent_close) >= 6 else 0
            if ret_5d < -0.03 and vol_ratio > 1.5:
                result["patterns"].append({
                    "type": "放量下跌",
                    "detail": (f"5日跌{ret_5d*100:.1f}%且量放大{vol_ratio:.1f}x"),
                    "severity": "warning",
                    "implication": "恐慌性出逃,关注是否超跌"
                })
                result["alerts"].append(
                    f"⚠️ 放量下跌: {symbol} — 5日{ret_5d*100:.0f}%, 量{vol_ratio:.1f}x"
                )
                result["score"] -= 2

            # Pattern 3: 缩量回踩
            min_pct = (current_close / min_60 - 1) if min_60 > 0 else 1
            if -0.05 < min_pct < 0.03 and vol_ratio < 0.7:
                result["patterns"].append({
                    "type": "缩量回踩",
                    "detail": f"价格在60日最低点附近({min_pct*100:+.1f}%),量萎缩{vol_ratio:.0%}",
                    "severity": "info",
                    "implication": "抛压衰竭,关注反弹机会"
                })
                result["score"] += 2

            # Pattern 4: 放量滞涨
            ret_20d = (current_close / recent_close[-21] - 1) if len(recent_close) >= 21 else 0
            vol_5d_avg = np.mean(recent_vol[-5:])
            vol_20d_avg = np.mean(recent_vol[-20:])
            vol_5d_ratio = vol_5d_avg / max(vol_20d_avg, 1)

            if abs(ret_20d) < 0.02 and vol_5d_ratio > 1.3:
                result["patterns"].append({
                    "type": "放量滞涨",
                    "detail": f"20日几乎没涨({ret_20d*100:+.1f}%)但量放大{vol_5d_ratio:.1f}x",
                    "severity": "danger",
                    "implication": "顶部换手,警惕出货"
                })
                result["alerts"].append(
                    f"🔴 放量滞涨: {symbol} — 可能顶部出货"
                )
                result["score"] -= 3

            result["score"] = round(max(-10, min(10, result["score"])), 1)

        except Exception as e:
            result["error"] = str(e)

        self.cache[cache_key] = result
        self.last_fetch = now
        return result

    def scan_multi(self, symbols: list[str]) -> dict[str, Any]:
        """扫描多指数/板块的量价背离。"""
        results = {}
        total_alerts = []
        for sym in symbols:
            r = self.compute(sym)
            results[sym] = r
            total_alerts.extend(r.get("alerts", []))
        return {"symbols": results, "total_alerts": total_alerts}


def main() -> None:
    vd = VolumeDivergence()
    # 主要指数
    for sym in ["sh000001", "sh000300", "sz399001", "sz399006"]:
        r = vd.compute(sym)
        print(f"\n{'='*50}")
        print(f"  {sym} — 量价背离分析")
        print(f"{'='*50}")
        print(f"  当前价: {r.get('current_close', 'N/A')}  "
              f"量比MA20: {r.get('vol_ratio_vs_ma20', 'N/A')}x")
        if r.get("patterns"):
            for p in r["patterns"]:
                sev = {"warning": "⚠️", "danger": "🔴", "info": "ℹ️"}
                print(f"  {sev.get(p['severity'],'•')} {p['type']}: {p['detail']}")
                print(f"    → {p['implication']}")
        else:
            print("  ✅ 无明显量价背离")


if __name__ == "__main__":
    main()
