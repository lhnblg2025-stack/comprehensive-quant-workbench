#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""技术分析采集器（2026-08-21 新增 —— 用户点名"技术分析为什么没有"）

用 K 线数据直接计算实战技术维度：
- 指数技术状态: 上证/深成/创业板 —— 均线多头/量能/RSI/MACD/布林 → 技术总分
- 主线龙头个股技术位: MA5/MA10/MA20/MA60 排列, 支撑/压力, 量价配合, 趋势
- 形态信号: 读取 pattern_report（K线形态引擎产物）
输出: SignalBlock(technical) 接入全面复盘报告。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

logger = logging.getLogger("technical_collector")

from daily_review_collectors import SignalBlock, _safe_collect  # noqa: E402

# 指数映射: 显示名 → index_daily_{name}.parquet（腾讯云产出）
INDEX_FILES = {
    "上证指数": "上证指数", "沪深300": "沪深300", "中证500": "中证500", "创业板指": "创业板指",
    "科创50": "科创50", "上证50": "上证50", "中证1000": "中证1000", "红利指数": "红利指数",
}


def _load_kline(code: str, days: int = 120, is_index: bool = False) -> Optional[object]:
    """读本地 K线 parquet（指数走 index_daily_{name}，个股走 kline/{code}）。"""
    try:
        import pandas as pd
        if is_index:
            # 上证指数读独立文件 index_daily_上证指数.parquet（由 fetch_index_daily.py 落盘）。
            # index_daily.parquet 是沪深300的主文件同步，不能当上证指数用。
            p = ROOT / "data_warehouse" / "market" / f"index_daily_{code}.parquet"
        else:
            p = ROOT / "data_warehouse" / "kline" / f"{code}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days)
        return df
    except Exception as e:  # noqa: BLE001
        print(f"[technical] 读K线失败 {code}: {str(e)[:50]}")
        return None


def _ma(df, n: int) -> Optional[float]:
    try:
        c = df["close"].astype(float)
        return float(c.tail(n).mean()) if len(c) >= n else None
    except Exception:  # noqa: BLE001
        return None


def _rsi(df, n: int = 14) -> Optional[float]:
    try:
        c = df["close"].astype(float)
        d = c.diff()
        up = d.clip(lower=0).rolling(n).mean()
        down = (-d.clip(upper=0)).rolling(n).mean()
        rs = up / down.replace(0, 1e-9)
        return float((100 - 100 / (1 + rs)).iloc[-1]) if len(c) > n else None
    except Exception:  # noqa: BLE001
        return None


def _macd(df) -> Optional[dict]:
    try:
        c = df["close"].astype(float)
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        hist = (dif - dea) * 2
        return {"dif": round(float(dif.iloc[-1]), 3),
                "dea": round(float(dea.iloc[-1]), 3),
                "hist": round(float(hist.iloc[-1]), 3),
                "golden": dif.iloc[-1] > dea.iloc[-1]}
    except Exception:  # noqa: BLE001
        return None


def _volume_trend(df) -> Optional[str]:
    try:
        v = df["volume"].astype(float)
        v5 = v.tail(5).mean()
        v20 = v.tail(20).mean()
        if v5 > v20 * 1.3:
            return "放量"
        if v5 < v20 * 0.7:
            return "缩量"
        return "平量"
    except Exception:  # noqa: BLE001
        return None


def _support_resistance(df) -> dict:
    """近60日支撑/压力。"""
    try:
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        recent = df.tail(60)
        return {
            "support": round(float(recent["low"].min()), 2),
            "resistance": round(float(recent["high"].max()), 2),
        }
    except Exception:  # noqa: BLE001
        return {}


def _index_tech(code: str, name: str) -> Optional[dict]:
    """单指数技术状态，读取足够历史数据计算MA300。"""
    df = _load_kline(code, days=420, is_index=True)
    if df is None or len(df) < 30:
        return None
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    ma5, ma10, ma20, ma60, ma300 = (_ma(df, n) for n in (5, 10, 20, 60, 300))
    # 均线排列
    if all(v is not None for v in (ma5, ma10, ma20)):
        bullish = ma5 > ma10 > ma20
        bearish = ma5 < ma10 < ma20
        trend = "多头排列" if bullish else ("空头排列" if bearish else "均线缠绕")
    else:
        trend = "数据不足"
    # 价格相对均线
    above_ma20 = last > ma20 if ma20 else False
    above_ma300 = last > ma300 if ma300 else None
    # 评分 0-100；跌破MA300不是普通扣分，而是长期趋势否决信号。
    score = 50
    if bullish: score += 15
    if above_ma20: score += 10
    rsi = _rsi(df)
    macd = _macd(df)
    if rsi is not None:
        if rsi > 70: score -= 5  # 超买
        elif rsi < 30: score += 5  # 超卖反弹潜力
        elif rsi > 50: score += 5
    if macd and macd.get("golden"): score += 10
    sr = _support_resistance(df)
    return {
        "name": name, "code": code, "last": round(last, 2),
        "ma5": round(ma5, 2) if ma5 else None, "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None, "ma60": round(ma60, 2) if ma60 else None,
        "ma300": round(ma300, 2) if ma300 else None, "above_ma300": above_ma300,
        "trend": trend, "above_ma20": above_ma20, "rsi": round(rsi, 1) if rsi else None,
        "macd": macd, "volume": _volume_trend(df),
        "support": sr.get("support"), "resistance": sr.get("resistance"),
        "score": min(100, max(0, score)),
    }


def _stock_tech(code: str, name: str) -> Optional[dict]:
    """个股技术（主线龙头候选），读取个股 K 线而非指数文件。"""
    df = _load_kline(code, days=420, is_index=False)
    if df is None or len(df) < 30:
        return None
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    ma5, ma10, ma20, ma60, ma300 = (_ma(df, n) for n in (5, 10, 20, 60, 300))
    bullish = all(v is not None for v in (ma5, ma10, ma20)) and ma5 > ma10 > ma20
    bearish = all(v is not None for v in (ma5, ma10, ma20)) and ma5 < ma10 < ma20
    trend = "多头排列" if bullish else ("空头排列" if bearish else "均线缠绕")
    above_ma20 = last > ma20 if ma20 else False
    above_ma300 = last > ma300 if ma300 else None
    score = 50 + (15 if bullish else 0) + (10 if above_ma20 else 0)
    rsi = _rsi(df)
    macd = _macd(df)
    if rsi is not None:
        score += -5 if rsi > 70 else (5 if rsi < 30 or rsi > 50 else 0)
    if macd and macd.get("golden"):
        score += 10
    sr = _support_resistance(df)
    return {
        "name": name, "code": code, "last": round(last, 2),
        "ma5": round(ma5, 2) if ma5 else None, "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None, "ma60": round(ma60, 2) if ma60 else None,
        "ma300": round(ma300, 2) if ma300 else None, "above_ma300": above_ma300,
        "trend": trend, "above_ma20": above_ma20, "rsi": round(rsi, 1) if rsi else None,
        "macd": macd, "volume": _volume_trend(df),
        "support": sr.get("support"), "resistance": sr.get("resistance"),
        "score": min(100, max(0, score)),
    }


def _load_pattern_signals(as_of: str | None = None) -> list[dict]:
    """读形态报告；历史复盘只接受 exact-date 产物。"""
    try:
        import glob
        cands = sorted(glob.glob(str(ROOT / "generated" / "pattern_report_*.md")))
        if as_of:
            exact = ROOT / "generated" / f"pattern_report_{as_of}.md"
            cands = [str(exact)] if exact.exists() else []
        if not cands:
            return []
        md = Path(cands[-1]).read_text(encoding="utf-8")
        # 提取关键形态行（★/▲/风险/突破等标记）
        signals = []
        for line in md.splitlines():
            if any(k in line for k in ("★", "▲", "突破", "看涨", "看跌", "金叉", "死叉", "预警")):
                if len(line.strip()) > 4:
                    signals.append(line.strip()[:80])
            if len(signals) >= 6:
                break
        return signals
    except Exception as e:  # noqa: BLE001
        print(f"[technical] pattern读取失败: {str(e)[:50]}")
        return []


def collect_technical(gateway=None, leader_codes: list | None = None, as_of: str | None = None) -> SignalBlock:
    """技术分析采集：指数技术 + 主线龙头个股技术位 + exact-date形态信号。"""
    def _run() -> dict:
        out = {"indices": [], "leaders": [], "patterns": [], "confidence": 0.5}
        # 1) 三大指数技术状态
        for name, code in INDEX_FILES.items():
            try:
                t = _index_tech(code, name)
                if t:
                    out["indices"].append(t)
            except Exception as e:  # noqa: BLE001
                print(f"[technical] {name} 失败: {str(e)[:40]}")
        # 2) 主线龙头个股技术位（传 code 或默认核心样本）
        targets = leader_codes or [
            ("600519", "贵州茅台"), ("000001", "平安银行"), ("300750", "宁德时代"),
        ]
        for code, name in targets:
            try:
                t = _stock_tech(code, name)
                if t:
                    t["name"] = name
                    out["leaders"].append(t)
            except Exception:  # noqa: BLE001
                continue
        # 3) 形态信号
        out["patterns"] = _load_pattern_signals(as_of)
        out["confidence"] = 0.7 if out["indices"] else 0.2
        return out
    return _safe_collect(_run, "technical", "K线技术分析", {}, timeout_seconds=30)


def render_technical_md(block: SignalBlock) -> str:
    """技术分析 → Markdown。"""
    L = ["## 📈 技术分析（K线实战维度）"]
    if block.error or not block.value.get("indices"):
        L.append(f"- ⚠️ 技术分析降级: {block.error or '无K线数据'}")
        return "\n".join(L)
    L.append("### 指数技术状态")
    for t in block.value["indices"]:
        color = "🟢" if t["score"] >= 60 else ("🔴" if t["score"] <= 40 else "🟡")
        L.append(f"- **{t['name']}** {t['last']} {color} 技术分{t['score']} | {t['trend']}"
                 f" | MA5/10/20: {t['ma5']}/{t['ma10']}/{t['ma20']}"
                 f" | RSI {t['rsi']} | {t['volume']}"
                 f" | 支撑{t['support']} 压力{t['resistance']}")
        if t.get("macd"):
            L.append(f"  - MACD: DIF{t['macd']['dif']} DEA{t['macd']['dea']} "
                     f"{'金叉✓' if t['macd']['golden'] else '死叉✗'}")
    L.append("### 核心标的（个股技术位）")
    for t in block.value.get("leaders", [])[:5]:
        L.append(f"- **{t['name']}({t['code']})** {t['last']} | {t['trend']}"
                 f" | MA20 {t['ma20']} RSI {t['rsi']} {t['volume']}"
                 f" | 支撑{t['support']} 压力{t['resistance']}")
    pats = block.value.get("patterns", [])
    if pats:
        L.append("### 形态信号（pattern引擎）")
        for p in pats[:6]:
            L.append(f"- {p}")
    return "\n".join(L)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    b = collect_technical()
    print(render_technical_md(b))