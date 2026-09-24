"""
daily_snapshot.py — 每日快照宽表生成器（云服务器版）

方案 v1.3 D1/D2 落地：
- 每日一行宽表 daily_snapshot.parquet（追加模式）
- 字段：日期/上证涨跌/全A等权/北向净额/两融余额/涨停家数/炸板率/最高连板/
        宏观温度计/情绪温度计/核心因子Top5/异动标签
- 标签摘要：3-sigma 离群检测（60日 Z-score），模板化输出 3 句话复盘
- 只读数据源，无任何交易/下单逻辑（辅助工具定位）

部署：云服务器 C:\\quant\\scripts\\daily_snapshot.py
计划任务：quant_snapshot 每日 18:30
输出：C:\\quant\\out\\daily_snapshot\\  ← 由 OpenClaw cron 拉回虚拟机
"""
from __future__ import annotations
import logging

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_WS = Path(__file__).resolve().parent.parent
# 2026-08-22 稳定化: 默认仓库内路径(本机可直跑), 云端仍可环境变量覆盖
ROOT = Path(os.environ.get("QUANT_ROOT", str(_WS)))
sys.path.insert(0, str(ROOT))

try:
    from quant_platform.legacy import nice_process
    nice_process()
except Exception as e:
    logging.getLogger(__name__).error(f"[daily_snapshot] 操作失败: {e}", exc_info=True)

OUT_DIR = Path(os.environ.get("QUANT_DESKTOP", str(_WS / "generated" / "daily_snapshot"))) / "daily_snapshot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_FILE = OUT_DIR / "daily_snapshot.parquet"
SUMMARY_FILE = OUT_DIR / "summary.json"

# 核心因子 Top5（与日报口径一致）
CORE_FACTORS = ["momentum_20d", "volatility_20d", "turnover_20d",
                "valuation_pe_ttm", "quality_roe"]


def fetch_spot_breadth() -> dict:
    """全A等权涨跌幅 + 涨跌家数（多接口 fallback：spot_em → spot → spot_qq）。"""
    import time as _t
    interfaces = ["stock_zh_a_spot_em", "stock_zh_a_spot", "stock_zh_a_spot_qq"]
    for fn in interfaces:
        for attempt in range(2):
            try:
                import akshare as ak
                df = getattr(ak, fn)()
                if df is None or df.empty:
                    continue
                # 统一列名：spot_em 用"涨跌幅"，spot/spot_qq 用"涨跌幅"或"涨幅"
                ret_col = "涨跌幅" if "涨跌幅" in df.columns else ("涨幅" if "涨幅" in df.columns else None)
                name_col = "名称" if "名称" in df.columns else ("股票名称" if "股票名称" in df.columns else None)
                if ret_col is None or name_col is None:
                    continue
                df = df[~df[name_col].astype(str).str.contains("ST|退", na=False)]
                df = df[~df["代码"].astype(str).str.startswith(("4", "8", "9"))]  # 排除北交所
                ret = pd.to_numeric(df[ret_col], errors="coerce").dropna()
                if len(ret) == 0:
                    continue
                up = int((ret > 0).sum())
                down = int((ret < 0).sum())
                flat = int((ret == 0).sum())
                limit_up = int((ret >= 9.9).sum())
                limit_down = int((ret <= -9.9).sum())
                return {
                    "全A等权涨跌幅": round(float(ret.mean()), 2),
                    "上涨家数": up, "下跌家数": down, "平盘家数": flat,
                    "涨停家数": limit_up, "跌停家数": limit_down,
                    "行情接口": fn,
                }
            except Exception as e:
                print(f"[snapshot] spot {fn} try{attempt}: {type(e).__name__}: {str(e)[:80]}")
                if attempt < 1:
                    _t.sleep(2)
    return {}


def fetch_margin() -> dict:
    """沪深两融余额（汇总）。"""
    try:
        import akshare as ak
        sse = ak.stock_margin_sse()
        szse = ak.stock_margin_szse()
        total = 0.0
        if sse is not None and not sse.empty:
            total += float(sse.iloc[-1]["融资余额"]) / 1e8  # 元→亿
        if szse is not None and not szse.empty:
            total += float(szse.iloc[-1]["融资余额"]) / 1e8  # 元→亿
        return {"两融余额(亿)": round(total, 1)}
    except Exception as e:
        print(f"[snapshot] margin err: {type(e).__name__}: {str(e)[:120]}")
        return {}


def fetch_index() -> dict:
    """上证指数涨跌幅 + 沪深300/中证1000/微盘股风格。"""
    try:
        import akshare as ak
        out = {}
        for name, code in [("上证指数", "sh000001"), ("沪深300", "sh000300"),
                           ("中证1000", "sh000852")]:
            df = ak.stock_zh_index_daily(symbol=code)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                chg = round((last["close"] / prev["close"] - 1) * 100, 2)
                out[f"{name}涨跌幅"] = chg
        return out
    except Exception as e:
        print(f"[snapshot] index err: {type(e).__name__}: {str(e)[:120]}")
        return {}


def compute_labels(history: pd.DataFrame, today: dict) -> list[str]:
    """3-sigma 离群检测（60 日 Z-score），返回异动标签。"""
    labels = []
    if history is None or history.empty or len(history) < 20:
        return labels
    rules = [
        ("两融余额(亿)", "两融", 2.0, ["两融_大幅流入", "两融_大幅流出"]),
        ("全A等权涨跌幅", "全A等权", 2.0, ["全A_强势", "全A_弱势"]),
        ("涨停家数", "涨停家数", 2.0, ["涨停_过热", "涨停_冰点"]),
    ]
    for col, tag, z_thr, (up_lab, dn_lab) in rules:
        if col not in history.columns or col not in today:
            continue
        ser = pd.to_numeric(history[col], errors="coerce").dropna()
        if len(ser) < 20:
            continue
        mu, sd = ser.mean(), ser.std()
        if sd == 0 or pd.isna(sd):
            continue
        z = (today[col] - mu) / sd
        if z > z_thr:
            # V10 审计 M2 修复：up_lab 已含 tag 前缀时不再叠加（避免“两融_两融_大幅流入”）
            lab = up_lab if up_lab.startswith(f"{tag}_") else f"{tag}_{up_lab}"
            labels.append(lab)
        elif z < -z_thr:
            lab = dn_lab if dn_lab.startswith(f"{tag}_") else f"{tag}_{dn_lab}"
            labels.append(lab)
    return labels


def build_summary(date_str: str, today: dict, labels: list[str],
                  prev: dict | None) -> str:
    """模板化 3 句话摘要（事实变化，无幻觉归因）。"""
    lines = [f"{date_str} 快照摘要："]
    # 1. 市场状态
    eq = today.get("全A等权涨跌幅")
    if eq is not None:
        state = "上涨" if eq > 0 else ("下跌" if eq < 0 else "平盘")
        lines.append(f"全A等权 {eq:+.2f}%（{state}）")
    # 2. 量能与两融变化
    if prev and "两融余额(亿)" in prev and "两融余额(亿)" in today:
        d = today["两融余额(亿)"] - prev["两融余额(亿)"]
        lines.append(f"两融余额 {today['两融余额(亿)']:.0f}亿（较前日 {d:+.0f}亿）")
    # 3. 异动标签
    if labels:
        lines.append("异动：" + "；".join(labels))
    else:
        lines.append("无显著异动标签。")
    return "\n".join(lines)


def main() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"[snapshot] {today} start")

    row: dict = {"日期": today}
    row.update(fetch_index())
    row.update(fetch_spot_breadth())
    row.update(fetch_margin())

    # 读取历史（追加模式）
    history = None
    if SNAPSHOT_FILE.exists():
        try:
            history = pd.read_parquet(SNAPSHOT_FILE)
        except Exception as e:
            print(f"[snapshot] read hist err: {e}")
            history = None

    # 去重：同日覆盖
    if history is not None and not history.empty:
        history = history[history["日期"] != today]
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])

    history.to_parquet(SNAPSHOT_FILE, index=False)
    print(f"[snapshot] saved {len(history)} rows -> {SNAPSHOT_FILE}")

    # 标签 + 摘要
    prev_row = None
    if len(history) >= 2:
        prev_row = history.iloc[-2].to_dict()
    labels = compute_labels(history.iloc[:-1], row)
    summary = build_summary(today, row, labels, prev_row)
    SUMMARY_FILE.write_text(json.dumps({
        "日期": today, "标签": labels, "摘要": summary,
        "宽表行数": int(len(history)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[snapshot] summary: {summary}")
    print(f"[snapshot] done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
