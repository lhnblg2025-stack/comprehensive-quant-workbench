"""
hedge_monitor — 对冲监控（IBKR/外部持仓框架）

监控用户对冲需求: V11 决策层给出 A 股风险信号时, 若用户持有外部市场
(美股/港股/商品) 多头仓位, 提示需要对冲; 反向(外部大跌)提示对 A 股情绪影响。
**框架 + 本地数据可算部分, IBKR 凭证缺失时降级为纯提示模式, 不空壳。**

定位:
  - 本机无 IBKR 凭证 → ib_sync 读环境变量 IBKR_ACCOUNT/IBKR_TOKEN/IBKR_HOST,
    未配置返回 {"available": False, "reason": "未配置 IBKR 凭证"}; 配置了但未装
    ib-insync → 提示 pip install ib-insync; 绝不硬编码假仓位。
  - 核心价值在对冲逻辑本身(本地数据可算): A股风险信号 × 外部持仓暴露 → 对冲建议。

数据(本地优先, 缺失标注 None + 提示, 不编造):
  - 美元/人民币: data_warehouse/market/rates__fx.parquet(已确认存在, 央行中间价/100)
  - 南向资金: data_warehouse/market/southbound_flow.parquet(若存在) 否则经 akshare
    stock_hsgt_fund_flow_summary_em(港股通沪+深合计); 均失败 → None + 数据缺失提示
  - 纳指/标普: 本地无 → None(数据缺失提示)
  - A股自身风险: 复用 generated/battle_map_{date}.json 与 self_heal_{date}.json;
    无产出时回退调用 battle_map.build_map()/self_healer.check()

用法:
  python3 -m quant_system.analysis_core.hedge_monitor [--date 2026-08-10]
"""

from __future__ import annotations
import logging

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.analysis_core.common import today  # noqa: E402

CST = timezone(timedelta(hours=8))
GEN_DIR = ROOT / "generated"

FX_PARQUET = MARKET_DIR / "rates__fx.parquet"
SOUTHBOUND_PARQUET = MARKET_DIR / "southbound_flow.parquet"

RISK_LEVELS = ("高", "中", "低")

# 对冲工具建议(按风险等级; 均为跨市场/衍生品工具, 不涉及个股推荐)
HEDGE_INSTRUMENTS = {
    "高": [
        "股指期货空头: IF(沪深300)/IM(中证1000)/IC(中证500)",
        "期权: 买沪深300/中证500 认沽(PUT)",
        "纳指空头ETF: SQQQ(三倍做空纳指) 或 恒指/纳指期货空单",
    ],
    "中": [
        "观察仓: 少量买 PUT 或纳指空头ETF(SQQQ)",
        "预留股指期货空头权限, 风险升级时立即启用",
    ],
}


# ── 本地数据读取 ──────────────────────────────────────────
def _load_fx(date: str | None = None) -> dict | None:
    """美元/人民币汇率: 本地 rates__fx.parquet(央行中间价, 每100外币人民币)。"""
    if not FX_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(FX_PARQUET)
        df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
        if date:
            df = df[df["日期"] <= date]
        df = df.sort_values("日期").drop_duplicates("日期")
        if df.empty:
            return None
        last = df.iloc[-1]
        raw = pd.to_numeric(last.get("央行中间价", pd.NA), errors="coerce")
        if pd.isna(raw):
            raw = pd.to_numeric(last.get("中行折算价", pd.NA), errors="coerce")
        if pd.isna(raw):
            return None
        usdcny = round(float(raw) / 100.0, 4)  # 原始单位: 每100外币人民币
        as_of = str(last["日期"])[:10]
        # 近5个交易日变动(USD/CNY 上升 = 人民币贬值)
        trend = None
        if len(df) >= 6:
            seq = df.tail(6)["日期"]
            vals = []
            for d in seq:
                row = df[df["日期"] == d].iloc[0]
                v = pd.to_numeric(row.get("央行中间价", pd.NA), errors="coerce")
                if pd.isna(v):
                    v = pd.to_numeric(row.get("中行折算价", pd.NA), errors="coerce")
                vals.append(v if not pd.isna(v) else None)
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                trend = round(float(vals[-1] - vals[0]) / 100.0, 4)
        note = None
        # 优先读 update_fx.py 写的新鲜度标注（rates__fx_meta.json），不假装新鲜
        meta = MARKET_DIR / "rates__fx_meta.json"
        try:
            if meta.exists():
                import json as _json
                m = _json.loads(meta.read_text(encoding="utf-8"))
                if m.get("stale") and m.get("as_of"):
                    note = (f"本地汇率数据截至 {m['as_of']}（{m.get('stale_days', '?')}天未更新），"
                            f"已标记 stale。联网运行 scripts/update_fx.py 可滚动补缺。")
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[hedge_monitor] 操作失败: {e}", exc_info=True)
        if note is None and as_of < (date or today())[:10] and (datetime.strptime(as_of, "%Y-%m-%d") <
                                                                 datetime.strptime(date or today(), "%Y-%m-%d") - timedelta(days=30)):
            note = f"本地汇率数据截至 {as_of}, 超过30天未刷新, 建议更新 rates__fx.parquet"
        return {
            "value": usdcny,
            "as_of": as_of,
            "trend_5d": trend,
            "unit": "USD/CNY(由央行中间价/100 换算)",
            "note": note,
        }
    except Exception as e:
        return None


def _load_southbound(date: str | None = None) -> dict | None:
    """南向资金(港股通净买入, 亿元): 本地 parquet 优先, 否则 akshare; 失败返回 None。"""
    if SOUTHBOUND_PARQUET.exists():
        try:
            df = pd.read_parquet(SOUTHBOUND_PARQUET)
            col = "date" if "date" in df.columns else ("日期" if "日期" in df.columns else None)
            val_col = "南向净流入" if "南向净流入" in df.columns else ("成交净买额" if "成交净买额" in df.columns else None)
            if col is None or val_col is None or df.empty:
                return None
            df = df.sort_values(col)
            if date:
                df = df[df[col].astype(str) <= date]
            if df.empty:
                return None
            last = df.iloc[-1]
            v = pd.to_numeric(last[val_col], errors="coerce")
            if pd.isna(v):
                return None
            return {
                "as_of": str(last[col])[:10],
                "net_buy_yi": round(float(v), 2),
                "source": "local_parquet",
                "note": "港股通(沪)+港股通(深)合计, 单位亿元",
            }
        except Exception:
            return None
    # 本地无 parquet → akshare(南向净买额 2024-08-19 后仍由交易所披露)
    try:
        import akshare as ak

        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty or "板块" not in df.columns:
            return None
        south = df[df["板块"].astype(str).str.contains("港股通", na=False)].copy()
        if south.empty or "成交净买额" not in south.columns:
            return None
        total = pd.to_numeric(south["成交净买额"], errors="coerce").dropna().sum()
        trade_date = str(south["交易日"].iloc[-1])[:10] if "交易日" in south.columns else ""
        if not trade_date or pd.isna(total):
            return None
        return {
            "as_of": trade_date,
            "net_buy_yi": round(float(total), 2),
            "source": "akshare",
            "note": "港股通(沪)+港股通(深)合计, 单位亿元",
        }
    except Exception:
        return None


def _ashare_risk(date: str | None = None) -> dict:
    """A股自身风险: 复用 battle_map / self_healer 输出(generated JSON 优先, 无则实算)。"""
    date = date or today()
    bm = None
    bm_err = ""
    try:
        bm_path = GEN_DIR / f"battle_map_{date}.json"
        if bm_path.exists():
            bm = json.loads(bm_path.read_text(encoding="utf-8"))
    except Exception as e:
        bm = None
    if bm is None:
        try:
            from quant_system.analysis_core import battle_map
            bm = battle_map.build_map(date)
        except Exception as e:
            bm = None
            bm_err = str(e)[:120]
    sh = None
    try:
        sh_path = GEN_DIR / f"self_heal_{date}.json"
        if sh_path.exists():
            sh = json.loads(sh_path.read_text(encoding="utf-8"))
    except Exception:
        sh = None

    level, evidence = "低", "battle_map 无明显风险信号"
    if sh and sh.get("decision_mode") == "conservative":
        level, evidence = "高", "系统自愈标记 conservative(只输出观察)"
    if isinstance(bm, dict) and bm:
        if bm.get("macro_veto") == "hard":
            level, evidence = "高", f"宏观硬否决"
        elif bm.get("recommended") == "防守":
            level, evidence = "高", f"作战地图建议{bm.get('recommended')}"
        elif bm.get("emotion_stage") in ("冰点", "退潮"):
            level, evidence = "高", f"情绪阶段:{bm.get('emotion_stage')}"
        elif bm.get("macro_veto") == "soft":
            level, evidence = "中", "宏观软否决"
        elif bm.get("recommended") == "试错":
            level, evidence = "中", f"作战地图建议{bm.get('recommended')}"
        elif (bm.get("confidence") or 0) < 0.5:
            level, evidence = "中", f"置信度{bm.get('confidence')}偏低"
        else:
            level, evidence = "低", (f"情绪{bm.get('emotion_stage')}/建议{bm.get('recommended')}"
                                     f"/置信度{bm.get('confidence')}")
    elif bm is None:
        level, evidence = "中", f"battle_map 不可用({bm_err or '无产出'}), 默认中等风险"

    return {
        "level": level,
        "source": "battle_map/self_healer",
        "evidence": evidence,
        "decision_mode": sh.get("decision_mode") if isinstance(sh, dict) else None,
        "battle_map_date": bm.get("date") if isinstance(bm, dict) else None,
    }


# ── 对外接口 ──────────────────────────────────────────────
def external_risk_signals(date: str | None = None) -> dict:
    """外部市场风险信号(本地可算): 汇率/南向/纳指/A股自身风险。"""
    date = date or today()
    missing: list[str] = []

    fx = _load_fx(date)
    if fx is None:
        missing.append("美元/人民币汇率: data_warehouse 无可用 rates__fx.parquet")

    south = _load_southbound(date)
    if south is None:
        missing.append("南向资金: 本地无 southbound_flow.parquet 且 akshare 获取失败, 港股情绪不可算")

    # 纳指/标普: 本地无数据 → None, 不编造
    missing.append("纳指/标普: 本地无指数数据, 无法评估美股涨跌, 标注 None")

    hk_sentiment = None
    if south and south.get("net_buy_yi") is not None:
        v = float(south["net_buy_yi"])
        hk_sentiment = "偏强" if v > 30 else ("偏弱" if v < -30 else "中性")

    return {
        "date": date,
        "fx_usdcny": fx,
        "southbound_flow": south,
        "us_index": None,
        "ashare_risk": _ashare_risk(date),
        "hk_sentiment": hk_sentiment,
        "external_risk_note": (
            "外部大跌(如纳指/恒指急跌)时通常压制A股风险偏好; "
            "本地无美股指数数据, 外部大跌信号需经 akshare/手动补充后接入"
        ),
        "data_missing": missing,
    }


def ib_sync() -> dict:
    """IBKR 持仓同步: 未配置 → unavailable; 配置后经 ib_insync 拉真实仓位, 不编造。"""
    account = os.getenv("IBKR_ACCOUNT", "").strip()
    token = os.getenv("IBKR_TOKEN", "").strip()
    host = os.getenv("IBKR_HOST", "").strip()
    if not (account and token):
        return {
            "available": False,
            "reason": "未配置 IBKR 凭证",
            "env": {"IBKR_ACCOUNT": bool(account), "IBKR_TOKEN": bool(token),
                    "IBKR_HOST": bool(host)},
        }
    try:
        from ib_insync import IB  # noqa: PLC0415
    except Exception:
        return {"available": False,
                "reason": "已配置 IBKR 凭证但未安装 ib-insync, 请先 pip install ib-insync"}
    try:
        ib = IB()
        port = int(os.getenv("IBKR_PORT", "7497"))
        ib.connect(host or "127.0.0.1", port, clientId=1)
        positions = [
            {
                "symbol": p.contract.symbol,
                "secType": p.contract.secType,
                "exchange": p.contract.exchange,
                "position": float(p.position),
                "market_price": float(p.marketPrice),
                "market_value": float(p.marketValue),
            }
            for p in ib.portfolio()
        ]
        ib.disconnect()
        exposure = round(sum(p["market_value"] for p in positions if p["position"] > 0), 2)
        return {"available": True, "account": account, "positions": positions,
                "long_exposure": exposure}
    except Exception as e:
        return {"available": False, "reason": f"IBKR 连接失败: {str(e)[:120]}"}


def hedge_advice(date: str | None = None, risk: str | None = None,
                 exposure: dict | None = None) -> dict:
    """对冲建议: A股风险等级 × 外部持仓暴露 → 具体对冲方案。

    risk: "高"/"中"/"低", 缺省时从 battle_map/self_healer 推导。
    exposure: 外部多头暴露 dict, 如 {"美股": 100000}; 缺省时尝试 ib_sync,
              不可用则按"无外部持仓"降级(仍可运行)。
    """
    date = date or today()
    if risk and risk in RISK_LEVELS:
        risk_source = "user_provided"
        risk_evidence = f"调用方指定风险等级: {risk}"
    else:
        ashare = _ashare_risk(date)
        risk = ashare.get("level", "中")
        risk_source = ashare.get("source", "battle_map/self_healer")
        risk_evidence = ashare.get("evidence", "")
        if risk not in RISK_LEVELS:
            risk = "中"

    exposure_source = "user_provided"
    if exposure is None:
        ib = ib_sync()
        if ib.get("available"):
            exposure = {}
            for p in ib.get("positions", []):
                if float(p.get("position", 0) or 0) > 0:
                    market = p.get("secType", "unknown")
                    exposure[market] = exposure.get(market, 0.0) + float(p.get("market_value", 0) or 0)
            exposure_source = "ibkr"
        else:
            exposure = {}
            exposure_source = "ibkr_unavailable"

    total = sum(float(v) for v in exposure.values() if v and float(v) > 0)
    has_exposure = total > 0

    if risk == "高":
        if has_exposure:
            hedge_needed, direction, instruments, size_hint, reason = (
                "是", "对冲外部多头", HEDGE_INSTRUMENTS["高"],
                f"对冲外部暴露的 30%-50% (约 {int(total * 0.3):,} ~ {int(total * 0.5):,} 元名义)",
                f"A股风险=高 且外部多头暴露 {total:,.0f} 元 → "
                f"建议股指期货空头/买PUT/纳指空头ETF对冲, 降低跨市场回撤")
        else:
            hedge_needed, direction, instruments, size_hint, reason = (
                "否", "提示降仓", [],
                "0",
                "A股风险=高 但未检测到外部多头持仓 → 不涉及跨市场对冲, 只需降低A股仓位")
    elif risk == "中":
        if has_exposure:
            hedge_needed, direction, instruments, size_hint, reason = (
                "否", "观察(可轻仓保险)", HEDGE_INSTRUMENTS["中"],
                f"10%-20% 外部暴露 (约 {int(total * 0.1):,} ~ {int(total * 0.2):,} 元名义, 可选)",
                f"A股风险=中 且外部多头暴露 {total:,.0f} 元 → 暂不强制对冲, 可轻仓买PUT/纳指空头ETF保险")
        else:
            hedge_needed, direction, instruments, size_hint, reason = (
                "否", "无需对冲", [], "0",
                "A股风险=中 且无外部多头持仓 → 无需对冲")
    else:
        hedge_needed, direction, instruments, size_hint, reason = (
            "否", "无需对冲", [], "0",
            "A股风险=低 → 无对冲需求")

    return {
        "date": date,
        "risk": risk,
        "risk_source": risk_source,
        "risk_evidence": risk_evidence,
        "exposure": {k: v for k, v in (exposure or {}).items()},
        "exposure_source": exposure_source,
        "hedge_needed": hedge_needed,
        "direction": direction,
        "instruments": instruments,
        "size_hint": size_hint,
        "reason": reason,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="对冲监控(IBKR/外部持仓框架)")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD")
    args = ap.parse_args()
    print(json.dumps(external_risk_signals(args.date), ensure_ascii=False, indent=2))
    print(json.dumps(hedge_advice(args.date), ensure_ascii=False, indent=2))
