"""
市场背景数据 — 为决策引擎提供全面的宏观/指数/商品/广度背景。

数据源:
  - 新浪: A股主要指数实时（上证/深证/创业板）
  - 腾讯: 宽基指数实时（上证50/沪深300/中证500/中证1000，P2-Q4-fix M417）
  - 东方财富: 全市场涨跌家数（不可达时估算回退并显式标注，P2-Q4-fix M416）
  - Yahoo: 黄金/白银/纳指100
"""

from __future__ import annotations
import logging

from datetime import timedelta, timezone
from typing import Any

import requests as _req

from quant_system.utils import to_float as _safe_float

CST = timezone(timedelta(hours=8))


def _to_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# ── 公共头 ──────────────────────────────────────────────────────

SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn",
}
TENCENT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.qq.com",
}
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


# ── 国内指数 ────────────────────────────────────────────────────

def _fetch_tencent_indices() -> dict[str, Any]:
    """腾讯 qt.gtimg.cn 宽基指数实时行情（上证50/沪深300/中证500/中证1000）。

    P2-Q4-fix(M417): 原 tencent_codes 字典定义后从未发起任何请求，四个宽基指数
    恒缺失；此处补齐真实腾讯请求。失败时静默缺省（返回空 dict，不影响主流程）。
    """
    tencent_codes = {
        "sh000016": "上证50",
        "sh000300": "沪深300",
        "sh000905": "中证500",
        "sh000852": "中证1000",
    }
    out: dict[str, Any] = {}
    try:
        url = f"https://qt.gtimg.cn/q={','.join(tencent_codes.keys())}"
        r = _req.get(url, headers=TENCENT_HEADERS, timeout=6)
        if r.status_code != 200:
            return out
        for line in r.text.strip().split("\n"):
            if '"' not in line:
                continue
            parts = line.split('"')[1].split("~")
            if len(parts) < 5:
                continue
            name = parts[1] or ""
            price = _safe_float(parts[3])
            prev_close = _safe_float(parts[4], price) if parts[4] else price
            if not price or price <= 0:
                continue
            chg = price - prev_close
            pct = (chg / prev_close * 100) if prev_close else 0.0
            amount_wan = _safe_float(parts[37]) if len(parts) > 37 else 0.0
            out[name] = {
                "index": price,
                "change": round(chg, 3),
                "change_pct": round(pct, 2),
                "amount_yi": round(amount_wan / 10000, 1),
                "source": "tencent",
            }
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_context] 操作失败: {e}", exc_info=True)
    return out


def fetch_a_indices() -> dict[str, Any]:
    """A股主要指数实时行情 + 全市场成交量。"""
    # Sina s_ 前缀返回简化版（current, change, change%, vol, amount）
    sina_codes = {
        "s_sh000001": "上证指数",
        "s_sz399001": "深证成指",
        "s_sz399006": "创业板指",
    }

    result = {}

    # Sina indices (快速)
    try:
        r = _req.get(
            f"http://hq.sinajs.cn/list={','.join(sina_codes.keys())}",
            headers=SINA_HEADERS, timeout=6
        )
        if r.status_code == 200:
            for line in r.text.strip().split("\n"):
                if "=" not in line:
                    continue
                try:
                    key = line.split("=")[0].split("_")[-1]
                    parts = line.split('"')[1].split(",")
                    name = parts[0]
                    idx = float(parts[1])
                    chg = float(parts[2])
                    pct = float(parts[3])
                    # P2-Q4-fix(L428): 新浪 s_ 接口成交量单位是"手"，非"万手"，
                    # 键名随之改为 volume_hand 避免 1e4 量级误解
                    vol = float(parts[4])  # 手
                    amt = float(parts[5])  # 万元
                    result[name] = {
                        "index": idx, "change": chg, "change_pct": pct,
                        "volume_hand": vol, "amount_yi": round(amt / 10000, 1),
                    }
                except Exception as e:
                    logging.getLogger(__name__).error(f"[market_context] 操作失败: {e}", exc_info=True)
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_context] 操作失败: {e}", exc_info=True)

    # P2-Q4-fix(M417): 补齐腾讯宽基指数请求（原 tencent_codes 死字典 + 虚假 docstring）
    result.update(_fetch_tencent_indices())

    # 全市场成交量汇总
    # P1-Q4-fix: 求和键与实际写入键一致(volume_hand/amount_yi)，修复 _total_volume/_total_amount 恒 0 导致的静默失真
    # P2-Q4-fix(M418): 深证成指成交额已含创业板，总量只取 沪+深 两市场口径
    total_vol = sum(v.get("volume_hand", 0) for k, v in result.items()
                    if k in ("上证指数", "深证成指") and isinstance(v, dict))
    total_amt = sum(v.get("amount_yi", 0) for k, v in result.items()
                    if k in ("上证指数", "深证成指") and isinstance(v, dict))
    result["_total_volume"] = total_vol
    result["_total_amount"] = total_amt

    return result


# ── 涨跌家数（东方财富真实统计；不可达时估算并显式标注）────────

def _finish_advance_decline(d: dict[str, Any]) -> dict[str, Any]:
    """补齐涨跌家数统一字段（advance_pct/decline_pct/ratio/total）。"""
    adv = int(d.get("advance", 0))
    dec = int(d.get("decline", 0))
    flat = int(d.get("flat", 0))
    total = int(d.get("total", adv + dec + flat))
    d["total"] = total
    d["advance_pct"] = round(adv / total * 100, 1) if total > 0 else 0
    d["decline_pct"] = round(dec / total * 100, 1) if total > 0 else 0
    d["ratio"] = round(adv / dec, 2) if dec > 0 else 0
    return d


def _fetch_real_advance_decline() -> dict[str, Any] | None:
    """从东方财富获取真实涨跌家数（沪市+深市），失败返回 None。

    P2-Q4-fix(M416): 优先用真实接口替代指数拍脑袋估算；
    东财不可达/返回占位数据时返回 None，由上层回退估算。
    """
    try:
        url = ("https://push2delay.eastmoney.com/api/qt/ulist.np/get?"
               "fltt=2&secids=1.000001,0.399001&fields=f104,f105,f106")
        r = _req.get(url, headers=TENCENT_HEADERS, timeout=5)
        if r.status_code != 200:
            return None
        data = (r.json().get("data") or {}).get("diff") or []
        adv = dec = flat = 0
        for d in data:
            if not isinstance(d, dict):
                continue
            adv += _to_int(d.get("f104"))
            dec += _to_int(d.get("f105"))
            flat += _to_int(d.get("f106"))
        total = adv + dec + flat
        if total <= 0:
            return None
        return {
            "advance": adv, "decline": dec, "flat": flat,
            "total": total, "estimate": False, "source": "eastmoney",
        }
    except Exception:
        return None


def fetch_advance_decline(indices: dict) -> dict[str, Any]:
    """
    市场涨跌家数（P2-Q4-fix M416）。

    优先请求东财真实涨跌家数（estimate=False）；失败时按指数涨跌做合理估算，
    估算结果恒带 estimate=True 与显式"[估算]"标注，禁止下游当真实市场广度使用。
    """
    real = _fetch_real_advance_decline()
    if real:
        return _finish_advance_decline(real)

    # 估算回退：从指数涨跌推算市场广度
    sh = indices.get("上证指数", {})
    sz = indices.get("深证成指", {})
    cy = indices.get("创业板指", {})

    sh_pct = sh.get("change_pct", 0)
    sz_pct = sz.get("change_pct", 0)
    cy_pct = cy.get("change_pct", 0)

    # 粗略估算：根据指数加权表现
    avg_pct = (sh_pct + sz_pct + cy_pct) / 3

    # 根据历史经验估算涨跌比
    if avg_pct > 0.5:
        est_adv_pct = 60 + avg_pct * 5
    elif avg_pct > 0:
        est_adv_pct = 50 + avg_pct * 10
    elif avg_pct > -1:
        est_adv_pct = 40 + avg_pct * 10
    elif avg_pct > -3:
        est_adv_pct = 25 + avg_pct * 5
    else:
        est_adv_pct = max(5, 15 + avg_pct * 3)

    est_adv_pct = max(5, min(95, est_adv_pct))
    # P2-Q4-fix(M416): A股约5400+家，原固定 5000 偏低
    est_total = 5400
    est_adv = int(est_total * est_adv_pct / 100)
    est_dec = est_total - est_adv

    est = {
        "advance": est_adv, "decline": est_dec, "flat": 0,
        "total": est_total, "estimate": True,
        "advance_pct": round(est_adv_pct, 1),
        "decline_pct": round(100 - est_adv_pct, 1),
        "ratio": round(est_adv / est_dec, 2) if est_dec > 0 else 0,
        "note": f"[估算] 基于指数加权估算(上证{sh_pct:+.1f}% 深证{sz_pct:+.1f}% 创{cy_pct:+.1f}%)",
    }
    return _finish_advance_decline(est)


# ── 商品 ────────────────────────────────────────────────────────

def fetch_commodities() -> dict[str, Any]:
    """
    主要商品现货价格。
    黄金/白银: Yahoo Finance (GC=F, SI=F)
    原油: Yahoo Finance (CL=F)
    """
    result = {}
    yahoo_symbols = {
        "GC=F": "黄金",
        "SI=F": "白银",
        "CL=F": "原油(WTI)",
    }

    for sym, name in yahoo_symbols.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=1d"
            r = _req.get(url, headers=YAHOO_HEADERS, timeout=8)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                prev = meta.get("chartPreviousClose")
                if price:
                    chg_pct = (price - prev) / prev * 100 if prev else 0
                    result[name] = {
                        "price": price,
                        "change_pct": round(chg_pct, 2),
                        "currency": meta.get("currency", "USD"),
                    }
        except Exception as e:
            logging.getLogger(__name__).error(f"[market_context] 操作失败: {e}", exc_info=True)

    return result


# ── 国际指数 ────────────────────────────────────────────────────

def fetch_global_indices() -> dict[str, Any]:
    """主要国际指数。"""
    result = {}
    yahoo_symbols = {
        "^NDX": "纳斯达克100",
        "^DJI": "道琼斯",
        "^GSPC": "标普500",
        "^HSI": "恒生指数",
        "000300.SS": "沪深300",
    }

    for sym, name in yahoo_symbols.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=1d"
            r = _req.get(url, headers=YAHOO_HEADERS, timeout=8)
            if r.status_code == 200:
                data = r.json()
                meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                prev = meta.get("chartPreviousClose")
                if price:
                    chg_pct = (price - prev) / prev * 100 if prev else 0
                    result[name] = {
                        "price": round(price, 2),
                        "change_pct": round(chg_pct, 2),
                    }
        except Exception as e:
            logging.getLogger(__name__).error(f"[market_context] 操作失败: {e}", exc_info=True)

    return result


# ── 综合市场背景 ────────────────────────────────────────────────

def get_market_context() -> dict[str, Any]:
    """
    全量市场背景数据 — 供决策引擎使用。

    Returns:
        {
            "indices": {...},           # A股指数
            "advance_decline": {...},   # 涨跌家数
            "commodities": {...},       # 黄金/白银/原油
            "global": {...},            # 纳指100/恒指等
            "summary": "..."            # 文本摘要
        }
    """
    result = {}
    errors = []

    # 1. 指数
    try:
        indices = fetch_a_indices()
        result["indices"] = {k: v for k, v in indices.items() if not k.startswith("_")}
        result["total_volume"] = indices.get("_total_volume", 0)
        result["total_amount"] = indices.get("_total_amount", 0)
    except Exception as e:
        errors.append(f"指数: {e}")

    # 2. 涨跌家数
    try:
        result["advance_decline"] = fetch_advance_decline(result.get("indices", {}))
    except Exception as e:
        errors.append(f"涨跌: {e}")

    # 3. 商品
    try:
        result["commodities"] = fetch_commodities()
    except Exception as e:
        errors.append(f"商品: {e}")

    # 4. 国际指数
    try:
        result["global_indices"] = fetch_global_indices()
    except Exception as e:
        errors.append(f"国际: {e}")

    # 5. 摘要文本
    result["_summary"] = _summarize_context(result)
    result["_errors"] = errors
    return result


def _summarize_context(ctx: dict) -> str:
    """生成一段简洁的市场背景文字。"""
    parts = []

    # 指数
    indices = ctx.get("indices", {})
    if indices:
        idx_lines = []
        for name in ["上证指数", "深证成指", "创业板指"]:
            v = indices.get(name)
            if v:
                arrow = "📈" if v.get("change_pct", 0) >= 0 else "📉"
                idx_lines.append(f"{arrow} {name}: {v.get('index', 0):.0f} ({v.get('change_pct', 0):+.2f}%)")
        if idx_lines:
            parts.append("\n".join(idx_lines))

    # 全市场成交量汇总
    # P2-Q4-fix(M418): 深证成指成交额已含创业板，取 沪+深 两市场口径，避免创业板重复计入虚高
    indices = ctx.get("indices", {})
    sh = indices.get("上证指数", {})
    sz = indices.get("深证成指", {})
    total_amt_yi = sh.get("amount_yi", 0) + sz.get("amount_yi", 0)
    if total_amt_yi:
        parts.append(f"📊 全市场成交额(沪+深): {total_amt_yi:.0f}亿")

    # 涨跌家数（P2-Q4-fix M416: 估算时显著标注，禁止当真实广度）
    ad = ctx.get("advance_decline", {})
    if ad and ad.get("total", 0) > 0:
        adv = ad.get("advance", 0)
        dec = ad.get("decline", 0)
        est_flag = "（估算）" if ad.get("estimate", True) else ""
        parts.append(f"📈 上涨{adv} 下跌{dec} (涨跌比{ad.get('ratio', 0):.2f}){est_flag}")

    # 商品
    comm = ctx.get("commodities", {})
    if comm:
        comm_lines = []
        for name in ["黄金", "白银", "原油(WTI)"]:
            v = comm.get(name)
            if v:
                arrow = "📈" if v.get("change_pct", 0) >= 0 else "📉"
                comm_lines.append(f"{arrow} {name}: ${v.get('price', 0):.1f} ({v.get('change_pct', 0):+.2f}%)")
        if comm_lines:
            parts.append("\n".join(comm_lines))

    # 国际
    gl = ctx.get("global_indices", {})
    if gl:
        gl_lines = []
        for name in ["纳斯达克100", "恒生指数"]:
            v = gl.get(name)
            if v:
                arrow = "📈" if v.get("change_pct", 0) >= 0 else "📉"
                gl_lines.append(f"{arrow} {name}: {v.get('price', 0):.2f} ({v.get('change_pct', 0):+.2f}%)")
        if gl_lines:
            parts.append("\n".join(gl_lines))

    return "\n".join(parts)
