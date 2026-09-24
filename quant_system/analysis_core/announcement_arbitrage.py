"""
announcement_arbitrage — 公告事件深度排雷/套利（V11）

把公告从"关键词匹配标签"升级为"可计算结论":
  强赎 / 要约收购 / 配股 / 减持 / 回购 → 正则提取关键参数
  → 结合本地行情(零网络)计算亏损比例/转股价值/安全边际
  → 输出"今天必须做什么"。

数据源(全部本地, 禁止网络):
  data_warehouse/market/cb_spot.parquet   可转债行情(现价/转股价若有)
  data_warehouse/kline/{code}.parquet     正股日线(最近收盘价)

用法:
  from quant_system.analysis_core.announcement_arbitrage import (
      analyze, convertible_bond_arb, announcement_risk,
  )
  python3 -m quant_system.analysis_core.announcement_arbitrage --self-test
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CB_SPOT = ROOT / "data_warehouse" / "market" / "cb_spot.parquet"
KLINE_DIR = ROOT / "data_warehouse" / "kline"

# ── 事件类型识别关键词（按优先级判断）────────────────────
EVENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("强赎", ("强赎", "强制赎回", "赎回", "最后交易日", "最后转股日")),
    ("要约收购", ("要约",)),
    ("配股", ("配股",)),
    ("减持", ("减持",)),
    ("回购", ("回购",)),
]

# ── 日期正则: 覆盖 "8月12日" / "2026-08-12" / "八月十二日" / "8/12" ──
_CN_NUM = r"[一二两三四五六七八九十]{1,3}"
_DATE_PATTERN = (
    r"(?:20\d{2}|19\d{2})[-/.]\d{1,2}[-/.]\d{1,2}"      # 2026-08-12
    r"|(?:[0-9]{1,2}|" + _CN_NUM + r")月(?:[0-9]{1,2}|" + _CN_NUM + r")日?"  # 8月12日/八月十二日
    r"|\d{1,2}/\d{1,2}"                                  # 8/12
)
_DATE_RE = re.compile(_DATE_PATTERN)

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _cn_to_int(s: str) -> int | None:
    """中文数字 → int（覆盖 0-99，如 十二→12、三十一→31、八→8）。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(s) == 1:
        return _CN_DIGITS.get(s)
    num = 0
    for ch in s:
        d = _CN_DIGITS.get(ch)
        if d is None:
            return None
        num = num * 10 + d
    return num


def _parse_date_token(token: str) -> tuple[int | None, int, int] | None:
    """解析单个日期串 → (年|None, 月, 日)。失败返回 None。"""
    token = token.strip()
    if not token:
        return None
    m = re.match(r"^(20\d{2}|19\d{2})[-/.](\d{1,2})[-/.](\d{1,2})$", token)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(\d{1,2})/(\d{1,2})$", token)
    if m:
        return None, int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{1,2})-(\d{1,2})$", token)
    if m:
        return None, int(m.group(1)), int(m.group(2))
    m = re.match(r"^([0-9]{1,2}|" + _CN_NUM + r")月([0-9]{1,2}|" + _CN_NUM + r")日?$", token)
    if m:
        mon_s, day_s = m.group(1), m.group(2)
        mon = int(mon_s) if mon_s.isdigit() else _cn_to_int(mon_s)
        day = int(day_s) if day_s.isdigit() else _cn_to_int(day_s)
        if mon and day:
            return None, mon, day
    return None


def _norm_date(parsed: tuple[int | None, int, int] | None,
               raw: str) -> tuple[str | None, str]:
    """归一化: 有年 → 'YYYY-MM-DD'；无年 → 'MM-DD'。返回 (归一值, 原文)。"""
    if not parsed:
        return None, raw
    y, m, d = parsed
    return (f"{y:04d}-{m:02d}-{d:02d}" if y else f"{m:02d}-{d:02d}"), raw


def _extract_dated(text: str, label: str) -> tuple[str | None, str | None]:
    """提取紧跟 label（如'最后转股日'）之后的日期。返回 (归一值, 原文)。"""
    for m in re.finditer(re.escape(label), text):
        tail = text[m.end():m.end() + 16]
        dm = _DATE_RE.search(tail)
        if dm:
            raw = dm.group(0)
            norm, _ = _norm_date(_parse_date_token(raw), raw)
            return norm, raw
    return None, None


def _extract_price(text: str, *patterns: str) -> float | None:
    """按顺序尝试提取价格数值。"""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _extract_pct(text: str, *patterns: str) -> float | None:
    """按顺序尝试提取百分比数值（返回原样数字，如 18.05）。"""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _extract_redemption_params(text: str) -> dict:
    params: dict = {}
    params["redeem_price"] = _extract_price(
        text,
        r"(?:强赎|强制赎回|赎回)(?:价|价格)(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)",
    )
    for key, label in (("last_trade_date", "最后交易日"),
                       ("last_convert_date", "最后转股日"),
                       ("record_date", "赎回登记日")):
        norm, raw = _extract_dated(text, label)
        if norm:
            params[key] = norm
            params[f"{key}_raw"] = raw
    params["convert_price"] = _extract_price(
        text,
        r"转股价(?:格)?(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)",
    )
    return params


def _extract_offer_params(text: str) -> dict:
    params: dict = {}
    params["offer_price"] = _extract_price(
        text,
        r"要约(?:收购)?(?:价|价格)(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)",
        r"收购价(?:格)?(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)",
    )
    params["offer_ratio"] = _extract_pct(
        text,
        r"要约(?:收购)?(?:股份)?比例(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)\s*%?",
        r"收购(?:股份)?比例(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)\s*%?",
    )
    return params


def _extract_rights_params(text: str) -> dict:
    params: dict = {}
    params["rights_price"] = _extract_price(
        text,
        r"配股价(?:格)?(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)",
    )
    m = re.search(r"每\s*(\d+)\s*股(?:配|转增)?(?:配)?\s*(\d+(?:\.\d+)?)\s*股", text)
    if m:
        base, per = float(m.group(1)), float(m.group(2))
        params["rights_ratio"] = round(per / base, 6)
        params["rights_ratio_raw"] = f"每{int(base)}股配{per:g}股"
    else:
        pct = _extract_pct(text, r"配股比例(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)\s*%?")
        params["rights_ratio"] = pct
        params["rights_ratio_raw"] = f"{pct}%" if pct is not None else None
    norm, raw = _extract_dated(text, "股权登记日")
    if norm:
        params["record_date"] = norm
        params["record_date_raw"] = raw
    return params


def _extract_reduce_params(text: str) -> dict:
    params: dict = {}
    params["reduce_ratio"] = _extract_pct(
        text,
        r"减持(?:股份)?比例(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)\s*%",
        r"减持[^。；;，,]{0,12}?不超过\s*(\d+(?:\.\d+)?)\s*%",
        r"减持[^。；;，,]{0,12}?(\d+(?:\.\d+)?)\s*%",
    )
    return params


def _extract_buyback_params(text: str) -> dict:
    params: dict = {}
    m = re.search(r"回购(?:金额|资金总额|资金)(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)\s*(亿|万)?", text)
    if m:
        params["buyback_amount"] = float(m.group(1))
        params["buyback_amount_unit"] = m.group(2) or ""
    params["buyback_price"] = _extract_price(
        text,
        r"回购价格(?:上限)?(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)",
    )
    return params


def analyze(title: str, content: str) -> dict:
    """识别公告事件类型并提取关键参数。

    返回: {event_type, params, conclusion}
    """
    text = f"{title or ''} {content or ''}"
    event_type = "unknown"
    for et, keys in EVENT_KEYWORDS:
        if any(k in text for k in keys):
            event_type = et
            break

    if event_type == "unknown":
        return {"event_type": "unknown", "params": {}, "conclusion": "无量化结论"}

    if event_type == "强赎":
        params = _extract_redemption_params(text)
        parts: list[str] = []
        if params.get("redeem_price") is not None:
            parts.append(f"赎回价{params['redeem_price']}元")
        if params.get("last_convert_date"):
            parts.append(f"最后转股日{params['last_convert_date']}")
        if params.get("record_date"):
            parts.append(f"赎回登记日{params['record_date']}")
        if params.get("convert_price"):
            parts.append(f"转股价{params['convert_price']}元")
        conclusion = (
            f"已触发强赎，{'，'.join(parts)}；现价高于赎回价时不转股将按赎回价兑付造成亏损"
            if parts else "已触发强赎，需在最后转股日前处理持仓（关键参数缺失，请人工核对公告）"
        )
    elif event_type == "要约收购":
        params = _extract_offer_params(text)
        parts = []
        if params.get("offer_price") is not None:
            parts.append(f"要约价{params['offer_price']}元")
        if params.get("offer_ratio") is not None:
            parts.append(f"要约比例{params['offer_ratio']}%")
        conclusion = (
            f"要约收购：{'，'.join(parts)}；对比现价与要约价决定是否接受要约"
            if parts else "要约收购公告，缺少要约价/比例，请人工核对"
        )
    elif event_type == "配股":
        params = _extract_rights_params(text)
        parts = []
        if params.get("rights_price") is not None:
            parts.append(f"配股价{params['rights_price']}元")
        if params.get("rights_ratio") is not None:
            parts.append(f"配股比例{params['rights_ratio_raw'] or params['rights_ratio']}")
        if params.get("record_date"):
            parts.append(f"股权登记日{params['record_date']}")
        conclusion = (
            f"配股方案：{'，'.join(parts)}；不参与需在股权登记日前卖出，参与需按时缴款"
            if parts else "配股公告，缺少配股价/比例，请人工核对"
        )
    elif event_type == "减持":
        params = _extract_reduce_params(text)
        conclusion = (
            f"股东减持{'比例' + str(params['reduce_ratio']) + '%' if params.get('reduce_ratio') is not None else ''}，短期或有抛压"
        )
    else:  # 回购
        params = _extract_buyback_params(text)
        parts = []
        if params.get("buyback_amount") is not None:
            unit = params.get("buyback_amount_unit") or ""
            parts.append(f"回购金额{params['buyback_amount']:g}{unit}元")
        if params.get("buyback_price") is not None:
            parts.append(f"回购价上限{params['buyback_price']}元")
        conclusion = f"公司回购（{'，'.join(parts)}），中性偏多" if parts else "公司回购，中性偏多"

    return {"event_type": event_type, "params": params, "conclusion": conclusion}


# ── 行情读取（本地 parquet，零网络）──────────────────────
def _load_cb_price(code: str) -> float | None:
    if not CB_SPOT.exists():
        return None
    try:
        df = pd.read_parquet(CB_SPOT)
    except Exception:
        return None
    if df.empty:
        return None
    code_s = str(code)
    row = df[(df["code"].astype(str) == code_s) | (df["symbol"].astype(str).str.endswith(code_s))]
    if row.empty:
        return None
    r = row.iloc[0]
    for col in ("trade", "settlement", "close"):
        if col in r and pd.notna(r[col]):
            v = float(r[col])
            if v > 0:
                return round(v, 3)
    return None


def _load_cb_convert_price(code: str) -> float | None:
    if not CB_SPOT.exists():
        return None
    try:
        df = pd.read_parquet(CB_SPOT)
    except Exception:
        return None
    if df.empty:
        return None
    code_s = str(code)
    row = df[(df["code"].astype(str) == code_s) | (df["symbol"].astype(str).str.endswith(code_s))]
    if row.empty:
        return None
    r = row.iloc[0]
    for col in df.columns:
        if "转股" in str(col) and "价" in str(col) and pd.notna(r[col]):
            v = float(r[col])
            if v > 0:
                return v
    return None


def _load_stock_price(code: str) -> float | None:
    p = KLINE_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    df = df.dropna(subset=["close"]) if "close" in df else df
    if df.empty:
        return None
    return round(float(df["close"].iloc[-1]), 3)


_bond_name_cache: dict[str, str] = {}


def _bond_name(code: str) -> str | None:
    if code in _bond_name_cache:
        return _bond_name_cache[code]
    if not CB_SPOT.exists():
        return None
    try:
        df = pd.read_parquet(CB_SPOT)
        code_s = str(code)
        row = df[(df["code"].astype(str) == code_s) | (df["symbol"].astype(str).str.endswith(code_s))]
        if not row.empty and "name" in row:
            name = str(row.iloc[0]["name"])
            _bond_name_cache[code] = name
            return name
    except Exception as e:
        logging.getLogger(__name__).error(f"[announcement_arbitrage] 操作失败: {e}", exc_info=True)
    _bond_name_cache[code] = ""
    return None


# ── 强赎套利计算 ───────────────────────────────────────
def _days_to(date_value: str, now: datetime | date | None = None) -> int | None:
    """距目标日期还有几天（date_value 无年份时按当前年推算，过期则顺延一年）。"""
    if not date_value:
        return None
    parsed = _parse_date_token(date_value)
    if not parsed:
        return None
    y, m, d = parsed
    today = (now or datetime.now()).date()
    if y is None:
        y = today.year
    target = date(y, m, d)
    days = (target - today).days
    if days < -300:
        target = date(y + 1, m, d)
        days = (target - today).days
    return days


def _cb_recommendation(bond_price: float | None, redeem_price: float | None,
                       convert_value: float | None, loss: float | None,
                       days: int | None) -> str:
    if bond_price is None or redeem_price is None:
        return "观望（行情缺失，无法量化）"
    if bond_price <= redeem_price:
        return "可持有等待赎回/观望（现价已低于赎回价，无强制亏损）"
    urgent = days is not None and days <= 1
    if convert_value is not None:
        base = "转股" if convert_value >= bond_price else "卖出"
    else:
        base = "卖出或转股"
    if urgent or (loss is not None and loss >= 0.10):
        return f"务必今日{base}"
    return f"尽快{base}"


def convertible_bond_arb(code: str, redeem_price: float, last_convert_date: str,
                         *, bond_price: float | None = None,
                         stock_price: float | None = None,
                         convert_ratio: float | None = None,
                         convert_price: float | None = None,
                         stock_code: str | None = None,
                         now: datetime | date | None = None) -> dict:
    """强赎套利计算。

    不转股亏损比例 = (转债现价 - 赎回价) / 转债现价
    转股价值 = 正股价 × 转股比例（转股比例 = 100 / 转股价，若行情或参数可得到）

    行情缺失时对应字段为 None，不抛错、不联网。
    """
    if bond_price is None:
        bond_price = _load_cb_price(code)

    loss = None
    if bond_price is not None and redeem_price:
        loss = round((bond_price - redeem_price) / bond_price, 4)

    if convert_ratio is None:
        if convert_price:
            convert_ratio = 100.0 / convert_price
        else:
            cp = _load_cb_convert_price(code)
            if cp:
                convert_ratio = 100.0 / cp

    if stock_price is None:
        stock_price = _load_stock_price(stock_code or code)

    convert_value = None
    if stock_price is not None and convert_ratio:
        convert_value = round(stock_price * convert_ratio, 2)

    days = _days_to(last_convert_date, now)
    recommendation = _cb_recommendation(bond_price, redeem_price, convert_value, loss, days)

    margin = None
    if bond_price is not None and redeem_price:
        if convert_value is not None:
            margin = round((convert_value - redeem_price) / redeem_price, 4)
        else:
            margin = round((bond_price - redeem_price) / redeem_price, 4)

    return {
        "code": code,
        "bond_price": bond_price,
        "redeem_price": redeem_price,
        "loss_if_not_convert": loss,
        "convert_value": convert_value,
        "margin": margin,
        "recommendation": recommendation,
        "last_convert_date": last_convert_date,
        "urgency_days": days,
    }


def _short_date(value: str) -> str:
    parsed = _parse_date_token(value)
    if not parsed:
        return value
    y, m, d = parsed
    return f"{y}/{m}/{d}" if y else f"{m}/{d}"


def _fmt_cb_desc(code: str, arb: dict) -> str:
    name = _bond_name(code)
    label = f"{name} {code}" if name else str(code)
    rp = arb["redeem_price"]
    bp = arb["bond_price"]
    loss = arb["loss_if_not_convert"]
    parts: list[str] = []
    if bp is not None and rp is not None:
        parts.append(f"现价{bp} vs 强赎价{rp}")
    if loss is not None:
        parts.append(f"不转股亏损{loss:.1%}")
    if arb["last_convert_date"]:
        parts.append(f"最后转股日{_short_date(arb['last_convert_date'])}")
    body = "，".join(parts) if parts else "强赎公告（行情缺失）"
    return f"{label}: {body} → {arb['recommendation']}"


def announcement_risk(code: str, title: str, content: str,
                      *, stock_code: str | None = None,
                      bond_price: float | None = None,
                      stock_price: float | None = None,
                      convert_price: float | None = None,
                      now: datetime | date | None = None) -> dict:
    """组合接口: 识别事件 + 结合行情计算 → {risk_level, actionable, numbers}。"""
    a = analyze(title, content)
    et = a["event_type"]
    if et == "unknown":
        return {
            "event_type": "unknown",
            "risk_level": "未知",
            "actionable": "无量化结论",
            "numbers": None,
            "description": None,
            "conclusion": a["conclusion"],
            "params": a["params"],
        }

    base = {"event_type": et, "conclusion": a["conclusion"], "params": a["params"]}

    if et == "强赎":
        rp = a["params"].get("redeem_price")
        lcd = a["params"].get("last_convert_date") or ""
        if rp is not None:
            arb = convertible_bond_arb(
                code, rp, lcd,
                bond_price=bond_price, stock_price=stock_price,
                convert_price=convert_price, stock_code=stock_code, now=now,
            )
            loss = arb["loss_if_not_convert"]
            days = arb["urgency_days"]
            if arb["bond_price"] is None:
                risk = "警示"
                action = "强赎已触发，行情缺失无法量化，尽快核实转债现价并处理"
                numbers = None
            elif loss is None:
                risk = "警示"
                action = arb["recommendation"]
                numbers = None
            elif loss >= 0.20 or (loss > 0 and days is not None and days <= 1):
                risk = "致命"
                action = arb["recommendation"]
                numbers = {k: arb[k] for k in (
                    "loss_if_not_convert", "convert_value", "margin",
                    "bond_price", "redeem_price", "urgency_days")}
            elif loss > 0:
                risk = "重大"
                action = arb["recommendation"]
                numbers = {k: arb[k] for k in (
                    "loss_if_not_convert", "convert_value", "margin",
                    "bond_price", "redeem_price", "urgency_days")}
            else:
                risk = "警示"
                action = arb["recommendation"]
                numbers = {k: arb[k] for k in (
                    "loss_if_not_convert", "convert_value", "margin",
                    "bond_price", "redeem_price", "urgency_days")}
            base.update({
                "risk_level": risk,
                "actionable": action,
                "numbers": numbers,
                "description": _fmt_cb_desc(code, arb),
            })
        else:
            base.update({
                "risk_level": "重大",
                "actionable": "公告触发强赎，未提取到赎回价，请人工核对后处理",
                "numbers": None,
                "description": f"{code}: 强赎公告（赎回价缺失）→ 尽快核实",
            })
        return base

    if et == "要约收购":
        op = a["params"].get("offer_price")
        sp = stock_price if stock_price is not None else _load_stock_price(stock_code or code)
        gap = None
        if op is not None and sp is not None:
            gap = round((op - sp) / sp, 4)
            risk = "重大" if gap > 0 else "警示"
            action = f"要约价{op}元 vs 现价{sp}元，{'溢价' if gap > 0 else '折价'}{abs(gap):.1%}；要约期内决定是否接受"
            numbers = {"offer_price": op, "stock_price": sp, "gap_pct": gap}
        else:
            risk = "警示"
            action = "关注要约价与要约期，对比现价决定是否接受"
            numbers = None
        base.update({"risk_level": risk, "actionable": action, "numbers": numbers,
                     "description": f"{code}: 要约收购公告"})
        return base

    if et == "配股":
        numbers = {
            "rights_price": a["params"].get("rights_price"),
            "rights_ratio": a["params"].get("rights_ratio"),
            "record_date": a["params"].get("record_date"),
        }
        base.update({
            "risk_level": "重大" if a["params"].get("record_date") or a["params"].get("rights_price") else "警示",
            "actionable": "股权登记日前必须决定：参与则按时缴款，不参与则登记日前卖出（不参与配股通常损失约10%）",
            "numbers": numbers,
            "description": f"{code}: 配股公告",
        })
        return base

    if et == "减持":
        base.update({
            "risk_level": "警示",
            "actionable": "关注减持比例与节奏，短期或有抛压",
            "numbers": {"reduce_ratio": a["params"].get("reduce_ratio")},
            "description": f"{code}: 减持公告",
        })
        return base

    # 回购
    base.update({
        "risk_level": "警示",
        "actionable": "回购中性偏多，观察金额与价格上限的持续性",
        "numbers": {
            "buyback_amount": a["params"].get("buyback_amount"),
            "buyback_amount_unit": a["params"].get("buyback_amount_unit"),
            "buyback_price": a["params"].get("buyback_price"),
        },
        "description": f"{code}: 回购公告",
    })
    return base


# ── 验收自测 ────────────────────────────────────────────
def self_test() -> bool:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        status = "✅" if cond else "❌"
        print(f"  {status} {msg}")
        if not cond:
            ok = False

    print("== 1. analyze 强赎（验收案例）==")
    r = analyze("正帆转债赎回实施的提示性公告",
                "最后交易日8月7日，最后转股日8月12日，赎回价格100.16元")
    check(r["event_type"] == "强赎", f"event_type=强赎 (实际 {r['event_type']})")
    check(r["params"].get("redeem_price") == 100.16,
          f"提取赎回价 100.16 (实际 {r['params'].get('redeem_price')})")
    check("8月12日" in str(r["params"]),
          f"提取 8月12日 (params={r['params']})")
    check(r["params"].get("last_trade_date") == "08-07",
          f"提取最后交易日 08-07 (实际 {r['params'].get('last_trade_date')})")
    print(f"  conclusion: {r['conclusion']}")

    print("== 2. 日期格式覆盖 ==")
    r2 = analyze("可转债赎回公告", "最后转股日2026-08-12，赎回价格100.16元")
    check(r2["params"].get("last_convert_date") == "2026-08-12",
          f"2026-08-12 → {r2['params'].get('last_convert_date')}")
    r3 = analyze("可转债赎回公告", "最后转股日八月十二日，赎回价格100.16元")
    check(r3["params"].get("last_convert_date") == "08-12",
          f"八月十二日 → {r3['params'].get('last_convert_date')}")
    r3b = analyze("可转债赎回公告", "最后转股日8/12，赎回价格100.16元")
    check(r3b["params"].get("last_convert_date") == "08-12",
          f"8/12 → {r3b['params'].get('last_convert_date')}")

    print("== 3. 其他事件类型 ==")
    r4 = analyze("关于要约收购的提示性公告", "要约收购价格28.5元，要约收购比例18.05%")
    check(r4["event_type"] == "要约收购" and r4["params"].get("offer_price") == 28.5
          and r4["params"].get("offer_ratio") == 18.05,
          f"要约收购 → {r4['event_type']} {r4['params']}")
    r5 = analyze("配股发行公告", "每10股配3股，配股价6.8元，股权登记日2026-08-15")
    check(r5["event_type"] == "配股" and r5["params"].get("rights_price") == 6.8
          and r5["params"].get("rights_ratio") == 0.3,
          f"配股 → {r5['event_type']} {r5['params']}")
    r6 = analyze("股东减持计划公告", "拟减持不超过1.5%")
    check(r6["event_type"] == "减持" and r6["params"].get("reduce_ratio") == 1.5,
          f"减持 → {r6['event_type']} {r6['params']}")
    r7 = analyze("回购股份方案公告", "回购金额不低于2亿元，回购价格上限15元")
    check(r7["event_type"] == "回购" and r7["params"].get("buyback_price") == 15.0,
          f"回购 → {r7['event_type']} {r7['params']}")
    u = analyze("日常经营公告", "公司召开董事会会议")
    check(u["event_type"] == "unknown" and u["conclusion"] == "无量化结论",
          f"unknown → {u}")

    print("== 4. convertible_bond_arb 构造数据 ==")
    arb = convertible_bond_arb("118053", 100.16, "2026-08-12",
                               bond_price=168.9, stock_price=65.98, convert_price=50.0)
    expect_loss = (168.9 - 100.16) / 168.9
    check(arb["loss_if_not_convert"] is not None
          and abs(arb["loss_if_not_convert"] - expect_loss) < 1e-3,
          f"不转股亏损 {arb['loss_if_not_convert']:.1%}")
    check(arb["convert_value"] == 131.96,
          f"转股价值 65.98×100/50 = {arb['convert_value']}")
    check("务必今日" in arb["recommendation"],
          f"recommendation={arb['recommendation']!r}")
    print(f"  arb={json.dumps(arb, ensure_ascii=False)}")

    print("== 5. announcement_risk 组合接口（构造数据）==")
    risk = announcement_risk("118053", "正帆转债赎回实施的提示性公告",
                             "最后交易日8月7日，最后转股日8月12日，赎回价格100.16元",
                             bond_price=168.9, stock_price=65.98, convert_price=50.0)
    check(risk["risk_level"] == "致命", f"risk_level={risk['risk_level']}")
    check("118053" in risk["description"], f"description={risk['description']}")
    check(risk["numbers"] is not None and "loss_if_not_convert" in risk["numbers"],
          "numbers 包含亏损比例")
    print(f"  description: {risk['description']}")
    print(f"  actionable : {risk['actionable']}")

    print("== 6. 行情缺失降级 ==")
    r_miss = announcement_risk("999999", "某转债赎回实施公告", "最后转股日8月12日，赎回价格100.16元")
    check(r_miss["risk_level"] == "警示" and r_miss["numbers"] is None,
          f"行情缺失 → numbers=None, risk={r_miss['risk_level']} (description={r_miss['description']})")

    print("== 7. 真实本地行情（若有则计算，无则降级不报错）==")
    real = announcement_risk("118053", "正帆转债赎回实施的提示性公告",
                             "最后交易日8月7日，最后转股日8月12日，赎回价格100.16元")
    print(f"  118053 真实行情: bond_price={real['numbers']['bond_price'] if real['numbers'] else None}, "
          f"risk={real['risk_level']}, actionable={real['actionable']}")
    print(f"  description: {real['description']}")

    print()
    print("全部通过 ✅" if ok else "存在失败项 ❌")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="公告事件深度排雷/套利（V11）")
    ap.add_argument("--self-test", action="store_true", help="运行验收自测（含构造数据）")
    ap.add_argument("--analyze-title", type=str, help="公告标题")
    ap.add_argument("--analyze-content", type=str, help="公告正文")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if args.analyze_title or args.analyze_content:
        print(json.dumps(analyze(args.analyze_title or "", args.analyze_content or ""),
                         ensure_ascii=False, indent=2))
    else:
        ap.print_help()
