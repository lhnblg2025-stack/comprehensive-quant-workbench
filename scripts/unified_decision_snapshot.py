#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一市场决策快照。

全市场是分析范围；主板只是当前用户可执行范围。该模块只读已有本地基座，
把市场状态、情绪、主线、资金、行业/ETF、风险和候选机会归一为一份快照，
供盘中预警、盘后复盘和 Web 看板复用。
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"
MARKET = ROOT / "data_warehouse" / "market"
CST = timezone(timedelta(hours=8))


def _safe(v: Any, default=None):
    if v is None:
        return default
    try:
        if isinstance(v, float) and math.isnan(v):
            return default
    except Exception:
        pass
    return v


def _json_ready(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_ready(x) for x in v]
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    try:
        item = v.item()
    except Exception:
        return v
    return _json_ready(item)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _local_datetime(values):
    """Parse naive trading dates as local dates and aware timestamps as CST."""
    import pandas as pd
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        if parsed.dt.tz is not None:
            return parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    except (AttributeError, TypeError, ValueError):
        pass
    return parsed


def _latest(pattern: str, not_after: str | None = None) -> Path | None:
    items = []
    for p in GEN.glob(pattern):
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", p.name)
        if not m or (not_after and m.group(1) > not_after):
            continue
        items.append((m.group(1), p))
    return sorted(items, key=lambda x: x[0], reverse=True)[0][1] if items else None


def _path_date(path: Path | None) -> str | None:
    if path is None:
        return None
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else None


def _snapshot_date() -> str | None:
    for p in (MARKET / "fusion.parquet", MARKET / "zt_daily_stats.parquet"):
        try:
            import pandas as pd
            if p.exists():
                df = pd.read_parquet(p, columns=["date"])
                if not df.empty:
                    return str(pd.to_datetime(df["date"]).max().date())
        except Exception:
            continue
    return None


def _load_fusion(date: str | None) -> dict:
    try:
        import pandas as pd
        p = MARKET / "fusion.parquet"
        if not p.exists():
            return {}
        df = pd.read_parquet(p)
        if df.empty:
            return {}
        if date and "date" in df.columns:
            df = df[pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") <= date]
        if df.empty:
            return {}
        if "date" in df.columns:
            df = df.assign(_decision_date=pd.to_datetime(df["date"], errors="coerce")).sort_values("_decision_date")
        return {k: _safe(v) for k, v in df.iloc[-1].to_dict().items() if not str(k).startswith("_")}
    except Exception:
        return {}


def _load_fund(date: str | None) -> dict:
    try:
        import pandas as pd
        p = MARKET / "fund_forces.parquet"
        if not p.exists():
            return {}
        df = pd.read_parquet(p)
        if date and "date" in df.columns:
            df = df[pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") <= date]
        if not df.empty and "date" in df.columns:
            df = df.assign(_decision_date=pd.to_datetime(df["date"], errors="coerce")).sort_values("_decision_date")
        return {k: _safe(v) for k, v in (df.iloc[-1].to_dict() if not df.empty else {}).items() if not str(k).startswith("_")}
    except Exception:
        return {}


def _load_chain(mode: str, date: str | None) -> dict:
    if mode == "intraday":
        p = _latest("intraday_chain_*.json", date)
        value = _read_json(p) if p else {}
        value["_source_dates"] = {"intraday_chain": _path_date(p)}
        return value
    if mode == "after_close":
        p = _latest("after_close_extra_*.json", date)
        base = _read_json(p) if p else {}
        intraday_path = _latest("intraday_chain_*.json", date)
        intraday = _read_json(intraday_path) if intraday_path else {}
        intraday_date = _path_date(intraday_path)
        # Candidate rows are execution-sensitive and must never cross trading days.
        if intraday and (not date or intraday_date == date):
            base["candidate_rows"] = intraday.get("candidate_rows") or intraday.get("rows") or []
            base["intraday_coverage"] = intraday.get("coverage") or {}
            base["n_scanned"] = intraday.get("n_scanned", 0)
            base["n_candidates"] = intraday.get("n_candidates", intraday.get("n_opportunities", 0))
            if not base["intraday_coverage"]:
                scanned = int(base.get("n_scanned") or 0)
                evaluated = len(base.get("candidate_rows") or [])
                base["intraday_coverage"] = {
                    "snapshot_rows": scanned,
                    "signal_evaluated": scanned,
                    "candidate_count": evaluated,
                    "evidence_evaluated": evaluated,
                    "ratio": 1.0 if scanned >= 1000 else 0.0,
                    "status": "complete" if scanned >= 1000 else "insufficient",
                }
        else:
            base["intraday_coverage"] = {
                "status": "date_mismatch" if intraday_date else "insufficient",
                "requested_date": date,
                "actual_date": intraday_date,
                "reason": "没有同日盘中链，旧盘后产物只能作历史结构参考" if not intraday_date else "盘中链与请求日不一致",
            }
        base["_source_dates"] = {
            "after_close_extra": _path_date(p),
            "intraday_chain": intraday_date,
        }
        return base
    p = _latest("battle_map_*.json", date)
    value = _read_json(p) if p else {}
    value["_source_dates"] = {"battle_map": _path_date(p)}
    return value


def _load_named_json(prefix: str, date: str | None) -> tuple[dict, str | None]:
    """Load an exact-date generated JSON, then the newest file at/before cutoff."""
    if date:
        exact = GEN / f"{prefix}_{date}.json"
        if exact.exists():
            return _read_json(exact), date
    p = _latest(f"{prefix}_*.json", date)
    return (_read_json(p), re.search(r"(20\d{2}-\d{2}-\d{2})", p.name).group(1)) if p else ({}, None)


def _load_research(date: str | None) -> tuple[dict, str | None]:
    return _load_named_json("research_flow", date)


def _load_holders() -> dict:
    return _read_json(GEN / "watchlist_holders.json")


def _load_inquiry() -> dict:
    return _read_json(GEN / "inquiry_letters.json")


def _load_factor_decay(date: str | None) -> tuple[dict, str | None]:
    """Load the nested factor-decay artifact without treating degraded output as valid IC."""
    candidates: list[tuple[str, Path]] = []
    for p in (GEN / "factor_decay").glob("*/factor_decay_*.json"):
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", p.name)
        if m and (not date or m.group(1) <= date):
            candidates.append((m.group(1), p))
    if not candidates:
        return {}, None
    actual, path = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    return _read_json(path), actual


def _research_evidence(research: dict) -> dict:
    """Normalize reports into A(code), B(name), and C(theme) evidence indexes."""
    parsed = research.get("parsed") or []
    code_map: dict[str, dict[str, Any]] = {}
    name_map: dict[str, dict[str, Any]] = {}
    concept_counts: dict[str, int] = {}
    decision_extracted_reports = 0
    for item in parsed:
        if not isinstance(item, dict):
            continue
        concepts = [str(x) for x in item.get("concepts") or [] if str(x).strip()]
        rating = item.get("rating")
        target_price = item.get("target_price")
        catalysts = [str(x) for x in item.get("catalysts") or [] if str(x).strip()]
        title = str(item.get("title") or "")
        has_decision_extract = bool(rating or target_price or catalysts)
        if has_decision_extract:
            decision_extracted_reports += 1
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        raw_companies = [str(x).strip() for x in item.get("companies") or [] if str(x).strip()]
        codes = {str(x).zfill(6) for x in item.get("codes") or [] if str(x).strip()}
        company_names: set[str] = set()
        for raw in raw_companies:
            match = re.search(r"(\d{6})", raw)
            if match:
                codes.add(match.group(1))
            name = re.sub(r"\(\d{6}\)", "", raw).strip()
            if len(name) >= 2:
                company_names.add(name)
        for name in company_names:
            evidence = name_map.setdefault(name, {"name": name, "reports": 0, "decision_extracted_reports": 0, "top_concepts": [], "top_catalysts": [], "rating": None, "target_price": None, "titles": []})
            evidence["reports"] += 1
            if has_decision_extract:
                evidence["decision_extracted_reports"] += 1
            evidence["top_concepts"] = list(dict.fromkeys(evidence["top_concepts"] + concepts))[:5]
            evidence["top_catalysts"] = list(dict.fromkeys(evidence["top_catalysts"] + catalysts))[:5]
            evidence["rating"] = evidence["rating"] or rating
            evidence["target_price"] = evidence["target_price"] if evidence["target_price"] is not None else target_price
            if title and len(evidence["titles"]) < 3:
                evidence["titles"].append(title[:100])
        for code in codes:
            evidence = code_map.setdefault(code, {"code": code, "reports": 0, "decision_extracted_reports": 0, "concepts": {}, "ratings": {}, "catalysts": {}, "target_prices": [], "titles": []})
            evidence["reports"] += 1
            if has_decision_extract:
                evidence["decision_extracted_reports"] += 1
            if target_price is not None:
                evidence["target_prices"].append(_safe(target_price))
            for concept in concepts:
                evidence["concepts"][concept] = evidence["concepts"].get(concept, 0) + 1
            if rating:
                evidence["ratings"][str(rating)] = evidence["ratings"].get(str(rating), 0) + 1
            for catalyst in catalysts:
                evidence["catalysts"][catalyst] = evidence["catalysts"].get(catalyst, 0) + 1
            if title and len(evidence["titles"]) < 3:
                evidence["titles"].append(title[:100])

    # ``all_codes`` is a useful coverage count even for legacy records that lack parsed rows.
    for code in research.get("all_codes") or []:
        code = str(code).zfill(6)
        code_map.setdefault(code, {"code": code, "reports": 0, "decision_extracted_reports": 0, "concepts": {}, "ratings": {}, "catalysts": {}, "target_prices": [], "titles": []})
    for evidence in code_map.values():
        evidence["top_concepts"] = [k for k, _ in sorted(evidence["concepts"].items(), key=lambda x: (-x[1], x[0]))[:5]]
        evidence["top_catalysts"] = [k for k, _ in sorted(evidence["catalysts"].items(), key=lambda x: (-x[1], x[0]))[:5]]
        evidence["rating"] = max(evidence["ratings"], key=evidence["ratings"].get) if evidence["ratings"] else None
        evidence["target_price"] = evidence["target_prices"][-1] if evidence["target_prices"] else None
        evidence["extraction_rate"] = round(evidence["decision_extracted_reports"] / evidence["reports"], 3) if evidence["reports"] else 0.0
        evidence.pop("target_prices", None)
        evidence.pop("concepts", None)
        evidence.pop("ratings", None)
        evidence.pop("catalysts", None)
    leader_map = {}
    for hit in research.get("leader_hits") or []:
        if not isinstance(hit, dict):
            continue
        concept = str(hit.get("concept") or "")
        if concept:
            leader_map[concept] = {k: _safe(hit.get(k)) for k in ("leader", "leader_boards", "zt_cnt", "diffusion", "signal")}
    top_concepts = [{"concept": k, "reports": v, "leader": leader_map.get(k)} for k, v in sorted(concept_counts.items(), key=lambda x: (-x[1], x[0]))[:20]]
    return {
        "reports": int(research.get("n_reports") or len(parsed) or 0),
        "parsed_reports": len(parsed),
        "decision_extracted_reports": decision_extracted_reports,
        "decision_extraction_rate": round(decision_extracted_reports / len(parsed), 3) if parsed else 0.0,
        "ocr": int(research.get("n_ocr") or 0),
        "concepts": top_concepts,
        "codes": code_map,
        "names": name_map,
        "leader_hits": list(research.get("leader_hits") or [])[:30],
        "chain_hits": list(research.get("chain_hits") or [])[:30],
        "all_concepts": list(research.get("all_concepts") or [])[:100],
        "all_codes_count": len(research.get("all_codes") or code_map),
        "all_companies_count": len(research.get("all_companies") or []),
        "name_attribution_count": len(name_map),
        "leader_active_concepts": _safe(research.get("leader_active_concepts"), 0),
    }


def _regulatory_evidence(inquiry: dict) -> dict:
    """Normalize cninfo/Eastmoney announcement tiers; tier1 is a hard execution veto."""
    by_code = {}
    for raw in inquiry.get("top_codes") or []:
        if not isinstance(raw, dict):
            continue
        code = str(raw.get("code") or "").zfill(6)
        if code:
            by_code[code] = {
                "code": code, "name": raw.get("name"), "count": _safe(raw.get("count"), 0),
                "tier1": _safe(raw.get("tier1"), 0), "latest": raw.get("latest"),
                "recent": list(raw.get("recent") or [])[:3],
            }
    return {
        "total_hits": _safe(inquiry.get("total_hits"), 0),
        "codes": _safe(inquiry.get("codes"), 0),
        "tier_counts": inquiry.get("tier_counts") or {},
        "by_code": by_code,
        "top_codes": list(by_code.values())[:20],
        "source": inquiry.get("source"),
    }


def _holder_evidence(holders: dict, date: str | None) -> dict:
    items = []
    for raw in holders.get("items") or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        as_of = str(item.get("as_of") or "")[:10]
        if date and as_of and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", as_of):
            try:
                item["age_days"] = max(0, (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(as_of, "%Y-%m-%d")).days)
            except ValueError:
                item["age_days"] = None
        else:
            item["age_days"] = None
        item["signal"] = "筹码变化待解释" if abs(float(item.get("change_pct") or 0)) >= 10 else "稳定/小幅变化"
        items.append(item)
    by_code = {str(x.get("code") or "").zfill(6): x for x in items if x.get("code")}
    return {"available": _safe(holders.get("available"), len(items)), "total": _safe(holders.get("total"), len(items)), "items": items, "by_code": by_code}


def _load_stock_industry() -> dict[str, dict[str, str]]:
    """Load the full A-share code-to-industry map used by decision gates."""
    path = MARKET / "sw_industry_map.parquet"
    try:
        import pandas as pd
        frame = pd.read_parquet(path, columns=["code", "name", "industry", "industry_code"])
        result = {}
        for row in frame.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            if code:
                result[code] = {
                    "industry": str(row.get("industry") or "").strip(),
                    "industry_code": str(row.get("industry_code") or "").strip(),
                    "mapped_name": str(row.get("name") or "").strip(),
                }
        return result
    except Exception:
        return {}


def _load_stock_concepts() -> dict[str, list[str]]:
    """Join concept membership with board names for per-stock short-term themes."""
    member_path = ROOT / "data_warehouse" / "classification" / "concept_member.parquet"
    board_path = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
    try:
        import pandas as pd
        members = pd.read_parquet(member_path, columns=["concept", "code"])
        boards = pd.read_parquet(board_path, columns=["board_code", "board_name"])
        names = (boards.dropna(subset=["board_code", "board_name"])
                 .drop_duplicates("board_code", keep="last")
                 .set_index("board_code")["board_name"].astype(str).to_dict())
        result: dict[str, list[str]] = {}
        for row in members.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            concept = str(row.get("concept") or "")
            name = names.get(concept)
            if code and name:
                result.setdefault(code, []).append(name)
        return {code: sorted(set(values)) for code, values in result.items()}
    except Exception:
        return {}


def _norm_industry(value: Any) -> str:
    text = re.sub(r"[ⅠⅡⅢIV\s]+$", "", str(value or "").strip())
    aliases = {"房地产开发": "房地产", "农产品加工": "农林牧渔", "汽车零部件": "汽车"}
    return aliases.get(text, text)


def _industry_matches(candidate: str, mainline: str) -> bool:
    left, right = _norm_industry(candidate), _norm_industry(mainline)
    if not left or not right:
        return False
    return left == right or (len(left) >= 3 and left in right) or (len(right) >= 3 and right in left)


def _board(code: str) -> str:
    code = str(code or "").zfill(6)
    if code.startswith(("600", "601", "603", "605")):
        return "沪主板"
    if code.startswith(("000", "001", "002", "003")):
        return "深主板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith("688"):
        return "科创板"
    if code.startswith(("8", "4", "92")):
        return "北交所"
    return "未知"


def _tradable(item: dict) -> dict:
    code = str(item.get("symbol") or item.get("code") or "")
    board = _board(code)
    name = str(item.get("name") or "")
    st = "st" in name.lower() or "*st" in name.lower()
    price = _safe(item.get("price"), 0) or 0
    reasons = []
    allowed = board in ("沪主板", "深主板") and not st and float(price or 0) > 0
    if board not in ("沪主板", "深主板"):
        reasons.append("非主板，仅作全市场观察")
    if st:
        reasons.append("ST/风险标的")
    if not price:
        reasons.append("缺少有效价格")
    return {"board": board, "trade_allowed": allowed, "trade_label": "可执行" if allowed else "观察", "trade_reasons": reasons}


def _mainline_score(row: dict) -> float:
    """Score structure, diffusion, momentum and *same-day* fund confirmation.

    A limit-up cluster is only structural evidence. Stale or negative fund flow
    must not turn it into a confirmed mainline.
    """
    zt = float(row.get("zt") or row.get("zt_cnt") or 0)
    board = float(row.get("max_board") or 0)
    pct = float(row.get("avg_pct") or row.get("change_pct") or row.get("pct_chg") or 0)
    diffusion = float(row.get("candidate_count") or 0)
    flow = float(row.get("main_net_yi") or row.get("net_yi") or 0)
    structure_score = min(30.0, zt * 6.0) + min(24.0, board * 8.0)
    diffusion_score = min(18.0, diffusion * 2.0)
    momentum_score = min(12.0, max(0.0, pct) * 2.0)
    flow_score = min(25.0, max(0.0, flow) * 0.25) if row.get("flow_fresh") else 0.0
    return round(min(100.0, structure_score + diffusion_score + momentum_score + flow_score), 1)


def _load_flow_values(date: str) -> tuple[dict[str, float], str | None]:
    """Load concept/theme flow from the concept source only.

    Use the explicit trading-day column for day selection. ``ts`` is only used
    to select the latest snapshot within that day, preventing a naive local
    timestamp from moving a record into the next trading day.
    """
    import pandas as pd
    candidates = [
        (MARKET / "concept_fund_flow_intraday.parquet", "intraday"),
        (MARKET / "sector_fund_flow.parquet", "legacy_daily"),
    ]
    for path, kind in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
            if kind == "intraday":
                if "type" in frame.columns:
                    frame = frame[frame["type"].astype(str).isin({"概念", "concept"})]
                name_col, value_col = "name", "main_net_yi"
                day_col = "date" if "date" in frame.columns else "ts"
                order_col = "ts" if "ts" in frame.columns else day_col
            else:
                name_col, value_col, day_col = "concept_name", "net_yi", "date"
                order_col = day_col
            if name_col not in frame.columns or value_col not in frame.columns or day_col not in frame.columns:
                continue
            frame = frame.copy()
            frame["_day"] = pd.to_datetime(frame[day_col], errors="coerce").dt.strftime("%Y-%m-%d")
            frame = frame[frame["_day"].notna() & (frame["_day"] <= date)]
            if frame.empty:
                continue
            latest_day = frame["_day"].max()
            day = frame[frame["_day"] == latest_day]
            if kind == "intraday" and order_col in day.columns:
                order = pd.to_datetime(day[order_col], errors="coerce")
                if order.notna().any():
                    day = day[order == order.max()]
            values = {str(row.get(name_col) or "").strip(): float(row.get(value_col) or 0)
                      for row in day.to_dict("records") if str(row.get(name_col) or "").strip()}
            if values:
                return values, latest_day
        except Exception:
            continue
    return {}, None


def _load_industry_flow_values(date: str) -> tuple[dict[str, float], str | None, str]:
    """Load real industry flow from the newest valid dedicated artifact."""
    import pandas as pd
    paths = [MARKET / "industry_fund_flow.parquet", MARKET / "sector_fund_flow_intraday.parquet"]
    errors = []
    for path in paths:
        if not path.exists():
            errors.append(f"{path.name}不存在")
            continue
        try:
            frame = pd.read_parquet(path)
            if frame.empty:
                errors.append(f"{path.name}为空")
                continue
            name_col = next((c for c in ("industry_name", "sector_name", "name") if c in frame.columns), None)
            value_col = next((c for c in ("net_yi", "main_net_yi", "main_net") if c in frame.columns), None)
            date_col = next((c for c in ("date", "trade_date", "ts") if c in frame.columns), None)
            if not name_col or not value_col or not date_col:
                errors.append(f"{path.name}字段不完整")
                continue
            if "type" in frame.columns:
                frame = frame[frame["type"].astype(str).isin({"行业", "industry"})]
            if frame.empty:
                errors.append(f"{path.name}没有行业类型记录")
                continue
            parsed = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            frame = frame.assign(_date=parsed)
            frame = frame[frame["_date"].notna() & (frame["_date"].dt.strftime("%Y-%m-%d") <= date)]
            if frame.empty:
                errors.append(f"{path.name}没有不晚于决策日{date}的记录")
                continue
            latest_day = frame["_date"].dt.strftime("%Y-%m-%d").max()
            day = frame[frame["_date"].dt.strftime("%Y-%m-%d") == latest_day]
            if "ts" in day.columns:
                timestamps = pd.to_datetime(day["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
                if timestamps.notna().any():
                    day = day[timestamps == timestamps.max()]
            values = {}
            for row in day.to_dict("records"):
                name = str(row.get(name_col) or "").strip()
                value = pd.to_numeric(row.get(value_col), errors="coerce")
                if not name or pd.isna(value):
                    continue
                number = float(value)
                if value_col == "main_net":
                    number /= 1e8
                values[name] = number
            if values:
                return values, latest_day, ""
            errors.append(f"{path.name}最新日期没有有效净流入记录")
        except Exception as exc:
            errors.append(f"{path.name}读取失败：{str(exc)[:100]}")
    return {}, None, "；".join(errors) or "行业资金数据不可用；概念资金不能替代行业资金"


def _load_feature_rows(date: str) -> dict[str, dict[str, Any]]:
    """Load the current full-market feature store row by stock code."""
    try:
        import pandas as pd
        path = ROOT / "data_warehouse" / "feature_store" / f"{date.replace('-', '')}.parquet"
        if not path.exists():
            return {}
        frame = pd.read_parquet(path)
        if "code" not in frame.columns:
            return {}
        frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
        return {str(row["code"]): {str(k): _safe(v) for k, v in row.items()}
                for row in frame.to_dict("records") if row.get("code")}
    except Exception:
        return {}


def _load_stock_flow_values(date: str) -> tuple[dict[str, dict[str, float]], str | None]:
    """Load the latest real per-stock main/super order net flow, never relabeling its date."""
    try:
        import pandas as pd
        frames = []
        for path in (MARKET / "fund_flow").glob("fund_flow_*.parquet"):
            frame = pd.read_parquet(path)
            if {"code", "date"}.issubset(frame.columns):
                frames.append(frame)
        if not frames:
            return {}, None
        frame = pd.concat(frames, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["date"].dt.strftime("%Y-%m-%d") <= date]
        if frame.empty:
            return {}, None
        latest = frame["date"].max()
        frame = frame[frame["date"] == latest].drop_duplicates("code", keep="last")
        values = {}
        for row in frame.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            if code:
                values[code] = {
                    "main_net_yi": round(float(row.get("main_net") or 0) / 1e8, 3),
                    "super_net_yi": round(float(row.get("super_net") or 0) / 1e8, 3),
                }
        return values, str(latest.date())
    except Exception:
        return {}, None


_LHB_FEATURE_CACHE: dict[str, dict[str, dict[str, float]]] = {}
# Public score contract is always 0-100. Raw points stay internal for audit.
_SHORT_SCORE_PUBLIC_MIN = 0.0
_SHORT_SCORE_PUBLIC_MAX = 100.0
_SHORT_SCORE_RAW_MAX = 135.0


def _load_lhb_features(date: str, lookback_days: int = 60) -> dict[str, dict[str, float]]:
    """Aggregate real 龙虎榜 evidence by stock before the decision date."""
    if date in _LHB_FEATURE_CACHE:
        return _LHB_FEATURE_CACHE[date]
    try:
        import pandas as pd
        frames = []
        for path in MARKET.glob("lhb_20*.parquet"):
            try:
                frame = pd.read_parquet(path, columns=["代码", "上榜日", "龙虎榜净买额", "解读"])
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            _LHB_FEATURE_CACHE[date] = {}
            return {}
        frame = pd.concat(frames, ignore_index=True)
        frame["代码"] = frame["代码"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
        frame["上榜日"] = pd.to_datetime(frame["上榜日"], errors="coerce")
        frame["龙虎榜净买额"] = pd.to_numeric(frame["龙虎榜净买额"], errors="coerce")
        end = pd.Timestamp(date)
        start = end - pd.Timedelta(days=lookback_days)
        frame = frame[(frame["上榜日"] >= start) & (frame["上榜日"] <= end)].dropna(subset=["代码", "上榜日"])
        out: dict[str, dict[str, float]] = {}
        for code, group in frame.groupby("代码"):
            group = group.sort_values("上榜日")
            net = group["龙虎榜净买额"].dropna()
            out[str(code)] = {
                "days": float(len(group)),
                "positive_days": float((net > 0).sum()),
                "net_yi": float(net.sum() / 1e8) if len(net) else 0.0,
                "latest_net_yi": float(net.iloc[-1] / 1e8) if len(net) else 0.0,
                "institution_days": float(group["解读"].astype(str).str.contains("机构", na=False).sum()),
            }
        _LHB_FEATURE_CACHE[date] = out
        return out
    except Exception:
        _LHB_FEATURE_CACHE[date] = {}
        return {}


def _load_zt_rows(date: str) -> dict[str, dict]:
    """Load only the requested day's limit-up/board evidence.

    A prior trading day's ladder may be shown as historical context, but it
    must not add points to today's short-term ranking.
    """
    try:
        import pandas as pd
        path = MARKET / "zt_pool_em_daily.parquet"
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["date"].dt.strftime("%Y-%m-%d") == date]
        if frame.empty:
            return {}
        return {str(r.get("code") or "").zfill(6): {k: _safe(v) for k, v in r.items()}
                for r in frame.to_dict("records") if r.get("code")}
    except Exception:
        return {}


def _load_concept_names() -> dict[str, list[str]]:
    """Load exact stock-to-concept membership names from the local taxonomy."""
    try:
        import pandas as pd
        member_path = ROOT / "data_warehouse" / "classification" / "concept_member.parquet"
        board_path = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
        members = pd.read_parquet(member_path, columns=["concept", "concept_name", "code"])
        board_names: dict[str, str] = {}
        if board_path.exists():
            boards = pd.read_parquet(board_path, columns=["board_code", "board_name"])
            board_names = {
                str(row.get("board_code") or ""): str(row.get("board_name") or "").strip()
                for row in boards.to_dict("records")
                if row.get("board_code") and row.get("board_name")
            }
        result: dict[str, set[str]] = {}
        for row in members.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            concept_id = str(row.get("concept") or "").strip()
            label = str(row.get("concept_name") or "").strip()
            label = board_names.get(concept_id, label)
            if code and label and _is_specific_theme(label):
                result.setdefault(code, set()).add(label)
        return {code: sorted(labels) for code, labels in result.items()}
    except Exception:
        return {}


def _concept_mainline_rows(date: str) -> list[dict]:
    """Aggregate same-day limit-up diffusion by concept, never by SW industry."""
    try:
        import pandas as pd
        path = MARKET / "zt_pool_em_daily.parquet"
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        # Do not combine an older diffusion snapshot with today's fund flow.
        frame = frame[frame["date"].dt.strftime("%Y-%m-%d") <= date]
        if frame.empty:
            return []
        latest = frame["date"].max()
        frame = frame[frame["date"] == latest]
        source_date = latest.strftime("%Y-%m-%d")
        if "is_zt" in frame.columns:
            frame = frame[frame["is_zt"].fillna(False).astype(bool)]
        concept_map = _load_concept_names()
        grouped: dict[str, dict] = {}
        for row in frame.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            for concept in concept_map.get(code, []):
                item = grouped.setdefault(concept, {
                    "name": concept, "concept": concept, "taxonomy": "concept",
                    "source": "zt_pool_concept_diffusion", "codes": set(),
                    "structure_as_of": source_date,
                    "zt": 0, "max_board": 0, "amount": 0.0, "pct_sum": 0.0,
                })
                if code not in item["codes"]:
                    item["codes"].add(code)
                    item["zt"] += 1
                    item["amount"] += float(row.get("amount") or 0)
                    item["pct_sum"] += float(row.get("pct_chg") or 0)
                item["max_board"] = max(item["max_board"], int(float(row.get("board_count") or 0)))
        rows = []
        for item in grouped.values():
            count = len(item.pop("codes"))
            if count < 2:
                continue
            item["candidate_count"] = count
            item["avg_pct"] = round(item.pop("pct_sum") / count, 2)
            item["amount"] = round(item["amount"], 2)
            rows.append(item)
        return rows
    except Exception:
        return []


def _is_specific_theme(name: str) -> bool:
    text = str(name or "").strip()
    generic = {"融资融券", "沪股通", "深股通", "MSCI中国", "标准普尔", "富时罗素", "证金持股",
               "机构重仓", "基金重仓", "养老金", "QFII重仓", "IPO受益", "昨日涨停", "昨日连板",
               "小盘股", "中盘股", "大盘股", "微盘股", "小盘成长", "中盘成长", "大盘成长", "创业板综", "创业成份",
               "参股银行", "参股券商", "央企改革", "国企改革", "AB股", "破增发价股", "低价股", "低价",
               "破发股", "昨日高振幅", "最近多板", "东方财富热股", "反转股", "活跃股", "高换手",
               "高市净率", "低市净率", "高市盈率", "低市盈率", "百元股", "高贝塔", "低贝塔",
               "绩优股", "绩差股", "涨停股", "跌停股", "破净股", "高股息", "次新股"}
    if not text or text in generic:
        return False
    if re.search(r"^(HS|中证|上证|深证|国证)\d|^20\d{2}中报", text):
        return False
    return True


def _build_realtime_concept_rows(rows: list[dict], concept_map: dict[str, list[str]], date: str) -> list[dict]:
    """Build current concept/theme structure from the full quote universe.

    This is independent of the limit-up archive. It keeps today's live concept
    diffusion visible even when the daily limit-up or concept-board refresh is
    one trading day behind.
    """
    stats: dict[str, dict[str, float]] = {}
    for raw in rows:
        code = str(raw.get("symbol") or raw.get("code") or "").zfill(6)
        concepts = raw.get("concepts") or concept_map.get(code, [])
        pct = float(raw.get("change_pct") or raw.get("pct_chg") or 0)
        amount = float(raw.get("amount") or raw.get("amount_wan") or 0)
        if amount and amount < 1e7:
            amount *= 10000
        for concept in set(str(x).strip() for x in concepts if str(x).strip()):
            if not _is_specific_theme(concept):
                continue
            item = stats.setdefault(concept, {"candidate_count": 0, "amount": 0.0, "pct_sum": 0.0, "positive_count": 0})
            item["candidate_count"] += 1
            item["amount"] += amount
            item["pct_sum"] += pct
            item["positive_count"] += 1 if pct > 0 else 0
    out = []
    for concept, item in stats.items():
        count = int(item["candidate_count"])
        avg_pct = item["pct_sum"] / max(count, 1)
        if count < 5 or item["amount"] < 5e8 or avg_pct <= 0:
            continue
        diffusion = min(45.0, count * 2.5) + min(25.0, max(0.0, avg_pct) * 3.0)
        participation = item["positive_count"] / max(count, 1) * 20.0
        out.append({"name": concept, "concept": concept, "direction": concept,
                    "taxonomy": "concept", "taxonomy_label": "概念/题材",
                    "source": "realtime_concept_diffusion", "structure_as_of": date,
                    "candidate_count": count, "amount": round(item["amount"], 2),
                    "avg_pct": round(avg_pct, 2), "score": round(min(100.0, diffusion + participation), 1)})
    return sorted(out, key=lambda x: x["score"], reverse=True)[:30]


def _build_intraday_themes(opportunities: list[dict], flow_values: dict[str, float], structure_date: str) -> list[dict]:
    """Find concrete themes with same-day stock diffusion and positive fund flow."""
    stats: dict[str, dict[str, float]] = {}
    for item in opportunities:
        amount = float(item.get("amount") or 0)
        pct = float(item.get("change_pct") or 0)
        for theme in set(item.get("concepts") or []):
            if not _is_specific_theme(theme) or float(flow_values.get(theme, 0)) <= 0:
                continue
            row = stats.setdefault(theme, {"candidate_count": 0, "amount": 0.0, "pct_sum": 0.0})
            row["candidate_count"] += 1
            row["amount"] += amount
            row["pct_sum"] += pct
    themes = []
    for name, row in stats.items():
        count = int(row["candidate_count"])
        if count < 2 or row["amount"] < 2e8:
            continue
        flow = float(flow_values[name])
        avg_pct = row["pct_sum"] / count
        score = min(100.0, flow * 0.18 + count * 5 + min(20, row["amount"] / 1e9 * 2) + max(0, avg_pct) * 2)
        themes.append({"name": name, "industry": name, "source": "intraday_theme_diffusion",
                       "structure_as_of": structure_date,
                       "candidate_count": count, "amount": round(row["amount"], 2),
                       "avg_pct": round(avg_pct, 2), "net_yi": flow, "score": round(score, 1)})
    return sorted(themes, key=lambda x: x["score"], reverse=True)[:20]


def _short_score_detail(item: dict, mainline: dict | None, flow_value: float | None,
                        zt_row: dict | None, market_force: float, breadth: float,
                        *, flow_reason: str | None = None,
                        stock_flow_value: float | None = None,
                        lhb_feature: dict | None = None,
                        research_item: dict | None = None) -> tuple[float, list[str], list[str], dict]:
    """Calculate a comparable multi-factor short-term ranking score.

    Mainline and industry flow are factors, not score switches. Missing data
    contributes zero for that factor and is recorded in the breakdown; the
    remaining price, liquidity, momentum, ladder, and market factors still
    determine the rank.
    """
    score, reasons, blockers, breakdown = 0.0, [], [], {}
    pct = float(item.get("change_pct") or 0)
    amount = float(item.get("amount") or 0)
    turnover = float(item.get("turnover") or 0)

    if 2 <= pct < 9.5:
        points = 14.0; reasons.append(f"涨幅{pct:+.1f}%且未封板")
        breakdown["涨幅强度"] = {"points": points, "max": 14, "status": "有效"}
    elif pct >= 9.5:
        points = 0.0; blockers.append("接近涨停，追涨成交风险高")
        breakdown["涨幅强度"] = {"points": points, "max": 14, "status": "过热不加分"}
    else:
        points = 0.0; blockers.append("当日未形成短线强势")
        breakdown["涨幅强度"] = {"points": points, "max": 14, "status": "偏弱"}
    score += points

    if amount >= 3e8:
        points = 12.0; reasons.append(f"成交额{amount / 1e8:.1f}亿")
    elif amount >= 1e8:
        points = 7.0; reasons.append(f"成交额{amount / 1e8:.1f}亿")
    else:
        points = 0.0; blockers.append("成交额不足")
    breakdown["成交额"] = {"points": points, "max": 12, "status": "有效" if points else "不足"}
    score += points

    if 3 <= turnover <= 18:
        points = 10.0; reasons.append(f"换手{turnover:.1f}%处于可交易区间")
    elif turnover > 25:
        points = 0.0; blockers.append(f"换手过热{turnover:.1f}%")
    else:
        points = 0.0; blockers.append("换手未进入有效区间")
    breakdown["换手结构"] = {"points": points, "max": 10, "status": "有效" if points else "未加分"}
    score += points

    momentum = float(item.get("composite_conf") or 0)
    momentum_20d = float(item.get("mom20") or item.get("momentum_20d") or 0)
    volume_ratio = float(item.get("vol_ratio20") or item.get("volume_ratio_20d") or 0)
    momentum_points = min(12.0, max(0.0, momentum) * 1.2 + max(0.0, momentum_20d) * 12.0 + max(0.0, volume_ratio - 1.0) * 2.0)
    if momentum_points:
        reasons.append(f"技术动量{momentum:.1f}")
    else:
        blockers.append("技术动量暂无有效贡献")
    breakdown["技术动量"] = {"points": round(momentum_points, 1), "max": 12, "status": "有效" if momentum_points else "暂无"}
    score += momentum_points

    if mainline:
        mainline_points = min(20.0, 6.0 + float(mainline.get("structure_score") or mainline.get("score") or 0) * 0.16)
        structure_status = mainline.get("structure_status") or mainline.get("level") or "结构参考"
        reasons.append(f"概念/题材:{mainline.get('concept') or mainline.get('name')}（{structure_status}）")
        breakdown["概念/题材结构"] = {"points": round(mainline_points, 1), "max": 20, "status": structure_status}
    else:
        mainline_points = 0.0
        concepts = [str(value).strip() for value in (item.get("concepts") or []) if str(value).strip()]
        blockers.append("该股没有匹配到概念/题材结构" if not concepts else "概念/题材有归属，但尚未进入主线结构")
        breakdown["概念/题材结构"] = {"points": 0.0, "max": 20, "status": "未匹配" if not concepts else "待确认"}
    score += mainline_points

    if flow_value is not None and flow_value > 0:
        flow_points = min(18.0, 8.0 + flow_value * 0.12)
        reasons.append(f"行业资金净流入{flow_value:+.1f}亿")
        flow_status = "当日正向"
    elif flow_value is not None:
        flow_points = 0.0
        blockers.append(f"行业资金净流出{flow_value:+.1f}亿")
        flow_status = "净流出"
    else:
        flow_points = 0.0
        blockers.append(flow_reason or "行业资金暂未采集，本轮不计分（不是利空判断）")
        flow_status = "缺失·不计分"
    breakdown["行业资金"] = {"points": round(flow_points, 1), "max": 18, "status": flow_status}
    score += flow_points

    ladder_points = 0.0
    if zt_row:
        boards = int(float(zt_row.get("board_count") or 0))
        zb = bool(zt_row.get("is_zb"))
        if boards >= 2:
            ladder_points = min(14.0, boards * 4.0); reasons.append(f"涨停梯队{boards}板")
        elif zt_row.get("is_zt"):
            ladder_points = 4.0; reasons.append("今日涨停")
        if zb:
            blockers.append("曾炸板，等待回封确认")
    else:
        blockers.append("没有对应涨停梯队记录")
    breakdown["涨停梯队"] = {"points": ladder_points, "max": 14, "status": "有效" if ladder_points else "暂无"}
    score += ladder_points

    if market_force >= 35:
        market_points = 5.0; reasons.append(f"市场资金合力{market_force:.0f}")
    else:
        market_points = 0.0; blockers.append(f"市场资金合力偏弱{market_force:.0f}")
    if breadth >= 0.5:
        breadth_points = 3.0; reasons.append(f"市场宽度{breadth:.0%}")
    else:
        breadth_points = 0.0; blockers.append(f"市场宽度偏弱{breadth:.0%}")
    breakdown["市场环境"] = {"points": market_points + breadth_points, "max": 8, "status": "偏强" if market_points + breadth_points else "偏弱"}
    score += market_points + breadth_points

    # Additional short-term evidence is additive and explicitly attributed.
    stock_flow_points = 0.0
    if stock_flow_value is not None:
        if stock_flow_value > 0:
            stock_flow_points = min(8.0, 2.0 + stock_flow_value * 0.18)
            reasons.append(f"个股主力净流入{stock_flow_value:+.1f}亿")
        elif stock_flow_value < 0:
            stock_flow_points = max(-6.0, stock_flow_value * 0.12)
            blockers.append(f"个股主力净流出{stock_flow_value:+.1f}亿")
        else:
            blockers.append("个股主力资金为零，未加分")
    else:
        blockers.append("个股主力资金暂无当日记录，本轮不计分（不是利空判断）")
    breakdown["个股主力资金"] = {"points": round(stock_flow_points, 1), "max": 8, "status": "有效" if stock_flow_value is not None else "缺失·不计分"}
    score += stock_flow_points

    lhb_points = 0.0
    if lhb_feature:
        lhb_points = min(7.0, float(lhb_feature.get("positive_days") or 0) * 1.5 + float(lhb_feature.get("institution_days") or 0) * 1.0)
        if lhb_points:
            reasons.append(f"龙虎榜近60日{int(lhb_feature.get('days') or 0)}次，净买{int(lhb_feature.get('positive_days') or 0)}次")
        if float(lhb_feature.get("net_yi") or 0) < 0:
            lhb_points = max(-5.0, lhb_points - 2.0)
            blockers.append("龙虎榜近60日累计净卖出")
    else:
        blockers.append("近60日无龙虎榜记录")
    breakdown["龙虎榜"] = {"points": round(lhb_points, 1), "max": 7, "status": "有记录" if lhb_feature else "无记录"}
    score += lhb_points

    research_item = research_item or {}
    research_points = min(5.0, float(research_item.get("decision_extracted_reports") or 0) * 1.5)
    if research_points:
        reasons.append(f"研报/催化证据{int(research_item.get('decision_extracted_reports') or 0)}条")
    else:
        blockers.append("暂无可用于短线判断的研报/催化证据，本轮不计分（不是利空判断）")
    breakdown["研报与催化"] = {"points": round(research_points, 1), "max": 5, "status": "有效" if research_points else "暂无·不计分"}
    score += research_points

    # Valuation/financial context is a small short-term tie-breaker, not a
    # long-term quality score. It is still included so the displayed total can
    # explain why two similarly strong price moves rank differently.
    pe = item.get("pe") if item.get("pe") is not None else item.get("pe_ttm")
    pb = item.get("pb")
    valuation_points = 0.0
    try:
        pe_value, pb_value = float(pe), float(pb)
        if 0 < pe_value <= 20 and 0 < pb_value <= 2.5:
            valuation_points = 5.0; reasons.append(f"估值适中PE{pe_value:.1f}/PB{pb_value:.1f}")
        elif 0 < pe_value <= 35 and 0 < pb_value <= 4:
            valuation_points = 2.0; reasons.append(f"估值中性PE{pe_value:.1f}/PB{pb_value:.1f}")
        else:
            blockers.append("估值未提供短线加分")
    except (TypeError, ValueError):
        blockers.append("估值数据缺失，估值因子未加分")
    breakdown["估值/财务"] = {"points": valuation_points, "max": 5, "status": "有效" if valuation_points else "缺失或中性"}
    score += valuation_points

    risk = item.get("regulatory_risk") or {}
    tier1 = float(risk.get("tier1") or 0)
    event_points = -min(8.0, tier1 * 2.0)
    if tier1:
        blockers.append(f"监管硬风险{int(tier1)}条，短线风险扣分")
    elif risk:
        event_points = -min(3.0, float(risk.get("count") or 0) * 0.2)
        if event_points < 0:
            blockers.append(f"监管记录{int(float(risk.get('count') or 0))}条，风险扣分")
    else:
        blockers.append("公告/监管数据暂无记录")
    breakdown["公告与监管"] = {"points": event_points, "max": 0, "floor": -8, "status": "有风险记录" if event_points < 0 else "无硬风险"}
    score += event_points

    holder = item.get("holder_evidence") or {}
    holder_change = abs(float(holder.get("change_pct") or 0))
    holder_points = -min(2.0, holder_change / 5.0) if holder_change >= 10 else 0.0
    if holder_points:
        blockers.append(f"股东户数变化{holder.get('change_pct')}%，筹码扰动扣分")
    breakdown["股东筹码"] = {"points": round(holder_points, 1), "max": 2, "status": "筹码扰动" if holder_points else "稳定或暂无"}
    score += holder_points

    raw_score = round(score, 1)
    max_raw_score = sum(float(detail.get("max") or 0) for detail in breakdown.values()) or _SHORT_SCORE_RAW_MAX
    normalized_score = round(min(_SHORT_SCORE_PUBLIC_MAX, max(_SHORT_SCORE_PUBLIC_MIN, score / max_raw_score * _SHORT_SCORE_PUBLIC_MAX)), 1)
    for detail in breakdown.values():
        maximum = float(detail.get("max") or 0)
        points = float(detail.get("points") or 0)
        detail["weight_pct"] = round(maximum / max_raw_score * 100.0, 1) if maximum else 0.0
        detail["normalized_points"] = round(max(0.0, points) / max_raw_score * 100.0, 1)
    breakdown["_meta"] = {
        "raw_score": raw_score,
        "max_raw_score": max_raw_score,
        "normalized_score": normalized_score,
        "normalization": "raw_points / sum(factor max points) * 100; negative points floor final score at 0",
    }
    return normalized_score, reasons, blockers, breakdown


def _short_score(item: dict, mainline: dict | None, flow_value: float | None,
                 zt_row: dict | None, market_force: float, breadth: float,
                 *, strict: bool = False, flow_reason: str | None = None) -> tuple[float | None, list[str], list[str]]:
    """Compatibility wrapper; strict mode rejects missing key flow evidence."""
    score, reasons, blockers, _ = _short_score_detail(
        item, mainline, flow_value, zt_row, market_force, breadth, flow_reason=flow_reason
    )
    if strict and flow_value is None:
        return None, [], list(dict.fromkeys(blockers + ["关键行业资金缺失，严格模式不评分"]))
    return score, reasons, blockers


def _observation_rank_score(item: dict, zt_row: dict | None) -> tuple[float, list[str]]:
    """Rank observable short-term strength without calling it executable."""
    score, reasons = 0.0, []
    pct = float(item.get("change_pct") or 0)
    amount = float(item.get("amount") or 0)
    turnover = float(item.get("turnover") or 0)
    momentum = float(item.get("composite_conf") or 0)
    if pct > 0:
        score += min(24.0, pct * 2.0); reasons.append(f"涨幅{pct:+.1f}%")
    if amount >= 3e8:
        score += 20.0; reasons.append(f"成交额{amount / 1e8:.1f}亿")
    elif amount >= 1e8:
        score += 12.0; reasons.append(f"成交额{amount / 1e8:.1f}亿")
    if 3 <= turnover <= 18:
        score += 18.0; reasons.append(f"换手{turnover:.1f}%适中")
    elif 18 < turnover <= 25:
        score += 9.0; reasons.append(f"换手{turnover:.1f}%偏高")
    elif turnover > 25:
        score -= 4.0; reasons.append(f"换手{turnover:.1f}%过热")
    if momentum > 0:
        score += min(16.0, momentum * 2.0); reasons.append("技术动量")
    if zt_row:
        boards = int(float(zt_row.get("board_count") or 0))
        if boards >= 2:
            score += min(18.0, boards * 4.0); reasons.append(f"涨停梯队{boards}板")
        elif zt_row.get("is_zt"):
            score += 5.0; reasons.append("今日涨停")
    return round(max(0.0, min(100.0, score)), 1), reasons


def build_snapshot(mode: str = "intraday", date: str | None = None,
                   live_rows: list[dict] | None = None) -> dict:
    """Build a decision snapshot; ``live_rows`` forces the current full universe."""
    if date is None and mode == "intraday":
        date = _path_date(_latest("intraday_chain_*.json"))
    elif date is None and mode == "battle_map":
        date = _path_date(_latest("battle_map_*.json"))
    date = date or _snapshot_date() or datetime.now(CST).strftime("%Y-%m-%d")
    fusion = _load_fusion(date)
    fund = _load_fund(date)
    fusion_date = str(_safe(fusion.get("date")) or "")[:10] or None
    fund_date = str(_safe(fund.get("date")) or "")[:10] or None
    chain = _load_chain(mode, date)
    source_dates = dict(chain.get("_source_dates") or {})
    intraday_coverage = chain.get("intraday_coverage") or chain.get("coverage") or {}
    coverage_status = str(intraday_coverage.get("status") or "missing")
    bias = chain.get("bias") or {}
    research_raw, research_date = _load_research(date)
    research = _research_evidence(research_raw)
    holders = _holder_evidence(_load_holders(), date)
    regulatory = _regulatory_evidence(_load_inquiry())
    factor_decay, factor_date = _load_factor_decay(date)
    industry_map = _load_stock_industry()
    concept_map = _load_stock_concepts()
    zt = {}
    try:
        import pandas as pd
        p = MARKET / "zt_daily_stats.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df = df[pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") <= date]
            if not df.empty:
                zt = {k: _safe(v) for k, v in df.iloc[-1].to_dict().items()}
    except Exception:
        pass

    flow = {}
    try:
        p = GEN / f"research_fusion_snapshot_{date}.json"
        flow = _read_json(p) if p.exists() else _read_json(_latest("research_fusion_snapshot_*.json", date) or Path(""))
    except Exception:
        flow = {}
    money = flow.get("money_flow") or {}
    sector = flow.get("sector_rotation") or {}
    etf = flow.get("etf_activity") or {}
    concept_flow_values, concept_flow_date = _load_flow_values(date)
    concept_flow_fresh = concept_flow_date == date
    industry_flow_values, industry_flow_date, industry_flow_error = _load_industry_flow_values(date)
    industry_flow_fresh = industry_flow_date == date
    stock_flow_values, stock_flow_date = _load_stock_flow_values(date)
    stock_flow_fresh = stock_flow_date == date
    lhb_features = _load_lhb_features(date)
    feature_rows = _load_feature_rows(date)

    raw_mainlines: list[dict] = []
    # Keep only explicitly tagged concept directions from old chain artifacts.
    # Untagged rows are discarded because their `industry` field may be SW
    # second-level classification and is not a mainline conclusion.
    # A request date is never evidence of structure. Use only artifact dates or
    # an explicit row date; legacy rows without either remain undated.
    chain_source_dates = chain.get("_source_dates") or {}
    chain_structure_date = chain_source_dates.get("intraday_chain")
    # A live full-universe round is authoritative for the current concept
    # structure. Do not merge its rows with stale mainlines from an older chain.
    chain_mainline_rows = [] if live_rows is not None else (chain.get("mainlines") or chain.get("directions") or [])
    for row in chain_mainline_rows:
        if isinstance(row, dict) and str(row.get("taxonomy") or "").lower() in {"concept", "theme"}:
            item = dict(row)
            # Legacy rows often omit a structure date. Do not infer it from
            # the requested date or an unrelated after-close artifact.
            if chain_structure_date and not item.get("structure_as_of"):
                item["structure_as_of"] = chain_structure_date
            raw_mainlines.append(item)
    # Rebuild concept/theme structure from every current quote row. The old
    # chain often has no mainlines and the limit-up archive may lag by a day;
    # neither condition should erase today's concept universe.
    chain_rows = live_rows if live_rows is not None else (chain.get("candidate_rows") or chain.get("rows") or [])
    realtime_structure_date = date if live_rows is not None else chain_structure_date
    if realtime_structure_date:
        raw_mainlines.extend(_build_realtime_concept_rows(chain_rows, concept_map, realtime_structure_date))
    # Keep limit-up diffusion as an additional concept structure source.
    raw_mainlines.extend(_concept_mainline_rows(date))

    # Merge duplicate labels before scoring; overlapping aliases no longer
    # receive multiple independent places in the mainline ranking.
    merged_mainlines: dict[str, dict] = {}
    for raw in raw_mainlines:
        name = str(raw.get("industry") or raw.get("name") or "").strip()
        if not name or not _is_specific_theme(name):
            continue
        key = _norm_industry(name)
        item = merged_mainlines.setdefault(key, {
            "name": name, "concept": name, "direction": name,
            "taxonomy": "concept", "sources": [],
        })
        item["zt"] = max(int(float(item.get("zt") or 0)), int(float(raw.get("zt") or raw.get("zt_cnt") or 0)))
        item["max_board"] = max(int(float(item.get("max_board") or 0)), int(float(raw.get("max_board") or 0)))
        item["candidate_count"] = max(int(float(item.get("candidate_count") or 0)), int(float(raw.get("candidate_count") or 0)))
        item["avg_pct"] = raw.get("avg_pct", item.get("avg_pct"))
        raw_date = str(raw.get("structure_as_of") or "")[:10]
        current_date = str(item.get("structure_as_of") or "")[:10]
        if raw_date and (not current_date or raw_date > current_date):
            item["structure_as_of"] = raw_date
        source = str(raw.get("source") or "intraday_chain")
        if source not in item["sources"]:
            item["sources"].append(source)

    mainlines = []
    for item in merged_mainlines.values():
        name = str(item.get("concept") or item.get("direction") or item.get("name") or "")
        matches = [(flow_name, value) for flow_name, value in concept_flow_values.items()
                   if _is_specific_theme(flow_name) and _industry_matches(name, flow_name)]
        flow_name, flow_value = max(matches, key=lambda pair: pair[1]) if matches else (None, None)
        item["flow_name"] = flow_name
        item["net_yi"] = flow_value
        item["flow_as_of"] = concept_flow_date
        item["industry"] = None
        item["taxonomy_label"] = "概念/题材"
        item["flow_fresh"] = concept_flow_fresh
        item["industry_flow_as_of"] = industry_flow_date
        item["industry_flow_fresh"] = industry_flow_fresh
        structure_current = str(item.get("structure_as_of") or "")[:10] == date
        structure_ok = structure_current and (int(item.get("zt") or 0) >= 4 or int(item.get("max_board") or 0) >= 3)
        diffusion_ok = structure_current and int(item.get("candidate_count") or 0) >= 3 and float(item.get("avg_pct") or 0) > 0
        if concept_flow_fresh and flow_value is not None and flow_value > 0 and (structure_ok or diffusion_ok):
            item["level"] = "主线"
            item["confirmation"] = "同日资金+扩散/梯队确认"
        elif structure_ok or diffusion_ok:
            item["level"] = "待资金确认"
            item["confirmation"] = "结构成立，但缺少同日正资金"
        else:
            item["level"] = "观察"
            item["confirmation"] = "扩散或梯队不足"
        item["structure_score"] = _mainline_score(item)
        item["score"] = item["structure_score"] if item["level"] == "主线" else None
        item["score_status"] = "可比较" if item["level"] == "主线" else "待确认"
        item["structure_status"] = "当日结构" if item.get("structure_as_of") == date else f"历史结构参考（截至{item.get('structure_as_of') or '未知日期'}）"
        mainlines.append(item)
    mainlines.sort(key=lambda x: (x.get("level") == "主线", float(x.get("score") or 0)), reverse=True)

    opportunities = []
    raw_rows = live_rows if live_rows is not None else (chain.get("decision_rows") or chain.get("candidate_rows") or chain.get("rows") or chain.get("opportunities") or [])
    if not raw_rows and mode == "after_close":
        # 盘后额外产物没有统一 rows，低估池和龙虎榜个股仍是可审计候选来源。
        raw_rows = list(chain.get("value_picks") or [])
        raw_rows.extend(chain.get("lhb", {}).get("stock_top") or [])
    seen_codes: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        code = str(item.get("symbol") or item.get("code") or "").zfill(6)
        if code in seen_codes:
            continue
        seen_codes.add(code)
        item["code"] = code
        feature = feature_rows.get(code, {})
        for key in ("peTTM", "pbMRQ", "mom20", "mom60", "mom120", "amp20", "vol_ratio20"):
            if feature.get(key) is not None:
                item[key] = feature[key]
        if feature.get("mom20") is not None:
            item["momentum_20d"] = feature["mom20"]
        if feature.get("vol_ratio20") is not None:
            item["volume_ratio_20d"] = feature["vol_ratio20"]
        mapped = industry_map.get(code, {})
        item["industry"] = item.get("industry") or item.get("sector") or mapped.get("industry")
        item["industry_code"] = mapped.get("industry_code")
        item["concepts"] = item.get("concepts") or concept_map.get(code, [])[:20]
        item.update(_tradable(item))
        research_item = dict(research.get("codes", {}).get(code) or {})
        research_item.setdefault("code", code)
        research_item.setdefault("reports", 0)
        if research_item.get("reports"):
            research_item["evidence_level"] = "A"
            research_item["attribution"] = "股票代码精确命中"
        else:
            company_name = str(item.get("name") or "").strip()
            name_evidence = dict(research.get("names", {}).get(company_name) or {})
            if not name_evidence and company_name:
                name_evidence = next((dict(v) for k, v in research.get("names", {}).items() if k in company_name or company_name in k), {})
            if name_evidence:
                research_item = {**name_evidence, "code": code, "evidence_level": "B", "attribution": "公司名/简称命中，未发现代码精确命中", "code_exact": False}
            else:
                research_item["evidence_level"] = "C"
                research_item["attribution"] = "仅主题/行业证据，未归因到该股票"
        inquiry_item = dict(regulatory.get("by_code", {}).get(code) or {})
        holder_item = dict(holders.get("by_code", {}).get(code) or {})
        item["research_evidence"] = research_item
        item["regulatory_risk"] = inquiry_item
        item["holder_evidence"] = holder_item
        item["hard_risk"] = bool(float(inquiry_item.get("tier1") or 0) > 0)
        angles = item.get("angles") or {}
        positive = sum(1 for x in angles.values() if isinstance(x, dict) and x.get("judge") in ("偏多", "安全", "强"))
        score_base = float(item.get("composite_conf") or item.get("score") or 0)
        scaled_base = score_base if score_base > 10 else score_base * 10
        # Pure code/title matches are coverage, not alpha. Only extracted decision fields earn evidence points.
        extracted_reports = float(research_item.get("decision_extracted_reports") or 0) if research_item.get("evidence_level") == "A" else 0.0
        research_bonus = min(12, extracted_reports * 3)
        if research_item.get("rating"):
            research_bonus += 2
        if research_item.get("target_price") is not None:
            research_bonus += 2
        research_bonus = min(16, research_bonus)
        holder_change = abs(float(holder_item.get("change_pct") or 0))
        item["decision_score"] = round(min(100, scaled_base + positive * 8 + research_bonus), 1)
        item["mainline_match"] = None
        item["mainline_score"] = 0
        item["decision"] = "观察/等待确认"
        if item["hard_risk"]:
            item["trade_allowed"] = False
            item["trade_label"] = "风险拦截"
            item.setdefault("trade_reasons", []).append(f"监管硬风险：{inquiry_item.get('tier1')} 条")
            item["decision"] = "禁止交易/仅作风险观察"
        elif inquiry_item:
            item.setdefault("trade_reasons", []).append(f"监管记录 {inquiry_item.get('count', 0)} 条，需复核公告")
        if holder_change >= 10:
            item.setdefault("trade_reasons", []).append(f"股东户数变化 {holder_item.get('change_pct'):+.1f}% ，筹码信号待解释")
        opportunities.append(item)
    # Concept membership is context only. It is not a confirmed mainline and
    # must not add points or grant execution access by itself.
    for item in opportunities:
        item["concept_context"] = [str(value).strip() for value in (item.get("concepts") or []) if str(value).strip()][:8]
        item["mainline_match"] = None
        item["mainline_match_type"] = None
        item["mainline_score"] = 0
    risk_temp = float(fusion.get("temperature") or 50)
    force = float(fund.get("force_index") or fusion.get("force_index") or 0)
    breadth = float(bias.get("breadth") or 0.5)
    if concept_flow_fresh:
        intraday_themes = _build_intraday_themes(opportunities, concept_flow_values, chain_structure_date)
        existing_names = {_norm_industry(m.get("concept") or m.get("name")) for m in mainlines}
        for theme in intraday_themes:
            if _norm_industry(theme.get("concept") or theme.get("name")) in existing_names:
                continue
            theme["flow_as_of"] = concept_flow_date
            theme["flow_fresh"] = True
            theme["taxonomy"] = "concept"
            theme["taxonomy_label"] = "概念/题材"
            theme["concept"] = theme.get("name")
            theme["direction"] = theme.get("name")
            theme["industry"] = None
            theme["level"] = "主线"
            theme["confirmation"] = "同日资金+候选扩散确认"
            theme["structure_score"] = _mainline_score(theme)
            theme["score"] = theme["structure_score"]
            theme["score_status"] = "可比较"
            mainlines.append(theme)
    mainlines.sort(key=lambda x: (x.get("level") == "主线", float(x.get("score") or 0)), reverse=True)
    confirmed_mainlines = [m for m in mainlines
                           if m.get("level") == "主线"
                           and str(m.get("structure_as_of") or "")[:10] == date
                           and m.get("flow_fresh") is True]
    zt_rows = _load_zt_rows(date)
    for item in opportunities:
        candidate_industry = str(item.get("industry") or "")
        concepts = item.get("concepts") or []
        # A concept/theme structure is usable for ranking even when its
        # same-day confirmation is pending. Only the execution decision keeps
        # the stricter confirmed-mainline requirement.
        mainline = next((m for m in mainlines
                         if str(m.get("structure_as_of") or "")[:10] == date and any(
                             _industry_matches(concept, str(m.get("concept") or m.get("name") or ""))
                             for concept in concepts
                         )), None)
        generic_themes = {"融资融券", "沪股通", "深股通", "MSCI中国", "标准普尔", "富时罗素", "证金持股", "机构重仓", "基金重仓"}
        # Short-term execution requires an industry flow record. Concept flow
        # remains a separate mainline signal and cannot satisfy this gate.
        industry_name = candidate_industry.strip()
        flow_matches = [(name, value) for name, value in industry_flow_values.items()
                        if name not in generic_themes and industry_name and _industry_matches(industry_name, name)]
        best_flow_name, best_flow = max(flow_matches, key=lambda pair: pair[1]) if industry_flow_fresh and flow_matches else (None, None)
        stock_flow = dict(stock_flow_values.get(str(item.get("code") or "").zfill(6)) or {})
        item["stock_fund_flow"] = {
            **stock_flow,
            "as_of": stock_flow_date,
            "fresh": stock_flow_fresh,
            "status": "同日真实主力流" if stock_flow_fresh and stock_flow else ("历史真实主力流，仅供参考" if stock_flow else "无逐股主力流数据"),
        }
        amount = float(item.get("amount") or 0)
        pct = float(item.get("change_pct") or 0)
        item["amount_yi"] = round(amount / 1e8, 2)
        item["active_flow_proxy_yi"] = round(amount / 1e8 * (1 if pct > 0 else -1 if pct < 0 else 0), 2)
        item["active_flow_proxy_note"] = "当日成交额×涨跌方向代理，不等于主力净流入"
        if mainline:
            item["mainline_match"] = mainline.get("industry") or mainline.get("name")
            item["mainline_score"] = mainline.get("score", 0)
        zt_row = zt_rows.get(item.get("code"))
        observation_rank, observation_reasons = _observation_rank_score(item, zt_row)
        item["observation_rank_score"] = observation_rank
        item["observation_rank_reasons"] = observation_reasons
        lhb_feature = lhb_features.get(item.get("code"))
        short_score, short_reasons, short_blockers, score_breakdown = _short_score_detail(
            item, mainline, best_flow, zt_row, force, breadth,
            flow_reason=industry_flow_error or None,
            stock_flow_value=stock_flow.get("main_net_yi") if stock_flow_fresh else None,
            lhb_feature=lhb_feature,
            research_item=research_item,
        )
        # Missing or stale key evidence is not a comparable short-term score.
        if mainline is None or mainline.get("level") != "主线" or str(mainline.get("structure_as_of") or "")[:10] != date:
            short_score = None
            short_reasons = []
            short_blockers = list(dict.fromkeys(short_blockers + ["概念/题材主线未获同日结构确认，短线分不可比"]))
            score_breakdown["_meta"]["status"] = "not_comparable"
        if best_flow is None or not industry_flow_fresh:
            short_score = None
            short_reasons = []
            short_blockers = list(dict.fromkeys(short_blockers + ["同日行业资金缺失或陈旧，短线分不可比"]))
            score_breakdown["_meta"]["status"] = "not_comparable"
        if mode in ("intraday", "after_close") and (fusion_date != date or fund_date != date):
            short_score = None
            short_reasons = []
            short_blockers = list(dict.fromkeys(short_blockers + ["市场环境关键源非决策日，短线分不可比"]))
            score_breakdown["_meta"]["status"] = "not_comparable"
        item["source_freshness"] = {
            "fusion": fusion_date == date,
            "fund_forces": fund_date == date,
            "industry_fund_flow": industry_flow_fresh,
            "concept_fund_flow": concept_flow_fresh,
        }
        item["short_term_score"] = short_score
        item["short_term_score_status"] = "可评分" if short_score is not None else "不可评分"
        item["score_breakdown"] = score_breakdown
        item["score_factors"] = {key: value for key, value in score_breakdown.items() if key != "_meta"}
        score_meta = score_breakdown.get("_meta") or {}
        item["short_term_score_raw"] = score_meta.get("raw_score")
        item["short_term_score_max"] = _SHORT_SCORE_PUBLIC_MAX
        item["short_term_score_raw_max"] = score_meta.get("max_raw_score", _SHORT_SCORE_RAW_MAX)
        item["short_term_score_method"] = score_meta.get("normalization")
        item["decision_score"] = short_score
        item["short_term_score_reason"] = "各短线因子加权加总并归一化到100"
        item["short_term_reasons"] = short_reasons
        item["short_term_blockers"] = short_blockers
        if short_score is None:
            # Evidence-only score is not comparable to a scored candidate.
            item["decision_score"] = None
        item["flow_match"] = ({"name": best_flow_name, "net_yi": best_flow, "as_of": industry_flow_date,
                                "fresh": industry_flow_fresh, "status": "同日行业资金"}
                               if best_flow_name else {"name": None, "net_yi": None, "as_of": industry_flow_date,
                                                       "fresh": False, "status": industry_flow_error or "同日行业资金尚未更新"})
        item["execution_strategy"] = {
            "entry": "竞价不弱，开盘前两轮5分钟成交额和行业资金继续增强时，分两次试探",
            "add": "回踩分时均价/开盘价不破且行业资金仍在前排，才允许加仓",
            "stop": "跌破开盘价并连续两轮弱于行业，或行业资金转负，停止执行/退出交易仓",
            "avoid": "接近涨停、炸板、缩量拉升或单日换手过热时不追",
        }
        base_ok = item.get("board") in ("沪主板", "深主板") and not item.get("hard_risk")
        evidence_ok = mainline is not None and mainline.get("level") == "主线" and best_flow is not None and best_flow > 0
        score_ok = short_score is not None and short_score >= 65 and not item.get("hard_risk")
        market_ok = force >= 35 and breadth >= 0.45 and fusion_date == date and fund_date == date
        key_sources_fresh_pre = all(value == date for value in (chain_structure_date, fusion_date, fund_date, concept_flow_date, industry_flow_date))
        coverage_ok = coverage_status == "complete" and (mode not in ("intraday", "after_close") or key_sources_fresh_pre)
        price_ok = float(item.get("change_pct") or 0) < 9.5 and float(item.get("amount") or 0) >= 1e8
        item["trade_allowed"] = bool(base_ok and evidence_ok and score_ok and market_ok and coverage_ok and price_ok)
        item["trade_label"] = "短线候选" if item["trade_allowed"] else "观察"
        item["decision"] = "满足盘口触发后试探" if item["trade_allowed"] else "观察/条件不足"
        if not item["trade_allowed"]:
            item["trade_reasons"] = list(dict.fromkeys((item.get("trade_reasons") or []) + short_blockers))
    # Ranking is score-first. Execution permission remains a separate gate and
    # only breaks ties after the comparable short-term score.
    opportunities.sort(key=lambda x: (
        x.get("short_term_score") is not None,
        float(x.get("short_term_score") or -1),
        x.get("trade_allowed") is True,
        float(x.get("observation_rank_score") or -1),
        str(x.get("code") or ""),
    ), reverse=True)
    for rank, item in enumerate(opportunities, 1):
        item["short_term_rank"] = rank if item.get("short_term_score") is not None else None
    selected = [x for x in opportunities if x.get("trade_allowed")][:10]
    if len(selected) < 5:
        selected = [x for x in opportunities
                    if x.get("board") in ("沪主板", "深主板")
                    and x.get("short_term_score") is not None
                    and x.get("short_term_score") >= 65
                    and not x.get("hard_risk")
                    and float(x.get("change_pct") or 0) < 9.5
                    and x.get("mainline_match")
                    and (x.get("flow_match") or {}).get("net_yi", 0) > 0][:10]
        for item in selected:
            if not item.get("trade_allowed"):
                item["trade_label"] = "高分观察"
                item["decision"] = "高分但市场/资金门控未通过，不执行"

    observation_candidates = [x for x in opportunities
                              if x.get("board") in ("沪主板", "深主板")
                              and not x.get("hard_risk")]
    observation_candidates.sort(key=lambda x: (
        x.get("short_term_score") is not None,
        float(x.get("short_term_score") or -1),
        float(x.get("observation_rank_score") or -1),
        str(x.get("code") or ""),
    ), reverse=True)
    observation_candidates = observation_candidates[:10]
    for rank, item in enumerate(observation_candidates, 1):
        item["observation_rank"] = rank
    for item in observation_candidates:
        if not item.get("trade_allowed"):
            item["trade_label"] = "观察"
            item["decision"] = "资金/主线未获同日确认，不执行"

    risk_flags = []
    if float(zt.get("zb_rate") or 0) >= 0.35:
        risk_flags.append("炸板率偏高，追涨降级")
    if float(zt.get("max_board") or 9) <= 2:
        risk_flags.append("连板高度不足，主线确认度低")
    if force < 35:
        risk_flags.append("资金合力偏弱，优先等待放量确认")
    if breadth < 0.4:
        risk_flags.append("市场宽度偏弱，缩小可执行范围")
    if not confirmed_mainlines:
        risk_flags.append("没有通过同日资金与扩散/梯队共同确认的主线，不生成进攻性结论")
    if not industry_flow_fresh:
        risk_flags.append(f"行业资金未更新（最近数据日{industry_flow_date or '缺失'}，决策日{date}）：{industry_flow_error or '仅作历史参考，不参与短线评分'}")
    if stock_flow_date and not stock_flow_fresh:
        risk_flags.append(f"逐股主力资金最新为{stock_flow_date}，页面展示时标记为历史数据，不参与盘中评分")
    tier1_count = int((regulatory.get("tier_counts") or {}).get("硬风险", 0) or 0)
    if tier1_count:
        risk_flags.append(f"监管硬风险索引 {tier1_count} 条，候选逐股拦截")
    if research.get("reports", 0) <= 0:
        risk_flags.append("研报证据未加载，不将主题叙事视为确认信号")
    if not factor_decay:
        risk_flags.append("日内因子衰减产物缺失，因子权重不调整")
    elif factor_decay.get("degraded"):
        risk_flags.append(f"日内因子衰减未具备有效窗口（{factor_decay.get('reason', 'unknown')}），不纳入权重")
    if mode in ("intraday", "after_close") and not (fusion_date == date and fund_date == date):
        risk_flags.append(f"市场关键源日期不一致（融合{fusion_date or '缺失'}、资金合力{fund_date or '缺失'}、决策日{date}），仅作观察")
    if mode in ("intraday", "after_close") and coverage_status != "complete":
        risk_flags.append(
            f"盘中全市场覆盖未完成（{coverage_status}，扫描{int(chain.get('n_scanned') or 0)}只），候选仅作降级观察"
        )

    decision_basis = {
        "research": {
            "status": "available" if research.get("reports", 0) else "unavailable",
            "as_of": research_date, "reports": research.get("reports", 0), "parsed_reports": research.get("parsed_reports", 0),
            "decision_extracted_reports": research.get("decision_extracted_reports", 0),
            "decision_extraction_rate": research.get("decision_extraction_rate", 0), "ocr": research.get("ocr", 0),
            "concepts": research.get("concepts", [])[:12], "all_codes": research.get("all_codes_count", 0),
            "all_companies": research.get("all_companies_count", 0),
        },
        "holders": {"status": "available" if holders.get("available", 0) else "unavailable", "as_of": _safe(holders.get("items", [{}])[0].get("as_of") if holders.get("items") else None), "available": holders.get("available", 0), "total": holders.get("total", 0), "items": holders.get("items", [])},
        "regulatory": {"status": "available" if regulatory.get("total_hits", 0) else "unavailable", "total_hits": regulatory.get("total_hits", 0), "codes": regulatory.get("codes", 0), "tier_counts": regulatory.get("tier_counts", {}), "top_codes": regulatory.get("top_codes", [])},
        "factor_decay": {"status": "degraded" if factor_decay.get("degraded") else ("available" if factor_decay else "unavailable"), "as_of": factor_date, "reason": factor_decay.get("reason") if factor_decay.get("degraded") else None, "n_windows": factor_decay.get("n_windows", 0), "factors": factor_decay.get("factors", [])},
    }
    source_chain = {
        "intraday_chain": bool(source_dates.get("intraday_chain")), "research_fusion": bool(flow), "concept_fund_flow": bool(concept_flow_values), "industry_fund_flow": bool(industry_flow_values), "fusion": bool(fusion), "fund_forces": bool(fund),
        "research_flow": bool(research_raw), "watchlist_holders": bool(holders.get("items")), "inquiry_letters": bool(regulatory.get("total_hits")),
        "factor_decay": bool(factor_decay),
    }
    fusion_actual = str(_safe(fusion.get("date")) or "")[:10] or None
    fund_actual = str(_safe(fund.get("date")) or "")[:10] or None
    fusion_fresh = fusion_actual == date
    fund_fresh = fund_actual == date
    source_dates.update({
        "fusion": str(fusion_actual)[:10] if fusion_actual else None,
        "fund_forces": str(fund_actual)[:10] if fund_actual else None,
        "research_flow": research_date,
        "factor_decay": factor_date,
        "concept_fund_flow": concept_flow_date,
        "industry_fund_flow": industry_flow_date,
        "stock_fund_flow": stock_flow_date,
    })
    primary_date = source_dates.get("intraday_chain") or source_dates.get("after_close_extra") or date
    date_mismatches = {name: actual for name, actual in source_dates.items() if actual and actual != date}
    key_dates = ("intraday_chain", "fusion", "fund_forces", "concept_fund_flow", "industry_fund_flow")
    key_sources_fresh = all(source_dates.get(key) == date for key in key_dates)

    return {
        "ok": True,
        "schema_version": "decision-snapshot.v1",
        "mode": mode,
        "requested_date": date,
        "as_of": primary_date,
        "generated_at": datetime.now(CST).isoformat(),
        "scope": {"analysis": "全市场", "analysis_detail": "概念/题材主线、行业资金、个股资金、涨停梯队与多因子短线评分", "execution": "仅展示最多5只具备完整证据链的主板候选；没有完整证据就不输出", "live_execution": "不连接券商、不下真实订单"},
        "scope_id": "analysis_decision_assist_only",
        "scope_detail": {"analysis": "全市场", "analysis_detail": "概念/题材主线、行业资金、个股资金、涨停梯队与多因子短线评分", "execution": "仅提供分析与决策辅助，不连接券商执行"},
        "market": {"temperature": risk_temp, "force_index": force, "breadth": breadth,
                   "emotion_stage": fusion.get("emotion_stage") or bias.get("tone") or "未知",
                   "zt": zt, "risk_flags": risk_flags},
        "mainlines": mainlines[:12],
        "source_freshness": {"fusion": fusion_fresh, "fund_forces": fund_fresh, "concept_fund_flow": concept_flow_fresh, "industry_fund_flow": industry_flow_fresh, "key_sources_fresh": key_sources_fresh},
        "mainline_method": {
            "rule": "只从概念/题材聚合；同日正概念资金 + 概念扩散/涨停梯队共同确认；行业只作归属，不独立判主线",
            "flow_as_of": concept_flow_date,
            "flow_fresh": concept_flow_fresh,
            "flow_label": "概念/题材资金",
            "industry_flow_as_of": industry_flow_date,
            "industry_flow_fresh": industry_flow_fresh,
            "industry_flow_error": industry_flow_error,
            "stock_flow_as_of": stock_flow_date,
            "stock_flow_fresh": stock_flow_fresh,
            "confirmed": len(confirmed_mainlines),
        },
        "money_flow": {"top_in": money.get("top_in", [])[:10], "top_out": money.get("top_out", [])[:10]},
        "sector_rotation": {"top_up": sector.get("top_up", [])[:10], "top_down": sector.get("top_down", [])[:10]},
        "etf_activity": {"top_amount": etf.get("top_amount", [])[:10], "top_main_net": etf.get("top_main_net", [])[:10]},
        "industry_fund_flow": {"status": "available" if industry_flow_values else "missing", "as_of": industry_flow_date, "rows": len(industry_flow_values), "error": industry_flow_error or None},
        "short_term_candidates": selected,
        "observation_candidates": observation_candidates,
        "opportunities": opportunities,
        "intraday_coverage": intraday_coverage,
        "evidence": decision_basis,
        "decision_basis": {
            "action": "全市场用于概念/题材主线和资金判断；只有概念/题材与同日行业资金齐全时才计算短线分数，主板候选还必须等盘口触发",
            "gates": ["board_and_st", "regulatory_tier1", "concept_or_theme_mainline", "same_day_industry_fund_flow", "amount_and_turnover", "market_breadth", "fund_force"],
            "research_note": "研报/IMA仅作为主题与催化证据，不替代价格、资金和公告核验",
            "factor_note": "日内因子仅在有效30分钟窗口具备后调整权重；degraded 时保持不纳入",
        },
        "counts": {"scanned": int(chain.get("n_scanned") or len(raw_rows)), "candidates": int(chain.get("n_candidates") or len(raw_rows)), "scored_candidates": sum(1 for x in opportunities if x.get("short_term_score") is not None), "short_term_selected": len(selected), "observation_candidates": len(observation_candidates), "execution_candidates": sum(1 for x in selected if x.get("trade_allowed")),
                   "hard_risk_candidates": sum(1 for x in opportunities if x.get("hard_risk")), "research_covered_candidates": sum(1 for x in opportunities if (x.get("research_evidence") or {}).get("reports", 0)),
                   "mainline_matched_candidates": sum(1 for x in opportunities if x.get("mainline_match")), "mainlines": len(mainlines), "risk_flags": len(risk_flags)},
        "source_chain": source_chain,
        "source_dates": source_dates,
        "date_mismatches": date_mismatches,
        "degraded": coverage_status != "complete" or bool(date_mismatches) or not key_sources_fresh,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("intraday", "after_close", "battle_map"), default="intraday")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = build_snapshot(args.mode, args.date)
    path = Path(args.out) if args.out else GEN / f"decision_snapshot_{args.mode}_{out['requested_date']}.json"
    path.write_text(json.dumps(_json_ready(out), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps({"ok": True, "path": str(path), "counts": out["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
