#!/usr/bin/env python3
"""Build a local, auditable daily research-fusion snapshot.

Only existing warehouse files are read.  No network access or data refresh is
performed.  A requested date is a cutoff: every source uses its own latest
real observation on or before that date and exposes that date as ``as_of``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data_warehouse"
DEFAULT_TOP_N = 10

SOURCES = {
    "etf_state": WAREHOUSE / "events" / "etf_state.parquet",
    "sector_fund_flow": WAREHOUSE / "market" / "sector_fund_flow.parquet",
    "fund_forces": WAREHOUSE / "market" / "fund_forces.parquet",
    "concept_board": WAREHOUSE / "classification" / "concept_board.parquet",
    "fusion": WAREHOUSE / "market" / "fusion.parquet",
    "vision_analysis": WAREHOUSE / "ima_export" / "media" / "vision_analysis.jsonl",
}

_DATE_COLUMNS = {
    "etf_state": ("date", "trade_date", "updated_at"),
    "sector_fund_flow": ("date", "trade_date", "ts"),
    "fund_forces": ("date", "trade_date", "ts"),
    "concept_board": ("ts", "date", "trade_date"),
    "fusion": ("date", "trade_date", "ts"),
}


def _iso_date(value: Any) -> str | None:
    """Normalize a date-like value without inventing a date."""
    if value is None:
        return None
    try:
        if bool(value != value):  # NaN/NaT
            return None
    except Exception:
        pass
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    match = re.search(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", text)
    if not match:
        return None
    try:
        return dt.date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values to strict, portable JSON values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            return str(value)
    try:
        if bool(value != value):
            return None
    except Exception:
        pass
    return str(value)


def _relative_path(value: Any) -> str | None:
    """Return a safe relative path; never expose an absolute local path."""
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute() and not re.match(r"^[A-Za-z]:/", text):
        return path.as_posix().lstrip("./")
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return Path(text).name


def _source_meta(name: str, status: str, as_of: str | None = None,
                 rows: int = 0, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "source": SOURCES[name].relative_to(ROOT).as_posix(),
        "as_of": as_of,
        "rows": rows,
    }
    if error:
        result["error"] = str(error)[:500]
    return result


def _read_parquet(name: str) -> tuple[Any | None, dict[str, Any]]:
    path = SOURCES[name]
    if not path.exists():
        return None, _source_meta(name, "missing", error="file not found")
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
        if frame.empty:
            return frame, _source_meta(name, "empty")
        return frame, _source_meta(name, "loaded", rows=len(frame))
    except Exception as exc:  # one broken source must not abort the snapshot
        return None, _source_meta(name, "error", error=exc)


def _dated_slice(frame: Any, candidates: Iterable[str], cutoff: str | None) -> tuple[Any | None, str | None, str | None]:
    """Select rows from the latest real source date not after cutoff."""
    if frame is None or getattr(frame, "empty", True):
        return None, None, None
    date_col = next((col for col in candidates if col in frame.columns), None)
    if date_col is None:
        return None, None, "no supported date column"
    normalized = frame[date_col].map(_iso_date)
    valid = normalized.notna()
    if cutoff:
        valid &= normalized <= cutoff
    available = normalized[valid]
    if available.empty:
        return None, None, f"no rows on or before {cutoff}" if cutoff else "no valid dates"
    as_of = str(available.max())
    selected = frame.loc[valid & (normalized == as_of)].copy()
    return selected, as_of, None


def _records(frame: Any, columns: Iterable[str]) -> list[dict[str, Any]]:
    cols = [col for col in columns if col in frame.columns]
    return [{col: _json_value(row.get(col)) for col in cols} for row in frame[cols].to_dict("records")]


def _rank(frame: Any, value_col: str, columns: Iterable[str], *, ascending: bool = False,
          sign: str | None = None, n: int = DEFAULT_TOP_N) -> tuple[list[dict[str, Any]], str | None]:
    """Rank a numeric field and return an explicit error instead of hiding it."""
    if frame is None:
        return [], "no selected rows"
    if value_col not in frame.columns:
        return [], f"missing field: {value_col}"
    try:
        import pandas as pd
        ranked = frame.copy()
        ranked[value_col] = pd.to_numeric(ranked[value_col], errors="coerce")
        ranked = ranked.dropna(subset=[value_col])
        if ranked.empty:
            return [], f"field has no numeric values: {value_col}"
        if sign == "positive":
            ranked = ranked[ranked[value_col] > 0]
        elif sign == "negative":
            ranked = ranked[ranked[value_col] < 0]
        ranked = ranked.sort_values(value_col, ascending=ascending).head(n)
        return _records(ranked, columns), None
    except Exception as exc:
        return [], f"cannot rank {value_col}: {str(exc)[:200]}"


def _frame_columns(value: Any) -> tuple[Any, ...]:
    """Get columns safely from a dataframe-like value, including None/list."""
    columns = getattr(value, "columns", ())
    try:
        return tuple(columns)
    except TypeError:
        return ()


def _has_rows(value: Any) -> bool:
    if value is None:
        return False
    empty = getattr(value, "empty", None)
    if empty is not None:
        return not bool(empty)
    try:
        return len(value) > 0
    except TypeError:
        return False


def _status_for_slice(meta: dict[str, Any], selected: Any, as_of: str | None,
                      error: str | None) -> dict[str, Any]:
    result = dict(meta)
    result["as_of"] = as_of
    try:
        result["rows"] = 0 if selected is None else len(selected)
    except TypeError:
        result["rows"] = 0
    if error:
        result["status"] = "unavailable"
        result["error"] = error
    elif _has_rows(selected):
        result["status"] = "available"
    return result


def _history_rows(frame: Any, candidates: Iterable[str], cutoff: str | None,
                  name_columns: Iterable[str], value_columns: Iterable[str],
                  *, days: int = 20, top_n: int = 20) -> list[dict[str, Any]]:
    """Return compact multi-day series for interactive report charts."""
    if frame is None or getattr(frame, "empty", True):
        return []
    date_col = next((col for col in candidates if col in frame.columns), None)
    name_col = next((col for col in name_columns if col in frame.columns), None)
    values = [col for col in value_columns if col in frame.columns]
    if date_col is None or name_col is None or not values:
        return []
    try:
        import pandas as pd
        data = frame.copy()
        data["_date"] = data[date_col].map(_iso_date)
        data = data[data["_date"].notna()]
        if cutoff:
            data = data[data["_date"] <= cutoff]
        dates = sorted(data["_date"].unique())[-days:]
        data = data[data["_date"].isin(dates)]
        for col in values:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        rank_col = values[0]
        rows = []
        for day in dates:
            batch = data[data["_date"] == day].dropna(subset=[rank_col])
            if batch.empty:
                continue
            batch = batch.assign(_abs=batch[rank_col].abs()).sort_values("_abs", ascending=False).head(top_n)
            for row in batch.to_dict("records"):
                item = {"date": day, "name": _json_value(row.get(name_col))}
                for col in values:
                    item[col] = _json_value(row.get(col))
                rows.append(item)
        return rows
    except Exception:
        return []


def _build_etf(frame: Any, meta: dict[str, Any], cutoff: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    selected, as_of, error = _dated_slice(frame, _DATE_COLUMNS["etf_state"], cutoff)
    source = _status_for_slice(meta, selected, as_of, error)
    base = {"source": source["source"], "as_of": as_of, "status": source["status"]}
    if selected is None:
        return {**base, "top_amount": [], "top_main_net": [], "share_delta": [], "history": []}, source
    common = ("code", "name", "price", "change_pct")
    top_amount, amount_error = _rank(selected, "amount", (*common, "amount"))
    top_main_net, main_net_error = _rank(selected, "main_net", (*common, "main_net"))
    # 当前源没有可靠逐日申赎流水，shares_delta=0 不能区分真零与未知。
    share_delta, shares_error = [], "daily subscription/redemption flow unavailable"
    field_errors = {
        key: value for key, value in {
            "top_amount": amount_error,
            "top_main_net": main_net_error,
            "share_delta": shares_error,
        }.items() if value
    }
    result = {
        **base,
        "top_amount": top_amount,
        "top_main_net": top_main_net,
        "share_delta": share_delta,
        "history": _history_rows(
            frame, _DATE_COLUMNS["etf_state"], cutoff,
            ("name", "code"), ("amount", "main_net"), days=20, top_n=18,
        ),
    }
    if field_errors:
        result["status"] = "partial" if len(field_errors) < 3 else "unavailable"
        result["errors"] = field_errors
        source["status"] = result["status"]
        source["field_errors"] = field_errors
    return result, source


def _build_flow(frame: Any, meta: dict[str, Any], cutoff: str | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected, as_of, error = _dated_slice(frame, _DATE_COLUMNS["sector_fund_flow"], cutoff)
    source = _status_for_slice(meta, selected, as_of, error)
    common = {"source": source["source"], "as_of": as_of, "status": source["status"]}
    selected_columns = _frame_columns(selected)
    value_col = next((c for c in ("net_yi", "net", "net_amount", "main_net") if c in selected_columns), "net_yi")
    name_cols = ("concept_name", "sector_name", "name", value_col)
    top_in, in_error = _rank(selected, value_col, name_cols, sign="positive")
    top_out, out_error = _rank(selected, value_col, name_cols, ascending=True, sign="negative")
    history = _history_rows(
        frame, _DATE_COLUMNS["sector_fund_flow"], cutoff,
        ("concept_name", "sector_name", "name"), (value_col,), days=20, top_n=24,
    )
    money = {**common, "unit": "yi" if value_col == "net_yi" else "source_native", "top_in": top_in, "top_out": top_out, "history": history}
    rotation = {**common, "metric": value_col, "top_up": top_in, "top_down": top_out, "history": history}
    rank_error = in_error or out_error
    if rank_error:
        money["status"] = rotation["status"] = "unavailable"
        money["error"] = rotation["error"] = rank_error
        source["status"] = "unavailable"
        source["error"] = rank_error
    return money, rotation, source


def _build_concepts(frame: Any, meta: dict[str, Any], cutoff: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    selected, as_of, error = _dated_slice(frame, _DATE_COLUMNS["concept_board"], cutoff)
    source = _status_for_slice(meta, selected, as_of, error)
    common = {"source": source["source"], "as_of": as_of, "status": source["status"]}
    cols = ("board_code", "board_name", "pct_chg", "price", "turnover", "up_count", "down_count", "leader_name", "leader_code")
    top_up, up_error = _rank(selected, "pct_chg", cols, sign="positive")
    top_down, down_error = _rank(selected, "pct_chg", cols, ascending=True, sign="negative")
    leaders = []
    for item in top_up:
        leaders.append({
            "board_code": item.get("board_code"),
            "board_name": item.get("board_name"),
            "leader_name": item.get("leader_name"),
            "leader_code": item.get("leader_code"),
        })
    result = {**common, "top_up": top_up, "top_down": top_down, "leader": leaders}
    rank_error = up_error or down_error
    if rank_error:
        result.update(status="unavailable", error=rank_error)
        source.update(status="unavailable", error=rank_error)
    return result, source


def _first_record(frame: Any) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    return {str(k): _json_value(v) for k, v in frame.iloc[-1].to_dict().items()}


def _build_market(fusion: Any, fusion_meta: dict[str, Any], forces: Any,
                  forces_meta: dict[str, Any], cutoff: str | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fusion_slice, fusion_date, fusion_error = _dated_slice(fusion, _DATE_COLUMNS["fusion"], cutoff)
    force_slice, force_date, force_error = _dated_slice(forces, _DATE_COLUMNS["fund_forces"], cutoff)
    fusion_status = _status_for_slice(fusion_meta, fusion_slice, fusion_date, fusion_error)
    force_status = _status_for_slice(forces_meta, force_slice, force_date, force_error)
    frow, force_row = _first_record(fusion_slice), _first_record(force_slice)
    field_errors = {}
    if fusion_slice is not None:
        fusion_columns = _frame_columns(fusion_slice)
        missing = [field for field in ("temperature", "emotion_stage") if field not in fusion_columns]
        if missing:
            field_errors["fusion"] = f"missing fields: {', '.join(missing)}"
            fusion_status.update(status="unavailable", error=field_errors["fusion"])
    if force_slice is not None and "force_index" not in _frame_columns(force_slice):
        field_errors["fund_forces"] = "missing field: force_index"
        force_status.update(status="unavailable", error=field_errors["fund_forces"])
    available_parts = sum(status.get("status") == "available" for status in (fusion_status, force_status))
    result = {
        "status": "available" if available_parts == 2 else ("partial" if available_parts else "unavailable"),
        "as_of": max(filter(None, (fusion_date, force_date)), default=None),
        "temperature": frow.get("temperature"),
        "emotion": frow.get("emotion_stage"),
        "force_index": force_row.get("force_index"),
        "temperature_tag": frow.get("tag"),
        "force_detail": {key: force_row.get(key) for key in ("youzi_net", "jg_net", "north_net", "margin_delta", "n_legs")},
        "history": _history_rows(
            forces, _DATE_COLUMNS["fund_forces"], cutoff,
            ("date", "trade_date"), ("force_index", "youzi_net", "jg_net", "margin_delta"), days=30, top_n=1,
        ),
        "sources": {
            "temperature_emotion": {"source": fusion_status["source"], "as_of": fusion_date, "status": fusion_status["status"]},
            "force_index": {"source": force_status["source"], "as_of": force_date, "status": force_status["status"]},
        },
    }
    errors = [x for x in (fusion_error, force_error) if x]
    errors.extend(field_errors.values())
    if errors:
        result["errors"] = errors
    return result, {"fusion": fusion_status, "fund_forces": force_status}


def _load_vision_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], _source_meta("vision_analysis", "missing", error="file not found")
    records, bad_lines = [], 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
                    else:
                        bad_lines += 1
                except (TypeError, ValueError, json.JSONDecodeError):
                    bad_lines += 1
    except OSError as exc:
        return [], _source_meta("vision_analysis", "error", error=exc)
    meta = _source_meta("vision_analysis", "loaded", rows=len(records))
    if bad_lines:
        meta["parse_errors"] = bad_lines
    return records, meta


def _vision_date(record: Mapping[str, Any]) -> str | None:
    return _iso_date(record.get("report_date") or record.get("source_date") or record.get("analyzed_at"))


def _summarize_vision(records: list[dict[str, Any]], meta: dict[str, Any],
                      cutoff: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    eligible = [(rec, _vision_date(rec)) for rec in records]
    eligible = [(rec, date) for rec, date in eligible if date and (not cutoff or date <= cutoff)]
    as_of = max((date for _, date in eligible), default=None)
    batch = [rec for rec, date in eligible if date == as_of] if as_of else []
    ok = [rec for rec in batch if rec.get("status") == "ok"]
    failed = [rec for rec in batch if rec.get("status") != "ok"]
    visual_types = Counter(str(rec.get("visual_type") or "other") for rec in ok)
    images = []
    for rec in ok:
        image = {
            "relative_path": _relative_path(rec.get("relative_path")),
            "title": _json_value(rec.get("title")),
            "visual_type": _json_value(rec.get("visual_type") or "other"),
            "source_date": _iso_date(rec.get("source_date")),
            "summary": _json_value(rec.get("summary")),
            "metrics": _json_value(rec.get("metrics") if isinstance(rec.get("metrics"), list) else []),
            "series": _json_value(rec.get("series") if isinstance(rec.get("series"), list) else []),
            "flow": _json_value(rec.get("flow") if isinstance(rec.get("flow"), dict) else {}),
        }
        images.append(image)
    status = "available" if ok else ("empty" if records else meta.get("status", "missing"))
    source = dict(meta, status=status, as_of=as_of, rows=len(batch))
    result = {
        "status": status,
        "source": source["source"],
        "as_of": as_of,
        "success_count": len(ok),
        "error_count": len(failed),
        "by_visual_type": dict(sorted(visual_types.items())),
        "images": images,
    }
    if meta.get("error"):
        result["error"] = meta["error"]
    if meta.get("parse_errors"):
        result["parse_errors"] = meta["parse_errors"]
    return result, source


def _cross_checks(statuses: Mapping[str, Mapping[str, Any]], requested: str | None,
                  as_of: str | None) -> dict[str, Any]:
    missing, errors, degraded, mismatches, future = [], [], [], [], []
    for name, status in statuses.items():
        state, source_date = status.get("status"), status.get("as_of")
        if state in {"missing", "empty", "unavailable"}:
            missing.append({"source": name, "status": state, "error": status.get("error")})
        elif state == "error":
            errors.append({"source": name, "error": status.get("error")})
        elif state == "partial":
            degraded.append({"source": name, "status": state, "errors": status.get("field_errors") or status.get("error")})
        if source_date and as_of and source_date != as_of:
            try:
                lag = (dt.date.fromisoformat(as_of) - dt.date.fromisoformat(str(source_date))).days
            except ValueError:
                lag = None
            mismatches.append({"source": name, "as_of": source_date, "reference_as_of": as_of, "lag_days": lag})
        if requested and source_date and source_date > requested:
            future.append({"source": name, "as_of": source_date, "requested_date": requested})
    notes = []
    if requested and as_of and as_of < requested:
        notes.append(f"requested {requested}; latest eligible warehouse observation is {as_of}")
    if mismatches:
        notes.append("sources have different real latest dates; older observations were not relabeled")
    if missing:
        notes.append("one or more optional/required inputs are unavailable")
    if degraded:
        notes.append("one or more inputs are only partially usable because fields are missing or invalid")
    return {
        "status": "ok" if not (missing or errors or degraded or mismatches or future) else "attention",
        "missing_sources": missing,
        "source_errors": errors,
        "degraded_sources": degraded,
        "date_mismatches": mismatches,
        "future_date_violations": future,
        "notes": notes,
    }


def _parse_requested_date(value: Any) -> str | None:
    if value is None:
        return None
    parsed = _iso_date(value)
    if parsed is None:
        raise ValueError(f"invalid date: {value!r}; expected YYYY-MM-DD")
    return parsed


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temp_name).replace(path)
    except Exception:
        try:
            Path(temp_name).unlink(missing_ok=True)
        finally:
            raise


def build_snapshot(date: Any = None, out_path: str | Path | None = None) -> dict[str, Any]:
    """Build and write a snapshot, returning the same JSON-compatible object.

    ``date`` is a cutoff, not a label imposed on source data.  If omitted, the
    maximum real date found across tabular sources is used as the snapshot date.
    Individual source errors are represented in ``data_status`` and never raised.
    """
    requested = _parse_requested_date(date)
    frames: dict[str, Any] = {}
    raw_status: dict[str, dict[str, Any]] = {}
    for name in ("etf_state", "sector_fund_flow", "fund_forces", "concept_board", "fusion"):
        frames[name], raw_status[name] = _read_parquet(name)

    vision_records, vision_meta = _load_vision_records(SOURCES["vision_analysis"])
    if requested is None:
        observed = []
        for name, frame in frames.items():
            _, source_date, _ = _dated_slice(frame, _DATE_COLUMNS[name], None)
            if source_date:
                observed.append(source_date)
        observed.extend(date for date in (_vision_date(rec) for rec in vision_records) if date)
        requested = max(observed, default=dt.date.today().isoformat())

    etf, etf_status = _build_etf(frames["etf_state"], raw_status["etf_state"], requested)
    money, sector, flow_status = _build_flow(frames["sector_fund_flow"], raw_status["sector_fund_flow"], requested)
    concepts, concept_status = _build_concepts(frames["concept_board"], raw_status["concept_board"], requested)
    market, market_statuses = _build_market(
        frames["fusion"], raw_status["fusion"], frames["fund_forces"], raw_status["fund_forces"], requested
    )
    images, vision_status = _summarize_vision(vision_records, vision_meta, requested)

    statuses = {
        "etf_state": etf_status,
        "sector_fund_flow": flow_status,
        "fund_forces": market_statuses["fund_forces"],
        "concept_board": concept_status,
        "fusion": market_statuses["fusion"],
        "vision_analysis": vision_status,
    }
    actual_dates = [str(s["as_of"]) for s in statuses.values() if s.get("as_of")]
    as_of = max(actual_dates, default=None)
    available = sum(s.get("status") in {"available", "partial"} for s in statuses.values())
    fully_available = sum(s.get("status") == "available" for s in statuses.values())
    snapshot = {
        "schema_version": "research-fusion.v1",
        "scope": "analysis_decision_assist_only",
        "scope_detail": {"analysis": "全市场资金、行业、概念与ETF证据汇总", "execution": "仅作分析与决策辅助，不连接券商执行"},
        "as_of": as_of,
        "requested_date": requested,
        "data_status": {
            "status": "complete" if fully_available == len(statuses) else ("partial" if available else "unavailable"),
            "available_sources": available,
            "fully_available_sources": fully_available,
            "total_sources": len(statuses),
            "sources": statuses,
        },
        "market_summary": market,
        "money_flow": money,
        "etf_activity": etf,
        "sector_rotation": sector,
        "concept_rotation": concepts,
        "research_images": images,
        "cross_checks": _cross_checks(statuses, requested, as_of),
    }
    output = Path(out_path) if out_path is not None else ROOT / "generated" / f"research_fusion_snapshot_{requested}.json"
    if not output.is_absolute():
        output = ROOT / output
    snapshot["output_path"] = output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output)
    _atomic_json_write(output, snapshot)
    return snapshot


def self_test() -> dict[str, Any]:
    """Run lightweight pure transformation checks (no warehouse or network I/O)."""
    assert _iso_date("20260824") == "2026-08-24"
    assert _iso_date("not-a-date") is None
    assert _json_value(float("nan")) is None
    assert not str(_relative_path("/tmp/private/chart.png")).startswith("/")
    records = [
        {"status": "ok", "report_date": "2026-08-24", "relative_path": "images/a.png", "visual_type": "ETF", "metrics": [{"label": "规模", "value": 1}], "series": [], "flow": {"net": 2}},
        {"status": "ok", "report_date": "2026-08-24", "relative_path": "/tmp/b.png", "visual_type": "ETF", "metrics": [], "series": [], "flow": {}},
        {"status": "error", "report_date": "2026-08-24", "relative_path": "images/c.png", "error": "bad"},
        {"status": "ok", "report_date": "2026-08-23", "relative_path": "images/old.png", "visual_type": "行业"},
    ]
    result, status = _summarize_vision(records, _source_meta("vision_analysis", "loaded", rows=4), "2026-08-24")
    assert result["success_count"] == 2 and result["error_count"] == 1
    assert result["by_visual_type"] == {"ETF": 2}
    assert result["as_of"] == status["as_of"] == "2026-08-24"
    assert all(not str(item["relative_path"]).startswith("/") for item in result["images"])
    checks = _cross_checks({"new": {"status": "available", "as_of": "2026-08-24"}, "old": {"status": "available", "as_of": "2026-08-13"}}, "2026-08-24", "2026-08-24")
    assert checks["date_mismatches"][0]["lag_days"] == 11
    try:
        import pandas as pd
        ranked, rank_error = _rank(pd.DataFrame({"value": ["bad"]}), "value", ("value",))
        assert ranked == [] and rank_error == "field has no numeric values: value"
        inflow, _ = _rank(pd.DataFrame({"value": [-1, 2]}), "value", ("value",), sign="positive")
        assert inflow == [{"value": 2}]
    except ImportError:
        pass
    return {"status": "ok", "checks": 11}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local research-fusion snapshot")
    parser.add_argument("--date", help="source date cutoff (YYYY-MM-DD)")
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument("--self-test", action="store_true", help="run built-in pure transformation checks")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return 0
    try:
        snapshot = build_snapshot(date=args.date, out_path=args.out)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": snapshot["data_status"]["status"], "as_of": snapshot["as_of"], "output": snapshot["output_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
