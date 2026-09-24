#!/usr/bin/env python3
"""update_financial_quarterly.py — 财务季度更新(云端每周任务)

用 akshare financial_abstract 拉取股票池财务指标(ROE/净利率/现金流等)
→ 写 data_warehouse/financial/{code}.parquet, 供 stock_lens/估值系统使用。
股票池默认: 自选+持仓; --pool all 拉取全市场(慢, 分批)。

断点续传(2026-08-29): 按最新报告期跳过已最新的股票，并用
generated/financial_update_checkpoint.json 记录逐只进度；中断后再次运行
自动从断点继续，不重复抓取、不覆盖已完成产物。

用法:
  python3 scripts/update_financial_quarterly.py            # 自选+持仓(断点续传)
  python3 scripts/update_financial_quarterly.py --pool all --limit 200
  python3 scripts/update_financial_quarterly.py --force    # 忽略已有产物，全部重抓
  python3 scripts/update_financial_quarterly.py --status   # 只看当前进度，不抓取
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
FINANCIAL_DIR = ROOT / "data_warehouse" / "financial"
CHECKPOINT_FILE = ROOT / "generated" / "financial_update_checkpoint.json"


def _watch_pool() -> list[str]:
    """自选+持仓股票池（6 位代码）。

    2026-08-23 修复：get_watchlist() 返回 list[dict]（symbol/name/pe/pb），
    此前 str(c).zfill(6) 把整个 dict 当代码，导致 409 只全部抓取失败。
    """
    codes: list[str] = []

    def _norm(x) -> str:
        if isinstance(x, dict):
            x = x.get("symbol", "")
        code = str(x or "").strip().zfill(6)
        return code if code and code != "000000" else ""

    try:
        from quant_system.watchlist import get_watchlist  # noqa: PLC0415
        codes += [c for c in (_norm(x) for x in get_watchlist()) if c]
    except Exception:  # noqa: BLE001
        pass
    try:
        from quant_system.trade_db import get_positions  # noqa: PLC0415
        codes += [c for c in (_norm(p) for p in get_positions()) if c]
    except Exception:  # noqa: BLE001
        pass
    return list(dict.fromkeys(codes))


def merge_financial_frames(old: pd.DataFrame | None, current: pd.DataFrame) -> pd.DataFrame:
    """按指标名对齐合并财务宽表，逐报告期用新抓取值覆盖旧值。

    AkShare 返回的是“指标为行、报告期为列”的宽表。直接 concat 后按“选项”
    去重会在报告期集合变化时留下重复指标行，并让下游误取财务数据。
    """
    frames = [x.copy() for x in (old, current) if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame(columns=["选项"])
    normalized = []
    for frame in frames:
        # 兼容旧抓取器使用“指标”命名；对无法识别行索引的旧文件
        # 保留当前新抓取结果，不让一只坏文件阻断整个季度更新。
        if "选项" not in frame.columns:
            if "指标" in frame.columns:
                frame = frame.rename(columns={"指标": "选项"})
            elif "日期" in frame.columns:
                frame = frame.rename(columns={"日期": "选项"})
            else:
                continue
        frame["选项"] = frame["选项"].astype(str).str.strip()
        frame = frame[frame["选项"].ne("")].copy()
        frame = frame.drop_duplicates(subset=["选项"], keep="last")
        normalized.append(frame.set_index("选项"))
    merged = normalized[0]
    for frame in normalized[1:]:
        merged = merged.combine_first(frame)
        for col in frame.columns:
            if col not in merged.columns:
                merged[col] = frame[col]
            else:
                # 新抓取的非空值优先，旧值只填补本轮缺失。
                merged[col] = frame[col].combine_first(merged[col])
    result = merged.reset_index()
    result = result.rename(columns={result.columns[0]: "选项"})
    date_cols = [c for c in result.columns if c != "选项"]
    date_cols.sort(key=lambda c: pd.to_datetime(c, errors="coerce") if pd.notna(pd.to_datetime(c, errors="coerce")) else pd.Timestamp.min)
    result = result[["选项"] + date_cols]
    if result["选项"].duplicated().any():
        raise AssertionError("财务指标合并后仍存在重复选项")
    return result


def _probe_latest_report_period() -> str | None:
    """探针一只股票推断数据源当前最新报告期（用于断点跳过判定）。

    akshare financial_abstract 以"报告期"为列名(YYYYMMDD 倒序)。全部股票共享
    同一批报告期，因此只探一次即可；探针失败时返回 None，此时不加跳过、安全重抓。
    """
    try:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol="000001")
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None
    date_cols = [str(c) for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    return max(date_cols) if date_cols else None


def _latest_period_in_file(code: str) -> str | None:
    """读取已落盘 parquet 里的最新报告期列；损坏/缺失返回 None。"""
    out = FINANCIAL_DIR / f"{code}.parquet"
    try:
        df = pd.read_parquet(out)
    except Exception:  # noqa: BLE001
        return None
    date_cols = [str(c) for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    return max(date_cols) if date_cols else None


def _load_checkpoint() -> dict:
    if not CHECKPOINT_FILE.exists():
        return {"schema": "financial_update_checkpoint/v1", "done": {}, "latest_report_period": None}
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"schema": "financial_update_checkpoint/v1", "done": {}, "latest_report_period": None}
        data.setdefault("done", {})
        data.setdefault("latest_report_period", None)
        return data
    except (OSError, ValueError):
        return {"schema": "financial_update_checkpoint/v1", "done": {}, "latest_report_period": None}


def _save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_FILE)


def _update_one(code: str) -> bool:
    """拉取单只财务指标并合并写 parquet（akshare financial_abstract 多期）。"""
    try:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol=code)
    except Exception as e:  # noqa: BLE001
        print(f"[fin] {code} 抓取失败: {str(e)[:70]}", flush=True)
        return False
    if df is None or df.empty:
        return False
    # 数值化
    for col in df.columns:
        if col == "选项":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 审计 2026-08-16：修正"按行 tail(8)"的依赖返回顺序问题。
    # 该接口以"选项"(指标名)为行、报告期为列；tail(8) 会丢前面指标且依赖行序。
    # 改为：若列名可解析为报告期日期，则按日期排序保留最新 8 个报告期列；
    # 否则保留全部列（不丢指标），并在最左侧保留"选项"。
    date_cols = []
    for col in df.columns:
        if col == "选项":
            continue
        try:
            pd.to_datetime(col, errors="raise")
            date_cols.append(col)
        except Exception:
            pass
    if date_cols:
        date_cols.sort(key=lambda c: pd.to_datetime(c))
        keep_cols = ["选项"] if "选项" in df.columns else []
        keep_cols += date_cols[-8:]
        drop_cols = [c for c in df.columns if c not in keep_cols and c != "选项"]
        df = df.drop(columns=drop_cols)
    FINANCIAL_DIR.mkdir(parents=True, exist_ok=True)
    out = FINANCIAL_DIR / f"{code}.parquet"
    old = None
    if out.exists():
        try:
            old = pd.read_parquet(out)
        except Exception:  # noqa: BLE001
            old = None
    merged = merge_financial_frames(old, df)
    merged.to_parquet(out, index=False)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="财务季度更新")
    ap.add_argument("--pool", choices=["watch", "all"], default="watch")
    ap.add_argument("--limit", type=int, default=0, help="最多拉取数量(0=不限)")
    ap.add_argument("--sleep", type=float, default=1.5, help="接口间隔秒")
    ap.add_argument("--force", action="store_true", help="忽略断点与已有最新产物，全部重抓")
    ap.add_argument("--status", action="store_true", help="只打印断点进度，不抓取")
    args = ap.parse_args()

    checkpoint = _load_checkpoint()

    if args.status:
        done = checkpoint.get("done") or {}
        total = len(done)
        skipped = sum(1 for v in done.values() if isinstance(v, dict) and v.get("skipped"))
        ok = sum(1 for v in done.values() if isinstance(v, dict) and v.get("ok") and not v.get("skipped"))
        failed = sum(1 for v in done.values() if isinstance(v, dict) and not v.get("ok") and not v.get("skipped"))
        latest = checkpoint.get("latest_report_period")
        print(json.dumps({
            "schema": "financial_update_checkpoint/v1",
            "latest_report_period": latest,
            "in_progress": bool(checkpoint.get("scope_key")),
            "total_recorded": total,
            "fetched_ok": ok,
            "skipped_already_latest": skipped,
            "failed": failed,
            "completed_at": checkpoint.get("completed_at"),
            "last_result": checkpoint.get("last_result"),
            "checkpoint": str(CHECKPOINT_FILE),
        }, ensure_ascii=False, indent=2))
        return

    if args.pool == "all":
        try:
            import akshare as ak
            codes = ak.stock_info_a_code_name()["code"].astype(str).str.zfill(6).tolist()
        except Exception as e:  # noqa: BLE001
            print(f"[fin] 全市场代码列表失败: {e}", flush=True)
            return
    else:
        codes = _watch_pool()
    if args.limit:
        codes = codes[: args.limit]

    # 断点续传：`done` 只记录"本次抓取周期"内已完成的股票，用于中断后继续。
    # 正常跑完会清空，下一次运行重新按"是否已含数据源最新报告期"评估，避免把
    # 已成功但尚未披露最新季报的股票永久跳过（滚动更新仍会再次尝试）。
    import hashlib as _hashlib
    scope_key = _hashlib.sha256("|".join([args.pool, str(args.limit), *codes]).encode("utf-8")).hexdigest()
    if checkpoint.get("scope_key") != scope_key:
        checkpoint["done"] = {}
        checkpoint["scope_key"] = scope_key
        checkpoint["last_started_at"] = datetime.now(CST).isoformat(timespec="seconds")
        _save_checkpoint(checkpoint)

    # 推断数据源最新报告期，跳过已含最新期的股票。
    latest_period = checkpoint.get("latest_report_period") or _probe_latest_report_period()
    if latest_period and not checkpoint.get("latest_report_period"):
        checkpoint["latest_report_period"] = latest_period
        _save_checkpoint(checkpoint)

    done_map = checkpoint.get("done") or {}
    todo: list[str] = []
    skip_already_latest = 0
    resume_skipped = 0
    for code in codes:
        if args.force:
            todo.append(code)
            continue
        done_entry = done_map.get(code)
        if isinstance(done_entry, dict) and done_entry.get("ok") and not done_entry.get("skipped"):
            # 同一次抓取周期内已成功（断点续传，不再重复）。
            resume_skipped += 1
            continue
        if latest_period and _latest_period_in_file(code) == latest_period:
            # 已含最新报告期，无需重抓；登记为 skipped 以便进度可审计。
            checkpoint["done"].setdefault(code, {"ok": True, "skipped": True, "as_of": datetime.now(CST).isoformat(timespec="seconds")})
            skip_already_latest += 1
            continue
        todo.append(code)
    _save_checkpoint(checkpoint)

    print(f"[fin] 股票池 {len(codes)} 只: 已最新跳过 {skip_already_latest}, 断点续传跳过 {resume_skipped}, 待抓 {len(todo)}", flush=True)
    ok = 0
    for i, code in enumerate(todo, 1):
        success = _update_one(code)
        if success:
            ok += 1
        checkpoint["done"][code] = {"ok": bool(success), "skipped": False,
                                      "as_of": datetime.now(CST).isoformat(timespec="seconds")}
        # 每只落一次进度，中断后只重抓未完成部分。
        _save_checkpoint(checkpoint)
        if i % 20 == 0:
            print(f"[fin] 进度 {i}/{len(todo)} 成功{ok}", flush=True)
        time.sleep(args.sleep)

    # 本次抓取周期完整结束：清空 done，下一次运行重新评估（滚动更新）。
    checkpoint["done"] = {}
    checkpoint["scope_key"] = None
    checkpoint["completed_at"] = datetime.now(CST).isoformat(timespec="seconds")
    checkpoint["last_result"] = {"fetched": ok, "total": len(todo), "skipped_already_latest": skip_already_latest}
    _save_checkpoint(checkpoint)
    print(f"[fin] 完成: 本轮抓取 {ok}/{len(todo)} 成功（另有 {skip_already_latest} 只已含最新期跳过）", flush=True)


if __name__ == "__main__":
    main()
