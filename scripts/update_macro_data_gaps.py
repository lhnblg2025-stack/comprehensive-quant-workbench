#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_macro_data_gaps.py — 宏观数据缺口补齐（2026-08-12 自检新增 8 个数据集）
=======================================================================
背景：消费者信心指数(CCI)/社零/固投/房地产指数/全社会用电量/货币供应M1M0/新增信贷/财政收入
此前缺失。自检确认全部可用 akshare 接口（东方财富/新浪）直接拉取，无需云爬虫。

本脚本:
  1. 对每个数据集用 akshare 全量拉取
  2. 月份归一化（"2026年06月份"→"2026-06"，与现有 *_yearly 文件风格一致；
     列名保持 akshare 原始中文列名）
  3. 与本地 parquet 按主键幂等合并（保留历史 + 补缺，重复跑不重复）
  4. 写 data_warehouse/data_freshness.json（月度 75 天阈值）
  5. 全部失败 → 该数据集标注 stale + 原因，绝不假装新鲜

用法:
  python3 scripts/update_macro_data_gaps.py               # 全量补齐
  python3 scripts/update_macro_data_gaps.py --only consumer_confidence,retail_sales_yoy
  python3 scripts/update_macro_data_gaps.py --no-network  # 离线：只重算新鲜度标注
  python3 scripts/update_macro_data_gaps.py --json        # 输出新鲜度 JSON

data_warehouse 对 codex 只读：落盘目录可用环境变量 MACRO_DATA_DIR 覆盖
（自查写 /tmp，本机真实环境重跑落盘 data_warehouse）。
"""
from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_macro_data_gaps] 操作失败: {e}", exc_info=True)

ROOT = Path(__file__).resolve().parent.parent
WH = Path(os.environ.get("MACRO_DATA_DIR", str(ROOT / "data_warehouse")))
FRESHNESS_JSON = WH / "data_freshness.json"
MONTHLY_STALE_DAYS = 75  # 月度阈值：超过 75 天视为 stale

# 目标：本地文件 → (akshare 函数名, 日期列, 频率, 去重主键)
# 列名保持 akshare 原始中文列名；月份归一化为 YYYY-MM
JOBS: list[dict] = [
    {"file": "macro/consumer_confidence.parquet", "fn": "macro_china_xfzxx",
     "date_col": "月份", "freq": "monthly", "keys": ["月份"],
     "desc": "消费者信心指数（东方财富经济数据一览）"},
    {"file": "macro/retail_sales_yoy.parquet", "fn": "macro_china_consumer_goods_retail",
     "date_col": "月份", "freq": "monthly", "keys": ["月份"],
     "desc": "社会消费品零售总额(社零)（东方财富经济数据一览）"},
    {"file": "macro/fixed_asset_investment.parquet", "fn": "macro_china_gdzctz",
     "date_col": "月份", "freq": "monthly", "keys": ["月份"],
     "desc": "中国城镇固定资产投资（东方财富）"},
    {"file": "macro/real_estate.parquet", "fn": "macro_china_real_estate",
     "date_col": "日期", "freq": "monthly", "keys": ["日期"],
     "desc": "国房景气指数（东方财富，月度）"},
    {"file": "macro/society_electricity.parquet", "fn": "macro_china_society_electricity",
     "date_col": "统计时间", "freq": "monthly", "keys": ["统计时间"],
     "desc": "全社会用电分类情况表（新浪财经）"},
    {"file": "macro/money_supply_full.parquet", "fn": "macro_china_supply_of_money",
     "date_col": "统计时间", "freq": "monthly", "keys": ["统计时间"],
     "desc": "货币供应量 M2/M1/M0（新浪财经）"},
    {"file": "macro/new_financial_credit.parquet", "fn": "macro_china_new_financial_credit",
     "date_col": "月份", "freq": "monthly", "keys": ["月份"],
     "desc": "新增信贷数据（东方财富）"},
    {"file": "macro/fiscal_revenue.parquet", "fn": "macro_china_czsr",
     "date_col": "月份", "freq": "monthly", "keys": ["月份"],
     "desc": "财政收入（东方财富）"},
]


def _parse_cn_date(v) -> pd.Timestamp | None:
    """把 '2026年06月份' / '2026年06月' / '2026-06' / date 等归一为 Timestamp（月首）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return pd.Timestamp(v.year, v.month, 1)
    if isinstance(v, (date, datetime)):
        return pd.Timestamp(v.year, v.month, 1)
    s = str(v).strip()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.search(r"(\d{4})-(\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.match(r"^(\d{4})(\d{2})$", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.search(r"(\d{4})\.(\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return None


def normalize_month(s: pd.Series) -> pd.Series:
    """归一化月份列为 'YYYY-MM' 字符串（失败保留原值，合并去重仍兜底）。"""
    out = s.map(lambda v: _parse_cn_date(v).strftime("%Y-%m") if _parse_cn_date(v) is not None else v)
    return out


def as_of_of(df: pd.DataFrame, date_col: str) -> date | None:
    """as_of = 日期列的最大值（月度数据取当月，发布滞后另由消费侧处理）。"""
    if df is None or df.empty or date_col not in df.columns:
        return None
    vals = df[date_col].map(_parse_cn_date).dropna()
    if vals.empty:
        return None
    return vals.max().date()


def merge_idempotent(old: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """保留历史 + 补缺 + 按主键去重（重复跑幂等）。"""
    if old is not None and len(old):
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new.copy()
    merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return merged


def load_freshness() -> dict:
    if FRESHNESS_JSON.exists():
        try:
            return json.loads(FRESHNESS_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_freshness(fresh: dict) -> None:
    fresh["_generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    FRESHNESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    _tmp = FRESHNESS_JSON.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(_tmp, FRESHNESS_JSON)


def update_one(job: dict, old: pd.DataFrame | None, no_network: bool) -> dict:
    """拉取 + 归一化 + 合并，返回新鲜度条目。任何失败 → stale + 原因，绝不假装新鲜。"""
    name = Path(job["file"]).name
    entry: dict = {
        "file": job["file"],
        "desc": job.get("desc", ""),
        "source": f"akshare.{job['fn']}",
        "freq": job["freq"],
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "threshold_days": MONTHLY_STALE_DAYS,
    }
    new = None
    fetch_note = ""
    if not no_network:
        try:
            import akshare as ak
            fn = getattr(ak, job["fn"])
            # 超时保护（非主线程无法用 signal.alarm → 直接调用）
            try:
                import signal as _sig

                def _timeout(*_a):
                    raise TimeoutError("fetch timeout")

                try:
                    _sig.signal(_sig.SIGALRM, _timeout)
                    _sig.alarm(45)
                    new = fn()
                    _sig.alarm(0)
                except (ValueError, TypeError):
                    new = fn()
                finally:
                    try:
                        _sig.alarm(0)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[update_macro_data_gaps] 操作失败: {e}", exc_info=True)
            except Exception as e:  # noqa: BLE001
                entry["last_fetch_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            if new is not None and len(new):
                fetch_note = f"akshare.{job['fn']}"
                # 月份归一化（YYYY-MM）
                dc = job["date_col"]
                if dc in new.columns:
                    new = new.copy()
                    new[dc] = normalize_month(new[dc])
                # 主键转字符串，避免 int/str 混型导致去重失效
                for k in job["keys"]:
                    if k in new.columns:
                        new[k] = new[k].astype(str).str.strip()
                entry["last_fetch_error"] = ""
        except Exception as e:  # noqa: BLE001
            entry["last_fetch_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"⚠️ {name} 抓取失败: {entry['last_fetch_error']}", flush=True)

    if new is not None and len(new):
        old_rows = 0 if old is None else len(old)
        merged = merge_idempotent(old, new, job["keys"])
        path = WH / job["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(path)
        entry["rows"] = len(merged)
        entry["source"] = fetch_note or entry.get("source", f"akshare.{job['fn']}")
        print(f"✅ {name}: {old_rows} → {len(merged)} 行（新抓 {len(new)}，源={fetch_note}）", flush=True)
    else:
        print(f"⚠️ {name}: 抓取为空/失败，保留本地数据", flush=True)
        entry["rows"] = 0 if old is None else len(old)

    # 新鲜度重算（即使离线/失败 → stale + 原因）
    cur = old if new is None else pd.read_parquet(WH / job["file"])
    as_of = as_of_of(cur, job["date_col"])
    entry["as_of"] = as_of.isoformat() if as_of else None
    if as_of:
        entry["stale_days"] = (date.today() - as_of).days
        entry["stale"] = entry["stale_days"] > MONTHLY_STALE_DAYS
    else:
        entry["stale_days"] = None
        entry["stale"] = True
    if entry.get("stale"):
        if entry.get("last_fetch_error"):
            entry["note"] = (f"拉取失败/降级：{entry['last_fetch_error']}；"
                             f"最新数据 {entry.get('as_of')}，超过 {MONTHLY_STALE_DAYS} 天阈值。请联网重跑。")
        else:
            entry["note"] = (f"外部源陈旧：最新数据 {entry.get('as_of')}，超过 {MONTHLY_STALE_DAYS} 天阈值。"
                             "请勿当作最新宏观数据使用；联网重跑本脚本会自动补缺。")
    else:
        entry["note"] = ""
    print(f"   {name}: as_of={entry.get('as_of')} stale={entry.get('stale')}", flush=True)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description="宏观数据缺口补齐（8 个数据集）+ 新鲜度标注")
    ap.add_argument("--only", default="", help="逗号分隔的文件名（如 consumer_confidence,retail_sales_yoy）")
    ap.add_argument("--no-network", action="store_true", help="离线：只重算新鲜度标注")
    ap.add_argument("--json", action="store_true", help="结束后输出新鲜度 JSON")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    fresh = load_freshness()
    print(f"=== 宏观数据缺口补齐（{len(JOBS)} 个数据集）===", flush=True)
    print(f"落盘目录: {WH}", flush=True)

    for job in JOBS:
        stem = Path(job["file"]).stem
        if only and stem not in only:
            continue
        name = Path(job["file"]).name
        path = WH / job["file"]
        old = None
        if path.exists():
            try:
                old = pd.read_parquet(path)
                # 历史文件月份列同样归一化（保证合并去重键一致）
                if job["date_col"] in old.columns:
                    old = old.copy()
                    old[job["date_col"]] = normalize_month(old[job["date_col"]])
                    for k in job["keys"]:
                        if k in old.columns:
                            old[k] = old[k].astype(str).str.strip()
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {name} 读取失败: {e}", flush=True)
                old = None
        entry = update_one(job, old, args.no_network)
        fresh[name] = entry

    save_freshness(fresh)
    print(f"✅ 新鲜度已写入 {FRESHNESS_JSON}", flush=True)
    if args.json:
        print(json.dumps(fresh, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
