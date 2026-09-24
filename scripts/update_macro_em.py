#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_macro_em.py — 宏观数据东财直连更新器（域C：新浪/jin10 源 → 东财宏观数据中心换源）
=======================================================================
背景：akshare 新浪/jin10 源宏观接口 2025-08/09 集体停更（CPI/PPI/PMI/GDP/财新PMI/工业增加值/
进出口/外储/社融等）。东财 datacenter-web.eastmoney.com 已验证本机直连可用且数据新鲜（2026-06/07）。

本脚本:
  1. requests 直连 datacenter-web.eastmoney.com（浏览器 UA，按 REPORT_DATE 升序分页全量拉取）
  2. 字段映射到本地 parquet 现有列名（先读现有文件列名/日期格式，映射对齐；不破坏下游）
  3. 日期归一化: REPORT_DATE "2026-07-01 00:00:00" → 现有文件月份格式
     （自动识别 "2026年07月份" / "2026-07" / "2026.7" / "YYYY-MM-DD"；GDP 用季度列）
  4. 与本地 parquet 幂等合并（保留历史 + 补缺 + 去重，重复跑不翻倍）
  5. 写 data_warehouse/data_freshness.json（月度 75 天 / 季度 100 天阈值）
  6. 失败降级: 单个 reportName 失败 → 该数据集标注 stale + 原因，继续其余

用法:
  python3 scripts/update_macro_em.py                     # 全量（东财直连）
  python3 scripts/update_macro_em.py --only cpi,ppi      # 只更新指定数据集
  python3 scripts/update_macro_em.py --no-network        # 离线：只重算新鲜度标注
  python3 scripts/update_macro_em.py --json              # 输出新鲜度 JSON
  MACRO_DATA_DIR=/tmp/em_test python3 scripts/update_macro_em.py   # 落盘目录覆盖（自查）

data_warehouse 对 codex 沙箱只读：真实落盘由本机执行；自查用 MACRO_DATA_DIR 覆盖到 /tmp。
"""
from __future__ import annotations
import logging

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_macro_em] 操作失败: {e}", exc_info=True)

ROOT = Path(__file__).resolve().parent.parent
# data_warehouse 对 codex 沙箱只读：MACRO_DATA_DIR 可覆盖落盘目录（自查写 /tmp，本机真实落盘）
WH = Path(os.environ.get("MACRO_DATA_DIR", str(ROOT / "data_warehouse")))
FRESHNESS_JSON = WH / "data_freshness.json"

API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
TIMEOUT = 30
PAGE_SIZE = 500
RETRY = 2

MONTHLY_STALE_DAYS = 75   # 月度阈值（简报要求）
QUARTERLY_STALE_DAYS = 100


# 目标：本地文件 → (reportName, 字段映射, 频率, 主键, 日期列)
# 字段名 = 东财 datacenter 原始列（与 akshare 东财接口一致）；列名 = 本地 parquet 现有列名。
JOBS: list[dict] = [
    {"key": "cpi", "file": "macro/cpi_yearly.parquet", "report": "RPT_ECONOMY_CPI",
     "freq": "monthly", "desc": "居民消费价格指数(CPI)", "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "全国-当月", "全国-同比增长", "全国-环比增长", "全国-累计",
                 "城市-当月", "城市-同比增长", "城市-环比增长", "城市-累计",
                 "农村-当月", "农村-同比增长", "农村-环比增长", "农村-累计"],
     "fields": {"全国-当月": "NATIONAL_BASE", "全国-同比增长": "NATIONAL_SAME",
                "全国-环比增长": "NATIONAL_SEQUENTIAL", "全国-累计": "NATIONAL_ACCUMULATE",
                "城市-当月": "CITY_BASE", "城市-同比增长": "CITY_SAME",
                "城市-环比增长": "CITY_SEQUENTIAL", "城市-累计": "CITY_ACCUMULATE",
                "农村-当月": "RURAL_BASE", "农村-同比增长": "RURAL_SAME",
                "农村-环比增长": "RURAL_SEQUENTIAL", "农村-累计": "RURAL_ACCUMULATE"}},
    {"key": "ppi", "file": "macro/ppi_yearly.parquet", "report": "RPT_ECONOMY_PPI",
     "freq": "monthly", "desc": "工业品出厂价格指数(PPI)", "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "当月", "当月同比增长", "累计"],
     "fields": {"当月": "BASE", "当月同比增长": "BASE_SAME", "累计": "BASE_ACCUMULATE"}},
    {"key": "pmi", "file": "macro/pmi_yearly.parquet", "report": "RPT_ECONOMY_PMI",
     "freq": "monthly", "desc": "采购经理人指数(PMI)", "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "制造业-指数", "制造业-同比增长", "非制造业-指数", "非制造业-同比增长"],
     "fields": {"制造业-指数": "MAKE_INDEX", "制造业-同比增长": "MAKE_SAME",
                "非制造业-指数": "NMAKE_INDEX", "非制造业-同比增长": "NMAKE_SAME"}},
    {"key": "gdp", "file": "macro/gdp_yearly.parquet", "report": "RPT_ECONOMY_GDP",
     "freq": "quarterly", "desc": "国内生产总值(GDP)", "date_col": "季度", "keys": ["季度"],
     "date_kind": "quarter", "columns": ["季度", "国内生产总值-绝对值", "国内生产总值-同比增长",
                                         "第一产业-绝对值", "第一产业-同比增长",
                                         "第二产业-绝对值", "第二产业-同比增长",
                                         "第三产业-绝对值", "第三产业-同比增长"],
     "fields": {"国内生产总值-绝对值": "DOMESTICL_PRODUCT_BASE", "国内生产总值-同比增长": "SUM_SAME",
                "第一产业-绝对值": "FIRST_PRODUCT_BASE", "第一产业-同比增长": "FIRST_SAME",
                "第二产业-绝对值": "SECOND_PRODUCT_BASE", "第二产业-同比增长": "SECOND_SAME",
                "第三产业-绝对值": "THIRD_PRODUCT_BASE", "第三产业-同比增长": "THIRD_SAME"}},
    {"key": "exports", "file": "macro/exports_yoy.parquet", "report": "RPT_ECONOMY_CUSTOMS",
     "freq": "monthly", "desc": "海关进出口(出口)", "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "当月出口额-金额", "当月出口额-同比增长", "当月出口额-环比增长",
                 "当月进口额-金额", "当月进口额-同比增长", "当月进口额-环比增长",
                 "累计出口额-金额", "累计出口额-同比增长",
                 "累计进口额-金额", "累计进口额-同比增长"],
     "fields": {"当月出口额-金额": "EXIT_BASE", "当月出口额-同比增长": "EXIT_BASE_SAME",
                "当月出口额-环比增长": "EXIT_BASE_SEQUENTIAL",
                "当月进口额-金额": "IMPORT_BASE", "当月进口额-同比增长": "IMPORT_BASE_SAME",
                "当月进口额-环比增长": "IMPORT_BASE_SEQUENTIAL",
                "累计出口额-金额": "EXIT_ACCUMULATE", "累计出口额-同比增长": "EXIT_ACCUMULATE_SAME",
                "累计进口额-金额": "IMPORT_ACCUMULATE", "累计进口额-同比增长": "IMPORT_ACCUMULATE_SAME"}},
    {"key": "imports", "file": "macro/imports_yoy.parquet", "report": "RPT_ECONOMY_CUSTOMS",
     "freq": "monthly", "desc": "海关进出口(进口)", "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "当月出口额-金额", "当月出口额-同比增长", "当月出口额-环比增长",
                 "当月进口额-金额", "当月进口额-同比增长", "当月进口额-环比增长",
                 "累计出口额-金额", "累计出口额-同比增长",
                 "累计进口额-金额", "累计进口额-同比增长"],
     "fields": {"当月出口额-金额": "EXIT_BASE", "当月出口额-同比增长": "EXIT_BASE_SAME",
                "当月出口额-环比增长": "EXIT_BASE_SEQUENTIAL",
                "当月进口额-金额": "IMPORT_BASE", "当月进口额-同比增长": "IMPORT_BASE_SAME",
                "当月进口额-环比增长": "IMPORT_BASE_SEQUENTIAL",
                "累计出口额-金额": "EXIT_ACCUMULATE", "累计出口额-同比增长": "EXIT_ACCUMULATE_SAME",
                "累计进口额-金额": "IMPORT_ACCUMULATE", "累计进口额-同比增长": "IMPORT_ACCUMULATE_SAME"}},
    {"key": "fx_reserves", "file": "macro/fx_reserves_yearly.parquet", "report": "RPT_ECONOMY_FOREX_DEPOSIT",
     "freq": "monthly", "desc": "外汇储备(国家外汇储备)", "date_col": "统计时间", "keys": ["统计时间"],
     "date_kind": "reserve", "columns": ["统计时间", "黄金储备", "国家外汇储备"],
     "fields": {"国家外汇储备": "BASE"}},
    {"key": "industrial_production", "file": "macro/industrial_production_yoy.parquet",
     "report": "RPT_ECONOMY_INDUS_GROW", "freq": "monthly", "desc": "规模以上工业增加值同比",
     "date_col": "日期", "keys": ["商品", "日期"], "date_kind": "date",
     "name_const": "中国规模以上工业增加值年率报告",
     "columns": ["商品", "日期", "今值", "预测值", "前值"],
     "fields": {"今值": "BASE_SAME"}},
    {"key": "fiscal_revenue", "file": "macro/fiscal_revenue.parquet", "report": "RPT_ECONOMY_INCOME",
     "freq": "monthly", "desc": "财政收入", "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "当月", "当月-同比增长", "当月-环比增长", "累计", "累计-同比增长"],
     "fields": {"当月": "BASE", "当月-同比增长": "BASE_SAME",
                "当月-环比增长": "BASE_SEQUENTIAL", "累计": "BASE_ACCUMULATE",
                "累计-同比增长": "ACCUMULATE_SAME"}},
    {"key": "fixed_asset_investment", "file": "macro/fixed_asset_investment.parquet",
     "report": "RPT_ECONOMY_ASSET_INVEST", "freq": "monthly", "desc": "城镇固定资产投资",
     "date_col": "月份", "keys": ["月份"],
     "columns": ["月份", "当月", "同比增长", "环比增长", "自年初累计"],
     "fields": {"当月": "BASE", "同比增长": "BASE_SAME",
                "环比增长": "BASE_SEQUENTIAL", "自年初累计": "BASE_ACCUMULATE"}},
]

JOB_BY_FILE = {j["file"]: j for j in JOBS}  # update_macro_rolling fallback 复用


def _fetch_report(report: str, page_size: int = PAGE_SIZE, fetcher=None) -> list[dict]:
    """分页拉取东财 reportName 全量数据（REPORT_DATE 升序，页大小 500）。"""
    out: list[dict] = []
    page = 1
    while True:
        params = {"reportName": report, "columns": "ALL", "pageSize": page_size,
                  "pageNumber": page, "sortColumns": "REPORT_DATE", "sortTypes": 1}
        last_err: Exception | None = None
        for attempt in range(RETRY + 1):
            try:
                if fetcher is not None:
                    j = fetcher(report, params)
                else:
                    r = requests.get(API_URL, params=params,
                                     headers={"User-Agent": UA}, timeout=TIMEOUT)
                    r.raise_for_status()
                    j = r.json()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < RETRY:
                    time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"{report} 第{page}页抓取失败: {type(last_err).__name__}: {str(last_err)[:120]}")
        if not isinstance(j, dict) or j.get("code") != 0 or not j.get("result"):
            raise RuntimeError(f"{report} 响应异常: code={j.get('code') if isinstance(j, dict) else '?'} "
                               f"msg={j.get('message') if isinstance(j, dict) else '?'}")
        res = j["result"]
        data = res.get("data") or []
        out.extend(data)
        total = res.get("count") or len(out)
        pages = res.get("pages") or math.ceil(total / page_size)
        if page >= pages or not data:
            break
        page += 1
    return out


def _detect_month_fmt(values: pd.Series) -> str:
    """从现有文件的日期列采样，识别月份字符串格式（cn/dash/dot/iso/quarter）。"""
    for v in values.head(200):
        s = str(v)
        if re.search(r"\d{4}年\d{1,2}月", s):
            return "cn"
        if re.match(r"^\d{4}-\d{2}$", s):
            return "dash"
        if re.match(r"^\d{4}\.\d{1,2}", s):
            return "dot"
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return "iso"
    return "cn"


def _fmt_month(fmt: str, y: int, m: int, d: int | None = None) -> str:
    if fmt == "cn":
        return f"{y}年{m:02d}月份"
    if fmt == "dash":
        return f"{y:04d}-{m:02d}"
    if fmt == "dot":
        return f"{y}.{m}"
    if fmt == "iso":
        return f"{y:04d}-{m:02d}-{d or 1:02d}"
    return f"{y:04d}-{m:02d}-{d or 1:02d}"


def _ym_from_time(time_str: str) -> tuple[int, int] | None:
    m = re.search(r"(\d{4})年(\d{1,2})月", str(time_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _date_label(rec: dict, job: dict, fmt: str) -> str | None:
    """把原始记录 → 目标日期列字符串（与现有文件格式一致）。"""
    kind = job.get("date_kind", "month")
    time_str = rec.get("TIME") or ""
    report_ts = pd.to_datetime(rec.get("REPORT_DATE"), errors="coerce")
    if kind == "quarter":
        t = str(time_str).strip()
        if re.match(r"^\d{4}年第\d+(?:-\d+)?季度$", t):
            return t
        if pd.notna(report_ts):
            q = (report_ts.month - 1) // 3 + 1
            return f"{report_ts.year}年第{q}季度"
        return None
    if kind == "date":  # 工业增加值：REPORT_DATE 即发布时间 → datetime.date（与现有文件列类型一致）
        if pd.notna(report_ts):
            return report_ts.date()
        return None
    ym = _ym_from_time(time_str)
    if ym is None and pd.notna(report_ts):
        ym = (report_ts.year, report_ts.month)
    if ym is None:
        return None
    return _fmt_month(fmt, ym[0], ym[1])


def _build_df(job: dict, records: list[dict], fmt: str) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in records:
        row: dict = {}
        for col, field in job["fields"].items():
            v = rec.get(field) if isinstance(rec, dict) else None
            row[col] = v
        if job.get("name_const"):
            row["商品"] = job["name_const"]
        if job.get("date_col") and job["date_col"] not in row:
            row[job["date_col"]] = _date_label(rec, job, fmt)
        rows.append(row)
    df = pd.DataFrame(rows, columns=job["columns"])
    for col in df.columns:
        if col != job.get("date_col") and col != "商品":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if job["key"] == "industrial_production":
        df = df.sort_values("日期").reset_index(drop=True)
        df["前值"] = df["今值"].shift(1)
        df["预测值"] = None
    if job["key"] == "fx_reserves":
        df["黄金储备"] = None
    return df


def _align_to_existing(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """列名对齐：以现有文件列结构为准（不破坏下游），缺失列补 NaN，多余列丢弃。"""
    if existing is None or len(existing) == 0:
        return new
    cols = list(existing.columns)
    for c in cols:
        if c not in new.columns:
            new[c] = None
    return new[cols]


def merge_idempotent(old: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """保留历史 + 补缺 + 按主键去重（重复跑幂等）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if old is not None and len(old):
            merged = pd.concat([old, new], ignore_index=True)
        else:
            merged = new.copy()
    return merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def _normalize_dtypes(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """写盘前统一列 dtype，防 pyarrow ArrowTypeError（真实数据日期列为 int/str/中文月份混合）。

    - 日期/月份/统计时间/季度 列统一 astype(str)（保留 None→None，避免 "nan" 字符串）
    - 数值列 pd.to_numeric(errors="coerce") 防混合类型（object 列只允许字符串）
    - 商品等字符串列保持不动
    """
    out = df.copy()
    # 1) 日期列 → str（保留 NaN 为 None）
    if date_col is not None and date_col in out.columns:
        col = out[date_col]
        if not pd.api.types.is_datetime64_any_dtype(col):
            out[date_col] = col.map(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        else:
            out[date_col] = col.astype("str")
    # 2) 数值列 → to_numeric（跳过日期列与字符串型列，如 商品）
    for c in out.columns:
        if c == date_col or c == "商品":
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        # object/混合列：若采样以数值为主则强制转数值，否则保持字符串（防 pyarrow 混合类型）
        sample = out[c].dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        numeric_ratio = sample.map(lambda s: _is_numeric_str(s)).mean()
        if numeric_ratio >= 0.9:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].map(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return out


def _is_numeric_str(s: str) -> bool:
    try:
        float(str(s).replace(",", "").strip())
        return True
    except (TypeError, ValueError):
        return False


def _parse_dates(series: pd.Series) -> pd.Series:
    """多格式中文日期 → Timestamp（失败 NaT）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s = pd.to_datetime(series, errors="coerce")
    if s.notna().any():
        return s
    out = pd.Series(pd.NaT, index=series.index)
    for i, v in enumerate(series.astype(str)):
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        m = re.search(r"(\d{4})年(\d{1,2})月", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
        m = re.search(r"(\d{4})年第(\d+)(?:-(\d+))?季度", v)
        if m:
            end_q = int(m.group(3) or m.group(2))
            month = {1: 3, 2: 6, 3: 9, 4: 12}.get(end_q, end_q * 3)
            out.iloc[i] = pd.Timestamp(int(m.group(1)), month, 1) + pd.offsets.MonthEnd(0)
            continue
        m = re.match(r"^(\d{4})(\d{2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
        m = re.match(r"^(\d{4})\.(\d{1,2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
        m = re.match(r"^(\d{4})-(\d{2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
    return out


def _as_of(df: pd.DataFrame | None, date_col: str) -> date | None:
    if df is None or df.empty or date_col not in df.columns:
        return None
    s = _parse_dates(df[date_col]).dropna()
    if s.empty:
        return None
    return s.max().date()


def _sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    tmp = df.assign(_t=_parse_dates(df[date_col]))
    tmp = tmp.sort_values("_t", na_position="last", kind="mergesort")
    return tmp.drop(columns="_t").reset_index(drop=True)


def load_freshness() -> dict:
    if FRESHNESS_JSON.exists():
        try:
            return json.loads(FRESHNESS_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_freshness(fresh: dict) -> None:
    fresh["_generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    FRESHNESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    _tmp = FRESHNESS_JSON.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(_tmp, FRESHNESS_JSON)


def _update_freshness(fresh: dict, job: dict, path: Path, merged: pd.DataFrame | None,
                      error: str | None = None) -> dict:
    name = Path(job["file"]).name
    entry = fresh.get(name, {})
    entry["file"] = job["file"]
    entry["source"] = f"eastmoney.datacenter.{job['report']}"
    entry["freq"] = job["freq"]
    entry["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    threshold = QUARTERLY_STALE_DAYS if job["freq"] == "quarterly" else MONTHLY_STALE_DAYS
    entry["threshold_days"] = threshold
    df = merged
    if df is None and path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            df = None
    entry["rows"] = len(df) if df is not None else entry.get("rows")
    as_of = _as_of(df, job["date_col"]) if df is not None else None
    entry["as_of"] = as_of.isoformat() if as_of else None
    if error:
        entry["last_fetch_error"] = str(error)[:200]
        entry["stale_days"] = (date.today() - as_of).days if as_of else None
        entry["stale"] = True
        entry["note"] = (f"东财 {job['report']} 抓取失败: {str(error)[:120]}；"
                         f"保留本地数据并按陈旧标注（阈值 {threshold} 天）。")
    elif as_of:
        entry["stale_days"] = (date.today() - as_of).days
        entry["stale"] = entry["stale_days"] > threshold
        entry["note"] = "" if not entry["stale"] else (
            f"东财 {job['report']} 最新数据 {entry['as_of']}，超过 {threshold} 天阈值，请勿当作最新使用。")
        entry["last_fetch_error"] = ""
    else:
        entry["stale_days"] = None
        entry["stale"] = True
        entry["note"] = f"东财 {job['report']} 无数据，标注 stale。"
    fresh[name] = entry
    return entry


def _match_only(job: dict, only: set[str]) -> bool:
    stem = Path(job["file"]).stem
    return job["key"] in only or stem in only


def run(only: set[str] | None = None, no_network: bool = False,
        write_freshness: bool = True, fetcher=None) -> dict:
    """执行更新。返回 {key: {ok, report, rows, as_of, stale, ...}}，供 update_macro_rolling fallback 复用。"""
    results: dict = {}
    fresh = load_freshness()
    print("=== 东财宏观直连更新 update_macro_em ===", flush=True)
    if no_network:
        print("（--no-network：跳过抓取，仅重算新鲜度标注）", flush=True)
    for job in JOBS:
        if only and not _match_only(job, only):
            continue
        name = Path(job["file"]).name
        path = WH / job["file"]
        existing = None
        if path.exists():
            try:
                existing = pd.read_parquet(path)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {name} 读取失败: {e}", flush=True)
                existing = None
        fmt = "cn"
        if existing is not None and len(existing) and job["date_col"] in existing.columns:
            fmt = _detect_month_fmt(existing[job["date_col"]])

        new = None
        error: str | None = None
        if not no_network:
            try:
                records = _fetch_report(job["report"], fetcher=fetcher)
                if not records:
                    raise RuntimeError("API 返回空数据")
                new = _build_df(job, records, fmt)
                new = _align_to_existing(existing, new)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"⚠️ {name} 抓取失败: {error}", flush=True)

        merged = None
        if new is not None and len(new):
            old_rows = 0 if existing is None else len(existing)
            merged = merge_idempotent(existing, new, job["keys"])
            merged = _sort_by_date(merged, job["date_col"])
            # 域C修复：写盘前统一 dtype（日期列 astype(str) + 数值列 to_numeric），防 ArrowTypeError
            merged = _normalize_dtypes(merged, job["date_col"])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp, index=False)
            tmp.replace(path)
            print(f"✅ {name}: {old_rows} → {len(merged)} 行（新抓 {len(new)}，源={job['report']}）", flush=True)
        elif not no_network:
            print(f"⚠️ {name}: 抓取为空，保留本地数据", flush=True)
            error = error or "no data"

        entry = _update_freshness(fresh, job, path, merged,
                                  error=error if (no_network or new is None or not len(new)) else None)
        print(f"   {name}: as_of={entry.get('as_of')} stale={entry.get('stale')} rows={entry.get('rows')}", flush=True)
        results[job["key"]] = {
            "ok": bool(new is not None and len(new)),
            "report": job["report"],
            "rows": entry.get("rows"),
            "as_of": entry.get("as_of"),
            "stale": entry.get("stale"),
            "error": error if (no_network or new is None or not len(new)) else None,
        }

    if write_freshness:
        save_freshness(fresh)
        print(f"✅ 新鲜度已写入 {FRESHNESS_JSON}", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="宏观数据东财直连更新器（域C 换源）")
    ap.add_argument("--only", default="", help="逗号分隔的子集（key 或文件 stem，如 cpi,ppi / cpi_yearly）")
    ap.add_argument("--no-network", action="store_true", help="离线：只重算新鲜度标注")
    ap.add_argument("--json", action="store_true", help="结束后输出新鲜度 JSON")
    args = ap.parse_args()
    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    run(only=only or None, no_network=args.no_network)
    if args.json:
        print(json.dumps(load_freshness(), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
