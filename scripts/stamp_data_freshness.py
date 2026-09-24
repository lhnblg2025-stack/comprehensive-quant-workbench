#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stamp_data_freshness.py — 全仓数据新鲜度标注（data_warehouse/data_freshness.json）
======================================================================
对 data_warehouse 下"数据集级" parquet 扫描日期列，计算 as_of 与 stale 标记。
覆盖无本地滚动脚本的历史遗留数据（gdhs_all / account_stat / esg_rating /
rates__fx 等）：外部源取不到新数据时也如实标注，绝不假装新鲜。

跳过：逐股目录（financial/kline/valuation）、按日快照目录（zt_pool*、
realtime_snapshot）、季度周期文件（quarterly/）、lhb_20* 区间文件。

快照/无日期列数据集（V13 P2-2 审计登记，2026-08-15）：
  * 存在可解析日期列（concept_board/theme_board 的 ts、lpr/rates__lpr 的
    TRADE_DATE）→ 以数据内最新日期为 as_of（真实值，非估计）。
  * 每日快照且文件名含 _YYYYMMDD（hot_rank_* / baidu_hot_* / weibo_*）
    → as_of = 文件名日期。
  * 其余快照/无日期列数据集登记于 SNAPSHOT_MTIME → as_of = 文件 mtime 日期，
    阈值取 THRESHOLD_OVERRIDE 或默认；mtime 表示"最近写入时刻"，非数据内值。
  * 静态分类映射（concept_member/sw_* 等）登记于 NO_DATE_EXEMPT →
    stale=False + note（内容随上游分类调整而更新，不适用逐日新鲜度判定）。
  以上所有登记均保证"as_of 或 stale+note 至少齐全"，绝不出现
  "as_of=None 且无 stale"的无信号条目。as_of 一律来自真实日期（文件内日期列
  / 文件名日期 / 目录或文件 mtime），绝不编造。

margin_detail_{sh,sz}/{YYYYMMDD}.parquet（P2-3 关联）：不再被 silent 跳过，
在扫描末尾按目录补登记，as_of = 目录内最新日文件（YYYYMMDD）日期。

用法:
  python3 scripts/stamp_data_freshness.py          # 全量扫描 + 写 data_freshness.json
  python3 scripts/stamp_data_freshness.py --json   # 打印结果

下游读取示例:
  freshness = json.load(open("data_warehouse/data_freshness.json"))
  if freshness.get("gdhs_all.parquet", {}).get("stale"): ...
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[stamp_data_freshness] 操作失败: {e}", exc_info=True)

ROOT = Path(__file__).resolve().parent.parent
WH = ROOT / "data_warehouse"
FRESHNESS_JSON = WH / "data_freshness.json"

DATE_HINTS = ["日期", "date", "trade_date", "交易日", "时间", "报告期", "上榜日",
              "交易日期", "数据日期", "公告日期", "变动日期", "统计时间", "月份", "季度",
              # V13 P2-2: 主题/概念板块快照的最新更新时间戳（datetime 对象/日期字符串）
              "ts"]
DATE_EXCL = ["封板时间", "开板日期", "上市日期", "时间戳", "datetime", "统计截止",
             "首次封板", "最后封板", "updated_at", "更新时间", "update_time"]

# 跳过目录/文件模式（逐股、按日、按季度、区间文件）
SKIP_DIRS = {"financial", "kline", "valuation", "quarterly", "feature_store",
             "realtime_snapshot", "predictions"}
SKIP_PREFIXES = ("lhb_20", "zt_pool", "zt_pool_", "zt_pool_dtgc", "zt_pool_strong",
                 "zt_pool_subnew", "gdfx_", "analyst_")
# 已知低频数据集的判定阈值覆盖（天）
THRESHOLD_OVERRIDE = {
    "account_stat.parquet": 3650,     # 中登账户统计 2023-08 后公开渠道停发（2026-08-14 登记）
    "esg_rating.parquet": 400,        # 年度 ESG 评级快照（低频）
    "gdhs_all.parquet": 3650,         # 股东户数源 2023-11 停更（2026-08-14 登记）
    "fund_portfolio_hold.parquet": 120,  # 季度基金持仓
    # 外部源停更/低频（2026-08-14 逐一核实后登记，防误报陈旧告警）
    "commodity__coal.parquet": 3650,  # 动力煤期货源 2022-12 停更（外部限制，不可修）
    "reserve_requirement_ratio.parquet": 730,  # 事件型数据：2025-05 末次降准后无变化（非陈旧）
    "csi_industry.parquet": 400,      # 中证行业指数：外部源低频发布（云端每日拉但源数据季度更新）
    "real_estate.parquet": 400,       # NBS 房地产投资数据 2025-12 后停发（2e2ed4b 已登记）
    "shrzgm.parquet": 180,            # 月度社融：东财源滞后（2026-08-14 核实中）
    "macro_monthly__shrzgm.parquet": 180,
    # 月度宏观（非 update_macro_rolling 管理的遗留 macro_monthly__*）
    "macro_monthly__cpi.parquet": 75,
    "macro_monthly__pmi.parquet": 75,
    "macro_monthly__ppi.parquet": 75,
    # 乘联会 CPCA 月度数据（月中发布上月，允许 60 天滞后）
    "car_segment_cpca.parquet": 60,
    # V13 P2-2 快照登记阈值：按对应更新脚本预期节奏放宽
    "car_man_rank_cpca.parquet": 60,       # 月度厂商零售排名（同 CPCA 60 天滞后）
    "car_sale_rank_gasgoo.parquet": 60,    # 盖世月度批发排名（月度）
    "cb_spot.parquet": 5,                  # 可转债现价快照（应每日盘中/盘后更新）
    "margin.parquet": 5,                   # 两融总量快照（应每交易日更新）
    "stock_names.parquet": 7,              # 股票名称/ST 标记（应每周内更新）
    "hot_rank.parquet": 5,                 # 热搜榜（应每日更新）；hot_rank_YYYYMMDD 同
    "margin_detail_sh": 30,                # 两融个股明细 sh 日更目录（as_of=最新日文件日期）
    "margin_detail_sz": 30,                # 两融个股明细 sz 日更目录
}
DEFAULT_THRESHOLD = 30

# 快照/无日期列数据集 → as_of 取文件 mtime 日期，stale 按 阈值判定。
# 注意：as_of 是"最近写入时刻（mtime）"非数据内日期，note 中如实说明。
SNAPSHOT_MTIME = {
    # 可转债现价快照：ticktime 仅含时刻(如 15:30:00)无日期，无法行内解析 as_of
    "cb_spot.parquet": "可转债现价快照，ticktime 仅含时刻(如 15:30:00)无日期；"
                       "as_of 取文件 mtime（最近写入时刻）",
    # 两融总量快照：无任何日期列
    "margin.parquet": "两融总量快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
    # 股票名称/ST 标记快照：无日期列
    "stock_names.parquet": "股票名称/ST 标记快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
    # 打新收益率 oneoff 快照：无行内日期列
    "dxsyl.parquet": "打新收益率 oneoff 快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
    # 龙虎榜席位累计快照：无行内日期列
    "lhb_ggtj_sina.parquet": "龙虎榜营业部席位累计快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
    # 季度基金持仓快照：无行内日期列，季度阈值见 THRESHOLD_OVERRIDE
    "fund_portfolio_hold.parquet": "季度基金持仓快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
    # 申万二级实时行情点：实时行情点，无日期列，仅价量
    "sw_second_spot.parquet": "申万二级实时行情点快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
    # 月度厂商排名：月份以列名形式存放（如 '2025年7月'），无行内日期列
    "car_man_rank_cpca.parquet": "乘联会月度厂商零售排名，月份在列名中(如 '2025年7月')；"
                                 "as_of 取文件 mtime（最近写入时刻）",
    "car_sale_rank_gasgoo.parquet": "盖世汽车月度批发排名，月份在列名中(如 '2026-8')；"
                                    "as_of 取文件 mtime（最近写入时刻）",
    # 每日快照但文件名不含统一 _YYYYMMDD 模式时兜底（正常走文件名日期分支）
    "sw_industry.parquet": "申万行业行情快照，无日期列；as_of 取文件 mtime（最近写入时刻）",
}

# 日期列无年份等格式限制 / 静态分类映射的数据集 → 有数据但不标 stale，
# 仅登 note 透明说明判定方式（2026-08-15 新增：分类映射为低频内容快照）。
NO_DATE_EXEMPT = {
    "car_total_cpca.parquet": "月份列仅含 1月..12月（年份在列名中），格式限制，无法解析 as_of",
    # 概念/题材成分（成员列表，随上游分类调整而更新，不适用逐日新鲜度判定）
    "concept_member.parquet": "概念板块成分静态映射（东财），内容随板块分类调整而更新；"
                              "非逐日行情，无行内日期列，不按天判定新鲜度",
    "concept_member_ths.parquet": "同花顺概念成分静态映射，内容随板块分类调整而更新；"
                                  "非逐日行情，无行内日期列，不按天判定新鲜度",
    "concept_ths_boards.parquet": "同花顺概念板块列表静态映射，无行内日期列；"
                                  "非逐日行情，不按天判定新鲜度",
    # 申万行业分类映射（一级/二级/板块映射，低频静态）
    "sw_first.parquet": "申万一级行业分类静态映射，无行内日期列；随官方分类调整更新，不按天判定",
    "sw_second.parquet": "申万二级行业分类静态映射，无行内日期列；随官方分类调整更新，不按天判定",
    "sw_industry_map.parquet": "个股—申万行业分类映射，无行内日期列；随官方分类调整更新，不按天判定",
}

# 每日快照文件的文件名日期模式（如 hot_rank_20260811.parquet）→ as_of=文件名日期。
# 优先级高于 SNAPSHOT_MTIME（文件名日期即数据所属交易日，比 mtime 更准）。
SNAPSHOT_FILENAME_DATE_RE = re.compile(r"_(\d{8})\.parquet$")


def _find_date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cs = str(c).lower()
        if any(h.lower() in cs for h in DATE_HINTS) and \
           not any(e.lower() in cs for e in DATE_EXCL):
            return str(c)
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    """兼容 中文月份/季度/年月日 等格式（与 update_macro_rolling 同逻辑）。"""
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().any():
        return s
    out = pd.Series(pd.NaT, index=series.index)
    for i, v in enumerate(series.astype(str)):
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))); continue
        m = re.search(r"(\d{4})年(\d{1,2})月", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1); continue
        # 2026-08-14 补充: 乘联会 CPCA 用 "2026-7月" 连字符中文月格式
        m = re.search(r"(\d{4})-(\d{1,2})月", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1); continue
        m = re.search(r"(\d{4})年第(\d)季度", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)) * 3, 1); continue
        m = re.match(r"^(\d{4})(\d{2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1); continue
        m = re.search(r"^(\d{4})-(\d{2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1); continue
    return out


def _as_of(df: pd.DataFrame, date_col: str) -> date | None:
    s = _parse_dates(df[date_col]).dropna()
    if s.empty:
        return None
    return s.max().date()


def _snapshot_filename_date(path: Path) -> date | None:
    """每日快照文件名含 _YYYYMMDD（如 hot_rank_20260811.parquet）→ 该日 as_of。"""
    m = SNAPSHOT_FILENAME_DATE_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:  # noqa: BLE001
        return None


def _snapshot_threshold(path: Path) -> int:
    """快照阈值：THRESHOLD_OVERRIDE 精确名 → SNAPSHOT_MTIME 精确名 → 文件名日期快照默认。"""
    if path.name in THRESHOLD_OVERRIDE:
        return THRESHOLD_OVERRIDE[path.name]
    # 每日快照文件名日期（hot_rank_*/baidu_hot_*/weibo_*）默认 5 天
    if _snapshot_filename_date(path):
        return 5
    return DEFAULT_THRESHOLD


def scan_dataset(path: Path) -> dict | None:
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        return {"file": str(path.relative_to(WH)), "error": str(e)[:100]}
    dcol = _find_date_col(df)
    rel = str(path.relative_to(WH))
    if dcol is not None:
        as_of = _as_of(df, dcol)
        thr = THRESHOLD_OVERRIDE.get(path.name, DEFAULT_THRESHOLD)
        entry = {
            "file": rel, "date_col": dcol, "rows": int(len(df)),
            "as_of": as_of.isoformat() if as_of else None,
            "threshold_days": thr,
            "stale_days": (date.today() - as_of).days if as_of else None,
        }
        # 日期列无年份等格式限制的数据集（有数据但 as_of 解析不出）→ 仍保留 note 不误报
        if path.name in NO_DATE_EXEMPT and as_of is None and len(df) > 0:
            entry["stale"] = False
            entry["note"] = NO_DATE_EXEMPT[path.name]
            return entry
        entry["stale"] = (entry["stale_days"] is not None
                          and entry["stale_days"] > thr) or entry["as_of"] is None
        if entry["stale"]:
            entry["note"] = (f"数据截至 {entry['as_of']}，超过 {thr} 天阈值；"
                             "请勿当作最新数据使用。")
        else:
            entry["note"] = ""
        return entry

    # ---- 无日期列（快照/无解析日期）分支：保证 as_of 或 stale+note 至少齐全 ----
    entry = {"file": rel, "rows": int(len(df)), "as_of": None}
    # 1) 每日快照文件名含 _YYYYMMDD → as_of=文件名日期（真实值）
    fdate = _snapshot_filename_date(path)
    if fdate is not None:
        thr = _snapshot_threshold(path)
        entry.update(date_from="filename", threshold_days=thr)
        as_of = fdate
        entry["as_of"] = as_of.isoformat()
        entry["stale_days"] = (date.today() - as_of).days
        entry["stale"] = entry["stale_days"] > thr
        entry["note"] = (f"每日快照，按文件名日期判定 as_of={as_of}（{thr} 天阈值）"
                         if not entry["stale"] else
                         f"每日快照按文件名日期 {as_of} 已超 {thr} 天阈值，请检查更新。")
        return entry
    # 2) SNAPSHOT_MTIME 登记 → as_of=文件 mtime 日期（最近写入时刻）
    if path.name in SNAPSHOT_MTIME:
        thr = _snapshot_threshold(path)
        mdate = date.fromtimestamp(path.stat().st_mtime)
        entry.update(date_from="mtime", threshold_days=thr)
        entry["as_of"] = mdate.isoformat()
        entry["stale_days"] = (date.today() - mdate).days
        entry["stale"] = entry["stale_days"] > thr
        entry["note"] = SNAPSHOT_MTIME[path.name] + ("；新鲜度按 mtime 判定"
                                                     if entry["stale"] is False
                                                     else "；新鲜度按 mtime 判定（已陈旧）")
        return entry
    # 3) NO_DATE_EXEMPT 登记（静态映射/格式限制）→ stale=False + note
    if path.name in NO_DATE_EXEMPT and len(df) > 0:
        entry["stale"] = False
        entry["note"] = NO_DATE_EXEMPT[path.name]
        return entry
    # 4) 兜底：未登记的无日期列数据集 → 明确 stale=True + note（绝不无信号）
    entry["stale"] = True
    entry["note"] = "无日期列且未登记判定方式：新鲜度不可靠，请登记 SNAPSHOT_MTIME / NO_DATE_EXEMPT。"
    return entry


def register_margin_detail() -> list[dict]:
    """P2-3 关联：margin_detail_{sh,sz} 按目录补登记，不再被 SKIP 后失去信号。
    as_of = 目录内最新日文件（YYYYMMDD）日期（真实值，来自文件名）。
    """
    out = []
    market_dir = WH / "market"
    for name in ("margin_detail_sh", "margin_detail_sz"):
        d = market_dir / name
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.parquet"))
        if not files:
            entry = {"file": f"market/{name}", "as_of": None, "rows": None,
                     "stale": True, "note": "目录为空，无任何日文件可判定新鲜度"}
            out.append((name, entry))
            continue
        latest = files[-1].stem  # YYYYMMDD
        try:
            as_of = datetime.strptime(latest, "%Y%m%d").date()
        except ValueError:  # noqa: BLE001
            as_of = date.fromtimestamp(files[-1].stat().st_mtime)
        thr = THRESHOLD_OVERRIDE.get(name, 30)
        stale_days = (date.today() - as_of).days
        stale = stale_days > thr
        entry = {
            "file": f"market/{name}",
            "as_of": as_of.isoformat(),
            "threshold_days": thr,
            "stale_days": stale_days,
            "stale": stale,
            "date_from": "latest_daily_filename",
            "note": (f"margin_detail 日更目录，as_of=最新日文件 {latest}（{thr} 天阈值）"
                     if not stale else
                     f"margin_detail 最新日文件 {latest} 已超 {thr} 天阈值，请检查更新。"),
        }
        out.append((name, entry))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="全仓数据新鲜度标注")
    ap.add_argument("--json", action="store_true", help="打印 JSON")
    args = ap.parse_args()

    fresh: dict = {}
    if FRESHNESS_JSON.exists():
        try:
            fresh = json.loads(FRESHNESS_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            fresh = {}

    scanned = 0
    for sub in sorted(p for p in WH.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        for f in sorted(sub.glob("*.parquet")):
            if f.name.startswith(SKIP_PREFIXES):
                continue
            entry = scan_dataset(f)
            if entry is None:
                continue
            prev = fresh.get(f.name, {})
            # update_macro_rolling.py 维护的宏观条目（带 freq）不被覆盖
            if prev.get("freq"):
                continue
            fresh[f.name] = {**prev, **entry,
                             "checked_at": datetime.now().astimezone().isoformat(timespec="seconds")}
            scanned += 1
            st = "STALE" if entry.get("stale") else "ok"
            print(f"  [{st:5s}] {entry.get('file')}: as_of={entry.get('as_of')} "
                  f"days={entry.get('stale_days')}", flush=True)

    # V13 P2-3 关联：margin_detail_{sh,sz} 按目录补登记（as_of=最新日文件日期）
    for name, entry in register_margin_detail():
        prev = fresh.get(name, {})
        if not prev.get("freq"):
            fresh[name] = {**prev, **entry,
                           "checked_at": datetime.now().astimezone().isoformat(timespec="seconds")}
            st = "STALE" if entry.get("stale") else "ok"
            print(f"  [{st:5s}] {entry.get('file')}: as_of={entry.get('as_of')} "
                  f"days={entry.get('stale_days')}", flush=True)

    fresh["_generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    fresh["_note"] = ("由 scripts/stamp_data_freshness.py 生成；macro/* 条目由 "
                      "scripts/update_macro_rolling.py 维护；快照/无日期列数据集按 "
                      "SNAPSHOT_MTIME / NO_DATE_EXEMPT / 文件名日期登记。")
    import os as _os
    _tmp = FRESHNESS_JSON.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(_tmp, FRESHNESS_JSON)
    print(f"✅ 扫描 {scanned} 个数据集 → {FRESHNESS_JSON}", flush=True)
    if args.json:
        print(json.dumps(fresh, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
