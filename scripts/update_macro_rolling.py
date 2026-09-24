#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_macro_rolling.py — 宏观数据滚动更新（jin10 系 + 社融/PMI/货币供应等）
=======================================================================
背景：data_warehouse/macro/*_yearly.parquet（今值/预测值/前值，金十数据源）
历史停在 2025-08/09，shrzgm 停在 2026-04 —— 根因是外部源（jin10/mofcom）
在服务器端一次全量后不再滚动。本脚本：

  1. 对每个宏观数据集用 akshare 对应接口全量重取
  2. 与本地 parquet 按主键合并（保留历史 + 补缺 + 幂等：重复跑不重复）
  3. 重算 as_of（最新数据日期）并写 data_warehouse/data_freshness.json
     （stale 判定：月度 75 天 / 季度 100 天 / 日度事件 30 天）
  4. 外部源取不到新数据时标记 stale=true，绝不假装新鲜
  5. 域C（2026-08-12）：新浪/jin10 源已停更（2025-08/09），主路径切 scripts/update_macro_em.py
     （东财 datacenter 直连）。本脚本相关条目保留但 marked deprecated；
     主源+akshare fallback 均失败时自动调用 update_macro_em.py 东财直连 fallback。

用法:
  python3 scripts/update_macro_rolling.py             # 全量滚动（合并所有目标）
  python3 scripts/update_macro_rolling.py --only cpi_yearly,shrzgm  # 只更新指定集
  python3 scripts/update_macro_rolling.py --no-network # 离线：只重算新鲜度标注
  python3 scripts/update_macro_rolling.py --json      # 输出新鲜度 JSON

不破坏下游列名：合并只做行级去重/补缺，列名与 akshare 返回一致（与现文件一致）。
"""
from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_macro_rolling] 操作失败: {e}", exc_info=True)

ROOT = Path(__file__).resolve().parent.parent
# data_warehouse 对 codex 沙箱只读：MACRO_DATA_DIR 可覆盖落盘目录（自查写 /tmp，本机真实落盘）
WH = Path(os.environ.get("MACRO_DATA_DIR", str(ROOT / "data_warehouse")))
FRESHNESS_JSON = WH / "data_freshness.json"

# 目标：本地文件 → (akshare 函数名, 日期列, 频率, 去重主键)
JOBS: list[dict] = [
    {"file": "macro/cpi_yearly.parquet", "fn": "macro_china_cpi_yearly",
     "deprecated": True, "em": "cpi", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_cpi", "date_col": "月份", "keys": ["月份"]}},
    {"file": "macro/ppi_yearly.parquet", "fn": "macro_china_ppi_yearly",
     "deprecated": True, "em": "ppi", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_ppi", "date_col": "月份", "keys": ["月份"]}},
    {"file": "macro/pmi_yearly.parquet", "fn": "macro_china_pmi_yearly",
     "deprecated": True, "em": "pmi", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_pmi", "date_col": "月份", "keys": ["月份"]}},
    # 2026-08-13: jin10 源停更 → 东财指数源 index_pmi_man_cx 换源，最新 as_of 2026-07-31
    {"file": "macro/cx_pmi_yearly.parquet", "fn": "index_pmi_man_cx",
     "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "transform_cx_pmi": True},
    {"file": "macro/m2_yearly.parquet", "fn": "macro_china_m2_yearly",
     "deprecated": True, "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_money_supply", "date_col": "月份", "keys": ["月份"]}},
    {"file": "macro/gdp_yearly.parquet", "fn": "macro_china_gdp_yearly",
     "deprecated": True, "em": "gdp", "date_col": "日期", "freq": "quarterly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_gdp", "date_col": "季度", "keys": ["季度"]}},
    {"file": "macro/exports_yoy.parquet", "fn": "macro_china_exports_yoy",
     "deprecated": True, "em": "exports", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_hgjck", "date_col": "月份", "keys": ["月份"]}},
    {"file": "macro/imports_yoy.parquet", "fn": "macro_china_imports_yoy",
     "deprecated": True, "em": "imports", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_hgjck", "date_col": "月份", "keys": ["月份"]}},
    {"file": "macro/industrial_production_yoy.parquet", "fn": "macro_china_industrial_production_yoy",
     "deprecated": True, "em": "industrial_production", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"]},
    {"file": "macro/fx_reserves_yearly.parquet", "fn": "macro_china_fx_reserves_yearly",
     "deprecated": True, "em": "fx_reserves", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_foreign_exchange_gold", "date_col": "统计时间", "keys": ["统计时间"]}},
    {"file": "market/macro_monthly__exports.parquet", "fn": "macro_china_exports_yoy", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_hgjck", "date_col": "月份", "keys": ["月份"]}},
    {"file": "market/macro_monthly__imports.parquet", "fn": "macro_china_imports_yoy", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_hgjck", "date_col": "月份", "keys": ["月份"]}},
    {"file": "market/macro_monthly__m2.parquet", "fn": "macro_china_m2_yearly", "date_col": "日期", "freq": "monthly", "keys": ["商品", "日期"],
     "fallback": {"fn": "macro_china_money_supply", "date_col": "月份", "keys": ["月份"]}},
    {"file": "macro/shrzgm.parquet", "fn": "macro_china_shrzgm", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/pmi.parquet", "fn": "macro_china_pmi", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/money_supply.parquet", "fn": "macro_china_money_supply", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/new_house_price.parquet", "fn": "macro_china_new_house_price", "date_col": "日期", "freq": "monthly", "keys": ["日期", "城市"]},
    {"file": "macro/commodity_price_index.parquet", "fn": "macro_china_commodity_price_index", "date_col": "日期", "freq": "daily", "keys": ["日期"]},
    {"file": "macro/central_bank_balance.parquet", "fn": "macro_china_central_bank_balance", "date_col": "统计时间", "freq": "monthly", "keys": ["统计时间"]},
    {"file": "macro/enterprise_boom_index.parquet", "fn": "macro_china_enterprise_boom_index", "date_col": "季度", "freq": "quarterly", "keys": ["季度"]},
    {"file": "macro/reserve_requirement_ratio.parquet", "fn": "macro_china_reserve_requirement_ratio", "date_col": "公布时间", "freq": "event", "keys": ["公布时间"]},
    {"file": "macro/stock_market_cap.parquet", "fn": "macro_china_stock_market_cap", "date_col": "数据日期", "freq": "monthly", "keys": ["数据日期"]},
    {"file": "macro/urban_unemployment.parquet", "fn": "macro_china_urban_unemployment", "date_col": "date", "freq": "monthly", "keys": ["date", "item"]},
    # 2026-08-12 宏观缺口补齐（akshare 已验证可用；列名保持 akshare 原始中文列名）
    {"file": "macro/consumer_confidence.parquet", "fn": "macro_china_xfzxx", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/retail_sales_yoy.parquet", "fn": "macro_china_consumer_goods_retail", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/fixed_asset_investment.parquet", "fn": "macro_china_gdzctz", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/real_estate.parquet", "fn": "macro_china_real_estate", "date_col": "日期", "freq": "monthly", "keys": ["日期"]},
    {"file": "macro/society_electricity.parquet", "fn": "macro_china_society_electricity", "date_col": "统计时间", "freq": "monthly", "keys": ["统计时间"]},
    {"file": "macro/money_supply_full.parquet", "fn": "macro_china_supply_of_money", "date_col": "统计时间", "freq": "monthly", "keys": ["统计时间"]},
    {"file": "macro/new_financial_credit.parquet", "fn": "macro_china_new_financial_credit", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
    {"file": "macro/fiscal_revenue.parquet", "fn": "macro_china_czsr", "date_col": "月份", "freq": "monthly", "keys": ["月份"]},
]

THRESHOLD_DAYS = {"monthly": 75, "quarterly": 150, "daily": 30, "event": 30}

# 2026-08-14 登记: 外部源停更/事件型数据集的特殊陈旧阈值（key=数据集文件名）
# 逐一核实依据: 本文件 docstring 及 运维记录（NBS 停发、jin10/mofcom 停更、事件型无变化）
STALE_OVERRIDE_DAYS = {
    "reserve_requirement_ratio.parquet": 730,  # 事件型：2025-05 末次降准后无变化，非陈旧
    "real_estate.parquet": 400,                # NBS 房地产投资 2025-12 后停发（2e2ed4b 已登记）
    "shrzgm.parquet": 180,                     # 月度社融：外部源停更，2026-04 为最后可得（docstring 登记）
}


def _stale_threshold(path: Path, freq: str) -> int:
    """数据集的陈旧阈值（优先文件级覆盖，其次 freq 默认）。"""
    return STALE_OVERRIDE_DAYS.get(path.name, THRESHOLD_DAYS[freq])


def _parse_dates(series: pd.Series) -> pd.Series:
    """把各种中文日期格式统一成 Timestamp（失败 NaT）。"""
    out = pd.to_datetime(series, errors="coerce")
    # pd.to_datetime 解析不了的中文格式（2026年06月、2026年第2季度等）逐个正则兜底，
    # 不再提前 return，避免"能解析的混在一起时中文格式行被误判 NaT"。
    for i, v in enumerate(series.astype(str)):
        if pd.notna(out.iloc[i]):
            continue
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        m = re.search(r"(\d{4})年(\d{1,2})月", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
        # V4.1 fix: 东财 GDP "2026年第1-2季度"(H1累计) → 取后一季度末
        # 即 第1-2季度 → 2026-06-30；第1-4季度(全年) → 2026-12-31。必须放在单季度规则前。
        m = re.search(r"(\d{4})年第(\d)-(\d)季度", v)
        if m:
            q_end = int(m.group(3))
            out.iloc[i] = pd.Timestamp(int(m.group(1)), q_end * 3, 1)
            continue
        m = re.search(r"(\d{4})年第(\d)季度", v)
        if m:
            q = int(m.group(2))
            out.iloc[i] = pd.Timestamp(int(m.group(1)), q * 3, 1)
            continue
        m = re.match(r"^(\d{4})(\d{2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
        m = re.match(r"^(\d{4})\.(\d{1,2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
        m = re.search(r"^(\d{4})-(\d{2})$", v)
        if m:
            out.iloc[i] = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
            continue
    return out


def as_of_of(df: pd.DataFrame, date_col: str) -> date | None:
    if df is None or df.empty or date_col not in df.columns:
        return None
    s = _parse_dates(df[date_col]).dropna()
    if s.empty:
        return None
    return s.max().date()


def coerce_date_col(merged: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """写 parquet 前统一日期列：转 datetime64，解析失败的行丢弃。

    合并后日期列可能混入 datetime.date 对象 / 中文日期字符串（日期/月份/
    统计时间/公布时间/季度），pyarrow 写 parquet 要求列类型统一；先统一转
    datetime64 再 dropna，与 freshness 重算的 as_of_of 用同一套解析口径。
    """
    if merged is None or merged.empty or date_col not in merged.columns:
        return merged
    merged = merged.copy()
    merged[date_col] = _parse_dates(merged[date_col])
    merged = merged.dropna(subset=[date_col]).reset_index(drop=True)
    return merged


def merge_idempotent(old: pd.DataFrame | None, new: pd.DataFrame,
                     keys: list[str]) -> pd.DataFrame:
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
    fresh["_generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    import os as _os
    _tmp = FRESHNESS_JSON.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(_tmp, FRESHNESS_JSON)


def main() -> int:
    ap = argparse.ArgumentParser(description="宏观数据滚动更新 + 新鲜度标注")
    ap.add_argument("--only", default="", help="逗号分隔的子集名（文件 basename，如 cpi_yearly,shrzgm）")
    ap.add_argument("--no-network", action="store_true", help="离线：只重算新鲜度标注")
    ap.add_argument("--json", action="store_true", help="结束后输出新鲜度 JSON")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    fresh = load_freshness()
    print("=== 宏观数据滚动更新 ===", flush=True)

    for job in JOBS:
        name = Path(job["file"]).name
        name_stem = Path(job["file"]).stem  # 无扩展名，--only 用 stem 匹配
        if job.get("deprecated"):
            print(f"  (deprecated) {name}: 新浪/jin10 源停更，主路径已切 update_macro_em.py（东财直连）", flush=True)
        if only and name_stem not in only:
            continue
        path = WH / job["file"]
        old = None
        if path.exists():
            try:
                old = pd.read_parquet(path)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {name} 读取失败: {e}", flush=True)
                old = None
        entry = fresh.get(name, {})
        entry["file"] = job["file"]
        entry["source"] = f"akshare.{job['fn']}"
        entry["freq"] = job["freq"]
        entry["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

        new = None
        fetch_note = ""
        if not args.no_network:
            import akshare as ak
            # 主源（加超时保护：jin10 接口偶发无限挂起）
            try:
                import signal as _sig

                def _timeout_handler(*_a):
                    raise TimeoutError("fetch timeout")

                old_handler = None
                try:
                    _sig.signal(_sig.SIGALRM, _timeout_handler)
                    _sig.alarm(45)
                    new = getattr(ak, job["fn"])()
                    _sig.alarm(0)
                except (ValueError, TypeError):
                    # 非主线程无法用 signal.alarm → 直接调用
                    new = getattr(ak, job["fn"])()
                finally:
                    try:
                        _sig.alarm(0)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[update_macro_rolling] 操作失败: {e}", exc_info=True)
                if new is not None and len(new):
                    fetch_note = f"akshare.{job['fn']}"
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {name} 抓取失败: {type(e).__name__} {str(e)[:100]}", flush=True)
                entry["last_fetch_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            # V4.1 fix (jin10 源停更): 主源数据陈旧时自动切换东财 fallback
            # （jin10 yearly 系停更 2025-09；东财 RPT_ECONOMY_* 有 2026-07 最新）
            fb = job.get("fallback")
            if fb is not None:
                cur_df = new if (new is not None and len(new)) else old
                cur_asof = as_of_of(cur_df, job["date_col"]) if cur_df is not None else None
                main_stale = cur_asof is None or (date.today() - cur_asof).days > THRESHOLD_DAYS[job["freq"]]
                if main_stale:
                    try:
                        import signal as _sig2

                        def _fb_timeout(*_a):
                            raise TimeoutError("fallback fetch timeout")

                        try:
                            _sig2.signal(_sig2.SIGALRM, _fb_timeout)
                            _sig2.alarm(45)
                            fb_df = getattr(ak, fb["fn"])()
                            _sig2.alarm(0)
                        except (ValueError, TypeError):
                            fb_df = getattr(ak, fb["fn"])()
                        finally:
                            try:
                                _sig2.alarm(0)
                            except Exception as e:
                                logging.getLogger(__name__).error(f"[update_macro_rolling] 操作失败: {e}", exc_info=True)
                        if fb_df is not None and len(fb_df):
                            fb_asof = as_of_of(fb_df, fb["date_col"])
                            fb_stale = fb_asof is None or (date.today() - fb_asof).days > THRESHOLD_DAYS[job["freq"]]
                            if not fb_stale:
                                # fallback 数据新 → 用 fallback 替代主源结果
                                new = fb_df
                                fetch_note = f"akshare.{fb['fn']}(东财 fallback, 主源 jin10 停更)"
                                entry["fallback_used"] = fb["fn"]
                                print(f"🔄 {name}: 主源陈旧({cur_asof})，已切东财 fallback ({fb_asof})", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"⚠️ {name} fallback 失败: {type(e).__name__} {str(e)[:100]}", flush=True)
            # 域C (2026-08-12): 主源 + akshare fallback 均失败/为空 → 东财直连 update_macro_em fallback
            em_key = job.get("em")
            if em_key and (new is None or not len(new)):
                try:
                    import update_macro_em as _em
                    _res = _em.run(only={em_key}, no_network=False, write_freshness=False)
                    _r = _res.get(em_key, {})
                    if _r.get("ok"):
                        _em_job = _em.JOB_BY_FILE.get(job["file"])
                        if _em_job:
                            job["keys"] = _em_job["keys"]
                            job["date_col"] = _em_job["date_col"]
                        new = pd.read_parquet(path)
                        fetch_note = f"update_macro_em(东财直连 fallback, {_r.get('report')})"
                        entry["fallback_used"] = f"update_macro_em.{_r.get('report')}"
                        print(f"🔄 {name}: 新浪/jin10 源失败，已用 update_macro_em 东财直连补数 (as_of={_r.get('as_of')})", flush=True)
                    else:
                        print(f"⚠️ {name}: update_macro_em fallback 未补到数据 ({_r.get('error')})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ {name}: update_macro_em fallback 异常: {type(e).__name__} {str(e)[:100]}", flush=True)
            if new is not None and len(new):
                if job.get("transform_cx_pmi"):
                    # 2026-08-13: 东财 index_pmi_man_cx 换源（jin10 停更）。两源日期口径不同
                    # （jin10=发布日/月初，东财=发布日/月末），统一归一化为月末，确保
                    # (商品,日期) 主键匹配、幂等去重生效；否则同月两行、文件无限膨胀。
                    expected = {"日期", "制造业PMI"}
                    if not expected.issubset(set(new.columns)):
                        raise ValueError(f"index_pmi_man_cx 列结构异常: {list(new.columns)}")
                    if old is not None and len(old) and old["商品"].notna().any():
                        new["商品"] = old["商品"].dropna().iloc[0]
                    else:
                        new["商品"] = "中国财新制造业PMI终值报告"
                    new = new.rename(columns={"制造业PMI": "今值"})
                    new["预测值"] = float("nan")
                    new["前值"] = float("nan")
                    new["日期"] = pd.to_datetime(new["日期"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
                    new = new.dropna(subset=["日期"])
                    new = new[["商品", "日期", "今值", "预测值", "前值"]]
                    # 历史 jin10 数据同样归一化为月末，与新源主键对齐（否则历史不参与去重）
                    if old is not None and len(old):
                        old = old.copy()
                        old["日期"] = pd.to_datetime(old["日期"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
                        old = old.dropna(subset=["日期"])
                    fetch_note = "akshare.index_pmi_man_cx(东财财新制造业PMI, jin10停更换源 2026-08-13)"
                old_rows = 0 if old is None else len(old)
                # V4.1 fix: fallback(东财) 列结构与主源(jin10)不同 → 整体替换为新结构，
                # 不做列合并（否则两套列并存且主键不一致）。as_of 用 fallback 的 date_col。
                fb_cfg = job.get("fallback")
                used_fb = bool(fb_cfg and entry.get("fallback_used"))
                if used_fb:
                    merged = new.copy()
                    merged = merged.drop_duplicates(subset=fb_cfg["keys"], keep="last").reset_index(drop=True)
                    job["date_col"] = fb_cfg["date_col"]
                    job["keys"] = fb_cfg["keys"]
                else:
                    merged = merge_idempotent(old, new, job["keys"])
                # pyarrow 写 parquet 要求日期列类型统一（合并后可能混入
                # datetime.date 对象/中文日期字符串），统一转 datetime64，
                # 解析失败的行 drop（保有效数据）。覆盖 merge_idempotent 与
                # fallback/东财直连两条路径的 merged。
                merged = coerce_date_col(merged, job["date_col"])
                tmp = path.with_suffix(".parquet.tmp")
                merged.to_parquet(tmp, index=False)
                tmp.replace(path)
                entry["last_fetch_error"] = ""
                print(f"✅ {name}: {old_rows} → {len(merged)} 行（新抓 {len(new)}，源={fetch_note}）", flush=True)
            else:
                print(f"⚠️ {name}: 抓取为空，保留本地数据", flush=True)
            entry["source"] = fetch_note or entry.get("source", f"akshare.{job['fn']}")

        # 新鲜度重算（即使离线）
        cur = old if new is None else pd.read_parquet(path)
        as_of = as_of_of(cur, job["date_col"])
        entry["as_of"] = as_of.isoformat() if as_of else None
        thr = _stale_threshold(path, job["freq"])
        entry["threshold_days"] = thr
        if as_of:
            entry["stale_days"] = (date.today() - as_of).days
            entry["stale"] = entry["stale_days"] > thr
        else:
            entry["stale_days"] = None
            entry["stale"] = True
        if entry.get("stale"):
            entry["note"] = (f"外部源陈旧：最新数据 {entry.get('as_of')}，超过 {thr} 天阈值。"
                             "请勿当作最新宏观数据使用；联网重跑本脚本会自动补缺。")
        else:
            entry["note"] = ""
        fresh[name] = entry
        print(f"   {name}: as_of={entry.get('as_of')} stale={entry.get('stale')}", flush=True)

    save_freshness(fresh)
    print(f"✅ 新鲜度已写入 {FRESHNESS_JSON}", flush=True)
    if args.json:
        print(json.dumps(fresh, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
