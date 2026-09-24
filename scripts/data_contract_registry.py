#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_contract_registry — 多源契约注册表 + 失败可见检查（W2.8 / 需求 D2-R5 方向1.5）

把「数据源 → 基座/数据集 → 消费方 → 新鲜度阈值 → 失败可见方式 → 兜底」唯一登记，
并提供 `--check` 快速自检：对每个源跑「存在性 + 新鲜度」，输出「降级源清单 + 消费方
+ 定位提示」，任一源降级即非零退出码（供 cron / 前端 / 巡检看板接入）。

复用现有 freshness 逻辑（不重复造轮子）：
  - 基座目录级新鲜度 → scripts/freshness_gate.scan_all()（读 parquet 日期列）
  - 云快照新鲜度     → quant_system.analysis_core.data_sources._cloud_snapshots_status()
  - 无日期列/子目录  → quant_system.analysis_core.data_health_check._latest_file_mtime()
  - 数据集级陈旧     → data_warehouse/data_freshness.json（stamp_data_freshness 产物）

用法:
  python3 scripts/data_contract_registry.py --check      # 快速检查（存在性 + 新鲜度）
  python3 scripts/data_contract_registry.py --dump       # 打印注册表 JSON
  python3 scripts/data_contract_registry.py --list       # 打印契约表 Markdown

退出码（--check）:
  0  全部源存在且新鲜
  1  存在降级源（缺失/陈旧）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DW = ROOT / "data_warehouse"
CST = timezone(timedelta(hours=8))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# ─────────────────────────────────────────────────────────────────────────────
# 多源契约注册表（JSON 单一事实源）
# 字段: id/name/provider/frequency/threshold_days/consumers/failure_visibility/
#       fallback/bases/datasets/check
#   check.type: gate_dir  基座目录（freshness_gate.scan_all 的 lag）
#               cloud     云快照（_cloud_snapshots_status 的 key，可多个 any）
#               mtime_dir 无日期列/子目录（_latest_file_mtime 的 mtime 天数）
# ─────────────────────────────────────────────────────────────────────────────
REGISTRY: list[dict] = [
    # ── 行情/估值/财务（腾讯云回传 + baostock 兜底）─────────────────────────
    {
        "id": "kline",
        "name": "个股日K线（前复权）",
        "provider": "腾讯云回传（update_kline_tencent 腾讯日线） + baostock 兜底",
        "frequency": "每日盘后 17:05 增量",
        "threshold_days": 5,
        "consumers": ["factors", "ml", "backtest", "data_base_bridge.style_pulse", "策略引擎"],
        "failure_visibility": "data_health_check base::kline 降权 → trust_scores；freshness_gate 门控；pull_cloud_data --only kline 退出码≠0",
        "fallback": "baostock 直连（update_kline_tencent 同源） / 腾讯云全量回传",
        "bases": ["kline"],
        "datasets": ["kline/{code}.parquet"],
        "check": {"type": "gate_dir", "dir": "kline"},
    },
    {
        "id": "valuation",
        "name": "估值（PE/PB/PS/PCF）",
        "provider": "腾讯云回传（baostock 源）",
        "frequency": "每日盘后 17:05 增量",
        "threshold_days": 5,
        "consumers": ["valuation", "ml", "fundamental", "data_base_bridge"],
        "failure_visibility": "data_health_check base::valuation 降权；update_valuation_baostock 退出码≠0",
        "fallback": "baostock 直连（update_valuation_baostock）",
        "bases": ["valuation"],
        "datasets": ["valuation/{code}.parquet"],
        "check": {"type": "gate_dir", "dir": "valuation"},
    },
    {
        "id": "financial",
        "name": "财务三表（季报，报告期≠刷新时间）",
        "provider": "腾讯云（akshare 东财，update_financial_quarterly）",
        "frequency": "季报发布后（季频）",
        "threshold_days": 7,
        "consumers": ["valuation", "fundamental", "stock_lens", "financial_data"],
        "failure_visibility": "data_health_check base::financial 降权（当前已降级可见：根因=云端旧脚本 _watch_pool dict bug，已修待刷新）",
        "fallback": "本机 akshare 直连（update_financial_quarterly）",
        "bases": ["financial"],
        "datasets": ["financial/{code}.parquet"],
        # 报告期(2026-03-31)是季度语义，新鲜度应以「最近刷新时间 mtime」衡量（refresh lag）
        "check": {"type": "mtime_dir", "dir": "financial"},
    },
    # ── 市场（短线链核心：涨停池/资金/主题/融合）───────────────────────────
    {
        "id": "market",
        "name": "市场数据（涨停池/资金流/两融/主题/融合）",
        "provider": "腾讯云回传 + 本机东财直连（push2delay）",
        "frequency": "每日盘后 18:50 回传",
        "threshold_days": 4,
        "consumers": ["emotion_cycle", "ladder", "scenario", "decision_card", "battle_map",
                      "fund_forces", "fusion", "theme_cycle"],
        "failure_visibility": "data_health_check base::market 降权；pull_cloud_data 退出码≠0（zt_pool/lhb/theme_cycle/zt_daily_stats/fusion/zt_pool_em_daily 各源）",
        "fallback": "本机 akshare 直连（ak.stock_zt_pool_em 等）",
        "bases": ["market"],
        "datasets": ["market/zt_daily_stats.parquet", "market/theme_cycle.parquet",
                     "market/fusion.parquet", "market/fund_forces.parquet",
                     "market/zt_pool_em_daily.parquet", "events/etf_state.parquet"],
        "check": {"type": "gate_dir", "dir": "market"},
    },
    {
        "id": "quarterly",
        "name": "季度机构持仓/股东/分析师",
        "provider": "腾讯云（东财，gdfx_*/analyst_*）",
        "frequency": "季频（披露后刷新）",
        "threshold_days": 120,
        "consumers": ["institutional", "data_base_bridge.institutional"],
        "failure_visibility": "data_health_check base::quarterly；无新季报时书面豁免（当前最新季 20260630 已是最新）",
        "fallback": "本机 akshare 直连；书面豁免",
        "bases": ["quarterly"],
        "datasets": ["quarterly/gdfx_*.parquet", "quarterly/analyst_*.parquet"],
        "check": {"type": "gate_dir", "dir": "quarterly"},
    },
    # ── 宏观 / 一次性 ─────────────────────────────────────────────────────
    {
        "id": "macro",
        "name": "宏观月频（CPI/PPI/PMI/M2/社融等 22 类）",
        "provider": "本机 akshare 东财 datacenter（update_macro_rolling）",
        "frequency": "月频",
        "threshold_days": 31,
        "consumers": ["macro_system", "macro_veto", "macro_calendar", "macro_overseas",
                      "data_base_bridge.macro_pulse"],
        "failure_visibility": "data_health_digest critical（CPI/PPI/M2/PMI 陈旧→退出码 2）；base::macro 降权",
        "fallback": "东财 datacenter fallback（jin10 停更后主源）；外部停发如实标注 stale",
        "bases": ["macro"],
        "datasets": ["macro/cpi_yearly.parquet", "macro/ppi_yearly.parquet", "macro/pmi_yearly.parquet",
                     "macro/m2_yearly.parquet", "macro/cx_pmi_yearly.parquet"],
        "check": {"type": "gate_dir", "dir": "macro"},
    },
    {
        "id": "oneoff",
        "name": "一次性/低频（打新/破净/拥挤度/巴菲特指数）",
        "provider": "本机 akshare 东财",
        "frequency": "事件/按需",
        "threshold_days": 60,
        "consumers": ["research_platform", "研究报告"],
        "failure_visibility": "freshness_gate（月频 cycle）；无更新如实标注",
        "fallback": "如实标注 stale（不做假新鲜）",
        "bases": ["oneoff"],
        "datasets": ["oneoff/a_below_net.parquet", "oneoff/dxsyl.parquet"],
        "check": {"type": "gate_dir", "dir": "oneoff"},
    },
    # ── 事件域 ────────────────────────────────────────────────────────────
    {
        "id": "events",
        "name": "事件（大宗/ETF资金流/龙虎榜/两融）",
        "provider": "本机东财直连",
        "frequency": "每日盘后",
        "threshold_days": 7,
        "consumers": ["event_pulse", "north_margin", "daily_report"],
        "failure_visibility": "data_health_check base::events 降权；refresh_events 退出码",
        "fallback": "腾讯云回传",
        "bases": ["events"],
        "datasets": ["events/block.parquet", "events/etf_flow.parquet", "events/margin.parquet"],
        "check": {"type": "gate_dir", "dir": "events"},
    },
    # ── 云爬虫源（腾讯云，关键云源）───────────────────────────────────────
    {
        "id": "cninfo",
        "name": "巨潮公告（官方源）",
        "provider": "腾讯云 crawler_cninfo.py（124.223.219.237）",
        "frequency": "每日 18:30 schtasks",
        "threshold_days": 7,
        "consumers": ["alternative_data（高扰动/密集披露）", "announcement_arbitrage"],
        "failure_visibility": "云快照自检 cninfo_cloud；pull_cloud_data --only cninfo 退出码（云目录不可达=-1）",
        "fallback": "akshare cninfo 直连（境外 IP 受限）",
        "bases": ["cninfo"],
        "datasets": ["cninfo/cninfo_*.json"],
        "check": {"type": "cloud", "keys": ["cninfo_cloud"]},
    },
    {
        "id": "industry",
        "name": "申万/中证/汽车行业",
        "provider": "腾讯云 crawler（申万/中证/汽车）",
        "frequency": "每日 19:00",
        "threshold_days": 5,
        "consumers": ["industry_panorama", "industry_roll", "fusion", "data_base_bridge.industry_roll"],
        "failure_visibility": "云快照自检 industry_sw/industry_csi/industry_car；update_industry_sw 退出码",
        "fallback": "本机 akshare 直连（update_industry_sw）",
        "bases": ["industry"],
        "datasets": ["industry/sw_first*.parquet", "industry/csi_industry*.parquet", "industry/car_*.parquet"],
        "check": {"type": "cloud", "keys": ["industry_sw", "industry_csi", "industry_car"], "any": True},
    },
    {
        "id": "hot_rank",
        "name": "东财热榜 top100",
        "provider": "腾讯云 crawler_hot_rank.py（emappdata POST）",
        "frequency": "每日 18:35 schtasks",
        "threshold_days": 3,
        "consumers": ["social_sentiment", "alternative_data", "data_base_bridge.hot_pulse"],
        "failure_visibility": "云快照自检 hot_rank；pull_cloud_data --only hot_rank 退出码",
        "fallback": "东财热榜直连（ak.stock_hot_rank_em）+ K线补齐涨跌幅",
        "bases": ["hot_rank"],
        "datasets": ["hot_rank/hot_rank_*.parquet"],
        "check": {"type": "cloud", "keys": ["hot_rank"]},
    },
    {
        "id": "social",
        "name": "社交舆情（微博/百度/股吧/B站/雪球/新闻）",
        "provider": "云端/本机 social_realtime + analysis_core.social_sentiment 统一采集",
        "frequency": "每日 18:40",
        "threshold_days": 7,
        "consumers": ["social_sentiment", "retail_sentiment", "data_base_bridge"],
        "failure_visibility": "云快照自检 social；social_pulse/social_realtime 退出码",
        "fallback": "东财热榜兜底社交情绪；百度失效→EMPTY 防御不覆盖历史",
        "bases": ["social"],
        "datasets": ["social/weibo_*.parquet", "social/baidu_*.parquet", "social/guba_*.parquet", "social/bilibili_*.parquet", "social/xueqiu_*.json", "market/social_sentiment.parquet"],
        "check": {"type": "cloud", "keys": ["social"]},
    },
    {
        "id": "classification",
        "name": "概念/主题成分（东财504 + THS361）",
        "provider": "腾讯云 crawler_ths_concept.py（直爬 q.10jqka.com.cn）",
        "frequency": "每日 18:45 schtasks",
        "threshold_days": 7,
        "consumers": ["theme_cycle", "industry_graph", "data_sources_v11"],
        "failure_visibility": "云快照自检 ths_concept；pull_cloud_data --only ths_concept 退出码",
        "fallback": "东财概念直连（concept_member）",
        "bases": ["classification"],
        "datasets": ["classification/concept_member_ths.parquet", "classification/concept_member.parquet"],
        "check": {"type": "cloud", "keys": ["ths_concept"]},
    },
    # ── 资金流（腾讯云，独立子目录）───────────────────────────────────────
    {
        "id": "fund_flow",
        "name": "东财个股资金流",
        "provider": "腾讯云 crawler_fund_flow.py（境外 IP 断连→云端抓取）",
        "frequency": "每日 18:45",
        "threshold_days": 3,
        "consumers": ["fund_forces", "fusion", "fund_flow_divergence"],
        "failure_visibility": "pull_cloud_data --only fund_flow 退出码；freshness_gate no_data",
        "fallback": "本机 akshare 直连（境外断连→依赖云端）",
        "bases": ["market"],
        "datasets": ["market/fund_flow/fund_flow_*.parquet"],
        "check": {"type": "mtime_dir", "dir": "market/fund_flow"},
    },
    # ── 龙虎榜 / 涨停池（本机东财直连）────────────────────────────────────
    {
        "id": "lhb_hist",
        "name": "龙虎榜历史",
        "provider": "本机东财直连（update_lhb_daily）",
        "frequency": "每日盘后",
        "threshold_days": 7,
        "consumers": ["broker_profile", "short_term_extra", "lhb 分析"],
        "failure_visibility": "data_health_check base::lhb_hist 降权；update_lhb_daily 退出码",
        "fallback": "腾讯云回传 lhb_*.parquet",
        "bases": ["lhb_hist"],
        "datasets": ["lhb_hist/lhb_*.parquet"],
        "check": {"type": "gate_dir", "dir": "lhb_hist"},
    },
    {
        "id": "zt_history",
        "name": "涨停池历史",
        "provider": "本机东财直连（ak.stock_zt_pool_em）",
        "frequency": "每日盘后",
        "threshold_days": 7,
        "consumers": ["ladder", "emotion_cycle", "short_term_extra"],
        "failure_visibility": "data_health_check base::zt_history 降权",
        "fallback": "腾讯云回传 zt_pool_*.parquet",
        "bases": ["zt_history"],
        "datasets": ["zt_history/zt_pool_history.parquet"],
        "check": {"type": "gate_dir", "dir": "zt_history"},
    },
    # ── 实时 / 名称 / 本地派生 ────────────────────────────────────────────
    {
        "id": "realtime_snapshot",
        "name": "盘中 5 分钟实时快照",
        "provider": "本机腾讯批量接口（qt.gtimg.cn）",
        "frequency": "盘中每 5 分钟（保留 60 个）",
        "threshold_days": 1,
        "consumers": ["intraday_decision", "intraday_monitor", "realtime"],
        "failure_visibility": "freshness_gate no_data（非交易时段正常）；realtime_snapshot 退出码",
        "fallback": "新浪源",
        "bases": ["realtime_snapshot"],
        "datasets": ["realtime_snapshot/YYYYMMDD/HHMMSS.parquet"],
        "check": {"type": "mtime_dir", "dir": "realtime_snapshot"},
    },
    {
        "id": "stock",
        "name": "股票名称/上市日期/ST 标记（准静态）",
        "provider": "本机（baostock/东财）",
        "frequency": "每周（名称映射）/ 上市日期准静态",
        "threshold_days": 30,
        "consumers": ["zt_pool_history", "broker_gaming", "stock_names"],
        "failure_visibility": "data_health_check stock_names 降权（名称映射）；上市日期准静态按月门控",
        "fallback": "baostock.query_all_stock",
        "bases": ["stock"],
        "datasets": ["stock/listing_dates.parquet", "market/stock_names.parquet"],
        "check": {"type": "gate_dir", "dir": "stock"},
    },
    {
        "id": "feature_store",
        "name": "因子特征宽表",
        "provider": "本地生成（build_feature_store）",
        "frequency": "每日盘后",
        "threshold_days": 5,
        "consumers": ["ml_signals", "feature_store", "ml_pipeline"],
        "failure_visibility": "freshness_gate 门控；build_feature_store 退出码",
        "fallback": "重建特征宽表",
        "bases": ["feature_store"],
        "datasets": ["feature_store/YYYYMMDD.parquet"],
        "check": {"type": "gate_dir", "dir": "feature_store"},
    },
    {
        "id": "chan_theory",
        "name": "缠论语料/索引（本地知识库）",
        "provider": "本地生成（chan_skill）",
        "frequency": "按需重建",
        "threshold_days": 30,
        "consumers": ["缠论引擎", "chan_rag"],
        "failure_visibility": "freshness_gate no_data（按需重建，非逐日新鲜度）",
        "fallback": "重新索引",
        "bases": ["chan_theory"],
        "datasets": ["chan_theory/chan_index.json", "chan_theory/chan_chapters.jsonl"],
        "check": {"type": "mtime_dir", "dir": "chan_theory"},
    },
    {
        "id": "patterns",
        "name": "形态知识库（本地知识库）",
        "provider": "本地生成",
        "frequency": "按需重建",
        "threshold_days": 30,
        "consumers": ["pattern_engine", "chart_pattern_system"],
        "failure_visibility": "freshness_gate 门控（按需重建，非逐日新鲜度）",
        "fallback": "重算形态库",
        "bases": ["patterns"],
        "datasets": ["patterns/knowledge_base.parquet"],
        "check": {"type": "gate_dir", "dir": "patterns"},
    },
    {
        "id": "predictions",
        "name": "预测/验证闭环",
        "provider": "本地生成（prediction_verify）",
        "frequency": "每日盘后",
        "threshold_days": 7,
        "consumers": ["prediction_verify", "calibration"],
        "failure_visibility": "freshness_gate 门控",
        "fallback": "回填历史预测",
        "bases": ["predictions"],
        "datasets": ["predictions/predictions.parquet"],
        "check": {"type": "gate_dir", "dir": "predictions"},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 复用现有 freshness 逻辑
# ─────────────────────────────────────────────────────────────────────────────
def _load_scan_all():
    """复用 scripts/freshness_gate.scan_all()（基座目录级 lag，读 parquet 日期列）。"""
    try:
        from scripts.freshness_gate import scan_all
        return scan_all()
    except Exception as exc:  # noqa: BLE001
        print(f"[contract] 无法加载 freshness_gate.scan_all: {exc}", file=sys.stderr)
        return {}


def _load_cloud_status():
    """复用 quant_system.analysis_core.data_sources._cloud_snapshots_status()。"""
    try:
        from quant_system.analysis_core.data_sources import _cloud_snapshots_status
        return _cloud_snapshots_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[contract] 无法加载云快照状态: {exc}", file=sys.stderr)
        return {}


def _load_mtime():
    """复用 quant_system.analysis_core.data_health_check._latest_file_mtime()。"""
    try:
        from quant_system.analysis_core.data_health_check import _latest_file_mtime
        return _latest_file_mtime
    except Exception:  # noqa: BLE001
        def _fallback(dir_path: Path) -> float | None:
            try:
                if not dir_path.is_dir():
                    return None
                latest: float | None = None
                for p in dir_path.rglob("*"):
                    if not p.is_file():
                        continue
                    try:
                        t = p.stat().st_mtime
                    except OSError:
                        continue
                    if latest is None or t > latest:
                        latest = t
                return latest
            except OSError:
                return None
        return _fallback


def _age_days(as_of: str | None) -> int | None:
    """YYYYMMDD 字符串 → 距今自然日天数（解析失败返回 None）。"""
    if not as_of:
        return None
    s = str(as_of)
    try:
        d = datetime.strptime(s[:8], "%Y%m%d").replace(tzinfo=CST)
        return max(0, (datetime.now(CST) - d).days)
    except Exception:  # noqa: BLE001
        return None


def check_contract(contract: dict, scan: dict, cloud: dict, latest_mtime) -> dict:
    """对单个源契约跑「存在性 + 新鲜度」，返回结构化结果。"""
    cid = contract["id"]
    thr = int(contract.get("threshold_days", 7))
    spec = contract.get("check", {})
    ctype = spec.get("type")
    now = datetime.now(CST)

    if ctype == "cloud":
        keys = spec.get("keys", [])
        sts = [cloud.get(k) for k in keys if cloud.get(k)]
        if not sts:
            return {"id": cid, "exists": False, "fresh": False, "age_days": None,
                    "detail": f"云快照无记录: {', '.join(keys)}", "as_of": None}

        def _fmt(k: str, s: dict) -> str:
            age = _age_days(s.get("as_of"))
            return (f"{k}: as_of={s.get('as_of')} 距今{age if age is not None else '?'}天 "
                    f"n={s.get('n_files', 0)}")

        parts = [_fmt(k, s) for k, s in zip(keys, sts)]
        if spec.get("any"):
            # 任一子源新鲜即算新鲜（行业三源互为冗余）
            ages = [a for a in (_age_days(s.get("as_of")) for s in sts) if a is not None]
            best_age = min(ages) if ages else None
            exists = any(s.get("n_files", 0) > 0 for s in sts)
            fresh = best_age is not None and best_age <= thr
            return {"id": cid, "exists": exists, "fresh": fresh, "age_days": best_age,
                    "detail": " | ".join(parts) + f"（阈值 {thr} 天）",
                    "as_of": None}
        st = sts[0]
        exists = st.get("n_files", 0) > 0
        age = _age_days(st.get("as_of"))
        fresh = age is not None and age <= thr
        return {"id": cid, "exists": exists, "fresh": fresh, "age_days": age,
                "detail": parts[0] + f"（阈值 {thr} 天）", "as_of": st.get("as_of")}

    if ctype == "gate_dir":
        d = spec.get("dir")
        v = scan.get(d) if isinstance(scan, dict) else None
        if not v:
            return {"id": cid, "exists": False, "fresh": False, "age_days": None,
                    "detail": f"目录 {d} 不在 freshness_gate 扫描结果（缺失或非基座）", "as_of": None}
        lag = v.get("lag")
        if lag is None or lag < 0:
            return {"id": cid, "exists": False, "fresh": False, "age_days": None,
                    "detail": f"目录 {d} 缺失或空（no_data）", "as_of": None}
        fresh = lag <= thr
        return {"id": cid, "exists": True, "fresh": fresh, "age_days": int(lag),
                "detail": f"最新 {v.get('latest')}（滞后 {lag} 天，阈值 {thr}）",
                "as_of": v.get("latest")}

    if ctype == "mtime_dir":
        d = spec.get("dir")
        m = latest_mtime(DW / d)
        if m is None:
            return {"id": cid, "exists": False, "fresh": False, "age_days": None,
                    "detail": f"目录 {d} 缺失或空", "as_of": None}
        as_of = datetime.fromtimestamp(m, tz=CST).strftime("%Y-%m-%d")
        # 盘中快照等按交易日衡量：周末/节假日不得把周五收盘误报为陈旧。
        try:
            from quant_system.market_clock import get_trade_calendar
            calendar = sorted(get_trade_calendar())
            today_day = datetime.now(CST).strftime("%Y-%m-%d")
            age = max(0, sum(as_of < x <= today_day for x in calendar))
            age_unit = "交易日"
        except Exception:
            age = max(0, (datetime.now(CST).timestamp() - m) / 86400.0)
            age_unit = "天"
        fresh = age <= thr
        return {"id": cid, "exists": True, "fresh": fresh, "age_days": round(age, 1),
                "detail": f"最新写入 {as_of}（距今 {age:.1f} {age_unit}，阈值 {thr}）", "as_of": as_of}

    return {"id": cid, "exists": False, "fresh": False, "age_days": None,
            "detail": f"未知 check 类型 {ctype}", "as_of": None}


def run_check(verbose: bool = True) -> dict:
    """跑全部源契约检查，返回汇总 + 每源结果。"""
    scan = _load_scan_all()
    cloud = _load_cloud_status()
    latest_mtime = _load_mtime()

    results = []
    for c in REGISTRY:
        results.append(check_contract(c, scan, cloud, latest_mtime))

    degraded = [r for r in results if not r["exists"] or not r["fresh"]]
    missing = [r for r in degraded if not r["exists"]]
    stale = [r for r in degraded if r["exists"] and not r["fresh"]]

    summary = {
        "total": len(results),
        "ok": len(results) - len(degraded),
        "degraded": len(degraded),
        "missing": len(missing),
        "stale": len(stale),
        "results": results,
        "degraded_ids": [r["id"] for r in degraded],
    }
    if verbose:
        _print_report(summary, degraded)
    return summary


def _print_report(summary: dict, degraded: list[dict]) -> None:
    by_id = {c["id"]: c for c in REGISTRY}
    lines = [
        f"📋 多源契约检查 | 共 {summary['total']} 源 | 正常 {summary['ok']} | "
        f"降级 {summary['degraded']}（缺失 {summary['missing']} / 陈旧 {summary['stale']}）",
        "-" * 78,
    ]
    for r in summary["results"]:
        icon = "✅" if (r["exists"] and r["fresh"]) else ("❌" if not r["exists"] else "⚠️")
        c = by_id[r["id"]]
        lines.append(f"  {icon} {r['id']:<18} {c['name']:<24} {r['detail']}")
    if degraded:
        lines.append("")
        lines.append("## 降级源清单（10 秒定位：哪个源 → 为什么 → 影响谁）")
        lines.append("")
        for r in degraded:
            c = by_id[r["id"]]
            kind = "缺失" if not r["exists"] else "陈旧"
            lines.append(f"### {'❌' if not r['exists'] else '⚠️'} {r['id']} — {c['name']}（{kind}）")
            lines.append(f"  - 原因: {r['detail']}")
            lines.append(f"  - 提供方: {c['provider']}")
            lines.append(f"  - 更新频率: {c['frequency']} | 阈值: {c['threshold_days']} 天")
            lines.append(f"  - 影响消费方: {', '.join(c['consumers'])}")
            lines.append(f"  - 可见/告警: {c['failure_visibility']}")
            lines.append(f"  - 兜底: {c['fallback']}")
            lines.append("")
    print("\n".join(lines))


def dump_registry() -> str:
    return json.dumps(REGISTRY, ensure_ascii=False, indent=2)


def list_markdown() -> str:
    """契约表 Markdown（与落盘文档一致）。"""
    lines = [
        "| 源/基座 | 提供方 | 更新频率 | 新鲜阈值 | 主要消费方 | 失败可见方式 | 兜底 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in REGISTRY:
        consumers = "、".join(c["consumers"][:5]) + ("…" if len(c["consumers"]) > 5 else "")
        lines.append(
            f"| **{c['id']}** {c['name']} | {c['provider']} | {c['frequency']} | "
            f"{c['threshold_days']} 天 | {consumers} | {c['failure_visibility']} | {c['fallback']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="多源契约注册表 + 失败可见检查（W2.8）")
    ap.add_argument("--check", action="store_true", help="对每个源跑存在性 + 新鲜度检查")
    ap.add_argument("--dump", action="store_true", help="打印注册表 JSON")
    ap.add_argument("--list", action="store_true", help="打印契约表 Markdown")
    args = ap.parse_args(argv)

    if args.dump:
        print(dump_registry())
        return 0
    if args.list:
        print(list_markdown())
        return 0
    if args.check:
        summary = run_check(verbose=True)
        return 1 if summary["degraded"] > 0 else 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
