"""
pipeline — V11 每日数据流水线（cron 入口）

滚动式数据更新，全部幂等（重复执行不产生脏数据）：

  daily_update():
    1. 股票名称映射刷新（>30天才重拉）
    2. EM 三池当日抓取（精确数据: 封板资金/炸板次数/行业）
    3. K线滚动增量重建最近 10 交易日（涨停/炸板/跌停/连板长表）
    4. 天梯滚动更新最近 12 交易日（情绪周期/晋级率地基）
    5. 情绪周期判定 + 预测入库（后续阶段逐步挂载）

用法:
  python3 -m quant_system.analysis_core.pipeline --daily   # 每日盘后
  python3 -m quant_system.analysis_core.pipeline --status
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import zt_pool_history, ladder, emotion_cycle  # noqa: E402
from quant_system.context import RunContext


def _write_generated(name: str, payload: str | dict) -> None:
    """写 generated/ 产物；失败仅记录日志，不阻断 pipeline 主流程。"""
    try:
        out_dir = ROOT / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / name
        if isinstance(payload, dict):
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")
        else:
            out.write_text(payload, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(f"[pipeline] 落盘失败 {name}: {e}")


def _multi_agent_summary(date: str) -> dict:
    """battle_map 产物 json 附 multi_agent 摘要（pipeline 不单独调度 multi_agent）。

    优先复用 battle_map 渲染时已落盘的 multi_agent_{date}.json；缺失时调用
    arbitrate（其自身会落盘 multi_agent_{date}.json），保证云端产物齐全。
    """
    try:
        p = ROOT / "generated" / f"multi_agent_{date}.json"
        if p.exists():
            ma = json.loads(p.read_text(encoding="utf-8"))
        else:
            from quant_system.analysis_core import multi_agent  # noqa: PLC0415
            ma = multi_agent.arbitrate(date)
        return {k: ma.get(k) for k in (
            "date", "consensus", "confidence", "weighted_sum",
            "votes", "disagreement", "weights_note",
        )}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:120]}


def _validate_core_dates(core_dates: dict[str, str], expected_day: str) -> dict:
    """Require every core artifact to match the completed target trading day."""
    required = {"zt_daily_stats", "fusion"}
    missing = sorted(required - set(core_dates))
    stale = {key: value for key, value in core_dates.items() if value < expected_day}
    future = {key: value for key, value in core_dates.items() if value > expected_day}
    return {"expected": expected_day, "core_dates": core_dates, "missing": missing,
            "stale": stale, "future": future, "ok": not (missing or stale or future)}


def daily_update(*, context: RunContext | None = None, as_of: str | None = None, run_id: str | None = None) -> dict:
    """每日盘后全链路；旧的无参调用仍使用最近完成交易日。"""
    run_context = context or RunContext.create(as_of=as_of, run_id=run_id)
    report: dict = {"run_id": run_context.run_id, "as_of": run_context.iso_date}
    # 1-4. 数据地基（2026-08-21 审计: 步骤1-4原裸露，网络抓取(步骤2)异常会中断整链，
    #      下游情绪/融合/日报全不执行 → 每步独立 try/except，失败记 *_error 不中断）
    try:
        nm = zt_pool_history.ensure_name_map()
        report["name_map"] = len(nm)
    except Exception as e:
        report["name_map_error"] = str(e)[:200]
    try:
        report["em_today"] = zt_pool_history.update_today()
    except Exception as e:
        report["em_today_error"] = str(e)[:200]
    try:
        tail = zt_pool_history.incremental_kline(days=10)
        report["kline_tail_rows"] = len(tail)
    except Exception as e:
        report["kline_error"] = str(e)[:200]
    try:
        stats = ladder.build_daily_stats(recent_days=12)
        report["stats_days"] = len(stats)
    except Exception as e:
        report["stats_error"] = str(e)[:200]
    # 5. 核心市场数据门禁延后到 fund_forces/fusion 重建之后。
    # 旧顺序先检查 fusion 再重建 fusion，导致每天永远消费上一交易日派生产物。
    try:
        from quant_system.market_clock import latest_completed_trading_day
        expected_day = run_context.iso_date or latest_completed_trading_day().isoformat()
        report["target_day"] = expected_day
        report["as_of"] = expected_day
        from quant_system.data_release import load_release
        release = load_release(ROOT, expected_day=expected_day)
        report["release_id"] = release["release_id"]
    except Exception as exc:
        report["data_gate_error"] = str(exc)[:180]
        report["analysis_error"] = "无法确定最近完成交易日，跳过日报生成"
        return report
    # 5. 情绪周期 + 题材生命周期 + 信号融合 + 推演 + 决策 + 日报
    try:
        report["emotion"] = emotion_cycle.run_today(expected_day)
    except Exception as e:
        report["emotion_error"] = str(e)[:200]
    # 6. 另类数据（云源+本地代理）摘要（只存 status/summary，不塞全文）
    try:
        from quant_system.analysis_core import alternative_data
        ad = alternative_data.run_report()
        cloud = ad.get("cloud_sources", {}) or {}
        proxies = ad.get("local_proxies", {}) or {}
        report["alternative_data"] = {
            "date": ad.get("date"),
            "cloud_status": cloud.get("status"),
            "hot_rank_status": (cloud.get("hot_rank") or {}).get("status"),
            "social_status": (cloud.get("social") or {}).get("status"),
            "local_signal": proxies.get("signal"),
        }
    except Exception as e:
        report["alternative_data_error"] = str(e)[:200]
    # 7. 短线输入先刷新：社交快照、资金合力都必须在 fusion 前落盘。
    try:
        from quant_system.analysis_core import social_sentiment
        social_sentiment.store(social_sentiment.collect_all())
        report["social_sentiment_refreshed"] = True
    except Exception as e:
        report["social_sentiment_error"] = str(e)[:200]
    # 8. 资金合力重建：当天新算的资金合力必须进入当天融合温度。
    try:
        from quant_system.analysis_core import fund_forces
        fund_forces.build_forces(days=90)
        report["fund_forces_rows"] = len(fund_forces.latest())
    except Exception as e:
        report["fund_forces_error"] = str(e)[:200]
    # 9. 题材生命周期 + 信号融合（fusion 落盘，行业广度摘要从 fusion 结果取）
    try:
        from quant_system.analysis_core import theme_cycle, fusion
        tc = theme_cycle.build_theme_cycle(days=60)
        report["theme_rows"] = len(tc)
        fu = fusion.fuse_today()
        fusion.store(fu)
        ind = fu.get("industry") or {}
        report["fusion"] = {
            "date": fu.get("date"),
            "temperature": fu.get("temperature"),
            "tag": fu.get("tag"),
            "emotion_stage": fu.get("emotion_stage"),
            "alt_status": (fu.get("alt") or {}).get("status"),
            "industry_breadth": ind.get("breadth"),
            "industry_top3": [{"name": x.get("name"), "pct": x.get("pct")}
                              for x in ind.get("top_up", [])],
        }
        report["fusion_temp"] = fu["temperature"]
        report["fusion_tag"] = fu["tag"]
    except Exception as e:
        report["theme_error"] = str(e)[:200]
    # fusion 已按本轮输入重建后才执行门禁，避免旧 fusion 阻断新数据。
    try:
        import pandas as _pd
        core_dates = {}
        for key, path in (("zt_daily_stats", ROOT / "data_warehouse" / "market" / "zt_daily_stats.parquet"),
                          ("fusion", ROOT / "data_warehouse" / "market" / "fusion.parquet")):
            if path.exists():
                _df = _pd.read_parquet(path, columns=["date"])
                if not _df.empty:
                    core_dates[key] = str(_pd.to_datetime(_df["date"]).max().date())
        gate = _validate_core_dates(core_dates, expected_day)
        if not gate["ok"]:
            report["data_stale"] = {**gate, "action": "skip_analysis"}
            report["analysis_error"] = "关键市场数据日期与目标交易日不一致，跳过日报生成"
            return report
    except Exception as exc:
        report["data_gate_error"] = str(exc)[:180]
        report["analysis_error"] = "数据日期门禁失败，跳过日报生成"
        return report
    # 10. 推演 + 决策 + 日报
    try:
        from quant_system.analysis_core import scenario, decision_card, daily_report
        fc = scenario.scenario_forecast(as_of=expected_day)
        ids = scenario.make_predictions(fc)
        card = decision_card.build_card(as_of=expected_day)
        report["predictions_added"] = len(ids)
        report["stance"] = card["stance"]
        md = daily_report.build_report(as_of=expected_day)
        out = ROOT / "generated" / "short_term_daily.md"
        out.write_text(md, encoding="utf-8")
        report["report"] = str(out)
    except Exception as e:
        report["analysis_error"] = str(e)[:300]
    # 10. 盘后增强: 龙虎榜短线 + 中长线低估池（after_close_extra 2026-08-14,
    # 用户硬诉求: 盘后必须含 龙虎榜短线 + 中长线低估提醒）
    try:
        from quant_system.analysis_core import after_close_extra
        extra = after_close_extra.build_extra()
        report["lhb_broker_top"] = len((extra.get("lhb") or {}).get("broker_top", []))
        report["lhb_stock_top"] = len((extra.get("lhb") or {}).get("stock_top", []))
        report["value_picks"] = len(extra.get("value_picks", []))
    except Exception as e:
        report["extra_error"] = str(e)[:200]
    # 11. 资金合力已提前至 step 7 重建（F-06），此处仅做龙头扩散落盘
    # （2026-08-14 审计P0: leader_follower 产物停在 08-07 → 挂链刷新）
    try:
        from quant_system.analysis_core import leader_follower
        lf = leader_follower.run_today()
        report["leader_follower_rows"] = len(lf)
    except Exception as e:
        report["leader_follower_error"] = str(e)[:200]
    # 11.5 研报工作流：研报 NLP + 新闻/图片 OCR + 产业链/龙头融合
    # （2026-08-20 新增：研报从本地目录或有道云 MCP 注入，融合 chain_map/leader_follower）
    try:
        from quant_system.analysis_core import research_flow
        rf = research_flow.run_today(date=expected_day)
        report["research_reports"] = rf.get("n_reports", 0)
        report["research_concepts"] = len(rf.get("all_concepts", []))
        report["research_chain_hits"] = len(rf.get("chain_hits", []))
        report["research_leader_hits"] = len(rf.get("leader_hits", []))
    except Exception as e:
        report["research_flow_error"] = str(e)[:200]
    # 12. 个股全景透视落盘（stock_lens 2026-08-15: 自选+持仓短线/中线双视角,
    # 数据基座: 龙虎榜/涨停池/概念/资金流/融资盘/股东户数/基金持仓/估值）
    try:
        from quant_system.analysis_core import stock_lens
        items = stock_lens.analyze_watchlist()
        f = stock_lens.save_report(items)
        report["stock_lens_items"] = len(items)
        report["stock_lens_report"] = str(f)
    except Exception as e:
        report["stock_lens_error"] = str(e)[:200]
    # 13. 研报融合快照：使用数据门禁确定的最近完成交易日，失败不阻断主流程
    try:
        from scripts.research_fusion_snapshot import build_snapshot
        snapshot = build_snapshot(date=expected_day)
        report["research_fusion_snapshot"] = {
            "status": (snapshot.get("data_status") or {}).get("status"),
            "as_of": snapshot.get("as_of"),
            "output": snapshot.get("output_path"),
        }
    except Exception as e:
        report["research_fusion_snapshot_error"] = str(e)[:200]
    return report


def pre_market() -> dict:
    """盘前 08:30 链路: 健康检查 → 机制 → 宏观 → 共振 → 作战地图 → 订单 → 留痕。"""
    from datetime import datetime, timedelta, timezone
    from quant_system.analysis_core import (data_health_check, regime_classifier,
                                           macro_veto, resonance_scorer, battle_map,
                                           order_dispatcher, audit_trail)
    rep: dict = {}
    date = datetime.now().astimezone(timezone(timedelta(hours=8))).date().isoformat()
    try:
        rep["health"] = data_health_check.run(skip_baostock=True)["overall_ok"]
    except Exception as e:
        rep["health"] = str(e)[:100]
    try:
        rep["regime"] = regime_classifier.classify(date)["regime"]
    except Exception as e:
        rep["regime"] = str(e)[:100]
    try:
        veto_res = macro_veto.veto(date)
        rep["veto"] = veto_res["level"]
        _write_generated(f"macro_veto_{date}.json", veto_res)
    except Exception as e:
        rep["veto"] = str(e)[:100]
    try:
        res = resonance_scorer.score_day(date)
        rep["resonance_rows"] = len(res)
    except Exception as e:
        rep["resonance"] = str(e)[:100]
    try:
        bm = battle_map.build_map(date)
        rep["battle_map"] = f"{bm['confidence']} {bm['position_range']} {bm['recommended']}"
        _write_generated(f"battle_map_{date}.md", battle_map.render_md(bm))
        bm_out = dict(bm)
        bm_out["multi_agent"] = _multi_agent_summary(date)
        _write_generated(f"battle_map_{date}.json", bm_out)
    except Exception as e:
        rep["battle_map"] = str(e)[:100]
    try:
        od = order_dispatcher.dispatch(date)
        rep["orders"] = len(od.get("orders", []))
    except Exception as e:
        rep["orders"] = str(e)[:100]
    try:
        audit_trail.log_battle_map(date)
        rep["audit"] = "logged"
    except Exception as e:
        rep["audit"] = str(e)[:100]
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V11 每日数据流水线")
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--pre-market", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    # 2026-08-14 修复: os.nice 仅 Unix, Windows 无此属性 → 静默跳过(非致命)
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except Exception as e:
            logging.getLogger(__name__).error(f"[pipeline] nice 失败: {e}")

    if args.daily:
        import json
        print(json.dumps(daily_update(), ensure_ascii=False, indent=2, default=str))
    elif args.pre_market:
        import json
        print(json.dumps(pre_market(), ensure_ascii=False, indent=2, default=str))
    else:
        zt_pool_history.status()
