# -*- coding: utf-8 -*-
"""每日A股复盘融合链 · 前端 API handler（quant_web）

端点:
  /api/review/fusion?date=2026-08-21   → 融合复盘结构化数据（供前端看板）
  /api/review/md?date=2026-08-21       → 复盘 markdown（原样展示）

读 generated/review_{date}.json（由 daily_review_chain.py 生成）
"""
from __future__ import annotations

import json
import re
from datetime import date as date_type
from pathlib import Path

# quant_web 运行目录 = workspace/quant_web，generated 在 workspace/generated
ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "generated"


_DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


def _load_review(date: str | None) -> tuple[dict | None, str | None]:
    """加载 review_{date}.json；date 缺省取最新。"""
    gen_root = GEN.resolve()
    if date:
        if not _DATE_RE.fullmatch(date):
            return None, None
        try:
            date_type.fromisoformat(date)
        except ValueError:
            return None, None
        p = (GEN / f"review_{date}.json").resolve()
        if gen_root not in p.parents or p.name != f"review_{date}.json":
            return None, None
        if p.is_file() and not p.is_symlink():
            return json.loads(p.read_text(encoding="utf-8")), str(p)
        return None, None
    # 只选择日期命名的正式复盘，排除 review_q1/review_c2 等临时或历史样例。
    import re
    files = [p for p in GEN.glob("review_*.json") if re.fullmatch(r"review_20\d{2}-\d{2}-\d{2}\.json", p.name)]
    files.sort(key=lambda p: p.name)
    if not files:
        return None, None
    p = files[-1]
    return json.loads(p.read_text(encoding="utf-8")), str(p)



def _make_decision_from_blocks(blocks: dict) -> dict:
    """复用 decision_engine 生成决策卡（供前端直接展示）。"""
    try:
        import sys as _sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parent.parent.parent
        _sys.path.insert(0, str(_root / "scripts"))
        from fusion_decision import make_decision  # noqa: PLC0415
        # 把 blocks 还原为 SignalBlock dict（simplify：直接用原始 review blocks 值）
        sig = {}
        from daily_review_collectors import SignalBlock as _SB  # noqa: PLC0415
        for k, v in blocks.items():
            sig[k] = _SB(k, v.get("source", "?"), v.get("value", {}),
                         confidence=v.get("confidence", 0.5))
        return make_decision(sig)
    except Exception as _e:  # noqa: BLE001
        return {"error": str(_e)[:100]}


def _block_source_dates(blocks: dict) -> dict[str, str]:
    """Expose only dates explicitly stored by each review block; never invent one."""
    dates: dict[str, str] = {}
    for name, block in (blocks or {}).items():
        value = (block or {}).get("value") if isinstance(block, dict) else None
        candidates = []
        if isinstance(value, dict):
            candidates.extend(value.get(key) for key in ("as_of", "date", "data_date", "source_date"))
            for nested in value.values():
                if isinstance(nested, dict):
                    candidates.extend(nested.get(key) for key in ("as_of", "date", "data_date", "source_date"))
        candidates.extend((block or {}).get(key) for key in ("as_of", "date", "data_date", "source_date")) if isinstance(block, dict) else []
        for candidate in candidates:
            text = str(candidate or "").strip()
            if len(text) >= 10 and text[:4].isdigit():
                dates[name] = text[:10]
                break
    return dates


def _frontend_model(review: dict) -> dict:
    """转前端友好结构（轻量，字段打平）。"""
    blocks = review.get("blocks", {})
    bm = review.get("battle_map", {})

    def _blk(key):
        b = blocks.get(key) or {}
        if b.get("error"):
            return {"error": b["error"]}
        return b.get("value", {})

    mt = _blk("market_temperature")
    sd = _blk("strong_direction")
    strong_error = (blocks.get("strong_direction") or {}).get("error")
    ls = _blk("leader_sentiment")
    lh = _blk("lhb")
    sp = _blk("stock_picks")
    mv = _blk("model_verdict")

    try:
        try:
            from scripts.report_contract import contract_from_daily_review
        except ImportError:
            from report_contract import contract_from_daily_review
        research_contract = contract_from_daily_review(review)
    except Exception as exc:  # Legacy view remains available if optional adapter fails.
        research_contract = {"contract_error": str(exc)[:160]}
    return {
        "date": review.get("date"),
        "research_contract": research_contract,
        "generated_at": review.get("generated_at"),
        "health": review.get("health", {}),
        "temperature": {
            "emotion": (mt.get("components") or {}).get("emotion", {}),
            "fund": (mt.get("components") or {}).get("fund", {}),
            "macro": (mt.get("components") or {}).get("macro", {}),
            "health_ok": (mt.get("data_health") or {}).get("overall_ok"),
        },
        "directions": sd.get("directions", []),
        "mainlines": sd.get("mainlines", {"status": "unavailable" if strong_error else "watch",
                                             "items": sd.get("directions", [])}),
        "directions_error": strong_error,
        "theme_cycle": (sd.get("components") or {}).get("theme_cycle", {}),
        "leader": {
            "ladder": (ls.get("components") or {}).get("ladder", {}),
            "diffusion": (ls.get("components") or {}).get("leader", {}),
            "signals": ls.get("signals", []),
            "breakout": (ls.get("components") or {}).get("breakout", {}),
        },
        "lhb": {
            "broker_top": (lh.get("lhb") or {}).get("broker_top", []),
            "stock_top": (lh.get("lhb") or {}).get("stock_top", []),
        },
        "pools": sp.get("pools", {}),
        "verdict": mv.get("verdict", {}),
        "verdict_components": mv.get("components", {}),
        # 量化分析层（v3 深化）——优先使用复盘链已落盘的权威 decision，避免前端重算丢失 battle_map/intraday 门控。
        "decision": review.get("decision") or _make_decision_from_blocks(blocks),
        "overseas": _blk("overseas").get("quotes") or [],
        "data_base": _blk("data_base"),
        "freshness": _blk("freshness"),
        "technical": _blk("technical"),
        "prediction_loop": _blk("prediction_loop"),
        "source_dates": review.get("source_dates") or _block_source_dates(blocks),
        "quant": {
            "groups": (_blk("factor_signal").get("groups") or {}),
            "top_factors": _blk("factor_signal").get("top_factors") or [],
            "weak_factors": _blk("factor_signal").get("weak_factors") or [],
            "opportunities": _blk("opportunity").get("opportunities") or [],
            "ml": _blk("ml_verdict").get("predictions") or [],
            "backtest": _blk("backtest_check").get("checks") or [],
        },
        "battle_map": {
            "action_card": bm.get("action_card") or bm.get("recommended"),
            "position_range": bm.get("position_range"),
            "confidence": bm.get("confidence"),
            "bid_watch": bm.get("bid_watch", []),
            "risk_stocks": bm.get("risk_stocks", []),
            "degraded": bm.get("degraded", []),
            "error": bm.get("error"),
        },
        "blocks_meta": {k: ({"error": (b or {}).get("error")} if (b or {}).get("error")
                             else {"confidence": (b or {}).get("confidence")})
                        for k, b in blocks.items()},
    }


def handler_review_fusion(query: dict, send_json):
    """GET /api/review/fusion?date=YYYY-MM-DD"""
    date = (query.get("date") or [None])[0]
    review, path = _load_review(date)
    if review is None:
        # 缺少指定日复盘时回退到最近可用交易日，返回明确日期而不是让首页空白。
        candidates = sorted(GEN.glob("review_*.json"), reverse=True)
        if candidates:
            try:
                review = json.loads(candidates[0].read_text(encoding="utf-8"))
                path = str(candidates[0])
            except (OSError, ValueError):
                review = None
        if review is None:
            send_json({"ok": False, "data_status": "missing", "error": "暂无可用复盘产物，请先运行盘后复盘任务", "path": str(GEN)})
            return
        fallback_date = review.get("date")
    else:
        fallback_date = None
    model = _frontend_model(review)
    # “最新”页允许用当前轻量采集修复旧产物里的降级块；显式历史日期
    # 始终保持原始快照，避免用今天的数据改写历史结论。
    if not date:
        try:
            import sys as _live_sys
            _live_sys.path.insert(0, str(ROOT / "scripts"))
            if model.get("directions_error") or not model.get("directions"):
                from daily_review_collectors import collect_strong_direction
                live_direction = collect_strong_direction(None, model.get("date"))
                if not live_direction.error and live_direction.value.get("directions"):
                    model["directions"] = live_direction.value["directions"]
                    model["mainlines"] = live_direction.value.get("mainlines", {})
                    model["directions_error"] = None
                    model["directions_reference"] = "当前重算"
            if len(model.get("overseas") or []) < 6:
                from overseas_collector import collect_overseas
                live_overseas = collect_overseas()
                if not live_overseas.error and len(live_overseas.value.get("quotes") or []) >= 6:
                    model["overseas"] = live_overseas.value["quotes"]
                    model["overseas_coverage"] = live_overseas.value.get("coverage", {})
                    model["overseas_reference"] = "当前参考"
            from freshness_gate import gate_scores
            model["freshness"] = gate_scores()
        except Exception as live_error:  # Current enrichment must not hide the saved review.
            model["live_enrichment_error"] = str(live_error)[:160]
    # 缠论+深度复盘+IMA 统一缓存(1h, 避免每次API重算20s+)
    try:
        import sys as _s2, time as _t2
        from pathlib import Path as _P2
        _ROOT2 = _P2(__file__).resolve().parent.parent.parent
        _CC = _ROOT2 / "generated" / "chan_depth_cache.json"
        _cdc = None
        if _CC.exists() and _t2.time() - _CC.stat().st_mtime < 3600:
            cached_chan = json.loads(_CC.read_text(encoding="utf-8"))
            if cached_chan.get("schema") == "chan_review/v2":
                _cdc = cached_chan
        if _cdc is None:
            _s2.path.insert(0, str(_ROOT2 / "scripts"))
            from chan_skill import (
                _load_index_df, buy_point_scan, chan_bigscan, script_judge_by_levels,
            )
            _cb = chan_bigscan("沪深300", days=180)
            _chan_df = _load_index_df("沪深300")
            _scenario = script_judge_by_levels(_chan_df) if _chan_df is not None else {}
            _buy_points = buy_point_scan(_chan_df) if _chan_df is not None else []
            from depth_review import depth_block
            _dp = depth_block()
            _levels = _scenario.get("levels", {})
            _cdc = {
                "schema": "chan_review/v2",
                "chan": {"note": _cb.get("note", ""), "cur_pos": _cb.get("cur_pos", ""),
                         "cur_zs": _cb.get("cur_zs", ""), "bis": _cb.get("bis", 0),
                         "fractals": _cb.get("fractals", 0),
                         "scenario": _scenario.get("script"), "action": _scenario.get("act"),
                         "operation": _scenario.get("op"), "last": _scenario.get("last"),
                         "levels": _levels, "buy_points": _buy_points,
                         "confirm": f"站稳MA144({_levels.get('MA144', '-')})" if _levels else "等待结构确认",
                         "invalidation": f"跌破MA300({_levels.get('MA300', '-')})" if _levels else "关键位缺失"},
                "depth": {"emotion": _dp.get("emotion", {}), "zt_board": _dp.get("zt_board", [])[:6],
                          "chain_stage": _dp.get("chain_stage", [])[:5]},
            }
            _CC.write_text(json.dumps(_cdc, ensure_ascii=False), encoding="utf-8")
        model.update(_cdc)
        # IMA只参与研报输入，不作为研报/决策接口的展示字段。来源健康信息由
        # research_fusion_snapshot 的 data_status 统一呈现，避免前端触发慢查询。
    except Exception as _e2:
        model["chan_err"] = str(_e2)[:80]
    # 财报深度(选股池/估值/公告/趋势)——缓存1h
    try:
        import time as _t4
        from pathlib import Path as _P3
        _R3 = _P3(__file__).resolve().parent.parent.parent
        _FC = _R3 / "generated" / "fund_cache.json"
        if _FC.exists() and _t4.time() - _FC.stat().st_mtime < 3600:
            _fd = json.loads(_FC.read_text(encoding="utf-8"))
        else:
            import sys as _s3
            _s3.path.insert(0, str(_R3 / "scripts"))
            from fundamental_pulse import stock_picker, valuation_pulse, announcement_calendar, trend_data
            _fd = {"picker": stock_picker(limit=6), "valuation": valuation_pulse(limit=6),
                   "ann": announcement_calendar(6), "trend": trend_data("000001", 5)}
            _FC.write_text(json.dumps(_fd, ensure_ascii=False, default=str), encoding="utf-8")
        model["fund"] = _fd
    except Exception as _ef:
        model["fund_err"] = str(_ef)[:80]
    model["_source"] = path
    if fallback_date:
        model["requested_date"] = date
        model["fallback_date"] = fallback_date
        model["date_mismatch"] = bool(date and date != fallback_date)
        model["fallback_notice"] = f"所选日期无复盘，已显示最近可用报告：{fallback_date}"
    send_json({"ok": True, "data": model})


def handler_review_md(query: dict, send_json):
    """GET /api/review/md?date=YYYY-MM-DD"""
    date = (query.get("date") or [None])[0]
    if date:
        p = GEN / f"review_{date}.md"
    else:
        files = sorted(GEN.glob("review_*.md"))
        p = files[-1] if files else None
    if p is None or not p.exists():
        send_json({"ok": False, "error": "复盘 md 不存在"})
        return
    send_json({"ok": True, "data": {"date": p.stem.replace("review_", ""),
                                    "md": p.read_text(encoding="utf-8")}})

def handler_social(query, send_json):
    """GET /api/social — 统一市场舆情(股吧/百度/B站/微博/新闻/小红书/雪球 + 综合情绪指标)."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from social_pulse import sentiment_pulse, xueqiu_pulse
        p = sentiment_pulse()
        xq = xueqiu_pulse()
        # 综合情绪指标：从盘后已落盘的 social_sentiment.parquet 读取(不重复爬网)，
        # 作为「市场第二情绪指标」替代已停更的北向日频资金。
        composite = {}
        try:
            import pandas as pd
            sys.path.insert(0, str(ROOT))
            from quant_system.analysis_core.social_sentiment import market_sentiment_composite
            sp = ROOT / "data_warehouse" / "market" / "social_sentiment.parquet"
            if sp.exists():
                df = pd.read_parquet(sp)
                latest = df[df["date"] == df["date"].max()].to_dict("records")
                composite = market_sentiment_composite(live=latest)
        except Exception as e:  # noqa: BLE001
            composite = {"error": str(e)[:80]}
        source_buckets = {
            "guba": p.get("guba") or {}, "baidu": p.get("baidu") or {},
            "bilibili": p.get("bilibili") or {}, "xueqiu": xq or {},
        }
        source_dates = [bucket.get("as_of") for bucket in source_buckets.values() if isinstance(bucket, dict)]
        source_dates.extend([composite.get("as_of"), composite.get("date")])
        sentiment_as_of = max((str(value)[:10] for value in source_dates if value), default=None)
        guba_items = source_buckets["guba"].get("items", [])[:8]
        baidu_items = source_buckets["baidu"].get("items", [])[:6]
        bilibili_items = [item for item in source_buckets["bilibili"].get("items", [])
                           if item.get("is_finance") is True][:8]
        xueqiu_posts = source_buckets["xueqiu"].get("posts", [])[:6]
        source_items = {"guba": guba_items, "baidu": baidu_items,
                        "bilibili": bilibili_items, "xueqiu": xueqiu_posts}
        sources = {}
        for name, bucket in source_buckets.items():
            status = bucket.get("status") or ("ok" if source_items[name] else "empty")
            if name == "bilibili" and bucket.get("items") and not bilibili_items:
                status = "empty"
            sources[name] = {"status": status, "as_of": bucket.get("as_of"),
                             "count": len(source_items[name]), "error": bucket.get("error")}
        available_sources = sum(meta["status"] == "ok" and meta["count"] > 0 for meta in sources.values())
        stale_sources = sum(meta["status"] == "stale" for meta in sources.values())
        overall_status = "ok" if available_sources >= 2 and not stale_sources else (
            "partial" if available_sources or stale_sources else "unavailable")
        send_json({"ok": True, "data": {
            "status": overall_status,
            "available_sources": available_sources,
            "stale_sources": stale_sources,
            "sources": sources,
            "as_of": sentiment_as_of, "note": p.get("note"), "total": p.get("total"),
            "guba": guba_items,
            "baidu": baidu_items,
            "bilibili": bilibili_items,
            "xueqiu": {"posts": xueqiu_posts,
                       "stocks": (xq.get("stocks") or [])[:6],
                       "note": xq.get("note", "")},
            "market_sentiment": composite,
        }})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:100]})


def handler_market_pano(query, send_json):
    """GET /api/market_pano — 市场基座全景(商品/利率/宏观/涨停/状态机)."""
    import sys, json, time
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    _C = ROOT / "generated" / "market_pano_cache.json"
    if _C.exists() and time.time() - _C.stat().st_mtime < 3600:
        cached = json.loads(_C.read_text(encoding="utf-8"))
        if cached.get("schema") == "market_pano/v4":
            # 缓存可能由旧进程在状态规则升级前写入；以 errors 重新归一化，
            # 防止“存在缺失源但 status=ok”继续污染前端。
            if cached.get("errors") and cached.get("status") == "ok":
                cached["status"] = "partial" if any(
                    cached.get(key) for key in ("commodity", "rates", "macro", "zt_stats", "regime", "margin", "fund_flow")
                ) else "unavailable"
            send_json({"ok": True, **cached})
            return
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from market_pano import market_pano
        d = market_pano()
        _C.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding="utf-8")
        send_json({"ok": True, **d})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:80]})
