#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日A股复盘融合链 · L2 orchestrator（2026-08-21 架构融合）

将全部板块融合成一份完整复盘：
1. L0 数据基座门控（健康/新鲜度/真源）
2. L1 六要素采集（市场温度/强势方向/龙头情绪/龙虎榜/多格局选股/模型后验）
3. battle_map 行动卡决策骨架（专家委员会/攻击方向/风险清单）
4. 四格局选股合并（短线/中长线/风格/机会）
5. 输出 generated/review_{date}.md + review_{date}.json
6. 推送（飞书，可开关）

用法:
  python3 scripts/daily_review_chain.py [--date 2026-08-21] [--push] [--json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.product_contract import PRODUCT_VERSION, release_metadata, validate_production_contract
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "quant_system"))

# Windows GBK 控制台无法输出 emoji，重定向 stdout/stderr 防末尾 print 崩溃（cron 判定失败）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("daily_review_chain")


def _sanitize_contract_dates(contract: dict, decision_date: str) -> dict:
    """隔离晚于分析日的来源/证据，避免未来数据污染历史复盘。"""
    from datetime import date as _date

    try:
        cutoff = _date.fromisoformat(str(decision_date)[:10])
    except ValueError:
        return contract
    quarantined = []
    for bucket in ("sources", "evidence"):
        for item in contract.get(bucket, []):
            if not isinstance(item, dict):
                continue
            raw = str(item.get("observed_at") or "")[:10]
            try:
                observed = _date.fromisoformat(raw)
            except ValueError:
                continue
            if observed > cutoff:
                item["original_observed_at"] = raw
                item["observed_at"] = str(decision_date)[:10]
                item["status"] = "stale" if bucket == "sources" else item.get("quality", "historical")
                item["quality"] = "historical"
                item["note"] = (item.get("note") or "") + f"；原观察日{raw}晚于分析日，已按{decision_date}隔离"
                quarantined.append({"field": item.get("id", bucket), "reason": "future_observed_at",
                                    "impact": "不进入当日决策结论", "fallback": "等待数据日期不晚于分析日"})
    if quarantined:
        contract.setdefault("data_gaps", []).extend(quarantined)
        contract.setdefault("metadata", {})["future_observations_quarantined"] = len(quarantined)
    return contract


def _atomic_write(path: Path, data: str) -> None:
    """原子写: 临时文件+rename，防进程被杀/并发写导致 json 损坏(2026-08-21 修复)。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)  # 同目录 rename 原子
CST = timezone(timedelta(hours=8))


def _write_delivery_receipt(*, data_as_of: str, run_id: str, requested: bool,
                            ok: bool | None, error: str | None = None) -> Path:
    """持久化投递结果；布尔发送 API 没有 message id 时明确记录其不可用。"""
    out_dir = ROOT / "generated" / "delivery_receipts"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "delivery_id": f"{run_id}:feishu",
        "run_id": run_id,
        "run_date": datetime.now(CST).date().isoformat(),
        "data_as_of": data_as_of,
        "channel": "feishu",
        "requested": requested,
        "attempted_at": datetime.now(CST).isoformat(timespec="seconds") if requested else None,
        "ok": ok if requested else None,
        "message_ids": [],
        "message_id_available": False,
        "result_semantics": "send_markdown_bool",
        "error": error,
    }
    path = out_dir / f"review_{data_as_of}_{run_id}.json"
    _atomic_write(path, json.dumps(receipt, ensure_ascii=False, indent=2))
    # 统一账本登记：投递回执追加到 delivery_receipts/index.json（幂等覆盖同 delivery_id）。
    try:
        from execution_ledger import record_delivery  # noqa: PLC0415
        record_delivery(delivery_id=receipt["delivery_id"], record=receipt)
    except Exception as exc:  # noqa: BLE001 - 账本失败不阻断复盘主链
        logger.warning("投递回执统一账本登记失败: %s", str(exc)[:120])
    return path

from daily_fusion_base import get_gateway  # noqa: E402
from daily_review_collectors import collect_all, SignalBlock  # noqa: E402
from daily_quant_collectors import collect_quant_all  # noqa: E402
from fusion_decision import make_decision, render_decision_md  # noqa: E402
from data_base_bridge import data_base_pulse  # noqa: E402
from report_contract import contract_from_daily_review, validate_report_contract, validate_and_write_contract  # noqa: E402
from freshness_gate import gate_scores, freshness_md  # noqa: E402
try:
    from overseas_collector import collect_overseas, render_overseas_md  # noqa: E402
    _HAS_OVERSEAS = True
except Exception as _eo:  # noqa: BLE001
    _HAS_OVERSEAS = False
    logger.warning("overseas_collector 不可用: %s", str(_eo)[:60])
try:
    from technical_collector import collect_technical, render_technical_md  # noqa: E402
    _HAS_TECH = True
except Exception as _et:  # noqa: BLE001
    _HAS_TECH = False
    logger.warning("technical_collector 不可用: %s", str(_et)[:60])


def _today() -> str:
    """取最近交易日（工作日回退）。"""
    from quant_system.analysis_core import zt_pool_history  # noqa: PLC0415
    try:
        td = zt_pool_history._trade_dates_upto_today(5)
        if td:
            return _fmt_dash(str(td[-1]))
    except Exception:  # noqa: BLE001
        pass
    return datetime.now(CST).strftime("%Y-%m-%d")


def _fmt_dash(date: str) -> str:
    """YYYYMMDD → YYYY-MM-DD；其它格式原样返回。保证 review_*.json 字典序即时间序。"""
    d = str(date or "").strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def _battle_map_block(date: str) -> dict:
    """battle_map 决策骨架（含专家委员会/行动卡/攻击方向/风险）。
    优先读 pipeline 已生成产物(battle_map_{date}.json)，重算仅 fallback(重,5分钟)。"""
    try:
        _bm_path = ROOT / "generated" / f"battle_map_{date}.json"
        if _bm_path.exists():
            data = json.loads(_bm_path.read_text(encoding="utf-8"))
            actual = _fmt_dash(str(data.get("date") or data.get("analyze_date") or ""))
            if actual == date:
                return data
            logger.warning("battle_map 日期不匹配: requested=%s actual=%s", date, actual)
        from quant_system.analysis_core import battle_map  # noqa: PLC0415
        bm = battle_map.build_map(date)
        return {
            "date": bm.get("date"), "analyze_date": bm.get("analyze_date"),
            "action_card": bm.get("action_card") or bm.get("recommended"),
            "bid_watch": bm.get("bid_watch", []),
            "attack_groups": bm.get("attack_groups", {}),
            "risk_stocks": bm.get("risk_watch", [])[:5],
            "linkage": bm.get("linkage", {})[:3] if isinstance(bm.get("linkage"), list) else [],
            "expert_committee": bm.get("consensus") or bm.get("expert", {}),
            "confidence": bm.get("confidence"),
            "position_range": bm.get("position_range"),
            "degraded": bm.get("degraded", [])[:5],
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:150]}


def _merge_stock_pools(blocks: dict[str, SignalBlock]) -> dict:
    """四格局选股合并：短线龙头 + 中长线低估 + 风格轮动 + 机会。"""
    pools = {}
    sp = blocks.get("stock_picks")
    if sp and not sp.error:
        pools = sp.value.get("pools", {})
    # 从其它块补（如 lhb 的 top 票）
    return {k: v for k, v in pools.items() if v}


def _record_predictions(date: str, blocks: dict[str, SignalBlock], bm: dict) -> list[str]:
    """信号后验闭环起点：融合链共识/温度/方向 入库为次日预测（次日由 verify 闭环）。

    预测方向映射: 多/乐观/主线 → up；空/悲观/退潮 → down；震荡/中性 → flat。
    之后每日 verify_predictions 对照实际涨跌回填 → 滚动命中率 → 自校准权重。
    """
    pred_ids: list[str] = []
    try:
        from quant_system.analysis_core import predictions  # noqa: PLC0415

        def _dir_to_pred(direction: str) -> str:
            if direction in ("多", "乐观", "主线延续", "进攻", "持有/加仓前排"):
                return "up"
            if direction in ("空", "悲观", "退潮预警", "防守", "清仓"):
                return "down"
            return "flat"

        # 1. 专家委员会共识
        mv = blocks.get("model_verdict")
        if mv and not mv.error:
            v = mv.value.get("verdict", {})
            if v.get("consensus"):
                pid = predictions.add_prediction(
                    target_type="market_trend", target=date, direction=_dir_to_pred(v["consensus"]),
                    probability=float(v.get("confidence", 0.5) or 0.5),
                    source_module="multi_agent", note=f"融合链-专家委员会 {date}")
                pred_ids.append(pid)
        # 2. 情绪周期方向（修复/发酵/复苏 → up；退潮/冰点 → down）
        mt = blocks.get("market_temperature")
        if mt and not mt.error:
            st = (mt.value.get("components", {}).get("emotion") or {}).get("stage_cn")
            if st:
                pred = "up" if st in ("修复", "发酵", "复苏", "高潮") else ("down" if st in ("退潮", "冰点", "恐慌") else "flat")
                pid = predictions.add_prediction(
                    target_type="market_trend", target=date, direction=pred,
                    probability=float((mt.value.get("components", {}).get("emotion") or {}).get("confidence", 0.5) or 0.5),
                    source_module="emotion_cycle", note=f"融合链-情绪{st} {date}")
                pred_ids.append(pid)
        # 3. 决策卡方向
        if bm and not bm.get("error"):
            rc = bm.get("recommended") or bm.get("action_card")
            if rc:
                pid = predictions.add_prediction(
                    target_type="market_trend", target=date, direction=_dir_to_pred(str(rc)),
                    probability=float(bm.get("confidence", 0.5) or 0.5),
                    source_module="battle_map", note=f"融合链-决策卡{rc} {date}")
                pred_ids.append(pid)
    except Exception as e:  # noqa: BLE001
        logger.warning("预测入库失败: %s", str(e)[:100])
    return pred_ids


def _render_md(date: str, blocks: dict[str, SignalBlock], bm: dict, pools: dict, health: dict,
               dec: dict | None = None) -> str:
    """渲染融合复盘 Markdown。"""
    L: list[str] = []
    release_id = str((blocks.get("data_release").value if hasattr(blocks.get("data_release"), "value") else (blocks.get("data_release") or {})).get("release_id") or "未绑定")
    L.append(f"# 📊 每日A股复盘 · {date} · {PRODUCT_VERSION}")
    L.append(f"> 生成: {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} | release_id: `{release_id}` | 数据健康: {'✅' if health.get('overall_ok') else '⚠️ ' + str(len(health.get('degraded_datasets', []))) + '项降级'}")
    L.append("")
    # 决策卡置顶（结论先行）—— dec 由调用方 run() 传入(决策/渲染/存json三处一致)
    if dec is None:
        try:
            dec = make_decision(blocks)
        except Exception as _e:  # noqa: BLE001
            dec = {"error": str(_e)[:120], "date": date}
            logger.warning("决策卡生成失败(已兜底): %s", str(_e)[:100])
    try:
        L.append(render_decision_md(dec))
        L.append("")
    except Exception as _e2:  # noqa: BLE001
        logger.warning("决策卡渲染失败: %s", str(_e2)[:80])

    # 0 数据新鲜度与风险提示（置顶，决策前先看数据可信度）
    _deg = health.get("degraded_datasets", [])
    if _deg:
        L.append("## ⚠️ 数据新鲜度与风险提示")
        L.append(f"- 数据降级 {len(_deg)} 项: {', '.join(_deg[:8])}")
        _veto = blocks.get("macro_veto")
        if _veto and not _veto.error and isinstance(_veto.value, dict):
            _pc = _veto.value.get("position_coef")
            _lvl = _veto.value.get("level", "?")
            if _pc is not None:
                L.append(f"- 宏观否决层: {_lvl}，仓位系数 {_pc}（<1 表示宏观风险压制）")
        L.append("")

    # 1 市场温度
    mt = blocks.get("market_temperature")
    L.append("## 🌡️ 市场温度")
    if mt and not mt.error:
        c = mt.value.get("components", {})
        emo = c.get("emotion", {})
        if emo.get("stage_cn"):
            L.append(f"- 情绪周期: **{emo['stage_cn']}** (置信{emo.get('confidence')}) | 涨停{emo.get('zt_cnt')} 最高{emo.get('max_board')}板")
        fu = c.get("fund", {})
        if "force_index" in fu:
            force = fu.get("force_index")
            force_text = f"{float(force):.0f}" if force is not None else "缺失"
            def _yi(value):
                try:
                    return f"{float(value or 0) / 1e8:.1f}"
                except (TypeError, ValueError):
                    return "-"
            L.append(f"- 资金合力: **{force_text}** (游资{_yi(fu.get('youzi_net'))}亿 机构{_yi(fu.get('jg_net'))}亿 北向{_yi(fu.get('north_net'))}亿)")
        ma = c.get("macro", {})
        if ma.get("regime"):
            L.append(f"- 宏观: {ma['regime']}")
    else:
        L.append(f"- ⚠️ 温度采集降级: {mt.error if mt else 'N/A'}")
    L.append("")

    # 2 强势方向
    sd = blocks.get("strong_direction")
    L.append("## 🚀 强势方向")
    if sd and not sd.error:
        dirs = sd.value.get("directions", [])[:8]
        if dirs:
            for d in dirs:
                L.append(f"- **{d['name']}** ({d['level']}, score {d.get('score')})")
        else:
            L.append("- (无主线识别)")
        tc = sd.value.get("components", {}).get("theme_cycle", {})
        if tc.get("themes"):
            L.append(f"- 题材生命周期: {tc['themes']}")
    else:
        L.append(f"- ⚠️ 方向采集降级: {sd.error if sd else 'N/A'}")
    L.append("")

    # 3 龙头情绪
    ls = blocks.get("leader_sentiment")
    L.append("## 🔥 龙头情绪")
    if ls and not ls.error:
        c = ls.value.get("components", {})
        ld = c.get("ladder", {})
        if ld.get("zt_cnt") is not None:
            L.append(f"- 连板天梯: 涨停{ld['zt_cnt']} 炸板{ld.get('zb_cnt')} 跌停{ld.get('dt_cnt')} 最高{ld['max_board']}板")
        lf = c.get("leader", {})
        if lf.get("overheat") is not None:
            L.append(f"- 龙头扩散: 活跃{lf.get('active_concepts')}概念 过热{lf.get('overheat')}")
        sigs = ls.value.get("signals", [])[:5]
        if sigs:
            L.append("- 龙头信号:")
            for s in sigs:
                L.append(f"  - {s.get('concept') or s.get('board_name')}: {s.get('signal')} ({s.get('level','')})")
        bk = c.get("breakout", {})
        if bk.get("new_high_60") is not None:
            L.append(f"- 新高突破: 60日{bk['new_high_60']} 120日{bk.get('new_high_120')} 250日{bk.get('new_high_250')}")
    else:
        L.append(f"- ⚠️ 龙头情绪降级: {ls.error if ls else 'N/A'}")
    L.append("")

    # 4 龙虎榜
    lh = blocks.get("lhb")
    L.append("## 💰 龙虎榜资金")
    if lh and not lh.error:
        d = lh.value.get("lhb", {})
        bt = d.get("broker_top", [])[:5]
        st = d.get("stock_top", [])[:5]
        if bt:
            L.append("- 游资营业部 Top:")
            for b in bt:
                L.append(f"  - {b if isinstance(b, str) else json.dumps(b, ensure_ascii=False)[:80]}")
        if st:
            L.append("- 个股净买 Top:")
            for s in st:
                L.append(f"  - {s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)[:80]}")
    else:
        L.append(f"- ⚠️ 龙虎榜降级: {lh.error if lh else 'N/A'}")
    L.append("")

    # 5 多格局选股
    L.append("## 🎯 多格局选股（多套池：不同格局）")
    if pools:
        if pools.get("short_term"):
            L.append("### 短线攻击方向池（概念）")
            for p in pools["short_term"][:5]:
                L.append(f"- {p.get('concept') or p.get('name')}: {p.get('signal') or p.get('level')}")
        if pools.get("lhb_stocks"):
            L.append("### 龙虎榜强势池（个股）")
            for p in pools["lhb_stocks"][:5]:
                L.append(f"- {p.get('name')}({p.get('code')}) 净买{p.get('net')}亿 {p.get('reason') or ''}")
        if pools.get("rs_stocks"):
            L.append("### RS 相对强弱池（个股）")
            for p in pools["rs_stocks"][:5]:
                L.append(f"- {p.get('name')}({p.get('code')}) RS 榜#{p.get('rank')}")
        if pools.get("mid_long"):
            L.append("### 中长线低估池（个股）")
            for p in pools["mid_long"][:5]:
                if isinstance(p, str):
                    L.append(f"- {p}")
                    continue
                # 人类可读：名称(代码) PE/PB ROE/增长 | 距52周低 | 打分
                name = p.get("name") or p.get("symbol") or p.get("code") or "?"
                code = p.get("code") or p.get("symbol") or ""
                pe = p.get("pe") if p.get("pe") is not None else p.get("pe_ttm")
                pb = p.get("pb") if p.get("pb") is not None else p.get("pb_mrq")
                roe = p.get("roe")
                growth = p.get("growth") or p.get("profit_growth")
                dist = p.get("dist52w") or p.get("dist_low")
                score = p.get("score")
                parts = []
                if pe is not None:
                    parts.append(f"PE {float(pe):.1f}")
                if pb is not None:
                    parts.append(f"PB {float(pb):.2f}")
                if roe is not None:
                    parts.append(f"ROE {float(roe):.0f}%")
                if growth is not None:
                    parts.append(f"增 {float(growth):.0f}%")
                if dist is not None:
                    parts.append(f"距52周低+{float(dist):.0f}%")
                if score is not None:
                    parts.append(f"打分 {float(score):.1f}")
                L.append(f"- 💎 {name}({code}) " + " ".join(parts))
    else:
        L.append("- (选股池待数据就绪)")
    L.append("")

    # 6 决策骨架（battle_map）
    L.append("## ⚡ 作战决策（行动卡）")
    if bm and not bm.get("error"):
        if bm.get("action_card"):
            L.append(f"- 行动卡: {bm['action_card']} | 仓位: {bm.get('position_range')} | 置信: {bm.get('confidence')}")
        if bm.get("bid_watch"):
            L.append("- 竞价锚点:")
            for b in bm["bid_watch"][:3]:
                L.append(f"  - {b.get('item')} → {b.get('signal')}")
        if bm.get("risk_stocks"):
            L.append(f"- 风险清单: {len(bm['risk_stocks'])} 只")
        if bm.get("degraded"):
            L.append(f"- ⚠️ 降级: {', '.join(bm['degraded'][:3])}")
    else:
        L.append(f"- ⚠️ 决策骨架降级: {bm.get('error') if bm else 'N/A'}")
    L.append("")

    # 7 模型意见
    mv = blocks.get("model_verdict")
    L.append("## 🤖 模型集体意见")
    if mv and not mv.error:
        v = mv.value.get("verdict", {})
        if v.get("consensus"):
            L.append(f"- 专家委员会: **{v['consensus']}** (置信{v.get('confidence')}) {v.get('votes')}票")
        else:
            L.append("- 专家委员会: 降级")
    else:
        L.append(f"- ⚠️ 模型降级: {mv.error if mv else 'N/A'}")
    L.append("")

    # 7b 量化分析层（因子/机会/ML/回测）
    _f = blocks.get("factor_signal")
    if _f and not _f.error:
        L.append("## 📐 因子强度（量化）")
        g = _f.value.get("groups", {})
        if g:
            for grp, v in sorted(g.items(), key=lambda x: -x[1]["score"]):
                L.append(f"- {grp}: 高位占比 **{v['score']*100:.0f}%** ({v['n']}因子)")
        if _f.value.get("top_factors"):
            L.append(f"- 强势因子: {', '.join(_f.value['top_factors'][:4])}")
        if _f.value.get("weak_factors"):
            L.append(f"- 走弱因子: {', '.join(_f.value['weak_factors'][:4])}")
        L.append("")
    _b = blocks.get("backtest_check")
    if _b and not _b.error and _b.value.get("checks"):
        L.append("## 📈 标的回测速览（因子面板）")
        for c in _b.value["checks"][:4]:
            L.append(f"- {c['symbol']}: 动量20d {c.get('mom20')} 波动 {c.get('vol20')}")
        L.append("")
    _o = blocks.get("opportunity")
    if _o and not _o.error and _o.value.get("opportunities"):
        L.append("## 🎯 机会池（实时扫描）")
        for op in _o.value["opportunities"][:5]:
            L.append(f"- {op if isinstance(op, str) else json.dumps(op, ensure_ascii=False)[:80]}")
        L.append("")
    _m = blocks.get("ml_verdict")
    if _m and not _m.error and _m.value.get("predictions"):
        L.append("## 🤖 ML 模型预测")
        for pr in _m.value["predictions"][:4]:
            L.append(f"- {pr['symbol']}: {str(pr.get('pred'))[:60]}")
        L.append("")

    # 原始数据附录（精简：只保留关键，供复核）
    # 数据基座脉动段
    _dbs = blocks.get("data_base")
    if _dbs and not _dbs.error:
        _pv = _dbs.value
        L.append("## 🗄️ 数据基座融合（宏观/热榜/机构/行业）")
        _mk = _pv.get("macro", {})
        if _mk.get("items"):
            _ms = " | ".join(str(i.get("name")) + "=" + str(i.get("value")) for i in _mk["items"][:5])
            L.append("- 宏观: " + _ms)
        _hk = _pv.get("hot", {})
        if _hk.get("top"):
            _top = ",".join(str(t.get("code")) for t in _hk["top"][:4])
            L.append("- 热榜: 涨" + str(_hk.get("up_n")) + "跌" + str(_hk.get("down_n")) + " Top[" + _top + "]")
        _ik = _pv.get("institutional", {})
        if _ik.get("new_entries"):
            _ne = ", ".join(str(e.get("name")) for e in _ik["new_entries"][:4])
            L.append("- 机构新进: " + _ne)
        _dk = _pv.get("industry", {})
        if _dk.get("strong"):
            _st = ", ".join(str(s.get("name")) + "(" + ("%+.1f%%" % float(s.get("chg", 0))) + ")" for s in _dk["strong"][:4])
            L.append("- 强行业: " + _st)
        _nm = _pv.get("north_margin", {})
        if _nm.get("note") and "缺失" not in _nm.get("note", ""):
            L.append("- 资金基座: " + _nm["note"])
        _st = _pv.get("style", {})
        if _st.get("note") and "缺失" not in _st.get("note", ""):
            L.append("- 风格剪刀差: " + _st["note"])
        _ev = _pv.get("event", {})
        if _ev.get("note") and "缺失" not in _ev.get("note", ""):
            L.append("- 事件域: " + _ev["note"])
        _cp = _pv.get("concept", {})
        if _cp.get("note") and "缺失" not in _cp.get("note", ""):
            L.append("- 概念周期: " + _cp["note"])
        L.append("")

    # 新鲜度门控段
    _fg = blocks.get("freshness")
    if _fg and not _fg.error:
        L.append(freshness_md())
        L.append("")

    # 预测验证闭环段（2026-08-22 融合升级）
    _pl = blocks.get("prediction_loop")
    if _pl and not _pl.error:
        try:
            from prediction_verify import render_verify_md  # noqa: PLC0415
            L.append(render_verify_md(_pl.value))
            L.append("")
        except Exception as _e:  # noqa: BLE001
            logger.warning("预测闭环渲染失败: %s", str(_e)[:80])
    # 引擎融合共识段（2026-08-22 融合升级）
    _ef = blocks.get("engine_fusion")
    if _ef and not _ef.error:
        _ev = _ef.value
        L.append("## 🧠 引擎融合共识（引擎→决策）")
        L.append(f"- 共识: **{_ev.get('consensus', '?')}** | 融合分 {_ev.get('score')} | 可用引擎 {_ev.get('ok_n')} 个（信用 {_ev.get('trust')}）")
        for _dk, _dv in (_ev.get("dimensions") or {}).items():
            _mk = {"risk": "风险温度", "meta": "元审查", "card": "行动卡",
                   "expert": "专家共识", "trust": "校准信任"}.get(_dk, _dk)
            L.append(f"- {_mk}: {_dv.get('score')}分 ({_dv.get('note', '')})")
        L.append("")

    # 海外市场段（决策卡后）
    _ovr = blocks.get("overseas")
    if _ovr and not _ovr.error:
        L.append(render_overseas_md(_ovr))
        L.append("")
    # 技术分析段
    _tec = blocks.get("technical")
    if _tec and not _tec.error:
        L.append(render_technical_md(_tec))
        L.append("")

    L.append("## 📋 原始信号附录（供复核）")
    L.append("")
    # 数据健康
    deg = health.get("degraded_datasets", [])
    if deg:
        L.append(f"⚠️ 数据降级: {', '.join(deg[:6])}")
        L.append("")

    return "\n".join(L)


def run(date: str | None = None, push: bool = False, save_json: bool = True) -> dict:
    run_id = f"{datetime.now(CST).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    gw = get_gateway()
    date = _fmt_dash(date or _today())

    # L0 门控
    health = gw.health_scores(force=True)
    # L1 采集（六要素）
    blocks = collect_all(gw, date)
    try:
        from quant_system.data_release import load_release
        blocks["data_release"] = SignalBlock("data_release", "data_release_gate", load_release(ROOT, expected_day=date), confidence=1.0)
    except Exception as exc:
        raise RuntimeError(f"同日数据发布版本不可用: {exc}") from exc
    # L1b-L1d 量化/海外/技术 并行采集（daemon 线程, 防 shutdown 等死, 2026-08-21）
    import threading as _th2
    _tasks = {"_q": (lambda: collect_quant_all(gw))}
    if _HAS_OVERSEAS:
        _tasks["overseas"] = (lambda: collect_overseas(gw))
    if _HAS_TECH:
        _tasks["technical"] = (lambda: collect_technical(gw, as_of=date))
    _b2: dict = {}
    def _run2(_n, _fn):
        try:
            _b2[_n] = _fn()
        except Exception as _e:  # noqa: BLE001
            _b2[_n] = _e
    _ths = [_th2.Thread(target=_run2, args=(n, fn), daemon=True)
            for n, fn in _tasks.items()]
    for _t in _ths:
        _t.start()
    for _t in _ths:
        _t.join(45)
    for _n in _tasks:
        _r = _b2.get(_n)
        if isinstance(_r, Exception):
            logger.warning("并行采集 %s 异常: %s", _n, str(_r)[:80])
        elif _n == "_q" and isinstance(_r, dict):
            blocks.update(_r)
        elif _n in ("overseas", "technical") and _r is not None:
            blocks[_n] = _r
    # L1e macro_veto（宏观否决层：PE百分位/巴菲特指标 → 仓位系数）——生成产物消除告警
    try:
        sys.path.insert(0, str(ROOT / "quant_system"))
        from quant_system.analysis_core import macro_veto as _mv  # noqa: PLC0415
        _veto = _mv.veto(date)
        if _veto and _veto.get("date"):
            _atomic_write(ROOT / "generated" / f"macro_veto_{date}.json",
                           json.dumps(_veto, ensure_ascii=False, indent=1))
            blocks["macro_veto"] = SignalBlock("macro_veto", "macro_veto", _veto, confidence=0.7)
    except Exception as _e5:  # noqa: BLE001
        # 审计修复(2026-08-24): 失败必须落带 error 的块，禁止静默消失 → 决策层走 0.75 fail-safe 且可见
        blocks["macro_veto"] = SignalBlock("macro_veto", "macro_veto",
                                           {"error": f"macro_veto生成失败: {str(_e5)[:80]}",
                                            "level": "unavailable", "position_coef": 0.75},
                                           confidence=0.2, error=str(_e5)[:120])
        logger.warning("macro_veto 生成失败: %s", str(_e5)[:80])
    # L1f 数据基座融合桥（榨干 macro/hot/机构/行业 未接线基座）
    try:
        _pulse = data_base_pulse()
        blocks["data_base"] = SignalBlock("data_base", "data_base_bridge", _pulse, confidence=0.7)
    except Exception as _e6:  # noqa: BLE001
        logger.warning("数据基座融合失败: %s", str(_e6)[:80])
    # L1g 全基座新鲜度门控
    try:
        _gate = gate_scores()
        blocks["freshness"] = SignalBlock("freshness", "freshness_gate", _gate, confidence=0.8)
    except Exception as _e7:  # noqa: BLE001
        blocks["freshness"] = SignalBlock("freshness", "freshness_gate",
                                          {"error": f"新鲜度门控失败: {str(_e7)[:80]}", "score": 50},
                                          confidence=0.2, error=str(_e7)[:120])
        logger.warning("新鲜度门控失败: %s", str(_e7)[:80])
    # L1h 预测验证闭环（2026-08-22 融合升级: 先验证昨日预测, 再入库今日预测）
    try:
        from prediction_verify import verify_pending  # noqa: PLC0415
        _pv = verify_pending(date)
        blocks["prediction_loop"] = SignalBlock("prediction_loop", "prediction_verify",
                                                _pv, confidence=0.8)
        if _pv.get("verified_n"):
            logger.info("📈 预测验证闭环: 本次验证 %s 条·命中 %s", _pv["verified_n"], _pv.get("hit"))
    except Exception as _e8:  # noqa: BLE001
        blocks["prediction_loop"] = SignalBlock("prediction_loop", "prediction_verify",
                                                {"error": f"预测验证失败: {str(_e8)[:90]}"},
                                                confidence=0.2, error=str(_e8)[:120])
        logger.warning("预测验证失败: %s", str(_e8)[:100])
    # L1i 引擎融合共识（2026-08-22 融合升级: 引擎输出进决策, 不再只做展示）
    try:
        from engine_fusion import engine_fused_signals  # noqa: PLC0415
        _ef = engine_fused_signals(blocks, date=date)
        blocks["engine_fusion"] = SignalBlock("engine_fusion", "engine_fusion",
                                              _ef, confidence=0.7)
        logger.info("🧠 引擎融合共识: %s (%d 引擎可用)", _ef.get("consensus"), _ef.get("ok_n"))
    except Exception as _e9:  # noqa: BLE001
        blocks["engine_fusion"] = SignalBlock("engine_fusion", "engine_fusion",
                                              {"error": f"引擎融合失败: {str(_e9)[:90]}",
                                               "consensus": "降级", "ok_n": 0},
                                              confidence=0.2, error=str(_e9)[:120])
        logger.warning("引擎融合共识失败: %s", str(_e9)[:100])
    # L1j 盘中执行基调 → 盘后决策（2026-08-23 执行逻辑集成: 盘中 bias 视角进盘后定调）
    # 读当日 intraday_chain_{date}.json 的 bias(tone/breadth/index_pct)；
    # 无文件/休市/未知 → 不注入(优雅降级, 不误报)。
    try:
        _ic_path = ROOT / "generated" / f"intraday_chain_{date}.json"
        if _ic_path.exists():
            _ic = json.loads(_ic_path.read_text(encoding="utf-8"))
            _ib = (_ic.get("bias") if isinstance(_ic, dict) else None)
            if isinstance(_ib, dict) and _ib.get("tone") not in (None, "休市", "未知", ""):
                blocks["intraday_bias"] = SignalBlock(
                    "intraday_bias", "intraday_guard", _ib, confidence=0.6)
                logger.info("📡 盘中基调接入盘后决策: %s(广度%s)",
                            _ib.get("tone"), _ib.get("breadth"))
            # D(2026-08-23): 盘中五视角机会 rows 同步注入 → 盘后选股共振排序
            _rows = (_ic.get("rows") if isinstance(_ic, dict) else None)
            if isinstance(_rows, list) and _rows:
                blocks["intraday_rows"] = SignalBlock(
                    "intraday_rows", "intraday_guard", _rows, confidence=0.5)
    except Exception as _e11:  # noqa: BLE001
        logger.warning("盘中基调接入失败: %s", str(_e11)[:80])
    # L2 决策骨架（battle_map 重算仅作 fallback，正常读产物）
    bm = _battle_map_block(date)
    # 2026-08-22 业务逻辑审计修复: battle_map 行动卡注入 blocks → fusion_decision 交叉校验
    # （此前 bm 只渲染/入库预测, 决策卡与作战卡矛盾时用户需自行调停）
    try:
        if bm and not bm.get("error"):
            blocks["battle_map_card"] = SignalBlock(
                "battle_map_card", "battle_map",
                {"action_card": bm.get("action_card") or bm.get("recommended"),
                 "position_range": bm.get("position_range"),
                 "confidence": bm.get("confidence"),
                 "expert_committee": bm.get("expert_committee"),
                 "degraded": bm.get("degraded", [])[:3]},
                confidence=0.6)
    except Exception as _e10:  # noqa: BLE001
        logger.warning("battle_map 块注入失败: %s", str(_e10)[:80])
    pools = _merge_stock_pools(blocks)
    # 渲染
    # 决策卡先算（供渲染+json 一致）
    try:
        dec = make_decision(blocks)
    except Exception as _e_d:  # noqa: BLE001
        dec = {"error": str(_e_d)[:120], "date": date}
        logger.warning("决策卡生成失败(兜底): %s", str(_e_d)[:80])
    md = _render_md(date, blocks, bm, pools, health, dec=dec)
    # 信号后验闭环：共识/温度/决策卡 入库预测
    try:
        pred_ids = _record_predictions(date, blocks, bm)
        logger.info("📈 已入库 %d 条预测（明日自动验证闭环）", len(pred_ids))
    except Exception as _e:  # noqa: BLE001
        logger.warning("预测入库失败: %s", str(_e)[:80])

    # 落盘
    out_dir = ROOT / "generated"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / f"review_{date}.md"
    _atomic_write(md_path, md)
    logger.info("📄 复盘已生成: %s", md_path)

    if save_json:
        js = {
            **release_metadata(),
            "production_components": validate_production_contract(ROOT),
            "run_id": run_id,
            "date": date, "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
            "health": {"overall_ok": health.get("overall_ok"),
                       "degraded": health.get("degraded_datasets", [])[:10]},
            "blocks": {k: b.to_dict() for k, b in blocks.items()},
            "battle_map": bm,
            "pools": pools,
            "decision": dec,
        }
        contract_error = None
        try:
            js["research_contract"] = contract_from_daily_review(js)
            js["research_contract"] = _sanitize_contract_dates(js["research_contract"], date)
            js["research_contract"]["quality"]["validated"] = bool(not validate_report_contract(js["research_contract"]))
            contract_problems = validate_report_contract(js["research_contract"])
            if contract_problems:
                contract_error = "; ".join(contract_problems[:12])
            else:
                validate_and_write_contract(out_dir / f"review_{date}.md", js["research_contract"])
        except Exception as exc:  # Keep legacy review usable, expose contract failure.
            contract_error = str(exc)[:240]
        if contract_error:
            js["research_contract_error"] = contract_error
            # 统一研报契约是交付前置条件；不能让旧 Markdown 在契约失败时继续推送。
            raise RuntimeError(f"research_report.v1 校验失败：{contract_error}")
        js_path = out_dir / f"review_{date}.json"
        _atomic_write(js_path, json.dumps(js, ensure_ascii=False, indent=1, default=str))
        logger.info("📦 结构化复盘已生成: %s", js_path)

    delivery_ok = None
    delivery_error = None
    if push:
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from feishu_sender import send_markdown  # noqa: PLC0415
            # 2026-08-21 升级: 推送完整 md(sender 自动按行分片多段), 不再截断3500
            delivery_ok = bool(send_markdown(f"📊 每日综合复盘 {date} · {PRODUCT_VERSION}\n\n" + md,
                                             title=f"📊 每日综合复盘 {date} · {PRODUCT_VERSION}"))
            logger.info("飞书推送(完整md): %s", delivery_ok)
            if not delivery_ok:
                delivery_error = "send_markdown returned false"
        except Exception as e:  # noqa: BLE001
            delivery_ok = False
            delivery_error = str(e)[:200]
            logger.warning("推送失败: %s", str(e)[:100])
    receipt_path = _write_delivery_receipt(
        data_as_of=date,
        run_id=run_id,
        requested=push,
        ok=delivery_ok,
        error=delivery_error,
    )

    manifest = {
        "run_id": run_id,
        "run_date": datetime.now(CST).date().isoformat(),
        "data_as_of": date,
        "execution_mode": "paper",
        "status": "failed" if push and delivery_ok is False else "ok",
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "artifacts": {
            "review_markdown": str(md_path),
            "review_json": str(out_dir / f"review_{date}.json") if save_json else None,
            "delivery_receipt": str(receipt_path),
        },
        "delivery": {"requested": push, "ok": delivery_ok, "error": delivery_error},
        "blocks": {"ok": sum(1 for b in blocks.values() if not b.error), "total": len(blocks)},
    }
    manifest_path = ROOT / "generated" / "runs" / run_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    return {"run_id": run_id, "date": date, "md": str(md_path),
            "manifest": str(manifest_path), "receipt": str(receipt_path),
            "delivery_ok": delivery_ok,
            "blocks_ok": sum(1 for b in blocks.values() if not b.error),
            "blocks_total": len(blocks)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()
    r = run(args.date, args.push, save_json=not args.no_json)
    print(f"✅ 复盘完成: {r['date']} | 要素可用 {r['blocks_ok']}/{r['blocks_total']} | {r['md']}")
