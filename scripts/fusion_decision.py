#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合决策引擎 v2（2026-08-21 —— 用户核心诉求"要融合不要独立"）

六大维度(技术/情绪/资金/因子/海外/主线)加权融合 → 综合大盘评分 → 定调/仓位
- 板块融合研判: 主线 × 强度评分
- 选股融合打分: 主线龙头个股+资金+趋势 多维共振排序
- 风险融合预警: 维度背离(情绪热技术弱/海外暴跌/游资撤退)

对外接口与旧 decision_engine 兼容:
  make_decision(blocks) / render_decision_md(dec)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logger = logging.getLogger("fusion_decision")

from decision_engine_helpers import mainline_leaders  # noqa: E402
from quant_system.product_contract import PRODUCT_VERSION, release_metadata  # noqa: E402


def _num(v, d=0.0) -> float:
    try:
        v = float(v)
        return v if v == v else d
    except (TypeError, ValueError):
        return d


def _sb(blocks, key) -> dict:
    """兼容 SignalBlock 或原生 dict 读取 value。"""
    v = blocks.get(key)
    if v is None:
        return {}
    return v.value if hasattr(v, "value") else (v if isinstance(v, dict) else {})


def _d(v) -> dict:
    """dict 守卫: 内嵌毒化值(float/str/list/None)一律转 {}（2026-08-22 稳定化）。"""
    return v if isinstance(v, dict) else {}


def _stage(blocks) -> str:
    emo = _d(_d(_sb(blocks, "market_temperature")).get("components")).get("emotion")
    return str(_d(emo).get("stage_cn") or "?")


def _fund(blocks) -> dict:
    return _d(_d(_d(_sb(blocks, "market_temperature")).get("components")).get("fund"))


def _directions(blocks) -> list:
    v = _d(_sb(blocks, "strong_direction")).get("directions")
    return v if isinstance(v, list) else []


def _ladder(blocks) -> dict:
    return _d(_d(_d(_sb(blocks, "leader_sentiment")).get("components")).get("ladder"))


def _pool(blocks, key) -> list:
    v = _d(_d(_sb(blocks, "stock_picks")).get("pools")).get(key)
    return v if isinstance(v, list) else []


def _technical(blocks) -> dict:
    return _sb(blocks, "technical")


def _overseas(blocks) -> dict:
    return _sb(blocks, "overseas")


def _block_error(blocks, key: str) -> str | None:
    """Read SignalBlock/dict errors without losing degraded-state metadata."""
    raw = blocks.get(key)
    if raw is None:
        return "missing"
    err = getattr(raw, "error", None)
    if err:
        return str(err)
    if isinstance(raw, dict) and raw.get("error"):
        return str(raw.get("error"))
    return None


def _factors(blocks) -> dict:
    return _sb(blocks, "factor_signal")


def _climax_blowout(blocks) -> bool:
    """P2 修复(2026-08-23): 高潮 + 炸板率>0.42 = 退潮前兆(高潮次日分歧)。

    与 _score_emotion 的阈值一致(0.42), 用于定调档位的硬门控(禁进攻档)。
    """
    if _stage(blocks) != "高潮":
        return False
    lad = _ladder(blocks)
    zt = _num(lad.get("zt_cnt"))
    zb = _num(lad.get("zb_cnt"))
    return (zt + zb) > 0 and zb / (zt + zb) > 0.42


def _intraday_bias(blocks) -> dict:
    """盘中执行基调块(2026-08-23 执行逻辑集成: intraday_chain 的 bias 注入)。"""
    return _sb(blocks, "intraday_bias")


def _intraday_rows(blocks) -> list:
    """盘中五视角机会行(2026-08-23 阶段D: intraday_chain 的 rows 注入)。"""
    v = _sb(blocks, "intraday_rows")
    return v if isinstance(v, list) else []


_CARD_RANK_KEYWORDS = (
    (("积极", "加仓", "打板"), 3),
    (("进攻",), 2),
    (("试错", "低吸"), 1),
    (("观望", "中性", "轮动"), 1),
    (("防守", "减仓", "清仓", "回避", "空仓"), 0),
)


def _card_rank(card: str) -> int | None:
    """B1(2026-08-23): 作战卡 action_card → 与 _POSTURE_RANK 同尺的秩。

    修正关键字语义: "低吸"是进攻动作(原被误归防守)、"观望"是中性(原被误归防守)，
    消除因关键字过宽导致的假分歧；无法归类返回 None(不参与交叉校验, 不误判)。
    """
    c = str(card or "")
    if not c:
        return None
    for keys, rank in _CARD_RANK_KEYWORDS:
        if any(k in c for k in keys):
            return rank
    return None


def _selection_bias(blocks) -> dict:
    """B2(2026-08-23): 选股共振权重按情绪阶段动态调整。

    风险期(退潮/恐慌/高潮) → 降主线龙头(dim3 打板易炸)、升 RS 强势回踩(相对强);
    启动期(冰点/修复) → 主线龙头优先; 其余中性。
    返回 {src: float 加减权}, src ∈ {mainline, lhb, rs, intraday}。
    """
    stage = _stage(blocks)
    if stage in ("退潮", "恐慌"):
        return {"mainline": -2.0, "lhb": 0.0, "rs": 1.0, "intraday": 0.5}
    if stage == "高潮":
        return {"mainline": -1.0, "lhb": 0.0, "rs": 0.5, "intraday": 0.3}
    if stage in ("冰点", "修复"):
        return {"mainline": 0.5, "lhb": 0.0, "rs": -0.5, "intraday": 0.0}
    return {"mainline": 0.0, "lhb": 0.0, "rs": 0.0, "intraday": 0.0}


def _rank_picks(attack: list, bias: dict) -> list:
    """按 dim + 情绪动态加减权 排序(dim 仍用于图标/语义, src 权重只调顺序)。"""
    return sorted(attack, key=lambda x: -((x.get("dim", 0) or 0)
                                          + bias.get(x.get("src", ""), 0)))


# ═══════════════════════════════════════════════════════════
# 维度评分 (每维 0-100)
# ═══════════════════════════════════════════════════════════
def _score_technical(blocks) -> tuple[float, str, bool]:
    tec = _technical(blocks)
    idx = tec.get("indices") or []
    if not idx:
        return 50.0, "技术数据缺失⚠️", False
    main = [t for t in idx if t.get("name") in ("沪深300", "中证500", "创业板指", "中证1000")]
    pool = main or idx
    score = sum(_num(t.get("score"), 50) for t in pool) / len(pool)
    bull = sum(1 for t in pool if "多头" in str(t.get("trend")))
    return score, f"{bull}/{len(pool)}指数多头", True


def _score_emotion(blocks) -> tuple[float, str, bool]:
    """返回 (score, note, available) —— 3元组契约（2026-08-22 修复: 原实现误返回5元组）。"""
    stage = _stage(blocks)
    map_ = {"冰点": 35, "修复": 60, "发酵": 85, "复苏": 80, "高潮": 75,
            "分歧": 50, "退潮": 30, "恐慌": 20}
    if stage == "?":
        return 50.0, "情绪阶段缺失⚠️", False
    s = map_.get(stage, 50)
    ladder = _ladder(blocks)
    zt = _num(ladder.get("zt_cnt"))
    zb = _num(ladder.get("zb_cnt"))
    tt = zt + zb
    zb_rate = zb / tt if tt else 0
    note = f"{stage}·涨停{int(zt)} 炸板率{zb_rate:.0%}"
    if zb_rate > 0.35:
        s -= 10
        note += "⚠️高炸板"
    # P2 修复(2026-08-23): 高潮 + 炸板率>0.42 = 退潮前兆(高潮次日分歧风险)。
    # 原 75-10=65 仍偏高, 不足以压制进攻档 → 再 -15, 情绪面诚实反映"退潮前兆"。
    if stage == "高潮" and zb_rate > 0.42:
        s -= 15
        note += "⚠️高潮炸板退潮前兆"
    return max(5, min(95, s)), note, True


def _score_fund(blocks) -> tuple[float, str, bool]:
    f = _fund(blocks)
    fi_v = f.get("force_index")
    if fi_v is None or (isinstance(fi_v, float) and fi_v != fi_v):
        return 5.0, "资金数据缺失⚠️", False
    fi = _num(fi_v)
    youzi = _num(f.get("youzi_net")) / 1e8
    s = fi
    note = f"合力{fi:.0f} 游资{youzi:+.1f}亿"
    if youzi > 10:
        s += 8; note += " 游资活跃"
    elif youzi < -10:
        s -= 8; note += " 游资撤退"
    lhb_n = len(_pool(blocks, "lhb_stocks"))
    if lhb_n >= 3:
        s += 5; note += f" 龙虎榜{lhb_n}股"
    return max(5, min(95, s)), note, True


def _score_factor(blocks) -> tuple[float, str, bool]:
    fgs = _factors(blocks)
    fg = fgs.get("groups") if isinstance(fgs.get("groups"), dict) else {}
    if not fg:
        return 50.0, "因子数据缺失⚠️", False
    # MAJOR-7 修复: 高位占比中性≈0.40(截面60分位均值)，中心改 (avg-0.40)*100
    picks = []
    for k in ("技术动量", "质量", "量价"):
        g = fg.get(k)
        if isinstance(g, dict):
            v = g.get("score")
            if v is not None and v == v:
                picks.append(v)
    if not picks:
        return 50.0, "因子数据缺失⚠️", False
    avg = float(sum(picks) / len(picks))
    s = 50 + (avg - 0.40) * 250  # 0.40→50(中性), 0.44→60, 0.36→40
    note = "动量/质量/量价占优" if s >= 58 else ("因子中性" if s >= 45 else "因子偏弱")
    return max(5, min(95, s)), note, True


def _score_overseas(blocks) -> tuple[float, str, bool]:
    data = _overseas(blocks)
    _q = data.get("quotes")
    ov = [q for q in (_q if isinstance(_q, list) else []) if isinstance(q, dict)]
    # 分组覆盖用于审计：海外因子是辅助确认，不以单一指数替代全球风险面。
    groups = data.get("factor_groups") if isinstance(data.get("factor_groups"), dict) else {}
    if data.get("score_eligible") is False:
        failed = (data.get("coverage") or {}).get("failed_groups") or []
        return 50.0, f"海外核心组覆盖不足⚠️({','.join(failed)})", False
    if len(ov) < 2:
        reason = data.get("error") or data.get("status") or "合格报价不足"
        return 50.0, f"海外数据降级⚠️({reason})", False
    s = 50.0
    notes = []
    groups_seen = {}
    for q in ov:
        label = q.get("label", "")
        asset_id = str(q.get("asset_id") or q.get("symbol") or "")
        chg = _num(q.get("chg_pct"))
        if label in ("纳指", "纳指100", "纳指(NDX)", "标普500", "道指") or asset_id in ("^IXIC", "usNDX", "gb_ixic", "^GSPC", "usINX", "gb_inx", "^DJI", "usDJI", "gb_dji"):
            if chg < -1:
                s -= 8; notes.append(f"{label}{chg:+.1f}%⚠️")
            elif chg > 1:
                s += 4; notes.append(f"{label}{chg:+.1f}%")
        if label == "伦敦金" and chg > 2:
            s -= 5; notes.append("金急涨避险⚠️")
        if label == "费城半导体":
            if chg < -2:
                s -= 6; notes.append(f"费城半导体{chg:+.1f}%⚠️")
            elif chg > 2:
                s += 3; notes.append(f"费城半导体{chg:+.1f}%")
        if label == "WTI原油" and abs(chg) > 3:
            s -= 3; notes.append(f"原油波动{chg:+.1f}%⚠️")
        if label == "美10年债殖" and chg > 1:
            s -= 3; notes.append("美债急升⚠️")
        # 新增海外组：商品/半导体/港股中概作为辅助确认，单组贡献封顶。
        if label in ("白银", "布伦特原油"):
            groups_seen.setdefault("commodities", []).append(chg)
        if label in ("恒生科技", "腾讯控股", "阿里巴巴", "美团", "京东"):
            groups_seen.setdefault("hk_china_internet", []).append(chg)
        if label in ("英伟达", "费城半导体"):
            groups_seen.setdefault("semiconductor", []).append(chg)
    if groups_seen.get("semiconductor"):
        avg = sum(groups_seen["semiconductor"]) / len(groups_seen["semiconductor"])
        s += max(-5.0, min(5.0, avg * 1.5))
        if avg < -2:
            notes.append(f"半导体组{avg:+.1f}%⚠️")
    if groups_seen.get("hk_china_internet"):
        avg = sum(groups_seen["hk_china_internet"]) / len(groups_seen["hk_china_internet"])
        s += max(-4.0, min(4.0, avg * 1.2))
        if avg < -2:
            notes.append(f"港股中概组{avg:+.1f}%⚠️")
    if groups_seen.get("commodities"):
        avg = sum(groups_seen["commodities"]) / len(groups_seen["commodities"])
        if abs(avg) > 3:
            s -= 2
            notes.append(f"商品波动组{avg:+.1f}%⚠️")
    coverage = ",".join(f"{k}:{len(v)}" for k, v in groups.items() if isinstance(v, list))
    note = " ".join(notes[:3]) if notes else "海外平稳"
    if coverage:
        note = f"{note} | 覆盖{coverage}"
    return max(5, min(95, s)), note, True


def _fallback_directions(blocks) -> list[dict]:
    """Use already-computed engine evidence when the dedicated collector times out."""
    ef = _sb(blocks, "engine_fusion")
    raw = ef.get("dimensions") if isinstance(ef, dict) else {}
    if not isinstance(raw, dict):
        return []
    candidates = []
    for key, item in raw.items():
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("sector") or item.get("theme")
        score = item.get("score")
        if name and score is not None:
            candidates.append({"name": str(name), "level": "回填观察", "score": _num(score), "fallback": True})
    return candidates[:10]


def _score_mainline(blocks) -> tuple[float, str, bool]:
    direction_block = _sb(blocks, "strong_direction")
    error = _block_error(blocks, "strong_direction")
    dirs = [d for d in (_directions(blocks) or []) if isinstance(d, dict)]
    fallback = False
    if error:
        dirs = _fallback_directions(blocks)
        fallback = bool(dirs)
        if not dirs:
            return 30.0, f"主线数据降级⚠️({error})", False
    mains = [d for d in dirs if d.get("level") == "主线" or d.get("fallback")]
    if not mains:
        return 30.0, "扫描完成但无明确主线", True
    s = 60 + min(len(mains), 4) * 5
    # MAJOR-10: 计入主线 score 强度(0-3 量级)
    sc = [d.get("score") for d in mains if d.get("score") is not None]
    if sc:
        s += sum(float(x) for x in sc if x == x) / len(sc) * 8
    ladder = _ladder(blocks)
    mb = _num(ladder.get("max_board"))
    if mb >= 4:
        s += 5
    note = f"{len(mains)}条主线" + (f" 最高{int(mb)}板" if mb else "")
    if fallback:
        note = "回填引擎证据·" + note
    return max(5, min(95, s)), note, True


def _score_engine(blocks) -> tuple[float, str, bool]:
    """引擎融合共识（第7维，2026-08-22）：读 engine_fusion 适配器输出。

    少于3个可用引擎视为信号不足（available=False，不进加权，防单引擎主导）。
    """
    ef = _sb(blocks, "engine_fusion")
    if not ef:
        return 50.0, "引擎信号缺失⚠️", False
    ok_n = _num(ef.get("ok_n"))
    if ok_n < 3:
        return 50.0, f"引擎信号不足({int(ok_n)}<3)⚠️", False
    s = _num(ef.get("score"), 50)
    consensus = str(ef.get("consensus") or "?")
    note = f"{consensus}·{int(ok_n)}引擎 {ef.get('note', '')}"[:60]
    return max(5, min(95, s)), note, True


WEIGHTS = {"technical": 0.1818181818, "emotion": 0.1818181818, "fund": 0.1636363636,
           "factor": 0.1090909091, "overseas": 0.1363636364, "mainline": 0.1363636364,
           "engine": 0.0909090909}  # 第7维: 引擎融合共识；比例与旧版一致，声明和=1.0


_POSTURE_RANK = {"防守观望": 0, "中性轮动": 1, "试探性进攻": 2, "积极进攻": 3}


def apply_intraday_guard(posture: str, pos: tuple[float, float],
                         bias: dict) -> tuple[str, tuple[float, float], str]:
    """盘中实际基调 → 盘后定调门控（2026-08-23 执行逻辑集成）。

    盘中 scan 的 bias.tone（防御/谨慎/进攻）反映当日真实执行环境：
      - 防御 → 最高"中性轮动"，仓位上限 ≤40%（禁进攻）
      - 谨慎 → 最高"试探性进攻"，仓位上限 ≤50%（禁积极进攻）
      - 进攻/未知/缺失 → 不调整
    返回 (新posture, 新pos, note)；note 为空表示无门控。
    """
    b = bias if isinstance(bias, dict) else {}
    tone = str(b.get("tone") or "")
    note = ""
    if tone == "防御":
        if _POSTURE_RANK.get(posture, 0) > 1:  # 仅当需降档时才产生门控 note
            posture = "中性轮动"
            pos = (min(pos[0], 0.2), min(pos[1], 0.4))
            note = f"盘中防御基调(广度{b.get('breadth')})→禁进攻"
    elif tone == "谨慎":
        if _POSTURE_RANK.get(posture, 0) > 2:
            posture = "试探性进攻"
            pos = (min(pos[0], 0.3), min(pos[1], 0.5))
            note = "盘中谨慎基调→禁积极进攻"
    return posture, pos, note


def make_decision(blocks: dict) -> dict:
    """融合决策 v3（审计修复: 缺失重归一/硬门槛/veto防fail-open）。"""
    profs = {
        "technical": _score_technical(blocks),
        "emotion": _score_emotion(blocks),
        "fund": _score_fund(blocks),
        "factor": _score_factor(blocks),
        "overseas": _score_overseas(blocks),
        "mainline": _score_mainline(blocks),
        "engine": _score_engine(blocks),  # 第7维 引擎融合共识 (2026-08-22)
    }
    # CRITICAL-1: 缺失维度不进加权, 余下权重重归一
    avail = {k: v[2] for k, v in profs.items()}
    active = [k for k, a in avail.items() if a]
    w_sum = sum(WEIGHTS[k] for k in active) or 1.0
    total = sum(profs[k][0] * WEIGHTS[k] / w_sum for k in active)
    n_miss = 7 - len(active)
    miss_txt = f"{n_miss}/7 维度缺失⚠️" if n_miss else ""

    # 全基座新鲜度门控 → 决策降权 (2026-08-21; 2026-08-22 升级: 决策关键域加权)
    # 陈旧基座比例高 ⇒ 决策置信下调, 最多 -8 分(防止用旧数据激进决策)
    # 修复: 原实现对全部18域一刀切(industry/social等非决策域过期也扣分) →
    #       改为只对决策关键域(kline/market/zt/lhb/财务/估值/宏观/事件/行业)计分。
    _gate = _sb(blocks, "freshness")
    _gscore = _num(_gate.get("score"), 100)
    _scan = _gate.get("scan") if isinstance(_gate.get("scan"), dict) else {}
    _CRIT = {"kline", "market", "zt_history", "lhb_hist", "financial",
             "valuation", "macro", "events", "industry"}
    gate_penalty = 0
    gate_note = ""
    if _scan:
        crit = {k: v for k, v in _scan.items()
                if k in _CRIT and isinstance(v, dict)}
        if crit:
            c_bad = {k: v for k, v in crit.items()
                     if str(v.get("level") or "") in ("expired", "stale", "no_data", "error", "unknown")}
            _gscore = round(100 * (1 - len(c_bad) / max(1, len(crit))), 1)
            if _gscore < 100 and _gscore > 0:
                gate_penalty = round((100 - _gscore) * 0.10, 1)
                gate_penalty = min(8.0, gate_penalty)
                total = total - gate_penalty
                gate_note = f"关键域门控{_gscore:.0f}分 -{gate_penalty}分"
                if c_bad:
                    gate_note += " 异常:" + ",".join(
                        f"{k}:{v.get('level')}({v.get('lag')}d)" for k, v in sorted(c_bad.items())[:4])
            elif _gscore <= 0:
                gate_note = "关键域基座数据缺失/陈旧"
    elif 0 < _gscore < 100:
        gate_penalty = round((100 - _gscore) * 0.10, 1)
        gate_penalty = min(8.0, gate_penalty)
        total = total - gate_penalty
        gate_note = f"基座门控{_gscore:.0f}分 -{gate_penalty}分"
    elif _gscore <= 0:
        gate_note = "基座门控数据缺失"

    # 定调档位：进攻档必须有真实可用的技术数据，缺失不能用占位50分通过硬门槛。
    _t, _e = profs["technical"][0], profs["emotion"][0]
    _technical_available = bool(profs["technical"][2])
    if total >= 68 and _technical_available and _t >= 55 and _e >= 60:
        posture, pos = "积极进攻", (0.5, 0.8)
    elif total >= 55 and _technical_available and _t >= 45:
        posture, pos = "试探性进攻", (0.3, 0.5)
    elif total >= 42:
        posture, pos = "中性轮动", (0.2, 0.4)
    else:
        posture, pos = "防守观望", (0.0, 0.2)
    if not _technical_available and _POSTURE_RANK.get(posture, 0) > 1:
        posture, pos = "中性轮动", (0.0, min(pos[1], 0.2))

    # L0 门控：关键数据全部不可用时，不生成可执行进攻仓位。
    if _gscore <= 0:
        posture, pos = "防守观望", (0.0, 0.0)

    # 长期趋势硬门控：上证指数收盘跌破MA300时，禁止“试探性/积极进攻”。
    # 不能只把它作为评分项，否则短线情绪或个别板块会把整体仓位重新推高。
    _sh = next((x for x in _d(_technical(blocks)).get("indices", [])
                if str(x.get("name") or "") in ("上证指数", "上证")), {})
    _sh_long_state = _sh.get("above_ma300")
    ma300_guard_note = ""
    if _sh_long_state is False:
        if _POSTURE_RANK.get(posture, 0) >= 2:
            posture = "中性轮动"
        pos = (0.0, min(pos[1], 0.2))
        ma300_guard_note = "⚠️ 上证跌破MA300 → 禁止进攻，最高20%观察仓"
    elif _sh_long_state is None:
        if _POSTURE_RANK.get(posture, 0) >= 3:
            posture = "试探性进攻"
        pos = (min(pos[0], 0.1), min(pos[1], 0.3))
        ma300_guard_note = "⚠️ 上证MA300数据不足 → 禁止积极进攻，最高30%试错仓"

    # CRITICAL-2 + MAJOR-5: veto 修复
    #  - 块缺失 → 视为 soft(0.6) 降档防 fail-open(无否决=全仓)
    #  - 只有 soft/mild 系数才乘档位(不再双重计数)?
    #   审计结论: 与情绪/资金共用源已被计一次 → 改为 veto 只收敛 limiter,
    #   乘在档位区间上但 soft 系数提至 0.75
    veto = _sb(blocks, "macro_veto")
    veto_raw = blocks.get("macro_veto")
    has_veto = veto_raw is not None and not getattr(veto_raw, "error", None) and bool(veto)
    if has_veto:
        coef = _num(veto.get("position_coef", 0.9), 0.75)
    else:
        coef = 0.75  # 缺失 → soft 降档 (fail-safe)
    veto_note = ""
    if coef < 0.95:
        vl = str(veto.get("level") or "soft") if has_veto else "缺失"
        pe = _num(veto.get("pe_percentile")) * 100
        _tr = (veto.get("trade_hits") or veto.get("reasons") or [])[:2]
        tr_txt = " ".join(str(x)[:24] for x in _tr) if _tr else ""
        veto_note = f"·宏观否决{vl}(PE分位{pe:.0f}% 仓位×{coef:.2f}) {tr_txt}".strip()
        if has_veto and str(veto.get("level")) == "hard":
            posture = "防守观望"
        elif not has_veto and total >= 55:
            posture = "防守观望"
        pos = (max(0.0, pos[0] * coef), pos[1] * coef)

    # P2 修复(2026-08-23): 高潮+炸板>0.42 退潮前兆 → 禁进攻档(宁保守勿激进)。
    # 在 veto 收敛后执行, 与 cross_note 同级; 语义: 情绪面已诚实降分, 档位再硬门控。
    guard_note = ""
    if _climax_blowout(blocks) and posture in ("积极进攻", "试探性进攻"):
        posture = "中性轮动"
        pos = (min(pos[0], 0.2), min(pos[1], 0.4))
        guard_note = "⚠️ 高潮炸板退潮前兆 → 禁进攻档(降至中性轮动)"
    # 执行逻辑集成(2026-08-23): 盘中实际基调 → 盘后定调门控(禁进攻/禁积极进攻)。
    _ib = _intraday_bias(blocks)
    _ib_note = ""
    if _ib:
        posture, pos, _ib_note = apply_intraday_guard(posture, pos, _ib)
        if _ib_note:
            guard_note = (guard_note + "；" if guard_note else "") + _ib_note

    # 2026-08-22 业务逻辑审计修复: battle_map 行动卡交叉校验（决策一致性）
    # 主链为唯一决策真源; 作战卡作为第二意见 → 分歧必须可见, 不静默覆盖。
    # 注: 必须在 veto 收敛 posture/pos 之后计算, 否则用未终值误判。
    cross_note = ""
    _bmc = _sb(blocks, "battle_map_card")
    if _bmc:
        _card = str(_bmc.get("action_card") or "")
        _bm_pos = str(_bmc.get("position_range") or "")
        _bm_up = None
        if "-" in _bm_pos:
            try:
                _bm_up = float(_bm_pos.split("-")[1].replace("%", "").strip()) / 100.0
            except Exception:  # noqa: BLE001
                _bm_up = None
        _cr = _card_rank(_card)
        _pos_lo, _pos_hi = pos[0], pos[1]
        _gap = (_pos_hi - _bm_up) * 100 if _bm_up is not None else 0.0
        if _cr is not None:
            _pr = _POSTURE_RANK.get(posture, 0)
            # 规则1: 方向分歧 —— 作战卡中性/防守 vs 主链进攻（主链仓位明显更满）
            if _cr <= 1 and _pr >= 2 and _gap > 5:
                cross_note = (f"⚠️ 决策分歧: 主链{posture}{_pos_lo*100:.0f}-{_pos_hi*100:.0f}% "
                              f"vs 作战卡[{_card}]{_bm_pos or '?'}，建议按下限执行")
            # 规则2: 宏观否决压制 —— 作战卡仓位远高于否决后主链仓位，以 veto 为准
            elif (str(veto.get("level")) == "hard" and _bm_up is not None and _gap < -20):
                cross_note = (f"⚠️ 宏观否决压制: 作战卡({_bm_pos})显著高于否决后主链"
                              f"({_pos_lo*100:.0f}-{_pos_hi*100:.0f}%)，以宏观否决为准")
            # 规则3: 作战卡激进 vs 主链防守（主链更保守时不追高）
            elif (_cr >= 2 and _pr <= 1 and _gap < -5):
                cross_note = (f"ℹ️ 作战卡更积极([{_card}]{_bm_pos or '?'})，主链保持{posture}——"
                              f"宁保守勿激进（引擎共识{_sb(blocks, 'engine_fusion').get('consensus', '?')}）")
            # 规则4: 幅度背离 —— 作战卡明显低于主链仓位
            elif _gap > 15:
                cross_note = (f"ℹ️ 作战卡仓位({_bm_pos or '?'})明显低于主链({_pos_lo*100:.0f}-{_pos_hi*100:.0f}%)"
                              f"，注意风险预算")

    _main_dirs = _directions(blocks)
    if _block_error(blocks, "strong_direction"):
        _main_dirs = _fallback_directions(blocks)
    mains = [d for d in _main_dirs if isinstance(d, dict) and (d.get("level") == "主线" or d.get("fallback"))]
    sectors = []
    for d in mains[:5]:
        ssec = min(90, 50 + _num(d.get("score")) * 12)
        sectors.append({"name": d.get("name"), "score": round(ssec), "level": d.get("level")})

    attack = _fused_stock_picks(blocks, [d.get("name") for d in mains])
    observe = _fused_observe(blocks)
    risks = _fused_risks(blocks, profs, total)
    # 执行逻辑集成(2026-08-23): 盘中基调为防御/谨慎时, 进风险预案(次日延续视角)。
    if _ib and str(_ib.get("tone") or "") in ("防御", "谨慎"):
        risks.insert(0, {
            "trigger": f"盘中实际基调{_ib.get('tone')}(广度{_ib.get('breadth')})",
            "action": ("次日延续谨慎、不追高" if str(_ib.get("tone")) == "谨慎"
                       else "防御延续、轻仓等待"),
        })
    # B3(2026-08-23): 风险预案↔仓位联动 —— 风险条数≥4 且处于进攻档 → 自动降一档。
    # 业务语义: 风险预案不是"提示后仍照旧", 高密度风险应实质收敛仓位(宁保守勿激进)。
    if len(risks) >= 4:
        _rp = _POSTURE_RANK.get(posture, 0)
        if _rp == 3:
            posture, pos = "试探性进攻", (min(pos[0], 0.3), min(pos[1], 0.5))
        elif _rp == 2:
            posture, pos = "中性轮动", (min(pos[0], 0.2), min(pos[1], 0.4))
        if _rp >= 2:
            guard_note = (guard_note + "；" if guard_note else "") + \
                f"⚠️ 风险预案{len(risks)}条偏多 → 自动降一档"
    if ma300_guard_note:
        guard_note = (guard_note + "；" if guard_note else "") + ma300_guard_note
    detail = {k: {"score": round(v[0], 1), "note": v[1], "available": v[2]} for k, v in profs.items()}

    return {
        "product_version": PRODUCT_VERSION,
        "date": _sb(blocks, "market_temperature").get("date", "?"),
        "fusion": {"total": round(total, 1), "dimensions": detail, "weights": WEIGHTS,
                   "gate": {"score": _gscore, "penalty": gate_penalty, "note": gate_note}},
        "定调": {
            "posture": posture,
            "position": f"{pos[0]*100:.0f}-{pos[1]*100:.0f}%",
            "basis": f"综合{total:.0f}/100 · {'、'.join(f'{k}{v[0]:.0f}分' for k, v in profs.items())}",
            "veto_note": veto_note,
            "cross_note": cross_note,
            "guard_note": guard_note,
        },
        "主线研判": {
            "main": "、".join(d.get("name", "") for d in mains[:4]) if mains else "无明确主线",
            "quality": "强" if total >= 60 else ("中" if total >= 45 else "弱"),
            "sectors": sectors,
        },
        "操作清单": {"attack": attack, "observe": observe, "avoid": []},
        "风险预案": risks,
        "决策依据": {"融合分": round(total, 1), "门控扣分": gate_penalty, "门控分": _gscore,
                  **{k: round(v[0], 1) for k, v in profs.items()}},
    }


def _fused_stock_picks(blocks, main_names: list) -> list:
    attack = []
    seen = set()
    bias = _selection_bias(blocks)  # B2: 情绪阶段动态权重
    try:
        leaders = mainline_leaders(blocks, [m for m in main_names if m], top_n=5)
        for l in leaders:
            nm = f'{l.get("name")}({l.get("code")})'
            if nm not in seen:
                attack.append({"name": nm, "type": f"主线{l.get('industry')}",
                               "action": "龙头跟随/低吸", "why": f"{l.get('board')}板",
                               "dim": 3, "src": "mainline"})
                seen.add(nm)
    except Exception as e:  # noqa: BLE001
        logger.warning("主线龙头取数失败: %s", str(e)[:60])
    for s in [x for x in _pool(blocks, "lhb_stocks")[:3] if isinstance(x, dict)]:
        nm = f'{s.get("name")}({s.get("code")})'
        if nm not in seen:
            attack.append({"name": nm, "type": "龙虎榜", "action": "观察溢价承接",
                           "why": f"净买{_num(s.get('net')):.1f}亿", "dim": 2, "src": "lhb"})
            seen.add(nm)
    for s in [x for x in _pool(blocks, "rs_stocks")[:2] if isinstance(x, dict)]:
        nm = f'{s.get("name")}({s.get("code")})'
        if nm not in seen:
            attack.append({"name": nm, "type": "RS强势", "action": "回踩关注",
                           "why": f"榜#{s.get('rank')}", "dim": 1, "src": "rs"})
            seen.add(nm)
    # D(2026-08-23): 盘中五视角机会(高置信)进盘后选股共振排序
    for r in [x for x in _intraday_rows(blocks)[:4] if isinstance(x, dict)]:
        conf = _num(r.get("composite_conf"))
        if conf < 0.55:
            continue
        code = str(r.get("symbol") or "").strip()
        nm = f'{r.get("name", "")}({code})' if code else ""
        if not code or nm in seen:
            continue
        attack.append({"name": nm, "type": "盘中机会", "action": "盘中强势回踩关注",
                       "why": f"置信{conf:.0%} {r.get('dominant_label', '')}",
                       "dim": 2, "src": "intraday"})
        seen.add(nm)
    return _rank_picks(attack, bias)[:6]


def _fused_observe(blocks) -> list:
    ob = []
    for s in [x for x in _pool(blocks, "lhb_stocks")[3:6] if isinstance(x, dict)]:
        ob.append({"name": f'{s.get("name")}({s.get("code")})', "type": "龙虎榜跟进",
                   "action": "看是否高开承接", "why": f"净买{_num(s.get('net')):.1f}亿"})
    return ob[:4]


def _fused_risks(blocks, profs, total) -> list:
    risks = []
    emo_s, tech_s = profs["emotion"][0], profs["technical"][0]
    if emo_s >= 70 and tech_s < 50:
        risks.append({"trigger": "情绪热但技术弱(指数未企稳)", "action": "只做超短快进快出"})
    if tech_s >= 70 and emo_s < 40:
        risks.append({"trigger": "技术强但情绪冰点(无量反弹)", "action": "轻仓试错防诱多"})
    if profs["overseas"][0] <= 40:
        risks.append({"trigger": "海外转弱(美指/金急涨)", "action": "降仓回避高波动"})
    f = _fund(blocks)
    if _num(f.get("youzi_net")) / 1e8 < -15:
        risks.append({"trigger": "游资大幅净流出", "action": "防空头反扑兑现利润"})
    # MAJOR-9 补: 北向大幅流出 / 跌停比 / 涨停断层 / 炸板率恶化
    north = _num(f.get("north_net"))
    if north < -50e8:
        risks.append({"trigger": f"北向大幅流出({north/1e8:.0f}亿)", "action": "防外资重仓股补跌"})
    ladder = _ladder(blocks)
    zt = _num(ladder.get("zt_cnt"))
    dt = _num(ladder.get("dt_cnt"))
    if dt > zt and zt > 0:
        risks.append({"trigger": f"跌停({int(dt)})多于涨停({int(zt)})", "action": "情绪恶化,降仓防御"})
    max_board = _num(ladder.get("max_board"))
    if max_board <= 2 and zt < 40:
        risks.append({"trigger": "连板断层(最高≤2板+涨停稀)", "action": "无高度,只做首板不做接力"})
    if _stage(blocks) == "高潮":
        risks.append({"trigger": "高潮次日分歧", "action": "龙头断板即减仓"})
    if total < 42:
        risks.append({"trigger": "综合分持续<42", "action": "空仓或3成以下防守"})
    # 严重度排序: hard风险在前
    return risks[:6]


def render_decision_md(dec: dict) -> str:
    if not isinstance(dec, dict):
        return "## 🎯 今日决策\n- 决策数据异常（非 dict）"
    L = ["## 🎯 今日决策（多维度融合）"]
    dz = dec.get("定调") or {}
    fz = dec.get("fusion") or {}
    L.append(f"- **明日定调**: {dz.get('posture')} | 建议仓位 **{dz.get('position')}**")
    if dz.get("veto_note"):
        L.append(f"- {dz.get('veto_note')}")
    if dz.get("cross_note"):
        L.append(f"- {dz.get('cross_note')}")
    if dz.get("guard_note"):
        L.append(f"- 🛡️ {dz.get('guard_note')}")
    L.append(f"- 融合评分: **{fz.get('total')}**/100 | {dz.get('basis')}")
    _gtg = fz.get("gate") if isinstance(fz.get("gate"), dict) else {}
    if _gtg.get("note"):
        L.append(f"- 🗃️ {_gtg['note']}")
    L.append("")
    dims = fz.get("dimensions") if isinstance(fz.get("dimensions"), dict) else {}
    if dims:
        L.append("### 📊 七大维度融合")
        for k, cn in [("technical", "技术面"), ("emotion", "情绪面"), ("fund", "资金面"),
                      ("factor", "因子面"), ("mainline", "主线面"), ("overseas", "海外面"),
                      ("engine", "引擎面")]:
            v = dims.get(k)
            if isinstance(v, dict) and v.get("score") is not None:
                L.append(f"- {cn}: **{v['score']:.0f}**分 ({v.get('note', '')})")
        L.append("")
    L.append("### 🔍 主线研判（融合）")
    mz = dec.get("主线研判") if isinstance(dec.get("主线研判"), dict) else {}
    L.append(f"- 主线: **{mz.get('main')}**（质量{mz.get('quality')}）")
    for sec in [s for s in (mz.get("sectors") or [])[:4] if isinstance(s, dict)]:
        L.append(f"- {sec['name']}: 强度{sec['score']} ({sec.get('level')})")
    L.append("")
    op = dec.get("操作清单") if isinstance(dec.get("操作清单"), dict) else {}
    L.append("### ✅ 进攻清单（多维共振排序）")
    if op.get("attack"):
        for a in [x for x in op["attack"] if isinstance(x, dict)]:
            tag = "🔥共振" if a.get("dim", 0) >= 3 else ("💧资金" if a.get("dim", 0) == 2 else "📈趋势")
            L.append(f"- {tag} **{a['name']}** [{a['type']}] {a['action']} — {a.get('why', '')}")
    else:
        L.append("- 无主线加持标的")
    L.append("")
    L.append("### 👀 观察清单")
    for o in [x for x in (op.get("observe") or [])[:4] if isinstance(x, dict)]:
        L.append(f"- {o['name']} [{o['type']}] {o['action']} — {o.get('why', '')}")
    L.append("")
    risks = [r for r in (dec.get("风险预案") or []) if isinstance(r, dict)]
    L.append("### 🚨 融合风险预警（维度背离）")
    if risks:
        for r in risks:
            L.append(f"- ⚠️ {r.get('trigger')} → **{r.get('action')}**")
    elif dz.get("veto_note"):
        L.append(f"- ⚠️ 宏观否决生效: {dz['veto_note']}")
    else:
        L.append("- 维度一致，无显著背离")
    return "\n".join(L)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from daily_fusion_base import get_gateway
    from daily_review_collectors import collect_all
    from daily_quant_collectors import collect_quant_all
    from daily_review_chain import _HAS_OVERSEAS, _HAS_TECH
    gw = get_gateway()
    blocks = collect_all(gw)
    blocks.update(collect_quant_all(gw))
    if _HAS_OVERSEAS:
        from overseas_collector import collect_overseas
        blocks["overseas"] = collect_overseas(gw)
    if _HAS_TECH:
        from technical_collector import collect_technical
        blocks["technical"] = collect_technical(gw)
    dec = make_decision(blocks)
    print(render_decision_md(dec))