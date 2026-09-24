#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⚠️ 已废弃（2026-08-22 标注）：本文件为 v1 旧版研报生成器，仅作历史保留。
正式研报生成器是 scripts/html_report_generator.py（详实版，含引擎矩阵/共识/预测闭环/缠论图）。
请勿再用本文件产出研报；quant_web 仅 glob 兼容旧文件展示。

旧版说明（2026-08-21 —— 用户要求"输出html保存到研报共享文件夹，内容丰富具体"）：
从融合复盘 review_{date}.json → 精美 HTML 研报（自包含单页，含图表/表格/决策卡），
保存到 ~/Desktop/研报共享/。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CST = timezone(timedelta(hours=8))

# 研报输出目录: 优先真实桌面 ~/Desktop/研报共享；沙箱只读时退回 workspace/研究报告
_HOME = Path.home()
_OUT_DESKTOP = _HOME / "Desktop" / "研报共享"
try:
    _OUT_DESKTOP.mkdir(parents=True, exist_ok=True)
    _t = _OUT_DESKTOP / ".wtest"
    _t.write_text("ok", encoding="utf-8"); _t.unlink()
    OUT_DIR = _OUT_DESKTOP
    _writable = True
except OSError:
    OUT_DIR = ROOT / "研究报告"
    _writable = False
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_WRITABLE = _writable


def _load_review(date: str | None = None) -> dict:
    gen = ROOT / "generated"
    if date:
        p = gen / f"review_{date}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    files = sorted(gen.glob("review_*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _fmt_num(v, d=1):
    try:
        f = float(v)
        if f >= 1e8: return f"{f/1e8:.2f}亿"
        if f >= 1e4: return f"{f/1e4:.1f}万"
        return f"{f:.{d}f}"
    except (TypeError, ValueError):
        return "-"


def _stage_color(stage: str) -> str:
    hot = {"发酵", "复苏", "高潮"}
    cold = {"退潮", "恐慌", "冰点"}
    if stage in hot: return "#e35d5b"
    if stage in cold: return "#4a90d9"
    return "#e0a13c"


def build_html(review: dict) -> str:
    date = review.get("date", "")
    blocks = review.get("blocks", {})
    dec = review.get("decision", {})
    fz = dec.get("fusion", {})
    bm = review.get("battle_map", {}) or {}

    def blk(k):
        b = blocks.get(k) or {}
        return b.get("value", {}) if isinstance(b.get("value"), dict) else {}

    # ── 数据 ─────────────────────────────
    temp = blk("market_temperature")
    emo = (temp.get("components") or {}).get("emotion", {})
    fund = (temp.get("components") or {}).get("fund", {})
    macro = (temp.get("components") or {}).get("macro", {})
    sector = blk("strong_direction").get("directions", [])
    leader = blk("leader_sentiment")
    ladder = (leader.get("components") or {}).get("ladder", {})
    lhb = blk("lhb").get("lhb", {})
    pools = blk("stock_picks").get("pools", {})
    factor = blk("factor_signal").get("groups", {})
    technical = blk("technical").get("indices", [])
    overseas = blk("overseas").get("quotes", [])
    patterns = blk("technical").get("patterns", [])
    verdict = blk("model_verdict").get("verdict", {})
    health = review.get("health", {})

    stage = emo.get("stage_cn", "?")
    dims = fz.get("dimensions") or {}
    total = fz.get("total")
    dz = dec.get("定调", {})
    mz = dec.get("主线研判", {})
    atk = (dec.get("操作清单") or {}).get("attack", [])
    obs = (dec.get("操作清单") or {}).get("observe", [])
    risks = dec.get("风险预案", [])
    veto_note = dz.get("veto_note", "")

    dim_order = [("technical", "技术面"), ("emotion", "情绪面"), ("fund", "资金面"),
                 ("factor", "因子面"), ("mainline", "主线面"), ("overseas", "海外面")]
    dim_html = "".join(
        f'''<div class="dim"> <div class="dim-name">{cn}</div>
        <div class="dim-bar"><div class="dim-fill" style="width:{min(100, v.get('score',0))}%;background:{'#e35d5b' if v.get('score',0)>=65 else ('#4a90d9' if v.get('score',0)<45 else '#e0a13c')}"></div></div>
        <div class="dim-score">{v.get('score',0):.0f}</div><div class="dim-note">{v.get('note','')}</div></div>'''
        for k, cn in dim_order if (v := dims.get(k)))

    # 决策矩阵卡
    posture = dz.get("posture", "?")
    pos_color = "#e35d5b" if "进攻" in posture else ("#4a90d9" if "防守" in posture else "#e0a13c")

    tech_rows = "".join(f'''<tr><td>{t.get('name')}</td><td class="num">{t.get('last')}</td>
        <td><span class="tag" style="background:{'#2d6a4f' if t.get('score',50)>=65 else ('#7a4d2d' if t.get('score',50)>=45 else '#3d5a80')}">{t.get('score',50):.0f}分</span></td>
        <td>{t.get('trend','')}</td><td class="num">MA20 {t.get('ma20')}</td><td class="num">RSI {t.get('rsi')}</td>
        <td class="num">S {t.get('support')} R {t.get('resistance')}</td></tr>'''
        for t in technical[:6]) or "<tr><td colspan=7>暂无</td></tr>"

    ov_html = "".join(f'''<tr><td>{q.get('label')}</td><td class="num">{q.get('price')}</td>
        <td class="{'up' if q.get('chg_pct',0)>0 else 'down'}">{'+' if q.get('chg_pct',0)>0 else ''}{q.get('chg_pct',0):.2f}%</td></tr>'''
        for q in overseas) or "<tr><td colspan=3>暂无</td></tr>"

    atk_html = "".join(f'''<li><span class="atk-tag {'t3' if a.get('dim',0)>=3 else ('t2' if a.get('dim',0)==2 else 't1')}">
        {'🔥共振' if a.get('dim',0)>=3 else ('💧资金' if a.get('dim',0)==2 else '📈趋势')}</span>
        <b>{a.get('name')}</b> <span class="muted">[{a.get('type')}]</span>
        <span>{a.get('action')}</span> <span class="muted">{a.get('why','')}</span></li>'''
        for a in atk) or "<li>暂无主线加持标的</li>"

    obs_html = "".join(f"<li>{o.get('name')} <span class='muted'>[{o.get('type')}] {o.get('action')}</span></li>"
                       for o in obs) or "<li>暂无</li>"

    risk_html = "".join(f"<li>⚠️ <b>{r.get('trigger')}</b> → {r.get('action')}</li>"
                        for r in risks) or "<li>维度一致，无显著背离</li>"

    factor_html = "".join(f'''<tr><td>{k}</td><td><div class="mini-bar"><div style="width:{min(100, v.get('score',0)*100)}%;background:{'#e35d5b' if v.get('score',0)>=0.38 else '#4a90d9'}"></div></div></td><td>{v.get('score',0)*100:.0f}%</td></tr>'''
                          for k, v in factor.items()) or "<tr><td colspan=3>暂无</td></tr>"

    pools_html = ""
    pool_shown = {
        "short_term": "短线攻击方向", "lhb_stocks": "龙虎榜强势(个股)",
        "rs_stocks": "RS相对强弱", "mid_long": "中长线低估",
    }
    for key, name in pool_shown.items():
        items = pools.get(key, [])
        if not items:
            continue
        inner = "".join(f"<li>{it.get('name') or it.get('concept') or it.get('code','')} "
                        f"<span class='muted'>{it.get('signal') or ('#'+str(it.get('rank',''))) if it.get('rank') else (it.get('level') or '')}</span></li>"
                        for it in items[:5])
        pools_html += f"<div class='pool'><h4>🎯 {name}</h4><ul>{inner or '<li class=muted>—</li>'}</ul></div>"

    pattern_html = "".join(f"<li>{p[:90]}</li>" for p in patterns[:6]) or "<li>暂无形态信号</li>"

    health_txt = "✅ 数据健康" if health.get("overall_ok") else \
        f"⚠️ 降级 {len(health.get('degraded', []))} 项: {', '.join(health.get('degraded', [])[:5])}"

    # ── HTML ─────────────────────────────
    # 数据基座/门控 HTML
    dbb = blk("data_base")
    _db_rows = []
    _mk2 = dbb.get("macro", {}); 
    if _mk2.get("items"):
        _db_rows.append(f"<tr><td>宏观</td><td>{' · '.join(str(i.get('name'))+'='+str(i.get('value')) for i in _mk2['items'][:3])}</td></tr>")
    _hk2 = dbb.get("hot", {})
    if _hk2.get("top"):
        _db_rows.append(f"<tr><td>热榜</td><td>涨{_hk2.get('up_n')}跌{_hk2.get('down_n')} {','.join(str(t.get('code')) for t in _hk2['top'][:4])}</td></tr>")
    _ik2 = dbb.get("institutional", {})
    if _ik2.get("new_entries"):
        _db_rows.append(f"<tr><td>机构新进</td><td>{', '.join(str(e.get('name')) for e in _ik2['new_entries'][:4])}</td></tr>")
    _dk2 = dbb.get("industry", {})
    if _dk2.get("strong"):
        _db_rows.append(f"<tr><td>强行业</td><td>{', '.join(str(x.get('name')) for x in _dk2['strong'][:4])}</td></tr>")
    _st2 = dbb.get("style", {})
    if _st2.get("note"):
        _db_rows.append(f"<tr><td>风格</td><td>{_st2['note']}</td></tr>")
    _ev2 = dbb.get("event", {})
    if _ev2.get("note"):
        _db_rows.append(f"<tr><td>事件</td><td>{_ev2['note']}</td></tr>")
    _db_html = f"<table>{''.join(_db_rows) or '<tr><td>无基座数据</td></tr>'}</table>"
    fg = blk("freshness")
    _gate_html = f"<p>健康度 <b>{fg.get('score','-')}</b>/100 · fresh {fg.get('fresh','-')} / stale {fg.get('stale','-')} / expired {fg.get('expired','-')}</p>"
    nf = fg.get("not_fresh") or []
    if nf:
        _gate_html += f"<p class='sub'>陈旧: {' '.join(str(x.get('dir'))+'('+str(x.get('lag'))+'d)' for x in nf[:6])}</p>"

    return f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>每日A股融合研报 {date}</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--card2:#1c2333;--line:#2d3748;--tx:#e6edf3;--sub:#8b949e;
--red:#e35d5b;--grn:#3fb950;--amber:#e0a13c;--blue:#4a90d9}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;padding:24px;max-width:1100px;margin:0 auto}}
h1{{font-size:24px;margin-bottom:4px}} .sub{{color:var(--sub);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}
.card h2{{font-size:17px;margin-bottom:12px;border-left:4px solid var(--blue);padding-left:8px}}
.verdict{{background:linear-gradient(135deg,#1a2438,#202c47);border:2px solid var(--blue);border-radius:14px;padding:22px;margin-bottom:18px}}
.verdict .row{{display:flex;align-items:center;gap:18px;flex-wrap:wrap}}
.verdict .act{{font-size:26px;font-weight:800;color:var(--red)}}
.verdict .pos{{font-size:15px;color:var(--sub)}}
.score-big{{font-size:38px;font-weight:800}}
.fusion-total{{text-align:center;padding:8px;background:var(--card2);border-radius:8px;margin:8px 0}}
.dim{{display:grid;grid-template-columns:90px 1fr 50px 180px;gap:8px;align-items:center;padding:5px 0;border-bottom:1px dashed var(--line)}}
.dim-name{{font-size:13px;color:var(--sub)}}
.dim-bar{{height:10px;background:var(--card2);border-radius:5px;overflow:hidden}}
.dim-fill{{height:100%;border-radius:5px}}
.dim-score{{text-align:right;font-weight:700;font-size:14px}}
.dim-note{{font-size:11px;color:var(--sub)}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid var(--line)}}
th{{color:var(--sub);font-weight:500}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.up{{color:var(--red)}} .down{{color:var(--grn)}} .muted{{color:var(--sub);font-size:12px}}
.tag{{padding:1px 8px;border-radius:10px;font-size:11px;color:#fff}}
ul{{list-style:none}} li{{padding:5px 0;font-size:13px;border-bottom:1px dashed var(--line)}}
li:last-child{{border:none}}
.pools{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.pool{{background:var(--card2);border-radius:10px;padding:12px}}
.atk-tag{{padding:1px 7px;border-radius:8px;font-size:11px;margin-right:6px}}
.t3{{background:#3d1f1f;color:var(--red);border:1px solid var(--red)}}
.t2{{background:#1f3d2f;color:var(--grn);border:1px solid var(--grn)}}
.t1{{background:#3d3020;color:var(--amber);border:1px solid var(--amber)}}
.mini-bar{{height:8px;background:var(--card);border-radius:4px;overflow:hidden}}
.mini-bar>div{{height:100%;border-radius:4px}}
.banner{{background:var(--card2);border-radius:10px;padding:12px 16px;font-size:13px;margin-bottom:16px}}
@media(max-width:700px){{.pools{{grid-template-columns:1fr}}.dim{{grid-template-columns:60px 1fr 40px}}}}
</style></head><body>
<h1>📊 每日A股融合研报 · {date}</h1>
<div class="sub">生成 {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} · 六大维度融合决策 · {health_txt} · 本报告仅供参考</div>

<div class="verdict"><div class="row">
  <div style="flex:1">
    <div class="sub">明日定调 · 综合评分 {total if total else '-'}</div>
    <div class="act">{posture}</div>
    <div class="pos">建议仓位 <b>{dz.get('position','?')}</b>{veto_note and f' · {veto_note}' or ''}</div>
    <div class="pos" style="margin-top:4px">{dz.get('basis','')}</div>
  </div>
  <div class="score-big" style="color:{pos_color}">{total if total else '-'}</div>
</div></div>

<div class="card"><h2>📊 六大维度融合评分</h2>{dim_html}</div>

<div class="card"><h2>🎯 进攻清单（多维共振）</h2><ul>{atk_html}</ul>
<h2 style="margin-top:14px">👀 观察清单</h2><ul>{obs_html}</ul>
<h2 style="margin-top:14px">🚨 风险预警（维度背离）</h2><ul>{risk_html}</ul></div>

<div class="card"><h2>🔍 主线研判</h2>
<p style="font-size:14px">主线: <b>{mz.get('main')}</b> <span class="tag" style="background:var(--blue)">{mz.get('quality')}</span></p>
<table><tr><th>方向</th><th>强度</th><th>等级</th></tr>
{''.join(f"<tr><td>{s.get('name')}</td><td class='num'>{s.get('score')}</td><td>{s.get('level')}</td></tr>" for s in (mz.get('sectors') or [])[:6]) or '<tr><td colspan=3>暂无</td></tr>'}
</table></div>

<div class="card"><h2>🌡️ 市场温度</h2>
<p style="font-size:14px">情绪 <b style="color:{_stage_color(stage)}">{stage}</b> · 置信{emo.get('confidence')} · 涨停 <b class="up">{ladder.get('zt_cnt')}</b> 炸板{ladder.get('zb_cnt')} 跌停{ladder.get('dt_cnt')} 最高<b>{ladder.get('max_board')}</b>板</p>
<p class="sub">资金合力 {fund.get('force_index',0):.0f} · 游资 {_fmt_num(fund.get('youzi_net'))} · 机构 {_fmt_num(fund.get('jg_net'))} · 北向 {_fmt_num(fund.get('north_net'))} · 宏观 {macro.get('regime','-')}</p></div>

<div class="card"><h2>📈 技术分析（指数）</h2>
<table><tr><th>指数</th><th>收盘</th><th>技术分</th><th>趋势</th><th>均线</th><th>RSI</th><th>支撑/压力</th></tr>{tech_rows}</table></div>

<div class="card"><h2>💰 龙虎榜资金</h2>
<div class="pools">
<div class="pool"><h4>游资营业部</h4><ul>{''.join(f"<li>{o.get('name','')[:18]} <span class='{'up' if (o.get('net') or 0)>=0 else 'down'}'>{_fmt_num(o.get('net'))}</span></li>" for o in (lhb.get('broker_top') or [])[:5]) or '<li class=muted>暂无</li>'}</ul></div>
<div class="pool"><h4>个股净买</h4><ul>{''.join(f"<li>{o.get('name','')}({o.get('code','')}) <span class='up'>{_fmt_num(o.get('net'))}</span> {o.get('pct',0):.1f}%</li>" for o in (lhb.get('stock_top') or [])[:5]) or '<li class=muted>暂无</li>'}</ul></div></div></div>

<div class="card"><h2>🎯 多格局选股池</h2><div class="pools">{pools_html}</div></div>

<div class="card"><h2>📐 因子强度</h2>
<table><tr><th>风格</th><th>高位占比</th><th>强度</th></tr>{factor_html}</table>
<p class="sub" style="margin-top:8px">强势因子: {', '.join(blk('factor_signal').get('top_factors', [])[:6]) or '-'}</p>
<p class="sub">走弱因子: {', '.join(blk('factor_signal').get('weak_factors', [])[:6]) or '-'}</p></div>

<div class="card"><h2>🗄️ 数据基座融合（八域）</h2>
{_db_html}
</div>
<div class="card"><h2>🗃️ 基座新鲜度门控</h2>
{_gate_html}
</div>
<div class="card"><h2>🌍 海外市场</h2>
<table><tr><th>标的</th><th>价格</th><th>涨跌</th></tr>{ov_html}</table></div>

<div class="card"><h2>🧬 形态信号（技术引擎）</h2><ul>{pattern_html}</ul></div>

<div class="card"><h2>🤖 模型集体意见</h2>
<p>专家委员会共识: <b style="font-size:16px">{verdict.get('consensus','-')}</b> (置信{verdict.get('confidence')}) · {verdict.get('votes')}票</p></div>

<div class="banner">⚠️ 免责声明：本报告由量化融合链自动生成，数据可能存在延迟或缺失，仅供研究参考，不构成投资建议。</div>
</body></html>'''


def main(date: str | None = None) -> Path:
    review = _load_review(date)
    if not review:
        print("❌ 无复盘 json（需先跑 daily_review_chain.py）")
        sys.exit(1)
    date = review.get("date", "unknown")
    html = build_html(review)
    p = OUT_DIR / f"A股融合研报_{date}.html"
    p.write_text(html, encoding="utf-8")
    print(f"✅ 研报已生成: {p} ({len(html)}B)")
    if not REPORT_WRITABLE:
        print("⚠️ 桌面只读(沙箱)，已存 workspace/研究报告 —— 可手动复制到 ~/Desktop/研报共享/")
    return p


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    main(date)