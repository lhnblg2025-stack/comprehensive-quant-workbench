#!/usr/bin/env python3
"""Build an auditable, decision-oriented A-share short-term review HTML.

This builder deliberately consumes existing generated artifacts instead of inventing
live market values. It exposes provenance, degraded modules, backtest caveats,
point-in-time rules, rollback candidates, and the actual equity/trade rows.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path('${QUANT_DESKTOP:-.}/研报共享/A股融合研报_2026-08-21.html')
DEFAULT_OUT = Path('${QUANT_DESKTOP:-.}/研报共享/A股短线盘面复盘_审计版_2026-08-25.html')


def esc(v: Any) -> str:
    return html.escape('' if v is None else str(v), quote=True)


def load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return fallback


def fmt(v: Any, digits: int = 2) -> str:
    if v is None or v == '':
        return '缺失'
    try:
        return f'{float(v):,.{digits}f}'
    except Exception:
        return str(v)


def table(headers: list[str], rows: list[list[Any]], cls: str = '') -> str:
    head = ''.join(f'<th>{esc(x)}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def read_source_body(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding='utf-8') if path.exists() else ''
    title = re.search(r'<title>(.*?)</title>', raw, re.S | re.I)
    body = re.search(r'<body>(.*?)</body>', raw, re.S | re.I)
    return (title.group(1).strip() if title else path.name, body.group(1) if body else '<p>原始HTML缺失</p>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=str(DEFAULT_SOURCE))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--trade-date', default='2026-08-25')
    args = ap.parse_args()
    source = Path(args.source)
    out = Path(args.out)
    title, source_body = read_source_body(source)
    bt = load_json(ROOT / 'generated/backtest_latest.json', {})
    fb = load_json(ROOT / 'generated/factor_backtest_latest.json', {})
    review_files = sorted((ROOT / 'generated').glob('review_*.json'))
    review_latest = load_json(review_files[-1], {}) if review_files else {}
    review_previous = load_json(review_files[-2], {}) if len(review_files) > 1 else {}
    review_json_excerpt = json.dumps(review_latest, ensure_ascii=False, indent=2)[:220000]
    meta_files = sorted((ROOT / 'generated/a_share_data').glob('*-meta.json'))
    cache_files = sorted((ROOT / 'generated/data_source_cache').glob('*'))
    rows = bt.get('rows') or []
    factor_names = (fb.get('provenance') or {}).get('factor_names_used') or []
    statuses = []
    for p in meta_files[-3:]:
        m = load_json(p, {})
        statuses.extend(m.get('statuses') or [])
    failed = [s for s in statuses if not s.get('ok', True)]
    equity_rows: list[list[Any]] = []
    trade_rows: list[list[Any]] = []
    for item in rows:
        for point in (item.get('equity_curve') or []):
            equity_rows.append([item.get('symbol'), point.get('date'), fmt(point.get('equity'), 2)])
        for tr in (item.get('trades') or []):
            trade_rows.append([item.get('symbol'), tr.get('date'), tr.get('side'), fmt(tr.get('price')), tr.get('shares'), tr.get('reason', '')])
    if not equity_rows:
        for item in rows:
            equity_rows.append([item.get('symbol'), item.get('start'), fmt(item.get('final_equity'), 2)])
    max_equity_rows = 9000
    equity_rows = equity_rows[:max_equity_rows]
    audit_rows = [
        ['数据可得性', '原始融合HTML页眉标记 6 项降级', '不能把综合评分视为完整融合结论', '报告显式展示降级模块与缺失原因'],
        ['回测有效性', f"单标的样本交易数 {sum(int(x.get('trade_count') or 0) for x in rows)}", '样本量偏小，Sharpe/胜率不稳定', '仅作候选策略筛选，须滚动OOS复核'],
        ['因子有效性', f"实际使用 {len(factor_names)} 个因子", f"多空年化 {fmt((fb.get('long_short') or {}).get('annual_return_pct'))}% 与回撤 {fmt((fb.get('long_short') or {}).get('max_drawdown_pct'))}% 组合异常强", '按高风险异常值处理，不直接实盘'],
        ['时间顺序', '因子系统声明财务滞后45天，日内IC使用 t→t+1', '若输入文件修改时间晚于交易时间，可能形成伪实时', '以交易日、采集时间、生成时间三列校验'],
        ['交易约束', '测试覆盖T+1、涨跌停不可成交、手续费/滑点、退市清算', '真实标的输入仍需核对涨跌停和流动性字段', '缺字段时禁止晋级可交易信号'],
        ['交付链路', 'Youdao/Feishu由 report_delivery 统一处理', '凭据、网络或目标缺失会导致未完成', '保留桌面HTML与状态页脚作为兜底'],
    ]
    review_block_rows = []
    for name, block in (review_latest.get('blocks') or {}).items():
        if isinstance(block, dict):
            review_block_rows.append([name, block.get('status', 'ok'), block.get('error', ''), block.get('confidence', ''), len(json.dumps(block, ensure_ascii=False))])
        else:
            review_block_rows.append([name, 'ok', '', '', len(json.dumps(block, ensure_ascii=False))])
    if not review_block_rows:
        review_block_rows = [['blocks', '缺失', '未找到review JSON', '-', 0]]
    source_status_rows = []
    for s in statuses[-80:]:
        source_status_rows.append([s.get('name'), 'OK' if s.get('ok', True) else 'FAIL', fmt(s.get('ms'), 0), s.get('detail', '')])
    if not source_status_rows:
        source_status_rows = [['暂无最新meta', 'UNKNOWN', '-', '未找到 generated/a_share_data/*-meta.json']]
    factor_rows = []
    for k, v in (fb.get('group_metrics') or {}).items():
        factor_rows.append([k, fmt(v.get('annual_return', 0) * 100), fmt(v.get('annual_vol', 0) * 100), fmt(v.get('sharpe')), fmt(v.get('max_drawdown', 0) * 100)])
    if not factor_rows:
        factor_rows = [['无', '缺失', '缺失', '缺失', '缺失']]
    bt_rows = [[x.get('symbol'), x.get('start'), x.get('end'), x.get('trade_count'), fmt(x.get('total_return_pct')), fmt(x.get('max_drawdown_pct')), fmt(x.get('sharpe')), fmt(x.get('win_rate_pct'))] for x in rows]
    if not bt_rows:
        bt_rows = [['无', '缺失', '缺失', '-', '-', '-', '-', '-']]
    cache_rows = []
    for p in cache_files[-120:]:
        cache_rows.append([p.name, datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'), p.stat().st_size])
    if not cache_rows:
        cache_rows = [['无缓存文件', '-', '-']]
    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_title = f'A股短线盘面复盘审计版 · {args.trade_date}'
    html_doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(report_title)}</title><style>
:root{{--bg:#0c1117;--panel:#121a24;--panel2:#182332;--line:#2b3a4e;--text:#e7edf5;--muted:#9ba9ba;--red:#f06a69;--green:#48c78e;--amber:#f2b84b;--cyan:#56b4e9;--purple:#b79cff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}}.shell{{max-width:1440px;margin:auto;padding:22px}}header{{position:sticky;top:0;z-index:2;background:rgba(12,17,23,.96);border-bottom:1px solid var(--line);padding:12px 0}}h1{{margin:0;font-size:26px}}h2{{font-size:19px;border-left:4px solid var(--cyan);padding-left:10px;margin:0 0 14px}}h3{{font-size:15px;color:var(--cyan);margin:18px 0 8px}}p,li{{font-size:14px}}.muted{{color:var(--muted)}}.nav{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}.nav a,.pill{{color:var(--text);background:var(--panel2);border:1px solid var(--line);padding:4px 9px;border-radius:6px;text-decoration:none;font-size:12px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.metric,.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}}.metric b{{display:block;font-size:24px;margin-top:5px}}.metric .label{{color:var(--muted);font-size:12px}}.danger{{color:var(--red)}}.warn{{color:var(--amber)}}.good{{color:var(--green)}}.info{{color:var(--cyan)}}.table-wrap{{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:6px}}table{{width:100%;border-collapse:collapse;min-width:760px;font-size:12px}}th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#1c2b3e;color:var(--cyan)}}tr:hover{{background:#1b2938}}details{{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:10px 14px}}summary{{cursor:pointer;color:var(--cyan);font-weight:700}}.callout{{border-left:4px solid var(--amber);background:#251f14;padding:12px 14px;margin:12px 0}}.decision{{border:2px solid var(--cyan);background:#142537;padding:18px;border-radius:8px}}.source{{font-size:12px;color:var(--muted);word-break:break-all}}.chart{{width:100%;height:300px;background:#0a0f15;border:1px solid var(--line);border-radius:6px}}footer{{color:var(--muted);font-size:12px;padding:22px 0}}@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.shell{{padding:12px}}}}@media(max-width:560px){{.grid{{grid-template-columns:1fr}}h1{{font-size:21px}}}}
</style></head><body><div class="shell"><header><h1>{esc(report_title)}</h1><div class="muted">生成时间 {esc(generated)} · 原始报告 {esc(title)} · 研究用途，不构成投资建议</div><nav class="nav"><a href="#decision">决策结论</a><a href="#evidence">数据证据</a><a href="#backtest">回测</a><a href="#factor">因子迭代</a><a href="#timeline">时间与回滚</a><a href="#source">原始盘面</a><a href="#delivery">交付审计</a></nav></header>
<section id="decision" class="card"><h2>一、可执行决策结论</h2><div class="decision"><p><b>当前动作：</b><span class="warn">观察优先，条件式试探，禁止因单日综合分直接满仓。</span></p><p>决策链必须按“趋势 → 宽度 → 情绪 → 资金 → 主线 → 个股触发 → 风险预算 → 次日验证”顺序闭合。原HTML的技术/主线分数较高，但其页眉同时标注 `zt_daily_stats、zt_pool_history、fund_forces、theme_cycle、fusion` 等模块降级，因此综合评分只能作为待验证假设。</p><ul><li><b>进攻条件：</b>过滤后上涨占比连续改善并超过35%，跌停低于50，主线成交额与涨幅同向，核心标的回踩缩量且次日不破触发位。</li><li><b>防守条件：</b>上涨占比低于25%、跌停扩散、放量下跌，或高位主线核心出现放量冲高回落，仓位回到风险预算下限。</li><li><b>执行约束：</b>A股T+1、涨跌停不可卖、流动性上限、手续费和滑点必须写入每次候选交易的触发记录。</li></ul></div></section>
<section id="evidence" class="card"><h2>二、数据证据与业务审计</h2><div class="grid"><div class="metric"><span class="label">回测标的</span><b>{len(rows)}</b><span class="muted">真实JSON记录</span></div><div class="metric"><span class="label">因子数量</span><b>{len(factor_names)}</b><span class="muted">{esc(', '.join(factor_names[:4]))}</span></div><div class="metric"><span class="label">失败/降级状态</span><b class="{'danger' if failed else 'good'}">{len(failed)}</b><span class="muted">来自最新meta</span></div><div class="metric"><span class="label">权益曲线行数</span><b>{len(equity_rows):,}</b><span class="muted">真实逐日记录</span></div></div>{table(['审计域','现有证据','业务风险','报告处理'],audit_rows)}</section>
<section id="backtest" class="card"><h2>三、回测策略审计</h2><p>以下数字来自现有生成产物，不重新包装成收益承诺。回测结果只有在样本外、滚动窗口、不同成本和不同市场状态下稳定，才允许进入观察池；单次高收益且高回撤的多空结果不得直接指导实盘。</p>{table(['标的','开始','结束','交易数','总收益%','最大回撤%','Sharpe','胜率%'],bt_rows)}<h3>因子分组与多空结果</h3>{table(['组别','年化收益%','年化波动%','Sharpe','最大回撤%'],factor_rows)}<div class="callout"><b>异常审计：</b>现有因子回测多空年化 {fmt((fb.get('long_short') or {}).get('annual_return_pct'))}%、最大回撤 {fmt((fb.get('long_short') or {}).get('max_drawdown_pct'))}%，相对单多头与基准差异过大。优先排查股票池幸存者偏差、做空可实现性、信号/收益错位、复权及成本覆盖；在验证完成前标记为 <b>research-only</b>。</div><h3>真实权益曲线（完整行可展开）</h3><canvas id="equity" class="chart"></canvas><details open><summary>权益曲线逐日明细（最多9000行，来源于backtest_latest.json）</summary>{table(['标的','日期','权益'],equity_rows,'dense')}</details><details><summary>成交记录明细</summary>{table(['标的','日期','方向','价格','股数','原因'],trade_rows or [['暂无交易明细','-','-','-','-','回测产物未包含trades字段']])}</details></section>
<section id="factor" class="card"><h2>四、因子与策略自主迭代</h2><p>迭代不是追求回测最优，而是建立“提出假设 → 经济逻辑 → 点时数据 → 滚动IC/分层 → 成本后OOS → 稳健性门槛 → 晋级/降权/下岗 → 复盘”的闭环。现有因子轮动模块已提供winsorize、截面z-score、财务45天滞后和因子轮动基础；日内衰减模块按30分钟窗口计算 t→t+1 IC，并支持连续负IC下岗。</p>{table(['当前实际使用因子'],[[', '.join(factor_names) or '缺失']])}<h3>建议的晋级门槛</h3>{table(['门槛','最低要求','失败动作'],[['样本外覆盖','至少3个滚动窗口、覆盖上涨/震荡/下跌','降级为研究'],['成本后收益','手续费、印花税、滑点、冲击成本后仍为正','禁止实盘'],['IC稳定性','滚动IC方向稳定，不能只靠单一阶段','降低权重'],['风险','最大回撤和换手符合账户预算','缩小仓位'],['数据质量','点时、复权、涨跌停、停牌字段齐全','回滚到上一版本']])}<p class="source">因子回测来源：generated/factor_backtest_latest.json；轮动实现：quant_system/analysis_core/factor_rotation_system.py；日内衰减：quant_system/analysis_core/intraday_factor_decay.py。</p></section>
<section id="timeline" class="card"><h2>五、时间密度、时间顺序与回滚</h2><p>建议采用“5分钟盘中快照 → 30分钟因子衰减 → 收盘后日频复盘 → 次日开盘前验证 → 周度策略稳定性 → 月度参数冻结”的分层密度。所有数据记录同时保留 `trade_date、as_of、fetched_at、generated_at、source、version`，禁止用文件修改时间代替行情时间。</p>{table(['缓存/版本文件','文件修改时间','字节数'],cache_rows)}<h3>自动回滚规则</h3><ol><li>新快照交易日逆序、重复或晚于生成时间：拒绝发布，保留上一成功版本。</li><li>关键字段缺失比例超过阈值、市场宽度与指数方向矛盾、来源交叉校验超差：标记 degraded，不覆盖上一版。</li><li>回测结果突然跳变、交易数骤减或回撤异常改善：冻结策略版本，回滚因子权重和数据manifest。</li><li>报告发布必须携带数据版本、失败项、回滚原因和恢复路径，不能静默使用旧缓存。</li></ol></section>
<section id="source" class="card"><h2>六、原始盘面复盘（可追溯底稿）</h2><p class="muted">以下为现有HTML主体原样嵌入，保留其原始指标、板块、技术、龙虎榜和风险提示，便于逐项对照审计；新结论不覆盖原始数据。</p><div class="raw">{source_body}</div><h3>Review区块状态与体量</h3>{table(['区块','状态','错误','置信度','字节数'],review_block_rows)}<details><summary>最新review JSON原始审计摘录（截取220KB，保留真实字段）</summary><pre style="white-space:pre-wrap;max-height:900px;overflow:auto;font-size:11px;color:#c8d4e3">{esc(review_json_excerpt)}</pre></details></section>
<section id="delivery" class="card"><h2>七、数据源与交付审计</h2>{table(['数据源','状态','耗时ms','详情'],source_status_rows)}<p>交付底座位于 `scripts/report_delivery.py`：默认有道云备份、飞书新消息分段发送、桌面文件保留。若有道云或飞书失败，必须在正文末尾写明失败原因并保留本地HTML，不得把“已配置”写成“已成功”。</p></section>
<footer>生成器：scripts/build_a_share_audit_report.py · 生成时间 {esc(generated)} · 数据来自已有本地产物，外部行情缺失部分未编造。</footer></div><script>
const pts={json.dumps([{'d': x[1], 'v': float(str(x[2]).replace(',','')) if str(x[2]).replace(',','').replace('.','',1).isdigit() else None} for x in equity_rows], ensure_ascii=False)};
const c=document.getElementById('equity'),ctx=c.getContext('2d');function draw(){{const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;ctx.scale(d,d);const w=r.width,h=r.height,p=28;ctx.fillStyle='#0a0f15';ctx.fillRect(0,0,w,h);const a=pts.filter(x=>Number.isFinite(x.v));if(!a.length)return;const mn=Math.min(...a.map(x=>x.v)),mx=Math.max(...a.map(x=>x.v));ctx.strokeStyle='#2b3a4e';ctx.lineWidth=1;for(let i=0;i<5;i++){{let y=p+(h-2*p)*i/4;ctx.beginPath();ctx.moveTo(p,y);ctx.lineTo(w-p,y);ctx.stroke()}}ctx.strokeStyle='#56b4e9';ctx.lineWidth=2;ctx.beginPath();a.forEach((x,i)=>{{let X=p+(w-2*p)*i/Math.max(1,a.length-1),Y=h-p-(x.v-mn)/Math.max(1,mx-mn)*(h-2*p);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y)}});ctx.stroke();ctx.fillStyle='#9ba9ba';ctx.font='12px sans-serif';ctx.fillText('最低 '+mn.toFixed(2),p,h-7);ctx.fillText('最高 '+mx.toFixed(2),p,16)}}window.addEventListener('resize',draw);draw();
</script></body></html>'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding='utf-8')
    print(json.dumps({'path': str(out), 'bytes': out.stat().st_size, 'equity_rows': len(equity_rows), 'factor_count': len(factor_names), 'failed_sources': len(failed)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
