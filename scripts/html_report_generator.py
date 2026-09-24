#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTML 研报生成器 v2·详实版（2026-08-22 —— 用户批评 13.7KB 太薄）

全量渲染全部 15 blocks: 决策卡(六维评分条)/多日趋势/操作清单/市场温度/
强势方向全/龙头情绪/龙虎榜全表/选股池四池全/模型意见/因子全/ML/
技术7指数全指标/海外8项/基座八域/门控16基座全表/宏观否决/形态信号
主报告目标 200-250KB；体量必须来自真实候选、来源、风险和历史证据，不使用填充文本。
"""
from __future__ import annotations

import glob, hashlib, json, os, re, shutil, sys, uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent.parent
CST = timezone(timedelta(hours=8))

sys.path.insert(0, str(ROOT))
from quant_system.product_contract import PRODUCT_NAME, PRODUCT_VERSION, release_metadata  # noqa: E402

_OUT_DESKTOP = Path(os.environ.get("QUANT_REPORT_ROOT", str(Path.home() / "Desktop" / "研报共享")).strip())
_VISION_JSONL = ROOT / "data_warehouse" / "ima_export" / "media" / "vision_analysis.jsonl"


def _report_root() -> Path:
    """研报共享根目录：共享目录优先，不可写时退到 workspace/研究报告（返回真实可用根）。"""
    try:
        _OUT_DESKTOP.mkdir(parents=True, exist_ok=True)
        _t = _OUT_DESKTOP / ".wtest"; _t.write_text("ok", encoding="utf-8"); _t.unlink()
        return _OUT_DESKTOP
    except OSError:
        return ROOT / "研究报告"


OUT_DIR = _report_root()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_review(date=None):
    """读最新 review json。2026-08-23 稳定化: 文件损坏/解析失败 → 降级空 dict, 不炸渲染。"""
    gen = ROOT / "generated"

    def _safe_load(p):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    if date:
        p = gen / f"review_{date}.json"
        return _safe_load(p) if p.exists() else {}
    try:
        files = sorted(gen.glob("review_*.json"))
    except Exception:  # noqa: BLE001
        return {}
    if files:
        rv = _safe_load(files[-1])
        return rv if isinstance(rv, dict) else {}
    return {}


def _fmt_num(v, d=1):
    try:
        f = float(v)
        if f >= 1e8: return f"{f/1e8:.2f}亿"
        if f >= 1e4: return f"{f/1e4:.1f}万"
        return f"{f:.{d}f}"
    except (TypeError, ValueError):
        return "-"


def _load_json_file(path: Path) -> dict:
    try:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
    except Exception:
        pass
    return {}


def _artifact_at_or_before(path: Path, report_date: str) -> bool:
    """A date-less cache may be used only if it predates the report close."""
    try:
        cutoff = datetime.strptime(report_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=CST
        ).timestamp()
        return path.exists() and path.stat().st_mtime <= cutoff
    except Exception:
        return False


def _load_json_as_of(path: Path, report_date: str) -> dict:
    return _load_json_file(path) if _artifact_at_or_before(path, report_date) else {}


def _render_strategy_conclusions(date: str) -> str:
    """回测/因子只输出对当前决策的启示，不展示审计脚手架。"""
    fac = _load_json_as_of(ROOT / "generated" / "factor_backtest_latest.json", date)
    bt = _load_json_as_of(ROOT / "generated" / "backtest_latest.json", date)
    fg = _load_json_as_of(ROOT / "generated" / "factor_generator_report_2026-08-22.json", date)
    parts = []

    if fac:
        prov = fac.get("provenance") or {}
        period = prov.get("period") or {}
        ls = fac.get("long_short") or {}
        g1 = fac.get("long_group1") or {}
        bm = fac.get("benchmark") or {}
        groups = fac.get("group_metrics") or {}
        parts.append('<h3>因子组合当前表现</h3>')
        parts.append(f'<p class="sub">样本{escape(str(prov.get("universe","-")))} · {escape(str(period.get("start","-")))}→{escape(str(period.get("end","-")))} · 月频调仓 · 持有{escape(str(prov.get("horizon","-")))}日 · 成本{escape(str(prov.get("cost_bps","-")))}bps · {escape(str(prov.get("n_groups","-")))}分组</p>')
        parts.append(_tbl(["组合", "年化%", "最大回撤%", "夏普"], [
            ["多空(G1-G5)", ls.get("annual_return_pct","-"), ls.get("max_drawdown_pct","-"), ls.get("sharpe","-")],
            ["多头G1", g1.get("annual_return_pct","-"), g1.get("max_drawdown_pct","-"), g1.get("sharpe","-")],
            ["基准", bm.get("annual_return_pct","-"), bm.get("max_drawdown_pct","-"), bm.get("sharpe","-")],
        ]))
        if groups:
            parts.append(_tbl(["分组", "年化", "波动", "夏普", "回撤"], [[k, v.get("annual_return","-"), v.get("annual_vol","-"), v.get("sharpe","-"), v.get("max_drawdown","-")] for k, v in groups.items()]))
        if isinstance(g1.get("max_drawdown_pct"), (int, float)) and g1.get("max_drawdown_pct") < -40:
            parts.append('<p>策略结论：多头分层在样本期内回撤极深，纯多头因子策略不适合满仓进攻，短线应控仓、低吸与轮动为主。</p>')
        if isinstance(ls.get("sharpe"), (int, float)) and ls.get("sharpe") >= 1:
            parts.append('<p>策略结论：多空维度强于多头，说明因子有区分度，但A股单边做空受限，只用于强弱排序而非直接做空。</p>')

    if fg:
        valid = fg.get("valid_factors") or []
        all_f = fg.get("factors") or []
        valid_names = [x.get("name") for x in valid]
        parts.append('<h3>有效因子与选股方向</h3>')
        parts.append(f'<p class="sub">候选{fg.get("n_candidates", len(all_f))}个，通过检验{fg.get("n_valid", len(valid))}个：{"、".join(valid_names) or "-"}</p>')
        if valid_names:
            parts.append('<p>策略结论：短期反转与低波合成因子有效，选股优先超跌企稳+低波动，与主线低吸一致，不追高动量。</p>')
        parts.append(_tbl(["因子", "家族", "IC均值", "ICIR", "覆盖率", "状态"], [[x.get("name",""), x.get("family",""), x.get("ic_mean","-"), x.get("icir","-"), x.get("coverage","-"), "有效" if x.get("is_valid") else "观察/淘汰"] for x in all_f[:40]]))

    if bt:
        rows = [[x.get("symbol",""), x.get("start",""), x.get("end",""), x.get("total_return_pct","-"), x.get("max_drawdown_pct","-"), x.get("sharpe","-"), x.get("trade_count","-")] for x in (bt.get("rows") or [])]
        parts.append('<h3>单标的策略回测（仓位参考）</h3>')
        parts.append(_tbl(["标的","开始","结束","总收益%","最大回撤%","夏普","交易数"], rows))
        parts.append('<p>策略结论：历史回测仅作仓位与风控参考，实盘以当日主线、资金、情绪为准。</p>')

    return "".join(parts) or '<p class="sub">暂无可用回测/因子结论，当前决策以当日量价、资金和情绪为主。</p>'


def _render_factor_effectiveness(report_date: str) -> str:
    """因子有效性与方向校准：仅使用报告日之前可验证的产物。"""
    import csv

    parts = []
    oos = _load_json_as_of(ROOT / "generated" / "ic_report" / "IC_OOS_REPORT.json", report_date)
    if oos:
        params = oos.get("params") or {}
        factors = oos.get("factors") or []
        keep = [f for f in factors if f.get("sign_keep")]
        flipped = [f for f in factors if not f.get("sign_keep")]
        keep = sorted(keep, key=lambda x: abs(x.get("icir") or 0), reverse=True)
        rows = [[f.get("factor", ""), f"{f.get('ic_mean_train', 0):+.4f}", f"{f.get('ic_mean_test', 0):+.4f}",
                 round(f.get("icir") or 0, 2), f"{f.get('winrate', 0):.0%}"] for f in keep[:30]]
        parts.append('<h3>样本外有效因子（OOS方向稳定）</h3>')
        parts.append(f'<p class="sub">150只样本 · 未来{escape(str(params.get("forward", "-")))}日 · 训练{escape(str(params.get("train_days", "-")))}天/测试{escape(str(params.get("test_days", "-")))}天 · 方向稳定 {len(keep)} 个 · 翻转 {len(flipped)} 个</p>')
        parts.append(_tbl(["因子", "训练IC", "测试IC", "测试ICIR", "胜率"], rows))
        parts.append('<p>策略结论：仅上述方向稳定因子进入选股权重；方向翻转因子已剔除或降权，避免用训练期结论误判样本外。</p>')

    ic_csv = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"
    if _artifact_at_or_before(ic_csv, report_date):
        rows = []
        try:
            with ic_csv.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                entries = [r for r in reader]
            grade_order = {"A": 0, "B": 1}
            entries = sorted(entries, key=lambda r: (grade_order.get(r.get("grade", "C"), 2), -abs(float(r.get("icir") or 0))))
            for r in entries[:40]:
                icir = float(r.get("icir") or 0)
                direction = "正向" if r.get("direction") == "1" else ("反向" if r.get("direction") == "-1" else r.get("direction", "-"))
                rows.append([r.get("factor", ""), r.get("category", ""), r.get("grade", ""),
                             f'{float(r.get("ic_mean") or 0):+.3f}', round(icir, 2), f'{float(r.get("winrate") or 0):.0%}', direction])
        except Exception:
            rows = []
        if rows:
            parts.append('<h3>全因子IC排名（当前有效方向）</h3>')
            parts.append(_tbl(["因子", "类别", "评级", "IC均值", "ICIR", "胜率", "方向"], rows))
            cats = [r[1] for r in rows[:15]]
            top_cat = max(set(cats), key=cats.count) if cats else "-"
            parts.append(f'<p>策略结论：当前高评级因子集中在 {escape(top_cat)} 等类别，选股时优先暴露这些方向，避免逆势押注低IC因子。</p>')

    calib_path = ROOT / "generated" / "ic_report" / "direction_calibration.json"
    calib = _load_json_file(calib_path) if _artifact_at_or_before(calib_path, report_date) else {}
    if calib:
        changed = calib.get("details") or []
        parts.append('<h3>因子方向校准</h3>')
        parts.append(f'<p class="sub">共 {calib.get("total", 0)} 因子 · 方向一致 {calib.get("consistent", 0)} · 已校准 {calib.get("changed", 0)} · IC为零 {calib.get("ic_zero", 0)} · 未登记 {len(calib.get("not_in_registry") or [])}</p>')
        rows = [[c.get("factor", ""), c.get("category", ""), c.get("old_direction", "-"), c.get("new_direction", "-"), f'{c.get("ic_mean", 0):+.4f}'] for c in changed[:15]]
        parts.append(_tbl(["因子", "类别", "旧方向", "新方向", "IC均值"], rows))
        parts.append('<p>策略结论：方向校准后的因子才可用于当前选股，未校准历史方向不作为权重依据。</p>')

    return "".join(parts) or '<p class="sub">暂无因子有效性/OOS产物，选股以量价、资金与主线为准。</p>'

# ── 研报图表解析章节（共享渲染，供 html_report_generator / build_html_daily_report 复用）──

_IMA_CSS = """
<style>
.degraded-alert{background:#3d1f1f;border:1px solid #f06a69;border-left:4px solid #f06a69;border-radius:6px;padding:9px 12px;margin:8px 0;color:#ffd9d8;font-size:12px}
.ima-sec{position:relative}
.ima-tools{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.ima-tools input,.ima-tools button{background:#0d1420;border:1px solid var(--line,#2a3650);color:var(--tx,#e8edf4);border-radius:6px;padding:6px 10px;font-size:12px}
.ima-card{border:1px solid var(--line,#2a3650);border-radius:10px;margin-bottom:12px;background:var(--card2,#182130);overflow:hidden}
.ima-card-head{padding:10px 14px;cursor:pointer;display:flex;align-items:center;gap:8px;flex-wrap:wrap;user-select:none}
.ima-badge{background:#1e3a5f;color:#7fb2ff;border-radius:10px;padding:2px 8px;font-size:11px;white-space:nowrap}
.ima-card-body{padding:12px 14px;border-top:1px dashed var(--line,#2a3650)}
.ima-tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.ima-tabs button{background:transparent;border:1px solid var(--line,#2a3650);color:var(--sub,#8ea0b8);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer}
.ima-tabs button.on{background:#1e3a5f;color:#cfe1ff;border-color:#2f5c9e}
.ima-img{max-width:100%;max-height:340px;object-fit:contain;background:#0a0e17;border-radius:8px;display:block;cursor:zoom-in;border:1px solid var(--line,#2a3650)}
.ima-ocr{white-space:pre-wrap;word-break:break-word;font-size:11px;line-height:1.6;color:var(--sub,#8ea0b8);max-height:260px;overflow:auto;background:#0d1420;border:1px solid var(--line,#2a3650);border-radius:8px;padding:10px}
.ima-summary{font-size:13px;margin-bottom:8px}
.ima-pts{list-style:disc;padding-left:18px;font-size:12px;margin-bottom:8px}
.ima-pts li{padding:2px 0;border-bottom:none}
table.ima-tbl{width:100%;border-collapse:collapse;font-size:12px;margin:6px 0}
table.ima-tbl th,table.ima-tbl td{padding:5px 8px;border-bottom:1px solid var(--line,#2a3650);text-align:left;word-break:break-all}
table.ima-tbl th{color:var(--sub,#8ea0b8);font-weight:500;background:#182334}
.ima-flowbar{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:8px 0;font-size:11px;color:#fff;min-width:160px}
.ima-fb-in{background:#00b368;display:flex;align-items:center;justify-content:center;white-space:nowrap;padding:0 6px}
.ima-fb-out{background:#f23645;display:flex;align-items:center;justify-content:center;white-space:nowrap;padding:0 6px}
.ima-chart{width:100%;height:210px}
.ima-range{display:flex;gap:6px;margin:6px 0;flex-wrap:wrap}
.ima-range button{background:transparent;border:1px solid var(--line,#2a3650);color:var(--sub,#8ea0b8);border-radius:6px;padding:2px 10px;font-size:11px;cursor:pointer}
.ima-range button.on{background:#1e3a5f;color:#cfe1ff;border-color:#2f5c9e}
.ima-entity{display:inline-block;border:1px solid var(--line,#2a3650);border-radius:10px;padding:1px 8px;font-size:11px;margin:2px 4px 2px 0;color:var(--sub,#8ea0b8)}
.ima-nodata{color:var(--sub,#8ea0b8);font-size:12px;font-style:italic}
.ima-conf{font-size:12px;color:var(--amber,#e0a13c)}
.ima-hidden{display:none!important}
.ima-fail{background:#3d1f1f;border:1px solid #6b2a2a;border-radius:8px;padding:8px 12px;font-size:12px;margin-bottom:8px}
.ima-fold{margin-left:auto;color:var(--sub,#8ea0b8);font-size:11px}
.ima-up{color:#f23645}.ima-down{color:#00b368}.ima-flat{color:var(--sub,#8ea0b8)}
@media(max-width:700px){.ima-card-head{flex-direction:column;align-items:flex-start}.ima-chart{height:180px}}
</style>
"""

_IMA_JS_TMPL = """
<script>
(function(){
  var PALETTE=['#4a90d9','#f0b429','#00b368','#f23645','#a06ee0','#4dd0e1'];
  var charts={};
  function num(v){
    if(typeof v==='number') return v;
    if(v==null||String(v).trim()==='') return null;
    var s=String(v).replace(/,/g,'').trim();
    var m=s.match(/^(-?\\d+(\\.\\d+)?)/);
    if(!m) return null;
    var n=parseFloat(m[1]);
    if(/亿/.test(s)) n*=1e8; else if(/万/.test(s)) n*=1e4;
    return n;
  }
  function isDateLike(x){ return /^\\d{4}[-/]\\d{1,2}/.test(String(x)); }
  function initCharts(){
    document.querySelectorAll('script.ima-series-data').forEach(function(s){
      var idx=s.getAttribute('data-idx');
      var wrap=document.getElementById('imaChart-'+idx);
      if(!wrap) return;
      var seriesList=[]; try{ seriesList=JSON.parse(s.textContent)||[]; }catch(e){ seriesList=[]; }
      var valid=seriesList.filter(function(sr){ return sr&&Array.isArray(sr.points)&&sr.points.length; });
      if(!valid.length){ wrap.innerHTML='<div class="ima-nodata">图片未提供可重绘数据</div>'; return; }
      if(!window.echarts){ wrap.innerHTML='<div class="ima-nodata">ECharts 未加载（经网页服务 /report.html 访问可显示图表）</div>'; return; }
      var lineLike=valid.every(function(sr){ var x=sr.points[0]&&sr.points[0].date; return isDateLike(x); });
      var allPoints=valid.map(function(sr){ return sr.points.map(function(p){ return {x:p.date,y:num(p.value)}; }).filter(function(p){ return p.y!=null; }); });
      function render(n){
        var sliced=allPoints.map(function(pts){ return n>0?pts.slice(-n):pts; });
        // 各序列按自己的日期映射到日期并集，避免复用第一序列日期造成错位。
        var cats=[];
        sliced.forEach(function(pts){ pts.forEach(function(p){ if(cats.indexOf(p.x)===-1) cats.push(p.x); }); });
        cats.sort(function(a,b){ return String(a).localeCompare(String(b)); });
        var option={
          tooltip:{trigger:'axis'},
          legend:{data:valid.map(function(sr,i){ return sr.name||('系列'+(i+1)); }),textStyle:{color:'#8ea0b8'}},
          grid:{left:52,right:16,top:30,bottom:24},
          xAxis:{type:'category',data:cats,axisLabel:{color:'#8ea0b8',fontSize:10,interval:Math.max(0,Math.floor(cats.length/10)-1)}},
          yAxis:{type:'value',axisLabel:{color:'#8ea0b8'}},
          series:valid.map(function(sr,i){
            var byDate={};
            sliced[i].forEach(function(p){ byDate[p.x]=p.y; });
            return {name:sr.name||('系列'+(i+1)),type:lineLike?'line':'bar',smooth:lineLike,
              data:cats.map(function(date){ return byDate[date]===undefined?null:byDate[date]; }),
              itemStyle:{color:PALETTE[i%PALETTE.length]}};
          })
        };
        if(charts[idx]){ charts[idx].setOption(option,true); }
        else { charts[idx]=window.echarts.init(wrap); charts[idx].setOption(option); }
      }
      render(0);
      var range=document.getElementById('imaRange-'+idx);
      if(range){ range.querySelectorAll('button').forEach(function(b){
        b.addEventListener('click',function(){
          range.querySelectorAll('button').forEach(function(x){x.classList.remove('on');});
          b.classList.add('on');
          render(parseInt(b.getAttribute('data-r')||'0',10));
        });
      }); }
    });
  }
  function bindTabs(){
    document.querySelectorAll('.ima-card').forEach(function(card){
      card.querySelectorAll('.ima-tabs button').forEach(function(b){
        b.addEventListener('click',function(){
          var t=b.getAttribute('data-tab');
          card.querySelectorAll('.ima-tabs button').forEach(function(x){x.classList.remove('on');});
          b.classList.add('on');
          card.querySelectorAll('.ima-tab').forEach(function(x){ x.hidden=(x.getAttribute('data-tab')!==t); });
        });
      });
      var head=card.querySelector('.ima-card-head'), body=card.querySelector('.ima-card-body');
      if(head&&body){
        head.addEventListener('click',function(){
          body.classList.toggle('ima-hidden');
          var fold=head.querySelector('.ima-fold');
          if(fold) fold.textContent=body.classList.contains('ima-hidden')?'▶':'▼';
        });
      }
    });
  }
  function bindFilter(){
    document.querySelectorAll('.ima-filter-input').forEach(function(inp){
      inp.addEventListener('input',function(){
        var q=inp.value.trim().toLowerCase();
        document.querySelectorAll('.ima-card').forEach(function(card){
          var hay=(card.getAttribute('data-search')||'').toLowerCase();
          card.classList.toggle('ima-hidden',!!q&&hay.indexOf(q)===-1);
        });
      });
    });
  }
  function boot(){ initCharts(); bindTabs(); bindFilter(); }
  if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded',boot); } else { boot(); }
})();
</script>
"""


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _safe_asset_name(name: str) -> str:
    stem, ext = os.path.splitext(str(name))
    stem = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", stem).strip("_") or "image"
    ext = (ext or ".png").lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        ext = ".png"
    return f"{stem[:80]}{ext}"


def _copy_image_asset(src: Path, assets_dir: Path, name: str) -> str:
    """复制原图到当天报告目录 assets/，返回相对 URL（assets/xxx.png）。

    名称 = 安全文件名 + 源路径短哈希：同名不同源不冲突，同源重复构建幂等复用。
    """
    import hashlib
    assets_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_asset_name(name)
    h = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    target = assets_dir / f"{Path(safe).stem[:60]}_{h}{Path(safe).suffix}"
    if target.exists():
        return f"assets/{target.name}"  # 已复制过（同一来源），幂等复用
    try:
        shutil.copy2(src, target)
    except OSError:
        return ""
    return f"assets/{target.name}"


def _load_vision_insights(date=None) -> tuple[list[dict], list[dict]]:
    """读 vision_analysis.jsonl → (成功记录, 失败记录)。

    只取 report_date 精确命中的当日记录；没有命中时明确返回空结果，不回退旧批次。
    """
    records: list[dict] = []
    failed: list[dict] = []
    if not _VISION_JSONL.exists():
        return records, failed
    for line in _VISION_JSONL.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("status") in ("ok", "partial"):
            records.append(rec)
        elif rec.get("status") == "error":
            failed.append(rec)
    if date:
        records = [r for r in records if r.get("report_date") == date]
        failed = [r for r in failed if r.get("report_date") == date]
    return records, failed


def _ima_metric_rows(metrics: list) -> str:
    if not metrics:
        return ""
    rows = []
    for m in metrics[:60]:
        if not isinstance(m, dict):
            continue
        label = _esc(m.get("label") or "")
        date = _esc(m.get("date") or "")
        value = _esc(m.get("value") if m.get("value") is not None else "")
        unit = _esc(m.get("unit") or "")
        d = str(m.get("direction") or "")
        arrow = {"up": '<span class="ima-up">↑</span>', "down": '<span class="ima-down">↓</span>',
                 "neutral": '<span class="ima-flat">→</span>'}.get(d, "")
        rows.append(f"<tr><td>{label}</td><td>{date}</td><td>{value}</td><td>{unit}</td><td>{arrow}</td></tr>")
    return ('<table class="ima-tbl"><thead><tr><th>指标</th><th>日期</th><th>数值</th><th>单位</th><th>方向</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>")


def _ima_flow_html(flow) -> str:
    if not isinstance(flow, dict):
        return ""
    vals = {k: flow.get(k) for k in ("inflow", "outflow", "net", "amount")}
    source = flow.get("source")
    if not any(v is not None for v in vals.values()) and not source:
        return ""
    rows = []
    for label, key in (("流入", "inflow"), ("流出", "outflow"), ("净额", "net"), ("金额", "amount")):
        v = vals.get(key)
        if v is None or v == "":
            continue
        rows.append(f"<tr><td>{label}</td><td>{_esc(v)}</td></tr>")
    fin, fout = vals.get("inflow"), vals.get("outflow")
    derived_net = None
    net_note = ""
    try:
        a, b = float(fin), float(fout)
        derived_net = a + b if b < 0 else a - b
        if vals.get("net") in (None, ""):
            rows.append(f'<tr><td>净额</td><td>{derived_net:g}（按流入/流出计算）</td></tr>')
        elif abs(float(vals["net"]) - derived_net) > 1e-9:
            net_note = "；净额与流入/流出按符号计算结果不一致，保留原始净额"
    except (TypeError, ValueError):
        pass
    unit = flow.get("unit") or flow.get("currency")
    basis = flow.get("basis") or flow.get("口径")
    rows.append(f'<tr><td>单位</td><td>{_esc(unit) if unit else "未提供"}</td></tr>')
    rows.append(f'<tr><td>口径</td><td>{_esc(basis) if basis else "未提供（不推断）"}</td></tr>')
    if source:
        rows.append(f"<tr><td>来源</td><td>{_esc(source)}</td></tr>")
    bar = ""
    fin, fout = vals.get("inflow"), vals.get("outflow")
    try:
        a, b = float(fin), float(fout)
        # 流入/流出可能已带符号，比例只用绝对值，展示值仍保留原符号。
        aa, bb = abs(a), abs(b)
        if aa + bb > 0:
            w_in = aa / (aa + bb) * 100
            w_out = 100.0 - w_in
            bar = (f'<div class="ima-flowbar"><div class="ima-fb-in" style="width:{w_in:.1f}%">流入 {_esc(fin)}</div>'
                   f'<div class="ima-fb-out" style="width:{w_out:.1f}%">流出 {_esc(fout)}</div></div>')
    except (TypeError, ValueError):
        pass
    if net_note:
        rows.append(f'<tr><td>校验</td><td>{_esc(net_note)}</td></tr>')
    return f"<b style='display:block;margin-top:6px'>资金流向</b>{bar}" + \
        f'<table class="ima-tbl"><tbody>{"".join(rows)}</tbody></table>'


def _ima_series_html(idx: str, series: list, rec: dict | None = None) -> str:
    """仅对明确可重绘记录生成 ECharts 容器。"""
    if rec and (rec.get("status") != "ok" or rec.get("renderable") is False):
        return ""
    if not isinstance(series, list) or not series:
        return ""
    valid = [s for s in series if isinstance(s, dict) and isinstance(s.get("points"), list) and s["points"]]
    if not valid:
        return ""
    payload = json.dumps(valid, ensure_ascii=False).replace("</", "<\\/")
    return (f'<b style="display:block;margin-top:6px">图表数据（可时间区间切换）</b>'
            f'<div class="ima-range" id="imaRange-{idx}">'
            '<button class="on" data-r="0">全部</button><button data-r="30">近30</button><button data-r="10">近10</button></div>'
            f'<div class="ima-chart" id="imaChart-{idx}"></div>'
            f'<script type="application/json" class="ima-series-data" data-idx="{idx}">{payload}</script>')


def _ima_card_html(idx: str, rec: dict, asset_rel: str) -> str:
    visual_type = _esc(rec.get("visual_type") or "其他")
    title = _esc((rec.get("title") or "").strip() or rec.get("relative_path", ""))
    meta = " · ".join(x for x in (_esc(rec.get("source_date") or ""), _esc(rec.get("module") or "")) if x)
    ocr = rec.get("ocr_text") or ""
    summary = rec.get("summary") or ""
    key_points = rec.get("key_points") or []
    metrics = rec.get("metrics") or []
    series = rec.get("series") or []
    flow = rec.get("flow") or {}
    entities = rec.get("entities") or []
    risks = rec.get("risks") or []
    uncertainties = rec.get("uncertainties") or []
    trend = rec.get("trend") or ""
    confidence = rec.get("confidence")

    search = " ".join([title, str(rec.get("source_date") or ""), str(rec.get("module") or ""),
                       summary, " ".join(str(x) for x in key_points), ocr[:3000]])
    img_block = (f'<a href="{asset_rel}" target="_blank" title="点击查看原图"><img class="ima-img" src="{asset_rel}" alt=""></a>'
                 if asset_rel else '<p class="ima-nodata">原图未复制（文件缺失或不可读）</p>')

    data_parts = []
    data_parts.append(f'<div class="ima-summary">{_esc(summary) or "图片未提供摘要"}</div>')
    if key_points:
        data_parts.append("<b>要点</b><ul class='ima-pts'>" + "".join(f"<li>{_esc(k)}</li>" for k in key_points[:30]) + "</ul>")
    metric_html = _ima_metric_rows(metrics)
    if metric_html:
        data_parts.append("<b>指标</b>" + metric_html)
    flow_html = _ima_flow_html(flow)
    if flow_html:
        data_parts.append(flow_html)
    series_html = _ima_series_html(idx, series, rec)
    if series_html:
        data_parts.append(series_html)
    quality_issues = rec.get("quality_issues") or []
    if rec.get("status") != "ok" or rec.get("renderable") is False or quality_issues:
        issues = quality_issues or ["记录状态不是 ok，未生成图表"]
        data_parts.append('<div class="ima-fail"><b>图片质量问题</b>：' + _esc("；".join(str(x) for x in issues)) + '</div>')
    if trend:
        data_parts.append(f"<b>趋势</b><p class='sub' style='margin-top:4px'>{_esc(trend)}</p>")
    if entities:
        data_parts.append("<b>涉及标的/行业</b><div style='margin-top:4px'>" +
                          "".join(f'<span class="ima-entity">{_esc(e)}</span>' for e in entities[:30]) + "</div>")
    if risks:
        data_parts.append("<b>风险点</b><ul class='ima-pts'>" + "".join(f"<li>⚠️ {_esc(r)}</li>" for r in risks[:15]) + "</ul>")
    if uncertainties:
        data_parts.append("<b>不确定处</b><ul class='ima-pts'>" + "".join(f"<li>❓ {_esc(u)}</li>" for u in uncertainties[:10]) + "</ul>")
    if confidence is not None and confidence != "":
        data_parts.append(f'<div class="ima-conf">识别置信度：{_esc(confidence)}%</div>')
    if not series_html and not (metric_html or flow_html):
        data_parts.append('<p class="ima-nodata">图片未提供可重绘数据，保留原图与完整识别文本。</p>')
    data_block = "".join(data_parts)

    return (f'<div class="ima-card" data-search="{_esc(search)}">'
            '<div class="ima-card-head"><span class="ima-badge">' + visual_type + "</span><b>" + title + "</b>"
            + (f'<span class="sub">{meta}</span>' if meta else "")
            + '<span class="ima-fold">▼</span></div>'
            '<div class="ima-card-body">'
            '<div class="ima-tabs"><button class="on" data-tab="img">🖼 原图</button>'
            '<button data-tab="txt">📄 识别文本</button><button data-tab="data">📊 图表化数据</button></div>'
            f'<div class="ima-tab" data-tab="img">{img_block}</div>'
            f'<div class="ima-tab" data-tab="txt" hidden><pre class="ima-ocr">{_esc(ocr) or "（无识别文本）"}</pre></div>'
            f'<div class="ima-tab" data-tab="data" hidden>{data_block}</div>'
            "</div></div>")


def render_vision_insights_section(records: list[dict], failed: list[dict] | None = None,
                                   report_dir: Path | None = None, prefix: str = "ima",
                                   max_cards: int = 40) -> str:
    """渲染 '📊 研报图表解析与资金/趋势跟踪' 章节（含样式/交互脚本）。

    report_dir: 当天报告日期目录；非空时把原图复制到 {report_dir}/assets/ 并用相对 URL 引用。
    prefix: 图表 id 前缀（多份报告同页避免冲突）。
    无任何数据时返回空字符串（不干扰现有研报渲染）。
    """
    records = [r for r in records if isinstance(r, dict)] or []
    failed = [r for r in (failed or []) if isinstance(r, dict)]
    if not records and not failed:
        return (_IMA_CSS + '<div class="card ima-sec"><h2>📊 研报图表解析与资金/趋势跟踪</h2>'
                '<p class="ima-nodata">无当日图片解析</p></div>')
    assets_dir = (report_dir / "assets") if report_dir is not None else None
    cards = []
    for i, rec in enumerate(records[:max_cards]):
        asset_rel = ""
        rel = rec.get("relative_path") or ""
        if assets_dir is not None and rel:
            src = ROOT / "data_warehouse" / "ima_export" / "media" / rel
            if src.is_file():
                asset_rel = _copy_image_asset(src, assets_dir, Path(rel).name)
        cards.append(_ima_card_html(f"{prefix}-{i}", rec, asset_rel))
    more = len(records) - max_cards if len(records) > max_cards else 0
    failed_html = ""
    if failed:
        rows = "".join(
            f'<div class="ima-fail">⚠️ 待重试 · {_esc(f.get("relative_path", ""))} — {_esc(f.get("error") or "识别失败")}</div>'
            for f in failed[:20])
        failed_html = f"<b style='display:block;margin:10px 0 4px'>识别失败/待重试（{len(failed)}）</b>" + rows
    note = f'（{len(records[:max_cards])}张）' + (f"，另有 {more} 张未展示" if more else "")
    return (_IMA_CSS
            + f'<div class="card ima-sec"><h2>📊 研报图表解析与资金/趋势跟踪{note}</h2>'
            + '<div class="ima-tools"><input class="ima-filter-input" placeholder="🔍 筛选图表卡（标题/模块/要点/OCR）">'
            + '<span class="sub" style="align-self:center">点击卡片标题可折叠展开；原图/识别文本/图表化数据切换查看；图表支持时间区间切换。</span></div>'
            + failed_html
            + "".join(cards)
            + "</div>"
            + _IMA_JS_TMPL)


def _finalize_report(html: str, date: str, report_root: Path | None = None) -> str:
    """报告收尾：卡片目录（sticky）+ 锚点 id + 报告日期 select（同根目录历史切换）。"""
    titles: list[str] = []

    def _sub(m: re.Match) -> str:
        idx = len(titles) + 1
        t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        titles.append(t[:50])
        return f'<div class="card" id="sec-{idx}"><h2>' + m.group(1) + "</h2>"

    html = re.sub(r'<div class="card"><h2>(.*?)</h2>', _sub, html, flags=re.S)
    if not titles:
        return html

    date_opts = ""
    if report_root is not None and report_root.is_dir():
        for d in sorted([x for x in report_root.iterdir()
                         if x.is_dir() and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", x.name)],
                        reverse=True):
            files = sorted(d.glob("A股*.html"))
            if d.name != date and not files:
                continue  # 非当前日期目录需已有产物；当前日期正在生成中，直接列出
            sel = ' selected' if d.name == date else ''
            if not files:
                files = [Path(f"综合详细研报_{d.name}.html")]  # 当前日期产物即将写入
            date_opts += (f'<option value="{_esc(d.name)}" data-file="../{_esc(d.name)}/{_esc(files[-1].name)}"{sel}>'
                          f'{_esc(d.name)}</option>')

    toc_links = "".join(f'<a href="#sec-{i + 1}">{_esc(t)}</a>' for i, t in enumerate(titles))
    nav = ('<nav id="tocTop" style="position:sticky;top:0;z-index:50;background:rgba(10,14,23,.96);'
           'border:1px solid var(--line,#2a3650);border-radius:10px;padding:8px 12px;margin-bottom:14px;'
           'display:flex;gap:10px;flex-wrap:wrap;align-items:center;backdrop-filter:blur(4px)">'
           '<b style="font-size:12px">📑 目录</b>'
           + (f'<select id="reportDateSel" title="切换历史报告日期">' + date_opts + "</select>" if date_opts else "")
           + '<input id="reportSearch" placeholder="🔍 搜索章节、表格与证据" style="background:#0d1420;border:1px solid var(--line,#2a3650);color:var(--tx,#e8edf4);border-radius:6px;padding:5px 8px;font-size:12px;min-width:210px"><button type="button" class="btn-sm" data-report-filter="risk">仅风险</button><button type="button" class="btn-sm" data-report-filter="decision">仅决策</button><button type="button" class="btn-sm" data-report-filter="all">显示全部</button>'
           + f'<div id="tocLinks" style="display:flex;gap:4px;flex-wrap:wrap;max-height:120px;overflow:auto">{toc_links}</div>'
           + '<script>(function(){var sel=document.getElementById("reportDateSel");if(!sel)return;'
           'sel.addEventListener("change",function(){var o=sel.options[sel.selectedIndex];'
           'if(location.protocol==="file:"){location.href=o.getAttribute("data-file");return;}'
           'var u=new URL(location.href);u.searchParams.set("date",o.value);'
           'location.href=u.pathname+"?"+u.searchParams.toString();});})();</script>'
           + "</nav>")
    report_filter_script = '''<script>(function(){var input=document.getElementById("reportSearch");function filter(kind){document.querySelectorAll(".card[id^=sec-]").forEach(function(card){var t=(card.innerText||"").toLowerCase(),q=(input&&input.value||"").trim().toLowerCase(),ok=!q||t.indexOf(q)>=0;if(kind==="risk")ok=ok&&/(风险|门控|监管|降级|回撤|止损)/.test(t);if(kind==="decision")ok=ok&&/(决策|候选|主线|执行|作战)/.test(t);card.style.display=ok?"":"none";});}if(input)input.addEventListener("input",function(){filter("all")});document.querySelectorAll("[data-report-filter]").forEach(function(button){button.addEventListener("click",function(){filter(button.getAttribute("data-report-filter"))})});})();</script>'''
    return html.replace("<body>", "<body>\n" + nav + report_filter_script, 1)



_ECharts_JS_TMPL = """<script src="/echarts.min.js"></script>
<script>
const CHAN = __CHAN_JSON__;
(function(){
  if(!document.getElementById('chanK') || !window.echarts || !CHAN.kdata || !CHAN.kdata.length) return;
  var ch = echarts.init(document.getElementById('chanK'));
  var dates = CHAN.kdata.map(function(d){return d[0];});
  var k = CHAN.kdata.map(function(d){return [d[1],d[2],d[3],d[4]];});
  var biMap = {};
  (CHAN.bi_pairs||[]).forEach(function(pr){ biMap[pr[0]]=pr[1]; });
  var lines = [];
  CHAN.kdata.forEach(function(d,i){
    var nxt = biMap[d[0]];
    if(nxt){ var j = dates.indexOf(nxt); if(j>0) lines.push({coords:[[i,CHAN.bi_prices[i]],[j,CHAN.bi_prices[j]]]}); }
  });
  // 情绪多日折线
  var emo = CHAN.emo_hist || [];
  if(emo.length && document.getElementById('emoChart')){
    var ec = echarts.init(document.getElementById('emoChart'));
    ec.setOption({
      tooltip:{trigger:'axis'},
      legend:{data:['涨停','最高板','炸板'], textStyle:{color:'#8ea0b8'}},
      xAxis:{type:'category', data:emo.map(function(e){return e.d;}), axisLabel:{color:'#8ea0b8', fontSize:10}},
      yAxis:[{type:'value', axisLabel:{color:'#8ea0b8'}},{type:'value', axisLabel:{color:'#8ea0b8'}}],
      series:[
        {name:'涨停', type:'line', data:emo.map(function(e){return e.zt;}), smooth:true, itemStyle:{color:'#f23645'}},
        {name:'最高板', type:'line', data:emo.map(function(e){return e.mb;}), smooth:true, itemStyle:{color:'#f0b429'}},
        {name:'炸板', type:'bar', yAxisIndex:1, data:emo.map(function(e){return e.zb;}), itemStyle:{color:'#8ea0b8'}}
      ]
    });
  }
  // 题材周期线
  var th = CHAN.theme_hist || [];
  if(th.length && document.getElementById('themeChart')){
    var fc2 = echarts.init(document.getElementById('themeChart'));
    fc2.setOption({
      tooltip:{trigger:'axis'}, legend:{data:['题材涨停','最高']},
      xAxis:{type:'category', data:th.map(function(x){return x.d;}), axisLabel:{color:'#8ea0b8',fontSize:10}},
      yAxis:{type:'value', axisLabel:{color:'#8ea0b8'}},
      series:[{name:'题材涨停', type:'line', data:th.map(function(x){return x.zt;}), smooth:true, itemStyle:{color:'#f23645'}},
              {name:'最高', type:'line', data:th.map(function(x){return x.mb;}), smooth:true, itemStyle:{color:'#f0b429'}}]
    });
  }
  // 门控健康度(横条: 新鲜/滞后/过期)
  var gs = CHAN.gate_stat || [];
  if(gs.length && document.getElementById('gateChart')){
    var gc = echarts.init(document.getElementById('gateChart'));
    gc.setOption({
      tooltip:{}, xAxis:{type:'category', data:['fresh','stale','expired','健康度'], axisLabel:{color:'#8ea0b8'}},
      yAxis:{type:'value', axisLabel:{color:'#8ea0b8'}},
      series:[{type:'bar', data:[{value:gs[0],itemStyle:{color:'#00b368'}},{value:gs[1],itemStyle:{color:'#e0a13c'}},
        {value:gs[2],itemStyle:{color:'#f23645'}},{value:gs[3],itemStyle:{color:'#4a90d9'}}],
        label:{show:true,position:'top',color:'#e8edf4'}}]
    });
  }
  // 沪深300近30日走势线
  var ovk = CHAN.ov_kline || [];
  if(ovk.length && document.getElementById('ovChart')){
    var oc = echarts.init(document.getElementById('ovChart'));
    oc.setOption({
      tooltip:{trigger:'axis'}, xAxis:{type:'category', data:ovk.map(function(x){return x.d;}), axisLabel:{color:'#8ea0b8',fontSize:9}},
      yAxis:{scale:true, axisLabel:{color:'#8ea0b8'}},
      series:[{type:'line', data:ovk.map(function(x){return x.c;}), smooth:true, areaStyle:{opacity:0.15}, itemStyle:{color:'#4a90d9'}}]
    });
  }
  // 财报趋势(ROE/EPS/增速)
  var ft = CHAN.ftrend || [];
  if(ft.length && document.getElementById('ftrendChart')){
    var fc5 = echarts.init(document.getElementById('ftrendChart'));
    fc5.setOption({
      tooltip:{trigger:'axis'}, legend:{data:['ROE%','EPS','增速%'], textStyle:{color:'#8ea0b8'}},
      xAxis:{type:'category', data:ft.map(function(x){return x.d;}), axisLabel:{color:'#8ea0b8',fontSize:10}},
      yAxis:[{type:'value', axisLabel:{color:'#8ea0b8'}},{type:'value', axisLabel:{color:'#8ea0b8'}}],
      series:[{name:'ROE%', type:'line', data:ft.map(function(x){return x.roe;}), smooth:true, itemStyle:{color:'#f23645'}},
              {name:'EPS', type:'line', data:ft.map(function(x){return x.eps;}), smooth:true, itemStyle:{color:'#4a90d9'}},
              {name:'增速%', type:'bar', yAxisIndex:1, data:ft.map(function(x){return x.grow;}), itemStyle:{color:'#00b368'}}]
    });
  }
  // 商品趋势(gold/copper/crude)
  var cm = CHAN.cm_trend || [];
  if(cm.length && document.getElementById('cmChart')){
    var cmc = echarts.init(document.getElementById('cmChart'));
    var names = ['gold','copper','crude'].filter(function(n){ return cm.some(function(x){return x.s==n;}); });
    cmc.setOption({
      tooltip:{trigger:'axis'},
      legend:{data:names, textStyle:{color:'#8ea0b8'}},
      xAxis:{type:'category', data:cm.filter(function(x){return x.s===(names[0]||'');}).map(function(x){return x.d;}), axisLabel:{color:'#8ea0b8',fontSize:9}},
      yAxis:{type:'value', axisLabel:{color:'#8ea0b8'}},
      series:names.map(function(n){return {name:n, type:'line', smooth:true, data:cm.filter(function(x){return x.s===n;}).map(function(x){return x.v;}), itemStyle:{color:(n==='gold'?'#f0b429':(n==='copper'?'#4a90d9':'#f23645'))}};})
    });
  }
  function initFusionFlow(){
    var node=document.getElementById('fusionFlowData'), el=document.getElementById('fusionFlowChart');
    if(!node||!el) return;
    var data={}; try{data=JSON.parse(node.textContent||'{}');}catch(e){data={};}
    var chart=window.echarts?window.echarts.init(el):null;
    var note=document.getElementById('fusionFlowDecision');
    function topNames(rows,key){
      var sums={}; (rows||[]).forEach(function(r){var v=Number(r[key]);if(Number.isFinite(v))sums[r.name]=(sums[r.name]||0)+v;});
      return Object.keys(sums).sort(function(a,b){return Math.abs(sums[b])-Math.abs(sums[a]);}).slice(0,12);
    }
    function draw(mode){
      document.querySelectorAll('[data-flow-mode]').forEach(function(b){b.classList.toggle('on',b.getAttribute('data-flow-mode')===mode);});
      var rows=[],key='',title='',unit='';
      var intraday=mode.indexOf('intraday-')===0,xkey=intraday?'time':'date';
      if(mode==='intraday-industry'){rows=data.intraday_industry||[];key='active_flow_proxy_yi';title='盘中行业方向成交代理';unit='亿元';}
      else if(mode==='intraday-etf'){rows=data.intraday_etf||[];key='active_flow_proxy_yi';title='盘中ETF方向成交代理';unit='亿元';}
      else if(mode==='intraday-market'){rows=data.intraday_market||[];key='active_flow_proxy_yi';title='盘中全市场方向成交代理';unit='亿元';}
      else if(mode==='sector'){rows=data.sector||[];key=data.sector_metric||'net_yi';title='行业/概念日频净流向';unit='亿元';}
      else if(mode==='etf-amount'){rows=data.etf||[];key='amount';title='ETF日频成交额轨迹';unit='元';}
      else if(mode==='etf-net'){rows=data.etf||[];key='main_net';title='ETF日频主力净流入';unit='元';}
      else if(mode==='etf-share'){rows=data.etf||[];key='shares_delta';title='ETF份额变化';unit='份';}
      else {rows=data.market||[];key='force_index';title='市场日频资金合力';unit='分';}
      if(!chart){el.innerHTML='<div class="sub">ECharts未加载，保留上方结构化表格。</div>';return;}
      var names=topNames(rows,key),dates=[];
      rows.forEach(function(r){if(dates.indexOf(r[xkey])<0)dates.push(r[xkey]);});dates.sort();
      var series=names.map(function(name,i){var by={};rows.filter(function(r){return r.name===name;}).forEach(function(r){by[r[xkey]]=Number(r[key]);});return{name:name,type:'line',smooth:true,connectNulls:true,showSymbol:false,data:dates.map(function(d){return Number.isFinite(by[d])?by[d]:null;})};});
      chart.setOption({title:{text:title,textStyle:{color:'#e8edf4',fontSize:14}},tooltip:{trigger:'axis'},legend:{type:'scroll',top:24,textStyle:{color:'#8ea0b8'}},grid:{left:70,right:25,top:70,bottom:75},xAxis:{type:'category',data:dates,axisLabel:{color:'#8ea0b8'}},yAxis:{type:'value',name:unit,axisLabel:{color:'#8ea0b8'}},dataZoom:[{type:'inside'},{type:'slider',bottom:15}],series:series},true);
      var latest=dates[dates.length-1],prev=dates[dates.length-2],latestRows=rows.filter(function(r){return r[xkey]===latest&&Number.isFinite(Number(r[key]));}).sort(function(a,b){return Number(b[key])-Number(a[key]);});
      var lead=latestRows.slice(0,3).map(function(r){return r.name+' '+Number(r[key]).toFixed(2);}).join('、');
      var weak=latestRows.slice(-3).reverse().map(function(r){return r.name+' '+Number(r[key]).toFixed(2);}).join('、');
      var persistent=names.filter(function(n){var vals=rows.filter(function(r){return r.name===n&&dates.slice(-3).indexOf(r[xkey])>=0;}).map(function(r){return Number(r[key]);});return vals.length>=2&&vals.every(function(v){return v>0;});}).slice(0,5);
      if(note)note.innerHTML='<b>'+(intraday?'盘中代理解读：':'日频资金解读：')+'</b>截至 '+(latest||'-')+'，领先 '+(lead||'无')+'；弱侧 '+(weak||'无')+'。'+(persistent.length?'连续三轮正向：'+persistent.join('、')+'，优先观察承接与成交持续性。':'未形成稳定连续正向，按轮动处理，避免追逐单点峰值。')+(prev?' 对比前一观测 '+prev+' 判断加速或退潮。':'')+(intraday?' 盘中值是成交增量乘涨跌方向的代理，不是真实主力净流入。':'');
    }
    document.querySelectorAll('[data-flow-mode]').forEach(function(b){b.addEventListener('click',function(){draw(b.getAttribute('data-flow-mode'));});});
    draw((data.intraday_industry||[]).length?'intraday-industry':'sector'); window.addEventListener('resize',function(){if(chart)chart.resize();});
  }
  var fh = CHAN.fund_hist || [];
  if(fh.length && document.getElementById('fundChart')){
    var fc = echarts.init(document.getElementById('fundChart'));
    fc.setOption({
      tooltip:{trigger:'axis'},
      legend:{data:['游资(亿)','机构(亿)','合力'], textStyle:{color:'#8ea0b8'}},
      xAxis:{type:'category', data:fh.map(function(f){return f.d;}), axisLabel:{color:'#8ea0b8', fontSize:10}},
      yAxis:[{type:'value', axisLabel:{color:'#8ea0b8'}},{type:'value', axisLabel:{color:'#8ea0b8'}}],
      series:[
        {name:'游资(亿)', type:'bar', data:fh.map(function(f){return f.yz;}), itemStyle:{color:'#f23645'}},
        {name:'机构(亿)', type:'bar', data:fh.map(function(f){return f.jg;}), itemStyle:{color:'#4a90d9'}},
        {name:'合力', type:'line', yAxisIndex:1, data:fh.map(function(f){return f.fi;}), smooth:true, itemStyle:{color:'#f0b429'}}
      ]
    });
  }
  initFusionFlow();
  ch.setOption({
    tooltip:{trigger:'axis'},
    xAxis:{type:'category', data:dates, axisLabel:{color:'#8ea0b8', fontSize:10}},
    yAxis:{scale:true, axisLabel:{color:'#8ea0b8'}},
    dataZoom:[{type:'inside'}],
    series:[
      {type:'candlestick', data:k, itemStyle:{color:'#f23645',color0:'#00b368',borderColor:'#f23645',borderColor0:'#00b368'}},
      {type:'lines', data:lines, silent:true, lineStyle:{color:'#f0b429',width:1.5,type:'dashed'}, effect:{show:false}},
      {type:'line', data:CHAN.bi_prices, showSymbol:false, silent:true, lineStyle:{opacity:0}}
    ]
  });
})();
</script>"""


def _stage_color(stage):
    if stage in ("发酵", "复苏", "高潮"): return "#f23645"
    if stage in ("退潮", "恐慌", "冰点"): return "#4a90d9"
    return "#e0a13c"


def _tbl(headers, rows):
    if not rows: return "<tr><td colspan=9>-</td></tr>"
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div style="overflow-x:auto"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _bar(v, max_v=100, color="#4aa3ff"):
    pct = max(2, min(100, (v / max_v) * 100)) if max_v else 2
    return f'<div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{color}"></div></div>'


def _as_of_frame(frame, date: str, date_column: str = "date"):
    """Return only observations on or before the report date, never future rows."""
    try:
        import pandas as pd
        out = frame.copy()
        if date_column not in out.columns:
            return out.iloc[0:0]
        out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
        cutoff = pd.Timestamp(date)
        return out[out[date_column].notna() & (out[date_column] <= cutoff)].sort_values(date_column)
    except Exception:
        return frame.iloc[0:0]


def _load_decision_snapshot(date: str) -> dict:
    path = ROOT / "generated" / f"decision_snapshot_after_close_{date}.json"
    value = _load_json_file(path)
    if value.get("as_of") != date:
        return {"_error": f"统一决策快照日期不匹配：{value.get('as_of') or 'missing'} != {date}"}
    return value


def _render_decision_snapshot(snapshot: dict) -> str:
    if snapshot.get("_error"):
        return f'<div class="degraded-alert">{_esc(snapshot["_error"])}</div>'
    counts, market = snapshot.get("counts") or {}, snapshot.get("market") or {}
    rows = []
    detail_cards = []
    for item in snapshot.get("short_term_candidates") or []:
        reasons = "；".join(str(x) for x in (item.get("short_term_reasons") or [])) or "关键短线证据不足"
        blockers = "；".join(str(x) for x in (item.get("short_term_blockers") or [])) or "无硬阻断"
        risk = item.get("regulatory_risk") or {}
        strategy = item.get("execution_strategy") or {}
        rows.append([
            _esc(item.get("code") or "-"), _esc(item.get("name") or "-"), _esc(item.get("industry") or "未映射"),
            _esc(item.get("trade_label") or "观察"), _esc(item.get("short_term_score") or "-"),
            _esc(item.get("mainline_match") or "未匹配"), _esc((item.get("flow_match") or {}).get("name") or "无"),
            _esc(reasons),
        ])
        detail_cards.append(
            f'<div class="decision-note"><b>{_esc(item.get("name") or item.get("code"))} ({_esc(item.get("code"))}) · '
            f'{_esc(item.get("short_term_score"))}分 · {_esc(item.get("trade_label"))}</b>'
            f'<p>核心理由：{_esc(reasons)}</p><p>限制：{_esc(blockers)}；监管硬风险{_esc(risk.get("tier1", 0))}条。</p>'
            f'<p><b>入场：</b>{_esc(strategy.get("entry") or "-")}<br><b>加仓：</b>{_esc(strategy.get("add") or "-")}<br>'
            f'<b>失效/退出：</b>{_esc(strategy.get("stop") or "-")}<br><b>避免：</b>{_esc(strategy.get("avoid") or "-")}</p></div>'
        )
    flags = "".join(f"<li>{_esc(x)}</li>" for x in (market.get("risk_flags") or [])) or "<li>无额外市场风险标记</li>"
    summary = (f'<p><b>统一决策日期：</b>{_esc(snapshot.get("as_of") or "-")} · '
               f'<b>全市场短线扫描：</b>{_esc(counts.get("scanned", 0))} · <b>异动候选：</b>{_esc(counts.get("candidates", 0))} · '
               f'<b>精选主板：</b>{_esc(counts.get("short_term_selected", 0))} · '
               f'<b>盘口触发后可试探：</b>{_esc(counts.get("execution_candidates", 0))} · '
               f'<b>监管拦截：</b>{_esc(counts.get("hard_risk_candidates", 0))}</p>')
    return summary + '<ul>' + flags + '</ul>' + _tbl(
        ["代码", "名称", "行业", "状态", "短线分", "主线", "资金匹配", "核心理由"], rows
    ) + ''.join(detail_cards) + '<p class="decision-note"><b>权威口径：</b>全市场用于判断短线主线、行业/概念资金和梯队；最终仅输出5-10只主板高分候选，宁缺毋滥。高分不等于立即买入，必须满足每只候选的盘口触发条件。</p>'


def _render_decision_audit(snapshot: dict) -> str:
    """Render auditable coverage, candidate, and risk evidence without filler."""
    if snapshot.get("_error"):
        return ""
    coverage = snapshot.get("intraday_coverage") or {}
    source_dates = snapshot.get("source_dates") or {}
    mismatches = snapshot.get("date_mismatches") or {}
    source_rows = [
        [_esc(name), _esc("可用" if enabled else "缺失"), _esc(source_dates.get(name) or "无日期"),
         _esc("日期不匹配" if name in mismatches else "-")]
        for name, enabled in (snapshot.get("source_chain") or {}).items()
    ]
    coverage_rows = [[
        _esc(coverage.get("status") or "missing"),
        _esc(coverage.get("snapshot_rows") or 0),
        _esc(coverage.get("signal_evaluated") or 0),
        _esc(coverage.get("candidate_count") or 0),
        _esc(coverage.get("evidence_evaluated") or 0),
        _esc(coverage.get("ratio") if coverage.get("ratio") is not None else "-"),
    ]]
    alert = ""
    if coverage.get("status") != "complete":
        alert = (f'<div class="degraded-alert">盘中覆盖降级：{_esc(coverage.get("status") or "missing")}。'
                 '该日期候选仅用于审计与观察，不形成可执行结论。</div>')

    mainline_rows = [[
        _esc(row.get("industry") or row.get("name") or "-"), _esc(row.get("source") or "-"),
        _esc(row.get("score") or 0), _esc(row.get("zt") or row.get("zt_cnt") or 0),
        _esc(row.get("max_board") or 0), _esc(row.get("net_yi") if row.get("net_yi") is not None else "-"),
    ] for row in (snapshot.get("mainlines") or [])]

    opportunity_rows = []
    for item in (snapshot.get("opportunities") or [])[:100]:
        flow = item.get("flow_match") or {}
        research = item.get("research_evidence") or {}
        risk = item.get("regulatory_risk") or {}
        reasons = "；".join(str(x) for x in (item.get("short_term_reasons") or [])) or "无充分加分证据"
        blockers = "；".join(str(x) for x in (item.get("short_term_blockers") or [])) or "无"
        opportunity_rows.append([
            _esc(item.get("code") or "-"), _esc(item.get("name") or "-"), _esc(item.get("board") or "-"),
            _esc(item.get("industry") or "未映射"), _esc(item.get("change_pct") if item.get("change_pct") is not None else "-"),
            _esc(_fmt_num(item.get("amount"))), _esc(item.get("short_term_score") or 0),
            _esc(item.get("mainline_match") or "未匹配"), _esc(flow.get("name") or "无"),
            _esc(_fmt_num(flow.get("net_yi"))), _esc(research.get("reports") or 0),
            _esc(risk.get("tier1") or 0), _esc(item.get("trade_label") or "观察"),
            _esc(reasons), _esc(blockers),
        ])

    evidence = snapshot.get("evidence") or {}
    holders = (evidence.get("holders") or {}).get("items") or []
    holder_rows = [[
        _esc(x.get("code") or "-"), _esc(x.get("name") or "-"), _esc(x.get("as_of") or "-"),
        _esc(x.get("holder_count") or "-"), _esc(x.get("change_pct") if x.get("change_pct") is not None else "-"),
        _esc(x.get("age_days") if x.get("age_days") is not None else "-"), _esc(x.get("signal") or "-"),
        _esc(x.get("source") or "-"),
    ] for x in holders]

    regulatory = evidence.get("regulatory") or {}
    regulatory_cards = []
    for item in (regulatory.get("top_codes") or [])[:20]:
        recent_rows = [[
            _esc(event.get("date") or "-"), _esc(event.get("type") or "-"), _esc(event.get("tier") or "-"),
            _esc(event.get("title") or "-"),
        ] for event in (item.get("recent") or [])]
        regulatory_cards.append(
            f'<div class="decision-note"><b>{_esc(item.get("name") or "-")} ({_esc(item.get("code") or "-")}) · '
            f'累计{_esc(item.get("count") or 0)} · 硬风险{_esc(item.get("tier1") or 0)} · 最新{_esc(item.get("latest") or "-")}</b>'
            + _tbl(["日期", "类型", "层级", "公告"], recent_rows) + "</div>"
        )

    factor = evidence.get("factor_decay") or {}
    factor_rows = [[
        _esc(x.get("factor") or x.get("name") or "-"), _esc(x.get("status") or "-"),
        _esc(x.get("ic") if x.get("ic") is not None else x.get("value", "-")), _esc(x.get("note") or "-"),
    ] for x in (factor.get("factors") or [])[:50]]

    return (
        alert
        + '<h3>扫描覆盖与来源日期</h3>'
        + _tbl(["状态", "快照行", "信号评估", "候选", "证据评估", "覆盖率"], coverage_rows)
        + _tbl(["来源", "可用性", "实际日期", "日期契约"], source_rows)
        + '<h3>主线证据排行</h3>'
        + _tbl(["方向", "来源", "分数", "涨停数", "最高板", "资金净额"], mainline_rows)
        + f'<h3>全市场候选审计（展示{len(opportunity_rows)}/{len(snapshot.get("opportunities") or [])}）</h3>'
        + _tbl(["代码", "名称", "板块", "行业", "涨幅%", "成交额", "短线分", "主线", "资金方向", "资金净额",
                "研报数", "硬风险", "状态", "加分证据", "阻断条件"], opportunity_rows)
        + '<h3>股东户数与筹码证据</h3>'
        + _tbl(["代码", "名称", "截至", "户数", "变化%", "滞后天", "信号", "来源"], holder_rows)
        + f'<h3>监管风险索引（{_esc(regulatory.get("total_hits") or 0)}条/{_esc(regulatory.get("codes") or 0)}只）</h3>'
        + "".join(regulatory_cards)
        + '<h3>盘中因子窗口</h3>'
        + f'<p class="sub">状态 {_esc(factor.get("status") or "unavailable")} · 截至 {_esc(factor.get("as_of") or "-")} · '
          f'窗口 {_esc(factor.get("n_windows") or 0)} · 原因 {_esc(factor.get("reason") or "-")}</p>'
        + _tbl(["因子", "状态", "数值", "说明"], factor_rows)
    )


def _load_research_fusion_snapshot(date=None) -> dict:
    """Build from local warehouse data; preserve each source's real date."""
    try:
        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from research_fusion_snapshot import build_snapshot
        value = build_snapshot(date=date)
        if not isinstance(value, dict):
            return {}
        intraday_path = ROOT / "generated" / f"intraday_flow_history_{date}.json"
        value["intraday_flow"] = _load_json_file(intraday_path) if intraday_path.exists() else {
            "status": "unavailable", "date": date, "reason": "intraday flow history missing"
        }
        return value
    except Exception as exc:  # optional section must not abort the full report
        return {"_error": str(exc)[:300]}


def _render_research_fusion_snapshot(snapshot: dict) -> str:
    if snapshot.get("_error"):
        return f'<p class="sub">融合快照不可用：{_esc(snapshot["_error"])}</p>'

    def name(item):
        return item.get("name") or item.get("concept_name") or item.get("sector_name") or item.get("board_name") or item.get("code") or item.get("board_code") or "-"

    def value(item, keys):
        return next((item.get(key) for key in keys if item.get(key) is not None), None)

    def top(title, items, keys, label):
        rows = [[_esc(name(item)), _esc(item.get("code") or item.get("board_code") or "-"), _esc(_fmt_num(value(item, keys), 2))]
                for item in (items or [])[:8] if isinstance(item, dict)]
        return f'<b style="display:block;margin-top:10px">{_esc(title)}</b>' + _tbl(["标的", "代码", label], rows)

    status = snapshot.get("data_status") or {}
    market = snapshot.get("market_summary") or {}
    money, etf = snapshot.get("money_flow") or {}, snapshot.get("etf_activity") or {}
    sector, concept = snapshot.get("sector_rotation") or {}, snapshot.get("concept_rotation") or {}
    detail = market.get("force_detail") or {}
    intraday = snapshot.get("intraday_flow") or {}
    intraday_etf = intraday.get("etf") or {}
    sources = [[_esc(key), _esc(meta.get("status") or "-"), _esc(meta.get("as_of") or "无日期"), _esc(meta.get("source") or "-")]
               for key, meta in (status.get("sources") or {}).items()]
    html = (f'<p><b>快照数据截至：</b>{_esc(snapshot.get("as_of") or "无可用日期")} · <b>请求截止：</b>{_esc(snapshot.get("requested_date") or "-")} · <b>来源状态：</b>{_esc(status.get("status") or "unknown")} ({_esc(status.get("available_sources", 0))}/{_esc(status.get("total_sources", 0))})</p>'
            f'<p class="sub">市场温度 {_esc(market.get("temperature") if market.get("temperature") is not None else "-")} ({_esc(market.get("temperature_tag") or "-")}) · 情绪 {_esc(market.get("emotion") or "-")} · 资金合力 {_esc(market.get("force_index") if market.get("force_index") is not None else "-")} · 游资 {_esc(_fmt_num(detail.get("youzi_net")))} · 机构 {_esc(_fmt_num(detail.get("jg_net")))}</p>')
    html += _tbl(["来源", "状态", "实际日期", "本地文件"], sources)
    html += top("资金流入 Top", money.get("top_in"), ("net_yi", "net", "net_amount", "main_net"), "净流向")
    html += top("资金流出 Top", money.get("top_out"), ("net_yi", "net", "net_amount", "main_net"), "净流向")
    html += top("ETF 成交额 Top", etf.get("top_amount"), ("amount",), "成交额")
    html += top("ETF 主力净额 Top", etf.get("top_main_net"), ("main_net",), "主力净额")
    html += top("ETF 份额变化 Top", etf.get("share_delta"), ("shares_delta",), "份额变化")
    html += top("行业轮动 Top", sector.get("top_up"), (sector.get("metric") or "net_yi", "net_yi", "net"), "净流向")
    def compact_series(rows, key, limit=12):
        totals = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            try:
                totals[str(row.get("name") or "-")] = totals.get(str(row.get("name") or "-"), 0.0) + abs(float(row.get(key) or 0))
            except (TypeError, ValueError):
                continue
        keep = {name for name, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]}
        return [row for row in (rows or []) if str(row.get("name") or "-") in keep]

    intraday_industry = compact_series(intraday.get("industry") or [], "active_flow_proxy_yi", limit=8)
    intraday_etf_rows = compact_series(intraday_etf.get("rows") or [], "amount_delta_yi", limit=8)
    chart_payload = {
        "sector": sector.get("history") or [],
        "etf": etf.get("history") or [],
        "market": market.get("history") or [],
        "sector_metric": sector.get("metric") or "net_yi",
        "intraday_market": intraday.get("market") or [],
        "intraday_industry": intraday_industry,
        "intraday_etf": intraday_etf_rows,
    }
    payload = json.dumps(chart_payload, ensure_ascii=False).replace("</", "<\\/")
    html += '''
    <div class="fusion-chart-tools">
      <button type="button" data-flow-mode="intraday-industry" class="on">盘中行业代理</button>
      <button type="button" data-flow-mode="intraday-etf">盘中ETF代理</button>
      <button type="button" data-flow-mode="intraday-market">盘中全市场代理</button>
      <button type="button" data-flow-mode="sector">日频行业净流向</button>
      <button type="button" data-flow-mode="etf-amount">日频ETF成交额</button>
      <button type="button" data-flow-mode="etf-net">日频ETF主力净额</button>
      <button type="button" data-flow-mode="market">日频市场合力</button>
    </div>
    <div id="fusionFlowChart" style="width:100%;height:380px"></div>
    <div id="fusionFlowDecision" class="decision-note"></div>
    '''
    html += f'<script type="application/json" id="fusionFlowData">{payload}</script>'
    periods = intraday.get("periods") or {}
    html += (f'<p class="sub"><b>盘中分时状态：</b>{_esc(intraday.get("status") or "unavailable")} · '
             f'快照{_esc(intraday.get("snapshot_count") or 0)}个 · 开盘{_esc(periods.get("open", False))} · '
             f'上午{_esc(periods.get("morning", False))} · 午后{_esc(periods.get("afternoon", False))} · '
             f'尾盘{_esc(periods.get("close", False))} · ETF {_esc(intraday_etf.get("status") or "unavailable")}'
             f'{("（" + _esc(intraday_etf.get("reason")) + "）") if intraday_etf.get("reason") else ""}</p>')
    return html + '<p class="sub" style="margin-top:10px">日频图展示真实资金字段；盘中行业/ETF/市场图展示“成交额增量×涨跌方向”代理，只用于比较持续性和轮动，不冒充真实主力净流入。数据缺失留空，不补零、不改标日期。</p>'


def _block_alert(blocks: dict, key: str, label: str) -> str:
    """Render degraded/error state visibly instead of silently showing an empty table."""
    raw = blocks.get(key) or {}
    err = getattr(raw, "error", None) if not isinstance(raw, dict) else raw.get("error")
    conf = getattr(raw, "confidence", None) if not isinstance(raw, dict) else raw.get("confidence")
    if isinstance(raw, dict):
        value = raw.get("value") or {}
        err = err or value.get("error")
    if err or (conf is not None and float(conf or 0) < 0.5):
        reason = escape(str(err or f"置信度 {conf}"))
        return f'<div class="degraded-alert">⚠️ {escape(label)}数据降级：{reason}。本节不作为完整决策证据。</div>'
    return ""


def build_html(review, report_dir=None):
    """渲染完整研报 HTML。report_dir: 当天日期目录（图片资产复制到 {report_dir}/assets）。"""
    date = review.get("date", "")
    blocks = review.get("blocks", {})
    dec = review.get("decision", {})
    fz = dec.get("fusion", {})

    def blk(k):
        b = blocks.get(k) or {}
        return b.get("value", {}) if isinstance(b.get("value"), dict) else {}

    bm = review.get("battle_map") or {}

    # 互联舆情(股吧/微博/百度)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from social_pulse import sentiment_md as _social_md_fn
        _social_md = _social_md_fn()
    except Exception:
        _social_md = ""
    # 缠论笔段自动生成(K线分型→笔→中枢)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from chan_skill import chan_bigscan, chan_weekly, chan_all_indices
        _chan_big = chan_bigscan("沪深300", days=180)
        _chan_wk = chan_weekly("沪深300")
        _chan_all = chan_all_indices()
    except Exception as _ce:
        _chan_big = {}
    # 资金多日历史(游资/机构近8日, 画柱状)
    _fund_hist = []
    try:
        import pandas as _pd2
        _ff = _pd2.read_parquet(ROOT / "data_warehouse" / "market" / "fund_forces.parquet")
        _ff = _as_of_frame(_ff, date)
        for _, r in _ff.tail(8).iterrows():
            _fund_hist.append({"d": str(r.get("date",""))[:10][5:], "yz": round(float(r.get("youzi_net",0) or 0)/1e8,1),
                               "jg": round(float(r.get("jg_net",0) or 0)/1e8,1),
                               "fi": round(float(r.get("force_index",0) or 0))})
    except Exception: _fund_hist = []
    # 情绪多日数据(画折线: 涨停/炸板率/温度, 近10日)
    _emo_hist = []
    try:
        import glob as _g2
        review_files = []
        for rf in _g2.glob(str(ROOT / "generated" / "review_*.json")):
            match = re.search(r"review_(20\d{2}-\d{2}-\d{2})\.json$", rf)
            if match and match.group(1) <= date:
                review_files.append(rf)
        for rf in sorted(review_files)[-10:]:
            try:
                rd = json.loads(Path(rf).read_text(encoding="utf-8"))
                rb = rd.get("blocks", {})
                rt = (rb.get("market_temperature") or {}).get("value", {}).get("components", {}).get("emotion", {})
                rl = (rb.get("leader_sentiment") or {}).get("value", {}).get("components", {}).get("ladder", {})
                _emo_hist.append({"d": str(rd.get("date",""))[-5:], "zt": int(rt.get("zt_cnt",0) or 0),
                                  "mb": int(rl.get("max_board",0) or 0), "zb": int(rl.get("zb_cnt",0) or 0)})
            except Exception: continue
    except Exception: _emo_hist = []
    # 题材周期近8日(画轮动线)
    _theme_hist = []
    try:
        import pandas as _pd3
        _tc = _pd3.read_parquet(ROOT / "data_warehouse" / "market" / "theme_cycle.parquet")
        _tc = _as_of_frame(_tc, date)
        for _, r in _tc.tail(8).iterrows():
            _theme_hist.append({"d": str(r.get("date",""))[:10][5:], "zt": int(r.get("zt_cnt",0) or 0),
                                "mb": int(r.get("max_board",0) or 0)})
    except Exception: _theme_hist = []
    # 海外指数走势(纳指近30日, 画线)
    _ov_kline = []
    try:
        import pandas as _pd4
        _kf = _pd4.read_parquet(ROOT / "data_warehouse" / "market" / "index_daily_沪深300.parquet")
        _kf = _as_of_frame(_kf, date)
        for _, r in _kf.tail(30).iterrows():
            _ov_kline.append({"d": str(r.get("date",""))[:10][5:], "c": round(float(r.get("close",0)),1)})
    except Exception: _ov_kline = []
    # 沪深300 K线近60日(画图)
    try:
        import pandas as pd
        _kdf = pd.read_parquet(ROOT / "data_warehouse" / "market" / "index_daily_沪深300.parquet")
        _kdf = _as_of_frame(_kdf, date)
        _k60 = _kdf.tail(60)
        _kdata = [[str(d)[:10], round(float(o), 1), round(float(c), 1), round(float(l), 1), round(float(h), 1)]
                  for d, o, c, l, h in zip(_k60["date"], _k60["open"], _k60["close"], _k60["low"], _k60["high"])]
    except Exception:
        _kdata = []
    # 引擎真实产出(工厂账本优先; 兜底 deep cache) — 关键内容展示
    _deepeng = {}
    _ledger = {}
    try:
        import glob as _gL
        _lds = sorted(_gL.glob(str(ROOT / "generated" / "engine_ledger_*.json")))
        if _lds:
            import time as _tL
            _LDP = Path(_lds[-1])
            if _tL.time() - _LDP.stat().st_mtime < 21600:
                _ledger = json.loads(_LDP.read_text(encoding="utf-8"))
    except Exception: _ledger = {}
    try:
        _DEP = ROOT / "generated" / "deep_engines_cache.json"
        if _DEP.exists():
            import time as _tD
            if _tD.time() - _DEP.stat().st_mtime < 14400:
                _deepeng = json.loads(_DEP.read_text(encoding="utf-8"))
    except Exception: _deepeng = {}
    # 全引擎(83可用) — 读全量缓存
    _alleng = {}
    try:
        _AEP = ROOT / "generated" / "all_engines_cache.json"
        if _AEP.exists():
            import time as _t8
            if _t8.time() - _AEP.stat().st_mtime < 14400:
                _alleng = json.loads(_AEP.read_text(encoding="utf-8"))
    except Exception: _alleng = {}
    # 六引擎(海外宏观/RS/日历/透镜) — 产物优先, 无则实时(约80s)
    _eng = {}
    try:
        _ep = ROOT / "generated" / "engine_pulse_cache.json"
        if _ep.exists():
            import time as _t7
            if _t7.time() - _ep.stat().st_mtime < 172800:
                _eng = json.loads(_ep.read_text(encoding="utf-8"))
    except Exception: _eng = {}
    # 财报深度(ROE/增速/负债/每股收益)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from fundamental_pulse import fundamental_data, stock_picker, valuation_pulse, announcement_calendar, trend_data
        _fund = fundamental_data()
        _picker = stock_picker()
        _val = valuation_pulse()
        _ann = announcement_calendar(10)
        _ftrend = trend_data("000001", 5)
    except Exception:
        _fund, _picker, _val, _ann, _ftrend = {}, [], [], [], []
    # 题材轮动(theme_cycle 近10日最热题材 zt数)
    _theme_rot = []
    try:
        import pandas as _pd5
        _tc5 = _pd5.read_parquet(ROOT / "data_warehouse" / "market" / "theme_cycle.parquet")
        _tc5 = _as_of_frame(_tc5, date)
        _last_d = _tc5["date"].max()
        _recent = _tc5[_tc5["date"] >= _last_d - _pd5.Timedelta(days=12)]
        _agg = _recent.groupby("concept").agg(zt=("zt_cnt", "sum"), maxd=("max_board", "max")).sort_values(["maxd", "zt"], ascending=False)
        _theme_rot = [{"n": str(c), "zt": int(zt), "mb": int(md)}
                      for c, (zt, md) in zip(_agg.index[:6], _agg.itertuples(index=False))] if len(_agg) else []
    except Exception: _theme_rot = []
    # 新基座: 龙虎榜历史近3日 涨停历史近7日
    _lhb_h, _zt_h = [], []
    try:
        import pandas as _pdH, glob as _gH
        lhb_files = [path for path in _gH.glob(str(ROOT / "data_warehouse" / "lhb_hist" / "*.parquet"))
                     if (match := re.search(r"(20\d{8})", Path(path).name)) and f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}" <= date]
        for _f in sorted(lhb_files)[-3:]:
            try:
                _df = _pdH.read_parquet(_f)
                _nm = "代码" if "代码" in _df.columns else _df.columns[0]
                for _, r in _df.head(3).iterrows():
                    _lhb_h.append({"d": _f.split("_")[-1].replace(".parquet",""), "n": str(r.get(_nm,""))[:6]})
            except Exception: pass
        for _f in sorted(_gH.glob(str(ROOT / "data_warehouse" / "zt_history" / "*.parquet")))[-7:]:
            try:
                _df = _pdH.read_parquet(_f)
                _zt_h.append({"d": _f.split("_")[-1].replace(".parquet","")[4:], "n": int(len(_df))})
            except Exception: pass
    except Exception: pass
    # ML全池扫描(强势/回避榜)
    _mlscan = {}
    try:
        _MSC = ROOT / "generated" / "ml_scan_cache.json"
        if _MSC.exists():
            import time as _t9
            if _t9.time() - _MSC.stat().st_mtime < 172800:
                _mlscan = json.loads(_MSC.read_text(encoding="utf-8"))
    except Exception: _mlscan = {}
    # 决策预测验证(近10日复盘趋势一致性)
    _pred_chk = []
    try:
        import glob as _g9
        _hist9 = sorted(_g9.glob(str(ROOT / "generated" / "review_*.json")))[-10:]
        for _hf in _hist9:
            try:
                _rd = json.loads(Path(_hf).read_text(encoding="utf-8"))
                rb = _rd.get("blocks", {})
                rt = (rb.get("market_temperature") or {}).get("value", {}).get("components", {}).get("emotion", {})
                rl = (rb.get("leader_sentiment") or {}).get("value", {}).get("components", {}).get("ladder", {})
                _pred_chk.append({"d": str(_rd.get("date",""))[-5:], "stage": rt.get("stage_cn","?"),
                                  "zt": int(rt.get("zt_cnt",0) or 0), "mb": int(rl.get("max_board",0) or 0)})
            except Exception: continue
        if len(_pred_chk) < 3:
            # 复盘历史不足 → 数据仓库真实情绪周期历史兜底（同口径）
            import pandas as _pdP
            sys.path.insert(0, str(ROOT / "quant_system"))
            from quant_system.analysis_core import emotion_cycle as _ec9
            _eh = _ec9.run_history()
            if hasattr(_eh, "tail"):
                for _, _er in _eh.tail(6).iterrows():
                    _pred_chk.append({"d": str(_er.get("date",""))[-5:],
                                      "stage": str(_er.get("stage_cn") or "?"),
                                      "zt": int(_er.get("zt_cnt",0) or 0),
                                      "mb": int(_er.get("max_board",0) or 0)})
    except Exception: _pred_chk = []
    # 商品趋势(commodity__gold/copper 近30日画线)
    _cm_trend = []
    try:
        import pandas as _pdC
        for _cn in ("gold", "copper", "crude"):
            _cf2p = ROOT / "data_warehouse" / "market" / f"commodity__{_cn}.parquet"
            if _cf2p.exists():
                _cdf = _pdC.read_parquet(_cf2p)
                date_col = next((col for col in ("date", "日期", "trade_date") if col in _cdf.columns), None)
                if date_col:
                    _cdf = _as_of_frame(_cdf, date, date_col)
                _cdf = _cdf.tail(25)
                for _, r in _cdf.iterrows():
                    try:
                        _cm_trend.append({"s": _cn, "d": str(r.iloc[0])[:10][5:], "v": round(float(r.iloc[-1]), 2)})
                    except Exception: pass
    except Exception: _cm_trend = []
    # 概念资金流(sector_fund_flow 最新日 Top/Flop)
    _cf_top, _cf_flop = [], []
    try:
        import pandas as _pdF
        _sff = _pdF.read_parquet(ROOT / "data_warehouse" / "market" / "sector_fund_flow.parquet")
        _sff = _as_of_frame(_sff, date)
        _lastd = _sff["date"].max()
        _day = _sff[_sff["date"] == _lastd].sort_values("net_yi", ascending=False)
        _cf_top = [{"n": str(r["concept_name"]), "v": round(float(r["net_yi"]), 1)} for r in _day.head(8).to_dict("records")]
        _cf_flop = [{"n": str(r["concept_name"]), "v": round(float(r["net_yi"]), 1)} for r in _day.tail(8).to_dict("records")][::-1]
    except Exception: pass
    # 全池回测策略(动量/三策略/合流选股)
    _btp = {}
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from backtest_pool import backtest_data
        _btp = backtest_data()
    except Exception: _btp = {}
    # 市场基座全景(商品/利率/宏观/资金/两融/涨停)
    _mpano = {}
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from market_pano import market_pano as _mpfn
        _mpano = _mpfn()
    except Exception: _mpano = {}
    # 缠论分析(epub RAG + 结构判定)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from chan_skill import chan_report_md
        _chan_md = chan_report_md()
        _chan_ok = True
    except Exception:
        _chan_md, _chan_ok = "", False
    # 波浪理论大盘结构(对应缠论三剧本)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from elliot_wave import wave_structure, wave_all_indices
        _wave = wave_structure()
        _wave_all = wave_all_indices()
    except Exception:
        _wave = {}
    # IMA 知识库研报检索(热点主题 → 研报标题)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from ima_news_pulse import research_data
        _ima_res = research_data(per_topic=3)
    except Exception:
        _ima_res = {}
    # 深度复盘增强(涨停分级/核心情绪/基准/关键词)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from depth_review import depth_block
        _depth = depth_block()
    except Exception:
        _depth = {}

    temp = blk("market_temperature")
    emo = (temp.get("components") or {}).get("emotion", {})
    fund = (temp.get("components") or {}).get("fund", {})
    macro = (temp.get("components") or {}).get("macro", {})
    _strong_alert = _block_alert(blocks, "strong_direction", "强势方向")
    _overseas_alert = _block_alert(blocks, "overseas", "海外市场")
    sector = blk("strong_direction").get("directions", [])
    leader = blk("leader_sentiment")
    lad = (leader.get("components") or {}).get("ladder", {})
    lhb = blk("lhb").get("lhb", {})
    pools = blk("stock_picks").get("pools", {})
    factor = blk("factor_signal")
    fg = factor.get("groups", {})
    factor_quality_note = "-"
    try:
        fq_path = ROOT / "generated" / "factor_quality_registry.json"
        if fq_path.exists():
            _fq = json.loads(fq_path.read_text(encoding="utf-8"))
            _cnt = _fq.get("counts", {})
            factor_quality_note = (f"可用于决策 {_cnt.get('eligible', _cnt.get('core', 0) + _cnt.get('candidate', 0))} 个；"
                                   f"观察 {_cnt.get('observe',0)} 个；阻断 {_cnt.get('blocked',0)} 个。"
                                   "仅输出合成方向和风险门控，不展示因子名册。")
    except Exception:
        factor_quality_note = "因子质量账本不可用，本报告不采用因子结论"
    technical = blk("technical").get("indices", [])
    patterns = blk("technical").get("patterns", [])
    overseas = blk("overseas").get("quotes", [])
    dbb = blk("data_base")
    freshness = blk("freshness")
    _gate_stat = ((freshness.get("fresh",0), freshness.get("stale",0), freshness.get("expired",0),
                   freshness.get("score",0)) if freshness else None)
    veto = blk("macro_veto")
    verdict = blk("model_verdict").get("verdict", {})
    mv = blk("model_verdict")
    ml = blk("ml_verdict").get("predictions", [])
    bt = blk("backtest_check").get("checks", [])
    opp = blk("opportunity").get("opportunities", [])

    stage = emo.get("stage_cn", "?")
    dz = dec.get("定调", {})
    fz_total = fz.get("total")
    fz_dims = fz.get("dimensions") or {}
    fz_gate = fz.get("gate") or {}
    atk = (dec.get("操作清单") or {}).get("attack", [])
    obs = (dec.get("操作清单") or {}).get("observe", [])
    risks = dec.get("风险预案", [])

    dim_order = [("technical", "技术面"), ("emotion", "情绪面"), ("fund", "资金面"),
                 ("factor", "因子面"), ("mainline", "主线面"), ("overseas", "海外面")]
    dim_html = "".join(
        f'<div class="dim"><div class="dn">{cn}</div>{_bar(v.get("score", 0), 100, "#f23645" if v.get("score", 0) >= 65 else ("#4a90d9" if v.get("score", 0) < 45 else "#e0a13c"))}<div class="ds">{v.get("score", 0):.0f}</div><div class="dnt">{v.get("note", "")}</div></div>'
        for k, cn in dim_order if (v := fz_dims.get(k)))

    posture = dz.get("posture", "?")
    pc = "#f23645" if "进攻" in posture else ("#4a90d9" if "防守" in posture else "#e0a13c")

    tech_rows = []
    for t in technical:
        macd = t.get("macd") or {}
        tech_rows.append([t.get("name", ""), str(t.get("last", "")), f'{t.get("score", 0):.0f}',
                          t.get("trend", ""), str(t.get("ma5", "-")), str(t.get("ma10", "-")),
                          str(t.get("ma20", "-")), str(t.get("ma60", "-")), str(t.get("rsi", "-")),
                          str(t.get("volume", "-")), f'{t.get("support", "-")} / {t.get("resistance", "-")}',
                          "金叉" if macd.get("golden") else ("死叉" if macd else "-")])

    broker_rows = [[str(o.get("name", ""))[:28], _fmt_num(o.get("buy"), 0), _fmt_num(o.get("sell"), 0),
                    _fmt_num(o.get("net"), 0), str(o.get("stocks", ""))[:40]]
                   for o in (lhb.get("broker_top") or [])]
    stock_rows = [[str(o.get("name", "")) + "(" + str(o.get("code", "")) + ")", _fmt_num(o.get("net"), 2),
                   f'{o.get("pct", 0):.1f}%', str(o.get("reason", ""))[:40]]
                  for o in (lhb.get("stock_top") or [])]

    pool_html = ""
    for key, name in [("short_term", "短线攻击方向"), ("lhb_stocks", "龙虎榜强势"),
                      ("rs_stocks", "RS相对强弱"), ("mid_long", "中长线低估")]:
        items = pools.get(key, [])
        if not items: continue
        rows = []
        for it in items:
            nm = it.get("name") or it.get("concept") or it.get("code", "")
            extra = it.get("signal") or ("#" + str(it.get("rank")) if it.get("rank") else (it.get("level") or ""))
            det = f"PE{it.get('pe')}/PB{it.get('pb')}" if it.get("pe") else str(extra)
            rows.append([str(nm), str(det), str(it.get("net") or it.get("why") or "")])
        pool_html += f'<div class="pool"><h4>{name} ({len(items)})</h4>{_tbl(["名称", "细节", "备注"], rows[:12])}</div>'

    factor_rows = [[k, _bar(v.get("score", 0) * 100, 100, "#f23645" if v.get("score", 0) >= 0.38 else "#4aa3ff"),
                    f'{v.get("score", 0) * 100:.0f}%', str(v.get("n", ""))] for k, v in fg.items()]

    _io = dbb.get("institutional", {}).get("new_entries", [])
    inst_html = " · ".join(f"{e.get('name', '')}({e.get('code', '')})" for e in _io[:10]) if _io else "-"
    ind_html = " · ".join(f"{x.get('name', '')}({x.get('chg', 0):+.1f}%)" for x in dbb.get("industry", {}).get("strong", [])[:10]) if dbb.get("industry", {}).get("strong") else "-"
    db_html = ""
    _mk = dbb.get("macro", {})
    if _mk.get("items"):
        db_html += f'<p><b>宏观</b>: {" · ".join(str(i.get("name")) + "=" + str(i.get("value")) for i in _mk["items"][:8])}</p>'
    _hk = dbb.get("hot", {})
    if _hk.get("top"):
        ht = " · ".join(f"{t.get('code','')}({t.get('pct_chg',0):+.1f}%)" for t in _hk["top"][:12])
        db_html += f'<p><b>热榜</b>: 涨{_hk.get("up_n")}跌{_hk.get("down_n")} · {ht}</p>'
    _ik = dbb.get("institutional", {})
    if _ik.get("new_entries"):
        db_html += '<p><b>机构新进</b>: ' + (inst_html if inst_html != "-" else "-") + '</p>'
    db_html += '<p><b>强行业</b>: ' + (ind_html if ind_html != "-" else "-") + '</p>'
    for nm_key, nm_lbl in [("north_margin", "资金基座"), ("style", "风格"), ("event", "事件域"), ("concept", "概念周期")]:
        _v = dbb.get(nm_key, {})
        if _v.get("note"):
            db_html += f"<p><b>{nm_lbl}</b>: {_v['note']}</p>"

    scan = freshness.get("scan", {})
    gate_rows = [[name, info.get("cycle", ""), info.get("latest", ""), f'{info.get("lag", "-")}天',
                  '<span class="tag fresh">新鲜</span>' if info.get("level") == "fresh"
                  else ('<span class="tag stale">滞后</span>' if info.get("level") == "stale"
                        else '<span class="tag expired">过期</span>')]
                 for name, info in sorted(scan.items(), key=lambda x: (x[1].get("lag") if x[1].get("lag") is not None else -1))]

    pattern_rows = [[str(p)[:110]] for p in patterns[:15]]
    ov_rows = [[q.get("label", ""), str(q.get("price", "")), q.get("currency", ""),
                f'{q.get("chg_pct", 0):+.2f}%' if q.get("chg_pct") is not None else "-",
                ('🟢' if (q.get("chg_pct") or 0) > 0 else ('🔴' if (q.get("chg_pct") or 0) < 0 else "⚪"))]
               for q in overseas]

    veto_html = ""
    if veto.get("date") or veto.get("level"):
        veto_html = f'''<p>级别: <b>{veto.get("level", "-")}</b> | 仓位系数 <b>{veto.get("position_coef", "-")}</b>
        | PE分位 {float(veto.get("pe_percentile", 0) or 0) * 100:.0f}% | 巴菲特指标 {float(veto.get("buffett_ratio", 0) or 0) * 100:.0f}%
        | 状态 <b>{veto.get("regime", "-")}</b></p>
        <ul>{''.join(f"<li>{r}</li>" for r in (veto.get("reasons") or [])[:8])}</ul>'''

    # 引擎融合共识（引擎→决策，2026-08-22 融合升级）
    _ef_html = '<p class="sub">引擎融合共识未生成(等 daily_review_chain)</p>'
    try:
        _efv = blk("engine_fusion")
        if _efv.get("ok_n") or _efv.get("score"):
            _ef_labels = {"risk": "风险温度", "meta": "元审查", "card": "行动卡",
                          "expert": "专家共识", "trust": "校准信任"}
            _ef_dim = "".join(
                f'<li class="sub">{_ef_labels.get(k, k)}: {d.get("score")}分 ({d.get("note", "")})</li>'
                for k, d in (_efv.get("dimensions") or {}).items())
            _ef_html = (f'<p>共识: <b>{_efv.get("consensus", "?")}</b> | 融合分 {_efv.get("score")} '
                        f'| 可用引擎 {_efv.get("ok_n")} 个（信用 {_efv.get("trust")}）</p>'
                        f'<ul>{_ef_dim}</ul>'
                        f'<p class="sub">{_efv.get("note", "")}</p>')
    except Exception:
        pass
    # 预测验证闭环（2026-08-22 融合升级）
    _pl_html = '<p class="sub">预测闭环未生成(等 daily_review_chain)</p>'
    try:
        _plv = blk("prediction_loop")
        if _plv.get("verified_n") is not None or _plv.get("note"):
            _pl_mods = "".join(
                f'<li class="sub">{m.get("module")}: {m.get("hit")}（n={m.get("n")}）'
                f'{" ⚠️降权" if m.get("n", 0) >= 5 and m.get("hit", 1) < 0.5 else ""}</li>'
                for m in (_plv.get("by_module") or []))
            _pl_html = (f'<p>本次验证 <b>{_plv.get("verified_n", 0)}</b> 条 · 判对 {_plv.get("correct_n", 0)} 条'
                        f' · 总命中率 <b>{_plv.get("hit", "-")}</b>（Brier {_plv.get("brier", "-")}）</p>'
                        f'<p class="sub">pending 剩余 {_plv.get("pending", 0)} / 累计 {_plv.get("total", 0)} · {_plv.get("note", "")}</p>'
                        + (f'<ul>{_pl_mods}</ul>' if _pl_mods else ''))
    except Exception:
        pass

    hist = []
    try:
        review_files = []
        for rf in glob.glob(str(ROOT / "generated" / "review_*.json")):
            match = re.search(r"review_(20\d{2}-\d{2}-\d{2})\.json$", rf)
            if match and match.group(1) <= date:
                review_files.append(rf)
        for rf in sorted(review_files)[-6:]:
            try:
                rd = json.loads(Path(rf).read_text(encoding="utf-8"))
                rb = rd.get("blocks", {})
                rt = (rb.get("market_temperature") or {}).get("value", {}).get("components", {}).get("emotion", {})
                rl = (rb.get("leader_sentiment") or {}).get("value", {}).get("components", {}).get("ladder", {})
                hist.append([str(rd.get("date", ""))[:10], rt.get("stage_cn", "?"),
                             str(rt.get("zt_cnt", "-")), str(rl.get("max_board", "-")),
                             f'{rl.get("zb_rate", "-")}'])
            except Exception:
                continue
        if len(hist) < 3:
            # 复盘历史不足 → 用数据仓库真实涨停/天梯历史兜底（同口径，非编造）
            import pandas as _pdZ
            _zs = _pdZ.read_parquet(ROOT / "data_warehouse" / "market" / "zt_daily_stats.parquet")
            _zs = _as_of_frame(_zs, date)
            for _, _zr in _zs.tail(6).iterrows():
                hist.append([str(_zr.get("date", ""))[:10],
                             str(_zr.get("emotion_stage_cn") or _zr.get("stage_cn") or "?"),
                             str(_zr.get("zt_cnt", "-")), str(_zr.get("max_board", "-")),
                             f'{float(_zr.get("zb_rate", 0) or 0):.2f}'])
    except Exception:
        pass
    hist_rows = hist[-6:]
    _maxzt = max([int(h[2]) for h in hist_rows if str(h[2]).isdigit()] or [1])
    _hist_chart = ""
    if hist_rows:
        _parts = []
        for _h in hist_rows:
            _zt = int(_h[2]) if str(_h[2]).isdigit() else 0
            _hh = max(4, (_zt / _maxzt) * 60)
            _parts.append('<div style="flex:1;text-align:center">'
                          + '<div style="background:#4a90d9;height:%.0fpx;border-radius:4px"></div>' % _hh
                          + '<div style="color:#8ea0b8;font-size:10px">' + str(_h[0])[5:] + '</div>'
                          + '<div style="font-size:10px">' + str(_h[1]) + '</div></div>')
        _hist_chart = ('<div style="display:flex;gap:6px;align-items:flex-end;height:80px;margin-top:10px">'
                       + "".join(_parts) + '</div>')

    verdict_html = f'<p>共识: <b>{verdict.get("consensus", "-")}</b> | 置信 {verdict.get("confidence", "-")} | {verdict.get("votes", "-")}票</p>'
    _ima_topics = _ima_res.get("topics", []) or []
    _ima_count = len(_ima_topics)
    _ima_html = ''.join(
        f'<b>{t.get("topic","")}</b><ul>' + ''.join(f'<li class="sub">{h[:70]}</li>' for h in (t.get("hits") or [])[:3]) + '</ul>'
        for t in _ima_topics) or '<p class="sub">IMA库搜索无命中</p>'
    _fund_more_html = ""
    if _picker:
        _fund_more_html += '<b style="display:block;margin-top:8px">🎯 财报选股池(高ROE+高增+低负债)</b>' + _tbl(
            ["代码", "ROE%", "增速%", "负债%", "EPS", "评分"], [[x["code"], f'{x["roe"]:.1f}', f'{x["grow"]:.0f}', f'{x["lev"]:.0f}', f'{x["eps"]:.2f}', f'{x["score"]:.0f}'] for x in _picker])
    if _val:
        _fund_more_html += '<b style="display:block;margin-top:8px">📊 估值(PE_TTM/PB 低估值排序)</b>' + _tbl(
            ["代码", "PE", "PB"], [[x["code"], f'{x["pe"]:.1f}' if x["pe"] else "-", f'{x["pb"]:.2f}' if x["pb"] else "-"] for x in _val[:8]])
    if _ann and _ann.get("items"):
        _fund_more_html += '<b style="display:block;margin-top:8px">📅 公告事件日历</b>' + _tbl(
            ["日期", "代码", "名称", "类型", "标题"], [[x["date"], x["code"], x["name"], x["type"], x["title"]] for x in _ann["items"][:8]])
    _fund_html = ""
    if _fund_hist:
        _last_f = _fund_hist[-1]
        _fund_html = f'<p class="sub">最新({_last_f["d"]}): 游资{_last_f["yz"]}亿 机构{_last_f["jg"]}亿 合力{_last_f["fi"]}</p>'
    _mpano_html = ""
    if _mpano:
        _mpc = _mpano.get("commodity") or []
        if _mpc:
            _cc = " · ".join("%s=%s" % (c["name"], c["last"]) for c in _mpc[:5])
            _mpano_html += f'<p><b>商品期货</b>: {_cc}</p>'
        _mpr = _mpano.get("rates") or []
        if _mpr:
            _rr = " · ".join("%s=%s" % (r["name"], r["last"]) for r in _mpr[:4])
            _mpano_html += f'<p><b>利率</b>: {_rr}</p>'
        _mpm = _mpano.get("macro") or {}
        if _mpm:
            _mm = " · ".join("%s=%s" % (k, v) for k, v in list(_mpm.items())[:8])
            _mpano_html += f'<p><b>月度宏观</b>: {_mm}</p>'
        for _k, _nm in [("zt_stats","涨停统计"),("regime","状态机"),("margin","两融"),("fund_flow","市场资金流")]:
            if _mpano.get(_k):
                _mpano_html += f'<p class="sub"><b>{_nm}</b>: {str(_mpano[_k])[:70]}</p>'
    else:
        _mpano_html = '<p class="sub">市场全景数据缺失</p>'
    _social_html = ""
    if _social_md:
        import html as _hs2
        _social_html = '<pre style="white-space:pre-wrap;font-family:inherit;font-size:12px">' + _hs2.escape(_social_md) + '</pre>'
    _chan_big_html = ""
    if _chan_big.get("note"):
        _cb = _chan_big
        _chan_big_html = (f'<p><b>自动缠论结构</b>: {_cb.get("fractals","-")}分型 {_cb.get("bis","-")}笔 {len(_cb.get("zhongshu",[]))}中枢 '
                          f'→ <b>{_cb.get("cur_pos","-")}</b> 当前中枢 {_cb.get("cur_zs","-")} 收盘{_cb.get("last","-")}</p>')
    # K线+缠论图数据挂全局(JS用)
    _KCHART = {"kdata": _kdata, "bi_pairs": _chan_big.get("bi_pairs", []), "bi_prices": _chan_big.get("bi_prices", []),
               "emo_hist": _emo_hist, "emo_cnt": len(_emo_hist),
               "fund_hist": _fund_hist,
               "theme_hist": _theme_hist, "ov_kline": _ov_kline, "gate_stat": _gate_stat,
               "ftrend": _ftrend, "cm_trend": _cm_trend}
    _chan_html = ""
    if _chan_all:
        _chan_html += _tbl(["指数", "收盘", "日线", "日中枢", "周线", "周中枢"],
                            [[x["index"], str(x["last"]), x["day_pos"], x["day_zs"], x["week_pos"], x["week_zs"]] for x in _chan_all]) + "<br>"
    if _chan_wk.get("note"):
        _chan_html = f'<p><b>周线级</b>: {_chan_wk.get("note","")}</p>' + _chan_html
    if _chan_ok and _chan_md:
        import html as _hs
        _chan_html += '<div class="chan-box"><pre style="white-space:pre-wrap;font-family:inherit;font-size:12px">' + _hs.escape(_chan_md) + '</pre></div>'
    _wave_all_html = ""
    if '_wave_all' in dir() and _wave_all:
        _wave_all_html = _tbl(["指数", "收盘", "结构", "60日分位", "60日高低"],
                              [[w.get("指数",""), str(w.get("收盘","")), w.get("结构",""),
                                f'{w.get("60日分位",0):.0f}%', f'{w.get("高60","")}/{w.get("低60","")}'] for w in _wave_all])
    _wave_html = ""
    if _wave.get("last"):
        _wave_html = (f'<p><b>{_wave.get("index")}</b> {_wave.get("last")} · {_wave.get("position")}'
                      f' · 位置距高{_wave.get("dist_high_pct")}%/距低{_wave.get("dist_low_pct")}% · 展望{_wave.get("outlook")}</p>'
                      f'<p class="sub">{_wave.get("ma_struct")}</p>'
                      + _tbl(["剧本", "触发", "含义"], [[x.get("剧本",""), x.get("触发",""), x.get("含义","")] for x in _wave.get("scripts", [])])
                      + f'<p class="sub">{_wave.get("note","")}</p>')
    _fundamental_html = ""
    _rt = _fund.get("roe_top", []) or []
    _gt = _fund.get("growth_top", []) or []
    _lt = _fund.get("leverage_least", []) or []
    _et = _fund.get("eps_top", []) or []
    if _rt or _gt:
        if _rt:
            _fundamental_html += '<b>ROE最高</b>' + _tbl(["代码", "ROE%", "EPS"], [[x.get("code",""), f'{x.get("val",0):.1f}%', f'{x.get("eps",0):.2f}'] for x in _rt[:6]])
        if _gt:
            _fundamental_html += '<b style="display:block;margin-top:6px">净利增速最快</b>' + _tbl(["代码", "增速%"], [[x.get("code",""), f'{x.get("val",0):+.0f}%'] for x in _gt[:6]])
        if _lt:
            _fundamental_html += '<b style="display:block;margin-top:6px">低负债</b>' + _tbl(["代码", "负债率%"], [[x.get("code",""), f'{x.get("val",0):.0f}%'] for x in _lt[:5]])
        if _et:
            _fundamental_html += '<b style="display:block;margin-top:6px">每股收益最高</b>' + _tbl(["代码", "EPS元"], [[x.get("code",""), f'{x.get("val",0):.2f}'] for x in _et[:5]])
    else:
        _fundamental_html = '<p class="sub">财报数据缺失</p>'
    # 量化量化(从已载入的 factor 数据 + blocks 扩展)
    _quant_html = ""
    _qg = blk("quant_groups") if "quant_groups" in blocks else {}
    _qtop = factor.get("top_factors", []) or []
    _qweak = factor.get("weak_factors", []) or []
    _qrows = [[k, f'{v.get("score",0)*100:.0f}%', str(v.get("n",""))] for k, v in fg.items()]
    if _qrows:
        _quant_html = _tbl(["风格组", "强度", "因子数"], _qrows)
    if _qtop:
        _quant_html += f'<b style="display:block;margin-top:6px">强势因子</b><div class="sub">{"、".join(_qtop[:6])}</div>'
    if _qweak:
        _quant_html += f'<b style="display:block;margin-top:6px">走弱因子</b><div class="sub">{"、".join(_qweak[:6])}</div>'
    def _ml_txt(p):
        pd2 = p.get("pred") or {}
        if pd2.get("error"):
            return str(pd2["error"])[:50]
        return f'{pd2.get("prediction_label","-")} 置信{pd2.get("confidence",0):.2f}'
    _mlrows = [[p.get("symbol",""), _ml_txt(p)] for p in ml]
    if _mlrows:
        _quant_html += '<b style="display:block;margin-top:6px">ML 预测</b>' + _tbl(["标的", "预测"], _mlrows)
    _btrows = [[c.get("symbol",""), f'{c.get("mom20",0)*100:.1f}%', str(c.get("vol20","-"))] for c in bt]
    if _btrows:
        _quant_html += '<b style="display:block;margin-top:6px">回测速览(动量/波动)</b>' + _tbl(["标的", "20日动量", "波动率"], _btrows)
    _eng_html = ""
    _vol_html = ""
    _de_ok = (0 if not _deepeng else int(_deepeng.get("ok", 0)))
    _de_html = ""
    # 引擎工厂账本优先（87引擎真产状态）
    if _ledger.get("engines"):
        _le = _ledger["engines"]
        _ok_l = [k for k, v in _le.items() if v.get("status") == "真产"]
        _bad_l = [k for k, v in _le.items() if v.get("status") in ("空壳", "异常", "超时", "无recipe")]
        _de_ok = len(_ok_l)
        _de_html = _tbl(["引擎", "实质产出"],
                        [[k, str(_le[k].get("summary", ""))[:70]] for k in _ok_l[:18]])
        _de_html += f'<p class="sub">引擎工厂账本: 真产 {_de_ok}/{len(_le)} · 待办 {len(_bad_l)} (其余为配置面/降级面)</p>'
        if _bad_l:
            _de_html += f'<p class="sub">⚠️ 待办: {"、".join(_bad_l[:12])}</p>'
    elif _deepeng.get("engines"):
        _de_list = [k for k, v in _deepeng["engines"].items() if v.get("ok")]
        # 合并手动扫描(deep_manual.json) 真产引擎
        try:
            _jm2 = json.loads((ROOT / "generated" / "deep_manual.json").read_text(encoding="utf-8"))
        except Exception:
            _jm2 = {}
        _de_all2 = dict(_jm2)
        for k, v in _deepeng["engines"].items():
            if v.get("ok") and k not in _de_all2:
                _de_all2[k] = v
        _de_ok = len(_de_all2)
        _de_html = _tbl(["引擎", "实质产出"],
                        [[k, str(v.get("summary", ""))[:70]] for k, v in list(_de_all2.items())[:15]])
        _de_html += f'<p class="sub">深改造v4: 手动+自动合并, {_de_ok}个真实产出(其余依赖前置数据)</p>'
    else:
        _de_html = '<p class="sub">深改造待跑(engine_deep.py / engine_factory.py)</p>' 
    _ae_ok = (0 if not _alleng else int(_alleng.get("ok", 0)))
    _ae_total = (0 if not _alleng else int(_alleng.get("scanned", 0)))
    _ae_html = ""
    if _alleng.get("engines"):
        _ag = _alleng["engines"]
        _ok_l = [k for k, v in _ag.items() if v.get("ok")]
        _err_l = [k for k, v in _ag.items() if not v.get("ok")]
        if _ok_l:
            _ae_html = f'<b>✅ 运行成功 ({len(_ok_l)})</b><p class="sub">{"、".join(_ok_l[:36])}</p>'
        if _err_l:
            _ae_html += f'<b style="display:block;margin-top:6px">⚠️ 降级 ({len(_err_l)})</b><p class="sub">{"、".join(_err_l[:24])}</p>'
    else:
        _ae_html = '<p class="sub">全引擎扫描未完成(等 all_engines_scan.py)</p>' 
    _evol = _eng.get("vol") or {}
    if _evol.get("poc") or _evol.get("ok"):
        _vol_html = ("<b style='display:block;margin-top:6px'>量价分布(600519)</b>"
                     f"<p class='sub'>POC {_evol.get('poc',0):.1f} · 上轨{_evol.get('upper',0):.1f} · 下轨{_evol.get('lower',0):.1f}</p>")
    _emeta = _eng.get("meta") or {}
    if _emeta.get("conclusions"):
        _mc = _emeta["conclusions"][:3]
        _eng_html += "<b style='display:block;margin-top:6px'>元审查(报告质控)</b><ul>" + "".join(
            f'<li class="sub">{str(c.get("report_type",""))}: {str(c.get("stance",""))} 置信{c.get("confidence",0)} · {str(c.get("summary",""))[:40]}</li>' for c in _mc) + "</ul>"
    _etrend = _eng.get("trend") or {}
    if _etrend and isinstance(_etrend, dict):
        _tp = []
        for code, tv in list(_etrend.items())[:2]:
            if isinstance(tv, dict):
                _sg = tv.get("signal") or tv.get("trend") or ""
                _tp.append(f"{code}:{str(_sg)[:20]}")
        if _tp:
            _eng_html += "<b style='display:block;margin-top:6px'>趋势检测</b><p class='sub'>" + "、".join(_tp) + "</p>"
    _emj = _eng.get("macro") or {}
    if _emj.get("assets"):
        _parts = []
        for _v in list(_emj["assets"].values())[:4]:
            _ch = _v.get("chg5_pct")
            _ch_s = f"{_ch:.2f}%" if isinstance(_ch, (int, float)) else "-"
            _parts.append("%s %s %s" % (_v.get("name",""), _v.get("latest",""), _ch_s))
        _aa = " · ".join(_parts)
        _eng_html += f'<p><b>海外宏观</b>({_emj.get("_source","")}): {_aa}</p>'
    _ecal = _eng.get("calendar") or {}
    if _ecal.get("date"):
        _vn = {k: str(v) for k, v in (_ecal.get("verified") or {}).items() if v != "不支持" and v != False}
        _eng_html += f'<p><b>日历效应</b>: {"、".join(f"{k}={v}" for k, v in list(_vn.items())[:5]) or "今日无显著日历窗口"}</p>'
    _ers = _eng.get("rs") or {}
    if _ers.get("items"):
        _eng_html += f'<p><b>相对强弱</b>: ' + "、".join(str(i.get("symbol") or i.get("code") or "") for i in _ers["items"][:5]) + '</p>'
    _elens = _eng.get("lens") or {}
    if _elens and isinstance(_elens, dict):
        _lens_parts = []
        for code, lv in list(_elens.items())[:3]:
            nm = lv.get("name") or code
            sig = lv.get("signal") or lv.get("level") or ""
            _lens_parts.append(f"{nm}({sig})" if sig else nm)
        if _lens_parts:
            _eng_html += f'<p><b>个股透镜</b>: {"、".join(_lens_parts)}</p>'
    if not _eng_html:
        _eng_html = '<p class="sub">六引擎输出缺失(待本机跑 engine_pulse)</p>'
    _theme_rot_html = ""
    if _theme_rot:
        _theme_rot_html = _tbl(["概念", "涨停合计", "最高板"], [[x["n"], str(x["zt"]), str(x["mb"])] for x in _theme_rot])
    else:
        _theme_rot_html = '<p class="sub">题材数据缺失</p>'
    _btp_html = ""
    _bmom = _btp.get("mom", {}) or {}
    if _bmom.get("top"):
        _tp = "、".join("%s(%+.0f%%)" % (x["symbol"], x["mom60"]) for x in _bmom["top"][:8])
        _btp_html += f'<p><b>动量Top(60日)</b>: {_tp}</p>'
    _bpk = _btp.get("picks", {}) or {}
    if _bpk.get("picks"):
        _pk2 = "、".join("%s(%+.0f%%)" % (x["symbol"], x["mom20"]) for x in _bpk["picks"][:6])
        _btp_html += f'<p><b>策略合流选股({_bpk.get("count")}只)</b>: {_pk2}</p>'
    _bpa = _btp.get("panel", {}) or {}
    if _bpa:
        _pl = " | ".join("%s:%s(%+.1f%%)" % (sym, "、".join(sv["signals"]) or "无", sv["mom20"]) for sym, sv in list(_bpa.items())[:4])
        _btp_html += f'<p><b>核心票信号</b>: {_pl}</p>'
    if not _btp_html:
        _btp_html = '<p class="sub">回测数据待生成(backtest_pool.py)</p>'
    _mlscan_html = ""
    _msu = _mlscan.get("top_up") or []
    _msd = _mlscan.get("top_down") or []
    if _msu:
        _uu = "、".join("%s(up%.2f)" % (x["symbol"], x["up"]) for x in _msu[:8])
        _mlscan_html += f'<p><b>ML强势候选({_mlscan.get("up_n","-")})</b>: {_uu}</p>'
    if _msd:
        _dd = "、".join("%s(dn%.2f)" % (x["symbol"], x["down"]) for x in _msd[:8])
        _mlscan_html += f'<p><b>ML回避预警({_mlscan.get("down_n","-")})</b>: {_dd}</p>'
    if not _mlscan_html:
        _mlscan_html = '<p class="sub">ML全池扫描待跑(ml_market_scan.py --core)</p>'
    _pred_html = ""
    if _pred_chk:
        _pred_html = _tbl(["日期", "情绪", "涨停", "最高板"],
                          [[x["d"], x["stage"], str(x["zt"]), str(x["mb"])] for x in _pred_chk[-6:]])
        _st = [x["stage"] for x in _pred_chk]
        _pred_html += f'<p class="sub">情绪轨迹: {"→".join(_st[-6:])}</p>'
    else:
        _pred_html = '<p class="sub">历史复盘不足</p>'
    _cf_html = ""
    if _cf_top:
        _ct = "、".join("%s(%+.0f亿)" % (x["n"], x["v"]) for x in _cf_top[:6])
        _cf_html += f'<p><b>流入Top</b>: {_ct}</p>'
    if _cf_flop:
        _cf2 = "、".join("%s(%+.0f亿)" % (x["n"], x["v"]) for x in _cf_flop[:6])
        _cf_html += f'<p><b>流出Flop</b>: {_cf2}</p>'
    if not _cf_html:
        _cf_html = '<p class="sub">概念资金流缺失</p>'
    _dzt = _depth.get("zt_board", []) or []
    _demotion = _depth.get("emotion", {}) or {}
    _dbench = _depth.get("benchmark", []) or []
    _dkeys = _depth.get("reason_keywords", []) or []
    _depth_html = ""
    if _demotion:
        _depth_html += (f'<p><b>核心情绪</b>: 涨停{_demotion.get("zt","-")} 炸板{_demotion.get("zb","-")} '
                        f'封板率{_demotion.get("seal_rate","-")}% 空间龙头<b>{_demotion.get("space_leader","-")}</b> '
                        f'连板{_demotion.get("lianban","-")}</p>')
    if _dzt:
        _depth_html += _tbl(["板块", "涨停", "最高板", "龙头", "龙头板数"],
                            [[x.get("板",""), str(x.get("涨停","")), str(x.get("最高板","")),
                              str(x.get("龙头","")), str(x.get("龙头板数",""))] for x in _dzt])
    if _dbench:
        _depth_html += '<b style="display:block;margin-top:8px">基准指标对比</b>' + _tbl(
            ["指标", "阈值", "实测", "判定"], [[b.get("指标",""), b.get("阈值",""), b.get("实测",""), b.get("判定","")] for b in _dbench])
    _dcs = _depth.get("chain_stage", []) or []
    if _dcs:
        _depth_html += '<b style="display:block;margin-top:8px">产业链热度阶段</b>' + _tbl(
            ["板块", "涨停", "最高板", "炸板", "阶段"],
            [[x.get("板块",""), str(x.get("涨停","")), str(x.get("最高板","")), str(x.get("炸板","")), x.get("阶段","")] for x in _dcs])
    if _dkeys:
        _depth_html += f'<p class="sub" style="margin-top:6px">热点行业: {"、".join(_dkeys[:8])}</p>'
    _mav = mv.get("components", {}).get("multi_agent", {})
    if isinstance(_mav, dict) and _mav.get("error"):
        verdict_html += f'<p class="sub">⚠️ {_mav["error"]}</p>'

    # 作战决策(battle_map)渲染
    _bm_html = ""
    if bm:
        _bm_rows = []
        if bm.get("recommended"):
            _bm_rows.append(["行动卡", str(bm.get("recommended")), f'仓位{bm.get("position_range")} 置信{bm.get("confidence")}'])
        if bm.get("emotion_stage"):
            _bm_rows.append(["情绪", str(bm.get("emotion_stage")), str(bm.get("emotion_conf"))])
        if bm.get("regime"):
            _bm_rows.append(["状态", str(bm.get("regime")), ""])
        if bm.get("macro_veto"):
            _bm_rows.append(["宏观否决", str(bm.get("macro_veto")), ""])
        if bm.get("bid_watch"):
            for bwd in bm["bid_watch"][:6]:
                _bm_rows.append(["竞价锚点", str(bwd.get("item","")), f"{bwd.get('signal','')} → {bwd.get('action','')}"])
        _bm_html += _tbl(["项", "内容", "细节"], _bm_rows)
        _ag = bm.get("attack_groups") or {}
        if _ag.get("core"):
            _core_rows = [[str(a.get("name","")), str(a.get("score","")), str(a.get("strategy","")),
                           str((a.get("leader") or {}).get("name","")) + " " + str((a.get("leader") or {}).get("boards","")) + "板"]
                          for a in _ag["core"][:8]]
            _bm_html += '<b style="display:block;margin-top:8px">攻击方向·核心</b>' + _tbl(["方向", "强度", "策略", "龙头"], _core_rows)
        for gk in ("observe", "avoid"):
            gl = _ag.get(gk) or []
            if gl:
                _bm_html += f'<b style="display:block;margin-top:6px">{gk}</b>' + _tbl(
                    ["方向", "强度", "策略"], [[str(a.get("name","")), str(a.get("score","")), str(a.get("strategy",""))] for a in gl[:5]])
        rw = bm.get("risk_watch") or []
        if rw:
            _bm_html += f'<b style="display:block;margin-top:6px">风险清单({len(rw)})</b><p class="sub">{"、".join(str(r.get("name") or r.get("code","")) for r in rw[:10])}</p>'
    _opp_html = ""
    if opp:
        _opp_html = '<b style="display:block;margin-top:8px">🎯 机会池</b>' + _tbl(
            ["标的", "理由"], [[str(o.get("code") or o.get("name","")), str(o)[:60]] for o in opp[:8]])

        lnk = bm.get("linkage")
        if isinstance(lnk, list) and lnk:
            _bm_html += '<b style="display:block;margin-top:6px">产业链联动</b><ul>' + "".join(f'<li class="sub">{str(x)[:60]}</li>' for x in lnk[:5]) + '</ul>'
    _chart_json = json.dumps(_KCHART, ensure_ascii=False)
    _CHART_HTML = _ECharts_JS_TMPL.replace("__CHAN_JSON__", _chart_json) if _kdata else ""

    # 决策支持结论：回测/因子只产出“对当前决策的启示”，不展示审计脚手架。
    _strategy_conclusion_html = _render_strategy_conclusions(date)
    _factor_effectiveness_html = _render_factor_effectiveness(date)

    # 研报图表解析章节（读同一 vision_analysis.jsonl，图片资产复制到当天报告目录 assets/）
    _vis_records, _vis_failed = _load_vision_insights(date)
    _vis_html = render_vision_insights_section(_vis_records, _vis_failed, report_dir=report_dir)
    _decision_snapshot = _load_decision_snapshot(date)
    _decision_html = _render_decision_snapshot(_decision_snapshot)
    _decision_audit_html = _render_decision_audit(_decision_snapshot)
    _fusion_snapshot = _load_research_fusion_snapshot(date)
    _fusion_html = _render_research_fusion_snapshot(_fusion_snapshot)

    return f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>每日综合详细研报 {date} · {PRODUCT_VERSION}</title>
<style>
:root{{--color-bg:#0d1117;--color-panel:#161b22;--color-border:#30363d;--color-text:#e6edf3;--color-muted:#8b949e;--color-accent:#58a6ff;--color-up:#f6465d;--color-down:#2ebd85;--color-warn:#d29922;--color-danger:#f85149;--bg:var(--color-bg);--card:var(--color-panel);--card2:#1c2128;--line:var(--color-border);--tx:var(--color-text);--sub:var(--color-muted);--red:var(--color-up);--grn:var(--color-down);--amber:var(--color-warn);--blue:var(--color-accent);--font-num:"SF Mono","JetBrains Mono",Consolas,"Liberation Mono",monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{color-scheme:dark;scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--tx);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1360px;margin:0 auto;padding:20px 24px 44px;font-size:13px;line-height:1.55}}
.report-shell{{max-width:1280px;margin:0 auto}}
.terminal-bar{{position:sticky;top:0;z-index:50;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 0 14px;background:rgba(13,17,23,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.terminal-name{{font:700 12px/1 var(--font-num);letter-spacing:.08em;color:var(--color-accent)}}
.terminal-meta{{font:12px/1.4 var(--font-num);color:var(--sub);text-align:right}}
h1{{font-size:24px;letter-spacing:0;margin:18px 0 4px}}.meta{{color:var(--sub);font-size:12px;margin-bottom:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:12px;box-shadow:none}}
.card h2{{font-size:15px;margin-bottom:12px;padding-left:9px;border-left:3px solid var(--blue);letter-spacing:0}}
.verdict{{background:var(--card);border:1px solid var(--color-accent);border-left:4px solid var(--color-accent);border-radius:8px;padding:18px;margin-bottom:12px}}
.verdict .row{{display:flex;align-items:center;gap:20px;flex-wrap:wrap}}.act{{font-size:25px;font-weight:800}}
table{{width:100%;border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);word-break:break-word;font-variant-numeric:tabular-nums}}th{{color:#c9d1d9;font-weight:650;background:#1c2128;white-space:nowrap}}tbody tr:hover{{background:rgba(255,255,255,.03)}}tr:last-child td{{border-bottom:none}}
.up{{color:var(--red)}}.down{{color:var(--grn)}}
.tag{{display:inline-block;padding:2px 7px;border:1px solid transparent;border-radius:4px;font:11px/1.3 var(--font-num)}}.tag.fresh{{background:rgba(46,189,133,.14);border-color:rgba(46,189,133,.35);color:var(--grn)}}.tag.stale{{background:rgba(210,153,34,.14);border-color:rgba(210,153,34,.35);color:var(--amber)}}.tag.expired{{background:rgba(248,81,73,.14);border-color:rgba(248,81,73,.35);color:var(--red)}}
.dim{{display:grid;grid-template-columns:84px minmax(120px,1fr) 44px minmax(140px,1.6fr);gap:10px;align-items:center;padding:7px 0;border-bottom:1px dashed var(--line)}}.dn{{color:var(--sub);font-size:12px}}.ds{{font:700 12px var(--font-num);text-align:right}}.dnt{{font-size:11px;color:var(--sub)}}
.bar{{height:8px;background:var(--card2);border-radius:3px;overflow:hidden}}.fill{{height:100%;border-radius:3px}}.pool{{margin-bottom:12px}}.pool h4{{font-size:13px;color:var(--amber);margin-bottom:7px}}
ul{{list-style:none;padding-left:0}}li{{padding:5px 0;border-bottom:1px dashed var(--line)}}.sub{{color:var(--sub);font-size:11px}}
.fusion-chart-tools{{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 8px}}.fusion-chart-tools button{{background:transparent;border:1px solid var(--line);color:var(--sub);border-radius:6px;padding:6px 10px;cursor:pointer}}.fusion-chart-tools button.on{{background:rgba(88,166,255,.16);color:#c9e2ff;border-color:var(--blue)}}
.decision-note{{margin-top:10px;padding:10px 12px;background:#1c2128;border-left:3px solid var(--amber);line-height:1.7}}
details{{border:1px solid var(--line);border-radius:6px;padding:8px 10px;background:#0d1117}}summary{{cursor:pointer;color:#c9d1d9;font-weight:650}}pre{{font-family:var(--font-num)}}
@media(max-width:700px){{body{{padding:12px}}.terminal-bar{{align-items:flex-start;flex-direction:column;gap:5px}}.terminal-meta{{text-align:left}}.dim{{grid-template-columns:60px minmax(0,1fr) 38px}}.dnt{{grid-column:2 / -1}}.row{{flex-direction:column;align-items:flex-start!important}}}}
</style></head><body><div class="report-shell">
<div class="terminal-bar"><div class="terminal-name">MUYUN TERMINAL / RESEARCH REPORT</div><div class="terminal-meta">AS_OF {date} · GENERATED {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} CST · DATA-DRIVEN</div></div>
<h1>每日 A 股详细研报 · {date}</h1>
<div class="meta">融合决策 · 全维度数据 · 研究用途，不构成投资建议</div>

<div class="verdict"><div class="row">
  <div style="flex:1">
    <div class="meta">明日定调 · 综合评分 <b>{fz_total if fz_total else '-'}</b>/100{(' · 门控' + str(fz_gate.get("score")) + '分 -' + str(fz_gate.get("penalty")) + '分') if fz_gate.get("penalty") else ''}</div>
    <div class="act" style="color:{pc}">{posture}</div>
    <div class="meta">建议仓位 <b>{dz.get("position", "?")}</b>{(' · ' + dz.get("veto_note")) if dz.get("veto_note") else ''}{('<br>⚠️ <b>' + dz.get("cross_note")) if dz.get("cross_note") else ''}</div>
    <div class="meta">{dz.get("basis", "")}</div>
  </div>
  <div style="font-size:42px;font-weight:800;color:{pc}">{fz_total if fz_total else '-'}</div>
</div><div style="margin-top:12px">{dim_html}</div></div>

<div class="card"><h2>统一短线决策：全市场主线扫描与主板精选</h2>
{_decision_html}
</div>

<div class="card"><h2>统一决策审计：覆盖、候选、监管与筹码证据</h2>
{_decision_audit_html}
</div>

<div class="card"><h2>⚡ 作战决策（battle_map）</h2>
{_bm_html}
</div>
<div class="card"><h2>🎯 决策预测轨迹（近{len(_pred_chk)}复盘·验证跟踪）</h2>
{_pred_html}
</div>
<div class="card"><h2>📈 多日趋势（近{len(hist_rows)}复盘）</h2>
<div id="emoChart" style="width:100%;height:220px"></div>
{_tbl(["日期", "情绪", "涨停", "最高板", "炸板率"], hist_rows)}
{_hist_chart}</div>

<div class="card"><h2>💧 资金多日历史（游资/机构）</h2>
<div id="fundChart" style="width:100%;height:200px"></div>
{_fund_html}
</div>
<div class="card"><h2>次日执行剧本：按时间窗口行动</h2>
<table><thead><tr><th>时间窗口</th><th>必须观察</th><th>允许行动</th><th>撤退/降级条件</th></tr></thead><tbody>
<tr><td>09:15-09:30 集合竞价</td><td>主线核心股竞价强度、指数缺口、昨日强势方向是否集体高开、全市场首轮深扫覆盖</td><td>只保留作战地图核心方向；未完成开盘全栈扫描前不新增执行候选</td><td>主线核心低开超过2%、覆盖不足、数据源陈旧时降至观察</td></tr>
<tr><td>09:30-10:00 开盘确认</td><td>五分钟全市场快照、成交额扩散、ETF与行业资金是否同向、炸板率</td><td>资金连续两轮同向且龙头有承接，才按建议仓位下沿试探</td><td>指数与宽度背离、ETF放量流出、炸板率快速升高则撤单或减仓</td></tr>
<tr><td>10:00-11:30 上午主趋势</td><td>行业资金持续性、候选相对强弱、主线成交额排名变化</td><td>只加仓已验证方向，不临时追逐单日资金峰值</td><td>连续两轮跌出资金前排或核心股转弱，停止加仓</td></tr>
<tr><td>13:00-14:30 午��再确认</td><td>午后资金回流、指数新高/新低、ETF份额与价格是否背离</td><td>上午逻辑仍成立才续持；轮动方向仅做观察</td><td>午后回流失败、量价背离或风险清单触发时减至防守仓</td></tr>
<tr><td>14:30-15:00 尾盘决策</td><td>全天净流向、收盘位置、次日事件与公告风险</td><td>保留真正全天持续流入且收盘承接的方向</td><td>尾盘资金回吐、收盘跌破关键支撑或公告硬风险时不隔夜</td></tr>
</tbody></table>
<p class="decision-note"><b>硬规则：</b>五分钟全市场轻评估必须覆盖完整快照；开盘深因子/研究/监管证据未完成时，只能输出观察候选。页面展示条数不是扫描条数，飞书只推执行条件变化，不推单票噪声。</p>
</div>

<div class="card"><h2>🎯 操作清单（进攻{len(atk)}/观察{len(obs)}/风险{len(risks)}）</h2>
<b>进攻清单</b><ul>{''.join(f'<li>🔥 <b>{a.get("name","")}</b> [{a.get("type","")}] {a.get("action","")} <span class="sub">{a.get("why","")}</span></li>' for a in atk) or "<li>无</li>"}</ul>
<b>观察清单</b><ul>{''.join(f'<li>👀 {o.get("name","")} [{o.get("type","")}] {o.get("action","")}</li>' for o in obs) or "<li>无</li>"}</ul>
<b>风险预案</b><ul>{''.join(f'<li>⚠️ {r.get("trigger","")} → <b>{r.get("action","")}</b></li>' for r in risks) or "<li>无显著风险</li>"}</ul></div>

<div class="card"><h2>🌡️ 市场温度</h2>
<p>情绪 <b style="color:{_stage_color(stage)}">{stage}</b> 置信{emo.get("confidence")} · 涨停 <b class="up">{lad.get("zt_cnt","-")}</b> 炸板{lad.get("zb_cnt","-")} 跌停{lad.get("dt_cnt","-")} 最高<b>{lad.get("max_board","-")}</b>板 · 连板{lad.get("lianban","-")}</p>
<p class="sub">资金合力 {fund.get("force_index","-")} · 游资 {_fmt_num(fund.get("youzi_net"))} · 机构 {_fmt_num(fund.get("jg_net"))} · 北向 {_fmt_num(fund.get("north_net"))} · 宏观 {macro.get("regime","-")}</p></div>

<div class="card"><h2>资金/ETF/行业融合快照</h2>
{_fusion_html}
</div>

<div class="card"><h2>🔄 题材轮动（近12日最热概念）</h2>
{_theme_rot_html}
</div>
<div class="card"><h2>💧 概念资金流（最新日 Top/Flop）</h2>
{_cf_html}
</div>
<div class="card"><h2>🚀 强势方向（{len(sector)}）</h2>
{_strong_alert}
{_tbl(["方向", "等级", "强度分"], [[s.get("name",""), s.get("level",""), f'{s.get("score",0):.2f}'] for s in sector])}</div>

<div class="card"><h2>🔬 深度复盘（涨停分级/核心情绪/热点）</h2>
<div id="themeChart" style="width:100%;height:180px"></div>
{_depth_html}
</div>
<div class="card"><h2>💹 财报深度（最细基本面）</h2>
<div id="ftrendChart" style="width:100%;height:170px"></div>
{_fundamental_html}
{_fund_more_html}
</div>
<div class="card"><h2>📚 IMA知识库研报验证（{_ima_count}主题）</h2>
{_ima_html}
</div>
{_vis_html}
<div class="card"><h2>🔥 龙头情绪</h2>
<p>涨停{lad.get("zt_cnt","-")} 炸板{lad.get("zb_cnt","-")} 跌停{lad.get("dt_cnt","-")} 最高{lad.get("max_board","-")}板 · 炸板率{lad.get("zb_rate","-")}</p>
<ul>{''.join(f'<li>{s.get("concept","")}: {s.get("signal","")} {s.get("level","")}</li>' for s in leader.get("signals", [])) or "<li class=sub>无</li>"}</ul>
<p class="sub">{((leader.get("components") or {}).get("leader") or {}).get("note","")}</p></div>

<div class="card"><h2>💰 龙虎榜（营业部 {len(lhb.get("broker_top", []))} / 个股 {len(lhb.get("stock_top", []))}）</h2>
<b>游资营业部</b>{_tbl(["营业部", "买入", "卖出", "净买", "涉及个股"], broker_rows)}
<b style="display:block;margin-top:10px">个股净买</b>{_tbl(["股票", "净买", "涨幅", "上榜原因"], stock_rows)}
<b style="display:block;margin-top:8px">📊 近期{len(_lhb_h)}日龙虎榜前3</b><div class="sub">{" · ".join(f"{x['d']}:{x['n']}" for x in _lhb_h) or "无"}</div>
<b style="display:block;margin-top:6px">📈 近期涨停家数</b><div class="sub">{" · ".join(f"{x['d']}:{x['n']}家" for x in _zt_h) or "无"}</div></div>

<div class="card"><h2>🎯 多格局选股池</h2>{pool_html}</div>
<div class="card"><h2>📈 全池回测策略（动量/三策略/合流选股）</h2>
{_btp_html}
</div>

<div class="card"><h2>🤖 模型集体意见</h2>{verdict_html}
<p class="sub">回测速览: {", ".join(f"{c.get('symbol','')} mom{c.get('mom20','-')} vol{c.get('vol20','-')}" for c in bt) or "-"} · 机会池 {len(opp)}</p>
{_opp_html}
</div>

<div class="card"><h2>📐 因子强度（{len(fg)}组）</h2>
{_tbl(["风格", "强度", "高位占比", "因子数"], factor_rows)}
<p class="sub">强势: {", ".join(factor.get("top_factors", [])) or "-"}</p>
<p class="sub">走弱: {", ".join(factor.get("weak_factors", [])) or "-"}</p>
<p class="sub">有效因子池（真实IC/OOS，核心+待选）: {factor_quality_note}</p></div>

<div class="card"><h2>📊 量化全景（因子/ML/回测）</h2>
{_quant_html}
</div>
<div class="card"><h2>🤖 ML 全池扫描（强势/回避）</h2>
{_mlscan_html}
</div>
<div class="card"><h2>🤖 ML 预测（{len(ml)}）</h2>
<ul>{''.join(f'<li>{p.get("symbol","")}: {str(p.get("pred",""))[:80]}</li>' for p in ml) or "<li class=sub>ML 未就绪</li>"}</ul></div>

<div class="card"><h2>📈 技术分析（{len(technical)}指数）</h2>
{_tbl(["指数", "收盘", "技术分", "趋势", "MA5", "MA10", "MA20", "MA60", "RSI", "量能", "支撑/压力", "MACD"], tech_rows)}</div>

<div class="card"><h2>💬 互联舆情（股吧/微博/百度）</h2>
{_social_html}
</div>
<div class="card"><h2>📐 缠论分析（自动笔段+中枢+精确剧本+RAG）</h2>
{_chan_big_html}
<div id="chanK" style="width:100%;height:320px"></div>
{_chan_html}
</div>
<div class="card"><h2>📐 大盘趋势（波浪理论·多指数总览）</h2>
{_wave_all_html}
{_wave_html}
</div>
<div class="card"><h2>🛠️ 六引擎深挖（海外宏观/RS/日历/透镜）</h2>
{_eng_html}
{_vol_html}
</div>
<div class="card"><h2>🧠 引擎真实产出（深改造 {_de_ok}个·真内容）</h2>
{_de_html}
</div>
<div class="card"><h2>🧠 全引擎矩阵（{_ae_ok}运行OK / {_ae_total}个）</h2>
{_ae_html}
</div>
<div class="card"><h2>🧠 引擎融合共识（引擎→决策）</h2>
{_ef_html}
</div>
<div class="card"><h2>📈 预测验证闭环（进化回填）</h2>
{_pl_html}
</div>
<div class="card"><h2>🌐 市场基座全景（商品/利率/宏观/资金/两融/涨停）</h2>
<div id="cmChart" style="width:100%;height:180px"></div>
{_mpano_html}
</div>
<div class="card"><h2>🌍 海外市场（{len(overseas)}）</h2>
{_overseas_alert}
<div id="ovChart" style="width:100%;height:160px"></div>
{_tbl(["标的", "价格", "币", "涨跌", "方向"], ov_rows)}
<p class="sub">来源: {blk("overseas").get("_source", "-")}</p></div>

<div class="card"><h2>🗄️ 数据基座融合（八域）</h2>
<p><b>宏观</b>: {"; ".join(f"{i.get('name')}={i.get('value')}" for i in dbb.get('macro',{}).get('items',[])[:8]) or "-"}</p>
<p><b>热榜</b>: 涨{dbb.get('hot',{}).get('up_n','-')}跌{dbb.get('hot',{}).get('down_n','-')} ({len(dbb.get('hot',{}).get('top',[]))}只)</p>
<p><b>机构新进</b>: {inst_html}</p>
<p><b>强行业</b>: {ind_html}</p>
{''.join(f'<p><b>{lbl}</b>: {dbb.get(k,{}).get("note","-")}</p>' for k, lbl in [("north_margin", "资金基座"), ("style", "风格"), ("event", "事件域"), ("concept", "概念周期")] if dbb.get(k,{}).get("note"))}</div>

<div class="card"><h2>🗃️ 基座新鲜度门控（{len(scan)}基座 · 健康度{freshness.get("score","-")}/100）</h2>
<div id="gateChart" style="width:100%;height:150px"></div>
{_tbl(["基座", "周期", "最新", "滞后", "评级"], gate_rows)}</div>

<div class="card"><h2>🧭 宏观否决层</h2>{veto_html}</div>

<div class="card"><h2>🎛️ 策略结论（回测+因子对当前决策的启示）</h2>{_strategy_conclusion_html}</div>
<div class="card"><h2>🧬 因子有效性与方向校准（选股权重）</h2>{_factor_effectiveness_html}</div>

<div class="card"><h2>🧬 形态信号（{len(patterns)}）</h2>
{_tbl(["信号"], pattern_rows)}</div>

{_CHART_HTML}
<div class="meta" style="margin-top:12px">⚠️ 免责声明：本报告由量化融合链自动生成，数据可能存在延迟或缺失，仅供研究参考，不构成投资建议。</div>
</div></body></html>'''


def _atomic_write(path: Path, content: str) -> None:
    """Write a complete artifact, then publish it with one filesystem replace."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _report_run_id(review: dict) -> str:
    return str(os.environ.get("QUANT_RUN_ID") or review.get("run_id") or
               f"html-{datetime.now(CST).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}")


def main(date=None):
    review = _load_review(date)
    if not review:
        print("无复盘 json"); sys.exit(1)
    date = review.get("date", "unknown")
    day_dir = OUT_DIR / date
    day_dir.mkdir(parents=True, exist_ok=True)
    html = build_html(review, report_dir=day_dir)
    html = _finalize_report(html, date=date, report_root=OUT_DIR)
    p = day_dir / f"综合详细研报_{date}.html"
    run_id = _report_run_id(review)
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    _atomic_write(p, html)
    metadata = {
        "schema": "quant-report-artifact/v1",
        **release_metadata(),
        "report_date": date,
        "run_id": run_id,
        "sha256": digest,
        "completed_at": datetime.now(CST).isoformat(),
        "html": p.name,
    }
    _atomic_write(p.with_suffix(p.suffix + ".meta.json"), json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"详细研报: {p} ({len(html)}B) run_id={run_id}")
    return p


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    main(date)
