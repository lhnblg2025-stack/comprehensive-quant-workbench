#!/usr/bin/env python3
"""build_html_daily_report.py — 多因子复盘 HTML 日报 (V1)

用户要求:
- 日报保存到共享文件夹-研报共享，做成 HTML 形式
- 利用好现有的丰富的国内外数据源，做好个股的分析复盘
- 结合众多因子考虑，哪怕影响因子权重很小，但是出现了极端，也要提醒

输出: ~/Desktop/研报共享/A股多因子复盘日报_YYYY-MM-DD.html（不可写时回退到workspace/研究报告）
数据来源: data_warehouse（kline/valuation/financial/events/market/macro/oneoff）
         + 现有 a_share_daily_report 口径（宽度/热股/两融/板块）
"""

from __future__ import annotations
import logging

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2026-08-21 审计: Windows GBK 控制台无法输出 emoji 导致末尾 print 崩溃（html 已生成但 exit!=0）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(ROOT / "scripts"))
from factor_extreme_alert import analyze_batch, load_kline, load_valuation, WH
from html_report_generator import _load_vision_insights, render_vision_insights_section

# 与 html_report_generator 共用桌面研报共享根；不依赖云端专有的 /mnt/hgfs 路径。
_SHARED = Path.home() / "Desktop" / "研报共享"


def _report_root() -> Path:
    """研报共享根目录：共享目录优先，不可写时退到 workspace/研究报告（与 html_report_generator 一致）。"""
    try:
        _SHARED.mkdir(parents=True, exist_ok=True)
        _t = _SHARED / ".wtest"; _t.write_text("ok", encoding="utf-8"); _t.unlink()
        return _SHARED
    except OSError:
        return ROOT / "研究报告"


REPORT_DIR = _report_root()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 复盘股票池：持仓 + 关注 + 涨停池新面孔
WATCH_SYMBOLS = ["002714", "601899", "600519", "000858", "300750", "002594", "300059", "600036", "601318", "000333"]
WATCH_NAMES = {
    "002714": "牧原股份", "601899": "紫金矿业", "600519": "贵州茅台", "000858": "五粮液",
    "300750": "宁德时代", "002594": "比亚迪", "300059": "东方财富", "600036": "招商银行",
    "601318": "中国平安", "000333": "美的集团",
}


def load_name_map() -> dict[str, str]:
    """代码->名称 映射"""
    nm = {}
    try:
        p = ROOT / "quant_system" / "stock_name_map.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            # 支持 {name: code} 或 {code: name} 两种结构
            if isinstance(data, dict):
                for k, v in data.items():
                    if str(k).isdigit():
                        nm[str(k).zfill(6)] = str(v)
                    elif str(v).isdigit():
                        nm[str(v).zfill(6)] = str(k)
    except Exception as e:
        logging.getLogger(__name__).error(f"[build_html_daily_report] 操作失败: {e}", exc_info=True)
    nm.update(WATCH_NAMES)
    return nm


def recent_zt_stocks(days: int = 5, as_of: str | None = None) -> list[tuple[str, str, int]]:
    """近期涨停池股票，严格限制不晚于报告日，排除 ST。"""
    zt_dir = WH / "market" / "zt_pool"
    out: dict[str, tuple[str, int]] = {}
    if not zt_dir.is_dir():
        return []
    files = []
    for f in sorted(zt_dir.glob("*.parquet")):
        match = re.search(r"20\d{2}-\d{2}-\d{2}", f.stem)
        if as_of and match and match.group(0) > as_of:
            continue
        files.append(f)
    files = files[-days:]
    for f in files:
        try:
            df = pd.read_parquet(f)
            cc = "代码" if "代码" in df.columns else ("证券代码" if "证券代码" in df.columns else None)
            nc = "名称" if "名称" in df.columns else ("证券简称" if "证券简称" in df.columns else None)
            if not cc:
                continue
            for _, row in df.iterrows():
                code = str(row[cc]).zfill(6)
                name = str(row.get(nc, code)) if nc else code
                if "ST" in name.upper():
                    continue
                if code.startswith(("8", "4", "92")):  # 北交所
                    continue
                if code in out:
                    out[code] = (out[code][0], out[code][1] + 1)
                else:
                    out[code] = (name, 1)
        except Exception as e:
            logging.getLogger(__name__).error(f"[build_html_daily_report] 操作失败: {e}", exc_info=True)
            continue
    return [(c, n[0], n[1]) for c, n in out.items()]


def load_macro_latest() -> dict:
    """宏观数据最新值"""
    out = {}
    try:
        m = WH / "macro"
        for f, name, col in (
            ("pmi.parquet", "制造业PMI", None),
            ("cpi_yearly.parquet", "CPI", None),
            ("shibor_all.parquet", "SHIBOR隔夜", None),
            ("lpr.parquet", "LPR1Y", None),
        ):
            p = m / f
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    last = df.iloc[0] if str(df.iloc[0].iloc[0]).startswith("202") else df.iloc[-1]
                    v = last.iloc[1] if len(last) > 1 else last.iloc[0]
                    out[name] = {"date": str(last.iloc[0])[:10], "value": round(float(v), 2)}
    except Exception as e:
        logging.getLogger(__name__).error(f"[build_html_daily_report] 操作失败: {e}", exc_info=True)
    return out


def factor_table_html(results: list[dict], name_map: dict) -> str:
    """多因子极端扫描结果表格"""
    rows = []
    for r in results:
        if not r.get("ok"):
            continue
        name = name_map.get(r["symbol"], r["symbol"])
        level = r.get("level", "正常")
        badge = {"危险": "danger", "警惕": "warn", "正常": "ok"}.get(level, "ok")
        ext = r.get("extremes", [])
        ext_text = "；".join(e["message"] for e in ext) if ext else "—"
        rows.append(
            f"<tr><td>{name}<br><small>{r['symbol']}</small></td>"
            f"<td><span class='badge {badge}'>{level}</span></td>"
            f"<td>{r.get('score', 0)}</td>"
            f"<td>{r.get('n_extremes', 0)}</td>"
            f"<td class='ext-text'>{ext_text}</td></tr>"
        )
    if not rows:
        return "<p>暂无扫描结果</p>"
    return (
        "<table><thead><tr><th>股票</th><th>级别</th><th>综合分</th><th>极端数</th><th>极端因子明细</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_html(date_str: str) -> str:
    """生成完整 HTML 日报"""
    name_map = load_name_map()

    # 1. 多因子扫描（自选股 + 近期涨停股）
    scan_symbols = list(dict.fromkeys(WATCH_SYMBOLS))
    try:
        zt = recent_zt_stocks(5, as_of=date_str)
        for c, n, cnt in zt[:30]:
            if c not in scan_symbols:
                scan_symbols.append(c)
                name_map.setdefault(c, n)
    except Exception as e:
        logging.getLogger(__name__).error(f"[build_html_daily_report] 操作失败: {e}", exc_info=True)
    # 过滤 ST/退市/北交所（用名称映射判断）
    def _ok_symbol(s: str) -> bool:
        nm = name_map.get(s, "")
        if "ST" in str(nm).upper():
            return False
        if s.startswith(("8", "4", "92")):
            return False
        return True
    scan_symbols = [s for s in scan_symbols if _ok_symbol(s)]
    results = analyze_batch(scan_symbols)

    # 2. 宏观
    macro = load_macro_latest()

    # 3. 涨停池统计
    zt_5d = recent_zt_stocks(5, as_of=date_str)
    zt_5d_sorted = sorted(zt_5d, key=lambda x: -x[2])[:20]

    # 4. 市场宽度（从最近日报提取或直接读 market）
    breadth_html = ""
    try:
        daily_dir = ROOT / "generated" / "a_share_data"
        if daily_dir.is_dir():
            md = sorted(daily_dir.glob("*日报*.md"))
            if not md:
                md = sorted(daily_dir.glob("*.md"))
            if md:
                txt = md[-1].read_text(encoding="utf-8")
                # 提取核心结论段
                if "## 1. 核心结论" in txt:
                    seg = txt.split("## 1. 核心结论")[1].split("## 2.")[0]
                    breadth_html = "<div class='panel'><h3>📋 市场核心结论（最新日报）</h3><pre class='pre-wrap'>" + seg.strip()[:1500] + "</pre></div>"
    except Exception as e:
        logging.getLogger(__name__).error(f"[build_html_daily_report] 操作失败: {e}", exc_info=True)

    # 5. 生成 HTML
    dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = ""
    for r in results:
        if not r.get("ok"):
            continue
        name = name_map.get(r["symbol"], r["symbol"])
        level = r.get("level", "正常")
        badge = {"危险": "danger", "警惕": "warn", "正常": "ok"}.get(level, "ok")
        ext = r.get("extremes", [])
        ext_text = "；".join(e["message"] for e in ext) if ext else "—"
        rows += (
            f"<tr><td>{name}<br><small class='dim'>{r['symbol']}</small></td>"
            f"<td><span class='badge {badge}'>{level}</span></td>"
            f"<td class='score'>{r.get('score', 0):.0f}</td>"
            f"<td>{r.get('n_extremes', 0)}</td>"
            f"<td class='ext-text'>{ext_text}</td></tr>"
        )

    macro_html = ""
    for k, v in macro.items():
        macro_html += f"<div class='macro-item'><span>{k}</span><b>{v['value']}</b><small>{v['date']}</small></div>"

    zt_html = ""
    for c, n, cnt in zt_5d_sorted:
        zt_html += f"<span class='chip'>{n} <small>{c}·{cnt}次</small></span>"

    # 研报图表解析章节：读同一 vision_analysis.jsonl，从同一渲染函数产出（口径一致）
    _vis_records, _vis_failed = _load_vision_insights(date_str)
    vis_section = render_vision_insights_section(_vis_records, _vis_failed, report_dir=None, prefix="ima2") or '<div class="dim">研报图片解析数据暂缺（待 image_insights 运行）</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股多因子复盘日报 {date_str}</title>
<style>
:root {{
  --bg: #0f1115; --card: #171a21; --border: #262b36; --text: #d7dce4;
  --dim: #7a8494; --red: #e5484d; --orange: #f5a524; --green: #46a758; --blue: #4c8dff;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; padding: 24px; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
h2 {{ font-size: 17px; margin: 24px 0 12px; padding-left: 10px; border-left: 4px solid var(--blue); }}
.sub {{ color: var(--dim); font-size: 12px; margin-bottom: 20px; }}
.panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
.panel h3 {{ font-size: 14px; margin-bottom: 10px; color: #aeb7c5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
th {{ color: var(--dim); font-weight: 500; font-size: 12px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
.badge.danger {{ background: rgba(229,72,77,.15); color: var(--red); }}
.badge.warn {{ background: rgba(245,165,36,.15); color: var(--orange); }}
.badge.ok {{ background: rgba(70,167,88,.15); color: var(--green); }}
.ext-text {{ color: #c8b98a; font-size: 12px; line-height: 1.6; }}
.score {{ font-weight: 700; }}
.dim {{ color: var(--dim); }}
.pre-wrap {{ white-space: pre-wrap; font-family: inherit; font-size: 13px; line-height: 1.7; color: #b8c0cc; }}
.macro-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.macro-item {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 150px; }}
.macro-item span {{ display: block; font-size: 12px; color: var(--dim); }}
.macro-item b {{ font-size: 20px; }}
.macro-item small {{ display: block; color: var(--dim); font-size: 11px; margin-top: 2px; }}
.chip {{ display: inline-block; background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 5px 12px; margin: 3px; font-size: 12px; }}
.chip small {{ color: var(--dim); }}
.footer {{ margin-top: 28px; color: var(--dim); font-size: 11px; border-top: 1px solid var(--border); padding-top: 12px; }}
.danger-row {{ background: rgba(229,72,77,.06); }}
</style>
</head>
<body>
<h1>🐚 A股多因子复盘日报</h1>
<div class="sub">{date_str} · 生成于 {dt} · 数据源: data_warehouse（K线/估值/财务/事件/市场/宏观）+ akshare/baostock 多源</div>

{h2_html("1. 宏观速览")}
<div class="panel"><div class="macro-grid">{macro_html or "<div class='dim'>宏观数据缺失</div>"}</div></div>

{h2_html("2. 市场核心结论（来自每日日报口径）")}
{breadth_html or '<div class="panel"><div class="dim">日报数据未生成，请先运行 a_share_daily_report.py</div></div>'}

{h2_html("3. 个股多因子扫描（权重小但极端也提醒）")}
<div class="panel">
<div style="font-size:12px;color:var(--dim);margin-bottom:10px;">5大类29因子：技术面（乖离/MACD/RSI/KDJ/CCI/BOLL/量比/换手/ATR/RSRS/新高新低/跌幅）、估值面（PE/PB/PS分位）、财务面（ROE/毛利率/增速/负债率）、情绪面（涨停/连板）。任一因子达极端分位（95%/5%或z≥2.5）即提醒，不因权重小而忽略。</div>
<table>
<thead><tr><th>股票</th><th>级别</th><th>综合分</th><th>极端数</th><th>极端因子明细</th></tr></thead>
<tbody>{rows or "<tr><td colspan='5' class='dim'>无扫描结果</td></tr>"}</tbody>
</table>
</div>

{h2_html("4. 近5日涨停池（赚钱效应风向标）")}
<div class="panel">{zt_html or '<div class="dim">涨停池数据暂缺</div>'}</div>

{h2_html("5. 研报图表解析与资金/趋势跟踪")}
<div class="panel">{vis_section}</div>

<div class="footer">
⚠️ 本报告由多因子规则引擎自动生成，仅供研究参考，不构成投资建议。极端因子提醒≠卖出信号，需结合基本面与市场环境综合判断。<br>
数据截至 {date_str}，若部分板块数据未更新（云服务器抓取中），相关因子自动跳过并在明细中标注。
</div>
</body>
</html>"""


def h2_html(title: str) -> str:
    return f"<h2>{title}</h2>"


def main() -> int:
    date_str = datetime.now().strftime("%Y-%m-%d")
    html = build_html(date_str)
    day_dir = REPORT_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    out = day_dir / f"A股多因子复盘日报_{date_str}.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ HTML 日报已生成: {out} ({out.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
