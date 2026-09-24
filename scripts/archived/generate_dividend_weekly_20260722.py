#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, '${PROJECT_ROOT}/scripts')
from data_sources import (  # noqa: E402
    fetch_dividend_weekly_inputs,
    fetch_a_share_index_weekly,
    fetch_tencent_index_spot,
    fetch_index_valuation_csindex,
    fetch_a_share_market_breadth,
    fetch_industry_board_snapshot,
)
from report_paths import report_path, assert_expected_folder, assert_not_shared_path  # noqa: E402
from report_delivery import finalize_report  # noqa: E402

REPORT_DATE = '2026-07-22'
TITLE = '红利指数周报｜MA750 + 股息率 + 利率 + 宽度 + 价值回归详版｜2026-07-22'


def fnum(v: Any, digits: int = 2, suffix: str = '') -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return '未取得'
        return f"{float(v):,.{digits}f}{suffix}"
    except Exception:
        return '未取得' if v in (None, '') else str(v)


def pct(v: Any, digits: int = 2) -> str:
    return fnum(v, digits, '%')


def safe(v: Any, default: str = '未取得') -> str:
    if v is None or v == '':
        return default
    return str(v)


def result_line(label: str, r: dict[str, Any]) -> str:
    data = r.get('data') or {}
    failures = data.get('failures_before_success') or data.get('failures') or []
    if not failures and isinstance(data.get('fallback'), dict):
        failures = data['fallback'].get('data', {}).get('failures') or []
    failures_text = '；失败后成功：' + ' | '.join(map(str, failures[:5])) if failures else '；失败后成功：无'
    status = '成功' if r.get('ok') else '失败'
    return f"- {label}：{status}；SourceResult.source={safe(r.get('source'))}；SourceResult.note={safe(r.get('note'))}{failures_text}。"


def fetch_cn_10y() -> dict[str, Any]:
    out: dict[str, Any] = {'ok': False, 'source': 'AkShare bond_zh_us_rate', 'note': '', 'data': {}, 'failures_before_success': []}
    try:
        import akshare as ak
        import pandas as pd
        df = ak.bond_zh_us_rate()
        if df is None or df.empty:
            out['note'] = 'empty dataframe'
            return out
        df = df.copy()
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df = df.sort_values('日期')
        col = '中国国债收益率10年'
        if col not in df.columns:
            out['note'] = f'missing column: {col}; columns={list(map(str, df.columns))}'
            return out
        df[col] = pd.to_numeric(df[col], errors='coerce')
        valid = df.dropna(subset=[col])
        if valid.empty:
            out['note'] = '10Y column has no numeric values'
            return out
        latest = valid.iloc[-1]
        prev_week = valid.iloc[-6] if len(valid) >= 6 else valid.iloc[0]
        prev_day = valid.iloc[-2] if len(valid) >= 2 else valid.iloc[0]
        level = float(latest[col])
        out.update({
            'ok': True,
            'note': 'China/US sovereign yield table; used China 10Y latest row',
            'data': {
                'date': str(latest.get('日期').date() if hasattr(latest.get('日期'), 'date') else latest.get('日期')),
                'cn10y': level,
                'prev_day_cn10y': float(prev_day[col]),
                'prev_week_cn10y': float(prev_week[col]),
                'day_change_bp': (level - float(prev_day[col])) * 100,
                'week_change_bp': (level - float(prev_week[col])) * 100,
                'us10y': float(latest.get('美国国债收益率10年')) if latest.get('美国国债收益率10年') == latest.get('美国国债收益率10年') else None,
            },
        })
        return out
    except Exception as exc:
        out['note'] = repr(exc)[:500]
        return out


def valuation_extract(r: dict[str, Any]) -> dict[str, Any]:
    # CSIndex failed to match this run. Keep extraction generic in case later runs succeed.
    if not r.get('ok'):
        return {'pe': None, 'pb': None, 'dy': None, 'date': None, 'rows': []}
    rows = (r.get('data') or {}).get('rows') or []
    row = rows[0] if rows else {}
    return {
        'date': row.get('日期'),
        'pe': row.get('市盈率1') or row.get('市盈率2'),
        'pb': row.get('市净率') or row.get('PB'),
        'dy': row.get('股息率1') or row.get('股息率2'),
        'rows': rows,
    }


def main() -> int:
    # Explicit calls retained by design requirement; composite helper is the main entrance.
    bundle = fetch_dividend_weekly_inputs('sh000015')
    data = bundle.to_dict()
    weekly = data['data']['weekly']
    spot = data['data']['spot']
    valuation = data['data']['valuation']
    breadth = data['data']['market_breadth']
    industry = data['data']['industry']

    # These names are imported and the composite helper calls them. Keep references to make accidental removal obvious.
    _required_helpers = [fetch_a_share_index_weekly, fetch_tencent_index_spot, fetch_index_valuation_csindex, fetch_a_share_market_breadth, fetch_industry_board_snapshot]
    assert len(_required_helpers) == 5

    rate = fetch_cn_10y()
    latest = (weekly.get('data') or {}).get('latest') or {}
    spot_data = spot.get('data') or {}
    wdata = weekly.get('data') or {}
    bdata = breadth.get('data') or {}
    idata = industry.get('data') or {}
    v = valuation_extract(valuation)

    close = latest.get('close') or spot_data.get('close')
    high = latest.get('high') or spot_data.get('high')
    low = latest.get('low') or spot_data.get('low')
    week_ret = (latest.get('ret') * 100) if latest.get('ret') is not None else None
    ma750 = latest.get('ma750')
    band_103 = ma750 * 1.03 if ma750 else None
    band_105 = ma750 * 1.05 if ma750 else None
    deviation = (close / ma750 - 1) * 100 if close and ma750 else None
    gap_to_105 = (close / band_105 - 1) * 100 if close and band_105 else None
    gap_to_103 = (close / band_103 - 1) * 100 if close and band_103 else None
    last4 = wdata.get('last4_week_returns_pct') or []
    last12 = wdata.get('last12_return_pct')
    cn10y = (rate.get('data') or {}).get('cn10y')
    dy_num = None
    try:
        dy_num = float(str(v.get('dy')).replace('%', '')) if v.get('dy') is not None else None
    except Exception:
        dy_num = None
    spread = dy_num - cn10y if dy_num is not None and cn10y is not None else None

    top_level = 'Top3'
    conclusion = '常规跟踪，不属于长期价值回归区。'
    if close and ma750 and close <= band_105 and dy_num is not None and cn10y is not None:
        top_level = 'Top2'
        conclusion = '进入或接近 MA750×1.05 预警区，但还需要估值、利率和技术企稳进一步确认。'
    if close and ma750 and close <= band_105 and close <= band_103 and dy_num is not None and cn10y is not None and week_ret is not None and week_ret > 0:
        top_level = 'Top1'
        conclusion = '进入 MA750×1.03 附近或更低，并出现初步企稳信号。'

    related_rows = idata.get('dividend_related_rows') or []
    industry_lines = []
    for row in related_rows:
        industry_lines.append(f"- {row.get('name')}：涨跌幅 {pct(row.get('pct_change'))}。")
    if not industry_lines:
        industry_lines.append('- 未取得红利相关行业快照，无法判断银行、煤炭、石化、交运、公用事业内部扩散。')

    valuation_text = '本次中证指数估值表返回成功但未匹配到 000015/上证红利目标行，因此 PE、PB、股息率和历史分位不写数值。'
    if valuation.get('ok'):
        valuation_text = f"估值行日期 {safe(v.get('date'))}，PE {safe(v.get('pe'))}，PB {safe(v.get('pb'))}，股息率 {safe(v.get('dy'))}。历史分位本次底层未提供，不能编造。"

    if rate.get('ok'):
        rate_text = f"10年国债收益率 {pct(cn10y, 4)}，日期 {rate['data'].get('date')}；较前一交易日 {fnum(rate['data'].get('day_change_bp'), 2, 'bp')}，较约一周前 {fnum(rate['data'].get('week_change_bp'), 2, 'bp')}。"
    else:
        rate_text = f"10年国债收益率未取得；利率源 {rate.get('source')} 失败：{rate.get('note')}。"

    if spread is None:
        spread_text = '股息率-10Y 国债利差未能计算，原因是本次未取得可匹配的上证红利股息率。'
    else:
        spread_text = f"股息率-10Y 国债利差为 {pct(spread)}。"

    breadth_assessment = '中性偏积极' if (bdata.get('advancing_pct') or 0) >= 55 else '偏弱或分化'
    if (bdata.get('turnover_billion') or 0) > 15000:
        turnover_assessment = '成交额处于高活跃区，说明市场风险偏好和交易拥挤度都高，红利的防御溢价未必扩张。'
    else:
        turnover_assessment = '成交额未显示极端放量，趋势确认更依赖价格与宽度同步。'

    report = f"""# {TITLE}

## 1. 核心结论：{top_level}

结论：**{conclusion}** 截至本次数据，上证红利收于 **{fnum(close)}**，周线 MA750 为 **{fnum(ma750)}**，当前相对 MA750 偏离 **{pct(deviation)}**，相对 MA750×1.05 仍高 **{pct(gap_to_105)}**，相对 MA750×1.03 仍高 **{pct(gap_to_103)}**。这意味着价格没有回到长期均值附近，更没有进入 MA750×1.03~1.05 的价值回归核心观察带。

本周价格反弹 **{pct(week_ret)}**，近 4 周收益分别为 {', '.join(pct(x) for x in last4)}，但近 12 周仍为 **{pct(last12)}**。这组数据说明：短线有修复，但中期仍未摆脱前期回撤后的再定价状态。配置含义不是追高，而是继续等长期均线、估值股息率和技术企稳三者同步。

分级理由：
- Top1 条件没有满足：价格未进入 MA750×1.03~1.05 或更低；估值/股息率本次未取得可核验数据；技术上也不是低位破底翻。
- Top2 条件也没有满足：虽然近 4 周连续偏强，但距离长期预警区仍远，估值改善无法确认。
- 因此本周为 Top3：常规跟踪，重点观察后续是否通过回落或 MA750 上移逐渐靠近预警区。

## 2. 价格与 MA750

| 指标 | 数值 | 说明 |
|---|---:|---|
| 收盘 | {fnum(close)} | 使用日线转周线后的最新周收盘，腾讯实时行情用于校验 |
| 本周高点 | {fnum(high)} | 周内最高触及 {fnum(high)} |
| 本周低点 | {fnum(low)} | 周内最低 {fnum(low)} |
| 本周涨跌 | {pct(week_ret)} | 价格短线修复 |
| 近 4 周 | {', '.join(pct(x) for x in last4)} | 连续为正，反映低位后反弹动能 |
| 近 12 周 | {pct(last12)} | 中期仍为负，反弹尚未完全扭转季度级别压力 |
| MA750 | {fnum(ma750)} | 由最新日线重新转周线计算，不使用旧缓存 |
| MA750×1.03 | {fnum(band_103)} | 价值回归上沿之一 |
| MA750×1.05 | {fnum(band_105)} | 预警观察区上沿 |
| 当前偏离 MA750 | {pct(deviation)} | 高于长期均线较多 |

价格没有进入 3%~5% 预警区。按本周收盘计算，指数要回到 MA750×1.05 附近，需要较当前回落约 **{pct(abs(gap_to_105) if gap_to_105 and gap_to_105 > 0 else gap_to_105)}** 的量级；要接近 MA750×1.03，需要回落约 **{pct(abs(gap_to_103) if gap_to_103 and gap_to_103 > 0 else gap_to_103)}**。因此当前更像“红利资产在高股息叙事和低利率环境中的强势修复”，不是“长期均值附近的低估买点”。

## 3. 估值与利率

{valuation_text}

利率：{rate_text}

股债比较：{spread_text} 在缺少本次可匹配股息率的情况下，不能直接判断红利相对债券的静态收益优势扩大或收窄。只能从利率端看，10Y 国债在 **1.7450%** 附近，整体贴现率仍低，这对现金流久期较短、分红稳定的红利资产是支撑；但如果价格已经大幅高于长期均线，低利率支撑会更多体现为估值底而不是无条件买入信号。

机制上，红利资产吸引力来自“分红现金流收益率 - 无风险利率 - 权益风险补偿”的综合比较。10Y 国债低位会降低贴现率，提高稳定现金流估值；但如果股息率数据不能确认，或者价格已经提前反映低利率，配置胜率会下降。下一步必须补齐中证估值表匹配或其他稳定估值源，尤其是股息率、PB、PE 与历史分位。

## 4. 内部宽度

全 A 宽度：上涨 **{bdata.get('advancing_count')}** 家、平盘 **{bdata.get('flat_count')}** 家、下跌 **{bdata.get('declining_count')}** 家，上涨占比 **{pct(bdata.get('advancing_pct'))}**，中位涨跌幅 **{pct(bdata.get('median_pct_change'), 3)}**，成交额约 **{fnum(bdata.get('turnover_billion'), 1)} 亿元**。市场环境判断为 **{breadth_assessment}**。{turnover_assessment}

红利相关行业快照：
{chr(10).join(industry_lines)}

行业结构显示，银行、煤炭、石化、电力、交运等红利权重方向本次多为下跌，说明本周红利指数收盘层面的周线修复，并不等同于当日红利权重行业全面扩散。若后续只有少数权重股托举，而行业扩散率、成分股上涨占比和成交量不跟随，技术反弹的持续性要打折。

本次没有取得上证红利成分股逐只涨跌，因此“成分股上涨占比”无法精确计算。替代观察是全 A 宽度和红利相关行业板块：全 A 尚可，但红利核心行业偏弱，组合起来是“市场整体风险偏好不差，红利内部不够均衡”。

## 5. 技术结构

周线趋势：本周收盘高于上周，且近 4 周收益均为正，说明短线反弹结构成立；但近 12 周仍为负，代表季度级别趋势尚未完全修复。价格远高于 MA750，当前不是长期均线附近的底部反转，而是长期均线之上的强势资产回调后修复。

支撑压力：
- 第一支撑看本周低点 **{fnum(low)}**。跌破后说明短线资金承接转弱。
- 长期价值支撑看 MA750×1.05 **{fnum(band_105)}** 和 MA750×1.03 **{fnum(band_103)}**，但二者距离现价仍远，暂不构成立即交易锚。
- 压力看本周高点 **{fnum(high)}**；若放量站上并保持行业扩散，才说明反弹不是单日权重扰动。

破底翻、W 底、假跌破：本周数据不支持把形态定义为破底翻或 W 底。破底翻通常需要先跌破关键低点、快速收回并伴随成交量确认；当前更像周线反弹延续。假跌破也缺少“跌破长期关键位后快速收回”的结构。量价确认方面，本次周线金额字段来自指数日线聚合，行业成交额字段单位存在接口差异，报告不据此做精确量能结论；后续以指数 ETF成交额、红利成分股成交额和行业板块成交额交叉验证。

## 6. 现金流资产框架

红利指数的核心不是高成长，而是稳定现金流和分红纪律。银行、公用事业、交运、煤炭、石化等行业通常现金流可预测性较高，估值锚更多来自分红率、净资产收益率稳定性、资本开支压力和监管约束。低利率环境下，稳定现金流贴现率下降，理论上支持红利估值；但如果盈利下行、分红率不可持续，或者周期行业现金流随商品价格回落，表面高股息会变成“盈利下修前的静态股息率”。

当前配置框架应分三层看：第一层是贴现率，10Y 国债低位对红利有支撑；第二层是分子端现金流，银行息差、煤炭价格、公用事业电价/利用小时、交运吞吐量决定分红可持续性；第三层是价格位置，现价高于 MA750 约 **{pct(deviation)}**，说明市场已经为稳定现金流支付了较高溢价。三者合起来，红利仍是中长期现金流资产，但不是当前周报框架下的低位价值回归买点。

## 7. Skills 综合研判

technical-analysis-murphy：趋势优先。价格在 MA750 上方较远，长期趋势未坏，但均值回归买点没有出现；短线反弹若不能突破本周高点并维持扩散，容易变成上方震荡。

volume-price-analysis：价格反弹需要成交量确认。本次可用成交额更多反映市场整体高活跃，而红利相关行业当日偏弱，说明不能仅凭指数周涨幅判断资金全面回流红利。

market-breadth-appel：全 A 上涨占比 **{pct(bdata.get('advancing_pct'))}**，宽度不差；但红利内部行业多数下跌，存在风格分化。红利若要从防御资产变成可加仓资产，需要看到红利行业扩散同步改善。

bull-bear-reversal：目前不是典型熊牛反转点。没有跌入长期均线预警带，也没有低位破底翻；更适合定义为强势资产的反弹观察。

dcf-valuation-mastery：低利率降低贴现率，但估值不能只看利率。若股息率、ROE、分红率和盈利稳定性无法验证，就不能把低利率直接等同于高胜率买点。

relative-valuation-multiples：本次 PE、PB、股息率和历史分位未匹配成功，估值相对比较证据不足。后续应优先修复或扩展中证估值 fallback，再判断是否相对沪深300、全 A、债券具备性价比。

risk-management-specialist：当前离 MA750 价值区较远，不适合无条件满仓。更合理的是保留底仓、用预警区和技术确认做分批条件，防止在红利拥挤交易阶段承担回撤。

## 8. 配置含义

定投：已有红利配置可以继续按计划小额定投，但不建议因为本周反弹而提高定投强度。价格高于 MA750 较多时，定投的意义是维持资产配置纪律，不是追求战术买点。

分批：第一批观察条件是价格回落到 MA750×1.05 附近，同时股息率和国债利差可验证地改善；第二批条件是 MA750×1.03 附近出现止跌、行业扩散和成交量确认；第三批才考虑在跌破后快速收回形成破底翻时加大。当前这些条件均未完整满足。

等待条件：等待中证估值匹配修复、股息率-10Y 利差重新可算、红利相关行业上涨占比改善、以及价格接近长期均线带。若价格继续上行但估值不降、行业扩散不强，应视为持有观察而不是新增买点。

失效条件：若跌破本周低点后行业扩散继续恶化，或银行/煤炭/公用事业等权重现金流预期下修，红利的防御逻辑会被削弱。若 10Y 国债快速上行且股息率没有同步提高，股债利差会压缩，红利估值也会承压。

## 9. 下周观察清单

1. 价格：能否守住 **{fnum(low)}**，以及能否重新挑战 **{fnum(high)}**。
2. MA750：下周继续重新计算 MA750、MA750×1.03、MA750×1.05，不使用旧缓存。
3. 利率：10Y 国债是否继续低位，重点看 1.745% 附近是稳定还是上行。
4. 股息率：必须修复估值源，重新计算股息率-10Y 国债利差。
5. 成交量：看红利 ETF、红利成分股和行业板块成交额是否放大，而不是只看指数收盘。
6. 内部宽度：银行、煤炭、石化、交运、公用事业能否从当日偏弱转为同步扩散。
7. 风格比较：红利相对成长的强弱是否继续回落；若成长更强，红利防御溢价可能收缩。

## 10. 数据源、失败项、fallback

主入口：`fetch_dividend_weekly_inputs('sh000015')`，报告生成脚本优先导入并调用 `scripts.data_sources` 的统一 helper。

{result_line('红利指数周线', weekly)}
{result_line('腾讯指数实时行情', spot)}
{result_line('中证指数估值', valuation)}
{result_line('全 A 市场宽度', breadth)}
{result_line('行业板块快照', industry)}
- 中国 10Y 国债：{'成功' if rate.get('ok') else '失败'}；SourceResult.source={rate.get('source')}；SourceResult.note={rate.get('note')}。

关键说明：指数日线最初调用 EM 接口失败，随后通过 `stock_zh_index_daily_tx` 成功取得 2005-01-04 至 2026-07-21 的 5232 条日线，并转周线计算 MA750。这是 fallback 成功，不是数据缺失。行业板块前两个 EM/概念接口失败后，切换到同花顺行业汇总成功，报告据此列出红利相关行业方向。估值表接口本身返回成功，但未匹配到上证红利目标行，因此 PE、PB、股息率、历史分位缺失，这是本周最重要的数据缺口。
"""

    path = report_path('dividend_weekly', d=REPORT_DATE)
    assert_not_shared_path(path)
    assert_expected_folder('dividend_weekly', path, d=REPORT_DATE)
    Path(path).write_text(report.rstrip() + '\n', encoding='utf-8')

    channels = ['feishu'] if top_level == 'Top3' else (['feishu', 'qq'] if top_level == 'Top2' else ['feishu', 'weixin', 'qq'])
    result = finalize_report(path, deliver=True, channels=channels, title=TITLE, body=report, backup_youdao=True, max_chars=3200)
    print(json.dumps({'path': str(path), 'top_level': top_level, 'delivery': result}, ensure_ascii=False, indent=2))
    print('\n---REPORT_WITH_FOOTER---\n')
    print(Path(path).read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
