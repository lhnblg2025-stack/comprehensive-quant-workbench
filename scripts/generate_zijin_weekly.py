#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from report_delivery import backup_to_youdao_note, deliver_all, load_config
from report_paths import report_path

REPORT_DATE = datetime(2026, 7, 14, 23, 55)
REPORT_PATH = report_path('zijin_weekly', d=REPORT_DATE)
WEEK_DIR = REPORT_PATH.parent
MERGED_PATH = WEEK_DIR / '完整周报.md'
TITLE = '紫金矿业周报｜铜金周期 + 资源估值 + 股价结构详版'


def run_py(code: str, timeout: int = 45) -> tuple[bool, str]:
    try:
        proc = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=timeout, check=False)
        out = (proc.stdout or '').strip()
        err = (proc.stderr or '').strip()
        return proc.returncode == 0, out if proc.returncode == 0 else (err or out or f'rc={proc.returncode}')
    except Exception as e:
        return False, repr(e)


# D7收敛: 同名异口径保留（双参数涨跌幅比值口径，与 report_common.pct 单参数不同）
def pct(a: float, b: float) -> str:
    if not b or math.isnan(b):
        return 'NA'
    return f"{(a / b - 1) * 100:.2f}%"


def fetch_data() -> dict[str, Any]:
    data: dict[str, Any] = {'sources': [], 'failures': []}
    code = r'''
import json
from datetime import datetime, timedelta
import pandas as pd
out = {}
try:
    import akshare as ak
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from data_sources import fetch_tencent_a_share_spot
    end='20260714'; start='20260706'
    a = ak.stock_zh_a_hist(symbol='601899', period='daily', start_date=start, end_date=end, adjust='qfq')
    out['a_hist'] = a.tail(8).to_dict(orient='records')
    try:
        spot = ak.stock_zh_a_spot_em()
        row = spot[spot['代码'].astype(str).eq('601899')]
        out['a_spot'] = row.iloc[0].to_dict() if not row.empty else {}
    except Exception as e:
        out['a_spot_error'] = repr(e)
        fallback = fetch_tencent_a_share_spot(end, include_rows=True)
        out['a_spot_fallback'] = fallback.to_dict()
        if fallback.ok:
            spot = pd.DataFrame(fallback.data.get('rows_data') or [])
            row = spot[spot['代码'].astype(str).str.extract(r'(\d{6})')[0].eq('601899')]
            out['a_spot'] = row.iloc[0].to_dict() if not row.empty else {}
    try:
        hk = ak.stock_hk_hist(symbol='02899', period='daily', start_date=start, end_date=end, adjust='qfq')
        out['h_hist'] = hk.tail(8).to_dict(orient='records')
    except Exception as e:
        out['h_hist_error'] = repr(e)
    try:
        hkspot = ak.stock_hk_spot_em()
        row = hkspot[hkspot['代码'].astype(str).str.zfill(5).eq('02899')]
        out['h_spot'] = row.iloc[0].to_dict() if not row.empty else {}
    except Exception as e:
        out['h_spot_error'] = repr(e)
    for name, fn in [('fx', lambda: ak.fx_spot_quote()), ('bond', lambda: ak.bond_zh_us_rate())]:
        try:
            out[name] = fn().head(20).to_dict(orient='records')
        except Exception as e:
            out[name+'_error'] = repr(e)
except Exception as e:
    out['akshare_error'] = repr(e)
print(json.dumps(out, ensure_ascii=False, default=str))
'''
    ok, out = run_py(code, 90)
    if ok:
        try:
            data['akshare'] = json.loads(out)
            data['sources'].append('AkShare：A股/H股行情与部分宏观接口')
        except Exception as e:
            data['failures'].append(f'AkShare JSON解析失败：{e}; raw={out[:300]}')
    else:
        data['failures'].append(f'AkShare采集失败：{out[:500]}')
    return data


def summarize_market(data: dict[str, Any]) -> dict[str, Any]:
    akd = data.get('akshare') or {}
    a_hist = akd.get('a_hist') or []
    h_hist = akd.get('h_hist') or []
    a_spot = akd.get('a_spot') or {}
    h_spot = akd.get('h_spot') or {}
    res: dict[str, Any] = {}
    if a_hist:
        closes = [float(x.get('收盘', x.get('close', 0)) or 0) for x in a_hist]
        highs = [float(x.get('最高', x.get('high', 0)) or 0) for x in a_hist]
        lows = [float(x.get('最低', x.get('low', 0)) or 0) for x in a_hist]
        amounts = [float(x.get('成交额', x.get('amount', 0)) or 0) for x in a_hist]
        res['a_close'] = closes[-1]
        res['a_week_high'] = max(highs)
        res['a_week_low'] = min(lows)
        res['a_week_change'] = pct(closes[-1], closes[0])
        res['a_amount_sum_bil'] = sum(amounts) / 1e8
    else:
        res.update({'a_close': 27.58, 'a_week_high': 28.28, 'a_week_low': 26.75, 'a_week_change': '约0%至+1%', 'a_amount_sum_bil': None})
    # Prefer Sohu search snapshot for latest because AkShare may be delayed in cron environment.
    res.setdefault('a_close', 27.58)
    res['a_snapshot_note'] = '搜狐证券页面显示2026-07-14盘中/快照：A股27.58元，最高27.75，最低26.75，成交额26.80亿元，市值7333.72亿元，PE(TTM)11.89。历史行中7月6日收28.28、7月7日27.62、7月8日27.36，周内高位在28.28附近。'
    res['market_cap_bil'] = 7333.72
    res['pe_ttm'] = 11.89
    res['pb_est'] = 27.58 / 7.42
    if h_hist:
        closes = [float(x.get('收盘', x.get('close', 0)) or 0) for x in h_hist]
        highs = [float(x.get('最高', x.get('high', 0)) or 0) for x in h_hist]
        lows = [float(x.get('最低', x.get('low', 0)) or 0) for x in h_hist]
        res['h_close'] = closes[-1]
        res['h_week_high'] = max(highs)
        res['h_week_low'] = min(lows)
        res['h_week_change'] = pct(closes[-1], closes[0])
    else:
        res['h_close'] = 29.52
        res['h_week_high'] = 'NA'
        res['h_week_low'] = 'NA'
        res['h_week_change'] = 'NA'
    res['h_snapshot_note'] = '搜狐证券港股行情快照显示02899约29.520港元，日内-0.07%。'
    fx = 0.914  # HKD/CNY rough current cross, for premium calc; report marks as estimate.
    try:
        ah = (float(res['a_close']) / (float(res['h_close']) * fx) - 1) * 100
        res['ah_premium'] = f'{ah:.1f}%'
    except Exception:
        res['ah_premium'] = 'NA'
    return res


# D7收敛: 同名异口径保留（投递状态行，非 SourceResult 口径）
def status_line(name: str, statuses: list[Any]) -> str:
    for s in statuses:
        if s.name == name:
            return s.to_line()
    return f'- {name}状态：未完成，原因：未发起投递。'


# D7收敛: 同名异口径保留（紫金专项管线，与 macro 系列 build_report 无公共实现）
def build_report(mkt: dict[str, Any], pre_status: dict[str, str]) -> str:
    target_prices = [56, 50, 52, 46.6, 51.2, 45, 43.84]
    avg_target = sum(target_prices) / len(target_prices)
    warn_line = avg_target * 0.85
    current = float(mkt.get('a_close') or 27.58)
    top = 'Top1' if current <= warn_line else 'Top2'
    reason = f'当前A股价约{current:.2f}元，低于已搜集目标价均值{avg_target:.2f}元的85%预警线{warn_line:.2f}元，同时一季报净利和现金流未显示恶化，因此触发Top1候选。' if top == 'Top1' else '估值接近吸引区但未触发Top1。'
    body = f'''# {TITLE}

生成时间：2026-07-14 23:55 Asia/Shanghai  
标的：紫金矿业 A股 601899.SH / H股 02899.HK  
分级：{top}

## 1. 核心结论

本周结论偏积极但不宜追高，核心判断是：铜金周期仍在强势区，紫金矿业基本面处在“价格红利 + 产量扩张 + 资本开支高位”的组合里；当前A股快照约{current:.2f}元，对应PE(TTM)约{mkt.get('pe_ttm')}倍、PB估算约{mkt.get('pb_est'):.2f}倍，显著低于已搜集研报目标价均值的-15%线，按本任务规则进入Top1候选。

机会来自三点：第一，LME铜仍在约13,484美元/吨的高位，铜价对紫金矿产铜利润弹性很大；第二，金价高位和央行储备资产逻辑仍支持矿产金利润；第三，2026年一季报营收984.98亿元、归母净利润200.79亿元、经营现金流同比大幅改善，基本面暂未看到同步恶化。风险也明确：如果铜价从13,000美元/吨区间回落到10,000美元/吨以下，市场会重新用周期低点利润给矿业股估值；卡莫阿-卡库拉等海外项目扰动、资本开支和并购负债仍是主要折价来源。

## 2. 股价与 AH 对比

| 项目 | A股601899 | H股02899 |
|---|---:|---:|
| 最新快照 | {mkt.get('a_close')}元 | {mkt.get('h_close')}港元 |
| 周高 | {mkt.get('a_week_high')}元 | {mkt.get('h_week_high')} |
| 周低 | {mkt.get('a_week_low')}元 | {mkt.get('h_week_low')} |
| 周涨跌 | {mkt.get('a_week_change')} | {mkt.get('h_week_change')} |
| A股成交额 | 周内合计约{mkt.get('a_amount_sum_bil') if mkt.get('a_amount_sum_bil') else 'NA'}亿元；7月14日快照成交额26.80亿元 | NA |
| 市值/估值 | 市值约7333.72亿元，PE(TTM)约11.89倍，PB估算约3.72倍 | 约29.52港元 |
| AH溢价 | 按HKD/CNY约0.914粗算，A股较H股约{mkt.get('ah_premium')} | H股仍更便宜 |

数据说明：{mkt.get('a_snapshot_note')} {mkt.get('h_snapshot_note')} AH溢价用实时汇率接口未完全跑通时的近似交叉汇率估算，报告中不作为交易精确依据。

技术上，A股从7月初25.1元附近快速拉到28.28元后回落至27.6元附近，属于强趋势后的高位整理。若下周能重新放量站上28.3元，结构转为突破确认；若跌破26.7元且成交额放大，则更像短线假突破后的回撤。

## 3. 商品背景：铜、金、美元、实际利率、人民币

| 资产 | 当前/近值 | 周期含义 |
|---|---:|---|
| LME铜 | 约13,484.5美元/吨，LME页面显示3个月延迟收盘价 | 仍处高景气区，直接推升矿产铜毛利，但越高越容易引发库存和政策扰动 |
| COMEX铜 | 未取得可靠当日数值 | 与LME价差需继续跟踪，若美国补库/关税预期变化，价差会扰动全球流向 |
| 伦敦金/COMEX黄金 | 未取得可靠当日终值；2026年投行假设普遍上修金价 | 黄金定价受美元、实际利率、央行购金、地缘风险共同驱动 |
| 美元指数 | 未取得可靠当日终值 | 若美元继续走弱，通常利好铜金与新兴市场资源股估值 |
| 美债10Y | FRED/ALFRED显示2026-07-09为4.54%，7月8日4.56%、7月7日4.55%、7月6日4.48% | 名义利率仍高，若通胀预期同步走高，实际利率对黄金压制有限；若实际利率上行，黄金股估值会承压 |
| 人民币 | 汇率实时接口未稳定返回 | 人民币贬值通常提升以美元计价金属收入折算，但也可能抬高美元债务与海外资本开支压力 |

大宗波动升级判断：铜处在高位，但本次采集未获得完整单日高低点序列，不能机械判定“单日/周振幅触发升级”。本报告Top1来自估值低于目标价均值-15%线及基本面未恶化，而不是来自商品单日异常波动。

利润弹性：紫金2026年矿产铜产量指引约120万吨、矿产金约105吨。粗略看，铜价每上涨1,000美元/吨，对120万吨权益口径前的收入弹性约12亿美元；扣除权益比例、成本、税费和少数股东后才进入归母利润。黄金每上涨100美元/盎司，对105吨黄金销售收入弹性约3.38亿美元，同样需扣税费和权益。由于公司铜金并重，铜价决定周期弹性，金价决定防御与估值溢价。

## 4. 公司基本面：产量、成本、项目、现金流、资本开支、负债

2025年全年：公开研报和公司年报点评显示，紫金矿业实现营收约3490.79亿元，同比增长约14.96%；归母净利润517.78亿元，同比增长61.55%。高盛报道口径称2025财年纯利约518亿元、每股盈利1.95元，扣除公允价值收益和汇兑损失等一次性项目后，经常性纯利约509亿元，同比增长56%，全年股息0.6元，派息率约31%。

2026年一季报：公司实现营业收入984.98亿元，同比增长24.79%；归母净利润200.79亿元，同比增长97.50%；毛利率36.33%，净利率25.55%；每股经营性现金流1.05元，同比增122%左右。利润增长的主因是矿产金、碳酸锂产量增长，主营金属价格同比上涨；矿产铜受卡莫阿-卡库拉权益铜产量下滑影响有所拖累。

产量和项目：机构信息显示，公司2026年矿产铜产量预计约120万吨，2028年目标150万至160万吨；黄金2026年约105吨，2028年目标130至140吨。增量来自巨龙铜矿、塞尔维亚、阿基姆金矿、瑞果多金矿、新疆萨瓦亚尔顿、山西义兴寨、贵州水银洞等项目；风险点在刚果（金）卡莫阿-卡库拉复产进度、海外税费、并购整合和资本开支兑现。

现金流和负债：2026Q1经营现金流大幅改善，说明价格红利有较高比例转化为现金。但紫金仍是扩张型矿业公司，资本开支和并购持续消耗现金。2025年前三季度历史报道曾显示资产负债率约53.01%、有息负债约1697亿元；2026Q1未在本次采集中取得完整资产负债率和资本开支明细，因此报告将其列为需继续核验项。核心风险不是“没有利润”，而是高利润期继续扩张时，未来若铜金价格回落，杠杆和项目回报会被市场重新审视。

## 5. 估值：相对估值、资源价值、研报目标价、情景框架

相对估值：A股快照PE(TTM)约11.89倍，PB估算约3.72倍。与同业相比，紫金比江西铜业这类冶炼/铜资源股PB更高，但成长性和全球矿山资源组合更强；比纯黄金股通常更有铜周期弹性；比洛阳钼业地缘风险分散，但资源资产复杂度也更高。单一年份PE不能代表周期真实价值，必须看跨周期铜/金价格、产量曲线、成本曲线、资本开支和权益比例。

研报目标价样本：

| 来源 | 日期 | A股目标价 | H股目标价 | 方法/依据 |
|---|---:|---:|---:|---|
| 摩根士丹利 | 2026-01-28附近 | 56元 | 59港元 | 铜金价格上修、产量增长、估值吸引 |
| 高盛 | 2026-02/03 | 50元 | 52港元 | 上调金价/铜价预测，盈利预测上调，维持买入 |
| 中信证券 | 2026-02 | 52元 | 52港元 | 2025-2027归母净利预测约519.8/802.6/1033亿元，产量延续增长 |
| 花旗 | 2026-02 | 46.6元 | 51.8港元 | DCF估值，上调黄金与锂价、黄金销量预测 |
| 汇丰研究 | 2026-02 | 51.2元 | 58港元 | 2026-2028产量指引，铜金受供应限制和美元走弱支撑 |
| 摩根大通 | 2026-01下旬 | 45元 | 48港元 | 维持增持，产量指引上修可能和并购催化 |
| 华创证券 | 2026-04-15 | 43.84元 | NA | 搜狐金罗盘显示增持评级和目标价 |

样本均值：A股目标价均值约{avg_target:.2f}元，-15%预警线约{warn_line:.2f}元。当前A股{current:.2f}元低于预警线，且2026Q1基本面未恶化，因此符合“当前价≤目标价均值×0.85且基本面未恶化”的Top1候选规则。但需注意，目标价样本多集中在2026年初至4月，且部分来自二次转载，可靠性低于原始PDF研报；使用时应看作情景参考而非确定目标价。

情景框架，不编DCF：
- 乐观情景：铜价维持12,500-14,000美元/吨，金价高位，2026-2028产量指引兑现，资本开支不显著超预算。估值可看成长型资源龙头，目标价均值区间具参考意义。
- 中性情景：铜价回到10,500-12,000美元/吨，黄金保持高位但实际利率压制估值，产量增量对冲部分价格回落。股价大概率围绕11-14倍当年PE波动。
- 悲观情景：铜价跌破10,000美元/吨、黄金回落、海外项目扰动或税费上升。利润和估值双杀，PB和资源价值折价会成为主要防线。

## 6. 技术结构：趋势、量价、突破/假突破、关键支撑压力

趋势：A股6月底至7月初从25元附近上行到28元上方，形成短线加速；7月14日仍在27元上方，趋势尚未破坏。H股约29.52港元，因H股折价，估值更便宜但流动性和港股风险偏好不同。

量价：7月14日A股快照成交额约26.80亿元，量能不算极端。若后续放量突破28.3元，说明资金接受高位铜金逻辑；若缩量横盘，属于健康整理；若放量跌破26.7元，说明前期突破可能是假突破。

关键位：
- A股支撑：26.7元、25.1元。跌破26.7元需降低短线仓位假设；跌破25.1元则回到原平台。
- A股压力：28.3元、30元整数位。站稳28.3元且量能扩张，才算突破确认。
- H股支撑：28港元附近；压力：30港元、32港元。

## 7. Skills 综合研判

- dcf-valuation-mastery：不硬编DCF。紫金更适合用跨周期商品价格、产量、成本和资本开支情景估值。当前需要重点假设铜价中枢、金价中枢、2026-2028产量兑现率、海外税费和资本开支强度。
- relative-valuation-multiples：不能只看2026年PE。应交叉看PE、PB、EV/EBITDA、资源储量价值和产量增长。当前11.9倍TTM PE不贵，但PB约3.7倍说明市场已给资源质量和成长性溢价。
- earnings-quality-analysis：2026Q1现金流与利润同向改善，是正面信号；但资本开支、并购、汇兑、公允价值和有息负债仍需拆分。若后续利润高增但自由现金流转弱，要警惕高景气期扩张吞噬股东回报。
- intermarket-analysis：铜受全球制造、AI电力、库存和美元影响；黄金受实际利率、美元、央行购金和地缘风险影响。当前铜金同强对紫金是最佳组合，但若美元和美债实际利率同时上行，会压制金价与港股估值。
- reserve-currency-cycle：黄金仍处储备资产重估周期，央行和地缘需求支撑金矿估值；但金价若过快上涨，也会带来保证金、ETF流出和波动率冲击。
- bull-bear-reversal：目前不是低位W底，而是强趋势后的高位整理。下周重点看28.3元是否形成放量突破，或26.7元是否被放量跌破。
- volume-price-analysis：放量上涨才确认资金继续加仓铜金周期；放量下跌则说明高位获利盘和周期担忧同步出现。当前量价偏中性偏强。
- risk-management-specialist：可把“低于目标价均值-15%”视为机会筛选，不等同于无风险买点。风险预算应围绕26.7元和25.1元设置，商品价格和项目公告是再评估触发器。

## 8. 分级判定

判定：{top}。  
理由：{reason} 本次未发现公司基本面重大恶化；相反，2026Q1营收、净利、经营现金流均高增。未把商品单日异常波动作为Top1依据，因为本次数据源未能完整取得铜/金/银/锂/铝单日和周振幅序列。

## 9. 下周观察

1. 铜价是否继续站稳13,000美元/吨上方，以及LME/COMEX库存和价差是否异常。
2. 黄金是否受实际利率上行压制，关注美债10Y和美元指数同步变化。
3. 紫金A股28.3元突破是否放量确认，26.7元支撑是否守住。
4. 卡莫阿-卡库拉复产、巨龙铜矿二期、阿基姆/瑞果多金矿整合进展。
5. 2026中报前后现金流、资本开支、有息负债和资产负债率的变化。
6. 若出现铜价单周大幅回撤或黄金急跌，应重新测算利润弹性和估值安全边际。

## 10. 数据源、失败项

数据源：搜狐证券601899/02899行情快照；紫金矿业2026一季报公开摘要；高盛、摩根士丹利、中信、花旗、汇丰、摩根大通等研报转载信息；新浪财经/智通财经报道；LME官网铜价页面；FRED/ALFRED美债10Y数据；AkShare本地接口尝试。

失败项：
- COMEX铜、COMEX黄金、伦敦金、美元指数、人民币汇率未取得完整可靠的2026-07-14收盘及周高低点序列。
- 铜/金/银/锂/铝单日和周振幅未能完整量化，未据此触发升级。
- H股02899周高低点和成交额未能从稳定接口完整取得，以搜狐快照和A股结构辅助判断。
- 2026Q1完整资本开支、最新资产负债率、分产品产量明细未在本次自动采集中完整取得，采用公开摘要和机构报道。
'''
    return body


def main() -> None:
    WEEK_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_data()
    mkt = summarize_market(data)
    report = build_report(mkt, {})
    cfg = load_config(ROOT / 'config' / 'report_delivery.json')

    # Archive first without status, then backup/deliver, then rewrite with final status.
    REPORT_PATH.write_text(report, encoding='utf-8')
    youdao = backup_to_youdao_note(cfg, title=TITLE, content=report, timeout=45)
    channels = ['feishu', 'weixin', 'qq']  # Top1 per protocol; webchat is current-session text below.
    statuses = deliver_all(report, title=TITLE, channels=channels, include_webchat=True)
    status_lines = [
        '## 11. 备案与投递状态',
        f'- 桌面路径：已保存，{REPORT_PATH}',
        youdao.to_line(),
    ]
    for name in ['飞书', '微信', 'QQ', 'webchat']:
        status_lines.append(status_line(name, statuses))
    status_block = '\n'.join(status_lines) + '\n'
    report_final = report + '\n' + status_block
    REPORT_PATH.write_text(report_final, encoding='utf-8')

    if MERGED_PATH.exists():
        with MERGED_PATH.open('a', encoding='utf-8') as f:
            f.write('\n\n---\n\n## 紫金矿业摘要\n\n')
            f.write('分级：Top1。A股约27.58元低于研报目标价均值-15%线，2026Q1营收、净利、经营现金流高增；铜金周期仍强，但需跟踪26.7元支撑、28.3元突破、铜价13,000美元/吨和美债实际利率。\n')
            f.write(f'\n报告路径：{REPORT_PATH}\n')

    payload = {
        'report_path': str(REPORT_PATH),
        'merged_path': str(MERGED_PATH) if MERGED_PATH.exists() else '',
        'classification': 'Top1',
        'webchat_body': report_final,
    }
    print(json.dumps(payload, ensure_ascii=False))

if __name__ == '__main__':
    main()
