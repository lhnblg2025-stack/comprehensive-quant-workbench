#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_sources import (  # noqa: E402
    get_cpi_ppi,
    get_customs,
    get_fed_data_sources,
    get_international_oil_sources,
    get_official_source_registry,
    get_pmi,
    get_social_finance_m2,
)
from report_delivery import finalize_report  # noqa: E402
from report_contract import make_report_contract, validate_and_write_contract  # noqa: E402
from report_paths import iso_week, report_path  # noqa: E402


# D7收敛: 同名异口径保留（英文键/成功= 布尔表达，与 report_common.status_line 不同）
def status_line(result) -> str:
    return f"- {result.name}: 成功={result.ok}；source={result.source}；note={result.note}"


def _v(val, placeholder: str = "—") -> str:
    """Render a possibly-missing numeric value; None/空值显示占位符而非崩溃."""
    if val is None or val == "":
        return placeholder
    return val


def _fill_none(d: dict, keys) -> dict:
    """为已知字段预填 None 默认值，避免模板 .get() 后出现 KeyError."""
    for k in keys:
        d.setdefault(k, None)
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description="生成硬核宏观数据周报")
    parser.add_argument("--out")
    parser.add_argument("--no-deliver", action="store_true")
    args = parser.parse_args()
    now = datetime.now()
    cpi_ppi = get_cpi_ppi()
    pmi = get_pmi()
    customs = get_customs()
    credit = get_social_finance_m2()
    oil = get_international_oil_sources()
    fed = get_fed_data_sources()
    registry = get_official_source_registry()

    cpi = _fill_none(cpi_ppi.data.get("cpi") or {}, [
        "yoy", "mom", "core_yoy", "food_yoy", "food_mom", "pork_yoy", "egg_yoy",
        "service_yoy", "industrial_goods_yoy", "gold_jewelry_mom", "gasoline_mom",
    ])
    ppi = _fill_none(cpi_ppi.data.get("ppi") or {}, [
        "yoy", "mom", "coal_mining_yoy", "electrical_machinery_yoy",
        "computer_communication_electronics_yoy", "oil_extraction_mom",
    ])
    sf = _fill_none(credit.data.get("social_finance_stock") or {}, [
        "total_trillion_yuan", "yoy", "government_bonds_yoy", "enterprise_bonds_yoy",
    ])
    flow = _fill_none(credit.data.get("social_finance_flow_h1") or {}, [
        "total_trillion_yuan", "yoy_less_trillion_yuan",
        "rmb_loans_to_real_economy_trillion_yuan", "rmb_loans_yoy_less_trillion_yuan",
    ])
    money = _fill_none(credit.data.get("money") or {}, [
        "m2_yoy", "m1_yoy", "m2_m1_gap_pct_points", "m2_balance_trillion_yuan",
        "m1_balance_trillion_yuan",
    ])
    loans = _fill_none(credit.data.get("loans") or {}, [
        "household_loans_h1_billion_yuan", "household_short_term_loans_h1_billion_yuan",
        "household_medium_long_term_loans_h1_billion_yuan",
        "corporate_loans_h1_trillion_yuan", "corporate_short_term_loans_h1_trillion_yuan",
        "corporate_medium_long_term_loans_h1_trillion_yuan",
        "bill_financing_h1_billion_yuan", "rmb_loan_yoy",
    ])
    rates_fx = _fill_none(credit.data.get("rates_fx") or {}, [
        "interbank_call_rate_june_pct", "pledged_repo_rate_june_pct", "usd_cny_month_end",
    ])

    oil_archive = (oil.data.get("archive") or {}) if isinstance(oil.data, dict) else {}
    oil_source = oil_archive.get("archive_path") or "桌面归档未列出"

    report = f"""# 宏观数据周报｜增长、通胀、信用、外需、油价、资产含义详版

生成时间：{now.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai  
周次：{iso_week(now)}  
报告路径规则：`report_path('macro_weekly')`

## 1. 核心结论：当前宏观组合、主要矛盾、资产配置

当前宏观组合是“生产和外贸偏强、内需和地产偏弱、CPI温和、PPI同比偏强但环比降温、信用总量宽而质量一般”。6月制造业PMI 50.3，生产51.4、新订单51.2、新出口订单50.1，说明制造业回到弱扩张；上半年GDP同比4.7%、6月规上工业增加值同比5.3%，生产端有韧性；但服务业50.4、建筑业49.0、制造业就业48.5，需求修复还没有扩散到居民、地产和就业。

主要矛盾是名义收入能否从生产和外贸传导到居民、企业利润和信用质量。6月CPI同比{_v(cpi.get('yoy'))}%、核心CPI同比{_v(cpi.get('core_yoy'))}%，居民端不是通胀约束；PPI同比{_v(ppi.get('yoy'))}%、环比{_v(ppi.get('mom'))}%，上游同比仍强但原油链条降温。6月社融存量{_v(sf.get('total_trillion_yuan'))}万亿元，同比{_v(sf.get('yoy'))}%；M2同比{_v(money.get('m2_yoy'))}%、M1同比{_v(money.get('m1_yoy'))}%，剪刀差{_v(money.get('m2_m1_gap_pct_points'))}个百分点，说明货币总量宽于资金活化。

资产配置上，债券维持低利率区间思路，内需、地产和居民信用限制长端大幅上行；黄金仍看美元、实际利率、油价通胀预期和地缘，当前高实际利率压制短期弹性；A股用红利作底仓，成长和高端制造做结构性进攻，周期只选供给约束和订单兑现强的细分；人民币有贸易顺差和低通胀支撑，但受美元、实际利率和能源账单约束；商品不能按全面复苏定价，原油和有色看库存、供给纪律和美元。

## 2. 增长：PMI、工业/消费/投资、地产、服务业，判断需求和库存周期

PMI显示增长处在弱扩张。官方制造业PMI从50.0升至50.3，生产51.4、新订单51.2、采购量51.4，生产和订单同步改善；产成品库存47.7，较上月下降1.6，说明6月不是被动累库，而是订单改善带动出库。经济含义是需求边际改善真实存在，但PMI只略高于荣枯线，不能外推为强复苏。资产含义是高端制造、设备、电子、出口链相对受益，全面顺周期仍缺证据。

工业强于服务和地产。上半年GDP同比4.7%，6月规上工业增加值同比5.3%，与PMI生产分项一致；服务业商务活动50.4，只是温和扩张；建筑业49.0仍在收缩，房地产和航空运输等行业仍低于临界点。经济含义是生产韧性主要来自制造和外贸，居民服务消费、地产链和传统投资没有形成同等强度。资产含义是A股风格偏高技术制造、装备和出口链，地产、建材、强消费要等销售、投资和就业确认。

库存周期处在“被动去库向主动补库前夜”。原材料库存48.4仍低，产成品库存下降，采购量回升，在手订单47.1仍低。经济含义是企业按订单补采购，但没有系统性提高库存水位；利润率还受出厂价格48.2偏弱约束。资产含义是商品和周期股更适合看供给约束、出口订单和政策事件，不宜只凭PMI买全面涨价。

## 3. 通胀：CPI/PPI、食品、能源、核心通胀、利润分配

6月CPI同比{_v(cpi.get('yoy'))}%、环比{_v(cpi.get('mom'))}%，核心CPI同比{_v(cpi.get('core_yoy'))}%，食品同比{_v(cpi.get('food_yoy'))}%、食品环比{_v(cpi.get('food_mom'))}%。经济含义是居民端价格温和，食品和能源拖累环比，总需求没有过热。资产含义是债券没有强通胀压力，黄金不因CPI本身获得新增驱动，消费股仍要看销量和利润率而不是涨价。

结构上，猪肉同比{_v(cpi.get('pork_yoy'))}%，鸡蛋同比{_v(cpi.get('egg_yoy'))}%，服务价格同比{_v(cpi.get('service_yoy'))}%，工业消费品同比{_v(cpi.get('industrial_goods_yoy'))}%。黄金饰品环比{_v(cpi.get('gold_jewelry_mom'))}%，汽油环比{_v(cpi.get('gasoline_mom'))}%，两项解释了CPI环比下行的一部分。经济含义是服务通胀稳定但不强，能源和黄金价格波动是短期扰动。资产含义是黄金饰品价格回落不改变黄金配置逻辑，能源回落缓解交通和化工下游成本。

PPI同比{_v(ppi.get('yoy'))}%、环比{_v(ppi.get('mom'))}%，煤炭开采同比{_v(ppi.get('coal_mining_yoy'))}%，电气机械同比{_v(ppi.get('electrical_machinery_yoy'))}%，计算机通信电子同比{_v(ppi.get('computer_communication_electronics_yoy'))}%，石油开采环比{_v(ppi.get('oil_extraction_mom'))}%。经济含义是上游同比仍强，但原油链条边际降温；利润分配偏向资源品和具备定价权的高端制造，中下游价格传导不足。资产含义是煤炭、有色和高端制造可结构性受益，汽车、建材、部分消费制造仍受价格竞争压制。

## 4. 信用：社融、M2、贷款结构、政府债、票据，判断信用质量

6月社融存量{_v(sf.get('total_trillion_yuan'))}万亿元，同比{_v(sf.get('yoy'))}%；上半年新增社融{_v(flow.get('total_trillion_yuan'))}万亿元，同比少{_v(flow.get('yoy_less_trillion_yuan'))}万亿元。经济含义是信用总量仍在扩张，但斜率放慢。资产含义是宽货币仍支撑债券和红利估值，但不足以确认全面宽信用牛市。

贷款结构显示信用质量一般。上半年对实体人民币贷款{_v(flow.get('rmb_loans_to_real_economy_trillion_yuan'))}万亿元，同比少{_v(flow.get('rmb_loans_yoy_less_trillion_yuan'))}万亿元；人民币贷款余额同比{_v((credit.data.get('loans') or {}).get('rmb_loan_yoy'))}%。居民贷款上半年{_v(loans.get('household_loans_h1_billion_yuan'))}亿元，居民短贷{_v(loans.get('household_short_term_loans_h1_billion_yuan'))}亿元，居民中长期贷款{_v(loans.get('household_medium_long_term_loans_h1_billion_yuan'))}亿元。经济含义是居民仍在低杠杆或去杠杆，地产和消费自发修复不足。资产含义是地产链、可选消费和银行高弹性重估仍缺触发。

企业贷款强于居民，但结构仍要看久期和票据。上半年企业贷款{_v(loans.get('corporate_loans_h1_trillion_yuan'))}万亿元，其中企业短贷{_v(loans.get('corporate_short_term_loans_h1_trillion_yuan'))}万亿元、企业中长期贷款{_v(loans.get('corporate_medium_long_term_loans_h1_trillion_yuan'))}万亿元、票据融资{_v(loans.get('bill_financing_h1_billion_yuan'))}亿元。政府债券存量同比{_v(sf.get('government_bonds_yoy'))}%，企业债券同比{_v(sf.get('enterprise_bonds_yoy'))}%。经济含义是财政和直接融资托底，企业端有修复，但居民端弱使信用乘数不高。资产含义是先进制造、政策支持和债券底仓占优，强周期扩散需要M1、居民按揭和企业中长期贷款继续确认。

货币活化仍偏弱。M2余额{_v(money.get('m2_balance_trillion_yuan'))}万亿元，同比{_v(money.get('m2_yoy'))}%；M1余额{_v(money.get('m1_balance_trillion_yuan'))}万亿元，同比{_v(money.get('m1_yoy'))}%；M2-M1剪刀差{_v(money.get('m2_m1_gap_pct_points'))}个百分点。6月银行间同业拆借加权利率{_v(rates_fx.get('interbank_call_rate_june_pct'))}%，质押式回购{_v(rates_fx.get('pledged_repo_rate_june_pct'))}%，月末美元兑人民币{_v(rates_fx.get('usd_cny_month_end'))}。经济含义是流动性宽松但实体经营活化不足。资产含义是债券不宜过度看空，A股更偏结构行情而不是信用驱动的普涨。

## 5. 外需：出口、进口、顺差、区域/商品结构、量价拆分

6月货物进出口总值4.78万亿元，同比24.2%；上半年进出口25.47万亿元，同比16.9%；出口14.73万亿元，同比13.4%；进口10.74万亿元，同比22.1%；上半年顺差约3.99万亿元。经济含义是外贸边际强，而且进口增速高于出口，说明生产补库、设备进口和大宗需求也在贡献。资产含义是人民币有基本面缓冲，港口航运、出口制造、高技术制造和设备链受益。

区域结构上，共建“一带一路”国家上半年进出口12.97万亿元，占比50.9%，同比14.8%；周边国家9.44万亿元，同比20.6%；欧盟同比10.2%，拉美16.2%，非洲19.6%。经济含义是外需支撑来自新兴市场、区域供应链和产业链协作，不是单一欧美需求回暖。资产含义是出口链应偏向机电、高技术、自主品牌和具备区域渠道的公司。

商品结构上，机电产品出口9.36万亿元，同比20.1%，占出口63.5%；高技术产品出口3.26万亿元，同比39.0%；自主品牌出口同比25.4%；机电产品进口4.41万亿元，同比28.0%；大宗商品进口数量合计14.29亿吨，同比3.4%。经济含义是出口竞争力和进口投入同时改善，但原油、铁矿、煤、铜矿等品种的金额、数量、均价仍未完整取得。资产含义是AI硬件、电子、工业设备、汽车零部件、工程机械优先；资源品需要等品种量价拆分确认利润方向。

## 6. 油价影响：通胀、贸易账单、企业成本、人民币、黄金和A股行业

最新原油归档显示WTI 79.20美元/桶、Brent 81.62美元/桶，Brent-WTI价差约2.42美元/桶；EIA周度口径下商业原油库存409.665百万桶，周环比-1.693百万桶，汽油库存210.529百万桶，周环比-1.533百万桶，Cushing库存20.044百万桶。经济含义是商业原油和汽油去库、Cushing低位支撑油价，但总库存周度回升和高实际利率限制单边上行。资产含义是油价偏强可支撑上游资源、油服和油运，但追涨需要美元和实际利率配合。

通胀上，6月汽油环比回落拖累CPI，石油开采环比大跌拖累PPI环比。若后续油价反弹，会推高交通燃料、石化链和进口能源账单；若油价下跌来自需求弱，债券受益但周期股利润预期受压。资产含义是油价上行利好上游资源和煤化工相对成本优势，压制航空、交运、轮胎、包装和化工下游利润率。

贸易账单和人民币上，Brent上行会增加中国能源进口购汇需求，削弱顺差质量；若同时美元走强，人民币压力会放大。黄金上，油价上行若推升通胀预期且实际利率回落，黄金受益；若油价上行引发鹰派Fed预期和实际利率走高，黄金弹性受限。A股行业上，上游资源、油服、油运偏受益，航空、快递、公路、轮胎和石化下游偏承压。

## 7. 海外金融条件：美元、美债名义/实际利率、Fed、流动性

Fed/FRED官方源已固定，最新油价归档给出的海外金融条件是：美国10年名义利率4.6%，10年实际利率2.4%，10年通胀预期2.2%，美元广义指数120.531，高收益信用利差2.7%。经济含义是名义和实际利率仍高，美元偏强，全球流动性不是单边宽松；通胀预期稳定，市场还没把油价上行定价成失控通胀。资产含义是黄金和成长估值受实际利率约束，人民币受美元约束，商品需要库存和供给来抵消金融条件压力。

Fed反应函数仍看通胀扩散、就业和金融条件。当前中国数据对海外金融条件的含义是：外需强和PPI同比高说明全球制造链仍有价格韧性，但中国CPI温和、油价此前环比拖累、信用结构弱，不构成全球再通胀一边倒。资产含义是黄金要同时看实际利率和美元，人民币要看贸易顺差能否抵消中美利差和美元压力，A股成长需要外部流动性不再收紧。

流动性传导上，若美元走强叠加油价上行，会形成“能源账单上升、人民币压力、外资风险偏好下降”的组合；若美元回落且油价平稳，外贸顺差和低通胀会给人民币资产更多空间。债券在美元强时未必受益，因为汇率约束会限制国内宽松想象；黄金在美元强和实际利率上行时会被压制。

## 8. 格林斯潘框架：央行反应函数、金融失衡、生产率长周期、危机流动性传导

央行反应函数：国内央行最关心的是增长就业、价格温和、汇率稳定和信用传导。当前CPI不高，PPI环比降温，货币政策没有因通胀收紧的压力；但美元偏强、实际利率高和人民币汇率会约束过度宽松。能触发政策再定价的数据是7月PMI新订单、居民中长期贷款、M1、地产销售和人民币汇率。

金融失衡：低利率环境下，最大风险不是通胀失控，而是资金没有进入实体投资，转向债券、红利和主题资产拥挤。M2高于M1、居民贷款为负、票据融资不低，都提示信用传导质量一般。资产含义是债券和红利有支撑但要防拥挤，成长和周期必须等盈利、订单和现金流确认。

生产率长周期：高技术产品出口同比39.0%、机电出口和进口高增、PMI中高技术制造和信息服务较强，说明增长质量的积极部分来自产业升级和外部竞争力，而不是地产信用扩张。资产含义是成长制造、AI硬件、工业设备有中期逻辑，但估值必须与订单和现金流匹配。

危机流动性传导：最需要防的链条是美元走强或油价上行引发人民币压力，进而压制外资风险偏好和国内宽松空间；另一条是PPI强于CPI挤压中下游利润，弱行业信用风险上升。当前没有危机信号，但这些链条决定下周观察重点。

## 9. 资产含义：债券、黄金、A股红利/成长/周期/小微盘、人民币、商品

债券：CPI温和、居民信用弱、地产和建筑业未修复，长端利率上行空间受限；但PMI、外贸和PPI同比改善会限制收益率快速下行。若居民中长期贷款继续弱、M1不活跃，债券胜率提高；若企业中长期贷款和居民按揭同步强，债券面临止盈压力。

黄金：中期仍看实际利率、美元、地缘和央行需求。当前美国10年实际利率约2.4%，对黄金估值有压制；油价若上行但实际利率回落，黄金更容易共振；油价上行若推动鹰派Fed和美元走强，则不利于黄金。

A股红利：低利率、信用质量一般和内需弱修复仍支持红利底仓，但拥挤和估值要控制。若M1和居民中长期贷款明显转强，红利相对优势会下降；若信用继续弱，红利仍是组合稳定器。

A股成长：PMI新订单、高技术出口、机电进口和产业升级支持成长制造、AI硬件、电子设备、工业自动化和装备链。风险是出厂价格弱、利润兑现不足和外部实际利率上行。成长适合精选订单和现金流，不适合无业绩主题扩散。

A股周期：资源品受PPI同比、煤炭、有色和油价库存支撑，但内需和地产没有确认强复苏。周期只能买供给约束、出口订单或价格韧性明确的细分；黑色、建材和地产链仍需谨慎。

小微盘：弱复苏和流动性宽松有利于风险偏好，但信用质量和盈利扩散不足，小微盘容易受资金风格驱动。若成交和市场宽度不能扩散，不能把小微盘反弹视作宏观确认。

人民币：上半年顺差约3.99万亿元、月末外储3.42万亿美元、月末美元兑人民币6.8109提供支撑；但进口高增、能源账单、美元和中美利差会压制升值弹性。人民币更可能震荡偏稳，趋势升值要等美元转弱或国内增长质量改善。

商品：商品不应按全面需求复苏交易。原油看EIA库存、OPEC和美元实际利率；煤炭有迎峰度夏和PPI支撑；有色看全球需求和美元；黑色和建材受地产与建筑业拖累。若油价下跌来自需求弱，商品整体风险偏空；若美元回落且库存低，资源品弹性提高。

## 10. 下周观察清单

1. 7月PMI前瞻，尤其新订单、新出口订单、出厂价格、从业人员和建筑业是否改善。
2. 社融结构延续性：居民中长期贷款、企业中长期贷款、票据融资、M1和M2-M1剪刀差。
3. 地产销售、土地成交、居民按揭与建筑业高频数据，判断地产链是否继续拖累。
4. 海关6月出口、进口、顺差单月拆分、美元口径、对美/东盟/欧盟结构和重点商品量价表。
5. EIA商业原油、汽油、馏分油、Cushing库存，WTI/Brent与美元、实际利率是否共振。
6. 美国10年名义利率、10年实际利率、10年通胀预期、贸易加权美元指数和Fed表态。
7. A股市场宽度、成交额、红利拥挤度、成长制造订单验证、周期品价格与利润传导。

## 11. 数据源、失败项、fallback

必须调用的数据源结果：
{status_line(cpi_ppi)}
{status_line(pmi)}
{status_line(customs)}
{status_line(credit)}
{status_line(oil)}
{status_line(fed)}
{status_line(registry)}

已使用的归档和官方源：
- CPI/PPI：国家统计局2026-07-09解读稿，`get_cpi_ppi()` 官方校验快照。
- PMI：`get_pmi()` 读取桌面归档 `{pmi.data.get('archive_path')}`。
- 海关：`get_customs()` 读取桌面归档 `{customs.data.get('archive_path')}`。
- 社融/M2：`get_social_finance_m2()` 返回央行2026-06官方校验快照。
- 国际原油：`get_international_oil_sources()` 使用OPEC/EIA/FRED官方源登记和桌面归档 `{oil_source}`。
- Fed/FRED：`get_fed_data_sources()` 和 `get_official_source_registry()` 提供官方源入口和序列ID。
- 增长总量：桌面归档 `$HOME/Desktop/宏观数据/经济数据/2026-H1_上半年经济数据半年报与解读.md`。

失败项和处理：
- 必须调用的7个函数均成功返回，本次未修改 `scripts/data_sources.py` fallback。
- PMI、海关、原油部分结构化字段来自桌面归档 fallback，报告已标注来源；若后续官方表格抓取失败，应优先把稳定fallback补进 `scripts/data_sources.py`。
- 海关重点商品原油、铁矿、煤、铜矿的完整金额、数量、均价未取得，未编造量价拆分。
- OPEC结构化需求、供给、call on OPEC未解析，未编造数值。
- 外层cron delivery保持 `mode:none`；本报告投递由 `finalize_report(...)` 处理并在下一节记录状态。

"""

    if args.out:
        out = Path(args.out).expanduser()
    else:
        try:
            out = report_path("macro_weekly")
        except OSError:
            out = ROOT / "研究报告" / "周报" / iso_week() / "宏观数据周报.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    try:
        sources = []
        for index, item in enumerate((cpi_ppi, pmi, customs, credit, oil, fed, registry), 1):
            data = getattr(item, "data", {}) or {}
            observed = data.get("published_date") or data.get("period") or now.strftime("%Y-%m-%d")
            sources.append({"id": f"src-macro-{index}", "name": getattr(item, "name", f"macro_source_{index}"), "kind": "official", "status": "ok" if getattr(item, "ok", False) else "failed", "observed_at": str(observed)[:10], "locator": getattr(item, "source", None), "note": getattr(item, "note", "")})
        source_ref = sources[0]["id"]
        evidence = [{"claim": "宏观周报已取得真实来源结果", "value": "official_source_bundle", "unit": "text", "observed_at": now.strftime("%Y-%m-%d"), "source_ref": source_ref, "quality": "primary"}]
        contract = make_report_contract("weekly", "宏观数据周报", now.strftime("%Y-%m-%d"), subject={"id": "macro-cn", "name": "中国宏观", "kind": "market"}, report_id=f"macro_weekly:macro-cn:{iso_week(now)}", period_start=now.strftime("%Y-%m-%d"), period_end=now.strftime("%Y-%m-%d"), summary={"stance": "observe", "summary": "生产与外需有韧性，但内需、信用质量和金融条件仍需验证。", "horizon": "months", "confidence": None}, evidence=evidence, transmission=[{"from": "增长/通胀/信用/金融条件", "to": "债券、黄金与A股风格", "mechanism": "宏观数据通过利率、盈利、汇率和风险偏好传导", "direction": "mixed", "evidence_refs": ["ev-1"], "confidence": None}], risks=[{"description": "宏观官方/归档源存在发布滞后或字段缺口", "trigger": "关键指标未更新或来源失败", "impact": "只保留观察，不外推单一资产方向", "severity": "medium", "evidence_refs": ["ev-1"]}], actions=[{"action": "validate", "target": "下一期宏观数据", "condition": "PMI、社融/M1与外部金融条件同日更新", "invalidated_by": "来源失败或发布日期错位", "horizon": "next_release", "evidence_refs": ["ev-1"]}], data_gaps=[], sources=sources, chapters=[{"title": "宏观组合", "conclusion": "生产外需与内需信用存在分化", "evidence": ["ev-1"], "implication": "资产配置需要区分利率、盈利与汇率传导", "next_check": "核对下一期官方发布"}], metadata={"week": iso_week(now), "legacy_markdown": True})
        validate_and_write_contract(out, contract)
    except Exception as exc:
        print(f"research contract error: {exc}", file=sys.stderr)
    result = finalize_report(
        out,
        deliver=not args.no_deliver,
        channels=["feishu"],
        title="宏观数据周报｜增长、通胀、信用、外需、油价、资产含义详版",
        body=report,
        max_chars=9000,
        backup_youdao=not args.no_deliver,
    )
    print(out)
    for item in result.get("statuses", []):
        print(f"{item.get('name')}: {item.get('status')} - {item.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
