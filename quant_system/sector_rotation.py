"""
行业轮动分析 — 申万一级行业分类+资金流+轮动速度+强弱排名

核心功能:
  1. 行业分类映射: 409只自选股 → 申万一级行业
  2. 行业指数表现: 各行业日涨跌幅/周涨跌幅/月涨跌幅
  3. 资金流分析: 主力净流入/净流出排名 (今日/5日/10日)
  4. 轮动速度: 行业动量变化率，识别加速/减速行业
  5. 强弱排名: 相对强弱RS、轮动阶段判断

用法:
  python3 -m quant_system.sector_rotation
  python3 -m quant_system.sector_rotation --rank
  python3 -m quant_system.sector_rotation --fund-flow
  python3 -m quant_system.sector_rotation --momentum
  python3 -m quant_system.sector_rotation --rotation-map
"""

from __future__ import annotations
import logging

import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_system.utils import to_float as _to_float

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 60  # 缓存秒数


def _cache_get(key: str) -> Any:
    """从缓存获取"""
    item = _CACHE.get(key)
    if item and _time.time() - item['ts'] < _CACHE_TTL:
        return item['data']
    return None


def _cache_set(key: str, data: Any) -> None:
    _CACHE[key] = {'data': data, 'ts': _time.time()}


# 申万一级行业 → 东财板块名映射。
# P1-Q27-fix: 申万行业名与东财板块名是两套体系（如申万"煤炭开采加工"vs东财"煤炭行业"），
# 直接用申万名调 stock_board_industry_cons_em 会大量映射为空。
_SW_TO_EM_BOARDS: dict[str, list[str]] = {
    "农林牧渔": ["农牧饲渔", "农业种植", "养殖业", "渔业", "农产品加工"],
    "基础化工": ["化学原料", "化学制品", "农化制品", "橡胶制品", "塑料制品", "氟化工", "磷化工"],
    "钢铁": ["钢铁行业"],
    "有色金属": ["有色金属", "贵金属", "小金属", "能源金属", "工业金属"],
    "电子": ["半导体", "元件", "光学光电子", "消费电子", "电子化学品"],
    "汽车": ["汽车整车", "汽车零部件", "汽车服务"],
    "家用电器": ["家电行业", "小家电", "白色家电"],
    "食品饮料": ["酿酒行业", "食品饮料", "食品加工", "乳业", "调味品"],
    "纺织服饰": ["纺织服装", "服装家纺", "纺织制造"],
    "轻工制造": ["家用轻工", "造纸印刷", "包装材料", "家居用品"],
    "医药生物": ["化学制药", "中药", "生物制品", "医疗器械", "医疗服务", "医药商业"],
    "公用事业": ["公用事业", "电力行业", "燃气", "水务", "供气供热"],
    "交通运输": ["物流行业", "航空机场", "航运港口", "铁路公路"],
    "房地产": ["房地产开发", "房地产服务"],
    "商贸零售": ["商业百货", "贸易行业", "旅游零售"],
    "社会服务": ["旅游酒店", "教育", "旅游景点", "酒店餐饮"],
    "银行": ["银行"],
    "非银金融": ["保险", "证券", "多元金融", "期货"],
    "综合": ["综合行业"],
    "建筑材料": ["水泥建材", "装修建材", "玻璃玻纤"],
    "建筑装饰": ["工程建设", "装修装饰", "工程咨询服务"],
    "电力设备": ["电池", "光伏设备", "风电设备", "电网设备", "电源设备", "电机"],
    "机械设备": ["工程机械", "专用设备", "通用设备", "仪器仪表"],
    "国防军工": ["航天航空", "船舶制造", "军工电子", "航天装备", "航空装备"],
    "计算机": ["软件开发", "计算机设备", "互联网服务", "IT服务"],
    "传媒": ["文化传媒", "游戏", "影视院线", "广告营销", "数字媒体"],
    "通信": ["通信设备", "通信服务"],
    "煤炭": ["煤炭行业"],
    "石油石化": ["石油行业", "油气开采", "油服工程"],
    "环保": ["环保行业", "环保设备", "环境治理"],
    "美容护理": ["美容护理", "化妆品"],
}


class _SectorMap(dict):
    """带状态说明的行业映射结果（dict 子类，兼容现有调用方）。"""

    def __init__(self, *args, status: str = "ok", note: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.status = status
        self.note = note


# ───────── 1. 行业分类 ─────────

def get_sw_industry_list() -> pd.DataFrame:
    """获取申万一级行业列表（含PE/PB/股息率）"""
    cached = _cache_get('sw_industry_list_v2')
    if cached is not None:
        return cached
    try:
        import akshare as ak
        df = ak.sw_index_first_info()
        if df is None or df.empty:
            return pd.DataFrame()
        # P2-Q27-fix(M344): 按列名显式映射重命名，缺失列置 None。
        # 原实现 `df.columns = names[:len(cols)]` 在任一列缺失时后续列名错位
        # （如缺"静态市盈率"→ pe 标签落到 TTM 列），现改为显式映射 + 缺列补 None。
        _COL_MAP = {
            '行业代码': 'code',
            '行业名称': 'name',
            '成份个数': 'count',
            '静态市盈率': 'pe',
            'TTM(滚动)市盈率': 'pe_ttm',
            '市净率': 'pb',
            '静态股息率': 'div_yield',
        }
        df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
        for col in _COL_MAP.values():
            if col not in df.columns:
                df[col] = None  # 缺失列显式置 None
        df = df[list(_COL_MAP.values())]  # 统一规范列序
        _cache_set('sw_industry_list_v2', df)
        return df
    except Exception as e:
        print(f"⚠️ 获取申万行业列表失败: {e}")
        return pd.DataFrame()


def get_stock_industry_map() -> dict:
    """获取个股→申万行业映射（返回 {股票代码: 行业名称}）

    P1-Q27-fix:
      - 申万行业名 ≠ 东财板块名，改用 _SW_TO_EM_BOARDS 做名称映射后再调
        stock_board_industry_cons_em，避免名称不匹配导致映射大量为空。
      - 部分板块获取失败时打印可见告警（不静默返回空）；整体失败返回带状态对象。
    """
    cached = _cache_get('stock_sw_map')
    if cached is not None:
        return cached

    mapping: dict = {}
    warnings: list[str] = []
    try:
        import akshare as ak
        # 从东方财富板块成分批量获取
        sw_list = get_sw_industry_list()
        for _, row in sw_list.iterrows():
            sw_name = row['name']
            board_names = _SW_TO_EM_BOARDS.get(sw_name, [sw_name])
            got_any = False
            for board in board_names:
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=board)
                    for _, c in cons.iterrows():
                        s_code = str(c['代码']).zfill(6)
                        if s_code.startswith(('600', '601', '603', '605',
                                              '000', '001', '002',
                                              '300', '301', '688')):
                            mapping[s_code] = sw_name  # 统一存申万行业名
                    got_any = True
                except Exception:
                    warnings.append(f"东财板块[{board}] 获取成分失败")
                _time.sleep(0.3)  # 控制请求频率
                if got_any:
                    break  # 该申万行业已获取到成分股，不再重复请求
            if not got_any:
                warnings.append(f"申万行业[{sw_name}] 未获取到任何东财成分股")
        _cache_set('stock_sw_map', mapping)
        if warnings:
            print(f"⚠️ 行业映射部分失败({len(warnings)}条)，已覆盖{len(mapping)}只股票; "
                  f"示例: {warnings[:3]}")
        return _SectorMap(mapping, status="partial" if warnings else "ok",
                          note=f"warnings={len(warnings)}, covered={len(mapping)}")
    except Exception as e:
        print(f"⚠️ 获取行业映射失败: {e}")
        return _SectorMap({}, status="error", note=str(e))


# ───────── 2. 行业指数表现 ─────────

def get_sw_index_realtime() -> pd.DataFrame:
    """获取申万一级行业实时行情

    P1-Q27-fix: akshare 1.18.64 的 index_realtime_sw() 返回 tidy 宽表
    (指数代码/指数名称/昨收盘/今开盘/最新价/成交额/成交量/最高价/最低价)，
    旧实现按"列名以801开头的宽表"解析导致恒为空。此处按行结构解析一级行业(801开头)。
    """
    cached = _cache_get('sw_index_realtime')
    if cached is not None:
        return cached
    try:
        import akshare as ak
        try:
            df = ak.index_realtime_sw(symbol="一级行业")
        except TypeError:
            df = ak.index_realtime_sw()  # 旧版本无 symbol 参数
        if df is None or df.empty:
            return pd.DataFrame()

        code_col = "指数代码" if "指数代码" in df.columns else df.columns[0]
        rows = []
        for _, r in df.iterrows():
            code = str(r.get(code_col, "")).strip()
            if not code.startswith("801"):
                continue  # 仅一级行业（801 开头）
            rows.append({
                'code': code,
                'name': r.get("指数名称", code),
                'close': _to_float(r.get("最新价")),
                'open': _to_float(r.get("今开盘")),
                'pre_close': _to_float(r.get("昨收盘")),
                'high': _to_float(r.get("最高价")),
                'low': _to_float(r.get("最低价")),
                'amount': _to_float(r.get("成交额")),
            })
        result = pd.DataFrame(rows)
        if not result.empty and 'pre_close' in result.columns and 'close' in result.columns:
            pre = result['pre_close'].replace(0, float('nan'))
            result['pct'] = ((result['close'] - result['pre_close']) / pre * 100).round(2)
        _cache_set('sw_index_realtime', result)
        return result
    except Exception as e:
        print(f"⚠️ 获取申万实时行情失败: {e}")
        return pd.DataFrame()


def get_sector_performance(days: int = 5) -> pd.DataFrame:
    """获取行业N日涨跌幅排名

    P1-Q27-fix:
      - days 现在真正生效：按 start/end 日期窗口计算 N 日涨跌幅（原来算的是上市以来全历史累计）。
      - index_hist_sw 签名只有 (symbol, period)，没有 start_date/end_date；
        且 symbol 需去掉 ".SI" 后缀（如 "801010"）。
      - 不再用 sw.head(10)，改用全部 31 个一级行业。
    """
    try:
        import akshare as ak
        sw = get_sw_industry_list()
        if sw is None or sw.empty:
            return pd.DataFrame()
        now = datetime.now(CST)
        end = now.strftime('%Y%m%d')
        # V11 审计修复（Medium）: 原实现 days+30 日历日 ≈ days+21 交易日，
        # "5日涨跌幅"实为约 26 日涨跌，名实不符。
        # 改为按自然日 ×1.5 宽松窗口拉取，再在下方用 tail(days+1) 精确截取 N 个交易日。
        start = (now - timedelta(days=max(15, int(days * 1.5) + 10))).strftime('%Y%m%d')
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        results = []
        for _, row in sw.iterrows():  # 全量一级行业
            try:
                idx_code = str(row['code']).replace('.SI', '')
                hist = ak.index_hist_sw(symbol=idx_code, period="day")
                if hist is None or len(hist) <= 1:
                    continue
                hist = hist.copy()
                hist["日期"] = pd.to_datetime(hist["日期"])
                window = hist[(hist["日期"] >= start_ts) & (hist["日期"] <= end_ts)]
                # V11 审计修复: 精确截取最近 N+1 个交易日（含当日），保证"N日涨跌幅"名实一致
                window = window.tail(days + 1)
                if len(window) < 2:
                    # 数据不足则退而求其次用最近两个交易日
                    window = hist.tail(2)
                closes = window["收盘"].astype(float)
                pct = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
                # 计算RS（窗口内日收益年化）
                ret = window["收盘"].astype(float).pct_change().dropna()
                rs = ret.mean() / ret.std() * np.sqrt(252) if len(ret) > 1 and ret.std() > 0 else 0
                results.append({
                    'name': row['name'],
                    'chg_pct': round(pct, 2),
                    'rs': round(rs, 2),
                    'pe_ttm': row.get('pe_ttm', None),
                    'pb': row.get('pb', None),
                })
                _time.sleep(0.2)
            except Exception as e:
                logging.getLogger(__name__).error(f"[sector_rotation] 操作失败: {e}", exc_info=True)
                continue

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values('chg_pct', ascending=False)
    except Exception as e:
        print(f"⚠️ 获取行业表现失败: {e}")
        return pd.DataFrame()


# ───────── 3. 资金流向 ─────────

def get_sector_fund_flow(indicator: str = "今日") -> pd.DataFrame:
    """获取行业资金流向排名

    Args:
        indicator: "今日", "5日", "10日"

    Returns:
        DataFrame: 行业资金流排名
    """
    cache_key = f'sector_fund_flow_{indicator}'
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        import akshare as ak
        df = ak.stock_sector_fund_flow_rank(
            indicator=indicator,
            sector_type="行业资金流"
        )
        # 标准化列名
        col_map = {
            '名称': 'name', '最新价': 'price', '涨跌幅': 'pct',
            '今日主力净流入-净额': 'main_net_inflow',
            '今日主力净流入-净占比': 'main_net_ratio',
            '今日超大单净流入-净额': 'super_large_net',
            '今日大单净流入-净额': 'large_net',
            '今日中单净流入-净额': 'medium_net',
            '今日小单净流入-净额': 'small_net',
            '今日主力净流入最大股': 'top_stock',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df['indicator'] = indicator
        _cache_set(cache_key, df)
        return df
    except Exception as e:
        print(f"⚠️ 获取{indicator}资金流向失败: {e}")
        return pd.DataFrame()


def get_sector_fund_flow_summary() -> dict:
    """汇总今日/5日/10日资金流向前三和后三"""
    result = {}
    for period in ["今日", "5日", "10日"]:
        df = get_sector_fund_flow(period)
        if df.empty:
            continue
        if 'main_net_inflow' in df.columns:
            top3 = df.nlargest(3, 'main_net_inflow')[['name', 'main_net_inflow', 'main_net_ratio']]
            bottom3 = df.nsmallest(3, 'main_net_inflow')[['name', 'main_net_inflow', 'main_net_ratio']]
            result[period] = {
                'top3': top3.to_dict('records'),
                'bottom3': bottom3.to_dict('records'),
            }
    return result


# ───────── 4. 轮动分析 ─────────

def calc_rotation_momentum(days: int = 60) -> pd.DataFrame:
    """计算行业轮动动量

    公式:
      momentum = σ(最近N日涨跌幅) × sign(趋势方向)
      rotation_speed = |momentum_diff| / days

    P1-Q27-fix:
      - index_hist_sw 无 start_date/end_date 参数（1.18.64 签名仅 symbol/period），
        改为拉全量后按日期窗口过滤，避免 TypeError 被吞导致恒空。
      - acceleration 原实现用 df['rotation'].diff() 在行业横截面上做时间序列差分（无意义），
        改为按时间序列窗口计算：acceleration = 本期5日动量 − 上一期5日动量。
    """
    try:
        sw = get_sw_industry_list()
        if sw is None or sw.empty:
            return pd.DataFrame()
        start_dt = datetime.now(CST) - timedelta(days=days + 30)
        end_dt = datetime.now(CST)
        start_ts = pd.Timestamp(start_dt.strftime('%Y%m%d'))
        end_ts = pd.Timestamp(end_dt.strftime('%Y%m%d'))

        sectors = []
        for _, row in sw.iterrows():  # 全量一级行业
            try:
                import akshare as ak
                idx_code = str(row['code']).replace('.SI', '')
                hist = ak.index_hist_sw(symbol=idx_code, period="day")
                if hist is None or len(hist) <= 5:
                    continue
                hist = hist.copy()
                hist["日期"] = pd.to_datetime(hist["日期"])
                hist = hist[(hist["日期"] >= start_ts) & (hist["日期"] <= end_ts)].reset_index(drop=True)
                if len(hist) <= 5:
                    continue
                ret = hist['收盘'].astype(float).pct_change().dropna()
                mom_short = ret.tail(5).mean() * 100  # 5日动量
                mom_long = ret.tail(20).mean() * 100  # 20日动量
                # acceleration = 本期5日 − 上一期5日（同一行业的时间序列窗口对比）
                mom_prev = ret.iloc[-10:-5].mean() * 100 if len(ret) >= 10 else mom_short
                sectors.append({
                    'name': row['name'],
                    'momentum_5d': round(mom_short, 3),
                    'momentum_20d': round(mom_long, 3),
                    'rotation': round(mom_short - mom_long, 3),
                    'acceleration': round(mom_short - mom_prev, 3),
                    'volatility': round(ret.tail(20).std() * np.sqrt(252) * 100, 1),
                })
                _time.sleep(0.2)
            except Exception as e:
                logging.getLogger(__name__).error(f"[sector_rotation] 操作失败: {e}", exc_info=True)
                continue

        df = pd.DataFrame(sectors)
        if not df.empty:
            df = df.sort_values('rotation', ascending=False)
        return df
    except Exception as e:
        print(f"⚠️ 计算轮动动量失败: {e}")
        return pd.DataFrame()


# ───────── 5. 行业轮动图（表格版） ─────────

def build_rotation_heatmap() -> pd.DataFrame:
    """构建行业轮动热度矩阵（各周期涨跌幅排序）"""
    periods = {'1日': 1, '5日': 5, '20日': 20, '60日': 60}
    heatmap = {}
    try:
        sw = get_sw_industry_list()
        for label, days in periods.items():
            df = get_sector_performance(days=days)
            if not df.empty:
                heatmap[label] = dict(zip(df['name'], df['chg_pct']))
        result = pd.DataFrame(heatmap)
        return result.sort_index()
    except Exception:
        return pd.DataFrame()


# ───────── 6. 格式化输出 ─────────

def format_rank(df: pd.DataFrame, title: str = "") -> str:
    """格式化排名输出"""
    lines = []
    if title:
        lines.append(f"\n## {title}\n")
    if df.empty:
        return f"⚠️ 暂无{title}数据"
    lines.append(f"{'行业':<12}{'涨跌幅%':>8}{'PE_TTM':>8}{'PB':>8}{'RS':>8}")
    lines.append("-" * 48)
    for _, r in df.iterrows():
        name = str(r.get('name', ''))[:10]
        chg = r.get('chg_pct', 0)
        pe = r.get('pe_ttm', '-')
        pb = r.get('pb', '-')
        rs = r.get('rs', '-')
        arrow = "🟢" if chg > 0 else "🔴"
        lines.append(f"{arrow} {name:<10} {chg:>+8.2f} {str(pe):>8} {str(pb):>8} {str(rs):>8}")
    return "\n".join(lines)


def format_fund_flow(summary: dict) -> str:
    """格式化资金流向"""
    lines = ["\n## 💰 行业资金流向排名\n"]
    if not summary:
        return "⚠️ 暂无资金流向数据"

    for period, data in summary.items():
        lines.append(f"\n📅 {period}主力净流入:")
        for label, items in [("🟢 流入TOP3", data.get('top3', [])),
                             ("🔴 流出TOP3", data.get('bottom3', []))]:
            for item in items:
                name = item.get('name', '')
                inflow = item.get('main_net_inflow', 0)
                ratio = item.get('main_net_ratio', 0)
                if isinstance(inflow, (int, float)):
                    inflow_str = f"{inflow / 1e8:.2f}亿"
                else:
                    inflow_str = str(inflow)
                ratio_str = f"{ratio:.1f}%" if isinstance(ratio, (int, float)) else str(ratio)
                lines.append(f"  {label}: {name} {inflow_str} (占比{ratio_str})")
    return "\n".join(lines)


def format_momentum(df: pd.DataFrame) -> str:
    """格式化轮动动量"""
    lines = ["\n## 🔄 行业轮动动量分析\n"]
    if df.empty:
        return "⚠️ 暂无轮动数据"
    lines.append(f"{'行业':<12}{'5日动量':>10}{'20日动量':>10}{'轮动力度':>10}{'加速/减速':>10}")
    lines.append("-" * 52)
    for _, r in df.iterrows():
        name = str(r.get('name', ''))[:10]
        m5 = r.get('momentum_5d', 0)
        m20 = r.get('momentum_20d', 0)
        rot = r.get('rotation', 0)
        acc = r.get('acceleration', 0)
        acc_symbol = "🚀" if acc > 0 else "🐢"
        lines.append(f"  {name:<10} {m5:>+10.3f} {m20:>+10.3f} {rot:>+10.3f} {acc_symbol} {acc:>+.3f}")
    return "\n".join(lines)


def _col_mean_str(df: pd.DataFrame, col: str, ndigits: int = 1, suffix: str = "") -> str:
    """防御性列均值格式化：列缺失/全空/类型异常时返回 '-'（P2-Q27-fix M345）。"""
    if df is None or df.empty or col not in df.columns:
        return "-"
    vals = df[col].dropna()
    if vals.empty:
        return "-"
    try:
        return f"{vals.mean():.{ndigits}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def format_full_report() -> str:
    """生成完整行业轮动报告"""
    parts = []
    now = datetime.now(CST)
    parts.append(f"# 🗂️ 行业轮动报告 ({now.strftime('%Y-%m-%d %H:%M')})")
    parts.append(f"\n{'='*50}")

    # 1. 行业实时表现
    # P2-Q27-fix(M345): 原直接 sw['pe_ttm']/sw['pb']/sw['div_yield'] 索引，
    # 列缺失时 KeyError；改用 df.get 防御。
    sw = get_sw_industry_list()
    parts.append(f"\n## 1️⃣ 申万一级行业概览")
    parts.append(f"共 {len(sw)} 个行业 | "
                 f"PE均值={_col_mean_str(sw, 'pe_ttm')} | "
                 f"PB均值={_col_mean_str(sw, 'pb')} | "
                 f"平均股息率={_col_mean_str(sw, 'div_yield', 2, '%')}")

    # 2. 资金流向
    fund = get_sector_fund_flow_summary()
    parts.append(format_fund_flow(fund))

    # 3. 轮动动量
    momentum = calc_rotation_momentum()
    parts.append(format_momentum(momentum))

    # 4. 推荐关注
    parts.append("\n## 💡 当前关注信号\n")
    if not momentum.empty:
        top_rot = momentum.head(3)
        parts.append("🟢 轮动加速(强势): " + ", ".join(top_rot['name'].tolist()))
        bottom_rot = momentum.tail(3)
        parts.append("🔴 轮动减速(弱势): " + ", ".join(bottom_rot['name'].tolist()))

    return "\n".join(parts)


# ───────── 7. CLI ─────────

def main():
    """CLI入口"""
    args = sys.argv[1:]

    if "--rank" in args:
        df = get_sector_performance(days=5)
        print(format_rank(df, "申万行业5日涨跌幅排名"))
    elif "--fund-flow" in args:
        summary = get_sector_fund_flow_summary()
        print(format_fund_flow(summary))
    elif "--momentum" in args:
        df = calc_rotation_momentum()
        print(format_momentum(df))
    elif "--rotation-map" in args:
        hm = build_rotation_heatmap()
        if not hm.empty:
            print("\n## 🔥 行业轮动热度矩阵\n")
            print(hm.to_string())
    else:
        print(format_full_report())


if __name__ == "__main__":
    main()
