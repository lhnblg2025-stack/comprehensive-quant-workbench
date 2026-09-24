"""
K线形态识别 — 28种经典K线形态自动识别

功能:
  1. 单K形态: 锤子线/吊颈线/十字星/纺锤线/长阳/长阴/倒锤子/墓碑
  2. 双K形态: 看涨吞没/看跌吞没/刺透线/乌云盖顶/孕线/十字孕线
  3. 三K形态: 晨星/暮星/三白兵/三乌鸦/三内部上升/三内部下降
  4. 持续形态: 向上跳空/向下跳空/上升三法/下降三法
  5. 批量扫描: 全自选股扫描，输出形态出现频率

用法:
  python3 -m quant_system.chart_patterns             # 全量扫描
  python3 -m quant_system.chart_patterns --stock 600519  # 单票分析
  python3 -m quant_system.chart_patterns --bullish    # 只看买入形态
  python3 -m quant_system.chart_patterns --bearish    # 只看卖出形态
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# ───────── 形态定义 ─────────
PATTERN_CATEGORIES = {
    '单K线': ['锤子线', '吊颈线', '十字星', '长阳线', '长阴线', '倒锤子', '墓碑线', '纺锤线'],
    '双K线': ['看涨吞没', '看跌吞没', '刺透线', '乌云盖顶', '孕线', '十字孕线'],
    '三K线': ['晨星', '暮星', '三白兵', '三乌鸦', '三内部上升', '三内部下降'],
    '持续形态': ['向上跳空', '向下跳空', '上升三法(5日)', '下降三法(5日)'],
}


@dataclass
class PatternMatch:
    name: str          # 形态名称
    direction: int     # 1=看涨, -1=看跌, 0=中性
    confidence: float  # 0-1
    description: str   # 说明
    details: dict = field(default_factory=dict)


# ───────── 单K形态识别 ─────────
def _body_length(o, c, h, l):
    """K线实体长度占比"""
    body = abs(c - o)
    total = h - l
    if total == 0:
        return 0
    return body / total


def _upper_shadow(o, c, h):
    """上影线长度占比"""
    top = max(o, c)
    shadow = h - top
    total = h - min(o, c)
    if total == 0:
        return 0
    return shadow / total


def _lower_shadow(o, c, l):
    """下影线长度占比"""
    bot = min(o, c)
    shadow = bot - l
    total = max(o, c) - l
    if total == 0:
        return 0
    return shadow / total


def _is_bullish(o, c):
    return c >= o


def _is_bearish(o, c):
    return c < o


def identify_single_bar(open_p, close_p, high_p, low_p) -> list[PatternMatch]:
    """识别单根K线形态。

    P2-Q28-fix(M361): 单根 K 线只归类一种形态，doji 分支与独立锤/吊/倒锤分支互斥。
    原实现 doji 分支与独立锤子/吊颈/倒锤分支各自用独立 if 判定，导致：
      十字阳锤 → 同时输出两条"锤子线"（L102 与 L120）；
      十字阴吊 → 两条"吊颈线"；
      墓碑十字 → 同时输出"墓碑线(-1)"与"倒锤子(+1)"方向相反。
    现按 body_ratio 分段互斥判定，同一根 K 线至多返回一种形态。
    """
    results = []
    body_ratio = _body_length(open_p, close_p, high_p, low_p)
    upper = _upper_shadow(open_p, close_p, high_p)
    lower = _lower_shadow(open_p, close_p, low_p)
    bullish = _is_bullish(open_p, close_p)
    change_pct = (close_p / open_p - 1) * 100

    if body_ratio < 0.1:
        # ── 十字星系（互斥：只取一种）──
        if upper > 0.5 and lower < 0.1:
            results.append(PatternMatch("墓碑线", -1, 0.7, "上影线极长的十字星，顶部反转", {'body_ratio': round(body_ratio, 3)}))
        elif upper < 0.1 and lower > 0.5:
            if bullish:
                results.append(PatternMatch("锤子线", 1, 0.7, "下影线极长的十字星，底部反转", {'body_ratio': round(body_ratio, 3)}))
            else:
                results.append(PatternMatch("吊颈线", -1, 0.65, "下影线较长的小阴实体十字星，顶部反转", {'body_ratio': round(body_ratio, 3)}))
        elif upper > 0.4 and lower > 0.4:
            results.append(PatternMatch("纺锤线", 0, 0.4, "上下影线等长的高位/低位十字"))
        else:
            results.append(PatternMatch("十字星", 0, 0.5, "开盘=收盘，趋势犹豫"))
    elif body_ratio < 0.3:
        # ── 非十字的小实体（互斥：锤/吊/倒锤只取一种）──
        if lower > 0.6 and upper < 0.1:
            if bullish:
                results.append(PatternMatch("锤子线", 1, 0.75, "小阳实体+长下影，底部反转"))
            else:
                results.append(PatternMatch("吊颈线", -1, 0.65, "小阴实体+长下影，顶部反转"))
        elif bullish and upper > 0.6 and lower < 0.1:
            results.append(PatternMatch("倒锤子", 1, 0.6, "小阳实体+长上影，底部反转"))
    elif bullish and abs(change_pct) > 3 and body_ratio > 0.7:
        # 长阳线
        results.append(PatternMatch("长阳线", 1, 0.8, f"涨幅{change_pct:.1f}%的实体长阳", {'change_pct': round(change_pct, 2)}))
    elif not bullish and abs(change_pct) > 3 and body_ratio > 0.7:
        # 长阴线
        results.append(PatternMatch("长阴线", -1, 0.8, f"跌幅{change_pct:.1f}%的实体长阴", {'change_pct': round(change_pct, 2)}))

    return results


# ───────── 双K形态识别 ─────────
def identify_two_bars(bars: list[dict]) -> list[PatternMatch]:
    """识别双K形态（需要2根K线）"""
    results = []
    if len(bars) < 2:
        return results

    b1 = bars[-2]  # 前一根
    b2 = bars[-1]  # 当前

    o1, c1, h1, l1 = b1['open'], b1['close'], b1['high'], b1['low']
    o2, c2, h2, l2 = b2['open'], b2['close'], b2['high'], b2['low']

    b1_bull = _is_bullish(o1, c1)
    b2_bull = _is_bullish(o2, c2)
    b1_body = abs(c1 - o1)
    b2_body = abs(c2 - o2)

    # 看涨吞没 (Bullish Engulfing)
    if not b1_bull and b2_bull and o2 < c1 and c2 > o1 and b2_body > b1_body * 1.2:
        results.append(PatternMatch("看涨吞没", 1, 0.85, "阳线实体完全吞没前一个阴线", {
            'bar1_body': round(b1_body / (h1-l1), 2) if h1 != l1 else 0,
            'bar2_body': round(b2_body / (h2-l2), 2) if h2 != l2 else 0,
        }))

    # 看跌吞没 (Bearish Engulfing)
    if b1_bull and not b2_bull and o2 > c1 and c2 < o1 and b2_body > b1_body * 1.2:
        results.append(PatternMatch("看跌吞没", -1, 0.85, "阴线实体完全吞没前一个阳线"))

    # 刺透线 (Piercing Line)
    if not b1_bull and b2_bull and c2 > (o1 + c1) / 2 and c2 < o1:
        results.append(PatternMatch("刺透线", 1, 0.7, "阳线收盘价进入前阴线实体中部以上"))

    # 乌云盖顶 (Dark Cloud Cover)
    if b1_bull and not b2_bull and c2 < (o1 + c1) / 2 and c2 > o1:
        results.append(PatternMatch("乌云盖顶", -1, 0.7, "阴线收盘价进入前阳线实体中部以下"))

    # 孕线 (Harami)
    if b2_body < b1_body * 0.6 and ((b1_bull and not b2_bull) or (not b1_bull and b2_bull)):
        results.append(PatternMatch("孕线", 0, 0.5, "小实体被前一大实体包裹，趋势犹豫"))

    # 十字孕线 (Harami Cross)
    b2_body_ratio = _body_length(o2, c2, h2, l2)
    if b2_body_ratio < 0.1 and b2_body < b1_body * 0.5:
        results.append(PatternMatch("十字孕线", 0, 0.6, "十字星被前一大实体包裹，强烈反转信号"))

    return results


# ───────── 三K形态识别 ─────────
def identify_three_bars(bars: list[dict]) -> list[PatternMatch]:
    """识别三K形态（需要3根K线）"""
    results = []
    if len(bars) < 3:
        return results

    b1, b2, b3 = bars[-3], bars[-2], bars[-1]
    o1, c1 = b1['open'], b1['close']
    o2, c2 = b2['open'], b2['close']
    o3, c3 = b3['open'], b3['close']

    b1_bull = _is_bullish(o1, c1)
    b2_bull = _is_bullish(o2, c2)
    b3_bull = _is_bullish(o3, c3)

    # 晨星 (Morning Star): 长阴→小实体→长阳，有跳空
    if not b1_bull and b3_bull:
        b1_body_pct = abs(c1 - o1) / (b1['high'] - b1['low']) if b1['high'] != b1['low'] else 0
        b3_body_pct = abs(c3 - o3) / (b3['high'] - b3['low']) if b3['high'] != b3['low'] else 0
        if b1_body_pct > 0.5 and b3_body_pct > 0.5:
            if c3 > (o1 + c1) / 2:  # 第三根阳线进入阴线实体
                results.append(PatternMatch("晨星", 1, 0.9, "长阴→星线→长阳，底部反转"))

    # 暮星 (Evening Star): 长阳→小实体→长阴
    if b1_bull and not b3_bull:
        b1_body_pct = abs(c1 - o1) / (b1['high'] - b1['low']) if b1['high'] != b1['low'] else 0
        b3_body_pct = abs(c3 - o3) / (b3['high'] - b3['low']) if b3['high'] != b3['low'] else 0
        if b1_body_pct > 0.5 and b3_body_pct > 0.5:
            if c3 < (o1 + c1) / 2:  # 第三根阴线进入阳线实体
                results.append(PatternMatch("暮星", -1, 0.9, "长阳→星线→长阴，顶部反转"))

    # 三白兵 (Three White Soldiers)
    if b1_bull and b2_bull and b3_bull:
        c1_pct, c2_pct, c3_pct = (c1/o1-1)*100, (c2/o2-1)*100, (c3/o3-1)*100
        if all(p > 0.5 for p in [c1_pct, c2_pct, c3_pct]) and all(p < 5 for p in [c1_pct, c2_pct, c3_pct]):
            if o2 > o1 and o3 > o2:
                results.append(PatternMatch("三白兵", 1, 0.75, "连续三根阳线稳步上涨"))
            elif o2 < c1 and o3 < c2:
                results.append(PatternMatch("三内部上升", 1, 0.6, "强势三白兵变体"))

    # 三乌鸦 (Three Black Crows)
    if not b1_bull and not b2_bull and not b3_bull:
        if all((o1-c1)/o1*100 < 5 for o,c in [(o1,c1),(o2,c2),(o3,c3)]):  # 跌幅不过大
            results.append(PatternMatch("三乌鸦", -1, 0.75, "连续三根阴线稳步下跌"))
            if o2 > o1 and o3 > o2:
                results.append(PatternMatch("三内部下降", -1, 0.6, "弱势三乌鸦变体"))

    return results


# ───────── 持续形态 ─────────
def identify_continuation(bars: list[dict]) -> list[PatternMatch]:
    """识别持续形态"""
    results = []
    if len(bars) < 6:
        return results

    # 跳空缺口 (至少需要连续K线)
    if len(bars) >= 2:
        b2 = bars[-2]
        b1 = bars[-1]
        gap_up = b1['low'] > b2['high']
        gap_down = b1['high'] < b2['low']
        if gap_up:
            results.append(PatternMatch("向上跳空", 1, 0.6, f"跳空上涨({(b1['low']/b2['high']-1)*100:.2f}%)"))
        if gap_down:
            results.append(PatternMatch("向下跳空", -1, 0.6, f"跳空下跌({(b1['high']/b2['low']-1)*100:.2f}%)"))

    # 上升三法/下降三法 (Rising/Falling Three Methods)
    if len(bars) >= 5:
        b5 = bars[-5]
        b4 = bars[-4]
        b3 = bars[-3]
        b2 = bars[-2]
        b1 = bars[-1]

        b5_bull = _is_bullish(b5['open'], b5['close'])
        b1_bull = _is_bullish(b1['open'], b1['close'])

        # 上升三法: 长阳→3小阴→长阳
        if b5_bull and b1_bull:
            middle_bearish = sum(1 for i in range(3) if not _is_bullish(bars[-4+i]['open'], bars[-4+i]['close']))
            if middle_bearish >= 2:
                results.append(PatternMatch("上升三法(5日)", 1, 0.7, "长阳→调整→长阳，中继上涨"))

        # 下降三法: 长阴→3小阳→长阴
        if not b5_bull and not b1_bull:
            middle_bullish = sum(1 for i in range(3) if _is_bullish(bars[-4+i]['open'], bars[-4+i]['close']))
            if middle_bullish >= 2:
                results.append(PatternMatch("下降三法(5日)", -1, 0.7, "长阴→反弹→长阴，中继下跌"))

    return results


# ───────── 完整识别 ─────────
def identify_all_patterns(bars: list[dict]) -> list[PatternMatch]:
    """完整识别所有K线形态"""
    patterns = []
    patterns.extend(identify_single_bar(bars[-1]['open'], bars[-1]['close'],
                                        bars[-1]['high'], bars[-1]['low']))
    patterns.extend(identify_two_bars(bars))
    patterns.extend(identify_three_bars(bars))
    patterns.extend(identify_continuation(bars))
    return patterns


# ───────── 股票分析 ─────────
def analyze_stock(code: str, days: int = 120) -> dict:
    """分析单只股票的最新K线形态"""
    try:
        import akshare as ak
        end = datetime.now(CST).strftime('%Y%m%d')
        start = (datetime.now(CST) - timedelta(days=days)).strftime('%Y%m%d')
        hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   start_date=start, end_date=end,
                                   adjust="qfq")
        if hist is None or len(hist) < 5:
            return {'code': code, 'patterns': [], 'error': '数据不足'}

        bars = []
        for _, row in hist.iterrows():
            bars.append({
                'date': str(row['日期'])[:10],
                'open': float(row['开盘']),
                'close': float(row['收盘']),
                'high': float(row['最高']),
                'low': float(row['最低']),
                # P2-Q28-fix(L362): akshare stock_zh_a_hist 成交量单位为"手"（1手=100股），
                # 按项目标准统一为"股"。
                'volume': float(row['成交量']) * 100,
            })

        patterns = identify_all_patterns(bars)
        prices = hist[['日期', '开盘', '收盘', '最高', '最低']].tail(5)
        return {
            'code': code,
            'name': hist['股票简称'].iloc[-1] if '股票简称' in hist.columns else '',
            'current_price': bars[-1]['close'],
            'patterns': patterns,
            'recent_bars': prices.to_dict('records') if hasattr(prices, 'to_dict') else [],
        }
    except Exception as e:
        return {'code': code, 'patterns': [], 'error': str(e)}


def scan_all_stocks(max_codes: int | None = None,
                    checkpoint_file: str | None = None,
                    retries: int = 2) -> list[dict]:
    """扫描全部自选股的K线形态。

    P2-Q28-fix(L362):
      1) 原实现仅扫 watchlist 前 50 只（stocks[:50]），现支持全量扫描，
         max_codes 可限制数量（None=全部）；
      2) 支持断点续扫：checkpoint_file 记录已扫股票代码，中断后重跑自动跳过；
      3) 单票失败重试 retries 次，失败股票跳过而非中断全流程。
    """
    results = []
    try:
        from quant_system.watchlist import get_watchlist
        stocks = get_watchlist()
        # V11 审计修复（High）: get_watchlist 返回键是 symbol 不是 code，
        # 原实现恒取空串 → 扫描功能静默失效（409只×3次失败API调用）。
        codes = [str(s.get('symbol') or s.get('code') or '') for s in stocks]
    except ImportError:
        codes = ["600519", "601288", "600036"]

    if max_codes is not None:
        codes = codes[:max_codes]

    # 断点续扫：读取已完成的股票代码
    done: set[str] = set()
    if checkpoint_file:
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                done = {line.strip() for line in f if line.strip()}
        except OSError:
            done = set()

    import time as _time
    for code in codes:
        if code in done:
            continue
        result = None
        for attempt in range(retries + 1):
            result = analyze_stock(code)
            if not result.get('error') or attempt == retries:
                break
            _time.sleep(0.5 * (attempt + 1))  # 退避重试
        if result.get('patterns'):
            results.append(result)
        if checkpoint_file:
            try:
                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    f.write(f"{code}\n")
            except OSError:
                pass
        _time.sleep(0.3)

    return results


# ───────── 格式化 ─────────
def format_stock_analysis(result: dict) -> str:
    code = result.get('code', '')
    name = result.get('name', '')
    price = result.get('current_price', 0)
    patterns = result.get('patterns', [])

    lines = [f"🕯️ {name}({code[:6]}) | 现价: {price:.2f}"]
    if not patterns:
        lines.append("  无显著K线形态")
        return "\n".join(lines)

    for p in patterns:
        dir_icon = "🟢" if p.direction == 1 else ("🔴" if p.direction == -1 else "⚪")
        conf = "★" * int(p.confidence * 5) + "☆" * (5 - int(p.confidence * 5))
        lines.append(f"  {dir_icon} {p.name:<10} {conf} {p.description[:40]}")

    return "\n".join(lines)


def format_scan_report(results: list[dict]) -> str:
    bullish = [r for r in results if any(p.direction == 1 for p in r.get('patterns', []))]
    bearish = [r for r in results if any(p.direction == -1 for p in r.get('patterns', []))]

    now = datetime.now(CST)
    lines = [f"# 🕯️ K线形态扫描 ({now.strftime('%Y-%m-%d %H:%M')})"]
    lines.append(f"{'=' * 50}")

    lines.append(f"\n🟢 买入形态 ({len(bullish)}只)")
    for r in bullish[:10]:
        lines.append(format_stock_analysis(r))

    lines.append(f"\n🔴 卖出形态 ({len(bearish)}只)")
    for r in bearish[:10]:
        lines.append(format_stock_analysis(r))

    # 形态统计 (去重)
    pattern_counts = {}
    for r in results:
        for p in r.get('patterns', []):
            pattern_counts[p.name] = pattern_counts.get(p.name, 0) + 1
    if pattern_counts:
        lines.append(f"\n📊 形态分布")
        for name, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {name}: {count}家")

    total = len(results)
    scanned = total
    lines.append(f"\n扫描范围: {scanned}只 | 有形态: {len(bullish)+len(bearish)}只")
    return "\n".join(lines)


# ───────── CLI ─────────
def main():
    args = sys.argv[1:]

    if any(a.startswith('--stock') for a in args):
        idx = args.index('--stock')
        code = args[idx + 1] if idx + 1 < len(args) else "600519"
        result = analyze_stock(code)
        print(format_stock_analysis(result))
    elif "--bullish" in args:
        results = scan_all_stocks()
        for r in results:
            bullish = [p for p in r.get('patterns', []) if p.direction == 1]
            if bullish:
                print(format_stock_analysis(r))
    elif "--bearish" in args:
        results = scan_all_stocks()
        for r in results:
            bearish = [p for p in r.get('patterns', []) if p.direction == -1]
            if bearish:
                print(format_stock_analysis(r))
    else:
        results = scan_all_stocks()
        print(format_scan_report(results))


if __name__ == "__main__":
    main()
