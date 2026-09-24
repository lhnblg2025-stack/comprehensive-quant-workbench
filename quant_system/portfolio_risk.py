"""
组合级风控 — VaR、集中度、Kelly仓位、最大回撤预警

功能:
  1. VaR/CVaR计算: 持仓组合历史模拟法VaR(95%/99%)
  2. 集中度检查: 行业集中度/个股集中度/前5权重占比
  3. Kelly仓位: 根据历史胜率/盈亏比计算最优仓位
  4. 最大回撤预警: 组合回撤达阈值触发降仓建议
  5. 相关性分析: 持仓间的相关性矩阵
  6. 压力测试: 模拟大跌/大涨场景

用法:
  python3 -m quant_system.portfolio_risk           # 全量风控报告
  python3 -m quant_system.portfolio_risk --var      # VaR分析
  python3 -m quant_system.portfolio_risk --concentration  # 集中度
  python3 -m quant_system.portfolio_risk --correlation     # 相关性

D6收敛登记 (2026-08-11): 风控域收敛（保守策略）——
  1. compute_var = 组合级历史模拟VaR（akshare自取数、日期对齐联合分布、百分数多档输出），
     与 risk_management_pro._historical_var_cvar / portfolio_v2._historical_cvar_from_returns /
     portfolio_optimizer.cvar_optimize 异名异签名异口径 → 标注保留，不强迁。
  2. compute_concentration / compute_drawdown / compute_kelly_ratio(系) 与
     risk_management_pro.check_diversification / _max_drawdown_from_returns /
     portfolio_v2.KellyCriterion 同名近名异口径（输入/量纲/符号方向不同）→ 标注保留。
  3. compute_correlation / compute_ml_risk_score / full_risk_report / format_risk_report =
     独立能力保留（相关性矩阵/ML市场风险评分/报告层，无等价实现）。
"""

from __future__ import annotations
import logging

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# ⚠️ 预警阈值
VAR_THRESHOLD = 0.03           # 日VaR > 3%触发预警
CONCENTRATION_THRESHOLD = 0.3  # 单票 > 30%预警
INDUSTRY_CONCENTRATION = 0.4   # 单行业 > 40%预警
DRAWDOWN_CAUTION = 0.08        # 回撤8%注意
DRAWDOWN_WARNING = 0.15        # 回撤15%警告
DRAWDOWN_DANGER = 0.25         # 回撤25%危险


def load_positions_from_db() -> list[dict]:
    """从交易数据库加载持仓"""
    try:
        from quant_system.trade_db import TradeDB
        db = TradeDB()
        positions = db.get_positions()
        if positions:
            return positions
    except Exception as e:
        logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)

    # 尝试从仿真引擎加载
    try:
        sim_file = ROOT.parent / "config" / "simulation_state.json"
        if sim_file.exists():
            data = json.loads(sim_file.read_text())
            positions = []
            for code, pos in data.get('positions', {}).items():
                positions.append({
                    'stock': code,
                    'stock_name': pos.get('stock_name', ''),
                    'qty': pos.get('qty', 0),
                    'cost': pos.get('cost', 0),
                    'current_price': pos.get('current_price', 0),
                    'market_value': pos.get('qty', 0) * pos.get('current_price', 0),
                })
            return positions
    except Exception as e:
        logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)

    return []


def get_stock_sector(code: str) -> str:
    """获取股票所属行业"""
    try:
        import akshare as ak
        info = ak.stock_individual_info_em(symbol=code)
        if info is not None:
            d = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
            return d.get('行业', '未知')
    except Exception as e:
        logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)
    return '未知'


# ───────── VaR计算 ─────────
def compute_var(positions: list[dict], confidence: float = 0.95) -> dict:
    """计算组合VaR（历史模拟法）

    D6收敛: 异名异签名异口径保留 —— 与 risk_management_pro._historical_var_cvar /
    portfolio_v2._historical_cvar_from_returns / portfolio_optimizer.cvar_optimize 的
    VaR/CVaR 口径不同（自取数/日期对齐联合分布/百分数多档输出）。

    Args:
        positions: 持仓列表
        confidence: 置信度 0.95/0.99

    Returns:
        dict: VaR报告
    """
    if not positions:
        return {'var': 0, 'cvar': 0, 'total_value': 0, 'note': '无持仓'}

    total_value = sum(
        p.get('qty', 0) * p.get('current_price', 0) for p in positions
    )
    if total_value <= 0:
        return {'var': 0, 'cvar': 0, 'total_value': 0}

    # 获取每只持仓的历史收益率（带日期，供对齐）
    ret_by_code: dict[str, pd.Series] = {}
    weights_by_code: dict[str, float] = {}
    for pos in positions:
        code = str(pos.get('stock', ''))
        try:
            import akshare as ak
            hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                       start_date=(datetime.now(CST)-timedelta(days=252)).strftime('%Y%m%d'),
                                       end_date=datetime.now(CST).strftime('%Y%m%d'),
                                       adjust="qfq")
            if hist is not None and len(hist) > 20 and '日期' in hist.columns:
                s = pd.Series(hist['收盘'].astype(float).values,
                              index=hist['日期'].astype(str).str[:10]).sort_index()
                s = s[~s.index.duplicated(keep='last')]
                ret = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                if len(ret) > 20:
                    ret_by_code[code] = -ret.astype(float)  # 亏损方向
                    weights_by_code[code] = (pos.get('qty', 0) * pos.get('current_price', 0)) / total_value
        except Exception as e:
            logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)
            continue

    if not ret_by_code:
        return {'var': 0, 'cvar': 0, 'total_value': total_value, 'note': '无历史数据'}

    # 组合加权亏损：按日期对齐成面板再逐日加权求和
    # P2-Q16-fix (M117): 旧实现各股独立取"最后 min_len 个交易日"再求和——停牌/缺失日
    # 错位，联合分布失真（低估尾部相关）。改为 date 对齐：各股收益率按日期拼成面板，
    # 仅保留所有持仓都有收益的公共交易日（严格联合分布），再逐日加权。
    panel = pd.DataFrame(ret_by_code)  # index=date, columns=code, 值=负收益(亏损方向)
    panel = panel.dropna()
    if panel.empty or len(panel) < 20:
        return {'var': 0, 'cvar': 0, 'total_value': total_value,
                'note': f'按日期对齐后公共交易日不足({len(panel)}天)，无法估计联合尾部'}
    w_vec = np.array([weights_by_code[c] for c in panel.columns])
    weighted_losses = panel.values @ w_vec  # 逐日组合加权亏损

    # Always compute both 95% and 99% confidence levels for standard reporting
    p95 = 95.0
    p99 = 99.0
    var_95 = np.percentile(weighted_losses, p95)
    var_99 = np.percentile(weighted_losses, p99)
    cvar_95 = weighted_losses[weighted_losses >= var_95].mean() if np.any(weighted_losses >= var_95) else var_95
    cvar_99 = weighted_losses[weighted_losses >= var_99].mean() if np.any(weighted_losses >= var_99) else var_99

    # Also compute user-requested custom confidence level
    custom_pct = confidence * 100
    var_custom = np.percentile(weighted_losses, custom_pct)
    cvar_custom = weighted_losses[weighted_losses >= var_custom].mean() if np.any(weighted_losses >= var_custom) else var_custom

    return {
        'var_95': round(var_95 * 100, 2),
        'var_99': round(var_99 * 100, 2),
        'cvar_95': round(cvar_95 * 100, 2),
        'cvar_99': round(cvar_99 * 100, 2),
        'var_custom': round(var_custom * 100, 2),
        'cvar_custom': round(cvar_custom * 100, 2),
        'confidence_requested': confidence,
        'total_value': round(total_value, 2),
        'var_threshold_breached': bool(abs(var_95) * total_value > VAR_THRESHOLD * total_value),
    }


# ───────── 集中度分析 ─────────
def compute_concentration(positions: list[dict]) -> dict:
    """计算持仓集中度

    D6收敛: 异名/近名异口径保留 —— 持仓列表→行业/个股/前5权重(百分数), 与
    risk_management_pro.ComplianceChecker.check_diversification(weights+industry_map→HHI/
    有效持仓/分散评分) / portfolio_optimizer.portfolio_summary(平方口径) 不同。
    """
    if not positions:
        return {'max_single': 0, 'top5': 0, 'industry_map': {}, 'warnings': []}

    total_value = sum(
        p.get('qty', 0) * p.get('current_price', 0) for p in positions
    )
    if total_value <= 0:
        return {'max_single': 0, 'top5': 0, 'warnings': []}

    # 个股集中度
    weights = []
    for p in positions:
        val = p.get('qty', 0) * p.get('current_price', 0)
        w = val / total_value
        weights.append((p.get('stock_name', p.get('stock', '')), w, val))

    weights.sort(key=lambda x: x[1], reverse=True)
    max_single = weights[0][1] if weights else 0
    top5_weight = sum(w[1] for w in weights[:5])

    # 行业集中度
    industries = {}
    for p in positions:
        sector = get_stock_sector(str(p.get('stock', '')))
        val = p.get('qty', 0) * p.get('current_price', 0)
        industries[sector] = industries.get(sector, 0) + val
    industry_map = {
        k: round(v / total_value * 100, 1)
        for k, v in sorted(industries.items(), key=lambda x: x[1], reverse=True)
    }

    # 预警
    warnings = []
    if max_single > CONCENTRATION_THRESHOLD:
        warnings.append(f"⚠️ 单票集中度过高: {weights[0][0]} {max_single*100:.1f}%")
    for ind, wt in industry_map.items():
        if wt > INDUSTRY_CONCENTRATION * 100:
            warnings.append(f"⚠️ 行业集中度过高: {ind} {wt:.1f}%")

    return {
        'max_single': round(max_single * 100, 1),
        'max_single_name': weights[0][0] if weights else '',
        'top5_weight': round(top5_weight * 100, 1),
        'top5_details': [(n, round(w*100, 1), round(v, 0)) for n, w, v in weights[:5]],
        'industry_map': industry_map,
        'warnings': warnings,
    }


# ───────── Kelly公式 ─────────
def compute_kelly_ratio(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    """计算Kelly最优仓位比例

    D6收敛: 异名/近名异口径保留 —— 单资产Kelly(胜率/盈亏比, 限50%), 与
    portfolio_v2.KellyCriterion(多资产 f*=Σ^{-1}μ, fractional/杠杆约束) 不同。

    Args:
        win_rate: 胜率 (0-1)
        avg_win: 平均盈利
        avg_loss: 平均亏损 (正数)

    Returns:
        kelly_pct: Kelly比例
        half_kelly: 半凯利（推荐）
        quarter_kelly: 四分之一凯利（保守）
    """
    if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
        return {'kelly_pct': 0, 'half_kelly': 0, 'quarter_kelly': 0}

    b = avg_win / avg_loss  # 盈亏比
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b

    kelly = max(0, min(kelly, 0.5))  # 限制在50%以内

    return {
        'kelly_pct': round(kelly * 100, 1),
        'half_kelly': round(kelly * 50, 1),
        'quarter_kelly': round(kelly * 25, 1),
        'win_rate': round(win_rate * 100, 1),
        'profit_loss_ratio': round(b, 2),
    }


def _trade_pnl_pct(trade: dict) -> float | None:
    """从单笔交易记录提取收益率（百分数，与 portfolio.compute_kelly 的 pnl_pct 口径一致）。

    优先 'pnl_pct' 字段；无则若有本金/市值类字段用 pnl/本金 换算；都没有则以
    'pnl' 原值（绝对金额）退化使用（比值量纲仍正确，但受单笔仓位大小影响）。
    """
    raw = trade.get('pnl_pct')
    if raw is not None:
        try:
            v = float(raw)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            pass
    pnl = trade.get('pnl')
    if pnl is None:
        return None
    try:
        pnl = float(pnl)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(pnl):
        return None
    basis = None
    for key in ('notional', 'principal', 'invested', 'cost_basis'):
        b = trade.get(key)
        if b:
            try:
                basis = float(b)
                break
            except (TypeError, ValueError):
                continue
    if basis:
        return pnl / basis * 100.0
    return pnl


def compute_kelly_from_trades(trades: list[dict]) -> dict:
    """从交易记录计算Kelly。

    D6收敛: 异名/近名异口径保留 —— 单资产交易记录聚合, 与 portfolio_v2.KellyCriterion 多资产口径不同。

    P2-Q16-fix (M122): 统一 pnl 口径为百分比。旧实现直接拿 t['pnl'] 当收益率：
    若 pnl 为绝对金额（元），盈亏比受单笔仓位大小影响（与 portfolio.compute_kelly
    的 pnl_pct 语义不同）。现在优先 pnl_pct，其次 pnl/本金 换算，均无才退化用
    绝对金额并在返回中显式标注近似。
    """
    pct_values: list[float] = []
    used_abs = False
    for t in trades:
        p = _trade_pnl_pct(t)
        if p is None:
            continue
        if t.get('pnl_pct') is None:
            used_abs = True
        pct_values.append(p)

    wins = [p for p in pct_values if p > 0]
    losses = [abs(p) for p in pct_values if p < 0]

    if not wins or not losses:
        return {'kelly_pct': 0, 'note': '交易数据不足'}

    win_rate = len(wins) / (len(wins) + len(losses))
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)

    result = compute_kelly_ratio(win_rate, avg_win, avg_loss)
    if used_abs:
        result['note'] = '交易记录无 pnl_pct 且无本金字段，按绝对金额近似（盈亏比受仓位大小影响）'
    return result


# ───────── 相关性矩阵 ─────────
def compute_correlation(positions: list[dict]) -> dict:
    """计算持仓相关性矩阵

    D6收敛登记: 独立能力保留（持仓相关性矩阵, 无等价实现）。
    """
    if len(positions) < 2:
        return {'matrix': {}, 'avg_corr': 0, 'note': '持仓少于2只，无法计算相关性'}

    prices = {}
    for pos in positions[:10]:  # 最多10只
        code = str(pos.get('stock', ''))
        try:
            import akshare as ak
            hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                       start_date=(datetime.now(CST)-timedelta(days=120)).strftime('%Y%m%d'),
                                       end_date=datetime.now(CST).strftime('%Y%m%d'),
                                       adjust="qfq")
            if hist is not None and len(hist) > 20:
                prices[code] = hist['收盘'].pct_change().dropna().values
        except Exception as e:
            logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)
            continue

    codes = list(prices.keys())
    if len(codes) < 2:
        return {'matrix': {}, 'avg_corr': 0, 'note': '数据不足'}

    # 对齐长度
    min_len = min(len(prices[c]) for c in codes)
    aligned = {c: prices[c][-min_len:] for c in codes}

    corr_matrix = np.corrcoef(np.array([aligned[c] for c in codes]))

    # 平均相关系数（不含对角线）
    # P2-Q16-fix (L134): 删除死代码 mask（下一行用 np.array(codes) 重算，mask 从未被使用）
    codes_arr = np.array(codes)
    n = len(codes_arr)
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            off_diag.append(corr_matrix[i, j])
    avg_corr = np.mean(off_diag) if off_diag else 0

    # 可读格式
    matrix_readable = {}
    for i, ci in enumerate(codes):
        row = {}
        for j, cj in enumerate(codes):
            name_i = next((p.get('stock_name', ci[:6]) for p in positions if str(p.get('stock', '')) == ci), ci[:6])
            name_j = next((p.get('stock_name', cj[:6]) for p in positions if str(p.get('stock', '')) == cj), cj[:6])
            if i < j:
                row[f"{name_j}"] = round(corr_matrix[i, j], 3)
        if row:
            matrix_readable[name_i] = row

    return {
        'matrix': matrix_readable,
        'avg_corr': round(avg_corr, 3),
        'max_corr': round(np.max(off_diag), 3) if off_diag else 0,
        'min_corr': round(np.min(off_diag), 3) if off_diag else 0,
    }


# ───────── 最大回撤分析 ─────────
def compute_drawdown(positions: list[dict] = None) -> dict:
    """计算当前回撤状态

    D6收敛: 异名/近名异口径保留 —— 基于仿真净值序列(百分数, 含等级/降仓建议), 与
    risk_management_pro._max_drawdown_from_returns(损失正数小数)/portfolio_v2._max_drawdown(负向)/
    portfolio_optimizer 内联(负向) 方向与量纲不同。
    """
    # 从仿真引擎获取历史净值
    try:
        sim_file = ROOT.parent / "config" / "simulation_state.json"
        if sim_file.exists():
            data = json.loads(sim_file.read_text())
            history = data.get('pnl_history', [])
            if history:
                values = [h['total_value'] for h in history]
                peak = values[0]
                max_dd = 0
                current_dd = 0
                for v in values:
                    if v > peak:
                        peak = v
                    dd = (peak - v) / peak
                    if dd > max_dd:
                        max_dd = dd
                current_dd = (peak - values[-1]) / peak

                return {
                    'current_drawdown': round(current_dd * 100, 2),
                    'max_drawdown': round(max_dd * 100, 2),
                    'peak_value': round(peak, 2),
                    'current_value': round(values[-1], 2),
                    'level': '危险' if current_dd > DRAWDOWN_DANGER else (
                        '警告' if current_dd > DRAWDOWN_WARNING else (
                            '注意' if current_dd > DRAWDOWN_CAUTION else '正常')),
                    'suggested_action': '立即降仓50%+' if current_dd > DRAWDOWN_DANGER else (
                        '考虑降仓30%' if current_dd > DRAWDOWN_WARNING else (
                            '密切关注' if current_dd > DRAWDOWN_CAUTION else '持有')),
                }
    except Exception as e:
        logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)

    return {'current_drawdown': 0, 'max_drawdown': 0, 'note': '暂无净值数据'}


def compute_ml_risk_score() -> dict:
    """ML驱动的市场风险评分（基于市场温度、广度、风险信号）

    D6收敛登记: 独立能力保留（市场温度/广度/风险信号聚合, 无等价实现）。
    """
    result = {
        'ml_risk_score': 50,
        'ml_risk_level': '中性',
        'ml_confidence': 0.0,
        'signals': [],
        'market_regime': '未知',
    }
    try:
        from quant_system.market_temperature import fetch_market_data
        md = fetch_market_data()
        if not md:
            return result

        temp = float(md.get('temperature', 50))
        stage = str(md.get('stage', ''))
        risk = float(md.get('risk', 0))
        breadth_up = float(md.get('up_count', 0))
        breadth_down = float(md.get('down_count', 0))
        total = breadth_up + breadth_down
        breadth_ratio = breadth_up / total if total > 0 else 0.5
        pct_ma144 = float(md.get('pct_above_ma144', 0))
        pct_ma300 = float(md.get('pct_above_ma300', 0))

        signals = []
        score = 50  # 默认中性

        # 温度因子
        if temp > 80:
            score += 15
            signals.append('过热区(+15)')
        elif temp > 65:
            score += 8
            signals.append('偏热(+8)')
        elif temp < 30:
            score -= 10
            signals.append('冰点(-10)')
        elif temp < 20:
            score -= 20
            signals.append('深度冰点(-20)')

        # 广度因子
        if breadth_ratio < 0.3:
            score += 12
            signals.append(f'广度极弱(+12)')
        elif breadth_ratio < 0.4:
            score += 6
            signals.append(f'广度偏弱(+6)')
        elif breadth_ratio > 0.7:
            score -= 10
            signals.append(f'广度极强(-10)')

        # 均线位置
        if pct_ma144 < 10:
            score += 10
            signals.append(f'144线下<10%(+10)')
        if pct_ma300 < 10:
            score += 8
            signals.append(f'300线下<10%(+8)')

        # 风险信号
        if risk >= 8:
            score += 20
            signals.append(f'量化风险高分(+20)')
        elif risk >= 6:
            score += 12
            signals.append(f'量化风险偏高(+12)')

        # 趋势
        if '下跌' in stage or 'panic' in stage.lower():
            score += 15
            signals.append(f'阶段:{stage}(+15)')

        # 归一化到0-100
        score = max(0, min(100, score))

        if score >= 75:
            level = '高风险'
        elif score >= 55:
            level = '偏高风险'
        elif score >= 40:
            level = '中性'
        elif score >= 25:
            level = '偏安全'
        else:
            level = '安全'

        result.update({
            'ml_risk_score': score,
            'ml_risk_level': level,
            'ml_confidence': round(abs(score - 50) / 50, 2),
            'signals': signals,
            'market_temperature': temp,
            'breadth_ratio': round(breadth_ratio, 2),
            'pct_above_ma144': pct_ma144,
            'pct_above_ma300': pct_ma300,
            'market_regime': stage,
        })
    except Exception as e:
        result['error'] = str(e)
    return result


# ───────── 综合风控报告 ─────────
def full_risk_report() -> dict:
    """生成综合风控报告（含ML风险评分）

    D6收敛登记: 独立能力保留 —— 综合报告层, 与 risk_management_pro.RiskReportGenerator 异名异实现。
    """
    positions = load_positions_from_db()

    report = {
        'timestamp': datetime.now(CST).isoformat(),
        'position_count': len(positions),
        'positions': [
            {
                # P2-Q16-fix (M123): 'stock' 键存在但为 None/非字符串时旧代码
                # p.get('stock','')[:6] 抛 TypeError。统一 str() 防御。
                'name': str(p.get('stock_name') or ''),
                'code': str(p.get('stock') or '')[:6],
                'qty': p.get('qty', 0),
                'value': round(p.get('qty', 0) * p.get('current_price', 0), 2),
            } for p in positions[:10]
        ],
        'var': compute_var(positions),
        'concentration': compute_concentration(positions),
        'drawdown': compute_drawdown(positions),
        'correlation': compute_correlation(positions),
        'ml_risk': compute_ml_risk_score(),  # ML驱动市场风险评分
    }

    # 如果有交易记录，计算Kelly
    trades = []
    try:
        sim_file = ROOT.parent / "config" / "simulation_state.json"
        if sim_file.exists():
            data = json.loads(sim_file.read_text())
            trades = data.get('trades', [])
    except Exception as e:
        logging.getLogger(__name__).error(f"[portfolio_risk] 操作失败: {e}", exc_info=True)
    if trades:
        report['kelly'] = compute_kelly_from_trades(trades)

    return report


def format_risk_report(report: dict) -> str:
    lines = [
        f"# 🛡️ 组合风控报告 ({datetime.now(CST).strftime('%Y-%m-%d %H:%M')})",
        f"{'=' * 50}",
    ]

    # 仓位概览
    pos_count = report.get('position_count', 0)
    lines.append(f"\n📦 持仓: {pos_count}只")
    if report.get('positions'):
        for p in report['positions']:
            lines.append(f"  {p['name']}({p['code']}): {p.get('value', 0):,.0f} ({p['qty']}股)")

    # VaR
    var = report.get('var', {})
    lines.append(f"\n📉 VaR分析")
    lines.append(f"  日VaR(95%): {var.get('var_95', 0):.2f}%")
    lines.append(f"  日VaR(99%): {var.get('var_99', 0):.2f}%")
    lines.append(f"  CVaR(95%): {var.get('cvar_95', 0):.2f}%")
    if var.get('var_threshold_breached'):
        lines.append(f"  ⚠️  VaR超阈值! 建议减仓")

    # 集中度
    conc = report.get('concentration', {})
    lines.append(f"\n📊 集中度")
    lines.append(f"  最重仓: {conc.get('max_single_name', '')} {conc.get('max_single', 0):.1f}%")
    lines.append(f"  前5权重: {conc.get('top5_weight', 0):.1f}%")
    for ind, pct in conc.get('industry_map', {}).items():
        lines.append(f"  行业 {ind}: {pct:.1f}%")
    for w in conc.get('warnings', []):
        lines.append(f"  {w}")

    # 回撤
    dd = report.get('drawdown', {})
    lines.append(f"\n📉 回撤")
    lines.append(f"  当前回撤: {dd.get('current_drawdown', 0):.2f}% ({dd.get('level', '')})")
    lines.append(f"  最大回撤: {dd.get('max_drawdown', 0):.2f}%")
    if dd.get('suggested_action'):
        lines.append(f"  建议: {dd['suggested_action']}")

    # 相关性
    corr = report.get('correlation', {})
    if corr.get('avg_corr'):
        lines.append(f"\n🔗 相关性")
        lines.append(f"  平均: {corr['avg_corr']} (最大: {corr.get('max_corr', 0)} / 最小: {corr.get('min_corr', 0)})")

    # Kelly
    kelly = report.get('kelly', {})
    if kelly:
        lines.append(f"\n🎯 Kelly仓位")
        lines.append(f"  胜率: {kelly.get('win_rate', 0):.1f}% / 盈亏比: {kelly.get('profit_loss_ratio', 0):.2f}")
        lines.append(f"  全Kelly: {kelly.get('kelly_pct', 0):.1f}%")
        lines.append(f"  半Kelly(推荐): {kelly.get('half_kelly', 0):.1f}%")
        lines.append(f"  ¼Kelly(保守): {kelly.get('quarter_kelly', 0):.1f}%")

    return "\n".join(lines)


# ───────── CLI ─────────
def main():
    args = set(sys.argv[1:])

    if "--var" in args:
        positions = load_positions_from_db()
        var = compute_var(positions)
        print(json.dumps(var, ensure_ascii=False, indent=2))
    elif "--concentration" in args:
        positions = load_positions_from_db()
        conc = compute_concentration(positions)
        print(json.dumps(conc, ensure_ascii=False, indent=2, default=str))
    elif "--correlation" in args:
        positions = load_positions_from_db()
        corr = compute_correlation(positions)
        print(json.dumps(corr, ensure_ascii=False, indent=2))
    else:
        report = full_risk_report()
        print(format_risk_report(report))
        # 保存到文件
        out = ROOT.parent / "config" / "risk_report.json"
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n📁 已保存到 {out}")


if __name__ == "__main__":
    main()
