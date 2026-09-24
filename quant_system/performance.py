"""
投资组合绩效分析系统 (Performance Analysis Engine)

提供完整的投资组合绩效分析功能，包括:
  - 日收益计算
  - 全套绩效指标（Sharpe, Sortino, Calmar, VaR, CVaR, 最大回撤等）
  - Brinson 业绩归因
  - 因子归因
  - 滚动指标计算
  - 交易分析
  - 绩效报告生成（text/markdown）
  - 净值曲线图生成
"""

from __future__ import annotations
import logging

import json
import math
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 路径 ──
ROOT = Path(__file__).resolve().parent
CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CST = timezone(timedelta(hours=8))

# ── 懒加载导入 ──
_trade_db = None
_watchlist = None

# P2-Q24-fix (M276): 无风险利率配置缺键告警只打印一次，避免高频调用刷屏
_warned_risk_free_config = False


def _get_trade_db():
    """Lazy-import trade_db."""
    global _trade_db
    if _trade_db is None:
        sys.path.insert(0, str(ROOT))
        from quant_system import trade_db
        _trade_db = trade_db
    return _trade_db


def _get_watchlist():
    """Lazy-import watchlist."""
    global _watchlist
    if _watchlist is None:
        sys.path.insert(0, str(ROOT))
        from quant_system import watchlist
        _watchlist = watchlist
    return _watchlist


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _to_series(data: pd.Series | list | np.ndarray, name: str = "returns") -> pd.Series:
    """Ensure input is a pandas Series."""
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, name=name)


def _annual_factor(freq: str = "daily") -> float:
    """年化因子: 日=252, 周=52, 月=12."""
    return {"daily": 252, "weekly": 52, "monthly": 12}.get(freq, 252)


def _to_ak_index_symbol(symbol: str) -> str:
    """将基准代码（如 '000300'）转换为 akshare 指数代码（如 'sh000300'）。"""
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz")):
        return s
    # 沪深指数：3 开头多为深市(sz)，其余按沪市(sh)处理
    if s.startswith("3"):
        return f"sz{s}"
    return f"sh{s}"


def _fetch_benchmark_returns(start_date: str, end_date: str, benchmark_symbol: str = "000300") -> pd.Series:
    """
    获取基准指数日收益率（默认沪深300）。

    使用 akshare 获取指数日线数据，计算日收益率序列。
    P2-Q24-fix (L287): benchmark_symbol 透传，不再硬编码 sh000300。
    """
    try:
        import akshare as ak
        ak_symbol = _to_ak_index_symbol(benchmark_symbol)
        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        if df is None or df.empty:
            print(
                f"[performance] 警告: 基准({benchmark_symbol})数据为空，超额收益/信息比将不可用",
                file=sys.stderr,
            )
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["return"] = df["close"].pct_change()
        mask = (df.index >= start_date) & (df.index <= end_date)
        result = df.loc[mask, "return"].dropna()
        return result.astype(float)
    except Exception as e:
        # P1-Q24-fix (H08): akshare 失败显式告警，不再静默返回全 0
        print(
            f"[performance] 警告: 基准({benchmark_symbol})获取失败: {e}，"
            "超额收益/信息比将不可用",
            file=sys.stderr,
        )
        # P2-Q24-fix (L288): 原 watchlist.fetch_quotes fallback 只能取到最新价，
        # 无法计算日收益，属死代码，已删除（无可行数据源时宁缺勿假，直接返回空序列）
        return pd.Series(dtype=float)


# ════════════════════════════════════════════════════════════════
# 1. compute_returns
# ════════════════════════════════════════════════════════════════

def compute_returns(
    positions_history: list[dict],
    trades: list[dict],
    benchmark_symbol: str = "000300",
) -> pd.DataFrame:
    """
    计算每日组合收益。

    快照口径约定（P1-Q24-fix H07 统一契约）：
        - 主口径 total_value = 持仓市值，**不含现金**（cash 键仅供参考，
          不并入 daily_asset）。此时买卖现金变动通过 trades 现金流隐式处理，
          日收益公式: r_t = (V_t - V_{t-1} + CF_t) / V_{t-1}
          其中 CF_t = 当日卖出收入 - 买入金额（净流入市值）。
        - 兼容口径 total_asset = 总资产，**含现金**。买卖属内部调拨，
          不改变总资产；仅外部净流入 CF_ext（入金/出金）需调整:
          r_t = (A_t - A_{t-1} - CF_ext) / A_{t-1}（CF_ext 为净流入）。
        - 两种口径不得混用；若同时出现 total_value 与 total_asset，
          按市值口径（total_value）处理并告警。

    Args:
        positions_history: 持仓历史每日快照列表。
            每个元素: {date, total_value, cash, positions: [...]}（市值口径，推荐）
            或 {date, total_asset, positions: [...]}（含现金口径）
        trades: 交易流水列表。
            每个元素: {trade_date, symbol, trade_type(buy/sell/deposit/withdraw),
                       shares, price, total_amount, ...}
        benchmark_symbol: 基准代码, 默认沪深300 "000300"

    Returns:
        DataFrame, index=日期, columns=[
            'portfolio_return', 'benchmark_return', 'excess_return'
        ]
        基准缺失日 benchmark_return/excess_return 为 NaN（不再 fillna(0)，
        避免 akshare 失败时超额收益被静默虚高，见 H08）。
    """
    try:
        # ── 快照口径判定 ──
        mv_keys = [s for s in positions_history if "total_value" in s]
        asset_keys = [s for s in positions_history if "total_value" not in s and "total_asset" in s]
        include_cash = bool(asset_keys) and not bool(mv_keys)
        if mv_keys and asset_keys:
            # 混用口径无法可靠计算收益：明确告警并按市值口径处理
            print(
                "[performance] 警告: 持仓快照混用 total_value(市值) 与 total_asset(含现金) "
                "口径，已按市值口径计算，请统一口径。",
                file=sys.stderr,
            )

        # ── 从持仓历史构建日度资产序列 ──
        daily_asset = {}
        for snap in positions_history:
            date = snap.get("date", snap.get("trade_date", ""))
            if not date:
                continue
            if include_cash:
                total = snap.get("total_asset", 0)
            else:
                total = snap.get("total_value", snap.get("total_asset", 0))
            if total == 0:
                # 尝试从positions加总
                positions = snap.get("positions", [])
                total = sum(p.get("value", 0) or p.get("shares", 0) * p.get("price", 0) for p in positions)
                if include_cash:
                    # 含现金口径 fallback：positions 市值 + 现金
                    total += snap.get("cash", 0)
                # 市值口径 fallback：仅 positions 市值（不含现金），保持契约一致
            daily_asset[date] = total

        if not daily_asset:
            # 如果持仓历史为空, 从数据库拉取
            try:
                db = _get_trade_db()
                db_positions = db.get_positions()
                db_trades = db.get_trades(days=365)
                if db_positions:
                    current_value = sum(
                        p.get("total_value", 0) or p.get("shares", 0) * (p.get("current_price") or p.get("cost_price", 0))
                        for p in db_positions
                    )
                    today = datetime.now(CST).strftime("%Y-%m-%d")
                    daily_asset[today] = current_value
                if not trades and db_trades:
                    trades = db_trades
            except Exception as e:
                logging.getLogger(__name__).error(f"[performance] 操作失败: {e}", exc_info=True)

        if not daily_asset:
            return pd.DataFrame()

        # ── 从交易流水计算每日现金流 ──
        # 市值口径: daily_cf = 卖出收入 - 买入金额（净流入市值）
        # 含现金口径: daily_ext_cf = 入金 - 出金（外部净流入），买卖为内部调拨不计数
        daily_cf: dict[str, float] = defaultdict(float)
        daily_ext_cf: dict[str, float] = defaultdict(float)
        for t in trades:
            tdate = t.get("trade_date", t.get("date", ""))
            if not tdate:
                continue
            ttype = str(t.get("trade_type", t.get("type", ""))).lower()
            amount = abs(t.get("total_amount", t.get("amount", 0)))
            if ttype in ("buy", "买入"):
                daily_cf[tdate] -= amount  # 买入: 市值口径现金流出
            elif ttype in ("sell", "卖出"):
                daily_cf[tdate] += amount  # 卖出: 市值口径现金流入
            elif ttype in ("deposit", "inflow", "入金"):
                daily_ext_cf[tdate] += amount  # 外部净流入（含现金口径）
            elif ttype in ("withdraw", "outflow", "出金"):
                daily_ext_cf[tdate] -= amount  # 外部净流出（含现金口径）

        # ── 构建收益序列 ──
        dates = sorted(daily_asset.keys())
        if len(dates) < 2:
            # 单日快照无法计算收益率: 明确报错而非静默返回全0
            # （修复 Q24#1: 原实现导致一键周报/月报恒为空且无提示）
            raise ValueError(
                f"持仓快照不足2个（当前 {len(dates)} 个），无法计算收益率。"
                "请提供多日持仓快照序列（如 full_report 从交易流水重建的日度快照）。"
            )

        dates_series = pd.to_datetime(dates)
        asset_values = pd.Series([daily_asset[d] for d in dates], index=dates_series)

        portfolio_returns = []
        for i in range(1, len(dates)):
            prev_asset = asset_values.iloc[i - 1]
            curr_asset = asset_values.iloc[i]
            date_curr = dates[i]
            if prev_asset != 0:
                if include_cash:
                    # 含现金口径: 内部买卖不改变总资产，仅外部净流入需扣除
                    ext_cf = daily_ext_cf.get(date_curr, 0)
                    r = (curr_asset - prev_asset - ext_cf) / prev_asset
                else:
                    # 市值口径: 现金变动隐含在 V 中，r=(V_t−V_{t-1}+CF_t)/V_{t-1}
                    cf = daily_cf.get(date_curr, 0)
                    r = (curr_asset - prev_asset + cf) / prev_asset
            else:
                # P2-Q24-fix (L286): 前一日资产为0（空仓起点/缺快照）时不再静默计 0，
                # 显式告警并以 NaN 标记，避免虚假收益进入报告
                print(
                    f"[performance] 警告: {date_curr} 前一交易日资产为 0，"
                    "无法计算当日收益（返回 NaN 标记，请检查快照口径/空仓起点）",
                    file=sys.stderr,
                )
                r = float("nan")
            portfolio_returns.append(r)

        ret_dates = dates_series[1:]
        port_ret_series = pd.Series(portfolio_returns, index=ret_dates, name="portfolio_return")

        # ── 基准收益 ──
        start_date = ret_dates.min().strftime("%Y-%m-%d")
        end_date = ret_dates.max().strftime("%Y-%m-%d")
        # P2-Q24-fix (L287): 透传 benchmark_symbol（原实现忽略该参数，硬编码沪深300）
        benchmark_ret = _fetch_benchmark_returns(start_date, end_date, benchmark_symbol=benchmark_symbol)

        result = pd.DataFrame(index=ret_dates)
        result["portfolio_return"] = port_ret_series
        # P1-Q24-fix (H08): 基准缺失日保持 NaN，不再 fillna(0.0)，
        # 避免 akshare 失败/非交易日时超额收益被静默虚高。
        result["benchmark_return"] = benchmark_ret.reindex(ret_dates, method=None)
        result["excess_return"] = result["portfolio_return"] - result["benchmark_return"]

        return result

    except ValueError:
        raise  # 业务性错误（快照不足等）向上传播，由调用方明确处理
    except Exception as e:
        print(f"[performance] compute_returns error: {e}", file=sys.stderr)
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════
# 2. compute_metrics
# ════════════════════════════════════════════════════════════════

def compute_metrics(
    returns: pd.Series | np.ndarray | list,
    risk_free: float = 0.025,
    freq: str = "daily",
    benchmark_returns: pd.Series | np.ndarray | list | None = None,
) -> dict:
    """
    计算全套绩效指标。

    P1-Q24-fix (H03): 新增可选 benchmark_returns，信息比基于相对基准的超额
    收益序列计算；不传入时 information_ratio=NaN（不再用 r-rf 冒充基准超额）。

    Args:
        returns: 日收益序列 (可以是 Series / ndarray / list)
        risk_free: 年化无风险利率, 默认 2.5%
        freq: 频率, 默认 "daily"
        benchmark_returns: 基准日收益序列（与 returns 对齐），用于计算信息比。
            可缺省；缺省时 information_ratio 返回 NaN。

    Returns:
        dict: {
            total_return, annual_return, annual_vol,
            sharpe_ratio, sortino_ratio, calmar_ratio,
            information_ratio, max_drawdown, max_drawdown_duration,
            rolling_sharpe_60d, win_rate,
            skewness, kurtosis,
            var_95, var_99, cvar_95, cvar_99,
            num_observations
        }
    """
    try:
        global _warned_risk_free_config
        r = _to_series(returns)
        if len(r) < 5:
            return {"error": "样本数量不足", "num_observations": len(r)}

        # V4.1 fix: 从报告配置文件读取无风险利率
        # P2-Q24-fix (M276): 不再硬编码绝对路径——优先取环境变量 QUANT_RISK_FREE_RATE，
        # 其次读工作区 config/report_delivery.json（相对路径）；配置缺 risk_free_rate 键时
        # 显式告警（仅一次）而非静默走默认。
        try:
            _env_rf = os.environ.get("QUANT_RISK_FREE_RATE")
            if _env_rf is not None:
                risk_free = float(_env_rf)
            else:
                _cfg_path = ROOT.parent / "config" / "report_delivery.json"
                if _cfg_path.exists():
                    with open(_cfg_path) as _f:
                        _cfg = json.load(_f)
                    if "risk_free_rate" in _cfg:
                        risk_free = float(_cfg["risk_free_rate"])
                    elif not _warned_risk_free_config:
                        print(
                            f"[performance] 警告: {_cfg_path} 缺少 risk_free_rate 键，"
                            f"使用默认无风险利率 {risk_free}（可设置环境变量 "
                            "QUANT_RISK_FREE_RATE 覆盖）",
                            file=sys.stderr,
                        )
                        _warned_risk_free_config = True
                # 配置文件不存在时静默走默认（未部署配置的环境属正常）
        except Exception as _e:
            if not _warned_risk_free_config:
                print(
                    f"[performance] 警告: 读取无风险利率配置失败({_e})，使用默认 {risk_free}",
                    file=sys.stderr,
                )
                _warned_risk_free_config = True

        af = _annual_factor(freq)
        rf_daily = risk_free / af

        # ── 基础统计 ──
        n = len(r)

        # 累计收益
        total_return = float(np.prod(1 + r) - 1)

        # 年化收益 (几何)
        annual_return = float((1 + total_return) ** (af / n) - 1) if n > 0 else 0.0

        # 年化波动率
        daily_vol = float(r.std(ddof=1))
        annual_vol = daily_vol * math.sqrt(af)

        # ── Sharpe 比 ──
        excess_daily = r - rf_daily
        sharpe_ratio = float(excess_daily.mean() / excess_daily.std(ddof=1) * math.sqrt(af)) \
            if excess_daily.std(ddof=1) > 0 else 0.0

        # ── Sortino 比 ──
        # P1-Q24-fix (H05): 用标准下行偏差 sqrt(mean(min(r-rf,0)^2))（含0值、
        # 不做 ddof 截断），替换对"下行子样本"取 std 的非标准做法；无下行偏差时
        # 返回 NaN 而非 1e-10，避免仅1个亏损日时 Sortino 爆炸。
        downside_devs = np.minimum(r.values - rf_daily, 0.0)
        downside_var = float((downside_devs ** 2).mean())
        downside_vol = math.sqrt(downside_var)
        sortino_ratio = float(
            (r.mean() - rf_daily) * af / (downside_vol * math.sqrt(af))
        ) if downside_var > 0 else float("nan")

        # ── 最大回撤 ──
        cumulative = (1 + r).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())
        # 回撤持续时间 (最长)
        is_in_dd = drawdown < 0
        dd_durations = []
        current_duration = 0
        for flag in is_in_dd:
            if flag:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            dd_durations.append(current_duration)
        max_drawdown_duration = max(dd_durations) if dd_durations else 0

        # ── Calmar 比 ──
        # P1-Q24-fix (H04): 保留符号——亏损策略的负年化收益不应被 abs() 翻正。
        calmar_ratio = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # ── 信息比 IR = mean(r_p − r_b)/std(r_p − r_b)×√af ──
        # P1-Q24-fix (H03): 不再用 r-rf 近似（此前 IR≡Sharpe），改为基于基准
        # 超额收益；未传入基准或基准全缺失时返回 NaN，避免静默虚高。
        information_ratio = float("nan")
        if benchmark_returns is not None:
            bm = _to_series(benchmark_returns)
            if isinstance(r.index, pd.DatetimeIndex) and isinstance(bm.index, pd.DatetimeIndex):
                aligned = pd.concat([r.rename("port"), bm.rename("bench")], axis=1).dropna()
            else:
                n_al = min(len(r), len(bm))
                aligned = pd.DataFrame(
                    {"port": np.asarray(r.values)[:n_al], "bench": np.asarray(bm.values)[:n_al]}
                ).dropna()
            if len(aligned) >= 2:
                active = aligned["port"] - aligned["bench"]
                active_std = float(active.std(ddof=1))
                if active_std > 0:
                    information_ratio = float(active.mean() / active_std * math.sqrt(af))

        # ── 滚动60日夏普比 ──
        rolling_sharpe_60d = {}
        if len(r) >= 60:
            rolling_mean = r.rolling(60).mean()
            rolling_std = r.rolling(60).std(ddof=1)
            rolling_excess = rolling_mean - rf_daily
            # P2-Q24-fix (M272): 常数收益时 rolling_std=0 → 除零产生 inf/nan，
            # 先替换 inf 再丢弃（与 rolling_metrics 的 rolling_sharpe 处理一致）
            rolling_sharpe = (
                rolling_excess / rolling_std * math.sqrt(af)
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if rolling_sharpe.empty:
                # 全窗口夏普不可计算（如常数收益）：显式归 0，避免 inf/nan 进入报告
                rolling_sharpe_60d = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            else:
                rolling_sharpe_60d = {
                    "mean": float(rolling_sharpe.mean()),
                    "std": float(rolling_sharpe.std()),
                    "min": float(rolling_sharpe.min()),
                    "max": float(rolling_sharpe.max()),
                }

        # ── 胜率 ──
        win_rate = float((r > 0).sum() / n) if n > 0 else 0.0

        # ── 偏度/峰度 ──
        skewness = float(r.skew()) if n >= 3 else 0.0
        kurtosis = float(r.kurtosis()) if n >= 4 else 0.0

        # ── VaR / CVaR ──
        var_95 = float(np.percentile(r, 5))
        var_99 = float(np.percentile(r, 1))
        cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).sum() > 0 else var_95
        cvar_99 = float(r[r <= var_99].mean()) if (r <= var_99).sum() > 0 else var_99

        return {
            "total_return": round(total_return * 100, 4),
            "annual_return": round(annual_return * 100, 4),
            "annual_vol": round(annual_vol * 100, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "calmar_ratio": round(calmar_ratio, 4),
            "information_ratio": round(information_ratio, 4),
            "max_drawdown": round(max_drawdown * 100, 4),
            "max_drawdown_duration": max_drawdown_duration,
            "rolling_sharpe_60d": rolling_sharpe_60d,
            "win_rate": round(win_rate * 100, 2),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "var_95": round(var_95 * 100, 4),
            "var_99": round(var_99 * 100, 4),
            "cvar_95": round(cvar_95 * 100, 4),
            "cvar_99": round(cvar_99 * 100, 4),
            "num_observations": n,
        }

    except Exception as e:
        print(f"[performance] compute_metrics error: {e}", file=sys.stderr)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# 3. brinson_attribution
# ════════════════════════════════════════════════════════════════

def brinson_attribution(
    portfolio_weights: dict[str, float],
    sector_map: dict[str, str],
    benchmark_weights: dict[str, float],
    returns: dict[str, float],
) -> dict:
    """
    Brinson 业绩归因分解。

    将组合超额收益分解为:
      - 配置效应 (Allocation Effect): 行业配置决策带来的超额收益
      - 选股效应 (Selection Effect): 行业内选股带来的超额收益
      - 交互效应 (Interaction Effect): 配置与选股的交互影响

    Args:
        portfolio_weights: 组合权重 {symbol: weight (0~1)}
        sector_map: 行业映射 {symbol: sector_name}
        benchmark_weights: 基准权重 {symbol: weight (0~1)}
        returns: 区间收益 {symbol: return (小数)}

    Returns:
        dict: {
            allocation_effect: {sector: value, total: ...},
            selection_effect: {sector: value, total: ...},
            interaction_effect: {sector: value, total: ...},
            total_excess_return: float,
            sector_details: [{sector, p_weight, b_weight, p_return, b_return,
                              allocation, selection, interaction}, ...]
        }
    """
    try:
        all_symbols = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        # P2-Q24-fix (M277): 权重归一化——组合/基准权重和≠1时按比例缩放，
        # 避免行业权重直接相加导致的配置/选股效应整体偏移（与 performance_deep 版一致）
        total_pf_w = sum(portfolio_weights.values())
        total_bm_w = sum(benchmark_weights.values())
        _pf_scale = total_pf_w if total_pf_w > 0 else 1.0
        _bm_scale = total_bm_w if total_bm_w > 0 else 1.0

        # P2-Q24-fix (M277): 缺收益 symbol 显式告警，不再静默按 0 处理
        missing_ret = [s for s in all_symbols if s not in returns]
        if missing_ret:
            shown = ", ".join(missing_ret[:5])
            print(
                f"[performance] 警告: brinson_attribution 缺少 "
                f"{len(missing_ret)} 个标的收益 ({shown}{'...' if len(missing_ret) > 5 else ''})，"
                "收益按 0 处理（相关行业效应会失真）",
                file=sys.stderr,
            )

        # 按行业汇总
        sector_data: dict[str, dict] = defaultdict(lambda: {
            "p_weight": 0.0,  # 组合行业权重
            "b_weight": 0.0,  # 基准行业权重
            "p_return_sum": 0.0,  # 组合行业收益 (加权)
            "b_return_sum": 0.0,  # 基准行业收益 (加权)
            "p_return": 0.0,  # 组合行业加权平均收益
            "b_return": 0.0,  # 基准行业加权平均收益
            "symbols": [],
        })

        for sym in all_symbols:
            sector = sector_map.get(sym, "未分类")
            pw = portfolio_weights.get(sym, 0.0) / _pf_scale
            bw = benchmark_weights.get(sym, 0.0) / _bm_scale
            ret = returns.get(sym, 0.0)
            sd = sector_data[sector]
            sd["p_weight"] += pw
            sd["b_weight"] += bw
            sd["p_return_sum"] += pw * ret
            sd["b_return_sum"] += bw * ret
            sd["symbols"].append(sym)

        # 计算行业平均收益
        for sd in sector_data.values():
            sd["p_return"] = sd["p_return_sum"] / sd["p_weight"] if sd["p_weight"] > 0 else 0.0
            sd["b_return"] = sd["b_return_sum"] / sd["b_weight"] if sd["b_weight"] > 0 else 0.0

        # 基准总收益
        total_b_return = sum(
            (benchmark_weights.get(sym, 0.0) / _bm_scale) * returns.get(sym, 0.0)
            for sym in all_symbols
        )

        # Brinson 分解
        allocation_effect: dict[str, float] = {}
        selection_effect: dict[str, float] = {}
        interaction_effect: dict[str, float] = {}
        sector_details = []
        total_alloc = 0.0
        total_select = 0.0
        total_interact = 0.0

        for sector, sd in sorted(sector_data.items()):
            # 配置效应 = (Wp - Wb) * (Rb_sector - Rb_total)
            alloc = (sd["p_weight"] - sd["b_weight"]) * (sd["b_return"] - total_b_return)
            # 选股效应 = Wb * (Rp_sector - Rb_sector)
            select = sd["b_weight"] * (sd["p_return"] - sd["b_return"])
            # 交互效应 = (Wp - Wb) * (Rp_sector - Rb_sector)
            interact = (sd["p_weight"] - sd["b_weight"]) * (sd["p_return"] - sd["b_return"])

            # P2-Q24-fix (M277): 汇总用未舍入值，避免逐行业四舍五入的微小漂移
            total_alloc += alloc
            total_select += select
            total_interact += interact

            allocation_effect[sector] = round(alloc * 100, 4)
            selection_effect[sector] = round(select * 100, 4)
            interaction_effect[sector] = round(interact * 100, 4)

            sector_details.append({
                "sector": sector,
                "p_weight": round(sd["p_weight"] * 100, 2),
                "b_weight": round(sd["b_weight"] * 100, 2),
                "p_return": round(sd["p_return"] * 100, 4),
                "b_return": round(sd["b_return"] * 100, 4),
                "allocation": round(alloc * 100, 4),
                "selection": round(select * 100, 4),
                "interaction": round(interact * 100, 4),
            })

        return {
            "allocation_effect": {**allocation_effect, "total": round(total_alloc, 4)},
            "selection_effect": {**selection_effect, "total": round(total_select, 4)},
            "interaction_effect": {**interaction_effect, "total": round(total_interact, 4)},
            "total_excess_return": round(total_alloc + total_select + total_interact, 4),
            "benchmark_return": round(total_b_return * 100, 4),
            "sector_details": sector_details,
        }

    except Exception as e:
        print(f"[performance] brinson_attribution error: {e}", file=sys.stderr)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# 4. factor_attribution
# ════════════════════════════════════════════════════════════════

def factor_attribution(
    portfolio_weights: dict[str, float],
    factor_exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
    portfolio_returns: pd.Series | np.ndarray | list | None = None,
) -> dict:
    """
    因子归因分析。

    将组合收益分解为各因子贡献 + 残差 (特异性收益)。

    Args:
        portfolio_weights: 组合权重 {symbol: weight}
        factor_exposures: 因子暴露 DataFrame
            index=symbol, columns=factor names (如 'market', 'size', 'value', 'momentum' ...)
        factor_returns: 因子收益 DataFrame
            index=date (或单行), columns=factor names, 与 factor_exposures 列名对齐
        portfolio_returns: 组合实际收益序列 (date 索引, 小数)。
            提供后残差/R² 才基于真实组合收益计算；缺省时退化为
            因子收益中的 specific/residual 列，仍无则返回 NaN 并注明
            （不再用占位值虚构 residual=0 / R²=1）。

    Returns:
        dict: {
            factor_contributions: {factor_name: contribution_pct, ...},
            residual: float,       # 总收益 - 各因子贡献之和 (%)
            portfolio_return: float,
            r_squared: float,      # 因子解释比例 (实际收益对因子模型预测拟合优度)
            details: {factor_name: {exposure: float, return: float, contribution: float}, ...}
        }
    """
    try:
        # ── 计算组合对各因子的暴露 (市值加权平均) ──
        symbols = [s for s in portfolio_weights if s in factor_exposures.index]
        if not symbols:
            return {"error": "组合标的与因子暴露数据无交集"}

        # 筛选并对齐
        weights = np.array([portfolio_weights[s] for s in symbols])
        total_w = weights.sum()
        if total_w == 0:
            return {"error": "组合权重之和为0"}
        weights = weights / total_w  # 归一化

        exposures = factor_exposures.loc[symbols]
        portfolio_exposure = (exposures.T * weights).T.sum()  # 加权平均暴露
        portfolio_exposure = portfolio_exposure.to_dict()

        # ── 计算因子贡献 ──
        # 如果 factor_returns 是多行 (时间序列), 取均值
        if isinstance(factor_returns, pd.DataFrame) and len(factor_returns) > 1:
            fr_mean = factor_returns.mean()
        elif isinstance(factor_returns, pd.DataFrame):
            fr_mean = factor_returns.iloc[0]
        else:
            fr_mean = pd.Series(factor_returns)

        # 对齐因子
        common_factors = [f for f in portfolio_exposure if f in fr_mean.index]
        if not common_factors:
            return {"error": "因子暴露与因子收益无公共因子"}

        factor_contributions: dict[str, float] = {}
        details = {}
        for f in common_factors:
            exp = portfolio_exposure[f]
            fret = float(fr_mean[f])
            contrib = exp * fret
            factor_contributions[f] = round(contrib * 100, 4)
            details[f] = {
                "exposure": round(exp, 4),
                "return": round(fret * 100, 4),
                "contribution": round(contrib * 100, 4),
            }

        total_factor_return = sum(factor_contributions.values())  # 单位: %

        # ── 组合实际收益序列（残差与 R² 的基准） ──
        # 优先: 调用方传入的组合实际收益序列 portfolio_returns
        # 其次: factor_returns 中的 specific/residual 列（特异性收益）
        # 都无: 无法计算真实残差 → NaN 并注明（不再虚构 residual=0 / R²=1）
        actual_series: pd.Series | None = None
        if portfolio_returns is not None:
            s = _to_series(portfolio_returns)
            if len(s) > 0:
                actual_series = s.astype(float)

        # 因子模型预测收益序列（多期: Σ_f exposure_f × fr_{t,f}）
        model_series: pd.Series | None = None
        if isinstance(factor_returns, pd.DataFrame) and len(factor_returns) > 1:
            model_series = factor_returns[common_factors].dot(
                pd.Series({f: portfolio_exposure[f] for f in common_factors})
            )

        if actual_series is not None:
            if model_series is not None:
                # 按日期对齐；索引不匹配时按位置对齐到公共长度
                idx = actual_series.index.intersection(model_series.index)
                if len(idx) >= 2:
                    actual_series = actual_series.loc[idx]
                    model_series = model_series.loc[idx]
                else:
                    n = min(len(actual_series), len(model_series))
                    actual_series = pd.Series(
                        actual_series.values[:n], index=model_series.index[:n], dtype=float
                    )
            actual_mean = float(actual_series.mean())
        else:
            specific_cols = [
                c for c in factor_returns.columns
                if str(c).lower() in ("specific", "specific_return", "residual", "idiosyncratic")
            ]
            if specific_cols:
                actual_mean = float(factor_returns[specific_cols[0]].mean())
            else:
                actual_mean = float("nan")

        portfolio_return_pct = actual_mean * 100  # 输出口径: 百分数
        residual_pct = portfolio_return_pct - total_factor_return

        # ── R²: 实际组合收益对因子模型预测的拟合优度 1 − SS_res/SS_tot ──
        # 修复 Q24#2: 原实现用同一序列自比 → R² 恒为 1.0（虚构）
        r_squared = float("nan")
        if actual_series is not None and model_series is not None:
            aligned = pd.concat(
                [actual_series.rename("actual"), model_series.rename("model")], axis=1
            ).dropna()
            if len(aligned) >= 2:
                ss_tot = float(((aligned["actual"] - aligned["actual"].mean()) ** 2).sum())
                ss_res = float(((aligned["actual"] - aligned["model"]) ** 2).sum())
                if ss_tot > 0:
                    r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
                else:
                    r_squared = 0.0

        note = None
        if actual_series is None and np.isnan(actual_mean):
            note = "未提供组合实际收益序列(portfolio_returns)，残差/R² 无法计算"

        return {
            "factor_contributions": factor_contributions,
            "residual": round(residual_pct, 4),
            "portfolio_return": round(portfolio_return_pct, 4),
            "r_squared": round(r_squared, 4),
            "details": details,
            "common_factors": common_factors,
            "note": note,
        }

    except Exception as e:
        print(f"[performance] factor_attribution error: {e}", file=sys.stderr)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# 5. rolling_metrics
# ════════════════════════════════════════════════════════════════

def rolling_metrics(
    returns: pd.Series | np.ndarray | list,
    windows: list[int] | None = None,
    risk_free: float = 0.025,
) -> dict:
    """
    滚动指标计算。

    Args:
        returns: 收益序列
        windows: 滚动窗口列表 (交易日数), 默认 [21, 63, 252]
                 21=1个月, 63=3个月, 252=1年
        risk_free: 年化无风险利率, 默认 2.5%（与 compute_metrics 同口径）

    Returns:
        dict: {
            window_size: {
                sharpe: float,      # 窗口内夏普比均值（已扣除无风险利率）
                vol: float,         # 窗口内年化波动率均值
                max_dd: float,      # 窗口内最大回撤均值
                sharpe_series: [window_count],  # 逐窗口夏普比
                vol_series: [...],
                max_dd_series: [...],
            }
        }
    """
    try:
        r = _to_series(returns)
        if windows is None:
            windows = [21, 63, 252]
        if len(r) < min(windows):
            return {"error": f"数据长度 {len(r)} 小于最小窗口 {min(windows)}"}

        af = 252
        rf_daily = risk_free / af
        result: dict[int, dict] = {}

        for w in windows:
            if len(r) < w:
                result[w] = {"error": f"数据不足 (需{w}期)"}
                continue

            # 滚动 Sharpe (V4.1 fix: 与 compute_metrics 同口径, 扣除无风险利率)
            rolling_mean = r.rolling(w).mean()
            rolling_std = r.rolling(w).std(ddof=1)
            rolling_sharpe = ((rolling_mean - rf_daily) / rolling_std * math.sqrt(af)).dropna()
            rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).dropna()

            # 滚动年化波动率
            rolling_vol = (rolling_std * math.sqrt(af)).dropna()

            # 滚动最大回撤
            def _rolling_max_dd(series):
                cum = (1 + series).cumprod()
                run_max = cum.expanding().max()
                dd = (cum - run_max) / run_max
                return dd.min()

            rolling_max_dd = r.rolling(w).apply(_rolling_max_dd, raw=False).dropna()

            # 如果滚动窗口滑动, 计算每个窗口的统计量
            # 这里滚动值是逐日计算的
            n_windows = len(rolling_sharpe)

            result[w] = {
                "sharpe": round(float(rolling_sharpe.mean()), 4),
                "vol": round(float(rolling_vol.mean() * 100), 4),
                "max_dd": round(float(rolling_max_dd.mean() * 100), 4),
                "sharpe_series": [round(v, 4) for v in rolling_sharpe.tail(60).tolist()],
                "vol_series": [round(v, 4) for v in rolling_vol.tail(60).tolist()],
                "max_dd_series": [round(v, 4) for v in rolling_max_dd.tail(60).tolist()],
                "num_windows": n_windows,
            }

        return result

    except Exception as e:
        print(f"[performance] rolling_metrics error: {e}", file=sys.stderr)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# 6. trade_analysis
# ════════════════════════════════════════════════════════════════

def _match_closed_trades(trades: list[dict]) -> list[dict]:
    """按 FIFO 配对买入/卖出，返回每笔已平仓卖出的实现盈亏明细。

    P1-Q24-fix: 统一 H01/H02/H06 的数据来源——by_signal / profit_factor /
    total_pnl / avg_trade_pnl / monthly_returns 等一律基于传入的 trades
    计算（而非全库 signal_records），避免统计口径与报告期交易流水不一致，
    也避免 get_signal_performance() 无参分支缺 avg_win/avg_loss/profit_factor
    列导致的 KeyError 被静默吞掉、profit_factor 恒 0。

    说明: 买入价取 FIFO 最早未平仓批次；卖出量超出可匹配买入时（报告期外
    建仓的标的），该部分按卖出价近似成本（不虚增盈亏）。
    """
    lots: dict[str, deque] = defaultdict(deque)  # symbol -> deque[{shares, price, signal_type, date}]
    closed: list[dict] = []

    def _sort_key(t):
        # 同一日买入先于卖出（A股 T+1，正常同日不会配对，但排序保证确定性）
        ttype = t.get("trade_type", t.get("type", "")).lower()
        return (t.get("trade_date", t.get("date", "")), 0 if ttype == "buy" else 1)

    for t in sorted(trades, key=_sort_key):
        sym = t.get("symbol")
        ttype = t.get("trade_type", t.get("type", "")).lower()
        qty = int(t.get("shares", t.get("qty", 0)) or 0)
        price = float(t.get("price", 0) or 0)
        tdate = t.get("trade_date", t.get("date", ""))
        if not sym or qty <= 0 or not tdate:
            continue
        if ttype == "buy":
            lots[sym].append({
                "shares": qty,
                "price": price,
                "signal_type": t.get("signal_type", "") or "",
                "date": tdate,
            })
        elif ttype == "sell":
            remaining = qty
            cost = 0.0
            lot_signal = t.get("signal_type", "") or ""
            lot_buy_date = ""
            while remaining > 0 and lots[sym]:
                lot = lots[sym][0]
                matched = min(remaining, lot["shares"])
                cost += matched * lot["price"]
                lot_signal = lot["signal_type"] or lot_signal
                lot_buy_date = lot["date"] or lot_buy_date
                remaining -= matched
                lot["shares"] -= matched
                if lot["shares"] <= 0:
                    lots[sym].popleft()
            if remaining > 0:
                # 卖出多于窗口内可匹配买入：按卖出价近似成本，视为成本价未知的平仓
                cost += remaining * price
            # 审计 2026-08-16：净卖出额扣除交易费用（佣金/印花税/过户费，如有字段）
            proceeds = price * qty - float(t.get("commission", 0) or 0)                        - float(t.get("stamp_tax", 0) or 0)                        - float(t.get("transfer_fee", 0) or 0)
            closed.append({
                "symbol": sym,
                "sell_date": tdate,
                "buy_date": lot_buy_date,
                "shares": qty,
                "cost": cost,
                "sell_price": price,
                "pnl": proceeds - cost,
                "pnl_pct": round((proceeds - cost) / cost * 100, 2) if cost > 0 else 0.0,
                "signal_type": lot_signal,
            })
    return closed


def trade_analysis(trades: list[dict]) -> dict:
    """
    交易分析。

    统计各类交易指标。

    Args:
        trades: 交易记录列表。
            每个元素: {trade_date, symbol, name, trade_type(buy/sell),
                       shares, price, total_amount, signal_type, notes, ...}

    Returns:
        dict: {
            total_trades: int,
            buy_count: int, sell_count: int,
            by_signal: {
                signal_type: {
                    count, wins, losses, win_rate, avg_pnl_pct,
                    avg_win_pct, avg_loss_pct,
                    profit_factor (盈利/亏损比)
                }
            }
            profit_factor: float,          # 已平仓毛盈利 / 毛亏损
            avg_holding_period_days: float, # 平均持有期 (天)
            max_consecutive_losses: int,   # 最大连续亏损次数
            monthly_returns: {YYYY-MM: pnl_pct, ...}  # 按平仓月份成本加权
            total_cost: float,             # 买入总额（现金流，仅供参考）
            total_proceeds: float,         # 卖出总额（现金流，仅供参考）
            total_pnl: float,              # 平仓实现盈亏（基于 FIFO 配对）
            total_pnl_pct: float,
            avg_trade_pnl: float,          # 平均每笔已平仓交易盈亏
        }

    P1-Q24-fix: 所有盈亏类指标基于传入 trades 的 FIFO 平仓配对计算，
    不再读取全库 signal_records（修复 H01/H02/H06）。
    """
    try:
        if not trades:
            return {"total_trades": 0, "error": "无交易记录"}

        df = pd.DataFrame(trades)

        # ── 基础统计 ──
        total_trades = len(df)
        buy_count = int((df["trade_type"] == "buy").sum()) if "trade_type" in df.columns else 0
        sell_count = int((df["trade_type"] == "sell").sum()) if "trade_type" in df.columns else 0

        # P1-Q24-fix: 基于传入 trades 的 FIFO 配对，统一 H01/H02/H06 数据来源
        closed = _match_closed_trades(trades)

        # ── 按信号类型统计（仅统计已平仓交易，修复 Q24-H01/H02：
        #    不再读全库 signal_records，避免 avg_win/avg_loss/profit_factor 恒 0） ──
        by_signal: dict[str, dict] = {}
        if closed:
            sig_groups: dict[str, list[dict]] = defaultdict(list)
            for c in closed:
                sig_groups[c.get("signal_type") or "未知信号"].append(c)
            for sig, items in sorted(sig_groups.items()):
                wins = sum(1 for c in items if c["pnl"] > 0)
                losses = sum(1 for c in items if c["pnl"] < 0)
                total = len(items)
                pnl_pcts = [c["pnl_pct"] for c in items]
                win_pcts = [c["pnl_pct"] for c in items if c["pnl"] > 0]
                loss_pcts = [c["pnl_pct"] for c in items if c["pnl"] < 0]
                gross_profit = sum(c["pnl"] for c in items if c["pnl"] > 0)
                gross_loss = abs(sum(c["pnl"] for c in items if c["pnl"] < 0))
                if gross_loss > 0:
                    pf = round(gross_profit / gross_loss, 2)
                else:
                    pf = 0.0 if gross_profit <= 0 else float("inf")
                by_signal[sig] = {
                    "count": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(wins / total * 100, 2) if total else 0,
                    "avg_pnl_pct": round(float(np.mean(pnl_pcts)), 2) if pnl_pcts else 0.0,
                    "avg_win_pct": round(float(np.mean(win_pcts)), 2) if win_pcts else 0.0,
                    "avg_loss_pct": round(float(np.mean(loss_pcts)), 2) if loss_pcts else 0.0,
                    "profit_factor": pf,
                }

        # ── 总盈亏（平仓实现盈亏，修复 Q24-H06：
        #    原净现金流 total_proceeds-total_cost 在只有买入时会把 -全部买入成本误报为巨亏） ──
        total_cost = 0.0
        total_proceeds = 0.0
        if "trade_type" in df.columns and "total_amount" in df.columns:
            buys = df[df["trade_type"] == "buy"]
            sells = df[df["trade_type"] == "sell"]
            total_cost = float(buys["total_amount"].sum()) if not buys.empty else 0
            total_proceeds = float(sells["total_amount"].sum()) if not sells.empty else 0

        realized_pnl = sum(c["pnl"] for c in closed)
        closed_cost = sum(c["cost"] for c in closed)
        total_pnl = realized_pnl
        total_pnl_pct = round((realized_pnl / closed_cost * 100), 4) if closed_cost > 0 else 0

        # ── 盈利因子（按已平仓盈亏的毛盈利/毛亏损） ──
        gross_profit = sum(c["pnl"] for c in closed if c["pnl"] > 0)
        gross_loss = abs(sum(c["pnl"] for c in closed if c["pnl"] < 0))
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        else:
            profit_factor = 0.0 if gross_profit <= 0 else float("inf")

        # ── 平均持有期（基于 FIFO 配对结果） ──
        avg_holding_period = 0.0
        if closed:
            holding_periods = []
            for c in closed:
                if c.get("buy_date"):
                    try:
                        days = (pd.to_datetime(c["sell_date"]) - pd.to_datetime(c["buy_date"])).days
                        if days > 0:
                            holding_periods.append(days)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[performance] 操作失败: {e}", exc_info=True)
            avg_holding_period = round(float(np.mean(holding_periods)), 1) if holding_periods else 0.0

        # ── 最大连续亏损次数（按已平仓卖出日期排序） ──
        max_consecutive_losses = 0
        if closed:
            consec_loss = 0
            for c in sorted(closed, key=lambda x: x["sell_date"]):
                if c["pnl"] < 0:
                    consec_loss += 1
                    max_consecutive_losses = max(max_consecutive_losses, consec_loss)
                else:
                    consec_loss = 0

        # ── 月度收益分布（按已平仓卖出月份汇总成本加权收益，修复 Q24-H02：
        #    不再读取全库 signal_records） ──
        monthly_returns: dict[str, float] = {}
        if closed:
            month_pnl: dict[str, float] = defaultdict(float)
            month_cost: dict[str, float] = defaultdict(float)
            for c in closed:
                try:
                    m = pd.to_datetime(c["sell_date"]).strftime("%Y-%m")
                except Exception as e:
                    logging.getLogger(__name__).error(f"[performance] 操作失败: {e}", exc_info=True)
                    continue
                month_pnl[m] += c["pnl"]
                month_cost[m] += c["cost"]
            for m in sorted(month_pnl):
                monthly_returns[m] = round(
                    month_pnl[m] / month_cost[m] * 100, 2
                ) if month_cost[m] > 0 else 0.0

        avg_trade_pnl = round(realized_pnl / len(closed), 2) if closed else 0.0

        return {
            "total_trades": total_trades,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "by_signal": by_signal,
            "profit_factor": profit_factor,
            "avg_holding_period_days": avg_holding_period,
            "max_consecutive_losses": max_consecutive_losses,
            "monthly_returns": monthly_returns,
            "total_cost": round(total_cost, 2),
            "total_proceeds": round(total_proceeds, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": total_pnl_pct,
            "avg_trade_pnl": avg_trade_pnl,
        }

    except Exception as e:
        print(f"[performance] trade_analysis error: {e}", file=sys.stderr)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# 7. generate_performance_report
# ════════════════════════════════════════════════════════════════

def benchmark_deep_analysis(portfolio_returns: pd.Series | list, benchmark_returns: pd.Series | list,
                            risk_free_rate: float = 0.0) -> dict:
    """V11 融合: 基准深度分析（接入 performance_deep.benchmark_analysis）。

    计算追踪误差/信息比率/OLS alpha-beta/滚动夏普。
    """
    try:
        from quant_system.performance_deep.benchmark_analysis import BenchmarkAnalysis
    except ImportError:
        return {"available": False, "note": "performance_deep 不可用"}
    pr = _to_series(portfolio_returns, "portfolio") if not isinstance(portfolio_returns, pd.Series) else portfolio_returns
    br = _to_series(benchmark_returns, "benchmark") if not isinstance(benchmark_returns, pd.Series) else benchmark_returns
    if len(pr) < 2 or len(br) < 2:
        return {"available": False, "note": "样本不足"}
    try:
        ba = BenchmarkAnalysis(risk_free_rate=risk_free_rate)
        # 对齐
        idx = pr.index.intersection(br.index)
        if len(idx) < 2:
            return {"available": False, "note": "日期无交集"}
        result = ba.analyze(pr.loc[idx], br.loc[idx])
        return {"available": True, **result}
    except Exception as e:
        return {"available": False, "note": f"基准深度分析失败: {e}"}


def attribution_stability(attributions: list[dict], trend_threshold: float = 1e-4) -> dict:
    """V11 融合: 归因稳定性分析（接入 performance_deep.attribution_stability）。

    Args:
        attributions: 滚动窗口 Brinson 归因结果列表（每期含 allocation/selection 效应）
        trend_threshold: 趋势斜率阈值

    Returns:
        稳定性评分 + 趋势判断；数据不足返回说明。
    """
    try:
        from quant_system.performance_deep.attribution_stability import AttributionStability
    except ImportError:
        return {"available": False, "note": "performance_deep 不可用"}
    if not attributions:
        return {"available": False, "note": "无归因数据"}
    eng = AttributionStability(trend_threshold=trend_threshold)
    try:
        return {"available": True, **eng.analyze(attributions)}
    except Exception as e:
        return {"available": False, "note": f"归因稳定性分析失败: {e}"}


def generate_performance_report(
    result: dict,
    format: str = "text",  # noqa: A002
) -> str:
    """
    生成完整的绩效报告。

    Args:
        result: 包含以下 key 的字典:
            - returns_df: pd.DataFrame (from compute_returns)
            - metrics: dict (from compute_metrics)
            - brinson: dict (from brinson_attribution)
            - factor: dict (from factor_attribution)
            - trade_analysis: dict (from trade_analysis)
            - rolling: dict (from rolling_metrics)
            - positions: list[dict] (当前持仓)
            - report_title: str (可选)
            - period: str (可选, 如 "2026-07-01 至 2026-07-30")
        format: 输出格式, "text" 或 "markdown"

    Returns:
        格式化报告字符串。
    """
    try:
        title = result.get("report_title", "投资组合绩效报告")
        period = result.get("period", "")
        metrics = result.get("metrics", {})
        brinson = result.get("brinson", {})
        factor = result.get("factor", {})
        trade_anal = result.get("trade_analysis", {})
        rolling = result.get("rolling", {})
        positions = result.get("positions", [])
        returns_df = result.get("returns_df")

        lines: list[str] = []
        is_md = (format == "markdown")

        # ── 标题 ──
        if is_md:
            lines.append(f"# {title}")
            if period:
                lines.append(f"\n**报告期间**: {period}")
            lines.append("")
        else:
            lines.append(f"{'=' * 60}")
            lines.append(f"  {title}")
            if period:
                lines.append(f"  报告期间: {period}")
            lines.append(f"{'=' * 60}")
            lines.append("")

        # ── 收益概览 ──
        if metrics and "error" not in metrics:
            if is_md:
                lines.append("## 📊 收益概览")
                lines.append("")
                lines.append("| 指标 | 数值 |")
                lines.append("|------|------|")
                lines.append(f"| 累计收益 | {metrics.get('total_return', 0):.2f}% |")
                lines.append(f"| 年化收益 | {metrics.get('annual_return', 0):.2f}% |")
                lines.append(f"| 年化波动率 | {metrics.get('annual_vol', 0):.2f}% |")
                lines.append(f"| Sharpe 比 | {metrics.get('sharpe_ratio', 0):.4f} |")
                lines.append(f"| Sortino 比 | {metrics.get('sortino_ratio', 0):.4f} |")
                lines.append(f"| Calmar 比 | {metrics.get('calmar_ratio', 0):.4f} |")
                lines.append(f"| 信息比 | {metrics.get('information_ratio', 0):.4f} |")
                lines.append(f"| 胜率 (日) | {metrics.get('win_rate', 0):.2f}% |")
                lines.append("")
            else:
                lines.append("【收益概览】")
                lines.append(f"  累计收益:       {metrics.get('total_return', 0):>8.2f}%")
                lines.append(f"  年化收益:       {metrics.get('annual_return', 0):>8.2f}%")
                lines.append(f"  年化波动率:     {metrics.get('annual_vol', 0):>8.2f}%")
                lines.append(f"  Sharpe 比:      {metrics.get('sharpe_ratio', 0):>8.4f}")
                lines.append(f"  Sortino 比:     {metrics.get('sortino_ratio', 0):>8.4f}")
                lines.append(f"  Calmar 比:      {metrics.get('calmar_ratio', 0):>8.4f}")
                lines.append(f"  信息比:         {metrics.get('information_ratio', 0):>8.4f}")
                lines.append(f"  胜率 (日):      {metrics.get('win_rate', 0):>7.2f}%")
                lines.append("")

        # ── 基准数据缺失标注 (P1-Q24-fix H08) ──
        if returns_df is not None and "benchmark_return" in returns_df.columns:
            bm_col = returns_df["benchmark_return"]
            bm_missing = int(bm_col.isna().sum())
            if bm_missing > 0:
                warn = (f"⚠️ 基准(沪深300)数据缺失 {bm_missing}/{len(bm_col)} 个交易日，"
                        f"超额收益/信息比可能不完整或不可用")
                if is_md:
                    lines.append(f"> {warn}")
                    lines.append("")
                else:
                    lines.append(f"  {warn}")
                    lines.append("")

        # ── 风险指标 ──
        # P2-Q24-fix (L285): VaR/CVaR 以"损失幅度"展示（取绝对值、非负），
        # 空值/不可计算时显示 "—"，避免负值表达被误读
        def _loss_mag(v: Any) -> str:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "—"
            if not math.isfinite(v):
                return "—"
            return f"{max(0.0, -v):.2f}"

        if metrics and "error" not in metrics:
            if is_md:
                lines.append("## ⚠️ 风险指标")
                lines.append("")
                lines.append("| 指标 | 数值 |")
                lines.append("|------|------|")
                lines.append(f"| 最大回撤 | {metrics.get('max_drawdown', 0):.2f}% |")
                lines.append(f"| 回撤持续期 | {metrics.get('max_drawdown_duration', 0)} 日 |")
                lines.append(f"| VaR (95%) | {_loss_mag(metrics.get('var_95'))}% (损失幅度) |")
                lines.append(f"| VaR (99%) | {_loss_mag(metrics.get('var_99'))}% (损失幅度) |")
                lines.append(f"| CVaR (95%) | {_loss_mag(metrics.get('cvar_95'))}% (损失幅度) |")
                lines.append(f"| CVaR (99%) | {_loss_mag(metrics.get('cvar_99'))}% (损失幅度) |")
                lines.append(f"| 日收益偏度 | {metrics.get('skewness', 0):.4f} |")
                lines.append(f"| 日收益峰度 | {metrics.get('kurtosis', 0):.4f} |")
                lines.append("")
            else:
                lines.append("【风险指标】")
                lines.append(f"  最大回撤:       {metrics.get('max_drawdown', 0):>8.2f}%")
                lines.append(f"  回撤持续期:     {metrics.get('max_drawdown_duration', 0):>8} 日")
                lines.append(f"  VaR (95%):      {_loss_mag(metrics.get('var_95')):>8}% (损失幅度)")
                lines.append(f"  VaR (99%):      {_loss_mag(metrics.get('var_99')):>8}% (损失幅度)")
                lines.append(f"  CVaR (95%):     {_loss_mag(metrics.get('cvar_95')):>8}% (损失幅度)")
                lines.append(f"  CVaR (99%):     {_loss_mag(metrics.get('cvar_99')):>8}% (损失幅度)")
                lines.append(f"  日收益偏度:     {metrics.get('skewness', 0):>8.4f}")
                lines.append(f"  日收益峰度:     {metrics.get('kurtosis', 0):>8.4f}")
                lines.append("")

        # ── Brinson 归因 ──
        if brinson and "error" not in brinson:
            if is_md:
                lines.append("## 🏭 Brinson 业绩归因")
                lines.append("")
                lines.append(f"**基准收益**: {brinson.get('benchmark_return', 0):.2f}%")
                lines.append(f"**超额收益**: {brinson.get('total_excess_return', 0):.2f}%")
                lines.append("")
                lines.append("| 效应 | 贡献 (%) |")
                lines.append("|------|----------|")
                lines.append(f"| 配置效应 | {brinson.get('allocation_effect', {}).get('total', 0):.2f} |")
                lines.append(f"| 选股效应 | {brinson.get('selection_effect', {}).get('total', 0):.2f} |")
                lines.append(f"| 交互效应 | {brinson.get('interaction_effect', {}).get('total', 0):.2f} |")
                lines.append("")
            else:
                lines.append("【Brinson 业绩归因】")
                lines.append(f"  基准收益:             {brinson.get('benchmark_return', 0):>8.2f}%")
                lines.append(f"  超额收益:             {brinson.get('total_excess_return', 0):>8.2f}%")
                lines.append(f"  配置效应:             {brinson.get('allocation_effect', {}).get('total', 0):>8.2f}%")
                lines.append(f"  选股效应:             {brinson.get('selection_effect', {}).get('total', 0):>8.2f}%")
                lines.append(f"  交互效应:             {brinson.get('interaction_effect', {}).get('total', 0):>8.2f}%")
                lines.append("")

            # 行业详情 (只显示前10)
            sector_details = brinson.get("sector_details", [])
            if sector_details:
                if is_md:
                    lines.append("### 行业归因详情")
                    lines.append("")
                    lines.append("| 行业 | P权重(%) | B权重(%) | P收益(%) | B收益(%) | 配置效应 | 选股效应 |")
                    lines.append("|------|----------|----------|----------|----------|----------|----------|")
                    for sd in sector_details[:10]:
                        lines.append(
                            f"| {sd['sector']} | {sd['p_weight']:.1f} | {sd['b_weight']:.1f} "
                            f"| {sd['p_return']:.2f} | {sd['b_return']:.2f} "
                            f"| {sd['allocation']:.2f} | {sd['selection']:.2f} |"
                        )
                    lines.append("")
                else:
                    lines.append("  行业归因详情 (前10):")
                    lines.append(f"  {'行业':<10} {'P权重':>7} {'B权重':>7} {'P收益':>7} {'B收益':>7} {'配置':>7} {'选股':>7}")
                    lines.append(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
                    for sd in sector_details[:10]:
                        lines.append(
                            f"  {sd['sector']:<10} {sd['p_weight']:>6.1f}% {sd['b_weight']:>6.1f}% "
                            f"{sd['p_return']:>6.2f}% {sd['b_return']:>6.2f}% "
                            f"{sd['allocation']:>6.2f}% {sd['selection']:>6.2f}%"
                        )
                    lines.append("")

        # ── 因子归因 ──
        if factor and "error" not in factor:
            if is_md:
                lines.append("## 🧮 因子归因")
                lines.append("")
                lines.append("| 因子 | 暴露 | 因子收益(%) | 贡献(%) |")
                lines.append("|------|------|-------------|---------|")
                for fname, fdetail in factor.get("details", {}).items():
                    lines.append(
                        f"| {fname} | {fdetail['exposure']:.4f} | {fdetail['return']:.2f} "
                        f"| {fdetail['contribution']:.2f} |"
                    )
                lines.append("")
                lines.append(f"**残差**: {factor.get('residual', 0):.2f}%")
                lines.append("")
            else:
                lines.append("【因子归因】")
                for fname, fdetail in factor.get("details", {}).items():
                    lines.append(
                        f"  {fname:<12}  暴露: {fdetail['exposure']:>8.4f}  "
                        f"收益: {fdetail['return']:>7.2f}%  贡献: {fdetail['contribution']:>7.2f}%"
                    )
                lines.append(f"  残差: {factor.get('residual', 0):>8.2f}%")
                lines.append("")

        # ── 交易分析 ──
        if trade_anal and "error" not in trade_anal:
            if is_md:
                lines.append("## 📝 交易分析")
                lines.append("")
                lines.append("| 指标 | 数值 |")
                lines.append("|------|------|")
                lines.append(f"| 总交易次数 | {trade_anal.get('total_trades', 0)} |")
                lines.append(f"| 买入次数 | {trade_anal.get('buy_count', 0)} |")
                lines.append(f"| 卖出次数 | {trade_anal.get('sell_count', 0)} |")
                lines.append(f"| 总盈亏 | {trade_anal.get('total_pnl', 0):.2f} |")
                lines.append(f"| 总盈亏率 | {trade_anal.get('total_pnl_pct', 0):.2f}% |")
                lines.append(f"| 平均每笔盈亏 | {trade_anal.get('avg_trade_pnl', 0):.2f} |")
                lines.append(f"| 盈利因子 | {trade_anal.get('profit_factor', 0):.2f} |")
                lines.append(f"| 平均持有期 | {trade_anal.get('avg_holding_period_days', 0):.1f} 天 |")
                lines.append(f"| 最大连续亏损 | {trade_anal.get('max_consecutive_losses', 0)} 次 |")
                lines.append("")
            else:
                lines.append("【交易分析】")
                lines.append(f"  总交易次数:       {trade_anal.get('total_trades', 0):>8}")
                lines.append(f"  买入/卖出:        {trade_anal.get('buy_count', 0):>4}/{trade_anal.get('sell_count', 0)}")
                lines.append(f"  总盈亏:           {trade_anal.get('total_pnl', 0):>8.2f}")
                lines.append(f"  总盈亏率:         {trade_anal.get('total_pnl_pct', 0):>8.2f}%")
                lines.append(f"  平均每笔盈亏:     {trade_anal.get('avg_trade_pnl', 0):>8.2f}")
                lines.append(f"  盈利因子:         {trade_anal.get('profit_factor', 0):>8.2f}")
                lines.append(f"  平均持有期:       {trade_anal.get('avg_holding_period_days', 0):>7.1f} 天")
                lines.append(f"  最大连续亏损:     {trade_anal.get('max_consecutive_losses', 0):>8} 次")
                lines.append("")

            # 信号胜率详情
            by_signal = trade_anal.get("by_signal", {})
            if by_signal:
                if is_md:
                    lines.append("### 信号胜率")
                    lines.append("")
                    lines.append("| 信号类型 | 次数 | 胜 | 负 | 胜率(%) | 平均盈亏(%) | 盈利因子 |")
                    lines.append("|----------|------|----|----|---------|-------------|---------|")
                    for sig, stats in sorted(by_signal.items()):
                        lines.append(
                            f"| {sig} | {stats.get('count', 0)} | {stats.get('wins', 0)} "
                            f"| {stats.get('losses', 0)} | {stats.get('win_rate', 0):.1f} "
                            f"| {stats.get('avg_pnl_pct', 0):.1f} "
                            f"| {stats.get('profit_factor', 0):.2f} |"
                        )
                    lines.append("")
                else:
                    lines.append("  信号胜率:")
                    lines.append(
                        f"  {'信号类型':<14} {'次数':>5} {'胜':>5} {'负':>5} "
                        f"{'胜率':>7} {'平均盈亏':>9} {'盈利因子':>9}"
                    )
                    lines.append(f"  {'-'*14} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*9}")
                    for sig, stats in sorted(by_signal.items()):
                        lines.append(
                            f"  {sig:<14} {stats.get('count', 0):>5} {stats.get('wins', 0):>5} "
                            f"{stats.get('losses', 0):>5} {stats.get('win_rate', 0):>6.1f}% "
                            f"{stats.get('avg_pnl_pct', 0):>8.1f}% "
                            f"{stats.get('profit_factor', 0):>8.2f}"
                        )
                    lines.append("")

        # ── 持仓分析 ──
        if positions:
            if is_md:
                lines.append("## 💼 当前持仓")
                lines.append("")
                lines.append("| 代码 | 名称 | 股数 | 成本价 | 现价 | 市值 | 盈亏(%) | 行业 | 信号 |")
                lines.append("|------|------|------|--------|------|------|---------|------|------|")
                for p in positions[:20]:
                    lines.append(
                        f"| {p.get('symbol', '')} | {p.get('name', '')} "
                        f"| {p.get('shares', 0)} | {p.get('cost_price', 0):.2f} "
                        f"| {p.get('current_price', 0):.2f} "
                        f"| {p.get('total_value', 0):.0f} "
                        f"| {p.get('pnl_pct', 0):+.2f} "
                        f"| {p.get('sector', '')} | {p.get('signal_type', '')} |"
                    )
                lines.append("")
            else:
                lines.append("【当前持仓】")
                if len(positions) > 20:
                    positions = positions[:20]
                lines.append(
                    f"  {'代码':<8} {'名称':<8} {'股数':>6} {'成本价':>8} "
                    f"{'现价':>8} {'市值':>10} {'盈亏%':>8} {'行业':<8} {'信号':<10}"
                )
                lines.append(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
                for p in positions:
                    lines.append(
                        f"  {p.get('symbol', ''):<8} {p.get('name', ''):<8} "
                        f"{p.get('shares', 0):>6} {p.get('cost_price', 0):>8.2f} "
                        f"{p.get('current_price', 0):>8.2f} {p.get('total_value', 0):>10.0f} "
                        f"{p.get('pnl_pct', 0):>7.2f}% {p.get('sector', ''):<8} {p.get('signal_type', ''):<10}"
                    )
                lines.append("")

        # ── 滚动指标 (摘要) ──
        if rolling and "error" not in rolling:
            if is_md:
                lines.append("## 📈 滚动指标")
                lines.append("")
                lines.append("| 窗口 | Sharpe | 波动率(%) | 最大回撤(%) |")
                lines.append("|------|--------|-----------|-------------|")
                for w in sorted(rolling.keys()):
                    if isinstance(rolling[w], dict) and "error" not in rolling[w]:
                        lines.append(
                            f"| {w}日 | {rolling[w].get('sharpe', 0):.2f} "
                            f"| {rolling[w].get('vol', 0):.2f} "
                            f"| {rolling[w].get('max_dd', 0):.2f} |"
                        )
                lines.append("")
            else:
                lines.append("【滚动指标】")
                lines.append(f"  {'窗口':>6} {'Sharpe':>8} {'波动率':>8} {'最大回撤':>10}")
                lines.append(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
                for w in sorted(rolling.keys()):
                    if isinstance(rolling[w], dict) and "error" not in rolling[w]:
                        lines.append(
                            f"  {w:>4}日 {rolling[w].get('sharpe', 0):>8.2f} "
                            f"{rolling[w].get('vol', 0):>7.2f}% "
                            f"{rolling[w].get('max_dd', 0):>8.2f}%"
                        )
                lines.append("")

        # ── 尾部 ──
        now_str = datetime.now(CST).strftime("%Y-%m-%d %H:%M")
        if is_md:
            lines.append(f"---\n*报告生成时间: {now_str}*")
        else:
            lines.append(f"{'=' * 60}")
            lines.append(f"  报告生成时间: {now_str}")
            lines.append(f"{'=' * 60}")

        return "\n".join(lines)

    except Exception as e:
        print(f"[performance] generate_performance_report error: {e}", file=sys.stderr)
        return f"生成报告失败: {e}"


# ════════════════════════════════════════════════════════════════
# 8. plot_equity_curve
# ════════════════════════════════════════════════════════════════

def plot_equity_curve(
    returns: pd.Series | pd.DataFrame | None = None,
    title: str = "组合净值曲线",
    returns_df: pd.DataFrame | None = None,
) -> str:
    """
    生成净值曲线图并保存为文件。

    Args:
        returns: 收益序列。如果是 Series 则视为组合收益;
                 如果是 DataFrame 则使用 'portfolio_return' 和 'benchmark_return' 列
        title: 图表标题
        returns_df: 可替代 returns, 包含 portfolio_return 和 benchmark_return 列的 DataFrame

    Returns:
        图片文件路径。如果 matplotlib 不可用, 返回 CSV 数据文件路径。
    """
    try:
        # ── 确定数据源 ──
        if returns_df is not None:
            df = returns_df
        elif isinstance(returns, pd.DataFrame):
            df = returns
        elif isinstance(returns, pd.Series):
            df = returns.to_frame(name="portfolio_return")
        else:
            return _plot_equity_curve_csv(None, title)

        if df.empty:
            return _plot_equity_curve_csv(None, title)

        # ── 构建净值序列 ──
        has_benchmark = "benchmark_return" in df.columns
        has_portfolio = "portfolio_return" in df.columns

        if not has_portfolio:
            # 用第一列作为组合收益
            port_col = df.columns[0]
            nav_port = (1 + df[port_col]).cumprod()
        else:
            nav_port = (1 + df["portfolio_return"]).cumprod()

        nav_bench = None
        if has_benchmark:
            nav_bench = (1 + df["benchmark_return"]).cumprod()

        # ── 优先使用 matplotlib ──
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            from matplotlib.ticker import FuncFormatter

            # ── CJK 字体设置 (优先使用 Noto Sans CJK, 包含完整 Latin + CJK) ──
            for _cjk_candidate in ["Noto Sans CJK JP", "Noto Serif CJK JP",
                                    "Droid Sans Fallback", "SimHei", "WenQuanYi Micro Hei"]:
                try:
                    plt.rcParams["font.sans-serif"] = [_cjk_candidate, "DejaVu Sans"]
                    plt.rcParams["axes.unicode_minus"] = False
                    _test_fig, _test_ax = plt.subplots()
                    _test_ax.set_title("测试Abc123")
                    _test_fig.canvas.draw()
                    plt.close(_test_fig)
                    break  # 成功, 无警告
                except Exception:
                    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
                    continue

            fig, ax = plt.subplots(figsize=(14, 7))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            dates = nav_port.index

            # 组合净值 (主曲线)
            ax.plot(dates, nav_port.values, color="#00d2ff", linewidth=2,
                    label="组合净值", zorder=3)

            # 基准净值
            if nav_bench is not None:
                ax.plot(dates, nav_bench.values, color="#ff6b6b", linewidth=1.5,
                        alpha=0.8, label="沪深300", zorder=2)

            # ── 最大回撤标注 ──
            cumulative = nav_port
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd_idx = drawdown.idxmin()

            # 标记最大回撤区间
            if max_dd_idx is not None and not pd.isna(max_dd_idx):
                ax.annotate(
                    f"最大回撤: {drawdown.min() * 100:.2f}%",
                    xy=(max_dd_idx, nav_port.loc[max_dd_idx]),
                    xytext=(max_dd_idx, nav_port.loc[max_dd_idx] * 0.85),
                    arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.5),
                    fontsize=11, color="#ffd700", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                              edgecolor="#ffd700", alpha=0.8),
                )

                # 填充回撤区域
                ax.fill_between(dates, nav_port, running_max,
                                where=(nav_port < running_max),
                                color="#ff6b6b", alpha=0.15, label="回撤区间")

            # ── 风格美化 ──
            ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=20)
            ax.set_ylabel("净值", fontsize=12, color="white")
            ax.legend(loc="upper left", fontsize=11,
                      facecolor="#16213e", edgecolor="#0f3460",
                      labelcolor="white")
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.tick_params(colors="white")

            # X轴日期格式
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            plt.xticks(rotation=45)

            # Y轴百分比格式
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

            # 设置边框
            for spine in ax.spines.values():
                spine.set_color("#0f3460")

            plt.tight_layout()

            # ── 保存 ──
            safe_title = title.replace(" ", "_").replace("/", "_")
            filename = f"{safe_title}_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}.png"
            filepath = str(CHARTS_DIR / filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
            plt.close(fig)

            # 也保存CSV数据
            csv_path = _plot_equity_curve_csv(nav_port, title, nav_bench=nav_bench)

            return filepath

        except ImportError:
            # matplotlib 不可用, 保存 CSV
            result_path = _plot_equity_curve_csv(nav_port, title, nav_bench=nav_bench)
            return result_path

    except Exception as e:
        print(f"[performance] plot_equity_curve error: {e}", file=sys.stderr)
        return _plot_equity_curve_csv(None, title)


def _plot_equity_curve_csv(
    nav_port: pd.Series | None,
    title: str,
    nav_bench: pd.Series | None = None,
) -> str:
    """保存净值数据为 CSV (matplotlib 不可用时的fallback)."""
    try:
        safe_title = title.replace(" ", "_").replace("/", "_")
        filename = f"{safe_title}_data_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = str(CHARTS_DIR / filename)

        if nav_port is not None:
            # P2-Q28-fix(M366): 校验/转换索引为 datetime，避免整数 RangeIndex
            # 被当 "date" 列导出（历史 CSV 的 date 列实为 0,1,2…）。
            # 注意：pd.to_datetime 会把整数当作 ns 时间戳 → 1970 年假日期，
            # 故整数索引不做转换，直接可见降级为说明文件。
            idx = nav_port.index
            if isinstance(idx, pd.DatetimeIndex):
                pass
            elif idx.dtype.kind in "iu":
                with open(filepath, "w") as f:
                    f.write("# 净值数据索引为整数（无真实日期），已跳过 CSV 导出\n")
                    f.write(f"# index_type: {type(idx).__name__}, rows: {len(idx)}\n")
                return filepath
            else:
                parsed = pd.to_datetime(idx, errors="coerce")
                if parsed.isna().any():
                    with open(filepath, "w") as f:
                        f.write("# 净值数据索引无法解析为日期，已跳过 CSV 导出\n")
                        f.write(f"# index_type: {type(idx).__name__}\n")
                    return filepath
                nav_port = nav_port.copy()
                nav_port.index = parsed

            df_out = nav_port.to_frame(name="组合净值")
            if nav_bench is not None:
                b = nav_bench.reindex(nav_port.index)
                df_out["基准净值"] = b.values
            df_out.index.name = "date"
            # P2-Q28-fix(M366): CSV 落盘前列名断言，防止 date 列被静默写成整数索引
            assert df_out.index.name == "date", "CSV 导出: 索引名必须为 date"
            assert "组合净值" in df_out.columns, "CSV 导出: 缺少组合净值列"
            df_out.to_csv(filepath, encoding="utf-8-sig")
        else:
            with open(filepath, "w") as f:
                f.write("# 净值数据不可用\n")
                f.write(f"# title: {title}\n")
                f.write(f"# generated_at: {datetime.now(CST)}\n")

        return filepath

    except Exception as e:
        fallback = str(CHARTS_DIR / f"equity_data_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}.csv")
        # 空文件
        with open(fallback, "w") as f:
            f.write("# no data\n")
        return fallback


# ════════════════════════════════════════════════════════════════
# 便捷入口：一键生成完整周报
# ════════════════════════════════════════════════════════════════

def _build_snapshot_history(
    trades: list[dict],
    positions: list[dict],
    end_date: datetime,
    start_date: datetime,
) -> list[dict]:
    """从交易流水重建报告期内的多日持仓快照序列。

    方法：以当前持仓（当前价）为锚点，从最新到最旧逐日"撤销"交易，
    得到每个交易日的持仓市值快照。total_value = 持仓市值（不含现金），
    现金由 compute_returns 的现金流项（买卖金额）隐式处理，与
    r=(V_t−V_{t-1}+CF_t)/V_{t-1} 的契约一致。

    修复 Q24#1：full_report 原先只传单个当日快照 → len(dates)<2 →
    周报/月报永远无收益/风险/滚动指标。
    """
    if not trades and not positions:
        return []

    by_date: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        d = t.get("trade_date", t.get("date", ""))
        if d:
            by_date[str(d)].append(t)

    # 当前持仓: symbol -> (shares, price)
    cur = {p["symbol"]: p for p in positions}
    shares = {s: p.get("shares", 0) for s, p in cur.items()}
    cur_price = {
        s: (p.get("current_price") or p.get("cost_price") or 0.0)
        for s, p in cur.items()
    }

    # 每只股票的已知价格序列（交易价 + 当前价，按日期升序）
    symbol_prices: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for d in sorted(by_date):
        for t in by_date[d]:
            sym = t.get("symbol", "")
            price = t.get("price") or 0
            if sym and price:
                symbol_prices[sym].append((d, float(price)))
    end_str = end_date.strftime("%Y-%m-%d")
    for s, pr in cur_price.items():
        if pr:
            symbol_prices[s].append((end_str, float(pr)))
    for lst in symbol_prices.values():
        lst.sort()

    def price_at(sym: str, date_str: str) -> float:
        """该股票在 date_str 当日或之前最近已知价格。"""
        best = 0.0
        for d, pr in symbol_prices.get(sym, []):
            if d <= date_str:
                best = pr
            else:
                break
        return best

    def market_value(on_date: str) -> float:
        return sum(shares[s] * price_at(s, on_date) for s in shares)

    def make_snapshot(date_str: str) -> dict:
        return {
            "date": date_str,
            "total_value": market_value(date_str),
            "cash": 0,
            "positions": [
                {
                    "symbol": s,
                    "shares": shares[s],
                    "price": price_at(s, date_str),
                    "value": shares[s] * price_at(s, date_str),
                }
                for s in shares if shares[s] > 0
            ],
        }

    snapshots: list[dict] = [make_snapshot(end_str)]
    start_str = start_date.strftime("%Y-%m-%d")
    emitted = {end_str}

    # 从最新交易日向最早交易日回退，撤销当日交易还原盘前状态
    for d in sorted(by_date.keys(), reverse=True):
        for t in by_date[d]:
            sym = t.get("symbol", "")
            ttype = t.get("trade_type", t.get("type", ""))
            qty = int(t.get("shares", t.get("qty", 0)) or 0)
            if not sym or qty <= 0:
                continue
            if ttype == "buy":
                shares[sym] = shares.get(sym, 0) - qty   # 撤销买入
            elif ttype == "sell":
                shares[sym] = shares.get(sym, 0) + qty   # 撤销卖出
            if shares.get(sym, 0) < 0:
                shares[sym] = 0  # 交易窗口外建仓的标的, 防止负持仓
        if start_str <= d <= end_str and d not in emitted:
            snapshots.append(make_snapshot(d))
            emitted.add(d)

    snapshots.sort(key=lambda x: x["date"])
    return snapshots


def full_report(
    period_days: int = 7,
    format: str = "text",  # noqa: A002
) -> str:
    """
    一键生成完整绩效报告。

    从数据库读取持仓和交易数据, 自动计算所有指标。

    Args:
        period_days: 报告周期天数, 默认7天(周报)
        format: "text" 或 "markdown"

    Returns:
        格式化报告字符串。
    """
    try:
        db = _get_trade_db()
        positions = db.get_positions()
        trades = db.get_trades(days=period_days + 7)  # 多取一些

        end_date = datetime.now(CST)
        start_date = end_date - timedelta(days=period_days)

        # 修复 Q24#1: 从 DB 构建报告期内多日持仓快照序列（交易流水向后回推）
        snapshots = _build_snapshot_history(trades, positions, end_date, start_date)
        if len(snapshots) < 2:
            return (
                f"生成报告失败: 报告期内可用的持仓快照不足2个"
                f"（交易流水 {len(trades)} 条，持仓 {len(positions)} 个），"
                f"无法计算收益/风险指标，请确认报告期内存在交易记录。"
            )

        # 计算收益
        returns_df = compute_returns(snapshots, trades)
        if returns_df.empty or len(returns_df) < 2:
            return "生成报告失败: 收益序列为空或样本不足，无法生成周报/月报指标。"
        port_returns = returns_df["portfolio_return"]

        # 计算指标（P1-Q24-fix H03/H08: 传入基准序列计算信息比；基准缺失日为 NaN）
        metrics = compute_metrics(
            port_returns,
            benchmark_returns=returns_df["benchmark_return"] if "benchmark_return" in returns_df.columns else None,
        )

        # 交易分析
        trade_anal = trade_analysis(trades)

        # 滚动指标
        rolling = rolling_metrics(port_returns)

        result_dict = {
            "returns_df": returns_df,
            "metrics": metrics,
            "trade_analysis": trade_anal,
            "rolling": rolling,
            "positions": positions,
            "report_title": f"投资组合{'周' if period_days <= 7 else '月'}报",
            "period": f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}",
        }

        return generate_performance_report(result_dict, format=format)

    except ValueError as e:
        return f"生成报告失败: {e}"
    except Exception as e:
        return f"生成报告失败: {e}"


# ════════════════════════════════════════════════════════════════
# __name__ check
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("performance.py - 投资组合绩效分析系统")
    print(f"  函数列表:")
    print(f"    - compute_returns(positions_history, trades) -> pd.DataFrame")
    print(f"    - compute_metrics(returns, risk_free=0.025) -> dict")
    print(f"    - brinson_attribution(p_weights, sector_map, b_weights, returns) -> dict")
    print(f"    - factor_attribution(p_weights, factor_exposures, factor_returns) -> dict")
    print(f"    - rolling_metrics(returns, windows=[21,63,252]) -> dict")
    print(f"    - trade_analysis(trades) -> dict")
    print(f"    - generate_performance_report(result, format='text') -> str")
    print(f"    - plot_equity_curve(returns, title) -> str")
    print(f"    - full_report(period_days=7, format='text') -> str")
    print("OK")
