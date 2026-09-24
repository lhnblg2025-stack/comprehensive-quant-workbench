"""
monitor.py — P1-2 因子拥挤度 / 滚动 IC 失效监控（多因子七步法第 7 步扩展）
==========================================================================
审计_因子层.md P1-2：现有失效监控仅全样本 |IC|/ICIR/胜率 + PSI 静态漂移，
缺 滚动 IC+t 值、因子收益波动率（拥挤）、同类因子相关均值、多空换手率。
A 股因子衰减快、拥挤频繁，无在线指标无法在因子崩溃前降权。

本模块提供纯计算函数（无 IO），可独立测试；每日输出每因子：
  1. 滚动 IC（如 60 日）Series + 其 t 值
  2. 因子收益波动率（拥挤度上升信号）
  3. 同类因子相关均值（冗余/拥挤）
  4. 多空组合换手率

输出结构：每因子 dict/DataFrame 汇总；由 scripts/factor_health_monitor.py
落盘到 data_cache/factor_health/（parquet）。

D4收敛登记: 因子健康检查独特保留；本模块为其滚动/拥挤/换手在线监控补充。
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd


def rolling_ic(ic_series: pd.Series, window: int = 60,
               min_periods: int | None = None) -> pd.Series:
    """滚动 IC 均值（窗口内均值，非相关性重算——IC 已是每日关联度量）。

    ic_series: 逐日 IC Series(index=date)。返回滚动均值 Series（NaN 填充暖窗）。
    """
    if ic_series.empty:
        return pd.Series(dtype=float)
    minp = min_periods if min_periods is not None else max(2, window // 2)
    return ic_series.rolling(window, min_periods=minp).mean()


def rolling_ic_tvalue(ic_series: pd.Series, window: int = 60,
                      min_periods: int | None = None) -> pd.Series:
    """滚动 IC 的 t 值 = mean / (std / sqrt(n))。

    窗口内每日 IC 视为一个序列，t = |mean| / (std / sqrt(n))，
    符号沿用 IC 均值符号。n<min_periods 时 NaN。
    """
    if ic_series.empty:
        return pd.Series(dtype=float)
    minp = min_periods if min_periods is not None else max(2, window // 2)
    roll_mean = ic_series.rolling(window, min_periods=minp).mean()
    roll_std = ic_series.rolling(window, min_periods=minp).std(ddof=1)
    roll_n = ic_series.rolling(window, min_periods=minp).count()
    t = roll_mean / (roll_std / np.sqrt(roll_n))
    t = t.where(roll_std > 1e-12, np.nan)
    return t


def factor_return_volatility(factor_wide: pd.DataFrame,
                             ret_wide: pd.DataFrame,
                             window: int = 60,
                             min_periods: int | None = None) -> pd.Series:
    """因子收益波动率 = 每日多空组合（做多高因子值、做空低因子值）收益的滚动标准差。

    每日 long-short 收益 = 横截面上因子值 z 与未来收益的内积（等价多空 delta 组合），
    用 z 标准化避免量纲差异；滚动 std 度量拥挤度（波动加剧 → 拥挤升温）。

    factor_wide / ret_wide: index=date, columns=股票代码。
    返回 Series(index=date, name='factor_return_vol')。
    """
    dates = factor_wide.index.intersection(ret_wide.index)
    if len(dates) == 0:
        return pd.Series(dtype=float)
    ls_ret = {}
    for d in dates:
        f = factor_wide.loc[d].astype(float)
        r = ret_wide.loc[d].astype(float)
        valid = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if valid.sum() < 20:
            continue
        fv = f[valid]
        rv = r[valid]
        z = (fv - fv.mean())
        s = z.std()
        if s is None or (isinstance(s, float) and s == 0) or pd.isna(s) or s == 0:
            continue
        z = z / s
        ls_ret[d] = float((z * rv).mean())
    if not ls_ret:
        return pd.Series(dtype=float)
    ser = pd.Series(ls_ret).sort_index()
    minp = min_periods if min_periods is not None else max(5, window // 2)
    return ser.rolling(window, min_periods=minp).std(ddof=1)


def factor_correlation_mean(factor_wide: pd.DataFrame,
                            peer_names: list[str],
                            panel_dict: dict[str, pd.DataFrame],
                            window: int = 252,
                            min_periods: int | None = None,
                            corr_matrix: pd.DataFrame | None = None) -> float:
    """同类（族）因子相关均值：目标因子与同族其它因子的横截面 Spearman 相关均值。

    该指标衡量冗余/拥挤：同类因子相关越高 → 信号高度同源 → 拥挤/冗余风险高。

    优先使用传入的 corr_matrix（向量化预计算全相关矩阵，见 monitor_panel），
    否则回退到逐 (peer, date) 横截面秩相关再对时窗求均值。

    factor_wide: 目标因子面板(date×股票)；peer_names: 同类因子名列表；
    panel_dict: {因子名: 面板}，缺 panel 的 peer 跳过。
    """
    name = getattr(factor_wide, "name", "")
    if corr_matrix is not None and name and name in corr_matrix.index:
        peers = [c for c in peer_names
                 if c in corr_matrix.columns and c != name]
        vals = [corr_matrix.loc[name, c] for c in peers
                if pd.notna(corr_matrix.loc[name, c])]
        return float(np.mean(np.abs(vals))) if vals else float("nan")
    peers = [c for c in peer_names if c in panel_dict
             and panel_dict[c] is not None and not panel_dict[c].empty]
    if not peers:
        return float("nan")
    dates = factor_wide.index
    if len(dates) == 0:
        return float("nan")
    recent_dates = dates[-window:]
    corrs = []
    for peer in peers:
        p = panel_dict[peer]
        for d in recent_dates:
            if d not in p.index:
                continue
            f = factor_wide.loc[d].astype(float)
            g = p.loc[d].astype(float)
            mask = f.notna() & g.notna() & np.isfinite(f) & np.isfinite(g)
            n_valid = int(mask.sum()) if hasattr(mask, "sum") else 0
            if n_valid < 20:
                continue
            c = f[mask].rank().corr(g[mask].rank())
            if pd.notna(c):
                corrs.append(c)
    if not corrs:
        return float("nan")
    return float(np.mean(np.abs(corrs)))


def long_short_turnover(factor_wide: pd.DataFrame, window: int = 60,
                        long_q: float = 0.2, short_q: float = 0.2) -> pd.Series:
    """多空组合换手率：每日多/空腿成分相对前一日的变化比例。

    定义：t 日 long 腿 = 因子值前 long_q 分位股票；short 腿 = 后 short_q 分位股票。
    换手率 = (新增成分 + 移出成分) / 腿规模，对两腿平均。
    每日值对时间自然保值。

    factor_wide: index=date, columns=股票代码。返回 Series(index=date, name='ls_turnover')。
    """
    if factor_wide.empty:
        return pd.Series(dtype=float)
    dates = factor_wide.index
    prev_long: set | None = None
    prev_short: set | None = None
    out = {}
    for d in dates:
        f = factor_wide.loc[d].astype(float).dropna()
        if len(f) < 15:
            prev_long, prev_short = None, None
            continue
        q_long = f.quantile(1 - long_q)
        q_short = f.quantile(short_q)
        long_set = set(f[f >= q_long].index)
        short_set = set(f[f <= q_short].index)
        if prev_long is not None and prev_short is not None:
            long_turn = (len(long_set ^ prev_long) / max(len(long_set), 1)
                         if len(long_set) else 0.0)
            short_turn = (len(short_set ^ prev_short) / max(len(short_set), 1)
                          if len(short_set) else 0.0)
            out[d] = (long_turn + short_turn) / 2.0
        prev_long, prev_short = long_set, short_set
    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out, name="ls_turnover").sort_index()


def monitor_factor(ic_series: pd.Series | None,
                   factor_wide: pd.DataFrame | None,
                   ret_wide: pd.DataFrame | None,
                   name: str,
                   category: str = "",
                   peer_names: list[str] | None = None,
                   panel_dict: dict[str, pd.DataFrame] | None = None,
                   ic_window: int = 60,
                   vol_window: int = 60,
                   corr_window: int = 252,
                   corr_matrix: pd.DataFrame | None = None) -> dict:
    """单因子的滚动 IC / t值 / 波动率 / 同类相关 / 换手率 汇总。

    返回 dict：name/category + 最新滚动 IC+t + 波动率 + 相关均值 + 换手率 + 状态标记。
    任意输入缺失 → 对应指标 NaN，绝不伪造。
    corr_matrix: 可选的因子秩相关矩阵（monitor_panel 预计算一次传入，加速同类相关均值）。
    """
    row: dict = {"factor": name, "category": category}
    if factor_wide is not None:
        factor_wide.name = name  # 供 factor_correlation_mean 的 corr_matrix 路径定位
    if ic_series is not None and not ic_series.empty:
        ic_series = ic_series.copy()
        ic_series.name = name

    ric = rolling_ic(ic_series, ic_window) if ic_series is not None and not ic_series.empty \
        else pd.Series(dtype=float)
    row["rolling_ic"] = float(ric.dropna().iloc[-1]) if not ric.dropna().empty else np.nan

    tic = rolling_ic_tvalue(ic_series, ic_window) if ic_series is not None and not ic_series.empty \
        else pd.Series(dtype=float)
    tv = tic.dropna()
    row["ic_tvalue"] = float(tv.iloc[-1]) if not tv.empty else np.nan

    vol = (factor_return_volatility(factor_wide, ret_wide, vol_window)
           if factor_wide is not None and ret_wide is not None else pd.Series(dtype=float))
    vd = vol.dropna()
    row["factor_ret_vol"] = float(vd.iloc[-1]) if not vd.empty else np.nan

    row["peer_corr_mean"] = (factor_correlation_mean(
        factor_wide, peer_names or [], panel_dict or {}, corr_window,
        corr_matrix=corr_matrix)
        if factor_wide is not None and (panel_dict or corr_matrix is not None) else np.nan)

    to = (long_short_turnover(factor_wide) if factor_wide is not None
          else pd.Series(dtype=float))
    tod = to.dropna()
    row["ls_turnover"] = float(tod.iloc[-1]) if not tod.empty else np.nan

    # 失效/拥挤告警判定（阈值默认）
    alerts = []
    if pd.notna(row["ic_tvalue"]) and abs(row["ic_tvalue"]) < 1.0:
        alerts.append("ic_tvalue低(失效)")
    if pd.notna(row["factor_ret_vol"]) and row["factor_ret_vol"] > 0.05:
        alerts.append("波动加剧(拥挤)")
    if pd.notna(row["peer_corr_mean"]) and row["peer_corr_mean"] > 0.7:
        alerts.append("同类高度相关(冗余/拥挤)")
    if pd.notna(row["ls_turnover"]) and row["ls_turnover"] > 0.5:
        alerts.append("换手过高(拥挤)")
    row["alerts"] = ";".join(alerts)
    # W2.5 修复: 所有关键指标均 NaN（数据缺失）不得判 "ok"
    if all(pd.isna(row.get(k, float("nan"))) for k in
           ("ic_tvalue", "factor_ret_vol", "peer_corr_mean", "ls_turnover")):
        row["status"] = "insufficient_data"
    else:
        row["status"] = "alert" if alerts else "ok"
    return row


def monitor_panel(panels: dict[str, pd.DataFrame],
                  ic_map: dict[str, pd.Series] | None = None,
                  ret_wide: pd.DataFrame | None = None,
                  category_map: dict[str, str] | None = None,
                  ic_window: int = 60,
                  vol_window: int = 60,
                  alert_floor: int = 0,
                  use_fast_corr: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """批量监控。返回 (报告DataFrame, 告警因子名列表)。

    panels: {因子名: 面板(date×股票)}；ic_map: {因子名: ic Series}（None→缺省 NaN）。
    use_fast_corr: True 时先一次性向量化算因子相关矩阵，再复用计算各因子的
    "同类因子相关均值"，避免逐因子逐 (peer,date) 重算（大幅提速，结果等价）。
    """
    # 预计算相关矩阵（fast path）
    corr_matrix = None
    if use_fast_corr and panels:
        try:
            from quant_system.ic_factors.dedup import factor_corr_matrix as _fcm
            corr_matrix = _fcm(panels)
        except Exception as e:  # noqa: BLE001
            log.warning(f"[P1-2] 预计算因子相关矩阵失败，回退慢路径: {e}")
            corr_matrix = None
    rows = []
    alerted = []
    for name, panel in panels.items():
        ic = (ic_map.get(name) if ic_map else None)
        cat = (category_map or {}).get(name, "")
        r = monitor_factor(ic, panel, ret_wide, name, category=cat,
                           peer_names=[c for c in panels
                                       if c != name and (category_map or {}).get(c) == cat],
                           panel_dict=panels,
                           ic_window=ic_window, vol_window=vol_window,
                           corr_matrix=corr_matrix)
        rows.append(r)
        if r["status"] == "alert":
            alerted.append(name)
    df = pd.DataFrame(rows)
    return df, alerted
