"""
ic_vectorized.py — 向量化 IC 回测（全市场仓库版）
===================================================
解决 factor_ic_backtest.py 的两个限制：
  1. 采样设计：n_days=250 + dates[::5] 每5日采样 → 观测仅 49 天
     → 本工具对每只股票一次性向量化计算完整因子时间序列，
       IC 观测 49 → 600+（全日期），ICIR 统计可靠性大幅提升
  2. 数据源：支持读取本地 data_warehouse（零网络）或 DataLoaderV7

公式严格对照 factors/technical_v7.py 原实现（窗口语义一致）。
少数"窗口内重算"因子（obv_trend/滚动回撤）用等价向量化近似，
结果偏差已在数值校验中确认 < 1e-6 量级（见 _SELF_CHECK）。

用法:
  python3 scripts/ic_vectorized.py --source warehouse --n 800 --forward 5,10,20
  python -m quant_system.ic_factors.scripts.ic_vectorized --source loader --n 60     # 对照旧版
"""
from __future__ import annotations

import sys
from pathlib import Path

# 单栈化：脚本位于 workspace/scripts/，需把 workspace 根加入 sys.path 以导入 quant_system/quant_platform
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import gc

_FULL_REPORT = None  # 全量 IC 结果（CSV 落盘用，2026-08-14 修复）
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors import registry as reg
from quant_system.ic_factors.filter import ic_stats
from quant_system.ic_factors.neutralize_inputs import (
    build_industry_map,
    build_market_cap,
    neutralize_input_summary,
)

warnings.filterwarnings("ignore")
log = get_logger("qv6.ic_vectorized")


def _rsi_vec(gain_avg: pd.Series, loss_avg: pd.Series) -> pd.Series:
    """向量化 RSI（与 technical_v7._rsi 三态对齐）。

    loss==0 且 gain>0 → 100（连续上涨）；gain==0 且 loss>0 → 0（连续下跌）；
    gain==loss==0 → 50（无波动）。其余按标准公式 100-100/(1+gain/loss)。
    np.where 三态分支消除 loss=0 时 g/ls 的除零 RuntimeWarning。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain_avg / loss_avg
        rsi = 100 - 100 / (1 + rs)
    return pd.Series(
        np.where(
            (loss_avg == 0) & (gain_avg > 0), 100.0,
            np.where(
                (gain_avg == 0) & (loss_avg > 0), 0.0,
                np.where((gain_avg == 0) & (loss_avg == 0), 50.0, rsi))),
        index=gain_avg.index)


# ── 单只股票完整因子时间序列（向量化）────────────────────
def _prep(df: pd.DataFrame) -> dict[str, pd.Series]:
    """统一列名 + date index，返回 {col: Series}。"""
    rename = {"日期": "date", "开盘": "open", "收盘": "close",
              "最高": "high", "最低": "low", "成交量": "volume",
              "成交额": "amount", "换手率": "turnover"}
    df = df.rename(columns=rename).copy()
    if "date" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out = {}
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if c in df.columns:
            out[c] = df[c]
    return out


def _series_all(df: pd.DataFrame) -> dict[str, pd.Series]:
    """对单只股票算全部 58 个 tech 因子时间序列。

    返回 {因子名: Series(index=date)}。每项与原实现"在任意日期 t
    用截至 t 的历史窗口计算"的结果一致（rolling 天然仅用过去）。
    """
    s = _prep(df)
    if not s:
        return {}
    c = s["close"]
    h = s.get("high", c)
    l = s.get("low", c)
    o = s.get("open", c)
    v = s.get("volume", pd.Series(np.nan, index=c.index))
    a = s.get("amount", pd.Series(np.nan, index=c.index))
    t = s.get("turnover", pd.Series(np.nan, index=c.index))
    r = c.pct_change()
    out: dict[str, pd.Series] = {}

    # ── 动量 ────────────────────────────────────────────
    for n, nm in [(3, "tech_mom_3_10"), (10, "tech_mom_10"),
                  (20, "tech_mom_20"), (60, "tech_mom_60"),
                  (120, "tech_mom_120"), (5, "tech_ret_week"),
                  (250, "tech_ret_250"), (10, "tech_cum_ret_10")]:
        out[nm] = c.pct_change(n)
    out["tech_mom_accel"] = c.pct_change(20) - c.pct_change(60)
    out["tech_ret_20_60"] = (c.pct_change(20) / c.pct_change(60).abs())
    # 年初至今：原实现基准年 = 窗口(300日)起始日的年份，
    # 基准日 = 该年1月1日后的第一个交易日（且不早于窗口起点）
    dates_arr = c.index
    pos = np.arange(len(c))
    start_pos = np.maximum(pos - 299, 0)
    start_year = np.array([dates_arr[p].year for p in start_pos])
    jan1 = np.array([np.datetime64(f"{y}-01-01") for y in start_year])
    base_pos = np.searchsorted(dates_arr.values.astype("datetime64[D]"),
                               jan1.astype("datetime64[D]"), side="left")
    base_pos = np.maximum(base_pos, start_pos)
    base_close = c.to_numpy()[base_pos]
    out["tech_ret_ytd"] = pd.Series(c.to_numpy() / base_close - 1, index=c.index)

    # ── 均线偏离 / 趋势 ─────────────────────────────────
    for n, nm in [(5, "tech_ma5_dev"), (20, "tech_ma20_dev"),
                  (60, "tech_ma60_dev"), (250, "tech_close_ma250")]:
        ma = c.rolling(n).mean()
        out[nm] = c / ma - 1
    ma5, ma20, ma60 = (c.rolling(n).mean() for n in (5, 20, 60))
    # 注意：原实现 (ma5>ma20)+(ma20>ma60)+(c>ma5) 中 np.bool_ 的 + 是逻辑或！
    out["tech_ma_bull"] = ((ma5 > ma20) | (ma20 > ma60)
                           | (c > ma5)).astype(float)
    ma20r = c.rolling(20).mean()
    out["tech_ma20_slope"] = ma20r / ma20r.shift(5) - 1
    cross520 = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))
    out["tech_ma_cross_5_20"] = (cross520.rolling(3).max().fillna(0))

    # ── RSI（简单均值法，三态与 technical_v7._rsi 对齐）──
    diff = c.diff()
    gain = diff.clip(lower=0)
    loss = (-diff.clip(upper=0))
    for n, nm in [(6, "tech_rsi6"), (14, "tech_rsi14")]:
        out[nm] = _rsi_vec(gain.rolling(n).mean(), loss.rolling(n).mean())
    r20 = _rsi_vec(gain.rolling(20).mean(), loss.rolling(20).mean())
    r50 = _rsi_vec(gain.rolling(50).mean(), loss.rolling(50).mean())
    out["tech_rsi_diff"] = r20 - r50

    # ── KDJ / MACD / DMI ────────────────────────────────
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    out["tech_kdj_j"] = 3 * k - 2 * d
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    out["tech_macd_hist"] = hist
    gold = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    out["tech_macd_gold"] = gold.rolling(3).max().fillna(0)
    up = h.diff().clip(lower=0)
    dn = (-l.diff()).clip(lower=0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    pdi = 100 * up.rolling(14).mean() / atr14.replace(0, np.nan)
    out["tech_dmi_plus"] = pdi

    # ── 波动 ────────────────────────────────────────────
    out["tech_vol20"] = r.rolling(20).std()
    out["tech_vol_skew"] = r.rolling(60).skew()
    out["tech_sharpe_60"] = r.rolling(60).mean() / r.rolling(60).std() * np.sqrt(252)
    # V10 审计 H3 修复：原实现 r.where(r<0).rolling(60).std() 整列 NaN
    # （pandas 对负收益切片 + rolling.std 的 skipna 行为 → 因子恒失效）。
    # 改为：60日窗口内负收益的 std（只对负值），窗口无负收益时给中性分母。
    _neg = r.clip(upper=0)
    _cnt = (_neg != 0).rolling(60).sum()          # 窗口内负收益个数
    _s1 = _neg.rolling(60).sum()
    _s2 = (_neg ** 2).rolling(60).sum()
    with np.errstate(all="ignore"):
        _dsd = np.sqrt(np.maximum(_s2 / _cnt - (_s1 / _cnt) ** 2, 0))  # 负收益 std
    _dsd = _dsd.mask(_cnt < 2, 1.0)               # 负收益过少 → 分母中性
    _dsd = _dsd.mask(_dsd < 1e-9, 1.0)            # 无下行波动 → 分母中性
    out["tech_sortino_60"] = r.rolling(60).mean() / _dsd * np.sqrt(252)
    out["tech_vol_contract"] = r.rolling(5).std() / r.rolling(60).std()
    atr = tr.rolling(14).mean()
    out["tech_atr_pct"] = atr / c
    # 60日最大回撤：精确滚动窗口内相对前高的最大回撤
    # mdd_t = min_j (c_j / max_{i<=j, i∈窗口} c_i - 1)
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    c_v = c.to_numpy(dtype=float)
    mdd_s = np.full(len(c), np.nan)
    if len(c_v) >= 60:
        w = _swv(c_v, 60)
        cummax = np.maximum.accumulate(w, axis=1)
        mdd_s[59:] = (w / cummax - 1).min(axis=1)
    dd = pd.Series(mdd_s, index=c.index)
    out["tech_drawdown_60"] = dd
    ret60 = c / c.shift(59) - 1
    out["tech_calmar_60"] = ret60 / dd.abs()
    w20 = c.rolling(20)
    mid, sd = w20.mean(), w20.std()
    up_, lo_ = mid + 2 * sd, mid - 2 * sd
    out["tech_boll_pos"] = ((c - lo_) / (up_ - lo_)).where(sd >= 1e-9, 0.5)
    out["tech_boll_width"] = sd / mid * 4
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(20).mean()
    # md = 窗口内 |tp - 窗口固定均值| 的均值（sliding_window_view 精确向量化）
    tp_v = tp.to_numpy(dtype=float)
    md_s = np.full(len(tp), np.nan)
    if len(tp_v) >= 20:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        w = swv(tp_v, 20)
        m = w.mean(axis=1)
        md_s[19:] = np.abs(w - m[:, None]).mean(axis=1)
    md = pd.Series(md_s, index=tp.index)
    out["tech_cci20"] = ((tp - tp_ma) / (0.015 * md)).where(md >= 1e-9, 0.0)
    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    out["tech_williams_r"] = (hh14 - c) / (hh14 - ll14) * -100
    out["tech_hl_range10"] = ((h.rolling(10).max() - l.rolling(10).min())
                              / h.rolling(10).mean())

    # ── 量价 / 流动性 ───────────────────────────────────
    out["tech_vol_ratio"] = v / v.shift(1).rolling(5).mean()
    out["tech_vol_ma20"] = v / v.rolling(20).mean()
    out["tech_vol_std20"] = v.rolling(20).std() / v.rolling(20).mean()
    out["tech_amount_trend"] = (a.rolling(5).mean()
                                / a.shift(5).rolling(6).mean() - 1)
    amu, asd = a.rolling(20).mean(), a.rolling(20).std()
    out["tech_amount_z"] = (a - amu) / asd
    out["tech_price_vol_div"] = (c.pct_change(5) - v.pct_change(5))
    tu, tsd = t.rolling(20).mean(), t.rolling(20).std()
    out["tech_turnover_z"] = (t - tu) / tsd
    # OBV：符号量累积（全局 cumsum）
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    idx = np.arange(len(obv))
    z = obv * idx
    n_win = 25
    sum_y = obv.rolling(n_win).sum()
    sum_z = z.rolling(n_win).sum()
    x0 = idx - (n_win - 1)
    sum_x = pd.Series(idx, index=obv.index).rolling(n_win).sum()
    ssx = n_win * (n_win ** 2 - 1) / 12  # Σ(x-x̄)²
    slope = ((sum_z - x0 * sum_y) - (sum_x - x0 * n_win) * sum_y / n_win) / ssx
    out["tech_obv_slope"] = slope
    # OBV 趋势：原实现为 65日窗口内重算 obv（窗口第一天归零）后的
    # 20日均值 / 60日均值。窗口 [t-64..t] 内 obv = 全局 obv - obv[t-64]
    obv_off = obv.shift(64)
    m20 = obv.rolling(20).mean() - obv_off
    m60 = obv.rolling(60).mean() - obv_off
    out["tech_obv_trend"] = m20 / m60.abs()

    # ── 形态 / 事件 ─────────────────────────────────────
    out["tech_gap_up"] = o / c.shift(1) - 1
    out["tech_candle_body"] = (c - o) / c.shift(1)
    body = pd.concat([c, o], axis=1).max(axis=1)
    out["tech_upper_shadow"] = (h - body) / h
    out["tech_high_20d"] = (h >= h.shift(1).rolling(20).max()).astype(float)
    out["tech_low_20d"] = (l <= l.shift(1).rolling(20).min()).astype(float)
    out["tech_close_pos"] = ((c - c.rolling(20).min())
                             / (c.rolling(20).max() - c.rolling(20).min()))
    out["tech_high_52w_pos"] = ((c - c.rolling(250).min())
                                / (c.rolling(250).max() - c.rolling(250).min()))
    out["tech_breakout_20"] = c / h.shift(1).rolling(20).max() - 1
    gap = o > c.shift(1) * 1.005
    fill = l <= c.shift(1)
    out["tech_gap_fill"] = ((gap & fill).astype(float))
    # 连续上涨天数：累计上涨数 - 最近下跌处的累计上涨数（= 自最近下跌以来的连续上涨）
    upf = (r > 0).astype(float)
    cum_up = upf.cumsum()
    cum_at_down = cum_up.where(upf == 0).ffill().fillna(0)
    out["tech_consec_up"] = (cum_up - cum_at_down).where(upf == 1, 0)
    r20u = (r > 0).rolling(20).sum()
    r20d = (r < 0).rolling(20).sum()
    out["tech_up_down_ratio"] = (r20u / r20d).where(r20d > 0)

    return out


# ── zoo 35 个独立 K线因子向量化 ───────────────────────────────
# 这些是 legacy zoo（price/volatility/technical/volume/liquidity）中
# tech_ 之外的独立因子，全部为“完整序列 rolling 后取各日截面值”。
# 与原实现（每只股票在 t 日取 iloc[-1]）逐点一致：rolling 天然仅用过去。
ZOO_SERIES_FNS = {}


def _zoo_series_all(df: pd.DataFrame, include_gtja: bool = True) -> dict[str, pd.Series]:
    """对单只股票算 zoo 独立 35 因子的时间序列。"""
    s = _prep(df)
    if not s:
        return {}
    c = s["close"]
    h = s.get("high", c)
    l = s.get("low", c)
    v = s.get("volume", pd.Series(np.nan, index=c.index))
    a = s.get("amount", pd.Series(np.nan, index=c.index))
    t = s.get("turnover", pd.Series(np.nan, index=c.index))
    r = c.pct_change()
    out: dict[str, pd.Series] = {}

    # ── price: 动量/反转（rolling 收益率）──
    for n, nm in [(20, "mom20"), (21, "mom1m"), (63, "mom3m"),
                  (126, "mom6m"), (252, "mom12m")]:
        out[nm] = c.pct_change(n)
    out["rev5"] = -c.pct_change(5)   # 短期反转
    out["ma_cross"] = (c.rolling(5).mean() / c.rolling(20).mean() - 1)
    # macd_cross: 原实现 (gap - gap_prev)/px*100，gap = DIF-DEA（价格单位）
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    gap = dif - dea
    out["macd_cross"] = gap.diff() / c * 100
    # cci_20（TP 与 20日均值偏离 / 0.015*平均绝对偏差）
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(20).mean()
    md = (tp - tp_ma).abs().rolling(20).mean()
    out["cci_20"] = (tp - tp_ma) / (0.015 * md)
    # high_low_pos: (close - low252)/(high252 - low252)
    hi252, lo252 = c.rolling(252).max(), c.rolling(252).min()
    out["high_low_pos"] = ((c - lo252) / (hi252 - lo252)).where(hi252 > lo252, 0.5)
    out["bias12"] = (c / c.rolling(12).mean() - 1) * 100   # bias() 返回百分数
    out["new_high_dist"] = (c / h.rolling(60).max() - 1) * 100
    out["new_low_dist"] = (c / l.rolling(60).min() - 1) * 100

    # ── volume ──────────────────────────────────────────
    # volume_trend: 近5日均量 / 前15日均量 - 1（原实现 tail(5)/tail(20).head(15)）
    v5 = v.rolling(5).mean()
    prev15m = (v.rolling(20).sum() - v.rolling(5).sum()) / 15
    out["volume_trend"] = v5 / prev15m - 1
    # volume_surge: 当日量 / 前20日均量（不含当日）
    out["volume_surge"] = v / v.shift(1).rolling(20).mean()
    out["vol_20d"] = np.log1p(v.rolling(20).mean())
    # volume_ratio: 当日量 / 前5日均量（不含当日）
    out["volume_ratio"] = v / v.shift(1).rolling(5).mean()
    # obv_slope: 原实现 obv(sub).tail(20) 的 polyfit 斜率 / |统一基均值|
    # sub = tail(300) → OBV 从窗口起点累积。斜率对常数平移不变（用全序列 OBV
    # 窗口 polyfit），均值需减窗口起点 obv（统一基）
    obv = (np.sign(r) * v).fillna(0).cumsum()
    # obv_slope 完全向量化：20日窗口 polyfit 斜率 = cov(x,y)/var(x)
    # 配对注意：polyfit(x=0..19, s) 中 x=0 对应窗口最早值；
    # obv.shift(i) 的 i 越大越早，故权重取 (19-i) 保证最新值配 x=19。
    x = np.arange(20)
    x_sum = x.sum(); x_sq_sum = (x ** 2).sum()
    denom = x_sq_sum - 20 * (x_sum / 20) ** 2
    # Σ(x·y)：x 固定 0..19（与 polyfit 同向），对窗口内 obv 值加权求和
    roll_x_y = sum((19 - i) * obv.shift(i) for i in range(20))
    roll_sum = obv.rolling(20).sum()
    slope = (roll_x_y - x_sum * roll_sum / 20) / denom
    mean_base = obv.rolling(20).mean() - obv.shift(299)  # 统一基均值
    out["obv_slope"] = slope / (mean_base.abs() + 1e-9)
    # volume_price_fit: 20日量价相关系数（rolling corr 需两个 Series）
    out["volume_price_fit"] = r.rolling(20).corr(v.pct_change())
    # ── volatility ──────────────────────────────────────
    out["realized_vol"] = r.rolling(20).std(ddof=0) * np.sqrt(252) * 100
    # downside_vol 向量化：60日窗口内负收益的 std
    # std = sqrt(Σx²/n - (Σx/n)²)，x 为负收益（正收益 clip 为 0）
    negz = r.clip(upper=0)
    cnt = (negz != 0).rolling(60).sum()          # 有效负收益个数
    s1 = negz.rolling(60).sum()
    s2 = (negz ** 2).rolling(60).sum()
    with np.errstate(all="ignore"):
        dv = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)) * np.sqrt(252) * 100
    dv = dv.mask(cnt < 3, 0.0)
    out["downside_vol"] = dv
    # max_dd_12m: 原实现 max_drawdown(close.tail(250), 250)
    # = rolling(250, min_periods=2).max() 滚动累积 max（非窗口内独立 cummax）
    roll_max = c.rolling(250, min_periods=2).max()
    out["max_dd_12m"] = (c / roll_max - 1) * 100
    # vol_change: 20日std/60日std - 1
    out["vol_change"] = r.rolling(20).std(ddof=0) / r.rolling(60).std(ddof=0) - 1
    # atr_ratio
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_ratio"] = tr.rolling(14).mean() / c
    # beta_60d / downside_beta：无基准时对全体均值（截面一致）
    # 这里先留占位，面板层再算（需要全市场均值序列）

    # ── liquidity ───────────────────────────────────────
    out["amount_liquidity"] = np.log1p(a.rolling(20).mean())
    amt_ = a.replace(0, np.nan)
    out["amihud_illiq"] = (r.abs() / amt_).rolling(20).mean() * 1e6
    v20 = v.rolling(20)
    out["turnover_stability"] = (-v20.std(ddof=0) / v20.mean())
    out["spread_approx"] = ((h - l) / c.replace(0, np.nan)).rolling(20).mean()

    # ── technical ───────────────────────────────────────
    # macd_div: 30日窗口内价格新高但 hist 峰值下降 → -1/1/0
    hist30 = hist.rolling(30).max()
    price30 = c.rolling(30).max()
    hist_min30 = hist.rolling(30).min()
    price_min30 = c.rolling(30).min()
    top_div = ((c >= price30 * 0.98) & (hist < hist30 * 0.9)).astype(float)
    bot_div = ((c <= price_min30 * 1.02) & (hist > hist_min30 * 1.1)).astype(float)
    out["macd_div"] = top_div * (-1.0) + bot_div * 1.0
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    out["kdj"] = 3 * k - 2 * d
    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    # williams: 原实现返回 -WR（超卖→正→看多）
    out["williams"] = (hh14 - c) / (hh14 - ll14).replace(0, np.nan) * 100
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=c.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=c.index)
    atr_n = tr.rolling(14).sum()
    pdi = 100 * plus_dm.rolling(14).sum() / atr_n.replace(0, np.nan)
    mdi = 100 * minus_dm.rolling(14).sum() / atr_n.replace(0, np.nan)
    out["dmi"] = pdi - mdi
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    out["donchian"] = ((c - lo20) / (hi20 - lo20)).where(hi20 > lo20, 0.5)
    # V10 审计 M4 修复：boll_width 统一注册名（原 boll_width_zoo 与 zoo.py 注册名不一致
    # → 白名单 grade 查不到，C/D 降权失效）。数值 4*std/mean 与原 boll_width 一致。
    out["boll_width"] = (c.rolling(20).std(ddof=0) / c.rolling(20).mean()) * 4
    out["ma_trend"] = c.rolling(20).mean() / c.rolling(60).mean() - 1

    # ── V10 审计 M4 修复：补齐原 v6 zoo 41 因子中缺失的 6 个 ──
    # macd_hist / boll_pos / rsi14：原 price.py 横截面实现取“最新值”，
    # 向量化等价 = 完整序列（rolling 天然仅用过去，t 日值 = 原实现 iloc[-1]）。
    ema12_ = c.ewm(span=12, adjust=False).mean()
    ema26_ = c.ewm(span=26, adjust=False).mean()
    dif_ = ema12_ - ema26_
    dea_ = dif_.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (dif_ - dea_) * 2          # 原 macd() hist = (dif-dea)*2
    _mid = c.rolling(20).mean()
    _sd = c.rolling(20).std(ddof=0)
    _up = _mid + 2 * _sd
    _lo = _mid - 2 * _sd
    out["boll_pos"] = ((c - _lo) / (_up - _lo).replace(0, np.nan)).where(
        (_up - _lo) >= 1e-9, 0.5)
    _diff = c.diff()
    _gain = _diff.clip(lower=0).rolling(14).mean()
    _loss = (-_diff.clip(upper=0)).rolling(14).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        _rs = _gain / _loss
        out["rsi14"] = 100 - 100 / (1 + _rs)
    out["rsi14"] = out["rsi14"].fillna(50.0)     # 原 rsi() fillna(50)

    # beta_60d / downside_beta：原实现无基准时“对全体均值”（r.mean()*100 / 1.0）。
    # 向量化：个股 60 日收益均值×100 作为 beta 代理（无基准退化为均值收益），
    # 方向=-1（低 beta 抗跌偏好）在合成层统一处理。
    out["beta_60d"] = r.rolling(60).mean() * 100
    # downside_beta：无基准时原实现恒 1.0；用 60 日负收益均值×100 做代理
    # （下跌日越多/越深 → 值越大 → 方向=-1 偏好抗跌）。
    _dneg = r.clip(upper=0).rolling(60).mean() * 100
    out["downside_beta"] = _dneg.where(_dneg != 0, 1.0)

    # 2026-08-14: GTJA 因子时间序列（复用 gtja.py 辅助函数, 每期值,
    # 非截面最新值——IC 检验需要全序列）。调用方若单独构建 GTJA 面板，
    # 传 include_gtja=False 避免 5,000+ 股票重复计算一遍。
    if include_gtja:
        try:
            _gtja_df = pd.DataFrame(s)
            _gtja_series = _gtja_series_all(_gtja_df)
            out.update(_gtja_series)
        except Exception:
            pass

    # V12.3 因子扩充第二批: K线横截面 18 个(alpha101精简/微观结构/动量增强)
    try:
        from quant_system.ic_factors.kline_extra import kline_extra_series
        out.update({k: v for k, v in kline_extra_series(s).items()})
    except Exception as e:  # noqa: BLE001
        log.warning(f"[kline_extra] 计算失败: {e}")
    return out


def _gtja_series_all(df: pd.DataFrame) -> dict[str, pd.Series]:
    """GTJA 18 因子完整时间序列（复用 quant_system.ic_factors.gtja 辅助函数）。

    gtja.py 的因子函数是"截面最新值"式（供 zoo 截面使用），IC 检验需要
    每只股票的完整因子序列，故在此用其模块级辅助函数重算整列。
    """
    from quant_system.ic_factors import gtja as _g
    import numpy as _np
    eps = 1e-8
    out: dict[str, pd.Series] = {}
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float).replace(0, _np.nan)
    typ = (h + l + c) / 3.0

    # gtja_001: -CORR(RANK(ΔlnV,1), RANK((C-O)/O), 6)
    out["gtja_001"] = -_g._roll_corr(_g._tsrank(_g._delta(_np.log(v), 1), 6),
                                     _g._tsrank((c - o) / (o + eps), 6), 6)
    # gtja_002: -DELTA(((C-L)-(H-C))/(H-L), 1)
    pos = ((c - l) - (h - c)) / (h - l + eps)
    out["gtja_002"] = -_g._delta(pos, 1)
    # gtja_003: ROLLSUM(close==delay?0:close-(c>delay?min(l,delay):max(h,delay)), 6)
    prev = _g._delay(c, 1)
    term = pd.Series(0.0, index=df.index)
    m = prev.notna()
    term[m] = c[m] - _np.where(c[m] > prev[m], _np.minimum(l[m], prev[m]),
                               _np.maximum(h[m], prev[m]))
    term[c == prev] = 0.0
    out["gtja_003"] = _g._roll_sum(term, 6)
    # gtja_005: -TSMAX(CORR(TSRANK(V,5),TSRANK(H,5),5),3)
    out["gtja_005"] = -_g._roll_max(_g._roll_corr(_g._tsrank(v, 5), _g._tsrank(h, 5), 5), 3)
    # gtja_006: -RANK(SIGN(DELTA(O*0.85+H*0.15, 4)))
    out["gtja_006"] = -pd.Series(_np.sign(_g._delta(o * 0.85 + h * 0.15, 4).fillna(0.0)),
                                 index=df.index).rolling(df.shape[0]).rank(pct=True)
    # gtja_007: RANK(MAX(VWAP-C,3)+RANK(MIN(VWAP-C,3))*RANK(DELTA(V,3)))
    diff = typ - c
    dv = _g._delta(v, 3).fillna(0.0)
    x = _g._roll_max(diff, 3).fillna(0.0) + diff.rolling(df.shape[0]).rank(pct=True) * dv.rolling(df.shape[0]).rank(pct=True)
    out["gtja_007"] = x.rolling(df.shape[0]).rank(pct=True)
    # gtja_008: RANK(DELTA(((H+L)/2*0.2+VWAP*0.8), 4)*-1)
    y = (h + l) / 2 * 0.2 + typ * 0.8
    out["gtja_008"] = (-_g._delta(y, 4).fillna(0.0)).rolling(df.shape[0]).rank(pct=True)
    # gtja_009: SMA((H+L)/2-(DLYH+DLYL)/2)*(H-L)/V, 7, 2) ≈ ewm alpha=2/7
    z = ((h + l) / 2 - (_g._delay(h, 1) + _g._delay(l, 1)) / 2) * (h - l) / (v + eps)
    out["gtja_009"] = z.ewm(alpha=2 / 7, adjust=False).mean()
    # gtja_011: SUM(((C-L)-(H-C))/(H-L)*V, 6)
    out["gtja_011"] = _g._roll_sum(pos * v, 6)
    # gtja_012: RANK(O-VWAP)*RANK(C-VWAP)
    out["gtja_012"] = (o - typ).rolling(df.shape[0]).rank(pct=True) * (c - typ).rolling(df.shape[0]).rank(pct=True)
    # gtja_013: SQRT(H*L)-VWAP
    out["gtja_013"] = _np.sqrt(h * l) - typ
    # gtja_014: C-DELAY(C,5)
    out["gtja_014"] = _g._delta(c, 5)
    # gtja_015: O/DELAY(C,1)-1
    out["gtja_015"] = o / (_g._delay(c, 1) + eps) - 1
    # gtja_016: -MAX(CORR(RANK(V),RANK(VWAP),5),5)
    out["gtja_016"] = -_g._roll_max(_g._roll_corr(_g._tsrank(v, 5), _g._tsrank(typ, 5), 5), 5)
    # gtja_017: RANK(VWAP-MAX(VWAP,15))^DELTA(C,5) (符号保持, 用rank平滑)
    dd = (typ - _g._roll_max(typ, 15).fillna(typ)).rolling(df.shape[0]).rank(pct=True)
    out["gtja_017"] = _np.sign(_g._delta(c, 5).fillna(0.0)) * dd
    # gtja_018: C/DELAY(C,5)-1
    out["gtja_018"] = c / (_g._delay(c, 5) + eps) - 1
    # gtja_019: close<delay? (c-d)/d : (c-d)/c
    d5 = c - _g._delay(c, 5)
    s19 = d5 / (_g._delay(c, 5) + eps)
    up = (_g._delay(c, 5).notna()) & (c >= _g._delay(c, 5)) & (d5 != 0)
    s19 = s19.copy()
    s19[up] = d5[up] / c[up]
    out["gtja_019"] = s19
    # gtja_020: (C-DELAY(C,6))/DELAY(C,6)*100
    out["gtja_020"] = _g._delta(c, 6) / (_g._delay(c, 6) + eps) * 100
    return out


ZOO_FACTOR_NAMES = [
    "mom20", "mom1m", "mom3m", "mom6m", "mom12m", "rev5",
    "ma_cross", "macd_hist", "macd_cross", "boll_pos", "cci_20", "rsi14",
    "high_low_pos", "bias12", "new_high_dist", "new_low_dist",
    "volume_trend", "volume_surge", "vol_20d", "volume_ratio",
    "obv_slope", "volume_price_fit", "realized_vol", "downside_vol",
    "max_dd_12m", "beta_60d", "downside_beta", "vol_change", "atr_ratio",
    "amount_liquidity", "amihud_illiq", "turnover_stability",
    "spread_approx", "macd_div", "kdj", "williams", "dmi",
    "donchian", "boll_width", "ma_trend",
    # 2026-08-14: GTJA 国泰君安 Alpha 因子 18 个（日线近似, eGTAlpha MIT）
    "gtja_001", "gtja_002", "gtja_003", "gtja_005", "gtja_006",
    "gtja_007", "gtja_008", "gtja_009", "gtja_011", "gtja_012",
    "gtja_013", "gtja_014", "gtja_015", "gtja_016", "gtja_017",
    "gtja_018", "gtja_019", "gtja_020",
    # 2026-08-15: K线横截面扩充 18 个（alpha101精简/微观结构/动量增强）
    "vwap_bias", "amt_trend", "volume_std_20", "high_low_range_20", "gap_up_5d",
    "mom_risk_adj", "res_mom_approx", "wma_mom", "skew_20", "kurt_20",
    "ofi_approx", "amihud_5d", "roll_spread",
    "alpha_rank_cm", "alpha_corr_cv", "alpha_tsrank_20", "alpha_delta_vol_5",
    "candle_strength",
]


def _prepare_common(kl: dict[str, pd.DataFrame],
                    sample_dates: pd.DatetimeIndex | None = None
                    ) -> tuple[pd.DatetimeIndex | None, list[str]]:
    """第一遍：只读每只股票的日期索引求公共交易日（不做因子计算，省内存）。

    返回 (common, codes)；公共交易日不足(<30)时返回 (None, [])。
    与旧实现"全部股票算完因子再取交集"语义一致：
    因子序列的 index 就是预处理后 K 线的 date index。
    """
    stock_dates: dict[str, pd.DatetimeIndex] = {}
    for code, df in kl.items():
        s = _prep(df)
        if not s:
            continue
        stock_dates[code] = next(iter(s.values())).index
    common: pd.DatetimeIndex | None = None
    for idx in stock_dates.values():
        common = idx if common is None else common.intersection(idx)
    # 多数对齐：交集被极端次新股压垮时，回退到覆盖度 >=80% 的日期
    if common is not None and len(common) < 100:
        from collections import Counter
        cnt: Counter = Counter()
        for idx in stock_dates.values():
            cnt.update(idx)
        need = max(1, int(len(stock_dates) * 0.8))
        common = pd.DatetimeIndex(sorted(d for d, n in cnt.items() if n >= need))
    if common is None or len(common) == 0:
        return None, []
    common = common.sort_values()
    if sample_dates is not None:
        common = common.intersection(sample_dates)
    if len(common) < 30:
        return None, []
    return common, sorted(stock_dates.keys())


def _spill_panels(panels: dict[str, np.ndarray], common: pd.DatetimeIndex,
                  codes: list[str], spill_dir: Path,
                  reused: dict[str, Path] | None = None) -> dict[str, str]:
    """面板逐因子写 npy 落盘，返回 {因子名: npy路径}。

    reused: 已存在的因子 npy（V10 审计 M5 断点续跑），直接沿用路径不重写。
    """
    spill_dir.mkdir(parents=True, exist_ok=True)
    np.save(spill_dir / "dates.npy", common.values)
    np.save(spill_dir / "codes.npy", np.array(codes, dtype=object),
            allow_pickle=True)
    paths: dict[str, str] = {}
    if reused:
        for nm, oldp in reused.items():
            if oldp.resolve() != (spill_dir / f"{nm}.npy").resolve():
                import shutil as _sh
                _sh.copy2(oldp, spill_dir / f"{nm}.npy")
            paths[nm] = str(spill_dir / f"{nm}.npy")
    for name, arr in panels.items():
        p = spill_dir / f"{name}.npy"
        # _build_panels may already have created this exact file as a memmap.
        # Flush and reuse it instead of np.save-ing over itself.
        if isinstance(arr, np.memmap) and Path(arr.filename).resolve() == p.resolve():
            arr.flush()
        else:
            np.save(p, arr)
        paths[name] = str(p)
    log.info(f"面板已落盘 {len(paths)} 个 npy → {spill_dir}")
    return paths


def _load_spilled(name: str, spill_dir: Path, common: pd.DatetimeIndex,
                  codes: list[str]) -> pd.DataFrame:
    """从 npy 读回单个因子面板（index=common, columns=codes）。"""
    arr = np.load(spill_dir / f"{name}.npy")
    return pd.DataFrame(arr, index=common, columns=codes)


def _build_panels(kl: dict[str, pd.DataFrame], names: list[str], series_fn,
                  common: pd.DatetimeIndex, codes: list[str],
                  spill_dir: Path | None, label: str,
                  reuse_dir: Path | str | None = None) -> dict[str, object]:
    """流式面板构建：逐股票算因子→逐列写入→立即释放该股票序列。

    旧实现先算完 5106 只股票的全部因子序列并整体驻留（~3GB+），
    再填面板；本实现把峰值压到只有面板本身。
    spill_dir 为空 → 返回 {因子名: DataFrame}（内存驻留）；
    否则 → 每因子写 npy 落盘后释放，返回 {因子名: npy路径}。
    reuse_dir 不为空 → 复用 reuse_dir 下已存在的同名 npy（对齐校验后），
    只对缺失因子重新计算（V10 审计 M5：断点续跑，面板构建从 1h → 秒级）。
    ⚠️ spill 模式下复用的因子不加载进内存（只记路径，IC 阶段按需读），
    避免一次性载入全部面板导致 OOM（6GB+）。
    """
    n_d, n_c = len(common), len(codes)
    reused: dict[str, Path] = {}   # 因子名 → 已有 npy 路径（复用，不加载）
    arrs: dict[str, np.ndarray] = {}
    if reuse_dir is not None:
        reuse_dir = Path(reuse_dir)
        # 先确认日期/代码对齐：复用目录的 dates/codes 必须与本次一致
        try:
            old_dates = np.load(reuse_dir / "dates.npy")
            old_codes = np.load(reuse_dir / "codes.npy", allow_pickle=True)
            align_ok = (len(old_dates) == n_d and len(old_codes) == n_c
                        and np.array_equal(old_dates, common.values)
                        and np.array_equal(old_codes, np.array(codes, dtype=object)))
        except Exception:
            align_ok = False
        if align_ok:
            for nm in list(names):
                p = reuse_dir / f"{nm}.npy"
                if p.exists():
                    if spill_dir is not None:
                        reused[nm] = p          # 落盘模式：只记路径不加载
                    else:
                        arrs[nm] = np.load(p)   # 内存模式：直接载入
            if reused or arrs:
                log.info(f"{label} 复用已落盘面板 {len(reused) + len(arrs)} 个因子，仅重算缺失 {len(names) - len(reused) - len(arrs)} 个")
            else:
                log.info(f"{label} reuse_dir 无可用面板，全量重建")
        else:
            log.warning(f"{label} reuse_dir 日期/代码与本次不一致，跳过复用（全量重建）")
    memmaps: dict[str, np.memmap] = {}
    for nm in names:
        if nm not in arrs and nm not in reused:
            # Full-universe research must not allocate every factor panel in RAM.
            if spill_dir is not None:
                spill_dir = Path(spill_dir)
                spill_dir.mkdir(parents=True, exist_ok=True)
                memmaps[nm] = np.lib.format.open_memmap(
                    spill_dir / f"{nm}.npy", mode="w+", dtype=np.float32,
                    shape=(n_d, n_c),
                )
                memmaps[nm][:] = np.nan
            else:
                # Factor ranks/IC tolerate float32 while halving panel memory.
                arrs[nm] = np.full((n_d, n_c), np.nan, dtype=np.float32)
    codes_pos = {c: i for i, c in enumerate(codes)}
    touched: set[str] = set()
    for code, df in kl.items():
        if code not in codes_pos:
            continue
        ss = series_fn(df)
        if not ss:
            continue
        ci = codes_pos[code]
        for nm in list(arrs) + list(memmaps):  # 只填需要新算的因子
            s = ss.get(nm)
            if s is not None:
                target = arrs.get(nm) if nm in arrs else memmaps[nm]
                values = s.reindex(common).values
                target[:, ci] = values
                if np.isfinite(values).any():
                    touched.add(nm)
        del ss
    # 去掉全 NaN 面板（只针对新算的），释放 memmap 写句柄前 flush。
    for target in memmaps.values():
        target.flush()
    alive = {nm: a for nm, a in arrs.items() if nm in touched}
    alive.update({nm: memmaps[nm] for nm in memmaps if nm in touched})
    if spill_dir is not None:
        # 复用因子：沿用旧 npy（若目录不同则复制），不重写不加载
        for nm, oldp in reused.items():
            if oldp.resolve() != Path(spill_dir).resolve() / f"{nm}.npy":
                import shutil as _sh
                _sh.copy2(oldp, Path(spill_dir) / f"{nm}.npy")
        log.info(f"{label} 面板构建完成: {len(alive) + len(reused)} 因子 × {n_d} 观测 × {n_c} 股票（新算 {len(alive)} + 复用 {len(reused)}）")
        return _spill_panels(alive, common, codes, spill_dir, reused)
    return {nm: pd.DataFrame(a, index=common, columns=codes)
            for nm, a in alive.items()}


def build_zoo_panels(kl: dict[str, pd.DataFrame],
                     factor_names: list[str] | None = None,
                     sample_dates: pd.DatetimeIndex | None = None,
                     spill_dir: Path | None = None,
                     common: pd.DatetimeIndex | None = None,
                     codes: list[str] | None = None,
                     reuse_dir: Path | str | None = None) -> dict[str, object]:
    """zoo 独立 35 因子面板（与 build_tech_panels 同构，支持落盘）。"""
    names = factor_names or ZOO_FACTOR_NAMES
    if common is None or codes is None:
        common, codes = _prepare_common(kl, sample_dates)
        if not codes:
            return {}
    return _build_panels(kl, names, _zoo_series_all, common, codes,
                         spill_dir, "zoo", reuse_dir)


# ── 财务因子面板（fa_*：新浪财务分析指标，公告日锚点防前视）──────
# 新浪无真实公告日 → 锚点估算：季报报告期末 +90 天，年报 +150 天
# （与 financial_v7._est_pub_date 一致）。
FA_COLUMNS: dict[str, str] = {
    "fa_roa": "总资产利润率(%)",
    "fa_operating_margin": "营业利润率(%)",
    "fa_net_margin": "销售净利率(%)",
    "fa_cost_profit_ratio": "成本费用利润率(%)",
    "fa_roe_adj": "净资产报酬率(%)",
    "fa_eps_cfo": "每股经营性现金流(元)",
    "fa_capital_reserve": "每股资本公积金(元)",
    "fa_retained_eps": "每股未分配利润(元)",
    "fa_main_cost_ratio": "主营业务成本率(%)",
    "fa_asset_turnover": "总资产周转率(次)",
    "fa_interest_cover": "利息保障倍数",
    "fa_equity_growth": "股东权益增长率(%)",
    "fa_sustain_growth": "可持续增长率(%)",
    "fa_eps_adjusted": "扣除非经常性损益后的每股收益(元)",
    "fa_eps_weighted": "加权每股收益(元)",
    "fa_bps": "每股净资产_调整前(元)",
    "fa_total_asset_profit": "总资产净利润率(%)",
    "fa_main_biz_profit": "主营业务利润率(%)",
    "fa_roe_weighted": "净资产收益率加权(%)",
    "fa_quick_ratio": "速动比率",
    "fa_current_ratio": "流动比率",
    "fa_inventory_turnover": "存货周转率(次)",
    "fa_receivable_turnover": "应收账款周转率(次)",
    "fa_operating_cycle": "营业周期(天)",
    "fa_cash_ratio": "现金比率(%)",
}
# 组合因子：经营现金流 / 摊薄每股收益（两列相除）
FA_RATIO = ("fa_cfo_to_np", "每股经营性现金流(元)", "摊薄每股收益(元)")


def build_financial_panels(fin: dict[str, pd.DataFrame],
                           common: pd.DatetimeIndex,
                           codes: list[str],
                           spill_dir: Path | None = None,
                           reuse_dir: Path | str | None = None) -> dict[str, object]:
    """fa_* 财务因子面板（26 个，季度数据，公告日锚点防前视）。

    fin: {code: DataFrame}，新浪 stock_financial_analysis_indicator，
    必须含 "日期"（报告期）列。
    每只股票：报告期 → 估算公告日（季报+90d/年报+150d），
    对每个观测日取"截至该日已公告"的最新一期值 → date×code 面板。
    """
    if not fin:
        return {}
    n_d, n_c = len(common), len(codes)
    codes_pos = {c: i for i, c in enumerate(codes)}
    names = list(FA_COLUMNS) + [FA_RATIO[0]]
    reused: dict[str, Path] = {}
    arrs: dict[str, np.ndarray] = {}
    if reuse_dir is not None:
        reuse_dir = Path(reuse_dir)
        try:
            old_dates = np.load(reuse_dir / "dates.npy")
            old_codes = np.load(reuse_dir / "codes.npy", allow_pickle=True)
            if (len(old_dates) == n_d and len(old_codes) == n_c
                    and np.array_equal(old_dates, common.values)
                    and np.array_equal(old_codes, np.array(codes, dtype=object))):
                for nm in names:
                    p = reuse_dir / f"{nm}.npy"
                    if p.exists():
                        if spill_dir is not None:
                            reused[nm] = p
                        else:
                            arrs[nm] = np.load(p)
                if reused or arrs:
                    log.info(f"fa 复用已落盘面板 {len(reused)+len(arrs)} 个因子，仅重算缺失 {len(names)-len(reused)-len(arrs)} 个")
        except Exception as e:
            log.error(f"[ic_vectorized] 操作失败: {e}", exc_info=True)
    for nm in names:
        if nm not in arrs and nm not in reused:
            arrs[nm] = np.full((n_d, n_c), np.nan, dtype=float)
    common_ns = common.values.astype("datetime64[ns]")
    for code, df in fin.items():
        if code not in codes_pos or df is None or df.empty:
            continue
        if "日期" not in df.columns:
            continue
        r = pd.to_datetime(df["日期"], errors="coerce")
        m = r.notna()
        if not m.any():
            continue
        r = r[m]
        dsub = df[m]
        # 公告日锚点估算（与 financial_v7._est_pub_date 一致）
        pub = r + pd.to_timedelta(
            np.where(r.dt.month == 12, 150, 90), unit="D")
        ord_idx = np.argsort(pub.to_numpy(), kind="stable")
        pub_sorted = pub.to_numpy()[ord_idx]
        dsub_sorted = dsub.iloc[ord_idx]
        pos = np.searchsorted(pub_sorted, common_ns, side="right") - 1
        pos = np.clip(pos, 0, len(pub_sorted) - 1)
        has = common_ns >= pub_sorted[0]
        ci = codes_pos[code]
        for nm in names:
            if nm == FA_RATIO[0]:
                a = pd.to_numeric(dsub_sorted[FA_RATIO[1]],
                                  errors="coerce").to_numpy()
                b = pd.to_numeric(dsub_sorted[FA_RATIO[2]],
                                  errors="coerce").to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(np.abs(b) > 1e-9,
                                 a / np.where(np.abs(b) > 1e-9, b, np.nan),
                                 np.nan)
            else:
                col = FA_COLUMNS[nm]
                if col not in dsub_sorted.columns:
                    continue
                v = pd.to_numeric(dsub_sorted[col],
                                  errors="coerce").to_numpy()
            vals = v[pos]
            arrs[nm][:, ci] = np.where(has, vals, np.nan)
        del dsub_sorted
    alive = {nm: a for nm, a in arrs.items() if (~np.isnan(a)).sum() > 0}
    if spill_dir is not None:
        return _spill_panels(alive, common, codes, spill_dir, reused)
    return {nm: pd.DataFrame(a, index=common, columns=codes)
            for nm, a in alive.items()}


# ── 估值因子面板（val_*：东财日频估值）────────────────────
VAL_LEVEL_COLS: dict[str, str | None] = {
    "val_pe_ttm": "pe_ttm",
    "val_pb": "pb",
    "val_ps": "ps",
    "val_pe_pb_ratio": None,  # pe_ttm / pb
}
VAL_PCT_COLS: dict[str, str] = {
    "val_pe_ttm_pct": "pe_ttm",
    "val_pb_pct": "pb",
    "val_ps_pct": "ps",
    "val_pcf_pct": "pcf",
}
PCT_WINDOW = 750  # 近3年（约 750 交易日）分位窗口


def _rolling_pct(x: np.ndarray, win: int, min_count: int = 60) -> np.ndarray:
    """滚动窗口内当前值分位（窗口内 <= 当前值占比）。
    窗口有效值 < min_count 或当前值为 NaN → NaN。"""
    n = len(x)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    if n >= win:
        w = _swv(x, win)
        cur = w[:, -1]
        cnt = np.sum(~np.isnan(w), axis=1)
        le = np.sum(np.where(np.isnan(w), False, w <= cur[:, None]), axis=1)
        valid = (cnt >= min_count) & ~np.isnan(cur)
        out[win - 1:] = np.where(valid, le / np.where(cnt > 0, cnt, 1),
                                 np.nan)
    else:
        # 历史不足窗口：用全部可用历史（与 _percentile_cross len>=60 语义）
        cur = x[-1]
        if np.isnan(cur):
            return out
        w = x[:n]
        cnt = np.sum(~np.isnan(w))
        if cnt >= min_count:
            le = np.sum(np.where(np.isnan(w), False, w <= cur))
            out[n - 1] = le / cnt
    return out


def build_valuation_panels(val: dict[str, pd.DataFrame],
                           common: pd.DatetimeIndex,
                           codes: list[str],
                           spill_dir: Path | None = None,
                           reuse_dir: Path | str | None = None) -> dict[str, object]:
    """val_* 估值因子面板（8 个，日频，东财 stock_value_em）。

    val: {code: DataFrame(date, pe_ttm, pb, ps, pcf, ...)}。
    水平类：原值对齐公共日（ffill 缺口，不做 bfill 防前视）。
    分位类：每只股票滚动 750 日窗口内当前值分位（近3年分位）。
    兼容 baostock 旧列名（peTTM/pbMRQ/psTTM/pcfNcfTTM）。
    """
    if not val:
        return {}
    try:
        from quant_system.ic_factors.valuation_v7 import normalize_valuation_columns
    except Exception:  # noqa: BLE001
        normalize_valuation_columns = None
    n_d, n_c = len(common), len(codes)
    codes_pos = {c: i for i, c in enumerate(codes)}
    names = list(VAL_LEVEL_COLS) + list(VAL_PCT_COLS)
    reused: dict[str, Path] = {}
    arrs: dict[str, np.ndarray] = {}
    if reuse_dir is not None:
        reuse_dir = Path(reuse_dir)
        try:
            old_dates = np.load(reuse_dir / "dates.npy")
            old_codes = np.load(reuse_dir / "codes.npy", allow_pickle=True)
            if (len(old_dates) == n_d and len(old_codes) == n_c
                    and np.array_equal(old_dates, common.values)
                    and np.array_equal(old_codes, np.array(codes, dtype=object))):
                for nm in names:
                    p = reuse_dir / f"{nm}.npy"
                    if p.exists():
                        if spill_dir is not None:
                            reused[nm] = p
                        else:
                            arrs[nm] = np.load(p)
                if reused or arrs:
                    log.info(f"val 复用已落盘面板 {len(reused)+len(arrs)} 个因子，仅重算缺失 {len(names)-len(reused)-len(arrs)} 个")
        except Exception as e:
            log.error(f"[ic_vectorized] 操作失败: {e}", exc_info=True)
    for nm in names:
        if nm not in arrs and nm not in reused:
            arrs[nm] = np.full((n_d, n_c), np.nan, dtype=float)
    common_ns = common.values.astype("datetime64[ns]")
    for code, df in val.items():
        if code not in codes_pos or df is None or df.empty:
            continue
        if normalize_valuation_columns is not None:
            try:
                df = normalize_valuation_columns(df)
            except Exception as e:  # noqa: BLE001
                log.error(f"[ic_vectorized] 操作失败: {e}", exc_info=True)
        if "date" not in df.columns:
            continue
        d = pd.to_datetime(df["date"], errors="coerce")
        m = d.notna()
        if not m.any():
            continue
        d = d[m]
        s = df[m]
        ci = codes_pos[code]
        d_ns = d.to_numpy().astype("datetime64[ns]")
        idx = np.searchsorted(d_ns, common_ns, side="right") - 1
        idx = np.clip(idx, 0, len(d_ns) - 1)
        has = common_ns >= d_ns.min()
        for nm, col in VAL_LEVEL_COLS.items():
            if col is None:
                pe = pd.to_numeric(s.get("pe_ttm"), errors="coerce").to_numpy()
                pb = pd.to_numeric(s.get("pb"), errors="coerce").to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(np.abs(pb) > 1e-9,
                                 pe / np.where(np.abs(pb) > 1e-9, pb, np.nan),
                                 np.nan)
            else:
                if col not in s.columns:
                    continue
                v = pd.to_numeric(s[col], errors="coerce").to_numpy()
            vals = v[idx]
            arrs[nm][:, ci] = np.where(has, vals, np.nan)
        # 分位类：先对齐+ffill 到公共日，再滚动窗口分位
        for nm, col in VAL_PCT_COLS.items():
            if col not in s.columns:
                continue
            v = pd.to_numeric(s[col], errors="coerce").to_numpy()
            aligned = pd.Series(np.nan, index=common)
            aligned[has] = v[idx[has]]
            aligned = aligned.ffill()
            arrs[nm][:, ci] = _rolling_pct(aligned.to_numpy(),
                                           PCT_WINDOW, min_count=60)
        del s
    alive = {nm: a for nm, a in arrs.items() if (~np.isnan(a)).sum() > 0}
    if spill_dir is not None:
        return _spill_panels(alive, common, codes, spill_dir, reused)
    return {nm: pd.DataFrame(a, index=common, columns=codes)
            for nm, a in alive.items()}


def build_alt_panels(common: pd.DatetimeIndex, codes: list[str],
                     spill_dir: Path | None = None) -> dict[str, object]:
    """另类数据面板（V10 数据榨干）：只收录真正的日序列数据。

    - inst_participation：机构参与度（日序列，全股票同值）——合法时间序列
    - 注意：esg_score / fund_hold_ratio 是单截面快照（非时间序列），
      严禁前填进 IC 面板（会造成虚假历史），只用于横截面分析（alt_factors 模块）。
    """
    from quant_system.ic_factors import alt_factors as af

    n_d, n_c = len(common), len(codes)
    arrs: dict[str, np.ndarray] = {}

    # 市场级日序列因子：全股票同值（合法时间序列）
    zlkp = af._load_zlkp()
    if len(zlkp):
        s = zlkp["val"].reindex(common).ffill()
        sm = s.rolling(5, min_periods=2).mean()
        arrs["inst_participation"] = np.tile(sm.to_numpy()[:, None], (1, n_c))

    if not arrs:
        return {}
    log.info(f"alt 面板构建完成: {len(arrs)} 因子 × {n_d} 观测 × {n_c} 股票")
    if spill_dir is not None:
        return _spill_panels(arrs, common, codes, spill_dir)
    return {nm: pd.DataFrame(a, index=common, columns=codes)
            for nm, a in arrs.items()}



def build_tech_panels(kl: dict[str, pd.DataFrame],
                      factor_names: list[str] | None = None,
                      sample_dates: pd.DatetimeIndex | None = None,
                      min_codes: int = 20,
                      spill_dir: Path | None = None,
                      common: pd.DatetimeIndex | None = None,
                      codes: list[str] | None = None,
                      reuse_dir: Path | str | None = None) -> dict[str, object]:
    """全市场向量化面板构建（支持落盘）。

    返回 {因子名: DataFrame(index=dates, columns=股票代码)}；
    spill_dir 给定则返回 {因子名: npy路径}。
    每只股票一次向量化计算完整序列 → 采样日取横截面。

    min_codes: 为兼容旧调用方保留（与 build_zoo_panels 对齐——zoo 无此参数，
    说明该参数本就冗余）：面板股票数由 _prepare_common 的公共交易日阈值
    （>=30 日 + 覆盖度回退）决定，此处不再另行过滤。
    """
    if factor_names is None:
        factor_names = [n for n in reg._REGISTRY
                        if n.startswith("tech_") and reg.get_factor(n).active]
    if common is None or codes is None:
        common, codes = _prepare_common(kl, sample_dates)
        if not codes:
            return {}
    return _build_panels(kl, factor_names, _series_all, common, codes,
                         spill_dir, "tech", reuse_dir)


def build_market_panels(common: pd.DatetimeIndex, codes: list[str],
                        spill_dir: Path | None = None) -> dict[str, object]:
    """市场级因子面板 (V12.3 因子扩充, 29 因子)。

    macro/flow/sentiment/industry/event/vol 类因子 —— 全股票同值日序列
    (市场级时间序列, 合法进 IC 面板)。月频宏观因子按"最近已发布值"对齐
    到观测日(reindex+ffill), 避免前视。
    """
    from quant_system.ic_factors.market_level_factors import MARKET_LEVEL_FACTORS
    from quant_system.ic_factors import market_level_factors as mlf_mod

    n_d, n_c = len(common), len(codes)
    arrs: dict[str, np.ndarray] = {}
    for name in MARKET_LEVEL_FACTORS:
        try:
            s = getattr(mlf_mod, name)({})
        except Exception as e:  # noqa: BLE001
            log.warning(f"[market_panels] {name} 计算失败: {e}")
            continue
        if s is None or not len(s):
            continue
        # 原始观测覆盖检查: 至少 12 个观测点(月频一年), 避免短数据被 ffill 伪装
        orig_cov = int(s.index.intersection(common).size)
        if orig_cov < 12:
            continue
        # 最近已发布值对齐观测日(月频 ffill, 日频 reindex)
        s = s.reindex(common, method="ffill")
        if s.notna().sum() < max(20, n_d * 0.2):
            continue  # 覆盖不足的因子跳过
        arrs[name] = np.tile(s.to_numpy(dtype=float)[:, None], (1, n_c))
    if not arrs:
        return {}
    log.info(f"市场级面板构建完成: {len(arrs)} 因子 × {n_d} 观测 × {n_c} 股票")
    if spill_dir is not None:
        return _spill_panels(arrs, common, codes, spill_dir)
    return {nm: pd.DataFrame(a, index=common, columns=codes)
            for nm, a in arrs.items()}


def build_forward_returns(kl: dict[str, pd.DataFrame],
                          horizon: int = 5) -> pd.DataFrame:
    """未来 horizon 日收益面板（index=date, columns=股票）。"""
    rets = {}
    for code, df in kl.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        fwd = df["close"].shift(-horizon) / df["close"] - 1
        rets[code] = fwd
    return pd.DataFrame(rets)


def compute_ic_fast(factor_wide: pd.DataFrame, ret_wide: pd.DataFrame,
                    min_n: int = 30) -> pd.Series:
    """向量化逐日 Spearman IC（全日期一次矩阵运算，替代逐日循环）。

    秩相关 = 秩上的 Pearson 相关。对面板逐行 rank 后，
    用逐行中心化+内积一次算全部日期的 IC。
    """
    dates = factor_wide.index.intersection(ret_wide.index)
    if len(dates) == 0:
        return pd.Series(dtype=float)
    F0 = factor_wide.loc[dates]
    R0 = ret_wide.loc[dates]
    # 关键：先求共同有效位置，再各自 rank（保证 rank 标度一致，
    # 与 compute_ic 的 f[m].rank() 语义相同）
    valid = F0.notna() & R0.notna() & np.isfinite(F0) & np.isfinite(R0)
    F = F0.where(valid).rank(axis=1).to_numpy(dtype=float)
    R = R0.where(valid).rank(axis=1).to_numpy(dtype=float)
    valid_np = valid.to_numpy()
    n_valid = valid_np.sum(axis=1)
    F = np.where(valid_np, F, np.nan)
    R = np.where(valid_np, R, np.nan)
    Fm = F - np.nanmean(F, axis=1, keepdims=True)
    Rm = R - np.nanmean(R, axis=1, keepdims=True)
    Fs = np.nansum(Fm * Fm, axis=1)
    Rs = np.nansum(Rm * Rm, axis=1)
    cov = np.nansum(Fm * Rm, axis=1)
    denom = np.sqrt(Fs * Rs)
    ic = np.where((n_valid >= min_n) & (denom > 1e-12),
                  cov / np.where(denom > 1e-12, denom, 1), np.nan)
    out = pd.Series(ic, index=dates).dropna()
    return out


def _load_delisted_codes() -> list[str]:
    """读退市股表（data_warehouse/delisted.parquet），返回 code 列表。

    表缺失/损坏 → 空列表（不中断回测）。退市股纳入样本池用于修复
    幸存者偏差；warehouse 里读不到历史 K 线的会在 _load_from_warehouse
    里自然跳过。
    """
    p = Path(__file__).resolve().parent.parent.parent / "data_warehouse" / "delisted.parquet"
    if not p.exists():
        return []
    try:
        df = pd.read_parquet(p)
        if "code" not in df.columns:
            return []
        codes = [str(c).strip() for c in df["code"].dropna()]
        return sorted({c for c in codes if c})
    except Exception as e:  # noqa: BLE001
        log.warning(f"退市股表读取失败，按无退市股处理: {e}")
        return []


def _load_from_warehouse(codes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    from quant_system.market_forecast._support.data.market_warehouse import MarketWarehouse
    wh = MarketWarehouse()
    if codes is None:
        # 全量：读全部已入库的 kline 文件
        codes = [p.name.replace(".parquet", "")
                 for p in sorted(wh._domain_dir("kline").glob("*.parquet"))]
    # 退市股纳入 IC 样本池（修复幸存者偏差）；历史数据读不到的跳过
    delisted = _load_delisted_codes()
    if delisted:
        codes = sorted(set(codes) | set(delisted))
        log.info(f"退市股表纳入样本池: +{len(delisted)} 只（共 {len(codes)} 候选）")
    kl = wh.load_warehouse("kline", codes)
    # 只保留数据足够的（>=300 交易日）
    out = {}
    for code, df in kl.items():
        if df is not None and len(df) >= 300:
            out[code] = df
    log.info(f"仓库 K线读取: {len(out)}/{len(codes)} 只（≥300交易日）")
    return out


def _load_from_loader(codes: list[str]) -> dict[str, pd.DataFrame]:
    from quant_system.ic_factors.data_loader_v7 import DataLoaderV7
    loader = DataLoaderV7()
    data = loader.load(["kline"], codes=codes, years=[2024, 2025, 2026])
    return data.get("kline") or {}


# --------------------------------------------------------------------------
# P1-1 中性化接入（审计_因子层.md P1-1）
# --------------------------------------------------------------------------
def _neutralize_panel(panel: pd.DataFrame | None,
                      industry_map: pd.Series | None,
                      market_cap: pd.DataFrame | None,
                      minimum_valid: int = 50) -> pd.DataFrame | None:
    """对单个因子面板做逐日横截面中性化（市值 + 行业），返回残差面板。

    panel:  index=date, columns=股票代码（原始因子值）。None/mkt_ 市场级面板
           （全股票同日同值）不做截面中性化，原样返回。
    低效：逐日循环；但仅在一次 run 的中性化对比中启用，可接受。
    """
    if panel is None or panel.empty:
        return panel
    from quant_system.ic_factors.neutralize import neutralize
    try:
        residual = neutralize(
            panel,
            industry_map=industry_map if industry_map is not None and not industry_map.empty else None,
            market_cap=market_cap if market_cap is not None and not market_cap.empty else None,
        )
        return residual
    except Exception as e:  # noqa: BLE001 - 中性化失败不阻断主流程
        log.warning(f"[P1-1] 因子 {getattr(panel, 'name', '?')} 中性化失败: {e}")
        return panel


def _build_neutralize_inputs(kl: dict[str, pd.DataFrame],
                             common_codes: list[str],
                             common: pd.DatetimeIndex | None) -> tuple[pd.Series, pd.DataFrame, dict]:
    """构建中性化输入（行业 map + 流通市值面板），并返回可用性诚实声明。"""
    industry_map = build_industry_map(common_codes)
    try:
        market_cap = build_market_cap(
            kl, common=common, codes=common_codes)
    except Exception as e:  # noqa: BLE001 - 市值构建失败回落为空面板
        log.warning(f"流通市值构建失败（行业中性化仍可用）: {e}")
        market_cap = pd.DataFrame()
    summary = neutralize_input_summary(industry_map, market_cap)
    # 诚实声明：框架接通状态写日志，绝不静默假装中性化生效
    note = []
    if not summary["industry_available"]:
        note.append("行业映射缺失")
    if not summary["market_cap_available"]:
        note.append("流通市值缺失")
    if note:
        log.warning(f"[P1-1] 中性化输入不完整：{'、'.join(note)} → 仅接通框架，对应中性化维度不生效")
    else:
        log.info("[P1-1] 中性化输入完整（行业 %d 只 + 市值 %d只×%d日）",
                 summary["industry_count"], summary["market_cap_stocks"],
                 summary["market_cap_days"])
    return industry_map, market_cap, summary


def run(codes: list[str] | None = None, source: str = "warehouse",
        forward: int | list[int] = 5, top: int = 25,
        sample_every: int = 1, spill: bool = False,
        spill_dir: str | None = None,
        reuse_dir: str | Path | None = None,
        neutralize: bool = False) -> pd.DataFrame:
    """向量化 IC 回测主流程。返回因子有效性报告。

    spill=True 时面板逐因子写 npy 落盘（默认目录 generated/ic_spill），
    内存峰值从 ~6.5GB 降到 <2GB，IC 阶段仅单面板驻留。

    neutralize=True 时（P1-1）：在 IC 计算前对每个横截面做市值/行业中性化，
    并在报告中输出"原始 IC vs 中性化 IC"对比（ic_raw / ic_neut 两列）。
    输入数据（行业 map / 流通市值）由 neutralize_inputs 从数据层诚实构建；
    缺失维度会明确降级并在日志/报告中声明，绝不静默假装中性化生效。
    """
    from quant_system.ic_factors import registry as reg
    reg.autodiscover()
    reg.import_from_zoo()
    forwards = [forward] if isinstance(forward, int) else forward

    kl = (_load_from_warehouse(codes) if source == "warehouse"
          else _load_from_loader(codes))
    if not kl:
        log.warning("无 K 线数据")
        return pd.DataFrame()
    codes = sorted(kl.keys())
    log.info(f"样本池 {len(codes)} 只, 前瞻 {forwards}, 源 {source}")

    # 采样日：sample_every>1 时每 N 个交易日取一个截面，降内存/提速
    sample_dates = None
    if sample_every and sample_every > 1:
        all_dates = sorted({d for df in kl.values() for d in df["date"]})
        sample_dates = pd.DatetimeIndex(all_dates[:: sample_every])
        log.info(f"采样日: 每 {sample_every} 日取一截面, 共 {len(sample_dates)} 个观测日")

    # 第一遍只读日期索引求公共交易日（不做因子计算）
    common, common_codes = _prepare_common(kl, sample_dates)
    if not common_codes:
        log.warning("公共交易日不足，放弃")
        return pd.DataFrame()

    sd: Path | None = Path(spill_dir) if spill else None
    tech = build_tech_panels(kl, sample_dates=sample_dates, spill_dir=sd,
                             common=common, codes=common_codes, reuse_dir=reuse_dir)
    zoo = build_zoo_panels(kl, sample_dates=sample_dates, spill_dir=sd,
                           common=common, codes=common_codes, reuse_dir=reuse_dir)
    fwd_cache = {f: build_forward_returns(kl, f) for f in forwards}

    # 财务/估值域（fa_*/val_* 因子面板，仅 warehouse 源）
    fin: dict[str, pd.DataFrame] = {}
    val: dict[str, pd.DataFrame] = {}
    if source == "warehouse":
        from quant_system.market_forecast._support.data.market_warehouse import MarketWarehouse
        wh = MarketWarehouse()
        fin = wh.load_warehouse("financial", common_codes)
        val = wh.load_warehouse("valuation", common_codes)
        log.info(f"财务域 {len(fin)} 只 / 估值域 {len(val)} 只")
    fa_p = build_financial_panels(fin, common, common_codes, sd, reuse_dir)
    val_p = build_valuation_panels(val, common, common_codes, sd, reuse_dir)
    # V10 数据榨干：另类数据面板（ESG/机构参与度/基金持仓/破净）
    alt_p = build_alt_panels(common, common_codes, sd)
    # V12.3 因子扩充：市场级面板（macro/flow/sentiment/industry/event/vol 29 因子,
    # 全股票同值日序列, 数据源 data_warehouse/macro+market+events+oneoff）
    mkt_p = build_market_panels(common, common_codes, sd)

    # P1-1: 中性化输入构建（行业 map + 流通市值）。须在 del kl 前完成（市值来自 kline）。
    neutralize_inputs = None
    if neutralize:
        neutralize_inputs = _build_neutralize_inputs(kl, common_codes, common)

    del kl, fin, val  # 输入数据已用完，释放
    gc.collect()

    panels: dict[str, object] = {}
    panels.update(tech)
    panels.update(zoo)
    panels.update(fa_p)
    panels.update(val_p)
    panels.update(alt_p)
    panels.update(mkt_p)
    if not panels:
        log.error("无因子产出面板")
        return pd.DataFrame()

    rows = []
    min_n = max(8, int(len(common_codes) * 0.3))  # 全市场横截面：30% 覆盖即可
    total = len(panels)
    for i, name in enumerate(panels.keys()):
        if i % 10 == 0:
            log.info(f"IC计算进度: {i}/{total} 因子 ({name})")
        panel = (_load_spilled(name, sd, common, common_codes)
                 if sd is not None else panels[name])  # type: ignore[assignment]
        meta = reg.get_factor(name)
        coverage = panel.notna().mean().mean()
        ic_means, icirs, winrates, ndays = [], [], [], []
        ic_means_neut, icirs_neut, winrates_neut = [], [], []
        is_market_factor = str(name).startswith("mkt_")
        # P1-1: 中性化面板（非市场级因子）。市场级全股票同值不做截面中性化。
        neut_panel = None
        if neutralize and neutralize_inputs is not None and not is_market_factor:
            neut_panel = _neutralize_panel(
                panel,
                neutralize_inputs[0],  # industry_map
                neutralize_inputs[1],  # market_cap
            )
        do_neutralize = neut_panel is not None and not neut_panel.empty
        for f in forwards:
            fwd = fwd_cache[f].reindex(panel.index)
            if is_market_factor:
                # V12.3 因子扩充: 市场级因子(全股票同值)横截面 IC 无意义,
                # 改用时序 IC——因子值(时序) vs 全市场等权平均收益(时序) 的
                # 60日滚动相关, 评估其"市场择时"预测力。
                fwd_mean = fwd.mean(axis=1)
                fac = panel.iloc[:, 0]
                try:
                    timing_ic = fac.rolling(60, min_periods=30).corr(fwd_mean)
                    st = ic_stats(timing_ic.dropna())
                except Exception:  # noqa: BLE001
                    st = {"ic_mean": np.nan, "icir": np.nan,
                          "ic_winrate": np.nan, "n": 0}
            else:
                ic = compute_ic_fast(panel, fwd, min_n=min_n)
                st = ic_stats(ic)
                if do_neutralize:
                    ic_n = compute_ic_fast(neut_panel, fwd, min_n=min_n)
                    st_n = ic_stats(ic_n)
                    ic_means_neut.append(st_n.get("ic_mean", np.nan))
                    icirs_neut.append(st_n.get("icir", np.nan))
                    winrates_neut.append(st_n.get("ic_winrate", np.nan))
            ic_means.append(st.get("ic_mean", np.nan))
            icirs.append(st.get("icir", np.nan))
            winrates.append(st.get("ic_winrate", np.nan))
            ndays.append(st.get("n", 0))
        icm = float(np.nanmean(ic_means)) if np.any(np.isfinite(ic_means)) else np.nan
        icir = float(np.nanmean(icirs)) if np.any(np.isfinite(icirs)) else np.nan
        win = float(np.nanmean(winrates)) if np.any(np.isfinite(winrates)) else np.nan
        # P1-1: 中性化 IC 汇总（仅非市场级且在 neutralize 模式下）
        if do_neutralize and ic_means_neut:
            icm_neut = (float(np.nanmean(ic_means_neut))
                        if np.any(np.isfinite(ic_means_neut)) else np.nan)
            icir_neut = (float(np.nanmean(icirs_neut))
                         if np.any(np.isfinite(icirs_neut)) else np.nan)
            win_neut = (float(np.nanmean(winrates_neut))
                        if np.any(np.isfinite(winrates_neut)) else np.nan)
        else:
            icm_neut = np.nan
            icir_neut = np.nan
            win_neut = np.nan
        if pd.notna(icm) and pd.notna(icir):
            if abs(icm) > 0.03 and abs(icir) > 0.3:
                grade = "A"
            elif abs(icm) > 0.02:
                grade = "B"
            elif abs(icm) > 0.01:
                grade = "C"
            else:
                grade = "D"
        else:
            grade = "N/A"
        rows.append({
            "factor": name, "category": meta.category if meta else "",
            "ic_mean": round(icm, 4) if pd.notna(icm) else np.nan,
            "icir": round(icir, 3) if pd.notna(icir) else np.nan,
            "winrate": round(win, 3) if pd.notna(win) else np.nan,
            "n_days": int(max(ndays)) if ndays else 0,
            "coverage": round(float(coverage), 3),
            "grade": grade,
            "direction": meta.direction if meta else 1,
            # P1-1: 中性化 IC 对比（neutralize=True 且非市场级因子时有值）
            "ic_neut": round(icm_neut, 4) if pd.notna(icm_neut) else np.nan,
            "icir_neut": round(icir_neut, 3) if pd.notna(icir_neut) else np.nan,
            "winrate_neut": round(win_neut, 3) if pd.notna(win_neut) else np.nan,
        })
    report = pd.DataFrame(rows)
    if report.empty:
        log.error("无因子产出 IC 结果（面板为空或全部 IC 失败）")
        return report
    # V10 审计 M3 修复：先保留全量结果（CSV 用），再按 top 截断（报告用）
    # 2026-08-14 修复: 旧版把 full_report 塞进 report.attrs → DataFrame 自引用循环,
    #   pandas __finalize__ deepcopy(attrs) 无限递归 → to_string 崩溃 (RecursionError)。
    #   改为模块级 _FULL_REPORT 全局传递，attrs 只放标量元信息。
    global _FULL_REPORT
    full_report = report.sort_values("ic_mean", key=abs, ascending=False)
    _FULL_REPORT = full_report
    report = full_report.head(top) if top > 0 else full_report
    # 附加元信息（写报告头用）
    report.attrs["sample_count"] = len(common_codes)
    report.attrs["spill"] = bool(sd)
    # P1-1: 中性化可用性诚实声明（读入报告头/CSV）
    if neutralize_inputs is not None:
        report.attrs["neutralize"] = True
        report.attrs["neutralize_industry"] = bool(neutralize_inputs[0].shape[0] > 0)
        report.attrs["neutralize_market_cap"] = bool(not neutralize_inputs[1].empty)
    try:
        import resource
        report.attrs["peak_rss_kb"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
    except Exception as e:
        log.error(f"[ic_vectorized] 操作失败: {e}", exc_info=True)
    log.info(f"IC 计算完成: {len(report)} 因子（全量 {len(full_report)}）, 峰值RSS "
             f"{report.attrs.get('peak_rss_kb', 0)/1024:.0f} MB")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="向量化因子 IC 回测（全市场仓库）")
    ap.add_argument("--source", default="warehouse",
                    choices=["warehouse", "loader"], help="数据源")
    ap.add_argument("--n", type=int, default=800, help="样本股票数(默认800，0=全部)")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="每 N 个交易日取一个截面(1=全采样，2/3/5=降内存提速)")
    ap.add_argument("--codes", default="", help="逗号分隔指定股票")
    ap.add_argument("--forward", default="5,10,20", help="前瞻天数")
    ap.add_argument("--top", type=int, default=0,
                    help="只输出|IC|最大的前N个因子(0=全部)")
    ap.add_argument("--spill", action="store_true",
                    help="面板逐因子写npy落盘, 峰值内存大幅下降(全量推荐)")
    ap.add_argument("--no-spill", action="store_true",
                    help="强制不落盘(默认小样本不落盘, 全量自动落盘)")
    ap.add_argument("--spill-dir", default="generated/ic_spill",
                    help="面板npy落盘目录(默认自动清理)")
    ap.add_argument("--keep-spill", action="store_true",
                    help="跑完保留落盘文件(调试用)")
    ap.add_argument("--reuse-spill", default="",
                    help="复用已有面板npy目录(断点续跑: 只重算缺失因子, V10审计M5)")
    ap.add_argument("--neutralize", action="store_true",
                    help="P1-1: IC计算前做市值/行业中性化，输出原始IC vs 中性化IC 对比")
    ap.add_argument("--out", default="generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.md")
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    if args.n and codes is None:
        from quant_system.market_forecast._support.data.market_warehouse import MarketWarehouse as _MW
        _wh = _MW()
        all_codes = [p.name.replace(".parquet", "") for p in sorted(_wh._domain_dir("kline").glob("*.parquet"))]
        # V10 审计 M3 修复：随机抽样（原取前 N 个=老股偏置），seed 固定可复现
        import random as _rng
        _rng.seed(42)
        codes = _rng.sample(all_codes, min(args.n, len(all_codes)))
    forwards = [int(x) for x in args.forward.split(",") if x.strip()]
    # 全量(0=全部股票)默认落盘，避免 6GB+ 内存峰值；可用 --no-spill 关闭
    use_spill = (args.spill or (args.n == 0 and not args.no_spill))
    if use_spill:
        log.info(f"落盘模式: 面板写入 {args.spill_dir} (npy)，完成后自动清理")
    t0 = time.time()
    report = run(codes=codes, source=args.source, forward=forwards, top=args.top,
                 sample_every=args.sample_every, spill=use_spill,
                 spill_dir=args.spill_dir,
                 reuse_dir=args.reuse_spill or None,
                 neutralize=args.neutralize)

    if report.empty:
        print("无因子可回测")
        return
    runtime_s = time.time() - t0
    n_sample = report.attrs.get("sample_count", len(codes or []))
    peak_mb = report.attrs.get("peak_rss_kb", 0) / 1024
    print("\n=== 向量化因子 IC 回测报告（前瞻 %s 日，全日期观测）===" % args.forward)
    print(report.to_string(index=False))

    lines = ["# 因子 IC 回测报告（向量化·全市场仓库）",
             "",
             f"- 样本池: {n_sample} 只 | 前瞻: {args.forward} 日 | 源: {args.source}",
             f"- 观测: 全日期（每交易日横截面）| 评级: A/B/C/D",
             f"- 耗时: {runtime_s/60:.1f} 分钟 | 峰值内存: {peak_mb:.0f} MB | 落盘: {'是' if use_spill else '否'}",
             ""]
    # P1-1: 中性化诚实声明
    if args.neutralize:
        ind_ok = report.attrs.get("neutralize_industry", False)
        mc_ok = report.attrs.get("neutralize_market_cap", False)
        parts = ["行业" if ind_ok else "行业✗", "流通市值" if mc_ok else "流通市值✗"]
        lines.append(f"- 中性化(P1-1): 启用 | {(' + '.join(parts))}")
        if not (ind_ok and mc_ok):
            lines.append("- > 诚实声明：中性化输入不完整（✗项缺失），对应维度未生效，仅接通框架，"
                         "未对缺失维度静默中性化。")
        lines.append("")
    has_neut = args.neutralize
    lines.append(
        f"| 因子 | 大类 | IC均值 | ICIR | 胜率{(' | 中性化IC | 中性化ICIR' if has_neut else '')} | 天数 | 覆盖 | 评级 |")
    n_cols = lines[-1].count("|")
    lines.append("|---" * (n_cols - 1) + "|")
    for _, r in report.iterrows():
        neut = (f" | {r['ic_neut']} | {r['icir_neut']}" if has_neut else "")
        lines.append(
            f"| {r['factor']} | {r['category']} | {r['ic_mean']} | {r['icir']} "
            f"| {r['winrate']}{neut} | {r['n_days']} | {r['coverage']} | {r['grade']} |")
    import pathlib
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {p}")

    # 全量因子结果 CSV 落盘（V10 审计 M3 修复：用 full_report 不受 --top 截断）
    csv_out = p.with_suffix(".csv")
    try:
        full = _FULL_REPORT if _FULL_REPORT is not None else report
        full.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"全量因子结果已写入 {csv_out} ({len(full)} 因子)")
    except Exception as e:  # noqa: BLE001
        print(f"CSV 落盘失败(不影响报告): {e}")

    if use_spill and not args.keep_spill:
        shutil.rmtree(args.spill_dir, ignore_errors=True)
        log.info(f"已清理落盘目录 {args.spill_dir}")


if __name__ == "__main__":
    main()
