"""Visualization utilities for quant research and backtesting.

The module intentionally keeps dependencies light: matplotlib, numpy, and pandas.
Every public plotting function returns a matplotlib Figure, saves only when a
``save_path`` is supplied, and never calls ``plt.show()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised by environments where matplotlib is absent
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    font_manager = None  # type: ignore[assignment]
    Figure = Any  # type: ignore[misc,assignment]
    FuncFormatter = None  # type: ignore[assignment]
    PercentFormatter = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


PRIMARY = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
GRAY = "#7f7f7f"


def _warn_matplotlib() -> None:
    """Print a consistent warning when matplotlib is not available."""
    print("Warning: matplotlib is not available; visualization function returns None.")


def _require_matplotlib() -> bool:
    """Return whether plotting is available, printing a warning otherwise."""
    if not MATPLOTLIB_AVAILABLE:
        _warn_matplotlib()
        return False
    return True


def get_default_style() -> dict[str, Any]:
    """Return the shared color and style palette used by this module."""
    return {
        "primary": PRIMARY,
        "red": RED,
        "green": GREEN,
        "orange": ORANGE,
        "purple": PURPLE,
        "gray": GRAY,
        "grid_color": "#d9d9d9",
        "background": "#ffffff",
        "text": "#222222",
        "line_width": 2.0,
        "alpha": 0.25,
        "colors": [PRIMARY, ORANGE, GREEN, RED, PURPLE, "#8c564b", "#e377c2", GRAY],
    }


def set_chinese_font() -> str | None:
    """Detect and set an available Chinese font for Windows, Ubuntu, or macOS.

    Returns:
        The selected font family name, or ``None`` when matplotlib is unavailable
        or no known Chinese font can be found.
    """
    if not _require_matplotlib():
        return None

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "PingFang SC",
        "Heiti SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
        "Source Han Sans SC",
        "Source Han Serif SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _apply_style() -> dict[str, Any]:
    """Apply common matplotlib rcParams and return the palette."""
    style = get_default_style()
    set_chinese_font()
    plt.rcParams.update(
        {
            "axes.edgecolor": "#bfbfbf",
            "axes.labelcolor": style["text"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.facecolor": style["background"],
            "axes.facecolor": style["background"],
            "grid.color": style["grid_color"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    return style


def _save_figure(fig: Figure, save_path: str | Path | None) -> None:
    """Save a figure when requested."""
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)


def _to_series(data: Any, name: str = "value") -> pd.Series:
    """Convert common inputs to a clean pandas Series."""
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            series = pd.Series(dtype=float, name=name)
        else:
            series = data.iloc[:, 0].copy()
    else:
        series = pd.Series(data, name=name)
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.name is None:
        series.name = name
    return series


def _extract_pos_proba(y_pred_proba: Any, n_expected: int) -> pd.Series:
    """从概率输入提取正类概率列, 并校验与 y_true 长度一致。

    # P2-Q25-fix(M294): sklearn `predict_proba` 输出 (n,2) 时直接 `reshape(-1)`
    会展开为 2n 与 y_true(n) 错位拼接(pandas 截短后前 n 行实为负类列), 图形
    语义错乱。2D 输入统一取正类列 `proba[:, 1]`; 长度不一致时显式报错而非静默错位。
    """
    arr = np.asarray(y_pred_proba)
    if arr.ndim == 2:
        prob = arr[:, 1] if arr.shape[1] >= 2 else arr[:, 0]
    else:
        prob = arr.reshape(-1)
    series = pd.to_numeric(pd.Series(prob), errors="coerce")
    if len(series) != n_expected:
        raise ValueError(
            f"y_pred_proba 长度({len(series)}) 与 y_true 长度({n_expected})不一致; "
            "2D 输入请传入 sklearn predict_proba 的 (n,2) 输出或 1D 正类概率"
        )
    return series.reset_index(drop=True)


def _to_dataframe(data: Any) -> pd.DataFrame:
    """Convert common inputs to a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data)


def _normalize_curve(series: pd.Series) -> pd.Series:
    """Normalize a price/equity series to start at 1 when appropriate."""
    series = series.astype(float).dropna()
    if series.empty:
        return series
    first = series.iloc[0]
    if first != 0 and (first > 2 or series.max() > 3):
        return series / first
    return series


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute drawdown from an equity curve."""
    equity = _normalize_curve(equity)
    if equity.empty:
        return equity
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _percent_axis(ax: Any, decimals: int = 0) -> None:
    """Apply percentage formatting to a y-axis."""
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=decimals))


def _grid_legend(ax: Any) -> None:
    """Apply common grid and legend behavior."""
    ax.grid(True, axis="y", alpha=0.55)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Compute a normal PDF without scipy."""
    if std <= 0 or not np.isfinite(std):
        return np.zeros_like(x)
    coeff = 1.0 / (std * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)


def _max_streak(values: pd.Series, positive: bool) -> tuple[int, int, int]:
    """Return maximum streak length and its start/end positional indexes."""
    best_len = current_len = 0
    best_start = best_end = current_start = -1
    for i, value in enumerate(values):
        ok = value > 0 if positive else value < 0
        if ok:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = i
        else:
            current_len = 0
    return best_len, best_start, best_end


def plot_equity_curve(
    equity_curve: Any,
    benchmark_col: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure | None:
    """Plot net value curve with optional benchmark and max drawdown shading.

    Args:
        equity_curve: Series or DataFrame containing portfolio equity values.
        benchmark_col: Optional benchmark column name when ``equity_curve`` is a DataFrame.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure, or ``None`` when matplotlib is unavailable.
    """
    if not _require_matplotlib():
        return None
    style = _apply_style()
    df = _to_dataframe(equity_curve)
    fig, ax = plt.subplots(figsize=figsize)

    if df.empty:
        ax.set_title("净值曲线")
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        _save_figure(fig, save_path)
        return fig

    bench = None
    if benchmark_col and benchmark_col in df.columns:
        bench = _to_series(df[benchmark_col], "benchmark")
        portfolio_cols = [c for c in df.columns if c != benchmark_col]
        equity = _to_series(df[portfolio_cols[0]] if portfolio_cols else df.iloc[:, 0], "portfolio")
    else:
        equity = _to_series(df.iloc[:, 0], "portfolio")

    equity = _normalize_curve(equity)
    ax.plot(equity.index, equity - 1.0, color=style["primary"], lw=2.2, label="组合")
    if bench is not None and not bench.empty:
        bench = _normalize_curve(bench)
        ax.plot(bench.index, bench - 1.0, color=style["gray"], lw=1.8, label="基准")

    drawdown = _compute_drawdown(equity)
    if not drawdown.empty:
        trough = drawdown.idxmin()
        peak_slice = equity.loc[:trough]
        if not peak_slice.empty:
            peak = peak_slice.idxmax()
            ax.axvspan(peak, trough, color=style["red"], alpha=0.14, label="最大回撤区间")
            ax.annotate(
                f"最大回撤 {drawdown.min():.2%}",
                xy=(trough, equity.loc[trough] - 1.0),
                xytext=(10, -25),
                textcoords="offset points",
                color=style["red"],
                arrowprops={"arrowstyle": "->", "color": style["red"], "lw": 1.0},
            )

    ax.set_title("净值曲线")
    ax.set_ylabel("累计收益")
    _percent_axis(ax, 0)
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_drawdown(
    drawdown_series: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 3),
) -> Figure | None:
    """Plot a drawdown curve with red fill and max drawdown annotation."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    dd = _to_series(drawdown_series, "drawdown")
    # P2-Q25-fix(L305): 出现正值时统一转负向回撤(-abs), 避免正负值并存于同一
    # 回撤图的语义混淆(原逻辑在 max>0 且 mean<=0 时保持混合符号原样绘制)。
    if not dd.empty and dd.max() > 0:
        dd = -dd.abs()

    fig, ax = plt.subplots(figsize=figsize)
    if dd.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.fill_between(dd.index, dd.values, 0, color=style["red"], alpha=0.28)
        ax.plot(dd.index, dd.values, color=style["red"], lw=1.5)
        min_idx = dd.idxmin()
        min_val = dd.min()
        ax.scatter([min_idx], [min_val], color=style["red"], zorder=3)
        ax.annotate(
            f"最大回撤 {min_val:.2%}",
            xy=(min_idx, min_val),
            xytext=(10, 18),
            textcoords="offset points",
            color=style["red"],
            arrowprops={"arrowstyle": "->", "color": style["red"], "lw": 1.0},
        )
    ax.set_title("回撤曲线")
    ax.set_ylabel("回撤")
    _percent_axis(ax, 0)
    ax.grid(True, axis="y", alpha=0.55)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_monthly_returns(
    returns: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Figure | None:
    """Plot a monthly returns heatmap with annual returns on the right."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    ret = _to_series(returns, "returns")
    fig, ax = plt.subplots(figsize=figsize)

    if ret.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("月度收益热力图")
        _save_figure(fig, save_path)
        return fig

    if not isinstance(ret.index, pd.DatetimeIndex):
        ret.index = pd.to_datetime(ret.index, errors="coerce")
        ret = ret[ret.index.notna()]
    monthly = (1.0 + ret).resample("ME").prod() - 1.0
    years = sorted(monthly.index.year.unique())
    months = list(range(1, 13))
    heat = pd.DataFrame(index=years, columns=months, dtype=float)
    for idx, value in monthly.items():
        heat.loc[idx.year, idx.month] = value
    annual = (1.0 + ret).resample("YE").prod() - 1.0

    data = heat.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.05
    limit = max(float(limit), 0.01)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ret_rg", [GREEN, "#ffffff", RED])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"])
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_title("月度收益热力图")

    for i, year in enumerate(years):
        for j, month in enumerate(months):
            value = heat.loc[year, month]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8)
        ann_value = annual.loc[annual.index.year == year]
        ann_text = f"{ann_value.iloc[0]:.1%}" if not ann_value.empty else ""
        ax.text(12.35, i, ann_text, va="center", ha="left", fontsize=9, color=style["text"])
    ax.text(12.35, -0.75, "年度", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.5, 13.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_daily_returns_distribution(
    returns: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> Figure | None:
    """Plot daily return histogram with a normal-fit curve and moments."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    ret = _to_series(returns, "returns")
    fig, ax = plt.subplots(figsize=figsize)

    if ret.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        mean = float(ret.mean())
        std = float(ret.std(ddof=1))
        skew = float(ret.skew()) if len(ret) > 2 else np.nan
        kurt = float(ret.kurt()) if len(ret) > 3 else np.nan
        ax.hist(ret.values, bins=40, density=True, alpha=0.55, color=style["primary"], label="日收益")
        xs = np.linspace(float(ret.min()), float(ret.max()), 300)
        ax.plot(xs, _normal_pdf(xs, mean, std), color=style["red"], lw=2, label="正态拟合")
        ax.axvline(mean, color=style["gray"], ls="--", lw=1, label="均值")
        ax.text(
            0.98,
            0.95,
            f"偏度: {skew:.2f}\n峰度: {kurt:.2f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd"},
        )
    ax.set_title("日收益分布")
    ax.set_xlabel("日收益")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_backtest_summary(result: Any, save_dir: str | Path | None = None) -> dict[str, Figure | None]:
    """Create the standard four backtest summary figures.

    ``result`` may be a dict or an object with attributes. Recognized fields are
    ``equity_curve``, ``benchmark_col``, ``drawdown`` or ``drawdown_series``, and
    ``returns``.
    """
    if not _require_matplotlib():
        return {"equity_curve": None, "drawdown": None, "monthly_returns": None, "daily_distribution": None}

    def pick(*names: str) -> Any:
        for name in names:
            if isinstance(result, dict) and name in result:
                return result[name]
            if hasattr(result, name):
                return getattr(result, name)
        return None

    save_root = Path(save_dir) if save_dir is not None else None
    equity = pick("equity_curve", "equity", "portfolio_value", "nav")
    returns = pick("returns", "daily_returns")
    drawdown = pick("drawdown", "drawdown_series")
    benchmark_col = pick("benchmark_col")
    if drawdown is None and equity is not None:
        drawdown = _compute_drawdown(_to_series(_to_dataframe(equity).iloc[:, 0]))
    if returns is None and equity is not None:
        returns = _to_series(_to_dataframe(equity).iloc[:, 0]).pct_change().dropna()

    figures = {
        "equity_curve": plot_equity_curve(equity, benchmark_col=benchmark_col, save_path=save_root / "equity_curve.png" if save_root else None),
        "drawdown": plot_drawdown(drawdown, save_path=save_root / "drawdown.png" if save_root else None),
        "monthly_returns": plot_monthly_returns(returns, save_path=save_root / "monthly_returns.png" if save_root else None),
        "daily_distribution": plot_daily_returns_distribution(returns, save_path=save_root / "daily_returns_distribution.png" if save_root else None),
    }
    return figures


def plot_factor_ic_decay(
    ic_values: Any,
    periods: Any = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> Figure | None:
    """Plot information coefficient decay across forward-return periods."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    values = np.asarray(ic_values, dtype=float).reshape(-1)
    x = np.asarray(periods if periods is not None else np.arange(1, len(values) + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, values, marker="o", color=style["primary"], lw=2)
    ax.axhline(0, color=style["gray"], lw=1)
    ax.set_title("因子IC衰减")
    ax.set_xlabel("持有期")
    ax.set_ylabel("IC")
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_factor_long_short(
    portfolio_values: Any,
    benchmark: Any = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> Figure | None:
    """Plot long-short factor portfolio performance against an optional benchmark."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    pv = _normalize_curve(_to_series(portfolio_values, "long_short"))
    fig, ax = plt.subplots(figsize=figsize)
    if not pv.empty:
        ax.plot(pv.index, pv - 1.0, color=style["primary"], lw=2, label="多空组合")
    if benchmark is not None:
        bm = _normalize_curve(_to_series(benchmark, "benchmark"))
        if not bm.empty:
            ax.plot(bm.index, bm - 1.0, color=style["gray"], lw=1.8, label="基准")
    if pv.empty and benchmark is None:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("因子多空组合")
    ax.set_ylabel("累计收益")
    _percent_axis(ax, 0)
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_factor_ic_rolling(
    ic_series: Any,
    window: int = 60,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> Figure | None:
    """Plot raw and rolling mean IC series."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    ic = _to_series(ic_series, "ic")
    rolling = ic.rolling(window=max(int(window), 1), min_periods=max(min(int(window), 5), 1)).mean()
    fig, ax = plt.subplots(figsize=figsize)
    if ic.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(ic.index, ic.values, color=style["gray"], alpha=0.35, lw=1, label="IC")
        ax.plot(rolling.index, rolling.values, color=style["primary"], lw=2, label=f"{window}期滚动IC")
        ax.axhline(0, color=style["red"], lw=1)
    ax.set_title("因子IC滚动均值")
    ax.set_ylabel("IC")
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_factor_correlation_matrix(
    corr_df: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> Figure | None:
    """Plot a factor correlation matrix heatmap."""
    if not _require_matplotlib():
        return None
    _apply_style()
    corr = _to_dataframe(corr_df).astype(float)
    fig, ax = plt.subplots(figsize=figsize)
    if corr.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_yticklabels(corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("相关系数")
    ax.set_title("因子相关矩阵")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_factor_group_returns(
    group_returns: Any,
    n_groups: int = 5,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> Figure | None:
    """Plot cumulative returns by factor quantile/group."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    df = _to_dataframe(group_returns).apply(pd.to_numeric, errors="coerce")
    if df.shape[1] > n_groups:
        df = df.iloc[:, :n_groups]
    cumulative = (1.0 + df.fillna(0.0)).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=figsize)
    if cumulative.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, cumulative.shape[1]))
        for color, col in zip(colors, cumulative.columns):
            ax.plot(cumulative.index, cumulative[col], lw=1.8, color=color, label=str(col))
    ax.set_title("因子分组累计收益")
    ax.set_ylabel("累计收益")
    _percent_axis(ax, 0)
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_portfolio_weights(
    weights: Any,
    top_n: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Figure | None:
    """Plot top portfolio weights as a horizontal bar chart."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    w = _to_series(weights, "weight").sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
    w = w.sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    if w.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = [style["red"] if value >= 0 else style["green"] for value in w]
        ax.barh([str(i) for i in w.index], w.values, color=colors, alpha=0.82)
        ax.axvline(0, color=style["gray"], lw=1)
    ax.set_title(f"组合权重 Top {top_n}")
    ax.set_xlabel("权重")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, axis="x", alpha=0.55)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_weight_evolution(
    weight_history: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure | None:
    """Plot stacked portfolio weight evolution through time."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    df = _to_dataframe(weight_history).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.stackplot(df.index, [df[col].values for col in df.columns], labels=[str(c) for c in df.columns], alpha=0.85)
    ax.set_title("组合权重演化")
    ax.set_ylabel("权重")
    _percent_axis(ax, 0)
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_efficient_frontier(
    returns: Any,
    portfolios: Any = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Figure | None:
    """Plot an efficient frontier from provided portfolio points or simulations.

    Args:
        returns: Asset returns DataFrame used to simulate portfolios when
            ``portfolios`` is not supplied.
        portfolios: Optional DataFrame/dict with return, volatility, and Sharpe
            columns. Common column aliases are supported.
    """
    if not _require_matplotlib():
        return None
    style = _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    if portfolios is not None:
        pf = _to_dataframe(portfolios)
        lower = {str(c).lower(): c for c in pf.columns}
        ret_col = lower.get("return") or lower.get("returns") or lower.get("ret") or lower.get("收益")
        vol_col = lower.get("volatility") or lower.get("vol") or lower.get("risk") or lower.get("波动率")
        sharpe_col = lower.get("sharpe") or lower.get("sharpe_ratio") or lower.get("夏普")
        if ret_col is not None and vol_col is not None:
            x = pd.to_numeric(pf[vol_col], errors="coerce")
            y = pd.to_numeric(pf[ret_col], errors="coerce")
            c = pd.to_numeric(pf[sharpe_col], errors="coerce") if sharpe_col is not None else y / x.replace(0, np.nan)
            sc = ax.scatter(x, y, c=c, cmap="viridis", s=22, alpha=0.75)
            fig.colorbar(sc, ax=ax, label="夏普比率")
    else:
        ret_df = _to_dataframe(returns).apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if not ret_df.empty and ret_df.shape[1] >= 2:
            rng = np.random.default_rng(42)
            mean = ret_df.mean().values * 252.0
            cov = ret_df.cov().values * 252.0
            n_assets = ret_df.shape[1]
            weights = rng.dirichlet(np.ones(n_assets), size=2000)
            exp_ret = weights @ mean
            vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
            sharpe = np.divide(exp_ret, vols, out=np.zeros_like(exp_ret), where=vols > 0)
            sc = ax.scatter(vols, exp_ret, c=sharpe, cmap="viridis", s=16, alpha=0.55)
            fig.colorbar(sc, ax=ax, label="夏普比率")
    if not ax.collections:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("有效前沿")
    ax.set_xlabel("年化波动率")
    ax.set_ylabel("年化收益")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.55)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_market_regime(
    regime_history: Any,
    regime_colors: dict[Any, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 4),
) -> Figure | None:
    """Plot market regime history as colored timeline bands."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    regimes = _to_series(regime_history, "regime")
    default_colors = {"bull": RED, "bear": GREEN, "volatile": ORANGE, "sideways": GRAY, "牛市": RED, "熊市": GREEN, "震荡": GRAY}
    if regime_colors:
        default_colors.update(regime_colors)
    fig, ax = plt.subplots(figsize=figsize)
    if regimes.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        unique = list(pd.unique(regimes))
        fallback = style["colors"]
        color_map = {regime: default_colors.get(regime, fallback[i % len(fallback)]) for i, regime in enumerate(unique)}
        for i in range(len(regimes)):
            start = regimes.index[i]
            end = regimes.index[i + 1] if i + 1 < len(regimes) else start
            if start == end and i > 0:
                start = regimes.index[i - 1]
            ax.axvspan(start, end, color=color_map[regimes.iloc[i]], alpha=0.45)
        handles = [matplotlib.patches.Patch(color=color_map[r], alpha=0.45, label=str(r)) for r in unique]
        ax.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6), bbox_to_anchor=(0.5, 1.18))
        ax.set_xlim(regimes.index.min(), regimes.index.max())
        ax.set_yticks([])
    ax.set_title("市场状态切换")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_regime_radar(
    regime_scores: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (6, 6),
) -> Figure | None:
    """Plot regime scores as a radar chart."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    scores = _to_series(regime_scores, "score")
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    if scores.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [str(i) for i in scores.index]
        values = scores.astype(float).values
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
        values = np.concatenate([values, values[:1]])
        angles = np.concatenate([angles, angles[:1]])
        ax.plot(angles, values, color=style["primary"], lw=2)
        ax.fill(angles, values, color=style["primary"], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(1.0, float(np.nanmax(values))))
    ax.set_title("市场状态雷达图", pad=18)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_prediction_distribution(
    y_true: Any,
    y_pred_proba: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> Figure | None:
    """Plot predicted probability distribution split by true class."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    true = pd.Series(y_true).reset_index(drop=True)
    prob = _extract_pos_proba(y_pred_proba, len(true))  # P2-Q25-fix(M294)
    data = pd.DataFrame({"true": true, "prob": prob}).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    if data.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        pos = data.loc[data["true"] == 1, "prob"]
        neg = data.loc[data["true"] != 1, "prob"]
        ax.hist(neg, bins=30, alpha=0.55, color=style["green"], density=True, label="负类")
        ax.hist(pos, bins=30, alpha=0.55, color=style["red"], density=True, label="正类")
    ax.set_title("预测概率分布")
    ax.set_xlabel("预测为正类的概率")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_feature_importance(
    importances: Any,
    top_n: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> Figure | None:
    """Plot top feature importances."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    imp = _to_series(importances, "importance").sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    if imp.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.barh([str(i) for i in imp.index], imp.values, color=style["primary"], alpha=0.82)
    ax.set_title(f"特征重要性 Top {top_n}")
    ax.set_xlabel("重要性")
    ax.grid(True, axis="x", alpha=0.55)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_calibration_curve(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> Figure | None:
    """Plot a probability calibration curve without sklearn."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    true = pd.Series(y_true).astype(float).reset_index(drop=True)
    prob = _extract_pos_proba(y_prob, len(true)).astype(float)  # P2-Q25-fix(M294)
    data = pd.DataFrame({"true": true, "prob": prob}).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    if data.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = np.linspace(0, 1, max(int(n_bins), 2) + 1)
        data["bin"] = pd.cut(data["prob"], bins=bins, include_lowest=True, duplicates="drop")
        grouped = data.groupby("bin", observed=True).agg(pred=("prob", "mean"), actual=("true", "mean"), count=("true", "size"))
        ax.plot([0, 1], [0, 1], color=style["gray"], ls="--", lw=1.2, label="理想校准")
        ax.plot(grouped["pred"], grouped["actual"], marker="o", color=style["primary"], lw=2, label="模型")
        for _, row in grouped.iterrows():
            ax.scatter(row["pred"], row["actual"], s=max(20, row["count"] * 4), color=style["primary"], alpha=0.35)
    ax.set_title("校准曲线")
    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("实际正例比例")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_trade_analysis(
    trades: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> Figure | None:
    """Plot trade-level diagnostics: cumulative PnL, PnL distribution, win rate, and holding time."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    df = _to_dataframe(trades)
    lower = {str(c).lower(): c for c in df.columns}
    pnl_col = lower.get("pnl") or lower.get("profit") or lower.get("return") or lower.get("returns") or lower.get("收益")
    hold_col = lower.get("holding_period") or lower.get("hold_days") or lower.get("duration") or lower.get("持仓天数")
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna() if pnl_col is not None else pd.Series(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.ravel()
    if pnl.empty:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        cum = pnl.cumsum()
        ax1.plot(cum.index, cum.values, color=style["primary"], lw=2)
        ax1.set_title("累计交易PnL")
        ax1.grid(True, alpha=0.55)

        ax2.hist(pnl.values, bins=30, color=style["primary"], alpha=0.65)
        ax2.axvline(0, color=style["gray"], lw=1)
        ax2.set_title("单笔PnL分布")
        ax2.grid(True, axis="y", alpha=0.55)

        wins = int((pnl > 0).sum())
        losses = int((pnl <= 0).sum())
        ax3.bar(["盈利", "亏损"], [wins, losses], color=[style["red"], style["green"]], alpha=0.8)
        win_rate = wins / len(pnl) if len(pnl) else 0.0
        ax3.set_title(f"胜率 {win_rate:.1%}")
        ax3.grid(True, axis="y", alpha=0.55)

        if hold_col is not None:
            holding = pd.to_numeric(df[hold_col], errors="coerce").dropna()
            ax4.hist(holding.values, bins=25, color=style["orange"], alpha=0.7)
            ax4.set_title("持仓时间分布")
            ax4.set_xlabel("周期")
        else:
            gross_profit = pnl[pnl > 0].sum()
            gross_loss = abs(pnl[pnl < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
            ax4.axis("off")
            ax4.text(
                0.05,
                0.9,
                f"交易次数: {len(pnl)}\n平均PnL: {pnl.mean():.4f}\n盈亏比: {profit_factor:.2f}\n最大盈利: {pnl.max():.4f}\n最大亏损: {pnl.min():.4f}",
                va="top",
                transform=ax4.transAxes,
                fontsize=11,
            )
        ax4.grid(True, axis="y", alpha=0.55)
    fig.suptitle("交易分析", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_consecutive_trades(
    pnl_series: Any,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> Figure | None:
    """Plot consecutive winning and losing trade streaks from a PnL series."""
    if not _require_matplotlib():
        return None
    style = _apply_style()
    pnl = _to_series(pnl_series, "pnl")
    fig, ax = plt.subplots(figsize=figsize)
    if pnl.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
    else:
        signs = np.sign(pnl.values)
        colors = [style["red"] if value > 0 else style["green"] if value < 0 else style["gray"] for value in signs]
        ax.bar(np.arange(len(pnl)), signs, color=colors, alpha=0.78)
        win_len, win_start, win_end = _max_streak(pnl, positive=True)
        loss_len, loss_start, loss_end = _max_streak(pnl, positive=False)
        if win_len > 0:
            ax.axvspan(win_start - 0.5, win_end + 0.5, color=style["red"], alpha=0.12, label=f"最长连赢 {win_len}")
        if loss_len > 0:
            ax.axvspan(loss_start - 0.5, loss_end + 0.5, color=style["green"], alpha=0.12, label=f"最长连亏 {loss_len}")
        ax.axhline(0, color=style["gray"], lw=1)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["亏损", "持平", "盈利"])
        ax.set_xlim(-0.5, len(pnl) - 0.5)
    ax.set_title("连续交易盈亏")
    ax.set_xlabel("交易序号")
    _grid_legend(ax)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def save_all_figures(figures: dict[str, Figure | None], save_dir: str | Path) -> None:
    """Save every non-None figure in a dictionary to ``save_dir`` as PNG files."""
    if not _require_matplotlib():
        return
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        if fig is None:
            continue
        safe_name = str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.savefig(root / f"{safe_name}.png", bbox_inches="tight", dpi=150)


__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_monthly_returns",
    "plot_daily_returns_distribution",
    "plot_backtest_summary",
    "plot_factor_ic_decay",
    "plot_factor_long_short",
    "plot_factor_ic_rolling",
    "plot_factor_correlation_matrix",
    "plot_factor_group_returns",
    "plot_portfolio_weights",
    "plot_weight_evolution",
    "plot_efficient_frontier",
    "plot_market_regime",
    "plot_regime_radar",
    "plot_prediction_distribution",
    "plot_feature_importance",
    "plot_calibration_curve",
    "plot_trade_analysis",
    "plot_consecutive_trades",
    "save_all_figures",
    "set_chinese_font",
    "get_default_style",
]
