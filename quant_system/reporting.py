"""
reporting.py — 高级报告生成引擎
V4.1 feature

统一报告接口：日报/周报/月报/自定义，支持多格式输出。
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = __import__('logging').getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# P1-Q24-fix (H11): 与 performance.py 共用同一无风险利率配置，避免两模块口径不一致
# P2-Q24-fix (M276): 不再硬编码绝对路径——优先环境变量 QUANT_RISK_FREE_RATE，
# 其次读工作区 config/report_delivery.json（相对路径）；配置缺键时告警一次而非静默。
_warned_risk_free = False


def _load_risk_free() -> float:
    """读取年化无风险利率（默认 2.5%），配置缺失时回退默认值并告警一次。"""
    global _warned_risk_free
    try:
        env_rf = os.environ.get("QUANT_RISK_FREE_RATE")
        if env_rf is not None:
            return float(env_rf)
        _cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "report_delivery.json",
        )
        if os.path.exists(_cfg_path):
            with open(_cfg_path) as _f:
                _cfg = json.load(_f)
            if "risk_free_rate" in _cfg:
                return float(_cfg["risk_free_rate"])
        if not _warned_risk_free:
            print("[reporting] 警告: 无风险利率配置缺失，使用默认 0.025", file=sys.stderr)
            _warned_risk_free = True
        return 0.025
    except Exception as e:
        if not _warned_risk_free:
            print(f"[reporting] 警告: 读取无风险利率配置失败({e})，使用默认 0.025", file=sys.stderr)
            _warned_risk_free = True
        return 0.025


# ══════════════════════════════════════
# 1. Section Base
# ══════════════════════════════════════

class ReportSection:
    """报告段落基类"""
    def __init__(self, title: str = ""):
        self.title = title
        self.content = []

    def add_line(self, text: str):
        self.content.append(text)

    def add_table(self, df: pd.DataFrame, fmt: str = ""):
        self.content.append(f"\n[{self.title} 表格]")
        self.content.append(df.to_string(index=True))
        self.content.append("")

    def render(self) -> str:
        lines = [f"\n【{self.title}】", "-" * 40] if self.title else []
        lines.extend(self.content)
        return "\n".join(lines)


# ══════════════════════════════════════
# 2. 绩效报告块
# ══════════════════════════════════════

class PerformanceReport:
    """绩效报告"""

    def __init__(self, returns: pd.Series):
        self.returns = returns
        self.cum = (1 + returns).cumprod()
        self.peak = self.cum.expanding().max()
        self.dd = (self.cum - self.peak) / self.peak

    def summary_section(self) -> ReportSection:
        """基本绩效摘要"""
        sec = ReportSection("绩效摘要")
        # P1-Q24-fix (H11): 统一与 performance.py 的口径——
        # 年化收益改用几何 (1+total)^(252/n)−1，夏普扣除无风险利率
        n = len(self.returns)
        total_ret = float((1 + self.returns).prod() - 1) if n > 0 else 0.0
        ann_ret = float((1 + total_ret) ** (252 / n) - 1) if n > 0 else 0.0
        ann_vol = self.returns.std(ddof=1) * np.sqrt(252)
        rf_daily = _load_risk_free() / 252
        std_d = self.returns.std(ddof=1)
        sharpe = float((self.returns.mean() - rf_daily) / std_d * np.sqrt(252)) if std_d > 0 else 0.0
        max_dd = self.dd.min()

        sec.add_line(f"  年化收益: {ann_ret*100:.2f}%")
        sec.add_line(f"  年化波动: {ann_vol*100:.2f}%")
        sec.add_line(f"  夏普比率: {sharpe:.3f}")
        sec.add_line(f"  最大回撤: {max_dd*100:.2f}%")
        sec.add_line(f"  Calmar比: {ann_ret/max(abs(max_dd),1e-12):.2f}")
        sec.add_line(f"  总收益:   {total_ret*100:.2f}%")
        sec.add_line(f"  胜率:     {(self.returns>0).mean()*100:.1f}%")
        sec.add_line(f"  观测数:   {len(self.returns)}")
        return sec

    def monthly_section(self) -> ReportSection:
        """月度收益矩阵"""
        sec = ReportSection("月度收益")
        # P2-Q24-fix (L289): resample("ME") 依赖 pandas>=2.2，低版本兼容 "M"；
        # 同时兼容非 DatetimeIndex（整数/无法解析时给出明确提示而非崩溃）
        returns = self.returns
        if not isinstance(returns.index, pd.DatetimeIndex):
            _is_numeric = pd.api.types.is_integer_dtype(returns.index) or pd.api.types.is_float_dtype(returns.index)
            parsed = pd.to_datetime(returns.index, errors="coerce") if not _is_numeric else None
            if parsed is None or parsed.isna().any():
                sec.add_line("  收益索引非日期，无法生成月度收益矩阵")
                return sec
            returns = returns.copy()
            returns.index = parsed
        try:
            monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        except ValueError:
            monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly.index = monthly.index.strftime("%Y-%m")
        sec.add_table(monthly.to_frame("月收益"))
        sec.add_line(f"  月胜率: {(monthly>0).mean()*100:.1f}%")
        return sec

    def drawdown_section(self) -> ReportSection:
        """回撤分析"""
        sec = ReportSection("回撤分析")
        max_dd = self.dd.min()
        dd_duration = (self.dd < 0).astype(int).groupby(
            (self.dd >= 0).cumsum()).cumsum()

        # 找出最大回撤区间
        if max_dd < 0:
            trough_idx = self.dd.idxmin()
            peak_before = self.cum[:trough_idx].idxmax()
            recovery = self.cum[trough_idx:]
            recovery_date = recovery[recovery >= self.cum[peak_before]].index[0] if any(recovery >= self.cum[peak_before]) else "未恢复"

            sec.add_line(f"  最大回撤: {max_dd*100:.2f}%")
            sec.add_line(f"  回撤开始: {str(peak_before)[:10]}")
            sec.add_line(f"  回撤谷底: {str(trough_idx)[:10]}")
            sec.add_line(f"  恢复日期: {str(recovery_date)[:10]}")
            sec.add_line(f"  最长回撤期: {dd_duration.max()}日")

        return sec

    def risk_section(self) -> ReportSection:
        """风险分析"""
        sec = ReportSection("风险分析")
        # P2-Q24-fix (M273): 全为正收益时 neg_ret 为空 → CVaR/下跌波动率为 NaN。
        # 空序列返回 0；var_95 无负值时（95%分位为正）按"无损失"置 0。
        neg_ret = self.returns[self.returns < 0]
        var_95 = self.returns.quantile(0.05)
        if pd.isna(var_95):
            var_95 = 0.0
        elif var_95 > 0:
            var_95 = 0.0  # 95%分位为正 → 无损失
        if neg_ret.empty:
            cvar_95 = 0.0
            downside_vol = 0.0
        else:
            tail = neg_ret[neg_ret <= var_95]
            cvar_95 = float(tail.mean()) if not tail.empty else 0.0
            downside_vol = float(neg_ret.std()) if len(neg_ret) >= 2 else 0.0

        skew = self.returns.skew()
        kurt = self.returns.kurtosis()

        se = self.returns.std() / np.sqrt(len(self.returns))

        sec.add_line(f"  VaR(95%): {var_95*100:.2f}%")
        sec.add_line(f"  CVaR(95%): {cvar_95*100:.2f}%")
        sec.add_line(f"  偏度: {skew:.3f}")
        sec.add_line(f"  峰度: {kurt:.3f}")
        sec.add_line(f"  标准误: {se*100:.3f}%")
        sec.add_line(f"  下跌波动率: {downside_vol*np.sqrt(252)*100:.2f}%")
        return sec


# ══════════════════════════════════════
# 3. 日报/周报/月报
# ══════════════════════════════════════

class DailyReport:
    """每日绩效报告"""

    def __init__(self, strategy_name: str = "默认策略"):
        self.strategy_name = strategy_name
        self.sections: list[ReportSection] = []

    def add_section(self, section: ReportSection):
        self.sections.append(section)

    def generate(self, returns: pd.Series, positions: Optional[dict] = None,
                  trades: Optional[list] = None) -> str:
        """生成日报"""
        perf = PerformanceReport(returns)

        # 标题
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [
            "=" * 55,
            f"每日策略报告: {self.strategy_name}",
            f"报告日期: {today}",
            "=" * 55,
        ]

        # 当日绩效
        daily_ret = returns.iloc[-1] if len(returns) > 0 else 0
        cum_ret = (1 + returns).prod() - 1

        lines.append("")
        lines.append(f"  【当日】收益: {daily_ret*100:+.2f}%")
        lines.append(f"  【累计】收益: {cum_ret*100:+.2f}%")

        # 绩效摘要
        lines.append("")
        lines.append(perf.summary_section().render())

        # 回撤
        lines.append("")
        lines.append(perf.drawdown_section().render())

        # 持仓
        if positions:
            lines.append("")
            lines.append("【当前持仓】")
            for sym, vol in positions.items():
                lines.append(f"  {sym}: {vol}")

        # 交易
        # P1-Q24-fix (H09): 系统 trades 字段为 trade_type/shares（trade_db 建表），
        # 兼容旧 side/volume 写法
        if trades and len(trades) > 0:
            lines.append("")
            lines.append(f"【今日交易】共 {len(trades)} 笔")
            for t in trades[-10:]:
                side = t.get('trade_type', t.get('side', ''))
                vol = t.get('shares', t.get('volume', 0))
                lines.append(f"  {t.get('symbol','')} {side} {vol}@{t.get('price',0):.2f}")

        return "\n".join(lines)


class WeeklyReport:
    """每周绩效报告"""

    def generate(self, daily_returns: pd.Series, factor_performance: Optional[pd.DataFrame] = None) -> str:
        """生成周报"""
        # P2-Q24-fix (M274): 索引可能非 DatetimeIndex（整数/字符串），
        # 直接 `index[-1] - timedelta` 会 TypeError。DatetimeIndex 直接用；
        # 字符串索引解析为日期；整数/无法解析的索引用"最近 7 个观测"兜底。
        if len(daily_returns) > 0:
            _idx = daily_returns.index
            _is_dt = isinstance(_idx, pd.DatetimeIndex)
            _is_numeric = pd.api.types.is_integer_dtype(_idx) or pd.api.types.is_float_dtype(_idx)
            if _is_dt:
                week_start = _idx[-1] - timedelta(days=7)
                weekly = daily_returns[daily_returns.index >= week_start]
            elif _is_numeric:
                # 整数/浮点索引无法映射到真实日期：按位置取最近 7 个交易日
                _pos = max(0, len(daily_returns) - 7)
                week_start = _pos
                weekly = daily_returns.iloc[_pos:]
            else:
                _parsed = pd.to_datetime(_idx, errors="coerce")
                if _parsed.isna().any():
                    _pos = max(0, len(daily_returns) - 7)
                    week_start = _pos
                    weekly = daily_returns.iloc[_pos:]
                else:
                    daily_returns = daily_returns.copy()
                    daily_returns.index = _parsed
                    week_start = daily_returns.index[-1] - timedelta(days=7)
                    weekly = daily_returns[daily_returns.index >= week_start]
        else:
            week_start = datetime.now()
            weekly = daily_returns

        perf = PerformanceReport(daily_returns)
        wk_ret = (1 + weekly).prod() - 1 if len(weekly) > 0 else 0

        lines = [
            "=" * 55,
            f"每周策略报告",
            f"周期: {str(week_start)[:10]} ~ {str(daily_returns.index[-1])[:10] if len(daily_returns) > 0 else ''}",
            "=" * 55,
            "",
            f"  本周收益: {wk_ret*100:+.2f}%",
            "",
            perf.summary_section().render(),
            "",
            perf.drawdown_section().render(),
            "",
            perf.monthly_section().render(),
        ]

        if factor_performance is not None:
            lines.append("")
            lines.append("【因子表现】")
            lines.append(factor_performance.to_string(index=True))

        return "\n".join(lines)


class MonthlyReport:
    """月度绩效报告"""

    def generate(self, daily_returns: pd.Series) -> str:
        """生成月报"""
        perf = PerformanceReport(daily_returns)
        lines = [
            "=" * 55,
            "月度策略报告 (V4.1 feature)",
            f"报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
            "",
            perf.summary_section().render(),
            "",
            perf.drawdown_section().render(),
            "",
            perf.risk_section().render(),
            "",
            perf.monthly_section().render(),
        ]
        return "\n".join(lines)


# ══════════════════════════════════════
# 4. 图表生成
# ══════════════════════════════════════

class ChartGenerator:
    """图表生成器"""

    @staticmethod
    def equity_curve(returns: pd.Series, save_path: str = "",
                     title: str = "净值曲线") -> Optional[str]:
        """生成净值曲线图"""
        if not _HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        cum = (1 + returns).cumprod()
        ax.plot(cum.index, cum.values, linewidth=1.5, color="steelblue")
        ax.fill_between(cum.index, cum.values, 1, alpha=0.1, color="steelblue")
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("净值")
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.5)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return fig

    @staticmethod
    def drawdown_chart(returns: pd.Series, save_path: str = "") -> Optional[str]:
        """回撤图"""
        if not _HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(12, 3))
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak

        ax.fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.5)
        ax.set_title("回撤曲线", fontsize=12)
        ax.set_ylabel("回撤 %")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return fig

    @staticmethod
    def monthly_heatmap(returns: pd.Series, save_path: str = "") -> Optional[str]:
        """月度收益热力图"""
        if not _HAS_MPL:
            return None

        df = returns.to_frame("return").dropna(subset=["return"])
        # P2-Q24-fix (L290): df.index.year 要求 DatetimeIndex，非日期索引先转换，
        # 整数/无法解析时明确告警并返回 None（不绘制），避免直接报错
        if not isinstance(df.index, pd.DatetimeIndex):
            _is_numeric = pd.api.types.is_integer_dtype(df.index) or pd.api.types.is_float_dtype(df.index)
            if _is_numeric:
                print(
                    "[reporting] 警告: monthly_heatmap 收益索引为整数/非日期，跳过热力图",
                    file=sys.stderr,
                )
                return None
            df.index = pd.to_datetime(df.index, errors="coerce")
        if df.index.isna().any():
            print(
                "[reporting] 警告: monthly_heatmap 收益索引无法解析为日期，跳过热力图",
                file=sys.stderr,
            )
            return None
        df["year"] = df.index.year
        df["month"] = df.index.month
        monthly = df.groupby(["year", "month"])["return"].apply(lambda x: (1 + x).prod() - 1)
        heatmap_data = monthly.unstack()

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
        # P2-Q24-fix (L290): 月份列数可能不足12（如仅1个交易日/单月数据），
        # 标签数量需与列数一致，否则 matplotlib 报错
        _months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        _n_cols = len(heatmap_data.columns)
        ax.set_xticks(range(_n_cols))
        ax.set_xticklabels(_months[:_n_cols])
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index.astype(int))
        plt.colorbar(im, ax=ax)
        ax.set_title("月度收益热力图", fontsize=14)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        return fig


# ══════════════════════════════════════
# 5. ReportGenerator — 顶层整合
# ══════════════════════════════════════

class ReportGenerator:
    """报告生成器顶层整合"""

    def __init__(self, strategy_name: str = "默认策略"):
        self.strategy_name = strategy_name
        self.daily = DailyReport(strategy_name)
        self.weekly = WeeklyReport()
        self.monthly = MonthlyReport()
        self.charts = ChartGenerator()

    def generate_daily(self, returns: pd.Series, positions: Optional[dict] = None,
                        trades: Optional[list] = None) -> str:
        return self.daily.generate(returns, positions, trades)

    def generate_weekly(self, returns: pd.Series) -> str:
        return self.weekly.generate(returns)

    def generate_monthly(self, returns: pd.Series) -> str:
        return self.monthly.generate(returns)

    def generate_comprehensive(self, returns: pd.Series,
                                factor_df: Optional[pd.DataFrame] = None,
                                positions: Optional[dict] = None,
                                save_dir: str = "") -> dict:
        """生成综合报告（文本+图表）"""
        result = {
            "daily": self.daily.generate(returns, positions),
            "weekly": self.weekly.generate(returns),
            "monthly": self.monthly.generate(returns),
            "charts": {},
        }

        if save_dir and _HAS_MPL:
            os.makedirs(save_dir, exist_ok=True)
            equity_path = os.path.join(save_dir, "equity_curve.png")
            dd_path = os.path.join(save_dir, "drawdown.png")
            self.charts.equity_curve(returns, save_path=equity_path)
            self.charts.drawdown_chart(returns, save_path=dd_path)
            result["charts"]["equity"] = equity_path
            result["charts"]["drawdown"] = dd_path

        return result

    def export_to_json(self, returns: pd.Series, path: str = "") -> str:
        """导出为JSON"""
        # P1-Q24-fix (H11): 与 summary_section 统一几何年化 + 扣除无风险利率
        _n = len(returns)
        _tot = float((1 + returns).prod() - 1) if _n > 0 else 0.0
        _std = returns.std(ddof=1)
        _rf_daily = _load_risk_free() / 252
        stats = {
            "total_return": _tot,
            "annual_return": float((1 + _tot) ** (252 / _n) - 1) if _n > 0 else 0.0,
            "annual_vol": float(_std * np.sqrt(252)),
            "sharpe": float((returns.mean() - _rf_daily) / _std * np.sqrt(252)) if _std > 0 else 0.0,
            "max_dd": float((1 + returns).cumprod().div(
                (1 + returns).cumprod().expanding().max()).min() - 1),
            "win_rate": float((returns > 0).mean()),
            "n_observations": len(returns),
            "generated_at": datetime.now().isoformat(),
        }
        json_str = json.dumps(stats, indent=2, ensure_ascii=False)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
        return json_str


__all__ = [
    "PerformanceReport", "ReportSection",
    "DailyReport", "WeeklyReport", "MonthlyReport",
    "ChartGenerator", "ReportGenerator",
]
