#!/usr/bin/env python3
"""ETF 套利回测框架

功能模块：
  1. IOPV vs 市价折溢价监测
  2. 成分股一篮子跟踪误差分析
  3. 折溢价套利策略回测
  4. 历史折溢价分布统计

数据源：
  - AkShare fund_etf_hist_em: ETF 日线（含 IOPV 列）
  - 成分股权重: 中证指数官网 / fund_portfolio_hold_detail_em
  - 实时折溢价: fund_etf_spot_em

用法示例：
  from modules.etf_arbitrage import ETFArbitrageBacktest
  bt = ETFArbitrageBacktest("510050", "上证50ETF",
                             start="2025-01-01", end="2026-07-30")
  report = bt.run()        # 运行完整回测
  summary = bt.summary()   # 打印摘要统计
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


# ── Data Structures ──────────────────────────────────────────────────────


@dataclass
class ArbitrageSignal:
    date: str
    premium_pct: float         # +为溢价, -为折价
    iopv: float
    price: float
    volume: float
    signal_type: str           # "premium_arbitrage" | "discount_arbitrage" | "neutral"
    expected_profit_pct: float # 不考虑摩擦成本的期望利润
    notes: str = ""


@dataclass
class BacktestResult:
    etf_code: str
    etf_name: str
    period_start: str
    period_end: str
    total_days: int = 0
    avg_premium_pct: float = 0.0
    max_premium_pct: float = 0.0
    max_discount_pct: float = 0.0
    premium_std: float = 0.0
    premium_percentile_90: float = 0.0
    premium_percentile_10: float = 0.0
    signals: list[ArbitrageSignal] = field(default_factory=list)
    trade_count: int = 0
    win_count: int = 0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_approx: float = 0.0
    tracking_error_bp: float = 0.0  # 跟踪误差（基点/日）
    iopv_coverage_pct: float = 0.0  # IOPV 有效数据覆盖率
    fee_per_trade_pct: float = 0.03  # 单边交易成本 %
    notes: list[str] = field(default_factory=list)


# ── Data Fetching ────────────────────────────────────────────────────────


def fetch_etf_daily(code: str, start: str = "20250101", end: str = "") -> pd.DataFrame:
    """Fetch ETF daily OHLCV + IOPV from AkShare.

    推荐主源: fund_etf_hist_em (东方财富)
    字段包含 IOPV 折溢价信息。
    """
    try:
        import akshare as ak
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=(end or date.today().strftime("%Y%m%d")),
            adjust="qfq",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as exc:
        print(f"[etf_arbitrage] 获取 {code} 日线失败: {exc}", file=sys.stderr)
        return pd.DataFrame()


def fetch_etf_realtime_premium(code: str) -> dict[str, Any]:
    """获取 ETF 实时折溢价 (fund_etf_spot_em)."""
    out: dict[str, Any] = {"ok": False, "premium_pct": None, "iopv": None, "price": None}
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        if df is None or df.empty:
            return out
        row = df[df["代码"] == code]
        if row.empty:
            return out
        r = row.iloc[0]
        out.update({
            "ok": True,
            "code": code,
            "name": str(r.get("名称", "")),
            "price": float(r["最新价"]) if "最新价" in r else None,
            "iopv": float(r["IOPV"]) if "IOPV" in r else None,
            "premium_pct": float(r["折溢价率"]) if "折溢价率" in r else None,
        })
        return out
    except Exception as exc:
        print(f"[etf_arbitrage] 获取 {code} 实时折溢价失败: {exc}", file=sys.stderr)
        return out


def fetch_etf_constituents(code: str, date_str: str = "") -> list[dict[str, Any]]:
    """获取 ETF 持仓权重（使用 fund_portfolio_hold_detail_em）。

    Returns: [{"code": "600519", "name": "贵州茅台", "weight_pct": 15.2}, ...]
    """
    try:
        import akshare as ak
        df = ak.fund_portfolio_hold_detail_em(symbol=code, date=date_str)
        if df is None or df.empty:
            return []
        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            weight = float(r.get("占净值比例", r.get("持仓占比", 0)))
            if math.isnan(weight):
                weight = 0
            rows.append({
                "code": str(r.get("股票代码", "")),
                "name": str(r.get("股票名称", "")),
                "weight_pct": round(weight, 2),
            })
        return rows
    except Exception as exc:
        print(f"[etf_arbitrage] 获取 {code} 持仓失败: {exc}", file=sys.stderr)
        return []


# ── Core Analysis ────────────────────────────────────────────────────────


def compute_premium_stats(df: pd.DataFrame) -> dict[str, Any]:
    """从 ETF 日线计算折溢价统计。

    期望列名: "IOPV" 或 "折溢价" 或自动推导。
    """
    if df.empty:
        return {}

    # 尝试找 IOPV 和价格列
    price_col = None
    iopv_col = None
    for c in ["收盘价", "close", "Close", "净值", "IOPV", "单位净值"]:
        if c in df.columns:
            if price_col is None:
                if c in ["IOPV", "净值", "单位净值"]:
                    iopv_col = c
                else:
                    price_col = c
            else:
                if c in ["IOPV", "净值", "单位净值"]:
                    iopv_col = c

    if not price_col:
        price_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if not iopv_col:
        iopv_col = "IOPV"

    if iopv_col not in df.columns:
        return {"error": f"缺少 IOPV 列，可用列: {list(df.columns)}"}

    prices = pd.to_numeric(df[price_col], errors="coerce")
    iopvs = pd.to_numeric(df[iopv_col], errors="coerce")

    valid = prices.notna() & iopvs.notna() & (iopvs > 0)
    if valid.sum() < 2:
        return {"error": "有效 IOPV 数据不足"}

    premium = (prices[valid] / iopvs[valid] - 1) * 100

    stats = {
        "有效天数": int(valid.sum()),
        "总天数": len(df),
        "数据覆盖率_pct": round(float(valid.sum() / len(df) * 100), 1),
        "平均折溢价_pct": round(float(premium.mean()), 4),
        "中位数折溢价_pct": round(float(premium.median()), 4),
        "最大溢价_pct": round(float(premium.max()), 4),
        "最大折价_pct": round(float(premium.min()), 4),
        "标准差_pct": round(float(premium.std()), 4),
        "90分位溢价_pct": round(float(premium.quantile(0.90)), 4),
        "10分位折价_pct": round(float(premium.quantile(0.10)), 4),
        "大于0.5pct溢价天数": int((premium > 0.5).sum()),
        "小于-0.5pct折价天数": int((premium < -0.5).sum()),
    }

    return stats


def detect_signals(
    df: pd.DataFrame,
    premium_threshold: float = 0.5,
    fee_pct: float = 0.03,
) -> list[ArbitrageSignal]:
    """从日线数据检测折溢价套利信号。

    premium_threshold: 触发套利的折溢价阈值（百分比）
    fee_pct: 单边交易成本（百分比）
    """
    if df.empty:
        return []

    price_col = "收盘价"
    iopv_col = "IOPV"
    for c in ["close", "Close", "收盘价", "close_price"]:
        if c in df.columns:
            price_col = c
            break
    for c in ["IOPV", "净值", "单位净值", "iopv", "IOPV_NAV"]:
        if c in df.columns:
            iopv_col = c
            break

    if iopv_col not in df.columns:
        return []

    signals: list[ArbitrageSignal] = []
    prices = pd.to_numeric(df[price_col], errors="coerce")
    iopvs = pd.to_numeric(df[iopv_col], errors="coerce")

    for idx in range(len(df)):
        p = prices.iloc[idx]
        iv = iopvs.iloc[idx]
        if pd.isna(p) or pd.isna(iv) or iv <= 0:
            continue

        premium_pct = (p / iv - 1) * 100
        signal_type = "neutral"
        expected_profit = 0.0
        notes = ""

        if premium_pct > premium_threshold:
            signal_type = "premium_arbitrage"
            # 溢价套利：卖ETF + 买一篮子成分股
            expected_profit = premium_pct - 2 * fee_pct
            notes = f"溢价{premium_pct:.2f}%, 做空套利（卖ETF买成分股）, 预期净利{expected_profit:.2f}%"
        elif premium_pct < -premium_threshold:
            signal_type = "discount_arbitrage"
            # 折价套利：买ETF + 卖一篮子成分股
            expected_profit = -premium_pct - 2 * fee_pct
            notes = f"折价{premium_pct:.2f}%, 做多套利（买ETF卖成分股）, 预期净利{expected_profit:.2f}%"

        date_str = str(df.index[idx].date()) if hasattr(df.index[idx], "date") else str(df.iloc[idx].get("日期", ""))
        volume = float(df.iloc[idx].get("成交量", df.iloc[idx].get("volume", 0)))

        signals.append(ArbitrageSignal(
            date=date_str,
            premium_pct=round(premium_pct, 4),
            iopv=round(float(iv), 4),
            price=round(float(p), 4),
            volume=volume,
            signal_type=signal_type,
            expected_profit_pct=round(expected_profit, 4),
            notes=notes,
        ))

    return signals


def simple_backtest(
    signals: list[ArbitrageSignal],
    capital: float = 1000000.0,
) -> dict[str, Any]:
    """对信号序列做简单回测（每次等额开仓，完全成交假设）。"""
    if not signals:
        return {"trades": 0, "total_return": 0, "max_drawdown": 0}

    trades: list[dict] = []
    equity = capital
    peak = capital
    max_dd = 0.0
    wins = 0

    for sig in signals:
        if sig.signal_type == "neutral":
            continue
        profit_pct = sig.expected_profit_pct
        position = capital * 0.1  # 每次仅用10%资金
        profit = position * profit_pct / 100
        equity += profit
        if profit > 0:
            wins += 1
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
        trades.append({
            "date": sig.date,
            "signal": sig.signal_type,
            "premium_pct": sig.premium_pct,
            "profit": round(profit, 2),
            "cumulative_return_pct": round((equity - capital) / capital * 100, 4),
        })

    total_return = (equity - capital) / capital * 100
    return {
        "初始资金": capital,
        "最终资金": round(equity, 2),
        "总收益率_pct": round(total_return, 4),
        "交易次数": len(trades),
        "盈利次数": wins,
        "胜率_pct": round(wins / len(trades) * 100, 1) if trades else 0,
        "最大回撤_pct": round(max_dd, 4),
        "夏普近似": round(total_return / max(max_dd, 0.01), 2) if max_dd > 0 else 0,
    }


def tracking_error(
    etf_returns: pd.Series,
    constituent_returns: pd.DataFrame,
    weights: list[float],
) -> float:
    """计算 ETF 相对一篮子成分股的跟踪误差（年化基点）。

    etf_returns: ETF 日收益率序列
    constituent_returns: 成分股 DataFrame（列=股票代码, 值=日收益率）
    weights: 权重列表（与 constituent_returns 列顺序一致）
    """
    if len(weights) != len(constituent_returns.columns):
        return float("nan")
    if len(weights) == 0:
        return float("nan")

    weights_sum = sum(weights)
    if weights_sum <= 0:
        return float("nan")
    norm_weights = [w / weights_sum for w in weights]

    basket_return = pd.Series(0.0, index=constituent_returns.index)
    for i, col in enumerate(constituent_returns.columns):
        basket_return += constituent_returns[col].fillna(0) * norm_weights[i]

    te_series = (etf_returns - basket_return) * 10000  # 转基点
    te_daily = float(te_series.std())
    te_annual = te_daily * (252 ** 0.5)
    return round(te_annual, 2)


# ── Main Runner ──────────────────────────────────────────────────────────


class ETFArbitrageBacktest:
    """ETF 套利回测主类"""

    def __init__(
        self,
        etf_code: str,
        etf_name: str = "",
        start: str = "20250101",
        end: str = "",
        premium_threshold: float = 0.5,
        fee_pct: float = 0.03,
        capital: float = 1000000.0,
    ):
        self.etf_code = etf_code
        self.etf_name = etf_name or etf_code
        self.start = start
        self.end = end or date.today().strftime("%Y-%m-%d")
        self.premium_threshold = premium_threshold
        self.fee_pct = fee_pct
        self.capital = capital
        self._df: pd.DataFrame = pd.DataFrame()
        self._stats: dict[str, Any] = {}
        self._signals: list[ArbitrageSignal] = []
        self._result: BacktestResult | None = None

    def fetch_data(self) -> pd.DataFrame:
        """拉取 ETF 历史日线（含 IOPV）。"""
        self._df = fetch_etf_daily(self.etf_code, self.start, self.end)
        return self._df

    def analyze(self) -> dict[str, Any]:
        """执行折溢价分析。"""
        self._stats = compute_premium_stats(self._df)
        self._signals = detect_signals(self._df, self.premium_threshold, self.fee_pct)
        return self._stats

    def backtest(self) -> dict[str, Any]:
        """运行简单回测。"""
        return simple_backtest(self._signals, self.capital)

    def run(self) -> BacktestResult:
        """全流程：取数→分析→回测。"""
        self.fetch_data()
        self.analyze()

        stats = self._stats
        bt = simple_backtest(self._signals, self.capital)

        result = BacktestResult(
            etf_code=self.etf_code,
            etf_name=self.etf_name,
            period_start=self.start,
            period_end=self.end,
            total_days=stats.get("总天数", 0),
            avg_premium_pct=stats.get("平均折溢价_pct", 0),
            max_premium_pct=stats.get("最大溢价_pct", 0),
            max_discount_pct=stats.get("最大折价_pct", 0),
            premium_std=stats.get("标准差_pct", 0),
            premium_percentile_90=stats.get("90分位溢价_pct", 0),
            premium_percentile_10=stats.get("10分位折价_pct", 0),
            signals=self._signals,
            trade_count=bt.get("交易次数", 0),
            win_count=bt.get("盈利次数", 0),
            total_return_pct=bt.get("总收益率_pct", 0),
            max_drawdown_pct=bt.get("最大回撤_pct", 0),
            sharpe_approx=bt.get("夏普近似", 0),
            tracking_error_bp=0.0,
            iopv_coverage_pct=stats.get("数据覆盖率_pct", 0),
            fee_per_trade_pct=self.fee_pct,
            notes=[
                f"折溢价阈值: ±{self.premium_threshold}%",
                f"交易成本: 单边{self.fee_pct}%, 双边{2*self.fee_pct}%",
                f"每次开仓比例: 10%",
                f"IOPV 覆盖率: {stats.get('数据覆盖率_pct', 0)}%",
                f"溢价>0.5%天数: {stats.get('大于0.5pct溢价天数', 0)}",
                f"折价<-0.5%天数: {stats.get('小于-0.5pct折价天数', 0)}",
            ],
        )
        self._result = result
        return result

    def summary(self) -> str:
        """输出可读回测摘要。"""
        if self._result is None:
            self.run()
        r = self._result
        if not r:
            return "回测未执行"
        lines = [
            f"## ETF 套利回测：{r.etf_name} ({r.etf_code})",
            f"回测区间: {r.period_start} → {r.period_end} · {r.total_days} 个交易日",
            "",
            "### 折溢价统计",
            f"- 平均折溢价: {r.avg_premium_pct:+.4f}%",
            f"- 中位数折溢价: 待计算",
            f"- 最大溢价: {r.max_premium_pct:+.4f}%",
            f"- 最大折价: {r.max_discount_pct:+.4f}%",
            f"- 标准差: {r.premium_std:.4f}%",
            f"- 90分位溢价: {r.premium_percentile_90:+.4f}%",
            f"- 10分位折价: {r.premium_percentile_10:+.4f}%",
            f"- IOPV 覆盖率: {r.iopv_coverage_pct:.1f}%",
            "",
            "### 套利信号统计",
            f"- 触发信号: {r.trade_count} 次",
            f"- 盈利: {r.win_count} 次 / {r.win_count/max(r.trade_count,1)*100:.1f}% 胜率",
            f"- 总收益率: {r.total_return_pct:+.4f}%",
            f"- 最大回撤: {r.max_drawdown_pct:.4f}%",
            f"- 近似夏普: {r.sharpe_approx:.2f}",
            "",
            "### 备注",
        ]
        for note in r.notes:
            lines.append(f"- {note}")
        if r.signals:
            lines.append("")
            lines.append("### 部分信号明细（前20条）")
            for sig in r.signals[:20]:
                lines.append(f"  {sig.date} | {sig.signal_type:20s} | 折溢价={sig.premium_pct:+.4f}% | 预期净利={sig.expected_profit_pct:+.4f}%")
        lines.append("")
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="ETF 套利回测框架")
    parser.add_argument("code", help="ETF 代码，如 510050")
    parser.add_argument("--name", default="", help="ETF 名称")
    parser.add_argument("--start", default="20250101", help="回测开始日期")
    parser.add_argument("--end", default="", help="回测结束日期（默认今日）")
    parser.add_argument("--threshold", type=float, default=0.5, help="折溢价阈值%")
    parser.add_argument("--fee", type=float, default=0.03, help="单边交易成本%")
    parser.add_argument("--capital", type=float, default=1000000, help="初始资金")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    bt = ETFArbitrageBacktest(
        etf_code=args.code,
        etf_name=args.name or args.code,
        start=args.start,
        end=args.end,
        premium_threshold=args.threshold,
        fee_pct=args.fee,
        capital=args.capital,
    )
    result = bt.run()

    if args.json:
        d = asdict(result)
        d["signals"] = [asdict(s) for s in result.signals]
        print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
    else:
        print(bt.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
