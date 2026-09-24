"""
bond_equity — 股债性价比 (V5 跨市场验证)

核心问题：现在应该买股票还是买债券？
  1. 拉取上证指数的市盈率(PE)序列，取倒数得到股票"隐含收益率" (1/PE)
  2. 拉取10年期中国国债收益率
  3. 股权风险溢价 ERP = 1/PE - 10Y国债收益率
     ERP 越高 → 相对于债券，股票越有性价比（历史上是重要的择时信号）
  4. 把 ERP 放到自身历史序列里算分位，分位越低代表股票相对越贵
  5. 同时看国债收益率曲线的期限利差(30Y-10Y)，反映债市对经济/通胀预期

数据来源：
  - 股票估值: akshare.stock_market_pe_lg(symbol="上证") — 乐咕乐股，
    月度更新的上证指数平均市盈率（规格书中提到的 stock_market_fund_em()
    在当前 akshare 版本中不存在，这里用等价的估值数据源代替）。
  - 债券收益率: akshare.bond_china_yield(start_date, end_date) — 中债
    收益率曲线，取"中债国债收益率曲线"的10年/30年字段。

对标：经典的 "FED Model" / 股债性价比模型，国内常用 ERP = E/P - 10Y国债。
"""

from __future__ import annotations
import logging

import threading as _th
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent

ERP_HISTORY_YEARS = 10  # PE历史数据尽量拉长，以获得更可靠的分位基准
# P2-Q23-fix(M257): 债券收益率历史窗口原为 400 天，与"10年ERP历史分位"名不
# 副实（ERP 分位实际只覆盖 ≈13 个月）。拉长到覆盖整个 PE 历史窗口，使 ERP
# 分位真正基于 10 年序列；末尾 +30 天缓冲避免 PE 月末与债券交易日边界截断。
BOND_HISTORY_DAYS = 365 * ERP_HISTORY_YEARS + 30

ERP_HIGH_PCT = 80  # ERP处于高分位 → 股票相对债券便宜
ERP_LOW_PCT = 20  # ERP处于低分位 → 股票相对债券贵


def _call_with_timeout(fn: Callable[[], Any], timeout: float = 15.0) -> Any:
    """独立线程执行网络请求，超时放弃，避免整体阻塞。"""
    box: dict[str, Any] = {"result": None}

    def _run() -> None:
        try:
            box["result"] = fn()
        except Exception:
            box["result"] = None

    t = _th.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return box["result"]


class BondEquity:
    """股债性价比(ERP)分析引擎。

    核心假设：
      股票的隐含收益率(1/PE)与无风险利率(10Y国债)之差(ERP)是均值回归的。
      ERP处于历史高位 → 股票相对债券便宜，往后配置股票的性价比更高；
      ERP处于历史低位 → 股票相对债券贵，往后更应偏向债券或防御。

    Attributes:
        cache: 上一次 compute() 结果缓存
        last_fetch: 上次成功计算的时间戳
        cache_ttl: 缓存有效期（秒）
    """

    def __init__(self, cache_ttl: int = 1800) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0.0
        self.cache_ttl = cache_ttl

    # ────────────────────────── 数据获取 ──────────────────────────

    def _fetch_pe_history(self) -> pd.DataFrame:
        """获取上证指数市盈率历史（月度）。

        Returns:
            DataFrame(index=日期, columns=["index_level", "pe"])，
            失败返回空 DataFrame。
        """

        def _do() -> pd.DataFrame | None:
            import akshare as ak
            return ak.stock_market_pe_lg(symbol="上证")

        df = _call_with_timeout(_do, timeout=15.0)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df["平均市盈率"] = pd.to_numeric(df["平均市盈率"], errors="coerce")
        df["指数"] = pd.to_numeric(df["指数"], errors="coerce")
        df = df.dropna(subset=["日期", "平均市盈率"]).sort_values("日期")
        df = df[df["平均市盈率"] > 0]
        cutoff = datetime.now(CST) - timedelta(days=365 * ERP_HISTORY_YEARS)
        df = df[df["日期"] >= pd.Timestamp(cutoff.date())]
        return df.set_index("日期")[["指数", "平均市盈率"]].rename(
            columns={"指数": "index_level", "平均市盈率": "pe"}
        )

    def _fetch_bond_yield(self) -> pd.DataFrame:
        """获取中债国债收益率曲线（10年/30年）历史。

        Returns:
            DataFrame(index=日期, columns=["y10", "y30"])，
            失败返回空 DataFrame。
        """
        end = datetime.now(CST)
        start = end - timedelta(days=BOND_HISTORY_DAYS)

        def _do() -> pd.DataFrame | None:
            import akshare as ak
            # V11 审计修复（High）: akshare bond_china_yield 要求
            # start_date-end_date < 1 年（10年窗口实测返回 0 行），
            # 原 BOND_HISTORY_DAYS=10年 直接失效 → ERP 恒"数据源不可用"。
            # 修正: 按年分段拉取（每年一段 ≤1年）拼接出完整历史。
            frames = []
            seg_start = start
            while seg_start < end:
                seg_end = min(seg_start + timedelta(days=365), end)
                try:
                    seg = ak.bond_china_yield(
                        start_date=seg_start.strftime("%Y%m%d"),
                        end_date=seg_end.strftime("%Y%m%d"),
                    )
                    if seg is not None and not seg.empty:
                        frames.append(seg)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[bond_equity] 操作失败: {e}", exc_info=True)  # 单段失败跳过，其他段仍可用
                seg_start = seg_end + timedelta(days=1)
            if not frames:
                return None
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["日期", "曲线名称"], keep="last")
            return df

        # V11 审计修复: 10 段分段拉取需 ~60-90s，60s 超时不够
        #（实测 2 段就超 60s），放宽到 180s 确保完整历史。
        df = _call_with_timeout(_do, timeout=180.0)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if "曲线名称" not in df.columns:
            return pd.DataFrame()
        df = df[df["曲线名称"] == "中债国债收益率曲线"]
        if df.empty or "10年" not in df.columns or "30年" not in df.columns:
            return pd.DataFrame()
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df["10年"] = pd.to_numeric(df["10年"], errors="coerce")
        df["30年"] = pd.to_numeric(df["30年"], errors="coerce")
        df = df.dropna(subset=["日期", "10年"]).sort_values("日期")
        return df.set_index("日期")[["10年", "30年"]].rename(columns={"10年": "y10", "30年": "y30"})

    # ────────────────────────── 分析逻辑 ──────────────────────────

    @staticmethod
    def _percentile(value: float, history: list[float]) -> float:
        """value 在 history 序列中的百分位 (0~100)。"""
        if len(history) < 5:
            return 50.0
        arr = np.array(history, dtype=float)
        less = np.sum(arr < value)
        equal = np.sum(arr == value)
        return round(float((less + 0.5 * equal) / len(arr) * 100), 1)

    # ────────────────────────── 主计算 ──────────────────────────

    def compute(self) -> dict[str, Any]:
        """计算当前股债性价比状态。

        Returns:
            dict: {
                "timestamp": str,
                "equity_risk_premium": float,   # ERP, 单位小数(如0.02代表2%)
                "erp_percentile": float,        # ERP在自身历史序列中的分位
                "bond_yield_10y": float,
                "bond_yield_30y": float,
                "yield_curve_slope": float,     # 30Y - 10Y, 单位小数
                "equity_implied_yield": float,  # 1/PE
                "pe": float,
                "signal": str,
                "ok": bool,
            }
        """
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "equity_risk_premium": None,
            "erp_percentile": 50.0,
            "bond_yield_10y": None,
            "bond_yield_30y": None,
            "yield_curve_slope": None,
            "equity_implied_yield": None,
            "pe": None,
            "signal": "数据不足，无法判断",
            "ok": False,
        }

        try:
            pe_df = self._fetch_pe_history()
            bond_df = self._fetch_bond_yield()

            if pe_df.empty or bond_df.empty:
                missing = []
                if pe_df.empty:
                    missing.append("PE估值")
                if bond_df.empty:
                    missing.append("国债收益率")
                result["signal"] = f"数据源不可用: {', '.join(missing)}"
                self.cache = result
                self.last_fetch = now
                return result

            latest_pe = float(pe_df["pe"].iloc[-1])
            latest_y10 = float(bond_df["y10"].iloc[-1]) / 100.0  # 百分数转小数
            latest_y30 = float(bond_df["y30"].iloc[-1]) / 100.0

            equity_yield = 1.0 / latest_pe
            erp = equity_yield - latest_y10

            result["pe"] = round(latest_pe, 2)
            result["equity_implied_yield"] = round(equity_yield, 4)
            result["bond_yield_10y"] = round(latest_y10, 4)
            result["bond_yield_30y"] = round(latest_y30, 4)
            result["yield_curve_slope"] = round(latest_y30 - latest_y10, 4)
            result["equity_risk_premium"] = round(erp, 4)

            # ── ERP历史分位：用PE历史序列 + 历史10Y利率重建ERP历史 ──
            # P2-Q23-fix(M257): 原 `reindex(method="nearest").fillna(method="ffill")`
            # 有两个问题：(1) fillna(method=...) 在 pandas 2.3.3 已弃用(FutureWarning)，
            # pandas 3.x 将报错；(2) reindex(nearest) 会把债券窗口(原400天)之外的
            # PE 月份填成首日债券收益率常数段，或返回 NaN 被 dropna 静默丢弃——
            # 实测"10年历史分位"实际样本≈13个月。现改 merge_asof(backward, 45天
            # 容差) 对齐：只取 PE 日期之前最近且真实存在的债券收益率，超出债券
            # 覆盖区间者剔除；配合 BOND_HISTORY_DAYS 拉长到 10 年，分位真正
            # 基于 10 年样本。索引列名统一为"日期"（两帧 set_index 后原名）。
            _pe = pe_df["pe"].rename("pe").reset_index()
            _bond = bond_df["y10"].rename("y10").reset_index()
            aligned = pd.merge_asof(
                _pe.sort_values("日期"),
                _bond.sort_values("日期"),
                on="日期",
                direction="backward",
                tolerance=pd.Timedelta(days=45),
            )
            erp_series = (1.0 / aligned["pe"]) - (aligned["y10"] / 100.0)
            erp_history = erp_series.dropna().values.tolist()
            erp_valid = aligned.loc[aligned["y10"].notna(), "日期"]
            result["erp_percentile"] = self._percentile(erp, erp_history)
            result["erp_history_count"] = len(erp_history)
            result["erp_history_start"] = (
                erp_valid.min().strftime("%Y-%m-%d") if len(erp_valid) else None
            )

            # ── 信号 ──
            pct = result["erp_percentile"]
            if pct >= ERP_HIGH_PCT:
                base = f"ERP处于历史{pct:.0f}%分位,股票相对债券性价比较高"
            elif pct <= ERP_LOW_PCT:
                base = f"ERP处于历史{pct:.0f}%分位,股票相对债券性价比较低,债券更具吸引力"
            else:
                base = f"ERP处于历史{pct:.0f}%分位,股债性价比中性"

            slope = result["yield_curve_slope"]
            if slope is not None:
                if slope < 0.003:
                    base += "; 期限利差(30Y-10Y)偏平坦,债市对长期增长/通胀预期偏谨慎"
                elif slope > 0.008:
                    base += "; 期限利差偏陡峭,债市隐含较强的增长/通胀预期"

            result["signal"] = base
            result["ok"] = True

        except Exception as exc:
            result["error"] = repr(exc)[:200]

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    """CLI 演示：打印当前股债性价比状态。"""
    be = BondEquity()
    result = be.compute()

    print("═" * 65)
    print("  股债性价比 (ERP) 分析")
    print("═" * 65)
    print(f"  数据状态: {'正常' if result.get('ok') else '异常/部分失败'}")
    print(f"  上证PE: {result.get('pe', 'N/A')}")
    print(f"  股票隐含收益率(1/PE): {result.get('equity_implied_yield', 'N/A')}")
    print(f"  10年国债收益率: {result.get('bond_yield_10y', 'N/A')}")
    print(f"  30年国债收益率: {result.get('bond_yield_30y', 'N/A')}")
    print(f"  期限利差(30Y-10Y): {result.get('yield_curve_slope', 'N/A')}")
    print(f"  ERP: {result.get('equity_risk_premium', 'N/A')}")
    print(f"  ERP历史分位: {result.get('erp_percentile', 50)}%"
          f" (基于{result.get('erp_history_count', 0)}个月度样本,"
          f" 起于{result.get('erp_history_start', 'N/A')})")
    print()
    print(f"  📊 信号: {result.get('signal', '')}")
    if result.get("error"):
        print(f"\n  ⚠️ error: {result['error']}")


if __name__ == "__main__":
    main()
