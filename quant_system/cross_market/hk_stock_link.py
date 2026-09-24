"""
hk_stock_link — A-H 溢价分析 (V5 跨市场验证)

核心问题：A股相对港股是不是"贵"了？
  1. 拉取全市场 A+H 两地上市公司的实时比价（东方财富 AH 股比价）
  2. 算出全市场平均溢价率，近似恒生 AH 股溢价指数 (HSAHP) 的走势
  3. 把当前平均溢价写入本地历史文件，随着运行次数积累，换算出真实的
     "历史1年分位"（冷启动阶段用正态近似兜底）
  4. 溢价最高的个股列表 + 溢价趋势（20次采样的斜率）+ 信号文本

数据来源优先级：
  1. akshare.stock_zh_ah_spot_em()  — 东方财富 AH 股比价，字段里直接带
     "比价"和"溢价"，是目前最稳的数据源。
  2. akshare.stock_zh_ah_daily(symbol=h_code) / stock_zh_ah_name()
     （对应规格书里提到的 stock_ah_history() / stock_ah_name()，这两个
     旧函数名在新版 akshare 中已经改名，这里做了兼容尝试）作为补充/兜底，
     主要用来在主数据源失效时至少拿到 AH 股名单。

对标：恒生 AH 股溢价指数 (HK: HSAHP)。
"""

from __future__ import annotations
import logging

import json
import threading as _th
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))

# quant_system/ 包根目录
ROOT = Path(__file__).resolve().parent.parent

# 本地历史快照存储：每次 compute() 成功都会 append 一行，
# 用于随着时间积累计算真正的"历史1年分位"（而不是每次都是正态近似）。
DATA_DIR = ROOT / "data" / "cross_market"
HISTORY_FILE = DATA_DIR / "ah_premium_history.jsonl"

# 历史分位计算窗口（近似"1年"，按采样次数而非交易日计，冷启动期用正态近似兜底）
HISTORY_WINDOW_DAYS = 365

# 溢价预警阈值（基于经验：AH溢价长期均值在 25%~35% 附近波动）
PREMIUM_HIGH_PCT = 80  # 分位 > 80% → A股偏贵
PREMIUM_LOW_PCT = 20  # 分位 < 20% → A股偏便宜（港股偏贵）


def _call_with_timeout(fn: Callable[[], Any], timeout: float = 15.0) -> Any:
    """在独立线程里跑一个可能阻塞很久的网络请求，超时则放弃。

    AkShare 的很多接口在海外/受限网络环境下会长时间挂起（而不是抛异常），
    直接调用可能导致整条 compute() 卡死。用线程 + join(timeout) 兜底。

    Args:
        fn: 无参数可调用对象，内部自己处理异常
        timeout: 最长等待秒数

    Returns:
        fn() 的返回值；超时或异常返回 None
    """
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


class HkStockLink:
    """A-H 股溢价分析引擎。

    核心假设：
      恒生AH股溢价指数长期围绕均值波动（均值回归特征明显）。
      溢价处于历史高分位 → A股相对港股偏贵 → 边际上更应该谨慎；
      溢价处于历史低分位 → A股相对港股偏便宜（或港股偏贵）→ 边际上更从容。

    Attributes:
        cache: 上一次 compute() 结果缓存
        last_fetch: 上次成功计算的时间戳
        cache_ttl: 缓存有效期（秒）
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0.0
        self.cache_ttl = cache_ttl
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.getLogger(__name__).error(f"[hk_stock_link] 操作失败: {e}", exc_info=True)

    # ────────────────────────── 数据获取 ──────────────────────────

    def _fetch_ah_spot(self) -> pd.DataFrame:
        """获取全市场 AH 股实时比价（主数据源）。

        Returns:
            DataFrame，至少包含 [名称, H股代码, A股代码, 最新价-RMB,
            最新价-HKD, 比价, 溢价] 列；失败返回空 DataFrame。
        """

        def _do() -> pd.DataFrame | None:
            import akshare as ak
            return ak.stock_zh_ah_spot_em()

        df = _call_with_timeout(_do, timeout=15.0)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        return df

    def _fetch_ah_name_fallback(self) -> pd.DataFrame:
        """主数据源失效时的兜底：仅拿 AH 股名单（无比价数据）。

        兼容规格书中提到的旧函数名 stock_ah_name() / stock_ah_history()，
        新版 akshare 里对应的是 stock_zh_ah_name() / stock_zh_ah_daily()。

        Returns:
            DataFrame [代码, 名称]，失败返回空 DataFrame。
        """

        def _do() -> pd.DataFrame | None:
            import akshare as ak
            for fn_name in ("stock_ah_name", "stock_zh_ah_name"):
                fn = getattr(ak, fn_name, None)
                if fn is None:
                    continue
                try:
                    return fn()
                except Exception as e:
                    logging.getLogger(__name__).error(f"[hk_stock_link] 操作失败: {e}", exc_info=True)
                    continue
            return None

        df = _call_with_timeout(_do, timeout=20.0)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        return df

    # ────────────────────────── 历史快照 ──────────────────────────

    def _append_history(self, avg_premium: float, ah_premium_index: float) -> None:
        """把当次采样结果追加到本地历史文件（jsonl），供未来分位计算使用。

        P2-Q23-fix(M269): 同日去重——当日已有采样则跳过追加。原实现每次
        compute() 都 append，同一自然日多次计算会重复采样，按"采样次数"稀释
        "历史1年分位"（采样次数 ≠ 交易日数，周末/盘中多次刷新都会被算成样本）。
        """
        today = datetime.now(CST).strftime("%Y-%m-%d")
        if any(str(r.get("timestamp", ""))[:10] == today for r in self._load_history()):
            return
        record = {
            "timestamp": datetime.now(CST).isoformat(),
            "avg_premium": avg_premium,
            "ah_premium_index": ah_premium_index,
        }
        try:
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.getLogger(__name__).error(f"[hk_stock_link] 操作失败: {e}", exc_info=True)

    def _load_history(self) -> list[dict[str, Any]]:
        """读取近 HISTORY_WINDOW_DAYS 天内的历史快照（按日期去重，保留每日首条）。"""
        if not HISTORY_FILE.exists():
            return []
        cutoff = datetime.now(CST) - timedelta(days=HISTORY_WINDOW_DAYS)
        records: list[dict[str, Any]] = []
        seen_days: set[str] = set()
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ts = datetime.fromisoformat(rec["timestamp"])
                        if ts >= cutoff:
                            day = rec["timestamp"][:10]
                            if day in seen_days:
                                continue
                            seen_days.add(day)
                            records.append(rec)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[hk_stock_link] 操作失败: {e}", exc_info=True)
                        continue
        except Exception:
            return []
        return records

    @staticmethod
    def _percentile(value: float, history: list[float]) -> float:
        """value 在 history 序列中的百分位 (0~100)。

        样本不足 10 个时，用正态近似兜底（假设溢价服从均值25%/标准差12%的
        近似分布，这是A股历史AH溢价的粗略经验区间），避免冷启动阶段直接
        返回无意义的 50。
        """
        if len(history) >= 10:
            arr = np.array(history, dtype=float)
            less = np.sum(arr < value)
            equal = np.sum(arr == value)
            return round(float((less + 0.5 * equal) / len(arr) * 100), 1)
        # 冷启动兜底：正态近似 (经验均值25%, 标准差12%)
        z = (value - 25.0) / 12.0
        pct = 0.5 * (1 + _erf_approx(z / np.sqrt(2)))
        return round(float(max(0.0, min(100.0, pct * 100))), 1)

    @staticmethod
    def _trend_slope(history: list[float]) -> float:
        """最近若干次采样的溢价趋势斜率（简单线性回归斜率，单位：百分点/次采样）。"""
        if len(history) < 3:
            return 0.0
        y = np.array(history[-20:], dtype=float)
        x = np.arange(len(y), dtype=float)
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            slope = 0.0
        return round(slope, 4)

    # ────────────────────────── 主计算 ──────────────────────────

    def compute(self, top_n: int = 10) -> dict[str, Any]:
        """计算当前 A-H 溢价状态。

        Args:
            top_n: 返回溢价最高的前 N 只个股

        Returns:
            dict: {
                "timestamp": str,
                "ah_premium_index": float,   # 全市场加权/简单均值溢价(近似HSAHP), 单位%
                "premium_stocks": [{"code","name","a_price","h_price","premium","hist_percentile"}],
                "avg_premium": float,        # 简单平均溢价率, 单位%
                "premium_percentile": float, # 溢价指数的历史(采样)分位
                "trend_slope": float,        # 溢价趋势斜率
                "signal": str,
                "ok": bool,
                "sample_count": int,         # 参与统计的AH股数量
            }
        """
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "ah_premium_index": 0.0,
            "premium_stocks": [],
            "avg_premium": 0.0,
            "premium_percentile": 50.0,
            "trend_slope": 0.0,
            "signal": "数据不足，无法判断",
            "ok": False,
            "sample_count": 0,
        }

        try:
            df = self._fetch_ah_spot()

            if df.empty:
                # 主数据源失效，退化为仅报告股票名单（无溢价数据）
                names_df = self._fetch_ah_name_fallback()
                result["sample_count"] = 0 if names_df.empty else len(names_df)
                result["signal"] = "AH比价数据源暂时不可用，仅获取到股票名单"
                # P2-Q23-fix(M269): 降级后无溢价数据，显式标注（不静默返回空溢价）
                result["note"] = "主数据源(东财AH比价)不可用,降级为股票名单,无溢价/分位数据"
                self.cache = result
                self.last_fetch = now
                return result

            # 清洗：溢价/比价列转数值，剔除异常值
            df = df.copy()
            for col in ("溢价", "比价", "最新价-RMB", "最新价-HKD"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["溢价"])
            # 溢价字段是比例形式(例如0.28代表28%)，异常值(<-0.9 或 >5)剔除
            df = df[(df["溢价"] > -0.9) & (df["溢价"] < 5.0)]

            if df.empty:
                result["signal"] = "AH比价数据清洗后为空"
                self.cache = result
                self.last_fetch = now
                return result

            premium_pct = df["溢价"] * 100.0  # 转换为百分比数值
            avg_premium = float(premium_pct.mean())
            # 近似 HSAHP：以全市场AH股的简单平均溢价作为指数代理
            # （真实HSAHP按流通市值加权且只覆盖成分股，这里没有市值数据，
            #  用简单平均近似，量级和方向是一致的，仅数值有偏差）
            ah_premium_index = round(avg_premium, 2)

            result["avg_premium"] = round(avg_premium, 2)
            result["ah_premium_index"] = ah_premium_index
            result["sample_count"] = int(len(df))

            # ── 个股溢价排行 + 横截面分位 ──
            sorted_df = df.sort_values("溢价", ascending=False)
            top_df = sorted_df.head(top_n)
            cross_section = premium_pct.values.tolist()
            stocks: list[dict[str, Any]] = []
            for _, row in top_df.iterrows():
                p = float(row["溢价"]) * 100.0
                stocks.append({
                    "code": str(row.get("A股代码", "")),
                    "h_code": str(row.get("H股代码", "")),
                    "name": str(row.get("名称", "")),
                    "a_price": float(row.get("最新价-RMB", float("nan"))) if pd.notna(row.get("最新价-RMB")) else None,
                    "h_price": float(row.get("最新价-HKD", float("nan"))) if pd.notna(row.get("最新价-HKD")) else None,
                    "premium": round(p / 100.0, 4),
                    # 该股票溢价在当前全市场AH股中的横截面分位
                    "hist_percentile": self._percentile(p, cross_section),
                })
            result["premium_stocks"] = stocks

            # ── 时间序列分位（依赖本地积累的历史快照） ──
            # P2-Q23-fix(M269): 先读历史再追加本次采样——原顺序是先 append 再
            # load，本次当值被计入自己的历史，"分位含当前值自身"。改为先读后写，
            # 本次当值作为被排名对象、不进入历史序列；同日去重见 _append_history。
            history_records = self._load_history()
            hist_values = [r["ah_premium_index"] for r in history_records]
            result["premium_percentile"] = self._percentile(ah_premium_index, hist_values)
            result["trend_slope"] = self._trend_slope(hist_values)
            result["history_sample_count"] = len(hist_values)
            # P2-Q23-fix(M269): 明示分位口径——样本≥10 为经验分位，否则为冷启动
            # 正态近似兜底（均值25%/标准差12% 的经验参数），避免把近似当真实分位。
            result["percentile_method"] = (
                "empirical" if len(hist_values) >= 10 else "normal_approx_cold_start"
            )
            # P2-Q23-fix(M269): 幸存者偏差明示——只覆盖当前仍两地上市的AH股，
            # 已退市/私有化的个股不在样本内。
            result["note"] = "AH溢价仅覆盖当前仍两地上市的AH股,存在幸存者偏差"
            self._append_history(result["avg_premium"], ah_premium_index)

            # ── 信号 ──
            pct = result["premium_percentile"]
            slope = result["trend_slope"]
            if pct >= PREMIUM_HIGH_PCT:
                base = f"溢价处于历史{pct:.0f}%分位,A股相对港股明显偏贵,注意回调风险"
            elif pct <= PREMIUM_LOW_PCT:
                base = f"溢价处于历史{pct:.0f}%分位,A股相对港股偏便宜(或港股偏贵)"
            else:
                base = f"溢价处于历史{pct:.0f}%分位,处于中性区间"
            if slope > 0.3:
                trend_txt = ",且溢价呈扩张趋势"
            elif slope < -0.3:
                trend_txt = ",且溢价呈收缩趋势"
            else:
                trend_txt = ""
            result["signal"] = base + trend_txt
            result["ok"] = True

        except Exception as exc:
            result["error"] = repr(exc)[:200]

        self.cache = result
        self.last_fetch = now
        return result


def _erf_approx(x: float) -> float:
    """误差函数的 Abramowitz-Stegun 近似 (足够用于正态分位近似，避免依赖 scipy)。"""
    # 常数来自 A&S 7.1.26
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def main() -> None:
    """CLI 演示：打印当前 A-H 溢价状态。"""
    link = HkStockLink()
    result = link.compute()

    print("═" * 65)
    print("  A-H 股溢价分析")
    print("═" * 65)
    print(f"  数据状态: {'正常' if result.get('ok') else '异常/部分失败'}")
    print(f"  样本数量: {result.get('sample_count', 0)} 只AH股")
    print(f"  平均溢价: {result.get('avg_premium', 0):.2f}%")
    print(f"  溢价指数(近似HSAHP): {result.get('ah_premium_index', 0):.2f}")
    print(f"  历史分位: {result.get('premium_percentile', 50):.1f}%"
          f"  (基于{result.get('history_sample_count', 0)}次本地采样,"
          f"口径:{result.get('percentile_method', '?')})")
    print(f"  趋势斜率: {result.get('trend_slope', 0):+.3f}")
    if result.get("note"):
        print(f"  ℹ️ {result['note']}")
    print()
    print(f"  📊 信号: {result.get('signal', '')}")
    print()
    print("  ── 溢价最高的AH股 ──")
    for s in result.get("premium_stocks", [])[:10]:
        print(f"    {s['name']:<10} A:{s.get('code','-'):<8} "
              f"溢价{s['premium']:+.1%}  (横截面分位{s['hist_percentile']:.0f}%)")
    if result.get("error"):
        print(f"\n  ⚠️ error: {result['error']}")


if __name__ == "__main__":
    main()
