"""
fx_impact — 汇率 vs 行业表现 (V5 跨市场验证)

核心问题：人民币汇率变化，谁受益谁受损？
  1. 拉取美元兑人民币汇率（中行牌价，含央行中间价）近期走势
  2. 判断当前汇率处于"贬值/升值/震荡"哪个阶段
  3. 拉取出口导向 / 进口导向典型行业板块的历史行情
  4. 计算各行业指数收益与汇率变动的 60 日滚动相关性(fx_sensitivity)
  5. 综合给出信号：贬值利好出口板块，升值利好进口板块

数据来源：
  - 汇率: akshare.currency_boc_sina(symbol="美元")  —— 中行人民币牌价，
    "央行中间价"字段最接近官方 USDCNY 中间价（近似值：牌价除以100）。
    备选: akshare.fx_spot_quote()（实时报价，但历史序列缺失，仅用于校验现价）。
  - 行业表现: akshare.stock_board_industry_hist_em(symbol=板块名)

行业分类（出口受益 / 进口受益）用静态映射表，因为 AkShare 没有现成的
"贸易敞口"标签字段，这是行业研究中常见的近似做法（基于行业主营业务的
海外收入占比经验判断），非动态计算。
"""

from __future__ import annotations

import threading as _th
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent

# ── 行业敞口静态映射（经验分类，非动态计算） ──
# exposure: 1.0 = 高度出口敞口, -1.0 = 高度进口/内需敞口, 0 = 中性
EXPORT_ORIENTED_SECTORS = {
    "纺织服装": 0.8,
    "家用电器": 0.7,
    "电子": 0.6,
    "汽车零部件": 0.6,
    "机械设备": 0.5,
    "半导体": 0.5,
    "光伏设备": 0.6,
    "船舶制造": 0.7,
}
IMPORT_ORIENTED_SECTORS = {
    "石油行业": -0.7,
    "化学原料": -0.5,
    "有色金属": -0.4,
    "航空机场": -0.6,
    "钢铁行业": -0.3,
    "造纸印刷": -0.4,
}
SECTOR_EXPOSURE_MAP: dict[str, float] = {**EXPORT_ORIENTED_SECTORS, **IMPORT_ORIENTED_SECTORS}

# P2-Q23-fix(M258): 静态映射中的部分板块名不是东财真实板块名（stock_board_
# industry_hist_em 按东财板块列表取数，名字不对会静默失败跳过）：
#   "家用电器" → "家电行业"；"电子" → "电子元件"；"机械设备" → "通用设备"。
# 运行时用 stock_board_industry_name_em() 动态校验纠正，接口不可用时退回该
# 静态别名表（降级写回 result["notes"]，不静默）。
_SECTOR_ALIASES: dict[str, str] = {
    "家用电器": "家电行业",
    "电子": "电子元件",
    "机械设备": "通用设备",
}

ROLLING_WINDOW = 60  # 滚动相关性窗口(交易日)
FX_HISTORY_DAYS = 400  # 拉取多少天的汇率历史（覆盖60日窗口+缓冲）


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


class FxImpact:
    """汇率对行业表现影响分析引擎。

    核心假设：
      人民币贬值 → 出口企业以人民币计价的海外收入增加 → 出口板块受益；
      人民币升值 → 进口成本下降，内需/进口依赖型板块受益。
      用滚动相关性验证这个假设在当前样本区间是否成立(fx_sensitivity)。

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

    def _fetch_fx_history(self) -> pd.DataFrame:
        """获取美元兑人民币汇率历史（中行牌价）。

        Returns:
            DataFrame(index=日期, columns=["mid_rate"])，
            mid_rate 单位为 USDCNY（例如 7.25），失败返回空 DataFrame。
        """
        end = datetime.now(CST)
        start = end - timedelta(days=FX_HISTORY_DAYS)

        def _do() -> pd.DataFrame | None:
            import akshare as ak
            return ak.currency_boc_sina(
                symbol="美元",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )

        df = _call_with_timeout(_do, timeout=20.0)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        df = df.copy()
        # 央行中间价字段单位是"每100美元对应人民币元"，除以100换算成 USDCNY
        rate_col = "央行中间价" if "央行中间价" in df.columns else None
        if rate_col is None:
            return pd.DataFrame()
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
        df = df.dropna(subset=["日期", rate_col]).sort_values("日期")
        # P2-Q23-fix(M259): 原注释"用前值填充空缺"与实际实现不符——代码并未
        # ffill，NaN 行直接被 dropna 剔除（序列只含交易日，末值即最近工作日
        # 中间价）。若补 ffill 会引入周末零收益行，改变 regime 判定"近1月"的
        # 交易日/日历日语义，故保留交易日序列并修正注释。
        out = df.set_index("日期")[[rate_col]].rename(columns={rate_col: "mid_rate"})
        out["mid_rate"] = out["mid_rate"] / 100.0
        return out

    def _fetch_realtime_spot(self) -> float | None:
        """获取实时 USDCNY 现价（用于校验/补充最新值，历史序列不可用时的兜底）。"""

        def _do() -> float | None:
            import akshare as ak
            df = ak.fx_spot_quote()
            if df is None or df.empty:
                return None
            row = df[df.iloc[:, 0].astype(str).str.contains("USD/CNY", case=False, na=False)]
            if row.empty:
                return None
            for col in ("买报价", "卖报价"):
                if col in row.columns:
                    val = pd.to_numeric(row.iloc[0][col], errors="coerce")
                    if pd.notna(val) and val > 0:
                        return float(val)
            return None

        return _call_with_timeout(_do, timeout=10.0)

    def _fetch_sector_hist(self, sector: str, days: int = 90) -> pd.DataFrame:
        """获取单个行业板块的历史日线。

        Args:
            sector: 板块名称(东方财富行业板块命名，如"电子")
            days: 拉取最近多少天

        Returns:
            DataFrame(index=日期, columns=["close"])，失败返回空 DataFrame。
        """
        end = datetime.now(CST)
        start = end - timedelta(days=days)

        def _do() -> pd.DataFrame | None:
            import akshare as ak
            return ak.stock_board_industry_hist_em(
                symbol=sector,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                period="日k",
                adjust="",
            )

        df = _call_with_timeout(_do, timeout=15.0)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = df.copy()
        date_col = "日期" if "日期" in df.columns else df.columns[0]
        close_col = "收盘" if "收盘" in df.columns else None
        if close_col is None:
            return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)
        return df.set_index(date_col)[[close_col]].rename(columns={close_col: "close"})

    # ────────────────────────── 分析逻辑 ──────────────────────────

    def _resolve_sector_names(self) -> tuple[dict[str, float], list[str]]:
        """把静态行业映射校正为东财真实板块名（P2-Q23-fix M258）。

        用 ak.stock_board_industry_name_em() 拉取东财行业板块名单动态校验：
          1. 已在名单内 → 原样保留
          2. 命中 _SECTOR_ALIASES 静态别名 → 用别名（如"家用电器"→"家电行业"）
          3. 唯一包含/被包含模糊匹配 → 用匹配到的东财名
          4. 无匹配 → 剔除并记 note（不静默跳过）
        名单接口失败（网络抖动）→ 退回静态别名校正并记 note。

        Returns:
            (resolved_map, notes)：resolved_map={东财板块名: exposure}，
            notes 为可见降级/校正说明。
        """
        notes: list[str] = []
        valid: set[str] = set()

        def _do() -> list[str] | None:
            import akshare as ak
            df = ak.stock_board_industry_name_em()
            if df is None or df.empty:
                return None
            # V11 审计修复（High）: df.iloc[:,0] 是"排名"列（1,2,3…）而非板块名称
            # → 所有静态行业名校验失败被剔除，行业敏感度恒空。
            # 修正: 取"板块名称"列（不存在则取含'名称'的列）。
            name_col = next((c for c in df.columns if "名称" in str(c)), None)
            if name_col is None:
                return None
            return df[name_col].astype(str).tolist()

        names = _call_with_timeout(_do, timeout=10.0)
        if names:
            valid = set(names)
        else:
            notes.append("东财板块名单获取失败(stock_board_industry_name_em),"
                         "仅修正已知别名,未做完整性校验")

        resolved: dict[str, float] = {}
        for sector, exposure in SECTOR_EXPOSURE_MAP.items():
            if sector in valid:
                resolved[sector] = exposure
                continue
            alias = _SECTOR_ALIASES.get(sector)
            # 别名命中且（名单可用时）别名确在名单内 → 用别名
            if alias and (not valid or alias in valid):
                resolved[alias] = exposure
                if valid:
                    notes.append(f"板块名[{sector}]→东财名[{alias}]")
                continue
            if not valid:
                # 名单不可用时：无法校验，保留无别名纠正的静态名（沿用原行为），
                # 降级已在 notes 中明示——比静默跳过板块更诚实。
                resolved[sector] = exposure
                continue
            cands = [v for v in valid if sector in v or v in sector]
            if len(cands) == 1:
                resolved[cands[0]] = exposure
                notes.append(f"板块名[{sector}]→东财名[{cands[0]}](模糊匹配)")
                continue
            notes.append(f"板块[{sector}]无匹配东财板块名,已剔除")
        return resolved, notes

    @staticmethod
    def _classify_regime(fx_series: pd.Series) -> tuple[str, float]:
        """判断汇率所处阶段：贬值/升值/震荡。

        Args:
            fx_series: USDCNY 时间序列(升序)

        Returns:
            (regime, change_1m): regime 为 "贬值"/"升值"/"震荡"，
            change_1m 为近1个月(约21个交易日)变动幅度(比例)
        """
        if len(fx_series) < 5:
            return "数据不足", 0.0
        window = min(21, len(fx_series) - 1)
        change_1m = float(fx_series.iloc[-1] / fx_series.iloc[-1 - window] - 1)
        if change_1m > 0.005:
            regime = "贬值"
        elif change_1m < -0.005:
            regime = "升值"
        else:
            regime = "震荡"
        return regime, round(change_1m, 4)

    def _rolling_correlation(
        self, fx_returns: pd.Series, sector_returns: pd.Series, window: int = ROLLING_WINDOW
    ) -> float:
        """计算行业收益与汇率变动的滚动相关性(取最后一个窗口的相关系数)。

        Args:
            fx_returns: 汇率日收益率序列(USDCNY变动, 正=贬值)
            sector_returns: 行业指数日收益率序列
            window: 滚动窗口长度

        Returns:
            float: 相关系数(-1~1)，数据不足返回 0.0
        """
        aligned = pd.concat([fx_returns, sector_returns], axis=1, join="inner").dropna()
        if len(aligned) < min(window, 20):
            return 0.0
        tail = aligned.tail(window)
        try:
            corr = float(tail.iloc[:, 0].corr(tail.iloc[:, 1]))
        except Exception:
            corr = 0.0
        if np.isnan(corr):
            corr = 0.0
        return round(corr, 3)

    # ────────────────────────── 主计算 ──────────────────────────

    def compute(self) -> dict[str, Any]:
        """计算当前汇率状态及其对各行业的影响。

        Returns:
            dict: {
                "timestamp": str,
                "usdcny": float,
                "usdcny_change_1m": float,
                "fx_regime": str,
                "sector_impact": {sector: {"exposure","window_perf","fx_sensitivity"}},
                "signal": str,
                "ok": bool,
            }
        """
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "usdcny": None,
            "usdcny_change_1m": 0.0,
            "fx_regime": "数据不足",
            "sector_impact": {},
            "signal": "数据不足，无法判断",
            "ok": False,
        }

        try:
            fx_df = self._fetch_fx_history()

            if fx_df.empty:
                # 汇率历史失败，退化到只取实时现价
                spot = self._fetch_realtime_spot()
                if spot is not None:
                    result["usdcny"] = round(spot, 4)
                    result["signal"] = "汇率历史数据源失败，仅获取到实时现价"
                self.cache = result
                self.last_fetch = now
                return result

            result["usdcny"] = round(float(fx_df["mid_rate"].iloc[-1]), 4)
            regime, change_1m = self._classify_regime(fx_df["mid_rate"])
            result["fx_regime"] = regime
            result["usdcny_change_1m"] = change_1m

            fx_returns = fx_df["mid_rate"].pct_change().dropna()

            # ── 各行业敏感度分析 ──
            # P2-Q23-fix(M258): 先动态校验/纠正板块名→东财真实板块名
            # （"家用电器"→"家电行业"、"电子"→"电子元件"、"机械设备"→
            # "通用设备"），避免名字不对导致接口静默失败、板块被跳过。
            sector_impact: dict[str, dict[str, Any]] = {}
            sector_exposures, sector_notes = self._resolve_sector_names()
            if sector_notes:
                result["notes"] = sector_notes
            if not sector_exposures:
                result["sector_impact"] = {}
                result["industry_data_available"] = False
            else:
                # 先用一只板块做"探针"：行业板块历史接口在受限网络环境下经常
                # 整体失效(RemoteDisconnected)，如果逐个尝试14个板块、每个等
                # 满超时，总耗时会累积到几分钟。探针失败就直接跳过整个分析，
                # 避免长时间挂起。
                sectors = list(sector_exposures.items())
                canary_sector, canary_exposure = sectors[0]
                canary_df = self._fetch_sector_hist(canary_sector, days=120)
                if canary_df.empty:
                    result["sector_impact"] = {}
                    result["industry_data_available"] = False
                else:
                    result["industry_data_available"] = True
                    remaining = [(canary_sector, canary_exposure, canary_df)] + [
                        (s, e, None) for s, e in sectors[1:]
                    ]
                    for sector, exposure, cached_df in remaining:
                        sec_df = cached_df if cached_df is not None else self._fetch_sector_hist(sector, days=120)
                        if sec_df.empty:
                            continue
                        sec_returns = sec_df["close"].pct_change().dropna()
                        sensitivity = self._rolling_correlation(fx_returns, sec_returns)
                        # P2-Q23-fix(M259): 该收益实为"近120个交易日窗口收益"，
                        # 并非年初至今，改名 window_perf 避免误导。
                        try:
                            window_perf = float(sec_df["close"].iloc[-1] / sec_df["close"].iloc[0] - 1)
                        except Exception:
                            window_perf = 0.0
                        sector_impact[sector] = {
                            "exposure": exposure,
                            "window_perf": round(window_perf, 4),
                            "fx_sensitivity": sensitivity,
                        }
                    result["sector_impact"] = sector_impact

            # ── 信号 ──
            if regime == "贬值":
                base = f"人民币贬值(近1月{change_1m:+.2%}),理论上利好出口板块,关注纺织/电子/机械等出口敞口行业"
            elif regime == "升值":
                base = f"人民币升值(近1月{change_1m:+.2%}),理论上利好进口/内需板块,关注航空/化工等进口依赖行业"
            else:
                base = f"汇率震荡(近1月{change_1m:+.2%}),汇率因子对行业的边际影响有限"

            # 验证：出口板块的 fx_sensitivity 是否确实为正(贬值时上涨)
            # P2-Q23-fix(M258): 板块键可能已被校正为东财名(如"家电行业")，
            # 不能再按原静态键 `k in EXPORT_ORIENTED_SECTORS` 判断，改按
            # exposure>0（出口敞口）过滤。
            export_sens = [
                v["fx_sensitivity"] for v in sector_impact.values()
                if v["exposure"] > 0 and v["fx_sensitivity"] != 0
            ]
            if export_sens:
                avg_export_sens = float(np.mean(export_sens))
                if avg_export_sens > 0.15:
                    base += "; 出口板块与汇率正相关性得到数据验证"
                elif avg_export_sens < -0.15:
                    base += "; 但当前出口板块与汇率呈负相关,理论关系暂未获数据支持"

            result["signal"] = base
            result["ok"] = True

        except Exception as exc:
            result["error"] = repr(exc)[:200]

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    """CLI 演示：打印当前汇率状态及行业影响。"""
    fx = FxImpact()
    result = fx.compute()

    print("═" * 65)
    print("  汇率 vs 行业表现分析")
    print("═" * 65)
    print(f"  数据状态: {'正常' if result.get('ok') else '异常/部分失败'}")
    print(f"  USDCNY: {result.get('usdcny', 'N/A')}")
    print(f"  近1月变动: {result.get('usdcny_change_1m', 0):+.2%}")
    print(f"  汇率状态: {result.get('fx_regime', 'N/A')}")
    print()
    print("  ── 行业敏感度 ──")
    for sector, info in result.get("sector_impact", {}).items():
        print(f"    {sector:<10} 敞口{info['exposure']:+.1f}  "
              f"窗口表现{info['window_perf']:+.2%}  汇率敏感度{info['fx_sensitivity']:+.3f}")
    for note in result.get("notes", []):
        print(f"  ⚠️ {note}")
    print()
    print(f"  📊 信号: {result.get('signal', '')}")
    if result.get("error"):
        print(f"\n  ⚠️ error: {result['error']}")


if __name__ == "__main__":
    main()
