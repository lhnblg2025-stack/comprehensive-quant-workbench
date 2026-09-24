"""
volume_spike.py — 成交量异常扫描 (V5)

核心问题：市场上有哪些股票正在被资金异常关注（放量）？

放量往往先于价格异动出现，是资金动向的早期信号。
本模块扫描全市场（或指定股票池），找出当日成交量 / 20日均量 > 3倍的个股。

对标：涨停敢死队/游资跟踪工具中的"异动量比"扫描
"""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

logger = __import__("logging").getLogger(__name__)

VOLUME_RATIO_THRESHOLD = 3.0
AVG_WINDOW = 20


class VolumeSpike:
    """成交量异常扫描引擎——发现全市场中成交量显著放大的个股。

    Attributes:
        threshold: 量比阈值（当日成交量 / 20日均量），默认3.0
        avg_window: 均量计算窗口，默认20日
    """

    def __init__(self, threshold: float = VOLUME_RATIO_THRESHOLD, avg_window: int = AVG_WINDOW) -> None:
        self.threshold = threshold
        self.avg_window = avg_window

    def _get_stock_list(self, market: str = "A股") -> list[dict[str, str]]:
        """获取待扫描的股票列表（代码+名称）。

        Args:
            market: 市场标识，当前仅支持"A股"（后续可扩展港股/美股）

        Returns:
            [{"code": "600519", "name": "贵州茅台"}, ...]
        """
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is None or "代码" not in spot.columns:
                return []
            return [
                {"code": str(row["代码"]), "name": str(row.get("名称", ""))}
                for _, row in spot.iterrows()
            ]
        except Exception as exc:
            logger.warning("获取股票列表失败: %s", exc)
            return []

    def _check_single_stock(self, code: str, name: str, start_date: str = "") -> dict[str, Any] | None:
        """检查单只股票是否放量异常。

        P1-Q21-fix(H06): 增加 start_date 参数，只拉取近一段时间的历史数据，
        避免每次请求都拉取全量历史（更慢且更易触发东财限流）。

        Returns:
            命中则返回 {code, name, volume_ratio, avg_volume, current_volume}，
            否则返回 None
        """
        try:
            import akshare as ak
            kwargs: dict[str, Any] = {"symbol": code, "period": "daily", "adjust": "qfq"}
            if start_date:
                kwargs["start_date"] = start_date
            df = ak.stock_zh_a_hist(**kwargs)
            if df is None or len(df) < self.avg_window + 1 or "成交量" not in df.columns:
                return None
            volume = df["成交量"].values.astype(float)
            current_volume = float(volume[-1])
            avg_volume = float(np.mean(volume[-self.avg_window - 1:-1]))
            if avg_volume <= 0:
                return None
            ratio = current_volume / avg_volume
            if ratio > self.threshold:
                return {
                    "code": code,
                    "name": name,
                    "volume_ratio": round(ratio, 2),
                    "avg_volume": round(avg_volume, 0),
                    "current_volume": round(current_volume, 0),
                }
        except Exception as exc:
            logger.debug("检查 %s 成交量异常失败: %s", code, exc)
        return None

    def scan(self, market: str = "A股", stock_pool: list[dict[str, str]] | None = None,
             limit: int | None = None, request_interval: float = 0.2,
             history_days: int = 45) -> list[dict[str, Any]]:
        """扫描市场（或给定股票池），返回放量异常个股列表。

        P1-Q21-fix(H06): 修复"默认全市场扫描被静默截断为 spot 前200只"的问题：
          1) limit 默认改为 None=全量，显式截断而非静默；给定 limit 时随机抽样，
             避免只取 spot 列表前 N 只（以 000xxx 深市代码为主）的选择偏差。
          2) 增加 request_interval 节流，避免串行全量请求触发东财限流。
          3) 指定 start_date 只拉近 history_days 个自然日（约 30 个交易日），
             满足 20 日均量窗口即可，显著减少每次请求的数据量。

        Args:
            market: 市场标识，默认"A股"
            stock_pool: 若提供，则只扫描该股票池而非全市场
                （全市场扫描耗时较长，建议实盘场景传入自选/持仓股票池）
            limit: 全市场扫描时的最大股票数上限；None 表示全量（不做静默截断）。
            request_interval: 相邻两次请求的间隔秒数（节流），默认 0.2s。
            history_days: 只拉取最近 N 个自然日的日线（默认 45 天 ≈ 30 个交易日）。

        Returns:
            [{code, name, volume_ratio, avg_volume, current_volume}, ...]
            按 volume_ratio 降序排列
        """
        try:
            pool = stock_pool if stock_pool is not None else self._get_stock_list(market)
            if not pool:
                logger.warning("股票池为空，跳过扫描")
                return []

            # P1-Q21-fix(H06): limit=None=全量；显式 limit 时随机抽样，消除顺序偏差
            if limit is not None and limit > 0 and len(pool) > limit:
                pool = random.sample(pool, limit)
                logger.info("全市场扫描按 limit=%d 随机抽样（从 %d 只中抽取）", limit, len(pool))

            # P1-Q21-fix(H06): 只拉近 history_days 个自然日（约30个交易日），
            # 覆盖 avg_window 均量窗口即可
            start_date = (datetime.now(CST) - timedelta(days=history_days)).strftime("%Y%m%d")

            spikes = []
            for i, item in enumerate(pool):
                hit = self._check_single_stock(item["code"], item["name"], start_date=start_date)
                if hit:
                    spikes.append(hit)
                # P1-Q21-fix(H06): 请求节流，避免触发东财限流
                if request_interval > 0 and i < len(pool) - 1:
                    time.sleep(request_interval)

            spikes.sort(key=lambda x: x["volume_ratio"], reverse=True)
            return spikes
        except Exception as exc:
            logger.error("成交量异常扫描失败: %s", exc)
            return []

    def scan_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """对已获取的成交量矩阵直接计算量比异常（便于单测/离线批量处理）。

        Args:
            df: columns至少含 code, name, volume(当日)，
                以及 avg_volume(20日均量) 或 volume_history(list)

        Returns:
            量比异常列表
        """
        if df is None or df.empty:
            return []
        spikes = []
        for _, row in df.iterrows():
            avg_volume = row.get("avg_volume")
            current_volume = row.get("volume")
            if avg_volume is None or current_volume is None or avg_volume <= 0:
                continue
            ratio = float(current_volume) / float(avg_volume)
            if ratio > self.threshold:
                spikes.append({
                    "code": str(row.get("code", "")),
                    "name": str(row.get("name", "")),
                    "volume_ratio": round(ratio, 2),
                    "avg_volume": round(float(avg_volume), 0),
                    "current_volume": round(float(current_volume), 0),
                })
        spikes.sort(key=lambda x: x["volume_ratio"], reverse=True)
        return spikes


# ════════════════════════════════════════════════════════════════
# main — 测试入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """独立运行测试：用合成 DataFrame 验证量比计算逻辑，不依赖网络。"""
    try:
        detector = VolumeSpike()
        print("=" * 60)
        print("VolumeSpike 成交量异常扫描 — 测试")
        print("=" * 60)

        # 离线单测：合成数据验证量比判定逻辑
        df = pd.DataFrame([
            {"code": "600519", "name": "贵州茅台", "volume": 5_000_000, "avg_volume": 1_000_000},  # ratio=5
            {"code": "000001", "name": "平安银行", "volume": 1_200_000, "avg_volume": 1_000_000},  # ratio=1.2
            {"code": "300750", "name": "宁德时代", "volume": 4_000_000, "avg_volume": 1_000_000},  # ratio=4
        ])
        spikes = detector.scan_dataframe(df)
        print(f"\n离线量比异常扫描结果 ({len(spikes)}只):")
        for s in spikes:
            print(f"  {s['code']} {s['name']} 量比={s['volume_ratio']}x "
                  f"当前量={s['current_volume']:.0f} 均量={s['avg_volume']:.0f}")

        assert len(spikes) == 2, "应识别出2只放量异常股票(ratio>3)"
        assert spikes[0]["code"] == "600519", "应按量比降序排列"

        # 网络场景尝试（若无akshare/无网络自动降级为空列表，不报错）
        try:
            live_spikes = detector.scan(stock_pool=[{"code": "600519", "name": "贵州茅台"}])
            print(f"\n实时扫描测试（1只股票）: {len(live_spikes)} 个异常")
        except Exception as exc:
            print(f"\n实时扫描跳过（环境限制）: {exc}")

        print("\n✅ VolumeSpike 测试通过")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ VolumeSpike 测试失败: {exc}")
        raise


if __name__ == "__main__":
    main()
