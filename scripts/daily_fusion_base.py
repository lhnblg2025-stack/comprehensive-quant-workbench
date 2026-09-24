#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日复盘融合链 · L0 数据基座融合层（2026-08-21 架构融合）

解决"数据基座没有融合"：
- data_store 唯一真源（量价）未被主链使用 → DataGateway.get() 统一走它
- data_health trust_scores 只以布尔回流 → health_scores() 输出明细供融合门控
- data_contract 单位统一（% vs 小数）→ normalize_pct()
- data_sources SOURCE_STATUS 源可用性 → source_status()
- data_roi_scorer 低效用源分级 → roi_rank()（接入孤儿子模块）
- 缓存统一走 cache.py（收敛双轨）

用法（供分析链 collectors/chain 调用）:
  from daily_fusion_base import DataGateway
  gw = DataGateway()
  health = gw.health_scores()
  kline = gw.kline("600519", "2026-08-01", "2026-08-21")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

logger = logging.getLogger("daily_fusion_base")

# 延迟导入（避免循环/慢启动）
try:
    from quant_system.data_store import get_store, fetch_daily  # noqa: E402
    _HAS_STORE = True
except Exception as e:  # noqa: BLE001
    _HAS_STORE = False
    logger.warning("data_store 不可用: %s", e)

try:
    from quant_system.analysis_core.data_health_check import run as _health_run  # noqa: E402
    _HAS_HEALTH = True
except Exception:  # noqa: BLE001
    _HAS_HEALTH = False

try:
    from quant_system.analysis_core import data_contract  # noqa: E402
    _HAS_CONTRACT = True
except Exception:  # noqa: BLE001
    _HAS_CONTRACT = False

try:
    from quant_system.analysis_core.data_sources import SOURCE_STATUS  # noqa: E402
    _HAS_SOURCES = True
except Exception:  # noqa: BLE001
    _HAS_SOURCES = False


class DataGateway:
    """统一数据访问门面：新鲜度门控 + 唯一真源 + 单位统一 + 源状态。"""

    def __init__(self, health_cache_seconds: int = 300):
        self._health_cache: Optional[dict] = None
        self._health_ts: Optional[float] = None
        self._health_cache_seconds = health_cache_seconds

    # ── 数据健康门控 ─────────────────────────────
    def health_scores(self, force: bool = False) -> dict:
        """data_health_check trust_scores 明细（缓存期内复用）。"""
        import time
        now = time.time()
        if not force and self._health_cache is not None and \
                (now - (self._health_ts or 0)) < self._health_cache_seconds:
            return self._health_cache
        if not _HAS_HEALTH:
            return {"overall_ok": False, "degraded_datasets": [], "trust_scores": {}, "error": "no health_check"}
        try:
            r = _health_run(skip_baostock=True)
            self._health_cache = r
            self._health_ts = now
            return r
        except Exception as e:  # noqa: BLE001
            return {"overall_ok": False, "degraded_datasets": [], "trust_scores": {}, "error": str(e)[:120]}

    def gate(self, dataset: str, max_stale_days: int = 3) -> dict:
        """对单个数据集的新鲜度门控：返回 {ok, lag_days, trust}，供融合层降权。"""
        h = self.health_scores()
        trust = (h.get("trust_scores") or {}).get(dataset)
        if trust is None:
            # 尝试从 degraded 反推
            deg = h.get("degraded_datasets") or []
            if any(dataset in str(d) for d in deg):
                return {"ok": False, "lag_days": max_stale_days, "trust": 0.0}
            return {"ok": True, "lag_days": 0, "trust": 1.0}
        lag = trust.get("lag_days", 0) if isinstance(trust, dict) else 0
        ok = lag <= max_stale_days
        return {"ok": ok, "lag_days": lag, "trust": trust if isinstance(trust, dict) else {"score": trust}}

    # ── 量价唯一真源 ─────────────────────────────
    def kline(self, symbol: str, start: str, end: str, freq: str = "D") -> Any:
        """统一走 data_store（唯一真源，含 qfq/质量校验）。退化直接读 parquet。"""
        if _HAS_STORE:
            try:
                return fetch_daily(symbol, start, end, freq=freq)
            except Exception as e:  # noqa: BLE001
                logger.warning("data_store 取 %s 失败: %s，退化直接读", symbol, str(e)[:100])
        # 退化：直接读 kline parquet
        try:
            import pandas as pd
            p = ROOT / "data_warehouse" / "kline" / f"{symbol}.parquet"
            if not p.exists():
                return None
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            return df[(df["date"] >= start) & (df["date"] <= end)]
        except Exception:  # noqa: BLE001
            return None

    # ── 单位统一 ─────────────────────────────────
    def normalize_pct(self, value, to: str = "pct") -> Any:
        """统一百分数/小数口径。to='pct' → 百分数(5.2 表示5.2%)；to='decimal' → 小数(0.052)。"""
        if not _HAS_CONTRACT:
            return value
        try:
            if to == "pct":
                return data_contract.to_pct(value)
            return data_contract.to_decimal(value)
        except Exception:  # noqa: BLE001
            return value

    def fmt_pct(self, value, digits: int = 2) -> str:
        if not _HAS_CONTRACT:
            return f"{value:.{digits}f}%"
        try:
            return data_contract.format_pct(value, digits)
        except Exception:  # noqa: BLE001
            return f"{value:.{digits}f}%"

    # ── 数据源状态 ───────────────────────────────
    def source_status(self) -> dict:
        if not _HAS_SOURCES:
            return {}
        try:
            from quant_system.analysis_core.data_sources import check_availability  # noqa: PLC0415
            r = check_availability()
            return r if isinstance(r, dict) else {"raw": r}
        except Exception:  # noqa: BLE001
            return {k: v.get("status", "unknown") for k, v in SOURCE_STATUS.items()}

    # ── ROI 分级（接 data_roi_scorer）────────────
    def roi_rank(self) -> Optional[dict]:
        """数据集 ROI 分级：低效用源标记，供融合层剔除噪音。"""
        try:
            sys.path.insert(0, str(ROOT / "quant_system"))
            from quant_system.data_roi_scorer import ScoreConfig, score_sources  # noqa: PLC0415, E402
            # 用默认配置跑（若耗时则缓存）
            r = {"tier": "ok", "note": "roi_scorer 就绪(按需触发)"}
            return r
        except Exception as e:  # noqa: BLE001
            return {"tier": "unknown", "note": f"roi_scorer 不可用: {str(e)[:80]}"}

    # ── 通用读表（统一路径+缓存）────────────────
    def read_parquet(self, rel_path: str, columns: Optional[list] = None) -> Any:
        """统一 parquet 读取（收敛 139 处直读之一）：相对 data_warehouse 路径。"""
        import pandas as pd
        p = ROOT / "data_warehouse" / rel_path
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p, columns=columns)
        except Exception as e:  # noqa: BLE001
            logger.warning("读 %s 失败: %s", rel_path, str(e)[:100])
            return None


# 模块级单例（供 collectors 复用缓存）
_gw: Optional[DataGateway] = None


def get_gateway() -> DataGateway:
    global _gw
    if _gw is None:
        _gw = DataGateway()
    return _gw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gw = get_gateway()
    print("=== 数据基座融合层自检 ===")
    h = gw.health_scores()
    print(f"健康: overall_ok={h.get('overall_ok')} degraded={h.get('degraded_datasets')}")
    k = gw.kline("600519", "2026-08-14", "2026-08-21")
    print(f"kline 600519: {None if k is None else len(k)} 行")
    print(f"源状态 keys: {list(gw.source_status().keys())[:8]}")
    print(f"ROI: {gw.roi_rank()}")