"""
engine.py — 因子计算引擎（Factor Engine）
V4.1 feature

提供：
1. 批量/单因子计算
2. LRU 缓存（内存 + Parquet 二级）
3. 流水线执行
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

import os
import gc
import pandas as pd
import numpy as np
from typing import Optional
from .registry import FactorRegistry, get_default_registry, FactorCategory

logger = __import__('logging').getLogger(__name__)

# 已告警过的 (因子名, 缺失列) —— 避免每次 compute 重复刷日志
_warned_requires: set = set()


def _df_fingerprint(df: pd.DataFrame):
    """V6 fix (Q5 CRITICAL: engine.py:34-35): 缓存键用 df 指纹。

    V5.4 缓存键只有因子名 → 不同日期/不同股票池计算同一因子返回第一次的结果。
    """
    try:
        vals = pd.util.hash_pandas_object(df, index=True)
        return (len(df), hash(tuple(vals.values)), tuple(df.columns))
    except Exception:
        return (len(df), id(df), tuple(df.columns))


class FactorEngine:
    """因子计算引擎"""

    def __init__(self, registry: Optional[FactorRegistry] = None,
                 cache_size: int = 200, parquet_dir: str = ""):
        self.registry = registry or get_default_registry()
        self._cache: dict[tuple, pd.Series] = {}
        self._cache_size = cache_size
        self.parquet_dir = parquet_dir or os.path.expanduser("~/.quant_system/factor_store/")
        os.makedirs(self.parquet_dir, exist_ok=True)

    def compute(self, name: str, df: pd.DataFrame, **kwargs) -> pd.Series:
        """计算单个因子"""
        # V6 fix (Q5 CRITICAL: engine.py:34-35): 缓存键 = (因子名, df 指纹, kwargs)
        # 不同日期/不同股票池不再串数据。
        key = (name, _df_fingerprint(df), frozenset(kwargs.items()))
        if key in self._cache:
            return self._cache[key]

        factor_def = self.registry.get(name)
        if factor_def is None:
            raise ValueError(f"未知因子: {name}")

        # V6 fix (Q5 HIGH: engine.py:41-42): compute 前校验 requires 数据列。
        #   合成因子的 requires 是「子因子名」而非数据列，跳过校验；
        #   其余因子缺失列时仅告警一次（不短路返回 NaN），由实现函数自行降级
        #   （缺失即返回 NaN，可见降级，不崩溃）。
        if factor_def.category != FactorCategory.COMPOSITE:
            missing = [c for c in factor_def.requires if c not in df.columns]
            if missing:
                _key = (name, tuple(missing))
                if _key not in _warned_requires:
                    _warned_requires.add(_key)
                    logger.warning(
                        f"[FactorEngine] {name} 缺少数据列 {missing}（requires={factor_def.requires}），"
                        f"该因子将返回 NaN/部分值（降级可见）"
                    )

        if len(df) < factor_def.min_stocks:
            logger.warning(f"[FactorEngine] {name} 数据不足 {len(df)} < {factor_def.min_stocks}")
            return pd.Series(np.nan, index=df.index)

        params = {**factor_def.parameters, **kwargs}
        result = factor_def.computation_fn(df, **params)
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)

        self._cache[key] = result
        self._evict_if_full()
        return result

    def compute_many(self, names: list[str], df: pd.DataFrame,
                     **kwargs) -> pd.DataFrame:
        """批量计算因子"""
        results = {}
        for name in names:
            try:
                results[name] = self.compute(name, df, **kwargs)
            except Exception as e:
                logger.error(f"[FactorEngine] {name} 计算失败: {e}")
                results[name] = pd.Series(np.nan, index=df.index)
        return pd.DataFrame(results)

    def compute_by_category(self, category: FactorCategory, df: pd.DataFrame,
                            **kwargs) -> pd.DataFrame:
        """按类别计算"""
        names = [f.name for f in self.registry.list_by_category(category)]
        return self.compute_many(names, df, **kwargs)

    def compute_pipeline(self, df: pd.DataFrame,
                         categories: Optional[list[FactorCategory]] = None,
                         **kwargs) -> pd.DataFrame:
        """流水线：按类别分组计算"""
        cats = categories or list(FactorCategory)
        results = []
        for cat in cats:
            r = self.compute_by_category(cat, df, **kwargs)
            results.append(r)
        return pd.concat(results, axis=1)

    def clear_cache(self):
        """清空内存缓存"""
        self._cache.clear()
        gc.collect()

    def save_to_disk(self, name: str, series: pd.Series, date: str):
        """保存因子值到 Parquet"""
        path = os.path.join(self.parquet_dir, f"{name}.parquet")
        df = pd.DataFrame({name: series})
        df.index.name = "date"
        if os.path.exists(path):
            old = pd.read_parquet(path)
            df = pd.concat([old, df])
            # V6 fix (Q5 CRITICAL/LOW: engine.py:93): 全列 drop_duplicates 会误删
            # "值恰好相同的新日期行"；改为按索引（date）去重，保留最新。
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(path)

    def load_from_disk(self, name: str) -> Optional[pd.Series]:
        """从 Parquet 加载因子值"""
        path = os.path.join(self.parquet_dir, f"{name}.parquet")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        return df.iloc[:, 0]

    def list_disk(self) -> list[str]:
        """列出磁盘上已有的因子"""
        files = os.listdir(self.parquet_dir)
        return [f.replace(".parquet", "") for f in files if f.endswith(".parquet")]

    def _evict_if_full(self):
        if len(self._cache) > self._cache_size:
            keys = list(self._cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._cache[k]
