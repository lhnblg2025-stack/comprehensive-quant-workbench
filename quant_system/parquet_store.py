"""
parquet_store.py — V5.1 Parquet 持久化存储层

提供高效的列式存储，替代部分 SQLite 分析功能。
支持:
  1. 因子数据分片存储 (按年月)
  2. 快照版本管理
  3. 快速读取与过滤
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_parquet_store")

CST = timezone(timedelta(hours=8))
PARQUET_ROOT = Path.home() / ".quant_system" / "parquet"


def _ensure_root() -> Path:
    """确保 Parquet 存储根目录存在。"""
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
    return PARQUET_ROOT


def _partition_path(dataset: str, year: int, month: int | None = None) -> Path:
    """获取分区路径: {root}/{dataset}/year={year}/month={month}/"""
    root = _ensure_root()
    if month is not None:
        path = root / dataset / f"year={year}" / f"month={month:02d}"
    else:
        path = root / dataset / f"year={year}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _version_path(dataset: str) -> Path:
    """获取版本文件路径。"""
    return _ensure_root() / f"{dataset}_versions.json"


# ── 写入 ──

def write_factor_data(
    df: pd.DataFrame,
    dataset: str = "factors",
    version_tag: str | None = None,
    partition_by: str = "date",
) -> dict[str, Any]:
    """写入因子数据到 Parquet，自动分区。
    
    Args:
        df: 因子DataFrame (必须含 date 列)
        dataset: 数据集名
        version_tag: 版本标签 (默认 auto)
        partition_by: 分区列
    
    Returns:
        {rows, files, version, path}
    """
    if df.empty:
        return {"rows": 0, "files": 0, "version": "", "path": ""}

    root = _ensure_root()
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # P2-Q1-fix (Q1-M013): 所有分支先 copy，不再原地修改调用方 DataFrame。
    #   无日期列时显式报错（原逻辑静默按「当前年月」分区 → 数据错位风险）。
    df = df.copy()
    if partition_by in df.columns:
        df[partition_by] = pd.to_datetime(df[partition_by])
        df["_year"] = df[partition_by].dt.year
        df["_month"] = df[partition_by].dt.month
    else:
        raise ValueError(
            f"write_factor_data: 缺少分区列 '{partition_by}'（需要含日期的 DataFrame），"
            f"拒绝静默按当前年月写入。实际列: {list(df.columns)}"
        )

    version = version_tag or datetime.now(CST).strftime("%Y%m%d_%H%M%S")

    files_written = []
    for (year, month), group in df.groupby(["_year", "_month"]):
        # P2-Q1-fix (Q1-L019): 复用 _partition_path（此前是死代码，路径在此内联）
        out_path = _partition_path(dataset, year, month) / f"{version}.parquet"

        out_df = group.drop(columns=["_year", "_month"], errors="ignore")
        out_df.to_parquet(str(out_path), index=False, compression="zstd")
        files_written.append(str(out_path))
    
    # 版本记录
    record = {
        "version": version,
        "timestamp": datetime.now(CST).isoformat(),
        "n_rows": len(df),
        "n_files": len(files_written),
        "files": files_written,
        "columns": list(df.columns.drop(["_year", "_month"])),
    }
    
    vpath = _version_path(dataset)
    versions = []
    if vpath.exists():
        try:
            versions = json.loads(vpath.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    versions.append(record)
    # 审计 2026-08-16：版本文件原子写（tmp + os.replace），避免并发读到半截 JSON
    import os
    tmp = vpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(versions, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, vpath)
    
    return {
        "rows": len(df),
        "files": len(files_written),
        "version": version,
        "path": str(dataset_dir),
    }


def read_factor_data(
    dataset: str = "factors",
    version: str | None = None,
    years: list[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """读取因子数据。
    
    Args:
        dataset: 数据集名
        version: 版本 (None=最新)
        years: 年份过滤
        columns: 读取列 (None=全部)
    
    Returns:
        DataFrame
    """
    root = _ensure_root()
    dataset_dir = root / dataset
    
    if not dataset_dir.exists():
        return pd.DataFrame()
    
    # 获取最新版本
    auto_version = version is None  # P2-Q1-fix (Q1-M014): 自动版本次数可回退，显式版本保持严格
    vpath = _version_path(dataset)
    if version is None and vpath.exists():
        try:
            versions = json.loads(vpath.read_text())
            if versions:
                version = versions[-1]["version"]
        except (json.JSONDecodeError, OSError):
            pass

    # 收集月目录（按年份过滤）
    month_dirs: list[Path] = []
    for year_dir in sorted(dataset_dir.glob("year=*")):
        try:
            y = int(year_dir.name.split("=")[1])
        except (ValueError, IndexError):
            continue
        if years and y not in years:
            continue
        month_dirs.extend(sorted(year_dir.glob("month=*")))

    # P2-Q1-fix (Q1-M014): 无 versions.json 时，回退到「全局最新版本」而非各月各自
    #   最新 —— 原逻辑 `pfs[-1]` 按目录取最新，不同月份可能来自不同版本，拼接出
    #   「杂交」数据（有 versions.json 时走精确版本分支，不受影响）。
    if version is None:
        all_names: set[str] = set()
        for md in month_dirs:
            for pf in md.glob("*.parquet"):
                all_names.add(pf.name)
        if all_names:
            version = sorted(all_names)[-1][:-8]  # 去掉 ".parquet" 后缀
            logger.warning("[parquet_store] 无 %s 版本记录，回退到全局最新版本 %s（缺失该版本的月份将被跳过）",
                           vpath.name, version)

    parquet_files = []
    skipped_months = 0
    for month_dir in month_dirs:
        if version:
            pf = month_dir / f"{version}.parquet"
            if pf.exists():
                parquet_files.append(pf)
            elif auto_version:
                # 自动版本在该月缺失 → 跳过该月，避免把旧版本拼进来形成「杂交」数据。
                # 可见降级：调用方应传显式 version 获取该月数据。
                skipped_months += 1
            # 显式/versions.json 版本：严格模式，缺失月份跳过，不混版本
        else:
            # 没有任何 parquet 文件（异常态）
            pfs = sorted(month_dir.glob("*.parquet"))
            if pfs:
                parquet_files.append(pfs[-1])
    if skipped_months:
        logger.warning("[parquet_store] %s: %d 个月份缺少版本 %s，已跳过（避免跨版本拼接）",
                       dataset, skipped_months, version)
    
    if not parquet_files:
        return pd.DataFrame()
    
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(str(pf), columns=columns)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"[parquet_store] 操作失败: {e}", exc_info=True)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def list_versions(dataset: str = "factors") -> list[dict[str, Any]]:
    """列出数据集的所有版本。"""
    vpath = _version_path(dataset)
    if not vpath.exists():
        return []
    try:
        return json.loads(vpath.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def delete_version(dataset: str, version: str) -> bool:
    """删除指定版本。"""
    vpath = _version_path(dataset)
    if not vpath.exists():
        return False
    
    try:
        import os
        versions = json.loads(vpath.read_text())
        versions = [v for v in versions if v["version"] != version]
        tmp = vpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(versions, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, vpath)
        
        # 删除文件
        root = _ensure_root()
        for f in root.glob(f"{dataset}/**/{version}.parquet"):
            f.unlink()
        
        return True
    except (json.JSONDecodeError, OSError):
        return False


# ── CLI ──

def main():
    """测试 Parquet 存储（W2.5 P2：演示写临时目录，不污染生产 PARQUET_ROOT）。"""
    import tempfile
    global PARQUET_ROOT
    PARQUET_ROOT = Path(tempfile.mkdtemp()) / "parquet_demo"
    print(f"Parquet 根目录(临时): {PARQUET_ROOT}")
    
    # 写入测试
    df = pd.DataFrame({
        "date": pd.date_range("2025-06-01", periods=100, freq="B"),
        "600519": np.random.randn(100),
        "000858": np.random.randn(100),
        "300750": np.random.randn(100),
    })
    result = write_factor_data(df, dataset="test_factors")
    print(f"写入: {result}")
    
    # 读取
    loaded = read_factor_data(dataset="test_factors")
    print(f"读取: {loaded.shape}")
    
    # 版本
    versions = list_versions("test_factors")
    print(f"版本数: {len(versions)}")
    
    print("\nOK")


if __name__ == "__main__":
    main()
