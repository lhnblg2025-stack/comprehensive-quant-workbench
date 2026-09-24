"""
analysis_core.common — 公共工具层（重复实现收敛目标）

统一提供 analysis_core 内多份同名同签名工具的唯一实现：
  today / fmt / num / norm_date / signal_view / read_kline_window /
  kline_files / load_names / load_zt_stats

约束：本模块不 import 任何 analysis_core 业务模块（避免循环导入），
只依赖 stdlib + numpy + pandas（pyarrow 可选）。
"""

from __future__ import annotations
import logging

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # noqa: F401
except ImportError:
    pq = None

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace 根
CST = timezone(timedelta(hours=8))
CODE_RE = re.compile(r"^\d{6}$")
KLINE_DIR = ROOT / "data_warehouse" / "kline"
MARKET_DIR = ROOT / "data_warehouse" / "market"


def today() -> str:
    """当前日期（CST）→ YYYY-MM-DD。"""
    return datetime.now(CST).date().isoformat()


def fmt(v, nd: int = 2) -> str:
    """数值格式化：None/非数值/非有限 → "—"，否则保留 nd 位小数。"""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:.{nd}f}" if np.isfinite(f) else "—"


def num(x, default=None):
    """float 转换 + 有限性检查（inf/nan 视为无效→default，避免静默传播上游脏数据）。"""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def norm_date(date: str | None) -> str:
    """日期归一化：空 → today()；YYYYMMDD → YYYY-MM-DD；/ 分隔转 -。"""
    d = (date or today()).strip().replace("/", "-")
    if len(d) == 8 and d.isdigit():
        d = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def signal_view(signal: str) -> str:
    """signal → multi_agent 标准 view 词表（输出：多/空/震荡）。
    '防守' 输入映射为 '空'；未知输入默认 '震荡'。"""
    return {"多": "多", "空": "空", "防守": "空", "看多": "多", "看空": "空",
            "中性": "震荡", "震荡": "震荡"}.get(signal, "震荡")


def read_kline_window(path: Path, cols: list[str], window_start: pd.Timestamp):
    """快速读取K线窗口：pyarrow 直读 + 日期过滤（≈2.4x 快于 pd.read_parquet）；失败回退。"""
    if pq is not None:
        try:
            return pq.read_table(path, columns=cols,
                                 filters=[("date", ">=", window_start)]).to_pandas()
        except Exception as e:
            logging.getLogger(__name__).error(f"[common] 操作失败: {e}", exc_info=True)
    try:
        return pd.read_parquet(path, columns=cols, filters=[("date", ">=", window_start)])
    except Exception:
        return pd.read_parquet(path, columns=cols)


def kline_files(limit: int | None = None, *, kline_dir: Path | None = None) -> list[Path]:
    """全A K线文件（仅 6 位代码 parquet，排除指数/别名文件）；limit 时确定性等距抽样（结果可能略少于 limit）。"""
    kline_dir = kline_dir or KLINE_DIR
    files = sorted(p for p in kline_dir.glob("*.parquet") if CODE_RE.match(p.stem))
    if limit and 0 < limit < len(files):
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    return files


def load_names(market_dir: Path | None = None) -> tuple[dict[str, str], dict[str, bool]]:
    """code -> (name, is_st)。优先 market/stock_names.parquet，回退 stock_name_map.json。market_dir 默认 common.MARKET_DIR。"""
    market_dir = market_dir or MARKET_DIR
    name_map: dict[str, str] = {}
    st_map: dict[str, bool] = {}
    try:
        df = pd.read_parquet(market_dir / "stock_names.parquet")
        if {"code", "name"}.issubset(df.columns):
            df["code"] = df["code"].astype(str).str.zfill(6)
            name_map = dict(zip(df["code"], df["name"].astype(str)))
            if "is_st" in df.columns:
                st_map = dict(zip(df["code"], df["is_st"].astype(bool)))
    except Exception as e:
        logging.getLogger(__name__).error(f"[common] 操作失败: {e}", exc_info=True)
    if not name_map:
        try:
            raw = json.loads((ROOT / "quant_system" / "stock_name_map.json").read_text(encoding="utf-8"))
            for nm, cd in raw.items():
                name_map[str(cd).zfill(6)] = str(nm)
            st_map = {c: ("ST" in n.upper()) for c, n in name_map.items()}
        except Exception as e:
            logging.getLogger(__name__).error(f"[common] 操作失败: {e}", exc_info=True)
    return name_map, st_map


def load_zt_stats(path: Path, date: str) -> pd.DataFrame | None:
    """连板天梯聚合 ≤date，取尾部 60 交易日（behavior/emotion 逐字等价版本）。"""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        hist = df[df["date"] <= pd.Timestamp(date)]
        return hist.tail(60).reset_index(drop=True) if len(hist) else None
    except Exception:  # noqa: BLE001
        return None
