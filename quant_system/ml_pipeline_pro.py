"""
DEPRECATED (2026-08-07): 无生产引用，保留仅供参考。主用模块见 项目文档/量化交易系统/系统梳理报告.md

ml_pipeline_pro.py — ML 研发管道增强版

本模块面向量化研究中的机器学习研发闭环，提供从特征持久化、模型注册、
自动重训练、模型解释、超参搜索到顶层训练/预测入口的一体化实现。

V4.1 feature: FeatureStorePro
V4.1 feature: ModelRegistry
V4.1 feature: AutoRetrainer
V4.1 feature: ModelExplainer
V4.1 feature: HyperparameterOptimizer
V4.1 feature: MLEngineV2

设计原则
--------
1. 研究可复现: 特征和模型都带版本、时间戳、参数和指标。
2. 避免泄漏: 超参搜索默认使用时间序列切分；可通过 GroupKFold 避免组间泄漏。
3. 可降级依赖: pandas/numpy/sklearn 为基础依赖；SHAP、LightGBM、XGBoost 为可选依赖。
4. 文件透明: 所有元数据以 JSON 保存，特征以 Parquet 优先保存，缺失 parquet engine 时降级 pickle。
5. 工程保守: 顶层引擎优先使用已有注册表和特征存储，不隐藏重要假设。

注意
----
本文件刻意保留较完整的中文注释，方便后续量化研究员直接审计研发流程。
生产环境接入前，建议在项目测试集中加入以下覆盖:
  - 特征保存/加载/版本回放
  - 模型注册/提升/回滚
  - 时间序列 CV 与 GroupKFold 参数搜索
  - LightGBM/XGBoost 缺失时的 sklearn 降级路径
"""

from __future__ import annotations
import logging

import hashlib
import itertools
import json
import math
import os
import pickle
import re
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # sklearn 是本模块的基础依赖；try/except 让导入错误更易读。
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import partial_dependence as sk_partial_dependence
    from sklearn.inspection import permutation_importance as sk_permutation_importance
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV, TimeSeriesSplit
    from sklearn.model_selection._split import BaseCrossValidator
except Exception as exc:  # pragma: no cover - 环境缺依赖时才触发
    BaseEstimator = object  # type: ignore[assignment]
    CalibratedClassifierCV = None  # type: ignore[assignment]
    GridSearchCV = None  # type: ignore[assignment]
    GroupKFold = None  # type: ignore[assignment]
    RandomForestClassifier = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]
    RandomizedSearchCV = None  # type: ignore[assignment]
    TimeSeriesSplit = None  # type: ignore[assignment]
    BaseCrossValidator = object  # type: ignore[assignment]
    clone = None  # type: ignore[assignment]
    sk_partial_dependence = None  # type: ignore[assignment]
    sk_permutation_importance = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    balanced_accuracy_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    log_loss = None  # type: ignore[assignment]
    mean_absolute_error = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    r2_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR: Exception | None = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:  # V4.1 feature: LightGBM 可选依赖。
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - 可选依赖
    LGBMClassifier = None  # type: ignore[assignment]
    LGBMRegressor = None  # type: ignore[assignment]

try:  # V4.1 feature: XGBoost 可选依赖。
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - 可选依赖
    XGBClassifier = None  # type: ignore[assignment]
    XGBRegressor = None  # type: ignore[assignment]

try:  # V4.1 feature: SHAP 可选依赖。
    import shap  # type: ignore
except Exception:  # pragma: no cover - 可选依赖
    shap = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 全局常量与基础工具
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path.home() / ".quant_system"
DEFAULT_FEATURE_DIR = DEFAULT_ROOT / "ml_features_pro"
DEFAULT_MODEL_DIR = DEFAULT_ROOT / "ml_registry_pro"
DEFAULT_RETRAIN_DIR = DEFAULT_ROOT / "ml_retrain_pro"

JsonDict = dict[str, Any]
TaskType = Literal["classification", "regression"]


def _utc_now() -> str:
    """返回统一的 UTC ISO 时间戳，避免本地时区差异影响元数据比对。"""
    # P2-Q7-fix (L505): datetime.utcnow() 在 Python 3.12+ 弃用，改用 timezone.utc。
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_sklearn() -> None:
    """在需要 sklearn 的方法入口显式报错，避免后续出现难懂的 NoneType 错误。"""
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError("ml_pipeline_pro 需要 scikit-learn 才能执行该操作") from _SKLEARN_IMPORT_ERROR


def _safe_name(name: str) -> str:
    """将任意特征名/模型名转换成文件系统安全名称。

    V4.1 feature: 名称既要对人可读，也要能兼容中文、空格和特殊字符。
    这里保留 ASCII 字母数字和常用分隔符，其他字符用下划线替换；如果替换后
    信息过少，再追加短哈希，避免不同中文名称都落到同一个路径。

    P1-Q7-fix: 用 sha1(raw).hexdigest()[:7] 确定性哈希替代 abs(hash(raw)) % 10_000_000。
    Python 内置 hash() 受 PYTHONHASHSEED 随机化影响，同一中文名在不同进程得到不同目录，
    落盘路径跨进程不可复现，按文件名恢复（legacy 候选路径）永远找不到别的进程写的文件。
    sha1 派生哈希跨进程稳定；截断时预留哈希长度，保证长名称截断后完整哈希仍在目录名中，
    避免两个共享 120 字符前缀的长 ASCII 名碰撞到同一目录。
    """
    raw = str(name).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not cleaned:
        cleaned = "item"
    # 确定性哈希: sha1 前 7 位 hex，跨进程/跨机器一致。
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:7]
    if cleaned != raw or len(cleaned) > 120:
        # 先拼哈希再截断: 为后缀预留空间，确保哈希始终保留在最终目录名中。
        suffix = f"_{digest}"
        limit = max(1, 120 - len(suffix))
        return f"{cleaned[:limit]}{suffix}"
    return cleaned


def _json_default(obj: Any) -> Any:
    """JSON 序列化辅助函数，覆盖 numpy/pandas/Path 等常见研究对象。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    # P2-Q7-fix (L509): np.bool_ 以前落到 str(obj) → JSON 里布尔值变 "True"/"False"，
    # 元数据比对时类型漂移；np.complex 显式转成 JSON 安全结构。
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return {str(k): _json_default(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="list")
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _read_json(path: Path, default: JsonDict | None = None) -> JsonDict:
    """安全读取 JSON；文件不存在或损坏时返回 default 的浅拷贝。"""
    if default is None:
        default = {}
    if not path.exists():
        return dict(default)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else dict(default)
    except Exception as exc:
        warnings.warn(f"读取 JSON 失败: {path} ({exc})")
        return dict(default)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """原子写入 JSON，避免进程中断留下半截元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=_json_default)
    tmp.replace(path)


def _normalize_date_range(date_range: Any) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """把各种日期范围表达统一成 `(start, end)`。

    支持:
      - None: 不过滤
      - (start, end) / [start, end]
      - slice(start, stop)
      - 单个日期字符串: 仅加载该日
    """
    if date_range is None:
        return None, None
    if isinstance(date_range, slice):
        start = pd.Timestamp(date_range.start) if date_range.start is not None else None
        end = pd.Timestamp(date_range.stop) if date_range.stop is not None else None
        return start, end
    if isinstance(date_range, (tuple, list)):
        if len(date_range) != 2:
            raise ValueError("date_range 元组/列表必须是 (start, end)")
        start = pd.Timestamp(date_range[0]) if date_range[0] is not None else None
        end = pd.Timestamp(date_range[1]) if date_range[1] is not None else None
        return start, end
    day = pd.Timestamp(date_range)
    return day, day


def _filter_date_index(series: pd.Series, date_range: Any) -> pd.Series:
    """按日期范围过滤 Series，非日期索引会尽力转换。"""
    start, end = _normalize_date_range(date_range)
    if start is None and end is None:
        return series

    out = series.copy()
    try:
        out.index = pd.to_datetime(out.index)
    except Exception:
        # 如果索引不可转日期，交给 pandas 做普通比较会更混乱，因此直接报错。
        raise ValueError("date_range 过滤要求特征索引可转换为日期")

    mask = pd.Series(True, index=out.index)
    if start is not None:
        mask &= out.index >= start
    if end is not None:
        mask &= out.index <= end
    return out.loc[mask]


def _coerce_series(series: pd.Series | Mapping[Any, Any] | Sequence[Any], name: str) -> pd.Series:
    """把输入转换成 float Series，同时保留日期索引。"""
    if isinstance(series, pd.Series):
        out = series.copy()
    else:
        out = pd.Series(series)
    out.name = name
    out = pd.to_numeric(out, errors="coerce")
    try:
        out.index = pd.to_datetime(out.index)
    except Exception as e:
        # 截面特征也可能用股票代码索引；此时不强制日期化。
        logging.getLogger(__name__).error(f"[ml_pipeline_pro] 操作失败: {e}", exc_info=True)
    return out.sort_index()


def _write_frame_prefer_parquet(df: pd.DataFrame, path: Path) -> JsonDict:
    """优先写 Parquet，失败时降级 pickle。

    V4.1 feature: 用户要求 Parquet 持久化；但量化研究环境里 pyarrow/fastparquet
    不一定已安装。为了让管道在基础依赖下仍可运行，这里保留降级路径，同时在
    元数据里记录实际格式和错误原因。生产部署只需安装 pyarrow 即会写 Parquet。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=True)
        return {"storage_format": "parquet", "path": str(path), "parquet_error": None}
    except Exception as exc:
        fallback = path.with_suffix(".pkl")
        df.to_pickle(fallback)
        warnings.warn(f"Parquet 写入失败，已降级 pickle: {path} ({exc})")
        return {"storage_format": "pickle", "path": str(fallback), "parquet_error": repr(exc)}


def _read_frame_auto(path: Path, storage_format: str | None = None) -> pd.DataFrame:
    """按元数据格式读取 DataFrame，同时兼容历史 Parquet/ pickle 文件。"""
    if storage_format == "pickle" or path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if storage_format == "parquet" or path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            fallback = path.with_suffix(".pkl")
            if fallback.exists():
                return pd.read_pickle(fallback)
            raise
    # 兼容未记录格式的旧文件。
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_pickle(path)
    fallback = path.with_suffix(".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(path)


def _parse_name_version(text: str) -> tuple[str, str | None]:
    """解析 `name@version`、`name:v001` 等简写。"""
    raw = str(text)
    if "@" in raw:
        name, version = raw.rsplit("@", 1)
        return name, version or None
    if ":v" in raw:
        name, version = raw.rsplit(":", 1)
        return name, version or None
    return raw, None


def _version_sort_key(version: str) -> tuple[int, str]:
    """对 v001_20260101 这类版本号做稳定排序。"""
    match = re.match(r"v?(\d+)", str(version))
    number = int(match.group(1)) if match else -1
    return number, str(version)


def _is_classifier_target(y: pd.Series | np.ndarray) -> bool:
    """根据标签形态推断分类/回归。

    量化里二分类标签常常是 0/1 浮点，三分类标签可能是 -1/0/1 或 0/1/2。
    这里采用保守规则: 唯一值较少且接近整数，则视为分类。
    """
    arr = pd.Series(y).dropna()
    if arr.empty:
        return False
    unique = pd.unique(arr)
    if len(unique) <= 20:
        numeric = pd.to_numeric(pd.Series(unique), errors="coerce")
        if numeric.notna().all() and np.allclose(numeric, np.round(numeric)):
            return True
        if arr.dtype == object or str(arr.dtype).startswith("category"):
            return True
    return False


def _infer_task_type(y: pd.Series | np.ndarray, explicit: str | None = None) -> TaskType:
    """推断模型任务类型，允许调用方显式覆盖。"""
    if explicit in {"classification", "regression"}:
        return explicit  # type: ignore[return-value]
    return "classification" if _is_classifier_target(y) else "regression"


def _make_2d_frame(X: pd.DataFrame | pd.Series | Mapping[str, Any] | np.ndarray, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """把预测/解释输入统一成二维 DataFrame。"""
    if isinstance(X, pd.DataFrame):
        out = X.copy()
    elif isinstance(X, pd.Series):
        out = X.to_frame().T
    elif isinstance(X, Mapping):
        out = pd.DataFrame([dict(X)])
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = pd.DataFrame(arr, columns=list(columns) if columns is not None else None)
    if columns is not None:
        missing = [c for c in columns if c not in out.columns]
        for col in missing:
            out[col] = np.nan
        out = out[list(columns)]
    return out


def _fill_numeric_frame(X: pd.DataFrame, medians: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """机器学习前的轻量缺失值处理: 转数值、用中位数填补。

    审计 2026-08-16：支持传入训练期保存的 medians（dict），预测/评估时
    复用训练 imputer，避免用当前/测试数据自身中位数引入分布泄漏。
    """
    out = X.apply(pd.to_numeric, errors="coerce")
    if medians is not None:
        med = pd.Series({c: medians.get(str(c)) for c in out.columns}, dtype=float)
        med = med.replace([np.inf, -np.inf], np.nan)
    else:
        med = out.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return out


def _align_target_to_frame(y: Any, X: pd.DataFrame, *, context: str) -> pd.Series:
    """把 y 与 X 显式对齐，避免 `pd.Series(y, index=X.index)` 在索引不一致时静默错位。

    P2-Q7-fix (M497): 原代码多处用 `pd.Series(y, index=X.index)` / `y.reindex(X.index)`，
    当 y 是带不同索引的 Series 时会产生全 NaN 或按位置错位，随后 dropna 得到空集
    （walk-forward 静默返回空结果）或 permutation_importance 报 NaN 错误。这里统一：
      - Series: 行数必须与 X 一致；索引不一致时按位置对齐并告警（请调用方确认行序）。
      - array: 长度必须与 X 行数一致。
    对齐后由调用方自行 dropna 并同步过滤 X。
    """
    n_rows = len(X)
    if isinstance(y, pd.DataFrame):
        raise TypeError(f"{context}: y 不能是 DataFrame，请传入 Series/array")
    if isinstance(y, pd.Series):
        if len(y) != n_rows:
            raise ValueError(
                f"{context}: y 长度 {len(y)} 与 X 行数 {n_rows} 不一致，请确认输入行序一致"
            )
        if not y.index.equals(X.index):
            warnings.warn(
                f"{context}: y 索引与 X 索引不一致，已按位置对齐（请确认行序一致）", RuntimeWarning
            )
        return pd.Series(np.asarray(y).reshape(-1), index=X.index, name=getattr(y, "name", None))
    arr = np.asarray(y).reshape(-1)
    if len(arr) != n_rows:
        raise ValueError(
            f"{context}: y 长度 {len(arr)} 与 X 行数 {n_rows} 不一致，请确认输入行序一致"
        )
    return pd.Series(arr, index=X.index, name="target")


def _frame_date_values(index: pd.Index) -> np.ndarray | None:
    """返回与行一一对应的日期值数组；索引不是日期结构时返回 None。

    P2-Q7-fix (M496): 用于按旧模型训练日期截断验证集，避免新旧模型评估尺度不一致
    （旧模型在验证集上为样本内、新模型为样本外导致的系统性对比偏置）。
    """
    if isinstance(index, pd.DatetimeIndex):
        return pd.to_datetime(index).to_numpy()
    if isinstance(index, pd.MultiIndex) and "date" in index.names:
        return pd.to_datetime(pd.Series(index.get_level_values("date"))).to_numpy()
    if isinstance(index, pd.RangeIndex):
        return None  # 位置索引不是日期，无法判断时序先后
    try:
        return pd.to_datetime(index).to_numpy()
    except Exception:
        return None


def _predict_numeric(model: Any, X: pd.DataFrame) -> np.ndarray:
    """返回可用于排序/解释的数值预测。"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        arr = np.asarray(proba)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return arr[:, -1].astype(float)
        return arr.reshape(-1).astype(float)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float).reshape(-1)


def _score_predictions(task_type: TaskType, y_true: pd.Series | np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> JsonDict:
    """统一计算分类/回归指标。"""
    _ensure_sklearn()
    y = pd.Series(y_true).reset_index(drop=True)
    metrics: JsonDict = {"task_type": task_type, "n_samples": int(len(y))}
    if task_type == "classification":
        labels = np.asarray(pred)
        # Q7-fix: 空样本/空折时 accuracy_score 直接抛异常（同函数 f1/roc_auc 均有保护），
        # 补上 try/except 并标记 n_samples=0。
        # P2-Q7-fix (M501): 新版 sklearn 对空数组返回 nan（不抛异常），这里显式判空，
        # 保证空样本时 accuracy=None 且 n_samples=0，避免 nan 被当作有效指标。
        if len(y) == 0 or len(labels) == 0:
            metrics["accuracy"] = None
            metrics["n_samples"] = 0
        else:
            try:
                metrics["accuracy"] = float(accuracy_score(y, labels))
            except Exception:
                metrics["accuracy"] = None
                metrics["n_samples"] = 0
        # P2-Q7-fix (M500): 类别不平衡下 accuracy 无意义，补充平衡准确率。
        try:
            if balanced_accuracy_score is not None:
                metrics["balanced_accuracy"] = float(balanced_accuracy_score(y, labels))
        except Exception:
            metrics["balanced_accuracy"] = None
        try:
            metrics["f1"] = float(f1_score(y, labels, average="weighted"))
        except Exception:
            metrics["f1"] = None
        if proba is not None:
            try:
                p = np.asarray(proba)
                metrics["log_loss"] = float(log_loss(y, p))
            except Exception:
                metrics["log_loss"] = None
            try:
                p = np.asarray(proba)
                if p.ndim == 2 and p.shape[1] > 1:
                    metrics["roc_auc"] = float(roc_auc_score(y, p[:, -1]))
            except Exception:
                metrics["roc_auc"] = None
            # P2-Q7-fix (M500): 类别不平衡下补充 PR-AUC（更关注少数类排序质量）。
            try:
                p = np.asarray(proba)
                if p.ndim == 2 and p.shape[1] > 1 and average_precision_score is not None:
                    metrics["pr_auc"] = float(average_precision_score(y, p[:, -1]))
            except Exception:
                metrics["pr_auc"] = None
    else:
        values = np.asarray(pred, dtype=float)
        metrics["mse"] = float(mean_squared_error(y, values))
        metrics["rmse"] = float(math.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(y, values))
        try:
            metrics["r2"] = float(r2_score(y, values))
        except Exception:
            metrics["r2"] = None
    return metrics


def _primary_metric(metrics: Mapping[str, Any]) -> tuple[str, float, bool]:
    """选择模型对比的主指标。

    返回 `(metric_name, value, higher_is_better)`。
    分类优先 roc_auc/f1/accuracy，回归优先 r2，其次负 rmse。
    """
    for key in ("roc_auc", "f1", "accuracy", "ic", "mean_ic"):
        value = metrics.get(key)
        if value is not None and not pd.isna(value):
            return key, float(value), True
    if metrics.get("r2") is not None and not pd.isna(metrics.get("r2")):
        return "r2", float(metrics["r2"]), True
    for key in ("log_loss", "rmse", "mse", "mae"):
        value = metrics.get(key)
        if value is not None and not pd.isna(value):
            return key, float(value), False
    return "score", float(metrics.get("score", 0.0) or 0.0), True


# ---------------------------------------------------------------------------
# 数据类: 元数据 schema
# ---------------------------------------------------------------------------


@dataclass
class FeatureVersionMeta:
    """单个特征版本的元数据。

    V4.1 feature: 特征不仅保存值，还保存计算时间、半衰期、版本和研究元数据。
    """

    name: str
    version: int
    path: str
    storage_format: str
    computed_at: str
    half_life: float | None = None
    category: str | None = None
    creator: str | None = None
    formula: str | None = None
    ic_history: Any = None
    is_active: bool = True
    n_obs: int = 0
    start: str | None = None
    end: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class ModelVersionMeta:
    """单个模型版本的元数据。"""

    name: str
    version: str
    model_path: str
    params: JsonDict
    metrics: JsonDict
    training_date: str
    task_type: str | None = None
    is_production: bool = False
    notes: str | None = None


# ---------------------------------------------------------------------------
# FeatureStorePro — 特征存储增强版
# ---------------------------------------------------------------------------


class FeatureStorePro:
    """增强版特征存储。

    V4.1 feature: 这是 FactorStore 思路的 ML 特征版本。与按日期保存横截面因子的
    FactorStore 不同，FeatureStorePro 按 `特征名/版本` 保存完整时间序列，便于
    在模型训练时按特征集合做对齐、回放和重要性评估。

    存储布局示例::

        ~/.quant_system/ml_features_pro/
          catalog.json
          momentum_20d_1234567/
            v001.parquet
            v001.meta.json
            v002.parquet
            v002.meta.json

    Parquet 表结构::

        index: 日期或原始索引
        columns:
          value        - 特征值
          computed_at  - 计算时间
          half_life    - 半衰期，可空
          version      - 整数版本号
          feature_name - 原始特征名
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_FEATURE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.base_dir / "catalog.json"
        self._catalog: JsonDict = self._load_catalog()

    # ------------------------------------------------------------------
    # 内部目录/目录表工具
    # ------------------------------------------------------------------

    def _load_catalog(self) -> JsonDict:
        """读取特征目录；不存在时创建空目录结构。"""
        catalog = _read_json(self.catalog_path, {"schema_version": "4.1", "features": {}})
        catalog.setdefault("schema_version", "4.1")
        catalog.setdefault("features", {})
        return catalog

    def _save_catalog(self) -> None:
        """写回特征目录。"""
        self._catalog["updated_at"] = _utc_now()
        _write_json(self.catalog_path, self._catalog)

    def _feature_dir(self, name: str) -> Path:
        """返回某个特征的安全目录。"""
        return self.base_dir / _safe_name(name)

    def _feature_path(self, name: str, version: int) -> Path:
        """返回某个特征版本的 Parquet 路径。"""
        return self._feature_dir(name) / f"v{version:03d}.parquet"

    def _meta_path(self, name: str, version: int) -> Path:
        """返回某个特征版本的元数据路径。"""
        return self._feature_dir(name) / f"v{version:03d}.meta.json"

    def _next_version(self, name: str, metadata: Mapping[str, Any] | None = None) -> int:
        """决定新版本号；metadata 可显式传入 version。"""
        if metadata:
            explicit = metadata.get("version") or metadata.get("版本号")
            if explicit is not None:
                if isinstance(explicit, str) and explicit.lower().startswith("v"):
                    explicit = explicit[1:]
                return int(explicit)
        item = self._catalog.get("features", {}).get(name, {})
        versions = item.get("versions", {}) if isinstance(item, dict) else {}
        if not versions:
            return 1
        ints = []
        for value in versions.keys():
            try:
                ints.append(int(str(value).lstrip("v")))
            except Exception as e:
                logging.getLogger(__name__).error(f"[ml_pipeline_pro] 操作失败: {e}", exc_info=True)
                continue
        return (max(ints) + 1) if ints else 1

    def _resolve_version_meta(self, name: str, version: str | int | None = None) -> FeatureVersionMeta:
        """按名称和版本查找元数据，同时兼容旧目录结构。"""
        feature_item = self._catalog.get("features", {}).get(name)
        if not feature_item:
            # 兼容历史文件: base_dir/{safe_name}.parquet 或 feature_dir/latest.parquet。
            legacy_candidates = [
                self.base_dir / f"{_safe_name(name)}.parquet",
                self.base_dir / f"{_safe_name(name)}.pkl",
                self._feature_dir(name) / "latest.parquet",
                self._feature_dir(name) / "latest.pkl",
            ]
            for candidate in legacy_candidates:
                if candidate.exists():
                    return FeatureVersionMeta(
                        name=name,
                        version=0,
                        path=str(candidate),
                        storage_format="pickle" if candidate.suffix == ".pkl" else "parquet",
                        computed_at="unknown",
                    )
            raise KeyError(f"未找到特征: {name}")

        versions = feature_item.get("versions", {})
        if version is None or str(version).lower() in {"latest", "last"}:
            version = feature_item.get("latest_version")
        elif str(version).lower() in {"active", "production"}:
            active = [v for v, m in versions.items() if m.get("is_active", True)]
            version = sorted(active, key=_version_sort_key)[-1] if active else feature_item.get("latest_version")
        if version is None:
            raise KeyError(f"特征 {name} 没有可用版本")

        key = str(version).lstrip("v")
        if key not in versions:
            # 兼容目录里用 v001 做键的情况。
            vkey = f"v{int(key):03d}" if key.isdigit() else str(version)
            if vkey in versions:
                key = vkey
            else:
                raise KeyError(f"特征 {name} 未找到版本 {version}")
        meta = versions[key]
        return FeatureVersionMeta(**meta)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def save_feature(self, name: str, series: pd.Series, metadata: dict | None = None) -> int:
        """保存一个特征序列。

        Parameters
        ----------
        name:
            特征名。可以是中文或包含特殊字符，目录层会自动安全转义。
        series:
            特征值序列。索引通常是日期，也可以是截面索引；值会转为数值。
        metadata:
            可选元数据，支持字段:
              - category / 类别
              - creator / 创建者
              - formula / 计算公式
              - ic_history / IC历史
              - half_life / 半衰期
              - is_active
              - version

        Returns
        -------
        int
            写入的版本号。
        """
        metadata = dict(metadata or {})
        version = self._next_version(name, metadata)
        computed_at = metadata.get("computed_at") or metadata.get("计算时间") or _utc_now()
        half_life = metadata.get("half_life", metadata.get("半衰期"))
        category = metadata.get("category", metadata.get("类别"))
        creator = metadata.get("creator", metadata.get("创建者"))
        formula = metadata.get("formula", metadata.get("计算公式"))
        ic_history = metadata.get("ic_history", metadata.get("IC历史"))
        is_active = bool(metadata.get("is_active", metadata.get("active", True)))

        values = _coerce_series(series, name)
        frame = pd.DataFrame(
            {
                "value": values,
                "computed_at": computed_at,
                "half_life": float(half_life) if half_life is not None and half_life != "" else np.nan,
                "version": int(version),
                "feature_name": name,
            }
        )
        frame.index.name = values.index.name or "date"

        path = self._feature_path(name, version)
        storage = _write_frame_prefer_parquet(frame, path)
        actual_path = Path(storage["path"])

        clean_metadata = dict(metadata)
        for key in ["version", "版本号", "computed_at", "计算时间"]:
            clean_metadata.pop(key, None)

        start = str(values.index.min()) if len(values) else None
        end = str(values.index.max()) if len(values) else None
        version_meta = FeatureVersionMeta(
            name=name,
            version=version,
            path=str(actual_path),
            storage_format=str(storage["storage_format"]),
            computed_at=str(computed_at),
            half_life=float(half_life) if half_life is not None and half_life != "" else None,
            category=str(category) if category is not None else None,
            creator=str(creator) if creator is not None else None,
            formula=str(formula) if formula is not None else None,
            ic_history=ic_history,
            is_active=is_active,
            n_obs=int(values.notna().sum()),
            start=start,
            end=end,
            metadata=clean_metadata,
        )

        # V4.1 feature: 每个版本单独落一份 meta，便于目录损坏时恢复。
        _write_json(self._meta_path(name, version), asdict(version_meta))

        features = self._catalog.setdefault("features", {})
        item = features.setdefault(
            name,
            {
                "name": name,
                "safe_name": _safe_name(name),
                "created_at": _utc_now(),
                "versions": {},
            },
        )
        item["latest_version"] = int(version)
        item["updated_at"] = _utc_now()
        item["category"] = version_meta.category
        item["creator"] = version_meta.creator
        item["formula"] = version_meta.formula
        item["is_active"] = is_active
        item.setdefault("versions", {})[str(version)] = asdict(version_meta)
        self._save_catalog()
        return version

    def load_feature(self, name: str, date_range: Any = None) -> pd.Series:
        """加载特征序列。

        Parameters
        ----------
        name:
            特征名；也支持 `feature@v2`、`feature@2` 这样的版本简写。
        date_range:
            日期范围，支持 None、(start, end)、slice(start, end) 或单个日期。

        Returns
        -------
        pd.Series
            特征值序列，名称为原始特征名。
        """
        base_name, version_spec = _parse_name_version(name)
        if version_spec is not None and version_spec.lower().startswith("v"):
            version_spec = version_spec[1:]
        meta = self._resolve_version_meta(base_name, version_spec)
        frame = _read_frame_auto(Path(meta.path), meta.storage_format)

        if "value" in frame.columns:
            series = frame["value"].copy()
        elif frame.shape[1] == 1:
            series = frame.iloc[:, 0].copy()
        else:
            # 历史宽表兼容: 优先同名列，否则取第一列。
            series = frame[base_name].copy() if base_name in frame.columns else frame.iloc[:, 0].copy()
        series.name = base_name
        return _filter_date_index(series, date_range)

    def list_features(self, category: str | None = None, is_active: bool | None = None) -> pd.DataFrame:
        """列出特征目录。

        Parameters
        ----------
        category:
            仅返回某一类别；None 表示不过滤。
        is_active:
            仅返回活跃/非活跃特征；None 表示不过滤。
        """
        rows: list[JsonDict] = []
        for name, item in self._catalog.get("features", {}).items():
            latest = item.get("latest_version")
            latest_meta = item.get("versions", {}).get(str(latest), {}) if latest is not None else {}
            row = {
                "name": name,
                "safe_name": item.get("safe_name"),
                "category": item.get("category") or latest_meta.get("category"),
                "creator": item.get("creator") or latest_meta.get("creator"),
                "formula": item.get("formula") or latest_meta.get("formula"),
                "latest_version": latest,
                "is_active": bool(item.get("is_active", latest_meta.get("is_active", True))),
                "n_versions": len(item.get("versions", {})),
                "n_obs": latest_meta.get("n_obs"),
                "start": latest_meta.get("start"),
                "end": latest_meta.get("end"),
                "computed_at": latest_meta.get("computed_at"),
                "half_life": latest_meta.get("half_life"),
                "ic_history": latest_meta.get("ic_history"),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if category is not None:
            df = df[df["category"].astype(str) == str(category)]
        if is_active is not None:
            df = df[df["is_active"] == bool(is_active)]
        return df.sort_values(["category", "name"], na_position="last").reset_index(drop=True)

    def feature_importance(self, feature_names: Sequence[str], target: pd.Series | str) -> pd.Series:
        """计算特征预测重要性。

        V4.1 feature: 使用 RandomForest 的 impurity importance 做快速筛选。
        对金融研究而言，这只是第一层过滤；正式上线前仍建议结合置换重要性和
        样本外检验，避免相关特征的替代效应误导。
        """
        _ensure_sklearn()
        if not feature_names:
            return pd.Series(dtype=float, name="importance")

        feature_frames: list[pd.Series] = []
        for feature in feature_names:
            try:
                feature_frames.append(self.load_feature(feature).rename(feature))
            except KeyError:
                warnings.warn(f"跳过未找到特征: {feature}")
        if not feature_frames:
            return pd.Series(dtype=float, name="importance")

        X = pd.concat(feature_frames, axis=1)
        y = self.load_feature(target) if isinstance(target, str) else pd.Series(target).rename("target")
        aligned = X.join(y.rename("target"), how="inner").dropna(subset=["target"])
        if aligned.empty:
            return pd.Series(dtype=float, name="importance")

        X_train = _fill_numeric_frame(aligned[list(X.columns)])
        y_train = aligned["target"]
        task_type = _infer_task_type(y_train)
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        importance = pd.Series(model.feature_importances_, index=X_train.columns, name="importance")
        return importance.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# ModelRegistry — 模型注册表
# ---------------------------------------------------------------------------


class ModelRegistry:
    """模型注册表。

    V4.1 feature: 注册表负责模型文件、参数、训练日期、指标和生产版本指针。
    每次 register 都创建不可变版本；promote 只改变生产标记，不重写模型 pickle。
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_MODEL_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.base_dir / "model_catalog.json"
        self._catalog: JsonDict = self._load_catalog()

    def _load_catalog(self) -> JsonDict:
        catalog = _read_json(self.catalog_path, {"schema_version": "4.1", "models": {}})
        catalog.setdefault("schema_version", "4.1")
        catalog.setdefault("models", {})
        return catalog

    def _save_catalog(self) -> None:
        self._catalog["updated_at"] = _utc_now()
        _write_json(self.catalog_path, self._catalog)

    def _model_dir(self, name: str) -> Path:
        return self.base_dir / _safe_name(name)

    def _next_version(self, name: str) -> str:
        item = self._catalog.get("models", {}).get(name, {})
        versions = item.get("versions", {}) if isinstance(item, dict) else {}
        nums: list[int] = []
        for version in versions.keys():
            match = re.match(r"v(\d+)", str(version))
            if match:
                nums.append(int(match.group(1)))
        seq = max(nums) + 1 if nums else 1
        # P2-Q7-fix (L505): datetime.utcnow() 弃用（3.12+），改用 timezone.utc。
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"v{seq:03d}_{stamp}"

    def _resolve_model_meta(self, name: str, version: str = "latest") -> ModelVersionMeta:
        base_name, embedded_version = _parse_name_version(name)
        if embedded_version:
            name = base_name
            version = embedded_version

        item = self._catalog.get("models", {}).get(name)
        if not item:
            raise KeyError(f"未找到模型: {name}")
        versions = item.get("versions", {})
        if not versions:
            raise KeyError(f"模型 {name} 没有可用版本")

        request = str(version or "latest").lower()
        if request in {"latest", "last"}:
            key = item.get("latest_version") or sorted(versions, key=_version_sort_key)[-1]
        elif request in {"production", "prod", "live"}:
            key = item.get("production_version")
            if key is None:
                promoted = [v for v, m in versions.items() if m.get("is_production")]
                key = sorted(promoted, key=_version_sort_key)[-1] if promoted else None
            if key is None:
                # P2-Q7-fix (M494): 原实现静默回退到 latest（is_production=False），
                # 导致 train() 的"无生产版本则自动 promote"成为死代码、delete_version 的
                # 生产保护与 summarize_model_registry 的生产模型报表全部失真（报表显示
                # "无生产模型"但预测实际在服务最新版）。这里直接抛 KeyError，由调用方
                # 决定回退（train 自动 promote / predict 显式回退 latest）。
                raise KeyError(f"模型 {name} 没有生产版本（production_version 未设置且无 is_production 版本）")
        else:
            key = version
        if key not in versions:
            # P2-Q7-fix (L510): 支持 name@v003 短版本前缀匹配 v003_* 的最新版本，
            # 与 FeatureStorePro 的短版本语义保持一致，避免必然 KeyError。
            candidates = {str(key), str(key).lstrip("v"), f"v{str(key).lstrip('v')}"}
            candidates.discard("")
            matches = [v for v in versions if any(str(v).startswith(c) for c in candidates)]
            if matches:
                key = sorted(matches, key=_version_sort_key)[-1]
            else:
                raise KeyError(f"模型 {name} 未找到版本 {version}")
        return ModelVersionMeta(**versions[key])

    def register(self, name: str, model_obj: Any, params: dict, metrics: dict) -> str:
        """注册模型版本。

        Parameters
        ----------
        name:
            模型逻辑名称，例如 `alpha_lgb_5d`。
        model_obj:
            可 pickle 的模型对象。
        params:
            训练参数、特征列表、标签名等。
        metrics:
            样本外绩效指标。

        Returns
        -------
        str
            新版本号。
        """
        version = str(params.get("version") or self._next_version(name))
        # P2-Q7-fix (M499): 版本号唯一性保护 —— 同名版本再注册会静默覆盖模型文件与
        # catalog 记录（无告警），破坏回滚能力；显式指定已存在版本时报错。
        existing = self._catalog.get("models", {}).get(name, {}).get("versions", {})
        if version in existing:
            raise ValueError(
                f"模型 {name} 已存在版本 {version}，不允许重复注册覆盖；"
                "请去掉显式 version 参数以自动编号，或使用新的版本号"
            )
        model_dir = self._model_dir(name)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{version}.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(model_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

        task_type = params.get("task_type") or metrics.get("task_type")
        meta = ModelVersionMeta(
            name=name,
            version=version,
            model_path=str(model_path),
            params=dict(params),
            metrics=dict(metrics),
            training_date=str(params.get("training_date") or _utc_now()),
            task_type=str(task_type) if task_type is not None else None,
            is_production=bool(params.get("is_production", False)),
            notes=params.get("notes"),
        )

        models = self._catalog.setdefault("models", {})
        item = models.setdefault(
            name,
            {
                "name": name,
                "safe_name": _safe_name(name),
                "created_at": _utc_now(),
                "versions": {},
            },
        )
        item["latest_version"] = version
        item["updated_at"] = _utc_now()
        item["task_type"] = meta.task_type
        item.setdefault("versions", {})[version] = asdict(meta)
        if meta.is_production:
            for vmeta in item["versions"].values():
                vmeta["is_production"] = False
            item["versions"][version]["is_production"] = True
            item["production_version"] = version
        self._save_catalog()
        return version

    def load(self, name: str, version: str = "latest") -> Any:
        """加载模型对象。"""
        meta = self._resolve_model_meta(name, version)
        path = Path(meta.model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as fh:
            return pickle.load(fh)

    def metadata(self, name: str, version: str = "latest") -> ModelVersionMeta:
        """返回模型元数据。"""
        return self._resolve_model_meta(name, version)

    def list_models(self, task_type: str | None = None) -> pd.DataFrame:
        """列出模型目录。"""
        rows: list[JsonDict] = []
        for name, item in self._catalog.get("models", {}).items():
            for version, meta in item.get("versions", {}).items():
                row = {
                    "name": name,
                    "version": version,
                    "task_type": meta.get("task_type") or item.get("task_type"),
                    "training_date": meta.get("training_date"),
                    "is_production": bool(meta.get("is_production", False)),
                    "model_path": meta.get("model_path"),
                }
                for key, value in (meta.get("metrics") or {}).items():
                    row[f"metric_{key}"] = value
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if task_type is not None:
            df = df[df["task_type"].astype(str) == str(task_type)]
        return df.sort_values(["name", "version"]).reset_index(drop=True)

    def promote(self, name: str, version: str) -> bool:
        """将某个版本提升为生产模型。"""
        item = self._catalog.get("models", {}).get(name)
        if not item or version not in item.get("versions", {}):
            return False
        for vmeta in item.get("versions", {}).values():
            vmeta["is_production"] = False
        item["versions"][version]["is_production"] = True
        item["production_version"] = version
        item["updated_at"] = _utc_now()
        self._save_catalog()
        return True

    def compare_models(self, names: list[str]) -> pd.DataFrame:
        """对比多个模型的最新/指定版本绩效。"""
        rows: list[JsonDict] = []
        for spec in names:
            model_name, version = _parse_name_version(spec)
            meta = self._resolve_model_meta(model_name, version or "latest")
            row = {
                "name": meta.name,
                "version": meta.version,
                "task_type": meta.task_type,
                "training_date": meta.training_date,
                "is_production": meta.is_production,
            }
            for key, value in meta.params.items():
                if key in {"feature_names", "selected_feature_names"}:
                    row[f"param_{key}_count"] = len(value) if isinstance(value, list) else None
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    row[f"param_{key}"] = value
            for key, value in meta.metrics.items():
                row[f"metric_{key}"] = value
            metric_name, metric_value, higher = _primary_metric(meta.metrics)
            row["primary_metric"] = metric_name
            row["primary_value"] = metric_value
            row["higher_is_better"] = higher
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        higher = bool(df["higher_is_better"].iloc[0]) if "higher_is_better" in df else True
        return df.sort_values("primary_value", ascending=not higher).reset_index(drop=True)

    def delete_version(self, name: str, version: str, remove_file: bool = False) -> bool:
        """删除非生产版本；默认只删目录记录，不碰模型文件。

        V4.1 feature: 研发目录可能有很多实验版本，但生产版本需要保护。
        """
        item = self._catalog.get("models", {}).get(name)
        if not item or version not in item.get("versions", {}):
            return False
        meta = item["versions"][version]
        if meta.get("is_production"):
            raise ValueError("不能删除生产版本，请先 promote 其他版本")
        if remove_file and meta.get("model_path"):
            path = Path(meta["model_path"])
            if path.exists():
                path.unlink()
        del item["versions"][version]
        versions = item.get("versions", {})
        item["latest_version"] = sorted(versions, key=_version_sort_key)[-1] if versions else None
        self._save_catalog()
        return True


# ---------------------------------------------------------------------------
# AutoRetrainer — 自动重训练器
# ---------------------------------------------------------------------------


class AutoRetrainer:
    """自动重训练器。

    V4.1 feature: 支持固定周期和 IC 衰减触发。retrain 方法采用保守策略:
      - 新模型必须在验证集主指标上不差于旧模型超过阈值；
      - 新模型更差时不会 promote，生产版本自动保持旧版本；
      - 所有重训练尝试会写日志，便于追踪研究试验次数。
    """

    def __init__(self, registry: ModelRegistry | None = None, base_dir: str | Path | None = None) -> None:
        self.registry = registry if registry is not None else ModelRegistry()
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_RETRAIN_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.schedule_path = self.base_dir / "schedule.json"
        self.log_path = self.base_dir / "retrain_log.json"
        self.schedule = self._load_schedule()

    def _load_schedule(self) -> JsonDict:
        data = _read_json(
            self.schedule_path,
            {
                "frequency_days": 20,
                "metric_threshold": 0.02,
                "rolling_window": 20,
                "last_retrain_at": None,
            },
        )
        data.setdefault("frequency_days", 20)
        data.setdefault("metric_threshold", 0.02)
        data.setdefault("rolling_window", 20)
        data.setdefault("last_retrain_at", None)
        return data

    def _save_schedule(self) -> None:
        _write_json(self.schedule_path, self.schedule)

    def _append_log(self, record: Mapping[str, Any]) -> None:
        log = _read_json(self.log_path, {"records": []})
        log.setdefault("records", []).append(dict(record))
        _write_json(self.log_path, log)

    def set_schedule(self, frequency_days: int = 20, metric_threshold: float = 0.02) -> None:
        """设置重训练条件。

        Parameters
        ----------
        frequency_days:
            固定周期触发间隔。
        metric_threshold:
            IC 或主指标相对滚动均值的衰减阈值。
        """
        if frequency_days <= 0:
            raise ValueError("frequency_days 必须为正数")
        if metric_threshold < 0:
            raise ValueError("metric_threshold 不能为负数")
        self.schedule["frequency_days"] = int(frequency_days)
        self.schedule["metric_threshold"] = float(metric_threshold)
        self.schedule["updated_at"] = _utc_now()
        self._save_schedule()

    def check_needs_retrain(self, current_ic: float, history: pd.Series) -> bool:
        """检查是否需要重训练。

        触发条件:
          1. 距离上次重训练超过 fixed frequency；
          2. 当前 IC 低于最近 N 期均值 - threshold；
          3. 最近 N 期均值低于更长窗口均值 - threshold。
        """
        threshold = float(self.schedule.get("metric_threshold", 0.02))
        window = int(self.schedule.get("rolling_window", self.schedule.get("frequency_days", 20)))
        history = pd.Series(history).dropna().astype(float)

        # 固定周期触发: 没有 last_retrain_at 时，不强制触发，避免首次运行误判。
        periodic_trigger = False
        last_retrain = self.schedule.get("last_retrain_at")
        if last_retrain:
            try:
                # P2-Q7-fix (L505): datetime.utcnow() 弃用；且存储的时间戳是 UTC（.replace("Z","")
                # 得到 naive），补上 timezone.utc 后再与 aware 的 now 比较，避免 TypeError。
                last_dt = datetime.fromisoformat(str(last_retrain).replace("Z", "")).replace(tzinfo=timezone.utc)
                periodic_trigger = datetime.now(timezone.utc) - last_dt >= timedelta(days=int(self.schedule["frequency_days"]))
            except Exception:
                periodic_trigger = False

        if history.empty:
            return periodic_trigger

        recent_mean = float(history.tail(window).mean())
        current_trigger = float(current_ic) < recent_mean - threshold

        long_trigger = False
        if len(history) >= window * 2:
            long_mean = float(history.tail(window * 2).head(window).mean())
            long_trigger = recent_mean < long_mean - threshold
        return bool(periodic_trigger or current_trigger or long_trigger)

    def _split_new_data(self, X: pd.DataFrame, y: pd.Series, validation_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """按时间顺序切分训练/验证集，避免随机打乱造成时间泄漏。"""
        if not 0 < validation_size < 0.8:
            validation_size = 0.2
        n = len(X)
        if n < 5:
            # P2-Q7-fix (M495): 原实现 n=1 时 split=max(1,0)=1 → split>=n → split=0 →
            # 空训练集 → fit 抛异常；n<5 时验证集过小，指标无统计意义。
            # 显式报错并要求调用方提供更多数据，避免样本内指标冒充验证指标。
            raise ValueError(
                f"_split_new_data: 样本数过少({n})，至少需要 5 个样本才能做有意义的"
                "训练/验证切分，请提供更多数据"
            )
        split = max(1, int(n * (1.0 - validation_size)))
        if split >= n:
            split = n - 1
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> JsonDict:
        """评估模型，分类时尽量记录 predict_proba 指标。"""
        pred = model.predict(X)
        proba = model.predict_proba(X) if task_type == "classification" and hasattr(model, "predict_proba") else None
        return _score_predictions(task_type, y, pred, proba)

    def retrain(self, model_name: str, new_data: Any) -> dict:
        """执行重训练并在新模型更差时自动回滚。

        Parameters
        ----------
        model_name:
            注册表中的模型名称。
        new_data:
            推荐传入 dict，字段包括:
              - X: pd.DataFrame
              - y: pd.Series / array
              - model_class: 可选，新模型类；缺省时 clone 旧模型
              - params: 可选，新模型参数
              - validation_size: 验证集比例
              - task_type: classification/regression
              - trainer: 可选 callable，签名 trainer(X_train, y_train, **params) -> model
        """
        _ensure_sklearn()
        if not isinstance(new_data, Mapping):
            raise TypeError("new_data 需要是包含 X/y 的 dict-like 对象")
        if "X" not in new_data or "y" not in new_data:
            raise ValueError("new_data 必须包含 X 和 y")

        # P2-Q7-fix (M494): 无生产版本时 metadata("production") 现在抛 KeyError，
        # 不再静默回退 latest —— 首次重训练直接走"训练并 promote"路径。
        try:
            old_meta = self.registry.metadata(model_name, "production")
            old_model = self.registry.load(model_name, old_meta.version)
        except KeyError:
            old_meta = None
            old_model = None

        X = _fill_numeric_frame(pd.DataFrame(new_data["X"]))
        # P2-Q7-fix (M497): 显式对齐 y 与 X（行数校验 + 索引告警），避免
        # pd.Series(y, index=X.index) 在索引不一致时静默错位/全 NaN。
        y = _align_target_to_frame(new_data["y"], X, context="AutoRetrainer.retrain")
        y = y.dropna()
        X = X.loc[y.index]
        if len(y) == 0:
            raise ValueError("AutoRetrainer.retrain: 对齐后无有效样本，请检查 y 与 X 的索引")
        # P2-Q7-fix (M496 关联): new_data 的特征列需按旧模型训练时的特征顺序重排，
        # 否则对列顺序敏感的树模型 predict 直接报 "feature names should match"。
        if old_meta is not None:
            old_feature_order = old_meta.params.get("selected_feature_names") or old_meta.params.get("feature_names")
            if old_feature_order:
                missing = [c for c in old_feature_order if c not in X.columns]
                if missing:
                    raise ValueError(f"AutoRetrainer.retrain: new_data 缺少旧模型训练特征: {missing}")
                X = X[list(old_feature_order)]
        task_type = _infer_task_type(y, new_data.get("task_type") or (old_meta.task_type if old_meta else None))
        validation_size = float(new_data.get("validation_size", 0.2))
        X_train, X_valid, y_train, y_valid = self._split_new_data(X, y, validation_size)

        validation_note: str | None = None
        if old_meta is not None:
            # P2-Q7-fix (M496): 若 new_data 覆盖旧模型训练窗口，旧模型在验证集上是
            # 样本内评估（指标虚高），新模型是样本外 → accept 系统性偏向拒绝新模型，
            # 重训练机制几乎失效。这里把验证集截断到旧模型训练日期之后；无法判断时
            # 显式告警并写入 result，降级必须可见。
            date_values = _frame_date_values(X_valid.index)
            if date_values is not None:
                try:
                    # 优先用旧模型训练段的最后日期（更精确），缺省回退到训练时间戳。
                    cutoff_str = old_meta.params.get("train_end") or old_meta.training_date
                    old_cutoff = pd.Timestamp(cutoff_str)
                except Exception:
                    old_cutoff = None
                if old_cutoff is not None:
                    # 存储的 training_date 带 "Z"（tz-aware UTC），而 X 索引通常是
                    # naive 日期；统一转成 naive UTC 再比较，避免 tz 比较 TypeError。
                    cutoff_naive = old_cutoff.tz_localize(None) if old_cutoff.tzinfo is not None else old_cutoff
                    date_series = pd.Series(date_values)
                    if getattr(date_series.dt, "tz", None) is not None:
                        date_series = date_series.dt.tz_localize(None)
                    after = np.asarray(date_series >= cutoff_naive)
                    if int(after.sum()) > 0:
                        if int(after.sum()) < len(X_valid):
                            X_valid = X_valid.loc[after]
                            y_valid = y_valid.loc[after]
                        validation_note = (
                            f"验证集已按旧模型训练日期 {old_cutoff.date()} 截断，"
                            f"排除旧模型样本内窗口（剩余 {len(X_valid)} 条）"
                        )
                    else:
                        validation_note = (
                            f"新数据全部落在旧模型训练窗口内（旧训练日 {old_cutoff.date()}），"
                            "无法做公平样本外对比，本次对比存在样本内偏置"
                        )
                else:
                    validation_note = (
                        f"无法解析旧模型训练日期 {old_meta.training_date!r}，跳过验证集截断；"
                        "请确保 new_data 全部晚于旧模型训练窗口"
                    )
            else:
                validation_note = (
                    "X 索引不是日期/MultiIndex(date,...)，无法按旧模型训练日期截断验证集；"
                    "请确保 new_data 全部晚于旧模型训练窗口"
                )

        trainer = new_data.get("trainer")
        params = dict(new_data.get("params", {}))
        if callable(trainer):
            new_model = trainer(X_train, y_train, **params)
        else:
            model_class = new_data.get("model_class")
            if model_class is not None:
                new_model = model_class(**params)
            else:
                if old_model is None:
                    raise RuntimeError("无历史生产模型时，必须提供 model_class 或 trainer 来构建新模型")
                if clone is None:
                    raise RuntimeError("sklearn clone 不可用，无法基于旧模型重训练")
                new_model = clone(old_model)
                for key, value in params.items():
                    if hasattr(new_model, "set_params"):
                        new_model.set_params(**{key: value})
            new_model.fit(X_train, y_train)

        threshold = float(new_data.get("metric_threshold", self.schedule.get("metric_threshold", 0.02)))
        if old_model is not None:
            old_metrics = self._evaluate_model(old_model, X_valid, y_valid, task_type)
            new_metrics = self._evaluate_model(new_model, X_valid, y_valid, task_type)
            old_metric_name, old_value, higher = _primary_metric(old_metrics)
            new_metric_name, new_value, _ = _primary_metric(new_metrics)
            improvement = new_value - old_value if higher else old_value - new_value
            accept = improvement >= -threshold
        else:
            # 首次注册：无旧模型可比，以新模型在验证集上的样本外指标为准并 promote。
            new_metrics = self._evaluate_model(new_model, X_valid, y_valid, task_type)
            new_metric_name, new_value, higher = _primary_metric(new_metrics)
            old_metric_name, old_value = None, None
            improvement = None
            accept = True

        result: JsonDict = {
            "model_name": model_name,
            "old_version": old_meta.version if old_meta is not None else None,
            "old_metric": old_metric_name,
            "old_value": old_value,
            "new_metric": new_metric_name,
            "new_value": new_value,
            "higher_is_better": higher,
            "improvement": improvement,
            "accepted": bool(accept),
            "rolled_back": not bool(accept),
            "checked_at": _utc_now(),
        }
        if validation_note is not None:
            result["validation_note"] = validation_note
        if old_meta is None:
            result["first_registration"] = True

        if accept:
            # P2-Q7-fix (M499): 继承旧参数时剥离版本/训练时间/生产标记等元字段，
            # 防止重训练把生产模型文件整体覆盖、破坏回滚能力。
            register_params = dict(old_meta.params) if old_meta is not None else {}
            for meta_key in ("version", "training_date", "is_production", "retrained_from", "promote"):
                register_params.pop(meta_key, None)
            register_params.update(params)
            register_params.update(
                {
                    "task_type": task_type,
                    "retrained_from": old_meta.version if old_meta is not None else None,
                    "feature_names": list(X.columns),
                    "training_date": _utc_now(),
                }
            )
            version = self.registry.register(model_name, new_model, register_params, new_metrics)
            self.registry.promote(model_name, version)
            self.schedule["last_retrain_at"] = _utc_now()
            self._save_schedule()
            result["new_version"] = version
        else:
            # V4.1 feature: 自动回滚只是保持旧生产版本，不删除新模型对象，因为它未注册。
            result["new_version"] = None

        self._append_log(result)
        return result


# ---------------------------------------------------------------------------
# ModelExplainer — 模型可解释性
# ---------------------------------------------------------------------------


class ModelExplainer:
    """模型解释工具。

    V4.1 feature: SHAP 为可选增强；置换重要性、偏依赖和交互效应仅依赖 sklearn。
    """

    def shap_explain(self, model: Any, X_sample: pd.DataFrame) -> dict:
        """计算 SHAP 值并输出特征重要性排序。

        如果 shap 未安装，返回 `available=False`，不抛出硬错误，方便研发环境降级。
        """
        X = _fill_numeric_frame(pd.DataFrame(X_sample))
        if shap is None:
            return {
                "available": False,
                "reason": "shap 未安装；可通过 pip install shap 启用",
                "importance": pd.Series(dtype=float),
            }
        try:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = shap.Explainer(model, X)
            shap_values = explainer.shap_values(X)
            values = shap_values
            if isinstance(values, list):
                # 分类模型通常返回每个类别一组 SHAP；取最后一类作为正类解释。
                values = values[-1]
            arr = np.asarray(values)
            if arr.ndim == 3:
                arr = arr[:, :, -1]
            importance = pd.Series(np.abs(arr).mean(axis=0), index=X.columns, name="mean_abs_shap")
            importance = importance.sort_values(ascending=False)
            expected_value = getattr(explainer, "expected_value", None)
            return {
                "available": True,
                "shap_values": arr,
                "importance": importance,
                "expected_value": expected_value,
                "feature_names": list(X.columns),
            }
        except Exception as exc:
            return {"available": False, "reason": repr(exc), "importance": pd.Series(dtype=float)}

    def permutation_importance(self, model: Any, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10) -> pd.DataFrame:
        """计算排列重要性。

        排列重要性比树模型 impurity importance 更接近样本外效果，但在高度相关特征
        中仍会受到替代效应影响，正式研究应结合特征聚类或分组置换。
        """
        _ensure_sklearn()
        if sk_permutation_importance is None:
            raise ImportError("sklearn.inspection.permutation_importance 不可用")
        X_frame = _fill_numeric_frame(pd.DataFrame(X))
        # P2-Q7-fix (M497): 显式对齐 y 与 X，避免 reindex(X.index) 在索引不一致时
        # 产生全 NaN → permutation_importance 报 NaN 错误。
        y_series = _align_target_to_frame(y, X_frame, context="ModelExplainer.permutation_importance")
        y_series = y_series.dropna()
        X_frame = X_frame.loc[y_series.index]
        if len(y_series) == 0:
            raise ValueError("permutation_importance: 对齐后无有效样本，请检查 y 与 X 的索引")
        result = sk_permutation_importance(
            model,
            X_frame,
            y_series,
            n_repeats=int(n_repeats),
            random_state=42,
            n_jobs=-1,
        )
        df = pd.DataFrame(
            {
                "feature": X_frame.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    def partial_dependence(self, model: Any, X: pd.DataFrame, feature_name: str) -> tuple[np.ndarray, np.ndarray]:
        """生成单特征偏依赖图数据。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(grid_values, averaged_predictions)`。
        """
        _ensure_sklearn()
        X_frame = _fill_numeric_frame(pd.DataFrame(X))
        if feature_name not in X_frame.columns:
            raise KeyError(f"特征不存在: {feature_name}")
        if sk_partial_dependence is not None:
            try:
                pdp = sk_partial_dependence(model, X_frame, [feature_name], grid_resolution=50)
                values = pdp.get("grid_values", pdp.get("values"))[0]
                average = pdp["average"][0]
                return np.asarray(values), np.asarray(average)
            except Exception as e:
                # 不同 sklearn 版本返回结构差异较大；失败后走手工路径。
                logging.getLogger(__name__).error(f"[ml_pipeline_pro] 操作失败: {e}", exc_info=True)

        series = X_frame[feature_name]
        grid = np.linspace(float(series.quantile(0.05)), float(series.quantile(0.95)), 50)
        averages: list[float] = []
        for value in grid:
            tmp = X_frame.copy()
            tmp[feature_name] = value
            averages.append(float(np.mean(_predict_numeric(model, tmp))))
        return grid, np.asarray(averages)

    def feature_interaction(self, model: Any, X: pd.DataFrame, feature1: str, feature2: str) -> pd.DataFrame:
        """计算两个特征的交互效应网格。"""
        X_frame = _fill_numeric_frame(pd.DataFrame(X))
        for feature in (feature1, feature2):
            if feature not in X_frame.columns:
                raise KeyError(f"特征不存在: {feature}")

        grid1 = np.linspace(float(X_frame[feature1].quantile(0.05)), float(X_frame[feature1].quantile(0.95)), 20)
        grid2 = np.linspace(float(X_frame[feature2].quantile(0.05)), float(X_frame[feature2].quantile(0.95)), 20)
        rows: list[JsonDict] = []
        for v1, v2 in itertools.product(grid1, grid2):
            tmp = X_frame.copy()
            tmp[feature1] = v1
            tmp[feature2] = v2
            effect = float(np.mean(_predict_numeric(model, tmp)))
            rows.append({feature1: v1, feature2: v2, "effect": effect})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Purged/Embargo 时间序列切分器
# ---------------------------------------------------------------------------


class PurgedEmbargoTimeSeriesSplit(BaseCrossValidator):
    """带 embargo 的时间序列交叉验证。

    V4.1 feature: 金融标签常常使用未来 N 日收益，普通 KFold 会泄漏未来信息。
    这里实现一个轻量切分器: 测试集前后的 embargo 样本不进入训练集。
    若需要严格按事件起止时间做 purging，可在上层传入 groups 并使用 GroupKFold。

    V4.1 fix: 本类已正确定义于 ml_pipeline_pro.py，继承 BaseCrossValidator。
    当 sklearn 不可用时降级为 object，但 split/get_n_splits 方法仍正常工作。
    引用位于: _build_cv() (行1476), walk_forward_evaluate() (行2128)。
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits 至少为 2")
        self.n_splits = int(n_splits)
        self.embargo = max(0, int(embargo))

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: D401
        # Q7-fix: 与 split() 实际产出一致 —— 小样本时首折（或前若干折）因
        # 无历史数据被跳过，不能再虚报满额折数（sklearn CV 契约）。
        if X is None:
            return self.n_splits
        n_samples = len(X)
        if n_samples < 2:
            return 0
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        n_real = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            current = stop
            if start - self.embargo > 0 and stop > start:
                n_real += 1
        return n_real

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if n_samples < 2:
            warnings.warn(f"PurgedEmbargoTimeSeriesSplit: 样本过少({n_samples})，无可用折", RuntimeWarning)
            return
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        n_yielded = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            # Q7-fix: 只保留测试窗之前的样本作为训练集（含 embargo）。
            # 原实现同时拼接测试窗之后的样本，导致第一折 train 全是"未来"
            # 数据 -> CV 分数系统性虚高（实测 train=10-29 全在未来）。
            train_idx = indices[: max(0, start - self.embargo)]
            current = stop
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            n_yielded += 1
            yield train_idx, test_idx
        if n_yielded < self.n_splits:
            warnings.warn(
                f"PurgedEmbargoTimeSeriesSplit: 实际产出 {n_yielded} 折 < 声明的 {self.n_splits} 折"
                f"（小样本或 embargo={self.embargo} 过大），请留意评估样本量", RuntimeWarning
            )


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """按股票分组的时间序列 + purge/embargo 切分器。

    审计 2026-08-16：解决“GroupKFold 与 purged/embargo 无法同时满足”的问题。
    对每个 group（股票），按行序（时间升序）切成连续块；fold k 使用该股票
    前 k 块训练、第 k 块测试，并从训练尾部剔除 embargo 行，防止标签与测试
    窗口重叠。既保证同股票训练严格早于测试，又支持多股票面板。
    """

    def __init__(self, n_splits: int = 5, embargo: int = 20) -> None:
        if n_splits < 2:
            raise ValueError("n_splits 至少为 2")
        self.n_splits = int(n_splits)
        self.embargo = max(0, int(embargo))

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits - 1  # 从第 1 块起才有训练集

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("PurgedGroupTimeSeriesSplit 需要 groups（每行股票 id）")
        g_arr = np.asarray(groups)
        n_samples = len(X)
        if len(g_arr) != n_samples:
            raise ValueError("groups 长度必须与 X 行数一致")
        group_pos: dict[int, list[int]] = {}
        for pos, g in enumerate(g_arr):
            group_pos.setdefault(int(g), []).append(pos)
        for k in range(1, self.n_splits):
            tr_all: list[int] = []
            te_all: list[int] = []
            for g, positions in group_pos.items():
                arr = np.asarray(positions, dtype=int)
                if len(arr) < self.n_splits:
                    tr_all.extend(arr.tolist())
                    continue
                bounds = np.linspace(0, len(arr), self.n_splits + 1, dtype=int)
                te = arr[bounds[k]:bounds[k + 1]]
                tr = np.concatenate([arr[bounds[j]:bounds[j + 1]] for j in range(k)])
                if len(tr) > self.embargo:
                    tr = tr[:-self.embargo]
                tr_all.extend(tr.tolist())
                te_all.extend(te.tolist())
            if tr_all and te_all:
                yield np.asarray(tr_all, dtype=int), np.asarray(te_all, dtype=int)


# ---------------------------------------------------------------------------
# HyperparameterOptimizer — 超参优化
# ---------------------------------------------------------------------------


class HyperparameterOptimizer:
    """超参优化器。

    V4.1 feature: 提供网格搜索、随机搜索和粗到细序贯搜索。默认不使用随机 KFold，
    优先使用 GroupKFold 或带 embargo 的时间序列切分，降低时间泄漏风险。
    """

    def __init__(self, groups: Sequence[Any] | None = None, embargo: int = 0, n_jobs: int = -1, random_state: int = 42) -> None:
        # P2-Q7-fix (M498): 不再立即 reset_index —— 保留调用方传入的索引，在 _run_search
        # 中校验与 X 原始索引一致后再按位置对齐，避免带日期索引的 groups 被静默丢弃
        # 造成 GroupKFold 分组错位（组间泄漏无提示）。
        self.groups = pd.Series(groups) if groups is not None else None
        self.embargo = int(embargo)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

    def set_groups(self, groups: Sequence[Any] | None) -> None:
        """设置 GroupKFold 分组。

        P2-Q7-fix (M498): groups 必须与 X 行序严格一致（按位置对齐）。
        传入带索引的 Series 时，索引须与 X 的索引一致，否则 _run_search 会抛错，
        避免静默错位造成组间泄漏。
        """
        self.groups = pd.Series(groups) if groups is not None else None

    def _extract_groups_from_X(self, X: pd.DataFrame | np.ndarray) -> pd.Series | None:
        """从 X 的索引中尝试提取分组。

        Q7-fix: 仅当索引是 MultiIndex(date, symbol) 这类截面结构时才自动取 groups；
        普通单层日期索引一律走时间序列切分（TimeSeriesSplit / PurgedEmbargoTimeSeriesSplit），
        避免按日期随机分组的 GroupKFold 静默破坏时序切分（训练/测试标签窗口互相重叠且无 purge）。
        显式传入 self.groups 时优先使用。
        """
        if self.groups is not None:
            return self.groups
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
            for level_name in ("group", "date", "month", "period"):
                if level_name in X.index.names:
                    groups = pd.Series(X.index.get_level_values(level_name)).reset_index(drop=True)
                    if len(groups) == len(X):
                        warnings.warn(
                            f"从 MultiIndex 自动提取分组用于 GroupKFold（level='{level_name}'），"
                            "若为纯时序数据请显式设置 groups=None 并调大 embargo", RuntimeWarning
                        )
                        return groups
        if isinstance(X, pd.DataFrame) and X.index.name in {"date", "month", "period"}:
            warnings.warn(
                f"单层索引名 '{X.index.name}' 不再自动转 GroupKFold："
                "时序数据将使用时间顺序切分（TimeSeriesSplit/PurgedEmbargo），"
                "截面数据请使用 MultiIndex 或显式 set_groups()", RuntimeWarning
            )
        return None

    def _build_cv(self, X: pd.DataFrame | np.ndarray, cv: int) -> tuple[Any, pd.Series | None]:
        """构建 CV splitter 和可选 groups。

        Q7-fix: 普通单层日期索引不再被静默转成 GroupKFold；只有截面 MultiIndex
        或显式 groups 才走分组 CV，其余一律时间顺序切分。
        P1-Q7-fix: 分组数不足以构成 cv 折时，降级为时间序列切分必须显式告警，
        避免调用方以为用了 GroupKFold（静默吞降级）。
        """
        _ensure_sklearn()
        groups = self._extract_groups_from_X(X)
        if groups is not None:
            if groups.nunique(dropna=False) >= cv:
                # 审计 2026-08-16：分组+时间净化同时满足（不再二选一）
                if self.embargo > 0:
                    return PurgedGroupTimeSeriesSplit(n_splits=int(cv), embargo=self.embargo), groups
                return GroupKFold(n_splits=int(cv)), groups
            warnings.warn(
                f"分组数 {groups.nunique(dropna=False)} < cv={cv}，分组 CV 不可用；"
                "已降级为时间序列切分并忽略分组，请检查 groups 设置", RuntimeWarning
            )
        if self.embargo > 0:
            return PurgedEmbargoTimeSeriesSplit(n_splits=int(cv), embargo=self.embargo), None
        return TimeSeriesSplit(n_splits=int(cv)), None

    def _make_estimator(self, model_class: Callable[..., Any] | Any, base_params: Mapping[str, Any] | None = None) -> Any:
        """实例化模型；支持传入类、callable 或已经初始化的 estimator。"""
        base_params = dict(base_params or {})
        if isinstance(model_class, type):
            return model_class(**base_params)
        if callable(model_class) and not hasattr(model_class, "fit"):
            return model_class(**base_params)
        estimator = model_class
        if base_params and hasattr(estimator, "set_params"):
            estimator = clone(estimator)
            estimator.set_params(**base_params)
        return estimator

    def _run_search(
        self,
        search_type: Literal["grid", "random"],
        model_class: Callable[..., Any] | Any,
        param_grid: Mapping[str, Any],
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int,
        scoring: str | None,
        n_iter: int | None = None,
        base_params: Mapping[str, Any] | None = None,
    ) -> dict:
        """统一执行 GridSearchCV / RandomizedSearchCV。"""
        _ensure_sklearn()
        estimator = self._make_estimator(model_class, base_params)
        splitter, groups = self._build_cv(X, cv)
        X_fit = _fill_numeric_frame(pd.DataFrame(X))
        y_fit = pd.Series(y).reset_index(drop=True)
        original_index = X_fit.index
        X_fit = X_fit.reset_index(drop=True)
        if groups is not None:
            groups_series = pd.Series(groups)
            # P2-Q7-fix (M498): 带非默认索引的 groups 必须与 X 的原始索引一致，否则
            # reset_index(drop=True) 后按位置对齐会静默错位 → GroupKFold 组间泄漏。
            if not groups_series.index.equals(pd.RangeIndex(len(groups_series))):
                if not groups_series.index.equals(original_index):
                    raise ValueError(
                        f"groups 索引与 X 索引不一致：GroupKFold 分组必须与 X 行序严格对应。"
                        "请传入与 X 行序一致的 groups（数组/无索引 Series 或按 X.index 对齐）"
                    )
            if len(groups_series) != len(X_fit):
                raise ValueError(
                    f"groups 长度 {len(groups_series)} 与 X 行数 {len(X_fit)} 不一致："
                    "GroupKFold 分组必须与 X 行序严格对应"
                )
            groups = groups_series.reset_index(drop=True)

        if search_type == "grid":
            if GridSearchCV is None:
                raise ImportError("GridSearchCV 不可用")
            search = GridSearchCV(estimator, dict(param_grid), cv=splitter, scoring=scoring, n_jobs=self.n_jobs, refit=True)
        else:
            if RandomizedSearchCV is None:
                raise ImportError("RandomizedSearchCV 不可用")
            search = RandomizedSearchCV(
                estimator,
                dict(param_grid),
                n_iter=int(n_iter or 50),
                cv=splitter,
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                refit=True,
            )
        search.fit(X_fit, y_fit, groups=groups)
        results = pd.DataFrame(search.cv_results_)
        return {
            "best_estimator": search.best_estimator_,
            "best_params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "cv_results": results,
            "search": search,
            "cv_type": type(splitter).__name__,
            "used_groups": groups is not None,
        }

    def grid_search(
        self,
        model_class: Callable[..., Any] | Any,
        param_grid: Mapping[str, Sequence[Any]],
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
        scoring: str | None = None,
    ) -> dict:
        """网格搜索。

        P2-Q7-fix (L508): 默认 scoring 由 neg_log_loss 改为 None —— 让 sklearn 使用
        estimator 默认 scorer（分类=accuracy、回归=r2），避免回归任务或某折单类标签
        GridSearchCV 直接失败；如需 log_loss / neg_mean_squared_error 可显式传入。
        """
        return self._run_search("grid", model_class, param_grid, X, y, cv, scoring)

    def random_search(
        self,
        model_class: Callable[..., Any] | Any,
        param_dist: Mapping[str, Any],
        n_iter: int = 50,
        X: pd.DataFrame | np.ndarray | None = None,
        y: pd.Series | np.ndarray | None = None,
        cv: int = 5,
    ) -> dict:
        """随机搜索。"""
        if X is None or y is None:
            raise ValueError("random_search 必须传入 X 和 y")
        return self._run_search("random", model_class, param_dist, X, y, cv, None, n_iter=n_iter)

    def sequential_search(
        self,
        model_class: Callable[..., Any] | Any,
        param_grids: Sequence[Mapping[str, Sequence[Any]]],
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
    ) -> dict:
        """序贯搜索: 先粗网格，再把最优参数固定后进入细网格。"""
        best_params: JsonDict = {}
        history: list[JsonDict] = []
        best_estimator: Any = None
        best_score = -np.inf
        for step, grid in enumerate(param_grids, start=1):
            result = self._run_search("grid", model_class, grid, X, y, cv, None, base_params=best_params)
            step_params = dict(result["best_params"])
            best_params.update(step_params)
            best_estimator = result["best_estimator"]
            best_score = float(result["best_score"])
            history.append(
                {
                    "step": step,
                    "grid_keys": list(grid.keys()),
                    "best_params": dict(best_params),
                    "best_score": best_score,
                    "cv_type": result["cv_type"],
                }
            )
        return {
            "best_estimator": best_estimator,
            "best_params": best_params,
            "best_score": best_score,
            "history": history,
        }


# ---------------------------------------------------------------------------
# MLEngineV2 — ML 引擎顶层整合
# ---------------------------------------------------------------------------


class MLEngineV2:
    """ML 引擎顶层整合。

    V4.1 feature: 统一训练、预测和评估入口。该类不强制绑定某个具体策略，
    而是通过 FeatureStorePro + ModelRegistry 组合出研发闭环。
    """

    def __init__(
        self,
        feature_store: FeatureStorePro | None = None,
        registry: ModelRegistry | None = None,
        optimizer: HyperparameterOptimizer | None = None,
    ) -> None:
        self.feature_store = feature_store if feature_store is not None else FeatureStorePro()
        self.registry = registry if registry is not None else ModelRegistry()
        self.optimizer = optimizer if optimizer is not None else HyperparameterOptimizer()
        self.explainer = ModelExplainer()

    def _load_training_matrix(self, feature_names: Sequence[str], target_name: str) -> tuple[pd.DataFrame, pd.Series]:
        """从特征存储加载训练矩阵。

        P1-Q7-fix: 此处只做数值化转换（to_numeric），不做中位数填补 —— 缺失值填补
        必须发生在 _chronological_split 之后、仅用训练段拟合，否则验证期分布会泄漏进
        填补值，导致评估指标系统性偏乐观。
        """
        if not feature_names:
            raise ValueError("feature_names 不能为空")
        features = [self.feature_store.load_feature(name).rename(name) for name in feature_names]
        X = pd.concat(features, axis=1)
        y = self.feature_store.load_feature(target_name).rename(target_name)
        frame = X.join(y, how="inner").dropna(subset=[target_name])
        if frame.empty:
            raise ValueError("训练数据为空，请检查特征和目标的索引是否对齐")
        X_out = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y_out = frame[target_name]
        return X_out, y_out

    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int | None = None,
        min_importance: float = 0.0,
    ) -> list[str]:
        """自动特征选择。"""
        _ensure_sklearn()
        if X.empty:
            return []
        task_type = _infer_task_type(y)
        if task_type == "classification":
            selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
        else:
            selector = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        selector.fit(X, y)
        importance = pd.Series(selector.feature_importances_, index=X.columns).sort_values(ascending=False)
        if min_importance > 0:
            importance = importance[importance >= min_importance]
        if max_features is not None and max_features > 0:
            importance = importance.head(int(max_features))
        selected = list(importance.index)
        return selected or list(X.columns)

    def _make_model(self, model_type: str, task_type: TaskType, params: Mapping[str, Any]) -> Any:
        """按模型类型创建 estimator。

        P2-Q7-fix (M500): LGBM/XGB 分类路径透传 scale_pos_weight/is_unbalance 等
        类别不平衡参数（由 train() 按样本比例自动注入默认 scale_pos_weight，
        调用方也可显式覆盖）。
        """
        _ensure_sklearn()
        model_type = str(model_type).lower()
        model_params = dict(params)
        # 顶层控制参数不传给 sklearn estimator。
        for key in [
            "model_name",
            "task_type",
            "auto_feature_selection",
            "max_features",
            "min_importance",
            "validation_size",
            "promote",
            "groups",
            "version",
            "notes",
            "calibrate",
        ]:
            model_params.pop(key, None)

        model_params.setdefault("random_state", 42)
        if model_type in {"lgb", "lightgbm"}:
            if task_type == "classification" and LGBMClassifier is not None:
                model_params.setdefault("n_estimators", 300)
                return LGBMClassifier(**model_params)
            if task_type == "regression" and LGBMRegressor is not None:
                model_params.setdefault("n_estimators", 300)
                return LGBMRegressor(**model_params)
            warnings.warn("LightGBM 未安装，降级为 RandomForest")
        if model_type in {"xgb", "xgboost"}:
            if task_type == "classification" and XGBClassifier is not None:
                model_params.setdefault("n_estimators", 300)
                model_params.setdefault("eval_metric", "logloss")
                return XGBClassifier(**model_params)
            if task_type == "regression" and XGBRegressor is not None:
                model_params.setdefault("n_estimators", 300)
                return XGBRegressor(**model_params)
            warnings.warn("XGBoost 未安装，降级为 RandomForest")
        if task_type == "classification":
            model_params.setdefault("n_estimators", 300)
            model_params.setdefault("n_jobs", -1)
            model_params.setdefault("class_weight", "balanced")
            return RandomForestClassifier(**model_params)
        model_params.setdefault("n_estimators", 300)
        model_params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**model_params)

    def _chronological_split(self, X: pd.DataFrame, y: pd.Series, validation_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """按时间顺序切分训练集和验证集。"""
        n = len(X)
        if n < 5:
            # P2-Q7-fix (M495): 原实现 n<5 时返回 (X,X,y,y)，样本内指标冒充验证指标
            # 被注册进 registry 并可能触发 promote（指标虚高）；n=1 时 split=0 产生
            # 空训练集 → fit 抛异常。改为显式报错，要求调用方提供足够样本。
            raise ValueError(
                f"_chronological_split: 样本数过少({n})，至少需要 5 个样本才能做有意义的"
                "训练/验证切分，请提供更多数据或改用 walk-forward/交叉验证"
            )
        validation_size = validation_size if 0 < validation_size < 0.8 else 0.2
        split = max(1, int(n * (1.0 - validation_size)))
        if split >= n:
            split = n - 1
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    def train(self, feature_names: Sequence[str], target_name: str, model_type: str = "lgb", **params: Any) -> str:
        """训练入口。

        V4.1 feature: 自动加载特征、自动特征选择、训练模型、写入 ModelRegistry。

        P1-Q7-fix: 先 _chronological_split 再做特征选择与缺失值填补。原先对全量 X,y
        跑 RandomForest 选特征（用到了验证期标签与未来分布），且中位数填补在切分前计算，
        验证集信息进入特征子集与填补值，评估指标系统性偏乐观。现在中位数只 fit 训练段、
        transform 验证段，特征重要性也只由训练段估计。

        Returns
        -------
        str
            `model_name@version`。
        """
        X_raw, y = self._load_training_matrix(feature_names, target_name)
        validation_size = float(params.get("validation_size", 0.2))
        X_raw_train, X_raw_valid, y_train, y_valid = self._chronological_split(X_raw, y, validation_size)

        # 中位数填补器只在训练段 fit，再 transform 到验证段（无泄漏）。
        med = X_raw_train.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
        X_train = X_raw_train.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        X_valid = X_raw_valid.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

        explicit_task = params.get("task_type")
        task_type = _infer_task_type(y_train, explicit_task)
        auto_select = bool(params.get("auto_feature_selection", True))
        if auto_select:
            selected = self._select_features(
                X_train,
                y_train,
                max_features=params.get("max_features"),
                min_importance=float(params.get("min_importance", 0.0)),
            )
        else:
            selected = list(X_raw.columns)
        X_train = X_train[selected]
        X_valid = X_valid[selected]

        # P2-Q7-fix (M500): LGBM/XGB 分类路径按样本比例注入 scale_pos_weight，
        # 缓解类别不平衡（调用方可在 params 显式覆盖）。
        if (
            task_type == "classification"
            and model_type in {"lgb", "lightgbm", "xgb", "xgboost"}
            and "scale_pos_weight" not in params
        ):
            counts = pd.Series(y_train).value_counts()
            if len(counts) == 2:
                neg, pos = int(counts.min()), int(counts.max())
                if pos > 0:
                    params["scale_pos_weight"] = neg / pos

        model = self._make_model(model_type, task_type, params)
        # P2-Q7-fix (M500): 可选概率校准 —— CalibratedClassifierCV 在训练段内部 CV
        # 拟合校准映射，验证集指标即为校准后的样本外指标；小类别样本不足以校准时
        # 显式告警并回退未校准模型。
        calibrate = bool(params.get("calibrate", False)) and task_type == "classification"
        calibrated = False
        if calibrate:
            if CalibratedClassifierCV is None:
                warnings.warn("CalibratedClassifierCV 不可用（sklearn 版本过低），跳过概率校准", RuntimeWarning)
            else:
                class_counts = pd.Series(y_train).value_counts()
                min_cls = int(class_counts.min()) if len(class_counts) >= 2 else 1
                if min_cls < 3:
                    warnings.warn(
                        f"train: 类别样本过少(min_class={min_cls})，无法进行概率校准，跳过；"
                        "请提供更多样本或关闭 calibrate 参数", RuntimeWarning
                    )
                else:
                    try:
                        model = CalibratedClassifierCV(model, method="sigmoid", cv=min(3, min_cls))
                        model.fit(X_train, y_train)
                        calibrated = True
                    except Exception as exc:
                        warnings.warn(f"train: 概率校准失败，回退未校准模型: {exc}", RuntimeWarning)
        if not calibrated:
            model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        proba = model.predict_proba(X_valid) if task_type == "classification" and hasattr(model, "predict_proba") else None
        metrics = _score_predictions(task_type, y_valid, pred, proba)

        model_name = str(params.get("model_name") or f"{model_type}_{target_name}")
        register_params = dict(params)
        register_params.update(
            {
                "model_type": model_type,
                "task_type": task_type,
                "feature_names": list(feature_names),
                "selected_feature_names": selected,
                "target_name": target_name,
                "n_train": int(len(X_train)),
                "n_valid": int(len(X_valid)),
                "calibrated": bool(calibrated),
                "training_date": _utc_now(),
                # P2-Q7-fix (M496): 记录训练段实际日期范围，供 AutoRetrainer 在
                # 重训练时把验证集截断到旧模型训练期之后（比 training_date 时间戳更精确）。
                "train_start": str(X_train.index.min()) if len(X_train) else None,
                "train_end": str(X_train.index.max()) if len(X_train) else None,
                # 审计 2026-08-16：保存训练期中位数 imputer，预测/评估复用
                "imputer_median": {str(k): (None if pd.isna(v) else float(v)) for k, v in med.items()},
            }
        )
        version = self.registry.register(model_name, model, register_params, metrics)

        # P2-Q7-fix (M494): _resolve_model_meta 在无生产版本时抛 KeyError，这里的
        # try/except 才能触发"首个模型自动 promote"（不再是死代码）。
        should_promote = bool(params.get("promote", False))
        try:
            self.registry.metadata(model_name, "production")
        except KeyError:
            should_promote = True
        except Exception:
            should_promote = True
        if should_promote:
            self.registry.promote(model_name, version)
        return f"{model_name}@{version}"

    def _resolve_model_for_prediction(self, model_name_or_version: str) -> tuple[Any, ModelVersionMeta]:
        """预测时优先加载生产模型；如果名称包含版本则加载指定版本。"""
        name, version = _parse_name_version(model_name_or_version)
        if version is not None:
            meta = self.registry.metadata(name, version)
            return self.registry.load(name, version), meta
        try:
            meta = self.registry.metadata(name, "production")
            return self.registry.load(name, meta.version), meta
        except Exception:
            meta = self.registry.metadata(name, "latest")
            return self.registry.load(name, meta.version), meta

    def predict(self, model_name_or_version: str, features: pd.DataFrame | pd.Series | Mapping[str, Any] | np.ndarray) -> pd.Series:
        """预测入口。"""
        model, meta = self._resolve_model_for_prediction(model_name_or_version)
        selected = meta.params.get("selected_feature_names") or meta.params.get("feature_names")
        X = _make_2d_frame(features, selected)
        X = _fill_numeric_frame(X, medians=meta.params.get("imputer_median"))
        values = _predict_numeric(model, X)
        return pd.Series(values, index=X.index, name=f"prediction_{meta.name}")

    def evaluate(self, model_name: str, test_data: pd.DataFrame | Mapping[str, Any]) -> dict:
        """模型评估入口。"""
        model, meta = self._resolve_model_for_prediction(model_name)
        target_name = meta.params.get("target_name", "target")
        selected = meta.params.get("selected_feature_names") or meta.params.get("feature_names")

        if isinstance(test_data, Mapping) and "X" in test_data and "y" in test_data:
            X = _make_2d_frame(test_data["X"], selected)
            y_raw = test_data["y"]
        else:
            frame = pd.DataFrame(test_data).copy()
            if target_name not in frame.columns:
                if "target" in frame.columns:
                    target_name = "target"
                else:
                    raise KeyError(f"test_data 必须包含目标列: {target_name}")
            y_raw = frame[target_name]
            X = _make_2d_frame(frame.drop(columns=[target_name]), selected)
        X = _fill_numeric_frame(X, medians=meta.params.get("imputer_median"))
        # P2-Q7-fix (M497): 显式对齐 y 与 X（行数校验 + 索引告警），避免索引错配
        # 产生全 NaN 静默进入指标计算。
        y = _align_target_to_frame(y_raw, X, context="MLEngineV2.evaluate")
        y = y.dropna()
        X = X.loc[y.index]
        if len(y) == 0:
            warnings.warn("MLEngineV2.evaluate: 对齐后无有效样本，返回空指标", RuntimeWarning)
            return {"model_name": meta.name, "version": meta.version, "task_type": meta.task_type, "n_samples": 0}
        task_type = _infer_task_type(y, meta.task_type)
        pred = model.predict(X)
        proba = model.predict_proba(X) if task_type == "classification" and hasattr(model, "predict_proba") else None
        metrics = _score_predictions(task_type, y, pred, proba)
        metrics.update({"model_name": meta.name, "version": meta.version})
        return metrics

    def explain(self, model_name: str, X_sample: pd.DataFrame) -> dict:
        """便捷 SHAP 解释入口。"""
        model, meta = self._resolve_model_for_prediction(model_name)
        selected = meta.params.get("selected_feature_names") or meta.params.get("feature_names")
        X = _make_2d_frame(X_sample, selected)
        return self.explainer.shap_explain(model, X)


# ---------------------------------------------------------------------------
# V4.1 feature: 研发审计与诊断工具
# ---------------------------------------------------------------------------


@dataclass
class FeatureQualityReport:
    """特征质量报告。

    V4.1 feature: 训练前先审计数据质量，避免模型把缺失值、极端值或常数列当作
    可学习信号。该 dataclass 主要用于 JSON/日志落地，也方便 notebook 展示。
    """

    n_rows: int
    n_columns: int
    missing_rate: dict[str, float]
    zero_rate: dict[str, float]
    infinite_rate: dict[str, float]
    constant_features: list[str]
    high_missing_features: list[str]
    high_correlation_pairs: list[dict[str, Any]]
    generated_at: str = field(default_factory=_utc_now)

    def to_frame(self) -> pd.DataFrame:
        """转换成每个特征一行的 DataFrame。"""
        rows: list[JsonDict] = []
        for feature, miss in self.missing_rate.items():
            rows.append(
                {
                    "feature": feature,
                    "missing_rate": miss,
                    "zero_rate": self.zero_rate.get(feature, np.nan),
                    "infinite_rate": self.infinite_rate.get(feature, np.nan),
                    "is_constant": feature in self.constant_features,
                    "high_missing": feature in self.high_missing_features,
                }
            )
        return pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)


@dataclass
class WalkForwardResult:
    """Walk-forward 评估结果。

    V4.1 feature: 金融 ML 的核心不是一次 train/test split，而是跨时间切片的稳定性。
    该结构记录每一折的样本区间、指标和最终聚合统计。
    """

    fold_metrics: list[JsonDict]
    aggregate_metrics: JsonDict
    task_type: TaskType
    n_folds: int
    generated_at: str = field(default_factory=_utc_now)

    def to_frame(self) -> pd.DataFrame:
        """返回每一折指标表。"""
        return pd.DataFrame(self.fold_metrics)


def audit_feature_matrix(
    X: pd.DataFrame,
    missing_threshold: float = 0.25,
    corr_threshold: float = 0.95,
) -> FeatureQualityReport:
    """审计特征矩阵质量。

    Parameters
    ----------
    X:
        待审计的特征矩阵。
    missing_threshold:
        缺失率超过该阈值即标记为高缺失特征。
    corr_threshold:
        两两相关系数绝对值超过该阈值即记录为高相关特征对。

    Returns
    -------
    FeatureQualityReport
        完整质量报告。
    """
    frame = pd.DataFrame(X).copy()
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    n_rows = int(len(numeric))
    n_cols = int(numeric.shape[1])
    if n_rows == 0:
        missing_rate = {str(c): 1.0 for c in numeric.columns}
        zero_rate = {str(c): 0.0 for c in numeric.columns}
        infinite_rate = {str(c): 0.0 for c in numeric.columns}
    else:
        missing_rate = {str(c): float(numeric[c].isna().mean()) for c in numeric.columns}
        zero_rate = {str(c): float((numeric[c] == 0).mean()) for c in numeric.columns}
        infinite_rate = {str(c): float(np.isinf(numeric[c].to_numpy(dtype=float, na_value=np.nan)).mean()) for c in numeric.columns}

    constant_features: list[str] = []
    for col in numeric.columns:
        values = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() <= 1:
            constant_features.append(str(col))

    high_missing = [name for name, rate in missing_rate.items() if rate >= missing_threshold]
    high_pairs: list[JsonDict] = []
    if n_cols > 1 and n_rows > 2:
        corr = numeric.replace([np.inf, -np.inf], np.nan).corr().abs()
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and float(value) >= corr_threshold:
                    high_pairs.append({"feature1": str(left), "feature2": str(right), "abs_corr": float(value)})

    return FeatureQualityReport(
        n_rows=n_rows,
        n_columns=n_cols,
        missing_rate=missing_rate,
        zero_rate=zero_rate,
        infinite_rate=infinite_rate,
        constant_features=constant_features,
        high_missing_features=high_missing,
        high_correlation_pairs=sorted(high_pairs, key=lambda row: row["abs_corr"], reverse=True),
    )


def neutralize_features(
    X: pd.DataFrame,
    exposures: pd.DataFrame,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """对特征做暴露中性化。

    V4.1 feature: 很多 alpha 特征会混入市值、行业、波动率等公共风险暴露。
    该函数按列回归并取残差，输出与原始 X 同维度的中性化特征矩阵。
    """
    X_frame = _fill_numeric_frame(pd.DataFrame(X))
    E_frame = _fill_numeric_frame(pd.DataFrame(exposures).reindex(X_frame.index))
    if E_frame.empty:
        return X_frame.copy()
    design = E_frame.copy()
    if add_intercept:
        design.insert(0, "intercept", 1.0)
    design_values = design.to_numpy(dtype=float)
    out = pd.DataFrame(index=X_frame.index)
    for col in X_frame.columns:
        y = X_frame[col].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(design_values).all(axis=1)
        if mask.sum() <= design_values.shape[1]:
            out[col] = X_frame[col]
            continue
        beta, *_ = np.linalg.lstsq(design_values[mask], y[mask], rcond=None)
        fitted = design_values @ beta
        out[col] = y - fitted
    return out


def winsorize_frame(
    X: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """按列缩尾，减少极端值对树模型和线性模型的影响。"""
    if not 0 <= lower < upper <= 1:
        raise ValueError("lower/upper 必须满足 0 <= lower < upper <= 1")
    frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    low = frame.quantile(lower)
    high = frame.quantile(upper)
    return frame.clip(lower=low, upper=high, axis=1)


def rank_normalize_frame(X: pd.DataFrame, by_date: bool = True) -> pd.DataFrame:
    """把特征转换成横截面百分位排名。

    V4.1 feature: 截面 alpha 训练通常更关心排序而非原始量纲。若索引是
    MultiIndex 且包含 date 层，则默认按日期做横截面排名；否则按全样本排名。

    P2-Q7-fix (L507): 该函数仅适用于横截面数据（MultiIndex(date, symbol)）。
    非截面输入会做全样本排名 —— 在时序单标的场景下，排名值会混入未来信息
    （全样本排名使用了整段数据的分位数）。请改用滚动排名；这里对非截面输入
    显式告警。
    """
    frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    if by_date and isinstance(frame.index, pd.MultiIndex) and "date" in frame.index.names:
        return frame.groupby(level="date", group_keys=False).rank(pct=True)
    if by_date and not isinstance(frame.index, pd.MultiIndex):
        warnings.warn(
            "rank_normalize_frame: 输入不是 MultiIndex(date, ...) 截面结构，已按全样本排名；"
            "时序单标的场景请使用滚动排名，避免未来信息泄漏", RuntimeWarning
        )
    return frame.rank(pct=True)


def information_coefficient(
    predictions: pd.Series | np.ndarray,
    forward_returns: pd.Series | np.ndarray,
    method: Literal["pearson", "spearman"] = "spearman",
    min_samples: int = 10,
) -> float:
    """计算预测值与未来收益的 IC。

    P2-Q7-fix (L504): 最小样本门槛从 3 提高到 min_samples（默认 10），
    3 个样本的 Spearman 相关统计意义极弱，t 统计与 IR 会被异常放大；
    样本不足返回 NaN。
    """
    pred = pd.Series(predictions, name="prediction")
    ret = pd.Series(forward_returns, name="forward_return")
    frame = pd.concat([pred, ret], axis=1).dropna()
    if len(frame) < min_samples:
        return float("nan")
    return float(frame["prediction"].corr(frame["forward_return"], method=method))


def daily_information_coefficient(
    predictions: pd.Series,
    forward_returns: pd.Series,
    method: Literal["pearson", "spearman"] = "spearman",
    min_samples: int = 10,
) -> pd.Series:
    """按日期计算横截面 IC。

    输入最好是 MultiIndex(date, symbol) 的 Series。若不是 MultiIndex，则退化为
    全样本单个 IC，并用当前日期作为索引返回一行。

    P2-Q7-fix (L503): pd.Timestamp.utcnow() 在 pandas 2.1+ 已弃用，改用 now(tz="UTC")。
    P2-Q7-fix (L504): 单日截面样本数 < min_samples 时返回 NaN 且不再 dropna，
    由 ic_summary 统计无效天数，避免小样本 IC 虚高被误读为有效。
    """
    pred = pd.Series(predictions, name="prediction")
    ret = pd.Series(forward_returns, name="forward_return")
    frame = pd.concat([pred, ret], axis=1).dropna()
    if frame.empty:
        return pd.Series(dtype=float, name="ic")
    if isinstance(frame.index, pd.MultiIndex) and "date" in frame.index.names:
        values = frame.groupby(level="date").apply(
            lambda g: g["prediction"].corr(g["forward_return"], method=method)
            if len(g) >= min_samples
            else np.nan
        )
        values.name = "ic"
        return values  # 保留 NaN 天数，由 ic_summary 统计无效天数
    return pd.Series(
        [information_coefficient(frame["prediction"], frame["forward_return"], method, min_samples)],
        index=[pd.Timestamp.now(tz="UTC")],
        name="ic",
    )


def ic_summary(ic_series: pd.Series) -> dict[str, float | int | None]:
    """汇总 IC 序列。"""
    raw = pd.Series(ic_series)
    # P2-Q7-fix (L504): 统计因样本不足/数据缺失而无效的天数，让"有效天数少"可见。
    n_invalid = int(raw.isna().sum())
    series = raw.dropna().astype(float)
    if series.empty:
        return {"n": 0, "n_invalid": n_invalid, "mean_ic": None, "std_ic": None, "ir": None, "hit_rate": None, "t_stat": None}
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    ir = mean / std if std > 0 else None
    t_stat = mean / (std / math.sqrt(len(series))) if std > 0 and len(series) > 1 else None
    return {
        "n": int(len(series)),
        "n_invalid": n_invalid,
        "mean_ic": mean,
        "std_ic": std,
        "ir": ir,
        "hit_rate": float((series > 0).mean()),
        "t_stat": t_stat,
    }


def top_bottom_spread(
    predictions: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 5,
) -> pd.DataFrame:
    """计算预测分组的多空收益差。

    V4.1 feature: 用分组收益辅助判断模型是否真的有排序能力，而不是只看单点指标。
    """
    if quantiles < 2:
        raise ValueError("quantiles 至少为 2")
    pred = pd.Series(predictions, name="prediction")
    ret = pd.Series(forward_returns, name="forward_return")
    frame = pd.concat([pred, ret], axis=1).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "mean_return", "count"])

    def assign_bucket(group: pd.DataFrame) -> pd.Series:
        # V11 审计修复（High）: 原实现小截面（股票数 < quantiles）时
        # pd.qcut 抛 ValueError，groupby.apply 直接崩整个调用。
        # 修正: 样本不足时按排名直接分桶（等价于均匀分桶的退化），不崩溃。
        ranks = group["prediction"].rank(method="first")
        if len(ranks) < quantiles:
            # 退化分桶: 按排名比例分到 [1..quantiles]
            labels = ((ranks - 1) * quantiles // max(len(ranks), 1)).clip(0, quantiles - 1) + 1
            return pd.Series(labels.astype(int).to_numpy(), index=group.index)
        return pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1

    if isinstance(frame.index, pd.MultiIndex) and "date" in frame.index.names:
        frame["bucket"] = frame.groupby(level="date", group_keys=False).apply(assign_bucket)
        grouped = frame.groupby([frame.index.get_level_values("date"), "bucket"])["forward_return"].mean()
        wide = grouped.unstack("bucket")
        wide["long_short"] = wide.get(quantiles, np.nan) - wide.get(1, np.nan)
        return wide

    frame["bucket"] = assign_bucket(frame)
    grouped = frame.groupby("bucket")["forward_return"].agg(["mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "mean_return"})
    return grouped


def walk_forward_evaluate(
    model_class: Callable[..., Any] | Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType | None = None,
    n_splits: int = 5,
    embargo: int | None = None,  # 审计 2026-08-16：默认从特征推断 horizon，避免标签泄漏
    params: Mapping[str, Any] | None = None,
) -> WalkForwardResult:
    """执行 walk-forward 评估。

    该函数比单次验证集更适合金融时间序列: 每一折只用过去训练、未来验证，
    并可设置 embargo 防止未来收益标签重叠。
    """
    _ensure_sklearn()
    X_frame = _fill_numeric_frame(pd.DataFrame(X))
    # P2-Q7-fix (M497): 统一走 _align_target_to_frame —— 行数校验 + 索引不一致告警，
    # 避免 pd.Series(y).reindex(X_frame.index) 在索引不一致时静默产生全 NaN 后
    # dropna 得到空集（walk-forward 静默返回空结果）。
    y_series = _align_target_to_frame(y, X_frame, context="walk_forward_evaluate")
    y_series = y_series.dropna()
    X_frame = X_frame.loc[y_series.index]
    if len(y_series) == 0:
        warnings.warn("walk_forward_evaluate: 对齐后无有效样本，返回空结果", RuntimeWarning)
        return WalkForwardResult(fold_metrics=[], aggregate_metrics={"n_folds": 0, "warning": "empty after alignment"}, task_type=task_type, n_folds=0)
    # 审计 2026-08-16：embargo 未显式给出时按特征 horizon 推断
    if embargo is None:
        embargo = embargo_from_features([str(c) for c in X_frame.columns])
    inferred = _infer_task_type(y_series, task_type)
    splitter = PurgedEmbargoTimeSeriesSplit(n_splits=n_splits, embargo=embargo)
    fold_metrics: list[JsonDict] = []
    for fold_no, (train_idx, test_idx) in enumerate(splitter.split(X_frame), start=1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            warnings.warn(f"walk_forward_evaluate: 第 {fold_no} 折训练/测试集为空，跳过", RuntimeWarning)
            continue
        if isinstance(model_class, type):
            model = model_class(**dict(params or {}))
        elif callable(model_class) and not hasattr(model_class, "fit"):
            model = model_class(**dict(params or {}))
        else:
            model = clone(model_class) if clone is not None else model_class
            if params and hasattr(model, "set_params"):
                model.set_params(**dict(params))
        X_train, X_test = X_frame.iloc[train_idx], X_frame.iloc[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test) if inferred == "classification" and hasattr(model, "predict_proba") else None
        metrics = _score_predictions(inferred, y_test, pred, proba)
        metrics.update(
            {
                "fold": fold_no,
                "train_start": str(X_train.index.min()),
                "train_end": str(X_train.index.max()),
                "test_start": str(X_test.index.min()),
                "test_end": str(X_test.index.max()),
            }
        )
        fold_metrics.append(metrics)

    aggregate: JsonDict = {"n_folds": len(fold_metrics)}
    if fold_metrics:
        frame = pd.DataFrame(fold_metrics)
        metric_cols = [c for c in frame.columns if c not in {"fold", "task_type", "train_start", "train_end", "test_start", "test_end"}]
        for col in metric_cols:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if not values.empty:
                aggregate[f"mean_{col}"] = float(values.mean())
                aggregate[f"std_{col}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return WalkForwardResult(fold_metrics=fold_metrics, aggregate_metrics=aggregate, task_type=inferred, n_folds=len(fold_metrics))


def probability_to_signal(
    scores: pd.Series | np.ndarray,
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
    neutral_value: int = 0,
) -> pd.Series:
    """把连续预测分数转换成 -1/0/1 信号。

    P2-Q7-fix (M500): 该函数直接对分数取分位数，未做概率校准。分类模型的概率输出
    建议先经过 CalibratedClassifierCV（train(calibrate=True)）再传入，否则分位点
    在类别不平衡下不反映真实概率。
    """
    series = pd.Series(scores, name="score").astype(float)
    if series.empty:
        return pd.Series(dtype=int, name="signal")
    if not 0 <= short_quantile < long_quantile <= 1:
        raise ValueError("需要满足 0 <= short_quantile < long_quantile <= 1")
    low = float(series.quantile(short_quantile))
    high = float(series.quantile(long_quantile))
    signal = pd.Series(neutral_value, index=series.index, dtype=int, name="signal")
    signal.loc[series >= high] = 1
    signal.loc[series <= low] = -1
    return signal


def rolling_metric_decay(
    metric_history: pd.Series,
    window: int = 20,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """计算滚动指标衰减，用于 AutoRetrainer 的触发解释。"""
    history = pd.Series(metric_history).dropna().astype(float)
    if history.empty:
        return pd.DataFrame(columns=["metric", "rolling_mean", "decay", "trigger"])
    rolling = history.rolling(window=window, min_periods=max(3, window // 3)).mean()
    decay = rolling - history
    return pd.DataFrame(
        {
            "metric": history,
            "rolling_mean": rolling,
            "decay": decay,
            "trigger": decay > float(threshold),
        }
    )


def align_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
    dropna_target: bool = True,
    fill_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """对齐特征和目标索引。"""
    X = pd.DataFrame(features).copy()
    y = pd.Series(target, name=getattr(target, "name", "target"))
    frame = X.join(y, how="inner")
    if dropna_target:
        frame = frame.dropna(subset=[y.name])
    y_out = frame[y.name]
    X_out = frame.drop(columns=[y.name])
    if fill_features:
        X_out = _fill_numeric_frame(X_out)
    return X_out, y_out


def save_research_snapshot(
    path: str | Path,
    feature_report: FeatureQualityReport | None = None,
    model_card_data: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    """保存一次 ML 研发快照。

    V4.1 feature: 将特征质量、模型卡和人工备注写成 JSON，方便复盘一次实验。
    """
    payload: JsonDict = {"generated_at": _utc_now()}
    if feature_report is not None:
        payload["feature_report"] = asdict(feature_report)
    if model_card_data is not None:
        payload["model_card"] = dict(model_card_data)
    if notes is not None:
        payload["notes"] = notes
    _write_json(Path(path), payload)


def load_research_snapshot(path: str | Path) -> dict:
    """读取研发快照。"""
    return _read_json(Path(path), {})


def estimate_feature_half_life(ic_series: pd.Series, min_periods: int = 5) -> float | None:
    """根据 IC 自相关估算特征半衰期。

    这里采用 AR(1) 近似: half_life = -ln(2) / ln(phi)。当 phi 不在 (0, 1)
    区间时，说明序列没有可解释的指数衰减结构，返回 None。
    """
    series = pd.Series(ic_series).dropna().astype(float)
    if len(series) < min_periods:
        return None
    lagged = series.shift(1).dropna()
    current = series.loc[lagged.index]
    if len(lagged) < min_periods:
        return None
    phi = float(lagged.corr(current))
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return None
    return float(-math.log(2.0) / math.log(phi))


def summarize_feature_catalog(store: FeatureStorePro) -> dict:
    """汇总 FeatureStorePro 目录。"""
    catalog = store.list_features()
    if catalog.empty:
        return {"n_features": 0, "n_active": 0, "categories": {}, "generated_at": _utc_now()}
    categories = catalog.groupby("category", dropna=False)["name"].count().to_dict() if "category" in catalog.columns else {}
    return {
        "n_features": int(len(catalog)),
        "n_active": int(catalog.get("is_active", pd.Series(dtype=bool)).sum()) if "is_active" in catalog else 0,
        "categories": {str(k): int(v) for k, v in categories.items()},
        "latest_computed_at": str(catalog["computed_at"].max()) if "computed_at" in catalog.columns else None,
        "generated_at": _utc_now(),
    }


def summarize_model_registry(registry: ModelRegistry) -> dict:
    """汇总 ModelRegistry 目录。"""
    models = registry.list_models()
    if models.empty:
        return {"n_models": 0, "n_versions": 0, "production_models": [], "generated_at": _utc_now()}
    production = models.loc[models["is_production"], ["name", "version"]].to_dict(orient="records")
    task_counts = models.groupby("task_type", dropna=False)["version"].count().to_dict() if "task_type" in models.columns else {}
    return {
        "n_models": int(models["name"].nunique()),
        "n_versions": int(len(models)),
        "task_counts": {str(k): int(v) for k, v in task_counts.items()},
        "production_models": production,
        "generated_at": _utc_now(),
    }


def export_registry_report(registry: ModelRegistry, path: str | Path) -> None:
    """导出模型注册表报告 JSON。"""
    rows = registry.list_models().to_dict(orient="records")
    payload = {"summary": summarize_model_registry(registry), "models": rows}
    _write_json(Path(path), payload)


def export_feature_catalog_report(store: FeatureStorePro, path: str | Path) -> None:
    """导出特征目录报告 JSON。"""
    rows = store.list_features().to_dict(orient="records")
    payload = {"summary": summarize_feature_catalog(store), "features": rows}
    _write_json(Path(path), payload)


def select_low_correlation_features(
    X: pd.DataFrame,
    importance: pd.Series | None = None,
    corr_threshold: float = 0.9,
    max_features: int | None = None,
) -> list[str]:
    """在高相关特征中保留更重要的一个。"""
    frame = _fill_numeric_frame(pd.DataFrame(X))
    if frame.empty:
        return []
    if importance is None:
        importance = pd.Series(1.0, index=frame.columns)
    importance = pd.Series(importance).reindex(frame.columns).fillna(0.0)
    ordered = list(importance.sort_values(ascending=False).index)
    corr = frame[ordered].corr().abs().fillna(0.0)
    selected: list[str] = []
    for candidate in ordered:
        if all(float(corr.loc[candidate, chosen]) < corr_threshold for chosen in selected):
            selected.append(str(candidate))
        if max_features is not None and len(selected) >= max_features:
            break
    return selected


def make_model_name(prefix: str, target_name: str, horizon: str | int | None = None, model_type: str | None = None) -> str:
    """生成稳定模型名。"""
    parts = [prefix, target_name]
    if horizon is not None:
        parts.append(f"h{horizon}")
    if model_type is not None:
        parts.append(model_type)
    return _safe_name("_".join(str(p) for p in parts if str(p)))


def infer_feature_horizon(name: str) -> int | None:
    """从特征名中推断 horizon，例如 `ret_20d` -> 20。"""
    match = re.search(r"(?:_|^)(\d+)(?:d|day|days)(?:_|$)", str(name).lower())
    if match:
        return int(match.group(1))
    match = re.search(r"(?:h|horizon_?)(\d+)", str(name).lower())
    if match:
        return int(match.group(1))
    return None


def embargo_from_features(feature_names: Sequence[str], default: int = 5) -> int:
    """根据特征名推断 embargo 长度。"""
    horizons = [infer_feature_horizon(name) for name in feature_names]
    valid = [h for h in horizons if h is not None]
    return int(max(valid) if valid else default)


def validate_no_future_leakage(
    feature_names: Sequence[str],
    target_name: str,
    max_allowed_horizon: int | None = None,
) -> dict:
    """基于命名规则做未来函数泄漏的轻量检查。

    这不是形式化证明，但能捕捉常见错误: 特征名里的 forward/future/label，或者
    特征 horizon 大于目标 horizon。
    """
    suspicious_keywords = ["future", "forward", "label", "target", "y_"]
    suspicious: list[str] = []
    for name in feature_names:
        lower = str(name).lower()
        if any(keyword in lower for keyword in suspicious_keywords):
            suspicious.append(str(name))
    target_horizon = infer_feature_horizon(target_name)
    limit = max_allowed_horizon if max_allowed_horizon is not None else target_horizon
    horizon_violations: list[JsonDict] = []
    if limit is not None:
        for name in feature_names:
            horizon = infer_feature_horizon(name)
            if horizon is not None and horizon > int(limit):
                horizon_violations.append({"feature": str(name), "horizon": horizon, "limit": int(limit)})
    ok = not suspicious and not horizon_violations
    return {"ok": ok, "suspicious_features": suspicious, "horizon_violations": horizon_violations}


def drift_ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """计算两个样本经验分布的 KS 距离，不依赖 scipy。"""
    ref = np.sort(pd.Series(reference).dropna().astype(float).to_numpy())
    cur = np.sort(pd.Series(current).dropna().astype(float).to_numpy())
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([ref, cur])))
    ref_cdf = np.searchsorted(ref, values, side="right") / len(ref)
    cur_cdf = np.searchsorted(cur, values, side="right") / len(cur)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """生成特征漂移报告。"""
    ref = pd.DataFrame(reference)
    cur = pd.DataFrame(current)
    common = [c for c in ref.columns if c in cur.columns]
    rows: list[JsonDict] = []
    for col in common:
        stat = drift_ks_statistic(ref[col], cur[col])
        rows.append({"feature": str(col), "ks_stat": stat, "drift_flag": bool(pd.notna(stat) and stat >= threshold)})
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature", "ks_stat", "drift_flag"])


def build_training_manifest(
    feature_names: Sequence[str],
    target_name: str,
    model_type: str,
    params: Mapping[str, Any] | None = None,
) -> dict:
    """生成训练清单，用于审计一次训练任务。"""
    params = dict(params or {})
    leakage = validate_no_future_leakage(feature_names, target_name)
    return {
        "created_at": _utc_now(),
        "feature_names": list(feature_names),
        "target_name": target_name,
        "model_type": model_type,
        "params": params,
        "n_features": len(feature_names),
        "embargo_suggestion": embargo_from_features(feature_names),
        "leakage_check": leakage,
    }


def restore_model_file(registry: ModelRegistry, name: str, version: str, destination: str | Path) -> Path:
    """把注册表中的模型文件复制到目标路径，用于离线部署。"""
    meta = registry.metadata(name, version)
    src = Path(meta.model_path)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = Path(destination)
    if dst.is_dir() or str(destination).endswith(os.sep):
        dst.mkdir(parents=True, exist_ok=True)
        dst = dst / src.name
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def prune_old_model_versions(
    registry: ModelRegistry,
    name: str,
    keep_latest: int = 5,
    remove_files: bool = False,
) -> list[str]:
    """清理旧模型版本，保留生产版本和最近 N 个版本。"""
    item = registry._catalog.get("models", {}).get(name)  # noqa: SLF001 - 本模块内部维护工具
    if not item:
        return []
    versions = sorted(item.get("versions", {}).keys(), key=_version_sort_key)
    production = item.get("production_version")
    if keep_latest <= 0:
        # P2-Q7-fix (L512): keep_latest<=0 时 versions[-0:] 等于全量，protected 覆盖
        # 全部版本，清理函数静默失效；显式报错，避免调用方误以为清理已执行。
        raise ValueError("keep_latest 必须为正整数（至少为 1）")
    protected = set(versions[-int(keep_latest) :])
    if production:
        protected.add(production)
    removed: list[str] = []
    for version in versions:
        if version in protected:
            continue
        if registry.delete_version(name, version, remove_file=remove_files):
            removed.append(version)
    return removed


def ensemble_average_predictions(
    predictions: Mapping[str, pd.Series],
    weights: Mapping[str, float] | None = None,
    rank_average: bool = False,
) -> pd.Series:
    """对多个模型预测做加权平均。

    P2-Q7-fix (L511):
      - 权重为 0 的模型从分母归一化中剔除，避免 fillna(0) 后稀释其他模型权重；
      - 提供 rank_average 选项（各预测先转百分位排名再平均），对预测分布差异大的
        模型更稳健；默认仍为简单加权平均，维持原有行为。
    """
    if not predictions:
        return pd.Series(dtype=float, name="ensemble_prediction")
    frame = pd.concat([pd.Series(v, name=k) for k, v in predictions.items()], axis=1)
    if weights is not None:
        w_all = pd.Series(weights).reindex(frame.columns).fillna(0.0).astype(float)
        if float(w_all.abs().sum()) == 0:
            # 全部权重为 0：回退等权平均（保留全部模型）。
            w = pd.Series(1.0 / frame.shape[1], index=frame.columns)
        else:
            drop = list(w_all[w_all == 0].index)
            if drop:
                warnings.warn(
                    f"ensemble_average_predictions: 以下模型权重为 0，已从集成中剔除: {drop}",
                    RuntimeWarning,
                )
            keep = w_all[w_all != 0]
            frame = frame[keep.index]
            w = keep / keep.sum()
    else:
        w = pd.Series(1.0 / frame.shape[1], index=frame.columns)
    if rank_average:
        frame = frame.rank(pct=True)
    result = frame.mul(w, axis=1).sum(axis=1)
    result.name = "ensemble_prediction"
    return result


def score_to_position_size(
    scores: pd.Series,
    gross_exposure: float = 1.0,
    max_position: float = 0.05,
    dollar_neutral: bool = True,
) -> pd.Series:
    """把模型分数转换成组合权重。"""
    series = pd.Series(scores, name="score").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.Series(dtype=float, name="weight")
    z = series - series.mean()
    if dollar_neutral:
        denom = float(z.abs().sum())
        weights = z / denom * gross_exposure if denom > 0 else z
    else:
        positive = series.clip(lower=0.0)
        denom = float(positive.sum())
        weights = positive / denom * gross_exposure if denom > 0 else positive
    weights = weights.clip(lower=-max_position, upper=max_position)
    denom2 = float(weights.abs().sum())
    if denom2 > 0:
        weights = weights / denom2 * gross_exposure
    # P2-Q7-fix (L506): 先 clip 再整体归一化后，权重可能重新突破 max_position 上限，
    # 风控约束被悄悄绕过。这里再 clip 一次，保证 max_position 为硬约束
    # （gross_exposure 在约束紧绑时可能略低于目标）。
    weights = weights.clip(lower=-max_position, upper=max_position)
    weights.name = "weight"
    return weights


def evaluate_signal_returns(
    scores: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 5,
) -> dict:
    """汇总模型分数对应的收益表现。"""
    ic_daily = daily_information_coefficient(scores, forward_returns)
    spread = top_bottom_spread(scores, forward_returns, quantiles=quantiles)
    result: JsonDict = {"ic": ic_summary(ic_daily)}
    if isinstance(spread, pd.DataFrame) and "long_short" in spread.columns:
        ls = spread["long_short"].dropna()
        result["long_short_mean"] = float(ls.mean()) if not ls.empty else None
        result["long_short_t"] = float(ls.mean() / (ls.std(ddof=1) / math.sqrt(len(ls)))) if len(ls) > 1 and ls.std(ddof=1) > 0 else None
    elif isinstance(spread, pd.DataFrame) and not spread.empty and "mean_return" in spread.columns:
        first = spread.loc[spread["bucket"] == 1, "mean_return"]
        last = spread.loc[spread["bucket"] == quantiles, "mean_return"]
        result["long_short_mean"] = float(last.iloc[0] - first.iloc[0]) if not first.empty and not last.empty else None
    result["spread_table"] = spread
    return result


def safe_model_params(model: Any) -> dict:
    """读取模型参数，无法读取时返回空 dict。"""
    if hasattr(model, "get_params"):
        try:
            return dict(model.get_params())
        except Exception:
            return {}
    return {}


def compact_feature_names(feature_names: Sequence[str], max_items: int = 20) -> str:
    """把特征名列表压缩成日志友好的字符串。"""
    names = [str(name) for name in feature_names]
    if len(names) <= max_items:
        return ", ".join(names)
    head = ", ".join(names[:max_items])
    return f"{head}, ... (+{len(names) - max_items})"


def assert_pipeline_ready(
    feature_store: FeatureStorePro,
    registry: ModelRegistry,
    required_features: Sequence[str] | None = None,
) -> dict:
    """检查 ML 管道的基础目录和依赖状态。"""
    required_features = list(required_features or [])
    available = set(feature_store.list_features().get("name", pd.Series(dtype=str)).astype(str))
    missing = [name for name in required_features if name not in available]
    return {
        "ok": not missing and _SKLEARN_IMPORT_ERROR is None,
        "feature_store_dir": str(feature_store.base_dir),
        "model_registry_dir": str(registry.base_dir),
        "missing_features": missing,
        "sklearn_available": _SKLEARN_IMPORT_ERROR is None,
        "shap_available": shap is not None,
        "lightgbm_available": LGBMClassifier is not None or LGBMRegressor is not None,
        "xgboost_available": XGBClassifier is not None or XGBRegressor is not None,
    }


# ---------------------------------------------------------------------------
# 研发辅助函数
# ---------------------------------------------------------------------------


def build_feature_matrix(store: FeatureStorePro, feature_names: Sequence[str], target_name: str | None = None) -> pd.DataFrame:
    """从 FeatureStorePro 快速拼接特征矩阵。"""
    frames = [store.load_feature(name).rename(name) for name in feature_names]
    matrix = pd.concat(frames, axis=1)
    if target_name is not None:
        matrix[target_name] = store.load_feature(target_name)
    return matrix


def train_test_time_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """时间序列友好的 train/test split。"""
    if not 0 < test_size < 0.9:
        raise ValueError("test_size 必须在 0 和 0.9 之间")
    n = len(X)
    split = max(1, int(n * (1.0 - test_size)))
    if split >= n:
        split = n - 1
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def model_card(registry: ModelRegistry, name: str, version: str = "production") -> dict:
    """导出模型卡片，便于报告和审计。"""
    meta = registry.metadata(name, version)
    metric_name, metric_value, higher = _primary_metric(meta.metrics)
    return {
        "name": meta.name,
        "version": meta.version,
        "task_type": meta.task_type,
        "training_date": meta.training_date,
        "is_production": meta.is_production,
        "primary_metric": metric_name,
        "primary_value": metric_value,
        "higher_is_better": higher,
        "params": meta.params,
        "metrics": meta.metrics,
    }


def compact_cv_results(cv_results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """把 GridSearchCV/RandomizedSearchCV 结果压缩成研究报告友好的表。"""
    if cv_results.empty:
        return cv_results
    cols = [c for c in cv_results.columns if c.startswith("param_")]
    for extra in ["mean_test_score", "std_test_score", "rank_test_score"]:
        if extra in cv_results.columns:
            cols.append(extra)
    out = cv_results[cols].copy()
    if "rank_test_score" in out.columns:
        out = out.sort_values("rank_test_score")
    elif "mean_test_score" in out.columns:
        out = out.sort_values("mean_test_score", ascending=False)
    return out.head(int(top_n)).reset_index(drop=True)


__all__ = [
    "FeatureStorePro",
    "ModelRegistry",
    "AutoRetrainer",
    "ModelExplainer",
    "HyperparameterOptimizer",
    "MLEngineV2",
    "PurgedEmbargoTimeSeriesSplit",
    "build_feature_matrix",
    "train_test_time_split",
    "model_card",
    "compact_cv_results",
]


def deep_factor_ensemble(data: dict | None = None) -> dict:
    """V11 融合: 接入 deep_factors 包（LSTM/Transformer/Autoencoder 深度因子）。

    TensorFlow 未安装时优雅降级——返回可用性说明，不抛异常。
    主栈无 TF 依赖；安装 TF 后自动启用深度因子段。

    Args:
        data: {"prices": DataFrame, "X": DataFrame, "symbols": list, "X_raw": DataFrame}
              省略时返回能力探测结果。

    Returns:
        {"available": bool, "models": [...], "alpha": DataFrame|None, "note": str}
    """
    try:
        from quant_system.deep_factors.ensemble import DeepFactorEnsemble
    except ImportError:
        return {"available": False, "note": "deep_factors 不可用"}
    ens = DeepFactorEnsemble()
    tf_ok = True
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        tf_ok = False
    if not tf_ok:
        return {"available": False, "note": "TensorFlow 未安装，深度因子降级为 sklearn 管线（ml_pipeline_pro 主流程）"}
    if data is None:
        return {"available": True, "note": "TensorFlow 已安装，可调用深度因子"}
    ens.add_lstm(seq_length=20)
    try:
        alpha = ens.predict(data)
        return {"available": True, "models": list(ens._models.keys()),
                "alpha": alpha, "note": "深度因子集成完成"}
    except Exception as e:
        return {"available": False, "note": f"深度因子预测失败: {e}"}


if __name__ == "__main__":  # pragma: no cover - 手工 smoke test 入口
    # V4.1 feature: 保留一个轻量命令行入口，方便研发环境快速验证导入和目录。
    fs = FeatureStorePro()
    mr = ModelRegistry()
    print("FeatureStorePro:", fs.base_dir)
    print("ModelRegistry:", mr.base_dir)
    print("features:", len(fs.list_features()))
    print("models:", len(mr.list_models()))
