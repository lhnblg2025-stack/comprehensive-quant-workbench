"""
model_registry.py — V5.1 ML模型注册表与性能监控

管理训练好的ML模型的注册、版本控制、性能跟踪和自动降级。
每当我们用 ml_signals.py 训练新模型时，注册到此模块，
实现: 模型版本管理 / 性能衰减监控 / 模型回滚 / 自动重新训练触发。
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
REGISTRY_DB = Path.home() / ".quant_system" / "model_registry.db"

# 模型名合法字符集（用于 model_id 生成，防注入/异常名）
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# ── 数据库初始化 ──

def _get_db() -> sqlite3.Connection:
    """获取或创建模型注册库数据库。"""
    REGISTRY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(REGISTRY_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            version INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            params_json TEXT,
            metrics_json TEXT,
            feature_names TEXT,
            n_features INTEGER,
            train_symbols TEXT,
            train_period TEXT,
            model_blob BLOB,
            hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            evaluation_date TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            ic REAL,
            icir REAL,
            sharpe REAL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            n_samples INTEGER,
            decay_score REAL,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'warning',
            created_at TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0,
            resolved_at TEXT
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


# ── 注册 ──

def register_model(
    model_name: str,
    algorithm: str,
    model_obj: Any,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    feature_names: list[str] | None = None,
    train_symbols: list[str] | None = None,
    train_period: str | None = None,
) -> dict[str, Any]:
    """注册新模型或创建新版本。
    
    Args:
        model_name: 模型名, 如 "xgboost_ensemble_v1"
        algorithm: 算法名, 如 "XGBoost/随机森林混合"
        model_obj: 可pickle的模型对象
        params: 超参数
        metrics: 评估指标 {ic, sharpe, ...}
        feature_names: 特征名列表
        train_symbols: 训练股票
        train_period: 训练期 "2025-01-01~2025-12-31"
    
    Returns:
        {model_id, version, status}
    """
    # P2-Q9-fix (Q9-L551): 校验 model_name 字符集与 feature_names 一致性
    if not model_name or not _MODEL_NAME_RE.fullmatch(model_name):
        raise ValueError(
            f"model_name 含非法字符（仅允许字母/数字/_/-）: {model_name!r}"
        )
    feature_names = list(feature_names or [])
    if feature_names and isinstance(model_obj, dict):
        obj_n = model_obj.get("n_features")
        if obj_n is not None and int(obj_n) != len(feature_names):
            raise ValueError(
                f"feature_names 数量({len(feature_names)}) 与模型声明的 "
                f"n_features({obj_n}) 不一致"
            )
        obj_fn = model_obj.get("feature_names")
        if obj_fn and list(obj_fn) != feature_names:
            raise ValueError("feature_names 与模型对象内保存的特征名不一致")

    db = _get_db()
    now = datetime.now(CST).isoformat()

    # 模型序列化
    blob = pickle.dumps(model_obj)

    # 简单的哈希校验
    import hashlib
    model_hash = hashlib.sha256(blob).hexdigest()[:16]

    # P2-Q9-fix (Q9-M543/M544): 版本分配与插入原子化——
    # ① BEGIN IMMEDIATE 串行化并发注册，IntegrityError 重试，杜绝并发同名
    #    注册产生相同 model_id 主键冲突崩溃；
    # ② 先插入新版本成功，再停用旧版本；任一失败整体回滚，避免 INSERT 失败
    #    时出现"无活跃模型"状态。
    for attempt in range(5):
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT MAX(version) FROM models WHERE model_name = ?",
                (model_name,),
            ).fetchone()
            version = (row[0] or 0) + 1
            model_id = f"{model_name}_v{version}"
            db.execute(
                """INSERT INTO models
                   (model_id, model_name, version, algorithm, status,
                    created_at, updated_at,
                    params_json, metrics_json,
                    feature_names, n_features,
                    train_symbols, train_period,
                    model_blob, hash)
                   VALUES (?, ?, ?, ?, 'active',
                           ?, ?,
                           ?, ?,
                           ?, ?,
                           ?, ?,
                           ?, ?)""",
                (
                    model_id, model_name, version, algorithm,
                    now, now,
                    json.dumps(params or {}), json.dumps(metrics or {}),
                    json.dumps(feature_names), len(feature_names),
                    json.dumps(train_symbols or []), train_period or "",
                    blob, model_hash,
                ),
            )
            db.execute(
                "UPDATE models SET status = 'superseded' "
                "WHERE model_name = ? AND status = 'active' AND model_id != ?",
                (model_name, model_id),
            )
            db.commit()
            break
        except sqlite3.IntegrityError:
            db.rollback()
            if attempt == 4:
                raise
            continue

    return {
        "model_id": model_id,
        "model_name": model_name,
        "version": version,
        "algorithm": algorithm,
        "status": "active",
        "hash": model_hash,
    }


# ── 查询 ──

def get_active_model(model_name: str) -> dict[str, Any] | None:
    """获取指定名称的当前活跃模型。"""
    db = _get_db()
    row = db.execute(
        """SELECT model_id, model_name, version, algorithm, status,
                  created_at, updated_at, params_json, metrics_json,
                  feature_names, n_features, train_symbols, train_period, hash
           FROM models
           WHERE model_name = ? AND status = 'active'
           ORDER BY version DESC LIMIT 1""",
        (model_name,),
    ).fetchone()
    
    if not row:
        return None
    
    return {
        "model_id": row[0],
        "model_name": row[1],
        "version": row[2],
        "algorithm": row[3],
        "status": row[4],
        "created_at": row[5],
        "updated_at": row[6],
        "params": json.loads(row[7]) if row[7] else {},
        "metrics": json.loads(row[8]) if row[8] else {},
        "feature_names": json.loads(row[9]) if row[9] else [],
        "n_features": row[10],
        "train_symbols": json.loads(row[11]) if row[11] else [],
        "train_period": row[12],
        "hash": row[13],
    }


def get_model_by_id(model_id: str) -> dict[str, Any] | None:
    """按ID获取模型元数据。"""
    db = _get_db()
    row = db.execute(
        """SELECT model_id, model_name, version, algorithm, status,
                  created_at, updated_at, params_json, metrics_json,
                  feature_names, n_features, train_symbols, train_period, hash
           FROM models WHERE model_id = ?""",
        (model_id,),
    ).fetchone()

    if not row:
        return None

    # P2-Q9-fix (Q9-L549): 查询列补上 hash，与 get_active_model() 返回结构一致
    return {
        "model_id": row[0], "model_name": row[1],
        "version": row[2], "algorithm": row[3],
        "status": row[4], "created_at": row[5],
        "updated_at": row[6], "params": json.loads(row[7]) if row[7] else {},
        "metrics": json.loads(row[8]) if row[8] else {},
        "feature_names": json.loads(row[9]) if row[9] else [],
        "n_features": row[10],
        "train_symbols": json.loads(row[11]) if row[11] else [],
        "train_period": row[12],
        "hash": row[13],
    }


def load_model(model_id: str) -> Any | None:
    """加载模型对象（校验 sha256 hash，异常打日志而非静默）。"""
    db = _get_db()
    row = db.execute(
        "SELECT model_blob, hash FROM models WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    blob, expected_hash = row[0], row[1]
    # P2-Q9-fix (Q9-M545): 注册时算了 hash 却从未校验——模型损坏/被篡改无法
    # 发现。加载后重算 sha256 并比对，不一致拒绝加载（失败可见）。
    if expected_hash:
        import hashlib
        actual_hash = hashlib.sha256(blob).hexdigest()[:16]
        if actual_hash != expected_hash:
            logger.warning(
                "model_registry: %s 模型 hash 校验失败 (期望 %s, 实际 %s)，拒绝加载",
                model_id, expected_hash, actual_hash,
            )
            return None
    try:
        return pickle.loads(blob)
    except Exception as e:
        logger.warning("model_registry: 加载 %s 模型对象失败: %s", model_id, e)
        return None


def list_models(model_name: str | None = None,
                status: str | None = None,
                limit: int = 20) -> list[dict[str, Any]]:
    """列出已注册模型。
    
    Args:
        model_name: 过滤模型名
        status: 过滤状态 (active/superseded/archived/failed)
        limit: 最大返回数
    """
    db = _get_db()
    query = """SELECT model_id, model_name, version, algorithm, status,
                      created_at, updated_at
               FROM models WHERE 1=1"""
    params = []
    
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    if status:
        query += " AND status = ?"
        params.append(status)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    rows = db.execute(query, params).fetchall()
    return [
        {
            "model_id": r[0], "model_name": r[1],
            "version": r[2], "algorithm": r[3],
            "status": r[4], "created_at": r[5], "updated_at": r[6],
        }
        for r in rows
    ]


# ── 性能跟踪 ──

def record_performance(
    model_id: str,
    metrics: dict[str, float],
    n_samples: int | None = None,
) -> dict[str, Any]:
    """记录一次模型性能评估。"""
    db = _get_db()
    # P2-Q9-fix (Q9-M548): evaluation_date 精确到秒时间戳，同一天多次评估可区分；
    # detect_decay 按自然日聚合窗口（见下）。
    now = datetime.now(CST).isoformat(timespec="seconds")
    ic = metrics.get("ic", 0)
    icir = metrics.get("icir", 0)
    sharpe = metrics.get("sharpe", 0)
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    
    db.execute(
        """INSERT INTO model_performance
           (model_id, evaluation_date, metrics_json,
            ic, icir, sharpe, accuracy, precision, recall, n_samples)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_id, now, json.dumps(metrics),
         ic, icir, sharpe, accuracy, precision, recall, n_samples or 0),
    )
    db.commit()
    
    return {"model_id": model_id, "date": now, "ic": ic, "icir": icir}


def get_performance_history(model_id: str,
                            n_days: int = 90) -> pd.DataFrame:
    """获取模型性能历史。"""
    db = _get_db()
    cutoff = (datetime.now() - timedelta(days=n_days)).strftime("%Y-%m-%d")
    rows = db.execute(
        """SELECT evaluation_date, ic, icir, sharpe, accuracy,
                  precision, recall, n_samples
           FROM model_performance
           WHERE model_id = ? AND evaluation_date >= ?
           ORDER BY evaluation_date""",
        (model_id, cutoff),
    ).fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=[
        "date", "ic", "icir", "sharpe", "accuracy",
        "precision", "recall", "n_samples",
    ])
    return df


# ── 性能衰减检测 ──

def detect_decay(model_id: str,
                 window: int = 20,
                 threshold: float = -0.3) -> dict[str, Any]:
    """检测模型性能是否在衰减。
    
    Args:
        model_id: 模型ID
        window: 评估窗口（最近N次）
        threshold: IC变化阈值（低于此值报警）
    
    Returns:
        {decayed, recent_ic, old_ic, ic_change, alert}
    """
    hist = get_performance_history(model_id, n_days=365)
    if len(hist) < window * 2:
        return {"decayed": False, "reason": "insufficient_data"}

    # P2-Q9-fix (Q9-M548): evaluation_date 已是时间戳，按自然日聚合 IC
    # （同一天多次评估取均值），避免记录频率影响窗口语义、随调用频率失真。
    daily = hist.copy()
    daily["date"] = daily["date"].astype(str).str[:10]
    daily = daily.groupby("date", as_index=False)["ic"].mean().sort_values("date")
    if len(daily) < window * 2:
        return {"decayed": False, "reason": "insufficient_data"}

    recent = daily["ic"].tail(window).mean()
    older = daily["ic"].iloc[-window*2:-window].mean() if len(daily) >= window*2 else recent
    ic_change = recent - older

    decayed = ic_change < threshold

    result = {
        "decayed": decayed,
        "recent_ic": round(float(recent), 4),
        "old_ic": round(float(older), 4),
        "ic_change": round(float(ic_change), 4),
        "threshold": threshold,
    }

    if decayed:
        # P2-Q9-fix (Q9-M546): 同一模型同类型未解决告警存在时不再重复创建，
        # 避免持续衰减期间高频调用产生告警风暴。
        db = _get_db()
        existing = db.execute(
            "SELECT COUNT(*) FROM model_alerts WHERE model_id = ? "
            "AND alert_type = 'decay' AND acknowledged = 0",
            (model_id,),
        ).fetchone()[0]
        if existing == 0:
            alert = create_alert(
                model_id, "decay",
                f"IC {ic_change:.2f} < {threshold:.1f} (衰减)",
                severity="warning",
            )
            result["alert"] = alert
        else:
            result["alert_suppressed"] = True

    return result


# ── 告警 ──

def create_alert(model_id: str,
                 alert_type: str,
                 message: str,
                 severity: str = "warning") -> dict[str, Any]:
    """创建模型告警。"""
    db = _get_db()
    now = datetime.now(CST).isoformat()
    db.execute(
        """INSERT INTO model_alerts
           (model_id, alert_type, message, severity, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (model_id, alert_type, message, severity, now),
    )
    db.commit()
    return {
        "model_id": model_id,
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "created_at": now,
    }


def get_alerts(model_id: str | None = None,
               unacknowledged_only: bool = True,
               limit: int = 20) -> list[dict[str, Any]]:
    """获取告警列表。"""
    db = _get_db()
    query = """SELECT id, model_id, alert_type, message, severity,
                      created_at, acknowledged, resolved_at
               FROM model_alerts WHERE 1=1"""
    params = []
    
    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)
    if unacknowledged_only:
        query += " AND acknowledged = 0"
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    rows = db.execute(query, params).fetchall()
    return [
        {
            "id": r[0], "model_id": r[1], "type": r[2],
            "message": r[3], "severity": r[4],
            "created_at": r[5], "acknowledged": bool(r[6]),
            "resolved_at": r[7],
        }
        for r in rows
    ]


def acknowledge_alert(alert_id: int) -> bool:
    """确认告警。"""
    db = _get_db()
    # P2-Q9-fix (Q9-L550): total_changes 依赖"连接内累计变更数"，新连接下
    # 模式脆弱易误判；改用 cursor.rowcount 判断是否真正更新了行。
    cur = db.execute(
        "UPDATE model_alerts SET acknowledged = 1 WHERE id = ?",
        (alert_id,),
    )
    db.commit()
    return cur.rowcount > 0


# ── 模型回滚 ──

def rollback(model_name: str, target_version: int | None = None) -> dict[str, Any]:
    """回滚到指定版本。

    # P2-Q9-fix (Q9-M547):
    # ① 目标版本不存在时 UPDATE 影响 0 行却不报错——现先校验存在性，返回 error；
    # ② 只允许回滚到 superseded 版本，archived/failed 状态拒绝激活；
    # ③ 目标已是活跃版本时直接返回 already_active，不再 superseded→active 产生
    #    无意义告警；并返回实际影响行数。

    Args:
        model_name: 模型名
        target_version: 目标版本号, None=回退到上一版本

    Returns:
        {model_id, version, status}
    """
    db = _get_db()

    if target_version is None:
        # 获取上一个非活跃版本
        row = db.execute(
            """SELECT model_id, version FROM models
               WHERE model_name = ? AND status = 'superseded'
               ORDER BY version DESC LIMIT 1""",
            (model_name,),
        ).fetchone()
        if not row:
            return {"error": f"No superseded version for {model_name}"}
        target_version = row[1]

    trow = db.execute(
        "SELECT model_id, version, status FROM models WHERE model_name = ? AND version = ?",
        (model_name, target_version),
    ).fetchone()
    if not trow:
        return {"error": f"Version {target_version} not found for {model_name}"}
    if trow[2] == "active":
        return {"status": "already_active", "model_name": model_name,
                "target_version": target_version, "model_id": trow[0]}
    if trow[2] != "superseded":
        return {"error": f"Version {target_version} 状态为 {trow[2]}，不可回滚激活（仅 superseded 可回滚）"}

    # 激活目标版本
    cur = db.execute(
        "UPDATE models SET status = 'active', updated_at = ? WHERE model_name = ? AND version = ?",
        (datetime.now(CST).isoformat(), model_name, target_version),
    )
    affected = cur.rowcount

    # 停用其他活跃版本
    db.execute(
        "UPDATE models SET status = 'superseded' WHERE model_name = ? AND status = 'active' AND version != ?",
        (model_name, target_version),
    )
    db.commit()

    # 创建回滚告警
    active = get_active_model(model_name)
    if active:
        create_alert(
            active["model_id"], "rollback",
            f"回滚到 v{target_version}",
            severity="info",
        )

    return {
        "model_name": model_name,
        "target_version": target_version,
        "status": "rolled_back",
        "affected_rows": affected,
    }


# ── 与训练管线的集成 ──────────────────────────────────
# # P1-Q9-fix (H06): 本模块原先全仓无 import 调用方（仅 __init__.py 列入
# __all__）——训练管线 ml_signals.train_ensemble 直接存 .pkl，ml_pipeline_pro
# 用另一套 ModelRegistry，本模块的版本管理/性能监控/回滚全部悬空，属未接线
# 孤岛。以下函数以训练管线落盘的磁盘产物为输入，把每次训练登记为不可变版本，
# 并可把注册库中的版本原子写回磁盘（在线管线始终读 ~/.quant_system/ml_ensemble.pkl），
# 从而让本注册库成为真实存储/版本管理层。

ENSEMBLE_PKL = Path.home() / ".quant_system" / "ml_ensemble.pkl"
ENSEMBLE_JSON = Path.home() / ".quant_system" / "ml_ensemble.json"


def _file_hash(path: Path) -> str:
    """SHA256 文件哈希（前 16 位），用于幂等判断。"""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ensemble_meta_from_disk() -> dict[str, Any] | None:
    """读取 ml_ensemble.json 元数据（可能缺失/损坏，返回 None）。"""
    if not ENSEMBLE_JSON.exists():
        return None
    try:
        return json.loads(ENSEMBLE_JSON.read_text("utf-8"))
    except Exception:
        return None


def _cv_aggregate(cv_scores: Any) -> dict[str, float]:
    """把 CV 折分数聚合成注册表指标（空数据时可见地返回 0，不静默）。"""
    if not cv_scores:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "sharpe": 0.0}
    acc = [float(s.get("accuracy", 0.0)) for s in cv_scores if isinstance(s, dict)]
    prec = [float(s.get("precision", s.get("precision_score", 0.0))) for s in cv_scores if isinstance(s, dict)]
    rec = [float(s.get("recall", s.get("recall_score", 0.0))) for s in cv_scores if isinstance(s, dict)]
    f1 = [float(s.get("f1", 0.0)) for s in cv_scores if isinstance(s, dict)]
    # V11 审计修复（High）: 原实现 precision/recall 都取 mean(f1)（假值），sharpe 恒 0。
    # 修正: 优先读真实 precision/recall 字段；缺失时用 accuracy 兜底（比 f1 冒充诚实）；
    # sharpe 从 sharpe_ratio 字段读取（缺失返回 0）。
    sharpe = [float(s.get("sharpe_ratio", s.get("sharpe", 0.0))) for s in cv_scores if isinstance(s, dict)]
    return {
        "accuracy": round(float(np.mean(acc)), 4) if acc else 0.0,
        "precision": round(float(np.mean(prec)), 4) if prec else (round(float(np.mean(acc)), 4) if acc else 0.0),
        "recall": round(float(np.mean(rec)), 4) if rec else (round(float(np.mean(acc)), 4) if acc else 0.0),
        "sharpe": round(float(np.mean(sharpe)), 4) if sharpe else 0.0,
    }


def register_ensemble_from_disk(
    model_name: str = "ensemble",
    algorithm: str = "XGBoost/随机森林/逻辑回归混合",
    force: bool = False,
) -> dict[str, Any]:
    """把磁盘上的 ml_ensemble.pkl（训练管线产物）注册进模型注册库。

    # P1-Q9-fix (H06): 以训练管线落盘产物为输入登记不可变版本（自动停用
    # 旧版）。幂等：pkl 文件哈希与当前活跃版本的 disk_hash 一致且未 force
    # 时跳过，避免重复登记。
    """
    if not ENSEMBLE_PKL.exists():
        return {"error": f"{ENSEMBLE_PKL} 不存在，请先运行 ml_signals.train_ensemble()"}
    pkl_hash = _file_hash(ENSEMBLE_PKL)
    active = get_active_model(model_name)
    if active and not force and active.get("params", {}).get("disk_hash") == pkl_hash:
        return {"status": "up_to_date", "model_id": active["model_id"],
                "version": active["version"], "hash": pkl_hash}

    try:
        with open(ENSEMBLE_PKL, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        return {"error": f"读取 {ENSEMBLE_PKL} 失败: {e}"}
    if not isinstance(obj, dict) or "models" not in obj:
        return {"error": f"{ENSEMBLE_PKL} 不是 ensemble 格式（缺 models 键）"}

    meta = _ensemble_meta_from_disk() or {}
    feature_names = obj.get("feature_names", []) if isinstance(obj, dict) else []
    cv_scores = (meta.get("cv_scores") or (obj.get("cv_scores") if isinstance(obj, dict) else []))
    metrics = _cv_aggregate(cv_scores)
    metrics["n_samples"] = int(meta.get("n_samples") or obj.get("n_samples", 0))
    params = {
        "disk_hash": pkl_hash,
        "training_date": meta.get("training_date", ""),
        "n_samples": meta.get("n_samples", 0),
        "n_symbols": meta.get("n_symbols", 0),
        "n_features": meta.get("n_features", len(feature_names)),
        "ensemble_weights": meta.get("ensemble_weights", {}),
        "cv_scores": cv_scores,
    }
    return register_model(
        model_name=model_name,
        algorithm=algorithm,
        model_obj=obj,
        params=params,
        metrics=metrics,
        feature_names=list(feature_names),
        train_symbols=[],
        train_period=str(meta.get("training_date", "")),
    )


def load_ensemble_from_registry(model_name: str = "ensemble") -> dict[str, Any] | None:
    """从注册库加载当前活跃 ensemble（{models, weights, feature_names, ...}）。

    # P1-Q9-fix (H06): 与 ml_signals.predict_ensemble 读取的 pkl 字典格式一致，
    为后续将推理路径切到注册库提供官方入口。
    """
    active = get_active_model(model_name)
    if active is None:
        return None
    obj = load_model(active["model_id"])
    if isinstance(obj, dict) and "models" in obj:
        return obj
    return None


def restore_ensemble_to(model_name: str = "ensemble",
                        target_version: int | None = None) -> dict[str, Any]:
    """把注册库中某版本回滚并原子写回磁盘，供在线管线加载。

    # P1-Q9-fix (H06): 在线管线 ml_signals.predict_ensemble 始终读
    # ~/.quant_system/ml_ensemble.pkl；仅翻转 DB 状态不会改变线上行为。
    # 本函数从 DB 取回目标版本 blob 原子写回（tmp+replace），再更新
    # ml_ensemble.json 元数据，使版本回滚真正作用于线上模型。
    """
    active = get_active_model(model_name)
    if active is None:
        return {"error": f"注册库中没有 {model_name}，请先 sync"}
    if target_version is None:
        roll_res = rollback(model_name, None)
        if roll_res.get("error"):
            return roll_res
        active2 = get_active_model(model_name)
        model_id = active2["model_id"]
        version = active2["version"]
    else:
        rows = list_models(model_name=model_name, status=None, limit=100)
        cand = [r for r in rows if r["version"] == target_version]
        if not cand:
            return {"error": f"未找到 {model_name} v{target_version}"}
        roll_res = rollback(model_name, target_version)
        # V11 审计修复（Medium）: 原实现不检查 rollback 返回值——target 是
        # archived/failed 时 rollback 返回 error，仍继续覆盖线上 pkl（DB 与磁盘不一致）。
        if isinstance(roll_res, dict) and roll_res.get("error"):
            return {"error": f"回滚失败: {roll_res['error']}（目标版本可能已归档/失败）"}
        model_id = cand[0]["model_id"]
        version = target_version

    obj = load_model(model_id)
    if obj is None:
        return {"error": f"加载 {model_id} 模型对象失败"}
    if not isinstance(obj, dict) or "models" not in obj:
        return {"error": f"{model_id} 不是 ensemble 格式，拒绝覆盖线上 pkl"}

    # 原子写回线上 pkl（tmp + replace，避免半写文件被管线读到）
    tmp = ENSEMBLE_PKL.with_name(ENSEMBLE_PKL.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(ENSEMBLE_PKL)

    meta_db = get_model_by_id(model_id)
    md = {
        "status": "restored",
        "training_date": (meta_db or {}).get("params", {}).get("training_date", ""),
        "n_samples": (meta_db or {}).get("params", {}).get("n_samples", 0),
        "n_symbols": (meta_db or {}).get("params", {}).get("n_symbols", 0),
        "n_features": (meta_db or {}).get("params", {}).get("n_features", 0),
        "ensemble_weights": (meta_db or {}).get("params", {}).get("ensemble_weights", {}),
        "cv_scores": (meta_db or {}).get("params", {}).get("cv_scores", []),
        "models_trained": [],
        "restore_source": f"{model_id}",
    }
    ENSEMBLE_JSON.write_text(json.dumps(md, ensure_ascii=False, indent=2), "utf-8")
    return {"status": "restored", "model_id": model_id, "version": version,
            "target": str(ENSEMBLE_PKL)}


def sync_from_disk(model_name: str = "ensemble",
                   force: bool = False) -> dict[str, Any]:
    """一键集成：把磁盘最新 ensemble 注册进注册库（哈希变化才登记）。

    # P1-Q9-fix (H06): 官方接线入口，供调度/CLI 在训练完成后调用。
    """
    return register_ensemble_from_disk(model_name=model_name, force=force)


# ── 仪表盘 ──

def registry_dashboard() -> dict[str, Any]:
    """模型注册表总览。"""
    db = _get_db()
    
    # 统计
    total = db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    active = db.execute(
        "SELECT COUNT(*) FROM models WHERE status = 'active'"
    ).fetchone()[0]
    alive = db.execute(
        "SELECT COUNT(*) FROM models WHERE status IN ('active', 'superseded')"
    ).fetchone()[0]
    
    # 按算法分布
    algo_dist = db.execute(
        """SELECT algorithm, COUNT(*) as cnt
           FROM models WHERE status IN ('active', 'superseded')
           GROUP BY algorithm ORDER BY cnt DESC"""
    ).fetchall()
    
    # 未处理告警
    unacked = db.execute(
        "SELECT COUNT(*) FROM model_alerts WHERE acknowledged = 0"
    ).fetchone()[0]
    
    alerts_detected = db.execute(
        "SELECT COUNT(*) FROM model_alerts WHERE alert_type = 'decay' AND acknowledged = 0"
    ).fetchone()[0]
    
    # 最近活跃模型
    recent = db.execute(
        """SELECT model_id, model_name, version, algorithm, created_at
           FROM models WHERE status = 'active'
           ORDER BY created_at DESC LIMIT 10"""
    ).fetchall()
    
    return {
        "total_models": total,
        "active_models": active,
        "total_registered": alive,
        "algo_distribution": [{"algo": a[0], "count": a[1]} for a in algo_dist],
        "unacknowledged_alerts": unacked,
        "decay_alerts": alerts_detected,
        "recent_active": [
            {"model_id": r[0], "name": r[1], "version": r[2],
             "algorithm": r[3], "created_at": r[4]}
            for r in recent
        ],
    }


# ── CLI ──

def main():
    """CLI：模型注册表操作。

    # P1-Q9-fix (H06): 新增 sync/list/rollback/restore/dashboard 子命令，
    # 提供与训练管线的官方接线入口（register_ensemble_from_disk 等）。
    """
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "sync":
            force = "--force" in sys.argv
            r = sync_from_disk(force=force)
            print(f"sync_from_disk: {r}")
            return
        if cmd == "list":
            rows = list_models(status=None, limit=50)
            print(f"共 {len(rows)} 条记录:")
            for r in rows:
                print(f"  {r['model_id']:<40} {r['algorithm']:<30} "
                      f"{r['status']:<12} {r['created_at']}")
            return
        if cmd == "rollback":
            name = sys.argv[2] if len(sys.argv) > 2 else "ensemble"
            r = rollback(name)
            print(f"rollback: {r}")
            return
        if cmd == "restore":
            name = sys.argv[2] if len(sys.argv) > 2 else "ensemble"
            ver = int(sys.argv[3]) if len(sys.argv) > 3 else None
            r = restore_ensemble_to(name, ver)
            print(f"restore: {r}")
            return
        if cmd == "dashboard":
            dash = registry_dashboard()
            print(f"总览: {dash['total_models']} 模型, {dash['active_models']} 活跃")
            for r in dash["recent_active"]:
                print(f"  {r['name']} v{r['version']} ({r['algorithm']})")
            return
        if cmd == "selftest":
            pass  # 走下方原有自测
        else:
            print(f"未知命令: {cmd}（可用: sync|list|rollback|restore|dashboard|selftest）")
            return

    # 原有自测
    import tempfile
    # W2.5 P2: 自测写临时库，不污染生产 REGISTRY_DB（原 dummy 模型会以 status=active 落生产库）
    REGISTRY_DB = Path(tempfile.mkdtemp()) / "model_registry_selftest.db"
    print("=== 模型注册表 ===")
    print(f"数据库(临时): {REGISTRY_DB}")

    # 注册测试
    reg = register_model(
        model_name="xgboost_test",
        algorithm="XGBoost",
        model_obj={"dummy": "model"},
        params={"n_estimators": 100, "max_depth": 5},
        metrics={"ic": 0.05, "sharpe": 1.2},
        feature_names=["mom_6m", "roe", "vol_20d"],
        train_period="2025-01~2025-06",
    )
    print(f"\n注册: {reg}")

    # 仪表盘
    dash = registry_dashboard()
    print(f"\n总览: {dash['total_models']} 模型, {dash['active_models']} 活跃")

    # 性能记录
    perf = record_performance(reg["model_id"], {"ic": 0.04, "sharpe": 1.0}, n_samples=100)
    print(f"性能记录: {perf}")

    # 衰减检测
    for ic_val in [0.05, 0.04, 0.03, 0.02, 0.01]:
        record_performance(reg["model_id"], {"ic": ic_val, "sharpe": ic_val * 20}, n_samples=100)
    decay = detect_decay(reg["model_id"], window=3)
    print(f"衰减检测: {decay}")

    print("\nOK")


if __name__ == "__main__":
    main()
