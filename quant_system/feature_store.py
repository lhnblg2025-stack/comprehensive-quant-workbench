"""
feature_store.py — V8 特征存储 & 模型漂移检测 & 增量学习

功能:
  1. 特征存储: SQLite 缓存已计算的特征向量，避免重复计算
  2. 漂移检测: 跟踪近期预测准确率 vs 历史基准，发出漂移告警
  3. 增量学习: 支持增量训练 (warm-start RF, LR partial_fit)
"""

from __future__ import annotations

import atexit
import logging
import sqlite3
import threading
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
def _db_path() -> Path:
    """DB 路径: ~/.quant_system 只读时降级 workspace tmp_tx/quant_cache(2026-08-22)."""
    _p = Path.home() / ".quant_system" / "feature_store.db"
    try:
        _p.parent.mkdir(parents=True, exist_ok=True)
        _t = _p.parent / ".wt"; _t.write_text("ok"); _t.unlink()
        return _p
    except OSError:
        _alt = Path(__file__).resolve().parent.parent / "tmp_tx" / "quant_cache"
        _alt.mkdir(parents=True, exist_ok=True)
        return _alt / "feature_store.db"


DB_PATH = _db_path()
MODEL_DIR = Path.home() / ".quant_system"

# 特征集版本：特征定义（FEATURES 列表/计算逻辑）变更时 bump 本版本并全量失效
# P2-Q9-fix (Q9-M536): 特征表新增 feature_set_version 维度并纳入主键，避免
# 特征逻辑变更后旧值被静默覆盖、新旧定义混存。
FEATURE_SET_VERSION = "v1"

# 漂移检测参数
DRIFT_WINDOW = 20           # 近期窗口大小
DRIFT_THRESHOLD = -0.10     # 准确率下降超过 10% 触发告警
MIN_SAMPLES_FOR_DRIFT = 30

# 增量学习参数
INCREMENTAL_BATCH = 100     # 每次增量学习最少样本
# P2-Q9-fix (Q9-L540): MAX_CACHED_FEATURES 原先定义了从未使用（死代码），
# 特征表无容量/清理机制。现由 _prune_features() 使用，写入后按日期裁剪。
MAX_CACHED_FEATURES = 10000

# 回填节流：避免每次 compute_drift 都在数据不足时反复联网回填
_BACKFILL_MIN_INTERVAL = 1800.0   # 两次回填尝试的最小间隔（秒，30 分钟）
_LAST_BACKFILL_TS: dict[str, float] = {}

# P2-Q9-fix (Q9-M541): 模块级单例连接 + 锁，替代"每次 _db() 新建连接且从不
# close"的连接泄漏实现。
_CONN: sqlite3.Connection | None = None
_CONN_LOCK = threading.Lock()


def _init_schema(conn: sqlite3.Connection) -> None:
    """建表（含特征集版本列）并执行旧库迁移。"""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_set_version TEXT NOT NULL DEFAULT 'v1',
            value REAL,
            PRIMARY KEY (symbol, date, feature_name, feature_set_version)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction INTEGER,
            probability REAL,
            actual INTEGER,
            PRIMARY KEY (symbol, date, model_name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            check_date TEXT NOT NULL,
            recent_accuracy REAL,
            baseline_accuracy REAL,
            drift_detected INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incremental_log (
            model_name TEXT NOT NULL,
            date TEXT NOT NULL,
            n_samples INTEGER,
            accuracy_before REAL,
            accuracy_after REAL,
            PRIMARY KEY (model_name, date)
        )
    """)
    conn.commit()
    _migrate_features_table(conn)
    _migrate_drift_log(conn)
    conn.commit()


def _migrate_features_table(conn: sqlite3.Connection) -> None:
    """旧库 features 表无 feature_set_version 列时重建（SQLite 无法改主键）。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(features)").fetchall()}
    if "feature_set_version" in cols:
        return
    conn.execute("""
        CREATE TABLE features_new (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_set_version TEXT NOT NULL DEFAULT 'v1',
            value REAL,
            PRIMARY KEY (symbol, date, feature_name, feature_set_version)
        )
    """)
    conn.execute("""
        INSERT INTO features_new (symbol, date, feature_name, feature_set_version, value)
        SELECT symbol, date, feature_name, 'v1', value FROM features
    """)
    conn.execute("DROP TABLE features")
    conn.execute("ALTER TABLE features_new RENAME TO features")


def _migrate_drift_log(conn: sqlite3.Connection) -> None:
    """旧库 drift_log 以 (model_name, check_date) 为主键、同日检查只留最后一条；
    重建为自增 id 保留全部检查历史。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(drift_log)").fetchall()}
    if "id" in cols:
        return
    conn.execute("""
        CREATE TABLE drift_log_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            check_date TEXT NOT NULL,
            recent_accuracy REAL,
            baseline_accuracy REAL,
            drift_detected INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        INSERT INTO drift_log_new (model_name, check_date, recent_accuracy, baseline_accuracy, drift_detected)
        SELECT model_name, check_date, recent_accuracy, baseline_accuracy, drift_detected FROM drift_log
    """)
    conn.execute("DROP TABLE drift_log")
    conn.execute("ALTER TABLE drift_log_new RENAME TO drift_log")


def _db() -> sqlite3.Connection:
    """Get or create feature store database.

    # P2-Q9-fix (Q9-M541): 原实现每次调用新建连接且从不 close，高频率写入会
    # 累积连接、触发 SQLite 锁与资源泄漏。改为模块级单例连接
    # （check_same_thread=False + busy_timeout 5s），进程退出由 close_db() 关闭。
    """
    global _CONN
    if _CONN is None:
        with _CONN_LOCK:
            if _CONN is None:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                _init_schema(conn)
                _CONN = conn
    return _CONN


def close_db() -> None:
    """显式关闭单例连接（进程退出时调用）。"""
    global _CONN
    with _CONN_LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            finally:
                _CONN = None


atexit.register(close_db)


# ── 特征缓存 ────────────────────────────────────────

def _norm_value(value: Any) -> Any:
    """把特征值归一化为可写 SQLite 的标量。

    # P2-Q9-fix (Q9-M535): NaN/None 存 SQLite NULL，不再静默替换为 0.0，
    # 避免缺失特征与真实 0 值混淆污染训练/信号；非数值类型抛 TypeError（失败可见）。
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"特征值 {value!r} 不是数值，无法缓存") from None
    if pd.isna(f):
        return None
    return f


def cache_feature(symbol: str, date: str, feature_name: str, value: float) -> None:
    """Store a single feature value."""
    db = _db()
    db.execute(
        "INSERT OR REPLACE INTO features (symbol, date, feature_name, feature_set_version, value) VALUES (?, ?, ?, ?, ?)",
        (symbol, date, feature_name, FEATURE_SET_VERSION, _norm_value(value)),
    )
    db.commit()


def cache_features(symbol: str, date: str, features: dict[str, float]) -> None:
    """Store all 60+ features for a symbol/date."""
    db = _db()
    rows = [(symbol, date, name, FEATURE_SET_VERSION, _norm_value(v))
            for name, v in features.items()]
    db.executemany(
        "INSERT OR REPLACE INTO features (symbol, date, feature_name, feature_set_version, value) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    db.commit()
    _prune_features()


def _prune_features(max_entries: int = MAX_CACHED_FEATURES) -> None:
    """按日期裁剪特征表，仅保留最新的 max_entries 条（防长期运行无限膨胀）。

    # P2-Q9-fix (Q9-L540): MAX_CACHED_FEATURES 原先定义了从未使用（死代码），
    # 特征表无容量/清理机制。现每次批量写入后检查并删除最旧日期。
    """
    db = _db()
    total = db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    if total <= max_entries:
        return
    row = db.execute(
        "SELECT date FROM features ORDER BY date DESC LIMIT 1 OFFSET ?",
        (max_entries - 1,),
    ).fetchone()
    if not row:
        return
    cutoff = row[0]
    db.execute("DELETE FROM features WHERE date < ?", (cutoff,))
    db.commit()


def get_cached_features(symbol: str, date: str) -> dict[str, float] | None:
    """Retrieve cached features for a symbol/date. Returns None if not cached."""
    db = _db()
    rows = db.execute(
        "SELECT feature_name, value FROM features WHERE symbol = ? AND date = ? AND feature_set_version = ?",
        (symbol, date, FEATURE_SET_VERSION),
    ).fetchall()
    if not rows:
        return None
    # P2-Q9-fix (Q9-M535): NULL（缺失）返回 NaN，调用方得以区分缺失与真实 0
    out: dict[str, float] = {}
    for name, v in rows:
        out[name] = float(v) if v is not None else float("nan")
    return out


def get_all_cached_dates(symbol: str) -> list[str]:
    """Get all dates with cached features for a symbol."""
    db = _db()
    rows = db.execute(
        "SELECT DISTINCT date FROM features WHERE symbol = ? AND feature_set_version = ? ORDER BY date",
        (symbol, FEATURE_SET_VERSION),
    ).fetchall()
    return [r[0] for r in rows]


# ── 预测记录 ────────────────────────────────────────

def record_prediction(
    symbol: str,
    date: str,
    model_name: str,
    prediction: int,
    probability: float | None = None,
    actual: int | None = None,
) -> None:
    """Record a model prediction for later accuracy tracking."""
    db = _db()
    db.execute(
        """INSERT OR REPLACE INTO predictions
           (symbol, date, model_name, prediction, probability, actual)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (symbol, date, model_name, int(prediction),
         round(float(probability), 4) if probability is not None else None,
         int(actual) if actual is not None else None),
    )
    db.commit()


def update_actual(symbol: str, date: str, model_name: str, actual: int) -> None:
    """Update actual outcome for a previously recorded prediction."""
    db = _db()
    db.execute(
        "UPDATE predictions SET actual = ? WHERE symbol = ? AND date = ? AND model_name = ?",
        (int(actual), symbol, date, model_name),
    )
    db.commit()


# ── 预测记录（集成自愈）─────────────────────────────

# 与 ml_signals._make_labels 一致的三重障碍标签口径（±2%，horizon=20）
DEFAULT_HORIZON_DAYS = 20
UP_THRESHOLD = 2.0
DOWN_THRESHOLD = -2.0

# 训练/预测常用股票池（与 ml_signals.train_ensemble 默认池一致）
DEFAULT_SYMBOLS = [
    "600519", "601288", "601398", "601939", "601988", "600036", "601166",
    "600900", "601318", "600276", "600887", "601888", "600585", "601668",
    "600028", "601857", "600030", "601211", "600837", "601688",
    "000002", "000001", "000651", "000333", "000858", "000568",
    "002415", "002714", "002475", "002304", "002230",
    "300750", "300059", "300124", "300274", "300760",
    "601012", "600809", "600438", "600309", "600690",
    "601225", "601088", "600031", "600104", "600019",
    "000725", "000538", "000063", "002352",
]


def _classify_forward_ret(future_ret: float) -> int:
    """把前向收益(%)映射为三重障碍标签 {0:跌, 1:盘, 2:涨}（与 ml_signals 一致）。"""
    if future_ret > UP_THRESHOLD:
        return 2
    if future_ret < DOWN_THRESHOLD:
        return 0
    return 1


def _load_ensemble() -> dict[str, Any] | None:
    """加载磁盘 ensemble 模型字典（ml_signals.train_ensemble 保存格式）。"""
    pkl_path = MODEL_DIR / "ml_ensemble.pkl"
    if not pkl_path.exists():
        return None
    try:
        import pickle
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        if not isinstance(d, dict) or not isinstance(d.get("models"), dict):
            logger.warning("model_registry: %s 格式异常，跳过加载", pkl_path)
            return None
        return d
    except Exception as e:
        logger.warning("加载 ensemble 失败: %s", e)
        return None


def _fetch_history(symbol: str) -> pd.DataFrame | None:
    """获取 A股前复权日线（与 ml_signals.train_ensemble 同数据源）。"""
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                            start_date="20230101", adjust="qfq")
    if df is None or len(df) < 300:
        return None
    return df


def _ensemble_proba(X_row: np.ndarray, ensemble: dict[str, Any]) -> np.ndarray | None:
    """单行特征 → 集成三分类概率 [跌,盘,涨]（与 ml_signals.predict_ensemble 一致）。"""
    models = ensemble.get("models", {})
    weights = ensemble.get("weights", {})
    wp = np.zeros(3)
    tw = 0.0
    for mn in ("xgb", "rf", "lr"):
        m = models.get(mn)
        w = weights.get(mn, 0.0)
        if m is None or w <= 0.0 or not hasattr(m, "predict_proba"):
            continue
        try:
            proba_raw = m.predict_proba(np.asarray(X_row, dtype=float).reshape(1, -1))[0]
        except Exception as e:
            logger.warning("ensemble 子模型 %s predict_proba 失败: %s", mn, e)
            continue
        try:
            from quant_system.ml_signals import _pad_proba_3class
            pa = _pad_proba_3class(proba_raw, m)[0]
        except Exception:
            pa = np.zeros(3)
            n = min(len(proba_raw), 3)
            pa[:n] = proba_raw[:n]
        wp[:3] += pa[:3] * w
        tw += w
    if tw <= 0:
        return None
    return wp / tw


def backfill_predictions(
    model_name: str = "ensemble",
    symbols: list[str] | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    verbose: bool = True,
) -> dict[str, Any]:
    """用磁盘 ensemble + 历史行情回填 predictions 表（含已实现标签 actual）。

    # P1-Q9-fix (H05): record_prediction()/update_actual() 全仓无调用方，
    # predictions 表恒空 → compute_drift 恒返回 insufficient_samples、增量学习
    # 恒为空，漂移检测/增量学习整条链路是死功能。本函数在 feature_store 内部
    # 自给自足：用训练管线同数据源（akshare 日线）对每个标的生成逐日预测 +
    # 前向收益标签，并同步缓存特征（供 get_training_samples 关联），使整条
    # 链路真正可用。

    Returns: {status, n_recorded, n_symbols, n_dates, n_errors, errors}
    """
    ensemble = _load_ensemble()
    if ensemble is None:
        return {"status": "skipped", "reason": "no_ensemble_model",
                "detail": f"{MODEL_DIR}/ml_ensemble.pkl 不存在，请先训练 ensemble"}
    # 审计 2026-08-16：只回填训练截止日之后的样本（样本外），避免把训练期
    # 预测当漂移依据；无 training_date 时保守跳过（不假装样本外）。
    training_date = str((ensemble.get("training_date") or "")[:10])
    if not training_date:
        return {"status": "skipped", "reason": "no_training_date",
                "detail": "ensemble 元数据缺少 training_date，无法区分样本内/外，拒绝样本内回填"}
    syms = symbols or DEFAULT_SYMBOLS
    n_recorded = 0
    n_symbols = 0
    n_dates = 0
    n_errors = 0
    errors: list[str] = []
    seen_dates: set[str] = set()
    # 训练标签 {-1,0,1} → predictions 表 pred 编码 {0,1,2}（与模型 classes_ 一致）
    _tri_to_code = {-1: 0, 0: 1, 1: 2}
    for sym in syms:
        try:
            df = _fetch_history(sym)
        except Exception as e:
            n_errors += 1
            if len(errors) < 10:
                errors.append(f"{sym}: fetch failed: {e}")
            continue
        if df is None or len(df) < 300:
            continue
        try:
            from quant_system.ml_signals import extract_features, FEATURES
            X, yc, _ = extract_features(df.to_dict("records"))  # yc 与训练同款三重障碍标签
        except Exception as e:
            n_errors += 1
            if len(errors) < 10:
                errors.append(f"{sym}: extract failed: {e}")
            continue
        if len(X) == 0:
            continue
        dates = df["日期"].astype(str).str[:10].tolist()
        closes = df["收盘"].astype(float).tolist()
        n_local = 0
        for i in range(len(X)):
            date = dates[i]
            if date <= training_date:
                continue  # 审计 2026-08-16：跳过训练期样本，只用样本外
            proba = _ensemble_proba(X[i], ensemble)
            if proba is None:
                continue
            pred = int(np.argmax(proba))
            actual = None
            # 仅当未来窗口完整时写 actual，且用训练同款三重障碍标签映射
            if i + horizon_days < len(closes):
                actual = _tri_to_code.get(int(yc[i]))
            feat_dict = {FEATURES[j]: float(X[i][j])
                         for j in range(min(len(FEATURES), X.shape[1]))}
            try:
                cache_features(sym, date, feat_dict)
                record_prediction(sym, date, model_name, pred,
                                  float(np.max(proba)), actual)
                n_recorded += 1
                n_local += 1
                seen_dates.add(date)
            except Exception as e:
                n_errors += 1
                if len(errors) < 10:
                    errors.append(f"{sym}@{date}: record failed: {e}")
        if n_local > 0:
            n_symbols += 1
    if verbose:
        print(f"[feature_store] backfill {model_name}: {n_recorded} predictions "
              f"({n_symbols} symbols, {n_errors} errors)")
    return {"status": "done", "n_recorded": n_recorded, "n_symbols": n_symbols,
            "n_dates": len(seen_dates), "n_errors": n_errors, "errors": errors}


def backfill_actuals(
    model_name: str = "ensemble",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    verbose: bool = True,
) -> dict[str, Any]:
    """为 predictions 表中 actual 为 NULL 的记录回填已实现标签。

    # P1-Q9-fix (H05): 配合 backfill_predictions，用真实行情计算前向收益并
    # 更新 actual，使 compute_drift 的准确率统计真正可以生效（失败可见，不静默）。
    """
    db = _db()
    rows = db.execute(
        "SELECT DISTINCT symbol, date FROM predictions "
        "WHERE model_name = ? AND actual IS NULL ORDER BY date",
        (model_name,),
    ).fetchall()
    updated = 0
    skipped = 0
    n_errors = 0
    errors: list[str] = []
    per_symbol: dict[str, list[str]] = {}
    for sym, dt in rows:
        per_symbol.setdefault(sym, []).append(dt)
    for sym, dt_list in per_symbol.items():
        try:
            df = _fetch_history(sym)
        except Exception as e:
            n_errors += 1
            if len(errors) < 10:
                errors.append(f"{sym}: fetch failed: {e}")
            continue
        if df is None:
            skipped += len(dt_list)
            continue
        dates = df["日期"].astype(str).str[:10].tolist()
        closes = df["收盘"].astype(float).tolist()
        # 审计 2026-08-16：actual 改用训练同款三重障碍标签（而非固定 ±2% 简单收益）
        try:
            from quant_system.ml_signals import extract_features
            _, yc, _ = extract_features(df.to_dict("records"))
            # extract_features 行从 LOOKBACK 开始，与 dates 前 LOOKBACK 行错位
            lookback = len(dates) - len(yc)
            yc_map = {-1: 0, 0: 1, 1: 2}
            yc_by_date = {}
            for k, lab in enumerate(yc):
                yc_by_date[dates[lookback + k]] = yc_map.get(int(lab))
        except Exception:
            yc_by_date = {}
        pos_map = {d: i for i, d in enumerate(dates)}
        for dt in dt_list:
            i = pos_map.get(dt)
            if i is None or i + horizon_days >= len(closes):
                skipped += 1
                continue
            actual = yc_by_date.get(dt)
            if actual is None:
                skipped += 1
                continue
            update_actual(sym, dt, model_name, actual)
            updated += 1
    if verbose:
        print(f"[feature_store] backfill_actuals {model_name}: {updated} updated, "
              f"{skipped} skipped, {n_errors} errors")
    return {"status": "done", "updated": updated, "skipped": skipped,
            "n_errors": n_errors, "errors": errors}


def _ensure_predictions(model_name: str = "ensemble",
                        force: bool = False) -> dict[str, Any]:
    """惰性自愈：predictions 已实现样本不足时自动回填（失败可见，不静默）。

    # P1-Q9-fix (H05): compute_drift/get_training_samples/incremental_train
    # 在数据不足时触发本函数，使漂移检测与增量学习链路不再恒为空。
    """
    db = _db()
    n_labeled = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_name = ? AND actual IS NOT NULL",
        (model_name,),
    ).fetchone()[0]
    if n_labeled >= MIN_SAMPLES_FOR_DRIFT and not force:
        return {"status": "ok", "backfilled": False, "n_labeled": n_labeled}
    if _load_ensemble() is None:
        return {"status": "skipped", "backfilled": False, "n_labeled": n_labeled,
                "reason": "no_ensemble_model"}
    # 节流：上次回填尝试后短时间内不重复联网（失败也会更新此时间戳）
    now = _time.time()
    last = _LAST_BACKFILL_TS.get(model_name, 0.0)
    if not force and now - last < _BACKFILL_MIN_INTERVAL:
        return {"status": "throttled", "backfilled": False,
                "n_labeled": n_labeled, "reason": "recent_backfill_attempt"}
    _LAST_BACKFILL_TS[model_name] = now
    result = backfill_predictions(model_name=model_name, verbose=False)
    result["n_labeled_before"] = n_labeled
    return result


# ── 漂移检测 ────────────────────────────────────────

def compute_drift(model_name: str = "ensemble",
                  self_heal: bool = True) -> dict[str, Any]:
    """Check if model accuracy has drifted.

    # P1-Q9-fix (H05): 增加 self_heal 惰性自愈——当 predictions 表已实现样本
    # 不足时先尝试回填（依赖磁盘 ensemble + 行情），并把回填结果显式写入返回
    # 字典的 backfill 字段，不再恒返回 insufficient_samples。

    Returns:
      - drift_detected: bool
      - recent_accuracy: float (last DRIFT_WINDOW predictions)
      - baseline_accuracy: float (all prior predictions)
      - delta: float (recent - baseline)
      - n_samples: int
    """
    heal: dict[str, Any] = {}
    if self_heal:
        heal = _ensure_predictions(model_name)

    db = _db()
    rows = db.execute(
        "SELECT date, prediction, actual FROM predictions WHERE model_name = ? AND actual IS NOT NULL ORDER BY date",
        (model_name,),
    ).fetchall()

    if len(rows) < MIN_SAMPLES_FOR_DRIFT:
        return {
            "drift_detected": False,
            "reason": "insufficient_samples",
            "n_samples": len(rows),
            "recent_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "delta": 0.0,
            "backfill": heal,
        }

    accuracies = [1.0 if p == a else 0.0 for _, p, a in rows]
    # P2-Q9-fix (Q9-M537): baseline 排除最近 DRIFT_WINDOW 条（即"近期窗口"
    # 本身），否则近期表现被自身污染、delta 被稀释，与 docstring 声称的
    # "all prior predictions" 不符。
    if len(accuracies) > DRIFT_WINDOW:
        baseline = float(np.mean(accuracies[:-DRIFT_WINDOW]))
    else:
        baseline = float(np.mean(accuracies))

    # Recent window
    recent = accuracies[-DRIFT_WINDOW:]
    recent_acc = float(np.mean(recent))

    delta = recent_acc - baseline
    drifted = delta < DRIFT_THRESHOLD

    # Log drift check
    today = datetime.now().strftime("%Y-%m-%d")
    # P2-Q9-fix (Q9-L542): 主键改为自增 id（见 _migrate_drift_log），同一
    # 自然日多次检查全部保留历史，不再 INSERT OR REPLACE 只留最后一次。
    db.execute(
        """INSERT INTO drift_log
           (model_name, check_date, recent_accuracy, baseline_accuracy, drift_detected)
           VALUES (?, ?, ?, ?, ?)""",
        (model_name, today, round(recent_acc, 4), round(baseline, 4), 1 if drifted else 0),
    )
    db.commit()

    return {
        "drift_detected": drifted,
        "recent_accuracy": round(recent_acc, 4),
        "baseline_accuracy": round(baseline, 4),
        "delta": round(delta, 4),
        "n_samples": len(rows),
    }


def get_drift_history(model_name: str = "ensemble") -> list[dict[str, Any]]:
    """Get full drift check history."""
    db = _db()
    rows = db.execute(
        "SELECT check_date, recent_accuracy, baseline_accuracy, drift_detected FROM drift_log WHERE model_name = ? ORDER BY id DESC",
        (model_name,),
    ).fetchall()
    return [
        {
            "date": r[0],
            "recent_accuracy": r[1],
            "baseline_accuracy": r[2],
            "drift_detected": bool(r[3]),
        }
        for r in rows
    ]


# ── 增量学习 ────────────────────────────────────────

def _resolve_feature_order(model_name: str = "ensemble") -> list[str]:
    """解析构造训练样本所用的特征顺序。

    # P2-Q9-fix (Q9-M538): 仅当取 ensemble 样本时采用磁盘模型训练时保存的
    # feature_names，并与当前 quant_system.ml_signals.FEATURES 校验对齐；
    # 不一致时可见告警，避免特征顺序/长度变更后与已训练模型静默错位。
    # 其他 model_name 无训练时特征记录，回退当前 FEATURES 顺序。
    """
    from quant_system.ml_signals import FEATURES
    stored = None
    if model_name == "ensemble":
        ensemble = _load_ensemble()
        if ensemble is not None and isinstance(ensemble, dict):
            stored = ensemble.get("feature_names")
    if not stored:
        return list(FEATURES)
    stored = list(stored)
    if stored != list(FEATURES):
        logger.warning(
            "特征顺序/长度与当前 FEATURES 不一致（模型保存 %d 项 vs 当前 %d 项），"
            "按模型训练时保存的 feature_names 构造样本", len(stored), len(FEATURES),
        )
    return stored


def get_training_samples(
    symbols: list[str] | None = None,
    days_back: int = 60,
    min_samples: int = 100,
    model_name: str = "ensemble",
) -> tuple[np.ndarray, np.ndarray]:
    """Get feature matrix and labels from stored predictions.

    # P2-Q9-fix (Q9-M538): model_name 改为参数（原先硬编码 'ensemble'，其他
    # 模型永远取不到样本）；import 移到函数外；特征顺序按训练时保存的
    # feature_names 校验对齐。

    Returns (X, y) where X = feature vectors, y = classification labels.
    """
    # P1-Q9-fix (H05): 空表自愈——先确保 predictions/features 有样本，
    # 避免增量学习链路恒为空（无模型时 _ensure_predictions 会 visible 跳过）。
    _ensure_predictions(model_name)

    db = _db()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    if symbols:
        placeholders = ",".join("?" * len(symbols))
        preds = db.execute(
            f"""SELECT p.symbol, p.date, p.prediction, p.actual
                FROM predictions p
                WHERE p.model_name = ? AND p.actual IS NOT NULL
                AND p.date >= ? AND p.symbol IN ({placeholders})
                ORDER BY p.date""",
            [model_name, cutoff] + symbols,
        ).fetchall()
    else:
        preds = db.execute(
            """SELECT p.symbol, p.date, p.prediction, p.actual
               FROM predictions p
               WHERE p.model_name = ? AND p.actual IS NOT NULL
               AND p.date >= ? ORDER BY p.date""",
            (model_name, cutoff),
        ).fetchall()

    if len(preds) < min_samples:
        return np.array([]), np.array([])

    # 特征顺序：优先用训练时保存的 feature_names，与当前 FEATURES 对齐校验
    feature_order = _resolve_feature_order(model_name)

    # Get feature vectors for each (symbol, date)
    X_list, y_list = [], []
    for sym, dt, _, actual in preds:
        feats = get_cached_features(sym, dt)
        if feats is None:
            continue
        vec = []
        for f in feature_order:
            v = feats.get(f, 0.0)
            # P2-Q9-fix (Q9-M535): 缓存缺失以 NULL 存储、读取为 NaN，此处
            # 显式把缺失（NaN）按 0 填充进训练向量（文档化的加载期插补，
            # 与"缓存层不落 0"区分开）。
            if v is None or (isinstance(v, float) and pd.isna(v)):
                vec.append(0.0)
            else:
                vec.append(float(v))
        X_list.append(vec)
        y_list.append(int(actual))

    if len(X_list) < min_samples:
        return np.array([]), np.array([])

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int32)


def incremental_train(
    model_name: str = "xgb",
    symbols: list[str] | None = None,
    days_back: int = 60,
) -> dict[str, Any]:
    """Use the real online XGBoost continuation path on cached samples."""
    # P1-Q9-fix (H05): 空表自愈（get_training_samples 内部也会触发）
    _ensure_predictions("ensemble")

    X, y = get_training_samples(symbols, days_back, model_name="ensemble")
    if len(X) < INCREMENTAL_BATCH:
        return {
            "status": "skipped",
            "reason": f"insufficient samples ({len(X)} < {INCREMENTAL_BATCH})",
            "n_samples": len(X),
        }

    # Compute accuracy before
    pkl_path = MODEL_DIR / "ml_ensemble.pkl"
    acc_before = 0.0
    if pkl_path.exists():
        import pickle
        try:
            with open(pkl_path, "rb") as f:
                ensemble = pickle.load(f)
            models = ensemble.get("models", {})
            m = models.get(model_name)
            if m is not None:
                preds = m.predict(X)
                acc_before = float(np.mean(preds == y))
        except Exception as e:
            # P2-Q9-fix (Q9-L539): 原实现静默吞异常；改为可见日志
            logger.warning("incremental_train accuracy_before 计算失败: %s", e)

    # The online model is shared by the signal service.  It persists its label
    # map and visibly falls back to a full retrain when a warm start is unsafe.
    from quant_system.ml_signals import online_partial_fit
    update = online_partial_fit(X, y)
    result = {
        "status": update.get("status", "error"),
        "model": model_name,
        "n_samples": len(X),
        "accuracy_before": round(acc_before, 4),
        "update": update,
    }

    # Log the incremental attempt
    today = datetime.now().strftime("%Y-%m-%d")
    db = _db()
    db.execute(
        "INSERT OR REPLACE INTO incremental_log (model_name, date, n_samples, accuracy_before, accuracy_after) VALUES (?, ?, ?, ?, ?)",
        (model_name, today, len(X), round(acc_before, 4), float(update.get("accuracy_after", 0.0) or 0.0)),
    )
    db.commit()

    return result


# ── 特征存储统计 ────────────────────────────────────

def feature_store_stats() -> dict[str, Any]:
    """Get statistics about the feature store."""
    db = _db()
    n_features = db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    n_unique_symbols = db.execute("SELECT COUNT(DISTINCT symbol) FROM features").fetchone()[0]
    n_unique_dates = db.execute("SELECT COUNT(DISTINCT date) FROM features").fetchone()[0]
    n_predictions = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_labeled = db.execute("SELECT COUNT(*) FROM predictions WHERE actual IS NOT NULL").fetchone()[0]
    drift_count = db.execute("SELECT COUNT(*) FROM drift_log WHERE drift_detected = 1").fetchone()[0]

    return {
        "n_features": n_features,
        "n_symbols": n_unique_symbols,
        "n_dates": n_unique_dates,
        "n_predictions": n_predictions,
        "n_labeled_predictions": n_labeled,
        "drift_alerts": drift_count,
    }


# ── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        stats = feature_store_stats()
        print(f"特征存储统计:")
        print(f"  特征条目: {stats['n_features']}")
        print(f"  标的:     {stats['n_symbols']}")
        print(f"  日期:     {stats['n_dates']}")
        print(f"  预测记录: {stats['n_predictions']} (已标注: {stats['n_labeled_predictions']})")
        print(f"  漂移告警: {stats['drift_alerts']}")
    elif cmd == "drift":
        result = compute_drift()
        print(f"漂移检测 ({result.get('n_samples', 0)} 样本):")
        print(f"  近期准确率: {result['recent_accuracy']:.2%}")
        print(f"  基线准确率: {result['baseline_accuracy']:.2%}")
        print(f"  差值:       {result['delta']:+.2%}")
        print(f"  漂移:       {'⚠️ 是' if result['drift_detected'] else '✅ 否'}")
        if result.get("backfill"):
            bf = result["backfill"]
            print(f"  回填状态:  {bf.get('status')} ({bf.get('reason', '')} "
                  f"记录={bf.get('n_recorded', 0)} 已有标签={bf.get('n_labeled', 0)})")
    elif cmd == "backfill":
        # P1-Q9-fix (H05): 提供显式 CLI 回填入口，接通 predictions/actuals 表
        model_name = sys.argv[2] if len(sys.argv) > 2 else "ensemble"
        r = backfill_predictions(model_name=model_name)
        print(f"回填预测: {r.get('status')} 记录={r.get('n_recorded', 0)} "
              f"标的={r.get('n_symbols', 0)} 日期={r.get('n_dates', 0)} "
              f"错误={r.get('n_errors', 0)}")
        if r.get("reason"):
            print(f"  原因: {r.get('reason')} {r.get('detail', '')}")
        if r.get("errors"):
            for e in r["errors"][:5]:
                print(f"  ⚠ {e}")
        r2 = backfill_actuals(model_name=model_name)
        print(f"回填实际: 更新={r2.get('updated', 0)} 跳过={r2.get('skipped', 0)} "
              f"错误={r2.get('n_errors', 0)}")
    else:
        print("用法: python3 -m quant_system.feature_store [status|drift|backfill]")
