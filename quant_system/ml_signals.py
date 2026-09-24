"""
ML 信号模块 v3 — 全指标 + SHAP + 相关性分析。

v3.1 additions: walk-forward validation, Optuna/GridSearch tuning,
triple-barrier labels, probability calibration, feature drift checks,
rolling feature updates, and optional online updates.

50+技术指标 + SHAP贡献度 + 特征相关性矩阵 + 交互特征。

用法:
  python3 -m quant_system.ml_signals --train          # 训练含SHAP+相关性
  python3 -m quant_system.ml_signals --importance     # 展示全部分析报告
  python3 -m quant_system.ml_signals --predict 600519 # 个股预测（集成模型）
  python3 -m quant_system.ml_signals --shap           # 触发完整训练并生成SHAP分析（当前实现）
  python3 -m quant_system.ml_signals --correlation    # 触发完整训练并生成相关性分析（当前实现）

  P2-Q8-fix (L529): --shap / --correlation 当前与 --train 共享完整 train_model()
  路径（内部一并产出 SHAP 与相关性），并非"仅 SHAP/仅相关性"；此处如实标注。
"""

from __future__ import annotations
import logging

import json, math, sys, time as _time, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
CST = timezone(timedelta(hours=8))

# ── 60个特征名称 ─────────────────────────────────────────
FEATURES = [
    # RSI系列
    "rsi6","rsi12","rsi24",
    # KDJ
    "kdj_k","kdj_d","kdj_j",
    # Williams %R
    "wr10","wr20",
    # CCI
    "cci20",
    # ROC
    "roc5","roc10","roc20",
    # BIAS
    "bias5","bias10","bias20",
    # PSY
    "psy12","psy24",
    # ADX/DMI
    "adx14","pdi14","mdi14",
    # MACD
    "macd_dif","macd_dea","macd_hist",
    # 均线偏离度
    "pct_ma5","pct_ma10","pct_ma20","pct_ma60","pct_ma144","pct_ma300",
    # 均线交叉
    "cross_5_20","cross_20_60","cross_60_144",
    # EXPMA
    "expma12_dist","expma50_dist",
    # P2-Q8-fix (L528): 删除冗余特征 boll_b（=boll_pos.copy() 完全重复）
    # BOLL
    "boll_pos","boll_width",
    # ATR
    "atr_pct","atr_ratio",
    # 量能
    "vol_ratio_5_20","vol_ratio_5_60","obv_slope","vr","mfi",
    # 价态
    "range_pct","gap_pct","pos_20d","change_5d",
    # P2-Q8-fix (L528): 删除冗余特征 vol_ratio（=std20/std60，与 vol_regime 近重复）
    # 波动率
    "vol_regime",
    # 交互特征
    "rsi_x_macd","ma20_x_ma60","ma60_x_ma144","vol_x_change","rsi_x_cci",
    # 日历
    "month_sin","month_cos","is_monday","is_friday",
    # 统计
    "ret_skew","ret_kurt","ret_ac1",
]
N_FEATURES = len(FEATURES)

# 单一共享股票池（A股主要蓝筹 + 银行，50 只）。
# P2-Q8-fix (M519/M520): 统一 v3 train_model / v7 train_ensemble / auto_retrain
# 的默认训练池，避免"自动重训后模型股票域漂移、与初始模型不可比"。
DEFAULT_TRAIN_SYMBOLS = [
    "600519","601288","601398","601939","601988","600036","601166",
    "600900","601318","600276","600887","601888","600585","601668",
    "600028","601857","600030","601211","600837","601688",
    "000002","000001","000651","000333","000858","000568",
    "002415","002714","002475","002304","002230",
    "300750","300059","300124","300274","300760",
    "601012","600809","600438","600309","600690",
    "601225","601088","600031","600104","600019",
    "000725","000538","000063","002352",
]


def _quant_cache_dir() -> Path:
    """Return the quant-system cache directory, creating it if needed.
    2026-08-22: ~/.quant_system 只读时降级到 workspace tmp_tx/quant_cache(可写)."""
    cache_dir = Path.home() / ".quant_system"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _t = cache_dir / ".wt"; _t.write_text("ok"); _t.unlink()
        return cache_dir
    except OSError:
        cache_dir = Path(__file__).resolve().parent.parent / "tmp_tx" / "quant_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


def _encode_labels(y: Any) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Encode arbitrary integer labels into contiguous sklearn/XGBoost labels."""
    arr = np.asarray(y).astype(int)
    classes = sorted(int(v) for v in np.unique(arr))
    label_to_code = {label: i for i, label in enumerate(classes)}
    code_to_label = {i: label for label, i in label_to_code.items()}
    encoded = np.array([label_to_code[int(v)] for v in arr], dtype=int)
    return encoded, label_to_code, code_to_label


def _default_xgb_params() -> dict[str, Any]:
    """Default conservative XGBoost parameters used when tuning is unavailable."""
    return {
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }


def label_triple_barrier(close: Any, pt_sl: tuple[float, float] = (1.02, 0.98), max_hold: int = 20,
                         high: Any = None, low: Any = None, width: Any = None) -> np.ndarray:
    """Create triple-barrier labels.

    Labels are 1 when the profit-taking barrier is touched first, -1 when the
    stop-loss barrier is touched first, and 0 when neither barrier is touched
    before ``max_hold`` bars. The scan moves forward from every observation and
    preserves first-touch ordering.

    P2-Q8-fix (M514): 触障检测从 close-only 改为优先使用 ``high``/``low``（提供时），
    与"价格先触上/下轨"的真实语义一致；障碍宽度可由 ``width``（标量或逐行数组）
    动态设定 —— 默认仍用 ``pt_sl`` 固定 ±2%，保持向后兼容。
    """
    try:
        prices = np.asarray(close, dtype=float)
        n = len(prices)
        labels = np.zeros(n, dtype=int)
        horizon = max(1, int(max_hold))
        hi = np.asarray(high, dtype=float) if high is not None else prices
        lo = np.asarray(low, dtype=float) if low is not None else prices
        w_arr = np.asarray(width, dtype=float) if width is not None else None
        for i in range(n):
            if not np.isfinite(prices[i]) or prices[i] <= 1e-10:
                continue
            if w_arr is not None:
                w = float(w_arr[i])
                if not np.isfinite(w) or w <= 0:
                    continue  # 无效宽度：不生成标签（保持 0 = 未触障）
                upper = prices[i] * (1.0 + w)
                lower = prices[i] * (1.0 - w)
            else:
                upper = prices[i] * float(pt_sl[0])
                lower = prices[i] * float(pt_sl[1])
            end = min(n, i + horizon + 1)
            for j in range(i + 1, end):
                if np.isfinite(hi[j]) and hi[j] >= upper:
                    labels[i] = 1
                    break
                if np.isfinite(lo[j]) and lo[j] <= lower:
                    labels[i] = -1
                    break
        return labels
    except Exception:
        return np.zeros(len(close) if hasattr(close, "__len__") else 0, dtype=int)


def walk_forward_validate(X: Any, y: Any, n_train: int = 252, n_test: int = 63,
                          groups: Any = None, n_splits: int = 5,
                          embargo: int = 20) -> dict[str, Any]:
    """Run walk-forward / grouped validation and return per-window metrics.

    When ``groups`` (per-row stock/panel ids) is provided, uses a
    **per-group time-series split** instead of random ``GroupKFold``:
    each stock's rows (assumed time-ascending in the concatenated array) are
    split into contiguous folds, and for fold k the training set is that
    stock's folds [0, k-1], test is fold k. This enforces both:
      - no same-stock future leakage (train strictly before test per stock);
      - embargo removes the last `embargo` rows from each stock's training
        tail so labels with forward horizon (triple-barrier max_hold=20)
        do not overlap the test window.

    Without ``groups``, uses sklearn ``TimeSeriesSplit`` with
    ``max_train_size=n_train`` and ``test_size=n_test`` plus a 20-bar embargo
    to approximate 252-day train / 63-day validation windows.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)
        if len(X_arr) < 2 or len(np.unique(y_arr)) < 2:
            return {"windows": [], "summary": {}, "error": "insufficient data"}

        # 审计 2026-08-16：分组 CV 从随机 GroupKFold 升级为“每组时间序列 + purge/embargo”。
        # 当前 X 由各股票按时间升序拼接（extract_features/训练管线保证），因此组内顺序即时间顺序。
        if groups is not None:
            g_arr = np.asarray(groups)
            if len(g_arr) != len(X_arr) or len(np.unique(g_arr)) < 2:
                return {"windows": [], "summary": {}, "error": "insufficient groups"}
            cv_type = "group_timeseries_purged"
            # 每个组按行序切成 n_splits 个连续时间块；fold k = 前 k 块训练、第 k 块测试
            n_groups = len(np.unique(g_arr))
            n_folds = max(2, min(int(n_splits), n_groups))
            # 收集每个组的样本位置
            group_pos: dict[int, list[int]] = {}
            for pos, g in enumerate(g_arr):
                group_pos.setdefault(int(g), []).append(pos)
            split_pairs = []
            # 每折：每只股票取前 k 块为训练、第 k 块为测试（k 从 1 开始，保证训练非空）
            for k in range(1, n_folds):
                tr_all: list[int] = []
                te_all: list[int] = []
                for g, positions in group_pos.items():
                    arr = np.asarray(positions)
                    if len(arr) < n_folds:
                        # 组样本太少：整组进训练，避免测试空
                        tr_all.extend(arr.tolist())
                        continue
                    # 等分索引（按行序=时间序）
                    bounds = np.linspace(0, len(arr), n_folds + 1, dtype=int)
                    te = arr[bounds[k]:bounds[k+1]]
                    tr = np.concatenate([arr[bounds[j]:bounds[j+1]] for j in range(k)])
                    # purge/embargo：剔除训练尾部 embargo 行，防止标签与测试窗口重叠
                    if len(tr) > embargo:
                        tr = tr[:-embargo]
                    tr_all.extend(tr.tolist())
                    te_all.extend(te.tolist())
                if tr_all and te_all and len(np.unique(y_arr[tr_all])) >= 2:
                    split_pairs.append((np.asarray(tr_all), np.asarray(te_all)))
        else:
            from sklearn.model_selection import TimeSeriesSplit
            cv_type = "timeseries"
            if len(X_arr) < n_train + n_test:
                return {"windows": [], "summary": {}, "error": "insufficient data"}
            n_folds = max(2, (len(X_arr) - n_train) // n_test)
            splitter = TimeSeriesSplit(n_splits=n_folds, max_train_size=n_train, test_size=n_test)
            # P1-Q8-fix (H04): embargo 必须 >= 标签最大前视（三重障碍 max_hold=20）。
            # 原 embargo=5 < 20 → 训练集末尾 15 行标签依赖测试窗价格，CV 指标虚高。
            embargo = 20  # 防止 label 前向收益重叠（>= 标签水平线 20）
            split_pairs = []
            for tr_idx, te_idx in splitter.split(X_arr):
                if len(tr_idx) == 0 or len(te_idx) == 0:
                    continue
                # 剪除训练集末尾 embargo 个点，避免 label 时效重叠
                if len(tr_idx) > embargo:
                    tr_idx = tr_idx[:-embargo]
                if len(tr_idx) >= 2:
                    split_pairs.append((tr_idx, te_idx))

        windows: list[dict[str, Any]] = []
        for fold, (tr_idx, te_idx) in enumerate(split_pairs):
            if len(tr_idx) == 0 or len(te_idx) == 0 or len(np.unique(y_arr[tr_idx])) < 2:
                continue
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_arr[tr_idx], y_arr[tr_idx])
            pred = model.predict(X_arr[te_idx])
            pos = np.where(pred == 1, 1.0, np.where(pred == -1, -1.0, 0.0))
            realized = np.where(y_arr[te_idx] == 1, 1.0, np.where(y_arr[te_idx] == -1, -1.0, 0.0))
            strat_ret = pos * realized
            sharpe = 0.0
            if np.std(strat_ret) > 1e-10:
                sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252))
            # P2-Q8-fix (L523): 该 sharpe 只是"预测类与真实类一致率"的代理指标
            # （pos=±1 直接当仓位、无成本/滑点/收益幅度），不得与策略 Sharpe 混读。
            sharpe_note = "label_agreement_proxy: mean(pos*realized)/std*sqrt(252)，未计成本/滑点/幅度"
            windows.append({
                "fold": int(fold),
                "cv_type": cv_type,
                "train_start": int(tr_idx[0]),
                "train_end": int(tr_idx[-1]),
                "test_start": int(te_idx[0]),
                "test_end": int(te_idx[-1]),
                "accuracy": round(float(accuracy_score(y_arr[te_idx], pred)), 4),
                "precision": round(float(precision_score(y_arr[te_idx], pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_arr[te_idx], pred, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y_arr[te_idx], pred, average="weighted", zero_division=0)), 4),
                "sharpe": round(sharpe, 4),
                "sharpe_note": sharpe_note,
            })

        summary: dict[str, dict[str, float]] = {}
        for key in ["accuracy", "precision", "recall", "f1", "sharpe"]:
            vals = np.array([w[key] for w in windows], dtype=float)
            if len(vals):
                summary[key] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        if summary:
            logger.info(f"[walk-forward ({cv_type})] avg +/- std: " + ", ".join(
                # P2-Q8-fix (L523): sharpe 明确标注为 label-agreement 代理指标，避免与策略 Sharpe 混读
                f"{'sharpe(label-agreement)' if k == 'sharpe' else k}={v['mean']:.4f}±{v['std']:.4f}"
                for k, v in summary.items()
            ))
        return {"windows": windows, "summary": summary, "cv_type": cv_type}
    except Exception as e:
        return {"windows": [], "summary": {}, "error": str(e)}


def tune_hyperparams(X_train: Any, y_train: Any, X_val: Any, y_val: Any, n_trials: int = 100) -> dict[str, Any]:
    """Tune XGBoost hyperparameters by validation AUC.

    Optuna is used when available. If Optuna is unavailable or fails, the
    function silently degrades to a coarse ``GridSearchCV``. Best parameters are
    saved to ``~/.quant_system/best_params.json``.
    """
    from sklearn.metrics import roc_auc_score

    best_params = _default_xgb_params()
    cache_file = _quant_cache_dir() / "best_params.json"
    try:
        X_tr = np.asarray(X_train, dtype=float)
        X_va = np.asarray(X_val, dtype=float)
        # Encode labels on training data only to avoid leaking validation class information
        y_tr_arr = np.asarray(y_train).astype(int)
        y_va_arr = np.asarray(y_val).astype(int)
        y_tr_enc, label_to_code, _ = _encode_labels(y_tr_arr)
        # Map validation labels using the same encoding; fallback to 0 for unseen labels
        y_va = np.array([label_to_code.get(int(v), 0) for v in y_va_arr], dtype=int)
        y_tr = y_tr_enc
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            cache_file.write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
            return best_params

        def _auc(model: Any) -> float:
            prob = model.predict_proba(X_va)
            if prob.shape[1] == 2:
                return float(roc_auc_score(y_va, prob[:, 1]))
            return float(roc_auc_score(y_va, prob, multi_class="ovr", average="weighted"))

        try:
            import optuna
            from xgboost import XGBClassifier

            def objective(trial: Any) -> float:
                params = {
                    "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8]),
                    "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 0.2]),
                    "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                    "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
                    "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
                    "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 7]),
                    "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.1, 1.0]),
                    "reg_lambda": trial.suggest_categorical("reg_lambda", [0, 0.1, 1.0]),
                }
                clf = XGBClassifier(**params, random_state=42, eval_metric="mlogloss", verbosity=0)
                clf.fit(X_tr, y_tr)
                return _auc(clf)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
            best_params = {**best_params, **study.best_params}
        except Exception:
            try:
                from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
                from xgboost import XGBClassifier
                scoring = "roc_auc" if len(np.unique(y_tr)) == 2 else "roc_auc_ovr_weighted"
                grid = {
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 300],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "min_child_weight": [1, 5],
                    "reg_alpha": [0, 0.1],
                    "reg_lambda": [0.1, 1.0],
                }
                # P1-Q8-fix (H04): 兜底 GridSearchCV 的 TimeSeriesSplit 也加 20-bar embargo，
                # 剪除训练折末尾标签可能触碰验证窗价格的样本（三重障碍 max_hold=20）。
                _ts = TimeSeriesSplit(n_splits=3)
                cv = [(tr[:-20] if len(tr) > 20 else tr, te) for tr, te in _ts.split(X_tr) if len(tr) > 20]
                search = GridSearchCV(XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0), grid, scoring=scoring, cv=cv, n_jobs=-1)
                search.fit(X_tr, y_tr)
                best_params = {**best_params, **search.best_params_}
            except Exception as e:
                logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)
        cache_file.write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        try:
            cache_file.write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)
    return best_params


def _pad_proba_3class(proba: Any, model: Any = None) -> np.ndarray:
    """把 predict_proba 输出对齐到规范三分类顺序 [跌, 盘, 涨] 的 3 列矩阵。

    Q8-fix: 若某折/某模型只含 2 个类别，predict_proba 列数会少于 3 且列序
    依赖 sklearn 对 classes_ 的排序；用 model.classes_ 显式对齐，缺列补 0，
    避免评估指标与概率列错位（如 P(上涨) 被当成 P(盘整)）。
    兼容两种标签编码：{0,1,2}（v3 编码后）与 {-1,0,1}（集成原始三重障碍标签）→
    统一映射为 [跌, 盘, 涨] = [0,1,2] 列序。
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    out = np.zeros((proba.shape[0], 3), dtype=float)
    # 规范列序：col0=跌, col1=盘, col2=涨
    canonical = {-1: 0, 0: 1, 1: 2}  # 原始三重障碍 {-1,0,1} → 规范
    if model is not None and hasattr(model, "classes_"):
        classes = np.asarray(model.classes_)
        cls_max = int(np.max(classes))
        cls_min = int(np.min(classes))
        for i, cls in enumerate(classes):
            if i >= proba.shape[1]:
                break
            if int(cls) in (0, 1, 2) and cls_max <= 2 and cls_min >= 0:
                # 已编码 {0,1,2}：列序即类别
                out[:, int(cls)] = proba[:, i]
            elif int(cls) in canonical:
                # 原始 {-1,0,1}：显式映射
                out[:, canonical[int(cls)]] = proba[:, i]
            else:
                # 未知类别：按出现顺序填入剩余空位
                empty = [c for c in range(3) if not np.any(out[:, c])]
                if empty:
                    out[:, empty[0]] = proba[:, i]
    else:
        n = min(proba.shape[1], 3)
        out[:, :n] = proba[:, :n]
    return out


def _canonical_class(raw_label: int, model: Any = None) -> int:
    """把模型原始预测类标签映射到规范 {0:跌, 1:盘, 2:涨}。

    Q8-fix: 集成模型直接训练在原始三重障碍标签 {-1,0,1} 上，predict 返回 -1/0/1；
    v3 路径经 _encode_labels 后为 {0,1,2}。统一经 classes_ 映射，避免
    "prediction_label" 查表错位（如 -1 查不到、1 被当成盘整）。
    """
    raw_label = int(raw_label)
    canonical = {-1: 0, 0: 1, 1: 2}
    if raw_label in canonical:
        return canonical[raw_label]
    if model is not None and hasattr(model, "classes_"):
        classes = np.asarray(model.classes_)
        for i, cls in enumerate(classes):
            if int(cls) == raw_label:
                return i if i < 3 else 1
    return raw_label if 0 <= raw_label <= 2 else 1


def calibrate_proba(y_true: Any, y_prob: Any, method: str = "sigmoid", positive_class: int = 2) -> dict[str, Any]:
    """Calibrate predicted probabilities and report Brier score before/after.

    ``method='sigmoid'`` performs Platt-style calibration; ``method='isotonic'``
    uses isotonic regression. A lightweight probability-feature estimator is
    wrapped by ``CalibratedClassifierCV`` to satisfy sklearn's calibration path.

    Q8-fix: 新增 ``positive_class`` 参数。原实现 ``y_bin = (y_arr == 1)`` 把
    编码类 1（盘整 FLAT）当正类，而 ``p_pos = p_arr[:, -1]`` 是 P(编码类 2=上涨)，
    用 P(上涨) 校准"是否盘整"导致 Brier 与校准曲线无意义。
    默认 ``positive_class=2``（上涨），调用方按目标类显式指定。
    # P1-Q8-fix (H03): 正类按目标类（默认"上涨"=2）显式构造 y_bin，与 p_pos 语义一致。
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss

        y_arr = np.asarray(y_true).astype(int)
        p_arr = np.asarray(y_prob, dtype=float)
        if p_arr.ndim > 1:
            p_pos = p_arr[:, -1]
        else:
            p_pos = p_arr
        X_prob = p_pos.reshape(-1, 1)
        y_bin = (y_arr == int(positive_class)).astype(int)
        base = LogisticRegression(max_iter=1000)
        calibrator = CalibratedClassifierCV(base, method=method if method in {"sigmoid", "isotonic"} else "sigmoid", cv=3)
        calibrator.fit(X_prob, y_bin)
        calibrated = calibrator.predict_proba(X_prob)[:, 1]
        return {
            "probabilities": calibrated,
            "brier_before": round(float(brier_score_loss(y_bin, np.clip(p_pos, 0, 1))), 6),
            "brier_after": round(float(brier_score_loss(y_bin, np.clip(calibrated, 0, 1))), 6),
            "method": method,
            "positive_class": int(positive_class),
        }
    except Exception as e:
        p_arr = np.asarray(y_prob, dtype=float)
        return {"probabilities": p_arr, "brier_before": None, "brier_after": None, "method": method, "error": str(e)}


def detect_feature_drift(X_new: Any, X_ref: Any, threshold: float = 0.05) -> dict[str, Any]:
    """Detect feature drift with KS tests and Population Stability Index.

    A feature triggers an alert when KS p-value is below ``threshold`` or PSI is
    above 0.1. Reference distribution summaries are persisted to
    ``~/.quant_system/feature_ref_dist.json``.
    """
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        try:
            expected = expected[np.isfinite(expected)]
            actual = actual[np.isfinite(actual)]
            if len(expected) == 0 or len(actual) == 0:
                return 0.0
            cuts = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
            if len(cuts) < 3:
                cuts = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), bins + 1)
            e_cnt, _ = np.histogram(expected, bins=cuts)
            a_cnt, _ = np.histogram(actual, bins=cuts)
            e_pct = np.clip(e_cnt / max(1, e_cnt.sum()), 1e-6, None)
            a_pct = np.clip(a_cnt / max(1, a_cnt.sum()), 1e-6, None)
            return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
        except Exception:
            return 0.0

    try:
        Xn = np.asarray(X_new, dtype=float)
        Xr = np.asarray(X_ref, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)
        if Xr.ndim == 1:
            Xr = Xr.reshape(1, -1)
        n_feat = min(Xn.shape[1], Xr.shape[1], N_FEATURES)
        try:
            from scipy.stats import ks_2samp
        except Exception:
            ks_2samp = None
        features: list[dict[str, Any]] = []
        for i in range(n_feat):
            ref_col = Xr[:, i]
            new_col = Xn[:, i]
            p_value = 1.0
            if ks_2samp is not None:
                try:
                    p_value = float(ks_2samp(ref_col[np.isfinite(ref_col)], new_col[np.isfinite(new_col)]).pvalue)
                except Exception:
                    p_value = 1.0
            psi = _psi(ref_col, new_col)
            alert = bool(p_value < threshold or psi > 0.1)
            features.append({"feature": FEATURES[i], "ks_pvalue": round(p_value, 6), "psi": round(psi, 6), "alert": alert})
        ref_summary = {
            FEATURES[i]: {
                "mean": float(np.nanmean(Xr[:, i])),
                "std": float(np.nanstd(Xr[:, i])),
                "p05": float(np.nanpercentile(Xr[:, i], 5)),
                "p50": float(np.nanpercentile(Xr[:, i], 50)),
                "p95": float(np.nanpercentile(Xr[:, i], 95)),
            }
            for i in range(n_feat)
        }
        (_quant_cache_dir() / "feature_ref_dist.json").write_text(json.dumps(ref_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        alerts = [f for f in features if f["alert"]]
        return {"drift_detected": bool(alerts), "n_alerts": len(alerts), "features": features}
    except Exception as e:
        return {"drift_detected": False, "n_alerts": 0, "features": [], "error": str(e)}


def _merge_kline_rows(df_hist: Any, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按日期合并历史 DataFrame 与增量 K 线 dict（new_rows 覆盖同日记录）。"""
    merged: dict[str, dict[str, Any]] = {}
    if df_hist is not None:
        for r in df_hist.to_dict("records"):
            d = str(r.get("日期", ""))[:10]
            if d:
                merged[d] = dict(r)
    for r in new_rows:
        d = str(r.get("日期", ""))[:10]
        if d:
            merged[d] = dict(r)
    # 日期格式为 YYYY-MM-DD，字符串排序即时间排序
    return [merged[k] for k in sorted(merged)]


def rolling_feature_update(symbol: str, new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Incrementally compute latest features from new K-line rows and predict.

    The function reuses the persisted ensemble model when available and returns
    ``predict_proba`` output for the latest feature row.

    P2-Q8-fix (M516): 原实现把几根新 K 线直接传给 extract_features，触发
    ``len(rows)<300`` 硬门槛必然返回空（"insufficient rows"），功能形同虚设。
    现改为：以 akshare 全量历史为上下文，合并增量行后统一重算特征，再取最新
    一行预测 —— 特征计算始终在完整序列上进行，增量行只负责"并入最新数据"。
    """
    try:
        import pickle
        import akshare as ak
        pkl_path = _quant_cache_dir() / "ml_ensemble.pkl"
        if not pkl_path.exists():
            return {"symbol": symbol, "error": "model not found"}
        df_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                     start_date="20230101", adjust="qfq")
        if df_hist is None:
            return {"symbol": symbol, "error": "history fetch failed"}
        merged = _merge_kline_rows(df_hist, new_rows)
        if len(merged) < 300:
            return {"symbol": symbol, "error": f"insufficient rows (history {len(df_hist)} + incremental {len(new_rows)})"}
        X, _, _ = extract_features(merged)
        if len(X) == 0:
            return {"symbol": symbol, "error": "insufficient rows after merge"}
        with open(pkl_path, "rb") as f:
            ensemble = pickle.load(f)
        x_latest = X[-1].reshape(1, -1)
        model_results: dict[str, Any] = {}
        for name, model in ensemble.get("models", {}).items():
            if model is None or not hasattr(model, "predict_proba"):
                continue
            try:
                prob = model.predict_proba(x_latest)[0]
                model_results[name] = [round(float(v), 6) for v in prob]
            except Exception as e:
                logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)
                continue
        return {"symbol": symbol, "n_rows": len(new_rows), "features": X[-1].tolist(), "probabilities": model_results}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


class _LabelMapMismatch(Exception):
    """新批次标签与已存 booster 的类映射不一致，无法安全热启动。"""


def _load_label_map(path: Any) -> tuple[dict[int, int] | None, dict[int, int] | None]:
    """读取持久化的 label_to_code / code_to_label 映射；缺失或损坏返回 (None, None)。"""
    try:
        if not path.exists():
            return None, None
        saved = json.loads(path.read_text(encoding="utf-8"))
        l2c = {int(k): int(v) for k, v in saved.get("label_to_code", {}).items()}
        c2l = {int(k): int(v) for k, v in saved.get("code_to_label", {}).items()}
        if not l2c or not c2l:
            return None, None
        return l2c, c2l
    except Exception:
        return None, None


def _encode_with_map(y: Any, l2c: dict[int, int]) -> np.ndarray | None:
    """用已有 label_to_code 编码新批次；若含未训练标签返回 None（无法热启动）。"""
    arr = np.asarray(y).astype(int)
    if l2c is None or not set(np.unique(arr)).issubset(set(l2c)):
        return None
    return np.array([l2c[int(v)] for v in arr], dtype=int)


def online_partial_fit(X: Any, y: Any) -> dict[str, Any]:
    """Attempt incremental XGBoost training, falling back to full retraining.

    XGBoost does not expose sklearn-style ``partial_fit`` for tree boosters, so
    this function continues training from the cached booster when possible. If
    that path is unavailable, it trains a fresh model on the supplied batch.

    P2-Q8-fix (M521): 原实现对每个新批次重新 ``_encode_labels`` —— 新批次缺某类时
    类编码整体平移，与已存 booster 的类映射错位，热启动模型预测标签静默错乱。
    现改为：把 label_to_code/code_to_label 持久化到 online_label_map.json，热启动
    时用训练时的映射编码新批次，并校验新批次类别集合与训练集合一致后才继续训练；
    校验失败则打印原因并回退全量重训（可见降级，不静默）。
    """
    try:
        from xgboost import XGBClassifier
        X_arr = np.asarray(X, dtype=float)
        params = _default_xgb_params()
        best_file = _quant_cache_dir() / "best_params.json"
        if best_file.exists():
            try:
                params.update(json.loads(best_file.read_text(encoding="utf-8")))
            except Exception as e:
                logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)
        model_path = _quant_cache_dir() / "online_xgb.json"
        map_path = _quant_cache_dir() / "online_label_map.json"
        y_arr = np.asarray(y).astype(int)

        def _persist(l2c: dict[int, int], c2l: dict[int, int]) -> None:
            map_path.write_text(json.dumps({
                "label_to_code": {str(k): int(v) for k, v in l2c.items()},
                "code_to_label": {str(k): int(v) for k, v in c2l.items()},
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        if model_path.exists():
            l2c, c2l = _load_label_map(map_path)
            if l2c is None:
                raise _LabelMapMismatch("无持久化类映射，无法安全热启动")
            y_enc = _encode_with_map(y_arr, l2c)
            if y_enc is None:
                raise _LabelMapMismatch(
                    f"新批次含未训练标签 {sorted(set(np.unique(y_arr))) - set(l2c)}")
            known_codes = set(range(len(c2l)))
            if set(np.unique(y_enc)) != known_codes:
                raise _LabelMapMismatch(
                    f"新批次类别集合 {sorted(set(np.unique(y_enc)))} != 训练集合 {sorted(known_codes)}")
            model = XGBClassifier(**params, random_state=42, eval_metric="mlogloss", verbosity=0)
            try:
                warm = XGBClassifier(**params, random_state=42, eval_metric="mlogloss", verbosity=0)
                warm.load_model(str(model_path))
                booster = warm.get_booster()
                model.fit(X_arr, y_enc, xgb_model=booster)
            except Exception as e:
                # 热启动失败 → 用同一映射对批次全新重训（保类一致，仍可见）
                logger.warning(f"[online_partial_fit] 热启动失败({e}) → 用存储映射对批次全新重训")
                model.fit(X_arr, y_enc)
            model.save_model(str(model_path))
            _persist(l2c, c2l)
            return {"status": "updated", "n_samples": int(len(X_arr)), "model_path": str(model_path)}
        else:
            # 首次训练
            y_enc, l2c, c2l = _encode_labels(y_arr)
            model = XGBClassifier(**params, random_state=42, eval_metric="mlogloss", verbosity=0)
            model.fit(X_arr, y_enc)
            model.save_model(str(model_path))
            _persist(l2c, c2l)
            return {"status": "updated", "n_samples": int(len(X_arr)), "model_path": str(model_path)}
    except _LabelMapMismatch as e:
        logger.warning(f"[online_partial_fit] {e} → 回退全量重训（可见降级）")
        try:
            return {"status": "fallback_retrain", "result": train_model(force_shap=False), "error": str(e)}
        except Exception as inner:
            return {"status": "error", "error": str(inner)}
    except Exception as e:
        try:
            return {"status": "fallback_retrain", "result": train_model(force_shap=False), "error": str(e)}
        except Exception as inner:
            return {"status": "error", "error": str(inner)}

# ── 技术指标计算 ─────────────────────────────────────────

def _ema(d, n):
    """Exponential Moving Average."""
    k, r = 2/(n+1), np.zeros(len(d))
    for i in range(len(d)):
        r[i] = d[i] if i==0 else d[i]*k + r[i-1]*(1-k)
    return r

def _sma(d, n):
    """Simple Moving Average."""
    r = np.zeros(len(d))
    for i in range(len(d)):
        r[i] = np.mean(d[max(0,i-n+1):i+1])
    return r

def _max_s(d, n):
    """Rolling max."""
    r = np.zeros(len(d))
    for i in range(len(d)):
        r[i] = np.max(d[max(0,i-n+1):i+1])
    return r

def _min_s(d, n):
    """Rolling min."""
    r = np.zeros(len(d))
    for i in range(len(d)):
        r[i] = np.min(d[max(0,i-n+1):i+1])
    return r

def _std_s(d, n):
    """Rolling std."""
    r = np.zeros(len(d))
    for i in range(len(d)):
        r[i] = np.std(d[max(0,i-n+1):i+1])
    return r


def _safe_div(num, den, default=0.0):
    """Vectorized division with a true nonzero mask; np.where still evaluates both branches."""
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    shape = np.broadcast_shapes(num_arr.shape, den_arr.shape)
    out = np.full(shape, default, dtype=float)
    np.divide(num_arr, den_arr, out=out, where=np.abs(den_arr) > 1e-10)
    return out


def _rsi(p, n=14):
    """RSI indicator with standard Wilder smoothing seed.

    P2-Q8-fix (L524): 原实现前 n 个点用原始涨跌值（不参与 Wilder 平滑），
    ``i>n`` 才进入平滑，前段跳变。现按 Wilder 标准：以前 n 段涨/跌均值作种子，
    之后逐期递推平滑。
    """
    m = len(p)
    if m < 2:
        return np.zeros(m)
    g = np.zeros(m)
    l = np.zeros(m)
    deltas = np.diff(p)  # deltas[k] = p[k+1]-p[k]
    seed_n = min(n, len(deltas))
    g[seed_n] = float(np.mean(np.maximum(deltas[:seed_n], 0.0)))
    l[seed_n] = float(np.mean(np.maximum(-deltas[:seed_n], 0.0)))
    for i in range(seed_n + 1, m):
        d = p[i] - p[i-1]
        g[i] = (g[i-1]*(n-1) + max(0.0, d)) / n
        l[i] = (l[i-1]*(n-1) + max(0.0, -d)) / n
    rs = _safe_div(g, l, default=1e10)
    return np.clip(100 - 100/(1 + rs), 0, 100)

def _kdj(h, l, c, n=9):
    """KDJ indicator."""
    hn, ln = _max_s(h, n), _min_s(l, n)
    rsv = np.where(hn>ln+1e-10, (c-ln)/(hn-ln)*100, 50)
    k = np.zeros(len(c))
    d = np.zeros(len(c))
    for i in range(len(c)):
        k[i] = 2/3*k[i-1]+1/3*rsv[i] if i>0 else 50
        d[i] = 2/3*d[i-1]+1/3*k[i] if i>0 else 50
    j = 3*k - 2*d
    return k, d, j

def _wr(h, l, c, n=10):
    """Williams %R."""
    hn, ln = _max_s(h, n), _min_s(l, n)
    return np.where(hn>ln+1e-10, (hn-c)/(hn-ln)*-100, -50)

def _cci(h, l, c, n=20):
    """CCI."""
    tp = (h+l+c)/3
    r = np.zeros(len(c))
    for i in range(n, len(c)):
        m, md = np.mean(tp[i-n:i]), np.mean(np.abs(tp[i-n:i]-np.mean(tp[i-n:i])))
        r[i] = (tp[i]-m)/(0.015*md+1e-10)
    return r

def _roc(c, n=10):
    """Rate of Change %."""
    r = np.zeros(len(c))
    for i in range(n, len(c)):
        r[i] = (c[i]/c[i-n]-1)*100 if c[i-n]>1e-10 else 0
    return r

def _bias(c, n):
    """BIAS = (price-MA)/MA*100."""
    ma = _sma(c, n)
    return np.where(ma>1e-10, (c-ma)/ma*100, 0)

def _psy(c, n=12):
    """Psychological Line = up days / window_len * 100.

    P2-Q8-fix (L525): 窗口不足 n 时原实现仍除以 n，序列前段被系统性低估；
    改为除以实际窗口长度。
    """
    r = np.zeros(len(c))
    for i in range(1, len(c)):
        win_start = max(1, i-n+1)
        up = sum(1 for j in range(win_start, i+1) if c[j] >= c[j-1])
        win_len = i - win_start + 1
        r[i] = up / win_len * 100 if win_len > 0 else 0.0
    return r

def _adx(h, l, c, n=14):
    """ADX / +DI / -DI."""
    tr = np.zeros(len(c))
    pdi, mdi = np.zeros(len(c)), np.zeros(len(c))
    for i in range(1, len(c)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up, down = h[i]-h[i-1], l[i-1]-l[i]
        pdi[i] = up if up>0 and up>down else 0
        mdi[i] = down if down>0 and down>up else 0
    atr = _ema(tr, n)
    spdi, smdi = _ema(pdi, n), _ema(mdi, n)
    pdi_s = _safe_div(spdi, atr) * 100
    mdi_s = _safe_div(smdi, atr) * 100
    dx = _safe_div(abs(pdi_s-mdi_s), pdi_s+mdi_s) * 100
    adx_v = _ema(dx, n)
    return adx_v, pdi_s, mdi_s

def _obv(c, v):
    """On-Balance Volume."""
    obv = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i]>c[i-1]: obv[i]=obv[i-1]+v[i]
        elif c[i]<c[i-1]: obv[i]=obv[i-1]-v[i]
        else: obv[i]=obv[i-1]
    return obv

def _mfi(h, l, c, v, n=14):
    """Money Flow Index."""
    tp = (h+l+c)/3
    mf = tp*v
    pmf, nmf = np.zeros(len(c)), np.zeros(len(c))
    for i in range(1, len(c)):
        if tp[i]>tp[i-1]: pmf[i]=mf[i]; nmf[i]=0
        else: nmf[i]=mf[i]; pmf[i]=0
    spmf = _sma(pmf, n)
    snmf = _sma(nmf, n)
    mfr = _safe_div(spmf, snmf, default=1e10)
    return np.clip(100-100/(1+mfr), 0, 100)

def _vr(c, v, n=26):
    """Volume Ratio (VR)."""
    av, bv, cv = np.zeros(len(c)), np.zeros(len(c)), np.zeros(len(c))
    for i in range(1, len(c)):
        for j in range(max(0,i-n+1), i+1):
            if c[j]>c[j-1]: av[i]+=v[j]
            elif c[j]<c[j-1]: cv[i]+=v[j]
            else: bv[i]+=v[j]
    denom = (bv/2+cv+1e-10)
    return np.where(denom>1e-10, (av+bv/2)/denom*100, 100)

# ── 主特征提取 ─────────────────────────────────────────

def extract_features(rows):
    """
    从OHLCV数据提取特征 + 分类标签 + 回归标签。

    分类标签 y_cls: 20 日三重障碍（high/low 触障、障碍宽度按波动率动态设定，
    clip 于 1.5%~8%，见 label_triple_barrier），取值 {1:上涨先触上轨, -1:下跌先触下轨, 0:未触障}。
    回归标签 y_reg: 5 日涨跌幅(%)。

    P2-Q8-fix (M514): 原 docstring 声称"5日涨>1%"、实际用固定 ±2%/20 日三重障碍
    （close-only 价格、不用 high/low 触障、无波动率缩放），分类口径与实际不一致；
    现 docstring 与实现统一。

    返回: (X, y_cls, y_reg) 每个都是numpy数组
    """
    if not rows or len(rows)<300:
        return np.array([]), np.array([]), np.array([])

    c = np.array([r["收盘"] for r in rows], dtype=float)
    h = np.array([r["最高"] for r in rows], dtype=float)
    l = np.array([r["最低"] for r in rows], dtype=float)
    o = np.array([r["开盘"] for r in rows], dtype=float)
    v = np.array([r["成交量"] for r in rows], dtype=float)
    ds = [str(r.get("日期", "")) for r in rows]
    n = len(c)

    # ── 均线 ──
    ma5  = _ema(c, 5)
    ma10 = _ema(c, 10)
    ma20 = _ema(c, 20)
    ma60 = _ema(c, 60)
    ma144 = _ema(c, 144)
    ma300 = _ema(c, 300)

    # ── RSI ──
    rsi6  = _rsi(c, 6)
    rsi12 = _rsi(c, 12)
    rsi24 = _rsi(c, 24)

    # ── KDJ ──
    kdj_k, kdj_d, kdj_j = _kdj(h, l, c)

    # ── Williams %R ──
    wr10 = _wr(h, l, c, 10)
    wr20 = _wr(h, l, c, 20)

    # ── CCI ──
    cci20 = _cci(h, l, c, 20)

    # ── ROC ──
    roc5  = _roc(c, 5)
    roc10 = _roc(c, 10)
    roc20 = _roc(c, 20)

    # ── BIAS ──
    bias5  = _bias(c, 5)
    bias10 = _bias(c, 10)
    bias20 = _bias(c, 20)

    # ── PSY ──
    psy12 = _psy(c, 12)
    psy24 = _psy(c, 24)

    # ── ADX ──
    adx14, pdi14, mdi14 = _adx(h, l, c)

    # ── MACD ──
    dif = _ema(c, 12) - _ema(c, 26)
    dea = _ema(dif, 9)
    macd_hist = (dif - dea) * 2

    # ── 均线交叉 ──
    cross_5_20 = np.where(ma5 > ma20, 1, 0)
    cross_20_60 = np.where(ma20 > ma60, 1, np.where(ma20 < ma60, -1, 0))
    cross_60_144 = np.where(ma60 > ma144, 1, np.where(ma60 < ma144, -1, 0))

    # ── EXPMA ──
    expma12 = _ema(c, 12)
    expma50 = _ema(c, 50)
    expma12_dist = np.where(expma12>1e-10, (c/expma12-1)*100, 0)
    expma50_dist = np.where(expma50>1e-10, (c/expma50-1)*100, 0)

    # ── BOLL ──
    boll_std = _std_s(c, 20)
    boll_mid = _sma(c, 20)
    boll_upper = boll_mid + 2*boll_std
    boll_lower = boll_mid - 2*boll_std
    boll_pos = np.clip(_safe_div(c-boll_lower, boll_upper-boll_lower, default=0.5), 0, 1)
    boll_width = np.where(boll_mid>1e-10, (boll_upper-boll_lower)/boll_mid*100, 0)
    # P2-Q8-fix (L528): 删除 boll_b —— boll_pos.copy() 完全重复，造成特征冗余、重要度分摊失真

    # ── ATR ──
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = _ema(tr, 14)
    atr_pct = np.where(c>1e-10, atr/c*100, 0)
    atr_base = _ema(atr, 20)
    atr_ratio = _safe_div(atr, atr_base, default=1)

    # ── 量能 ──
    v_ma5  = _sma(v, 5)
    v_ma20 = _sma(v, 20)
    v_ma60 = _sma(v, 60)
    vol_ratio_5_20 = np.where(v_ma20>1e-10, v_ma5/v_ma20, 1)
    vol_ratio_5_60 = np.where(v_ma60>1e-10, v_ma5/v_ma60, 1)

    obv = _obv(c, v)
    # P2-Q8-fix (L526): 原实现要求 obv[i-5]>1e-10 才计算斜率，OBV 常态为负
    # → 大量 0 值特征信息损失严重。改为 5 日 OBV 净变化 / 同期成交量（%），
    # 符号保留、无除零、对负 OBV 同样有效。
    obv_slope = np.zeros(n)
    for i in range(5, n):
        win_vol = v[i-4:i+1].sum()
        if win_vol > 1e-10:
            obv_slope[i] = (obv[i] - obv[i-5]) / win_vol * 100

    vr = _vr(c, v)
    mfi = _mfi(h, l, c, v)

    # ── 价态 ──
    range_pct = np.where(c>1e-10, (h-l)/c*100, 0)
    gap_pct = np.zeros(n)
    for i in range(1, n):
        gap_pct[i] = (o[i]/c[i-1]-1)*100 if c[i-1]>1e-10 else 0
    pos_20d = np.zeros(n)
    for i in range(20, n):
        hi, lo = np.max(h[i-19:i+1]), np.min(l[i-19:i+1])
        pos_20d[i] = (c[i]-lo)/(hi-lo)*100 if hi>lo+1e-10 else 50
    change_5d = np.zeros(n)
    for i in range(5, n):
        change_5d[i] = (c[i]/c[i-5]-1)*100 if c[i-5]>1e-10 else 0

    # ── 波动率 ──
    # P2-Q8-fix (L528): 合并 vol_regime / vol_ratio —— 两者都是 std20/std60，
    # 原 vol_ratio 用循环重算同一比值，特征冗余、重要度分摊失真，删除后者。
    vol_short = _std_s(c, 20)
    vol_long  = _std_s(c, 60)
    vol_regime = _safe_div(vol_short, vol_long, default=1)

    # ── 日历 ──
    month_arr = np.zeros(n)
    wday_arr = np.zeros(n)
    for i in range(n):
        try:
            dt = datetime.strptime(ds[i][:10], "%Y-%m-%d")
            month_arr[i] = dt.month
            wday_arr[i] = dt.weekday()
        except (ValueError, IndexError):
            pass
    month_sin = np.sin(2*math.pi*month_arr/12)
    month_cos = np.cos(2*math.pi*month_arr/12)
    is_monday = (wday_arr==0).astype(float)
    is_friday = (wday_arr==4).astype(float)

    # ── 统计特征 ──
    ret_skew = np.zeros(n)
    ret_kurt = np.zeros(n)
    ret_ac1  = np.zeros(n)
    for i in range(20, n):
        rets = [c[j]/c[j-1]-1 for j in range(max(1,i-19), i+1) if c[j-1]>1e-10]
        if len(rets)>5:
            ret_skew[i] = pd.Series(rets).skew()
            ret_kurt[i] = pd.Series(rets).kurtosis()
            ret_ac1[i] = pd.Series(rets).autocorr(lag=1) if len(rets)>10 else 0

    # ── 均线偏离度 (给交互特征用) ──
    pct_ma20 = np.where(ma20>1e-10, (c/ma20-1)*100, 0)
    pct_ma60 = np.where(ma60>1e-10, (c/ma60-1)*100, 0)
    pct_ma144 = np.where(ma144>1e-10, (c/ma144-1)*100, 0)

    # ── 交互特征 ──
    rsi_x_macd = (rsi12-50) * macd_hist
    ma20_x_ma60 = pct_ma20 * pct_ma60
    ma60_x_ma144 = pct_ma60 * pct_ma144
    vol_x_change = vol_ratio_5_20 * change_5d
    rsi_x_cci = (rsi12-50) * (cci20/100)

    # ── 拼装 ──
    LOOKBACK = 300
    X_list, yc_list, yr_list = [], [], []

    # P2-Q8-fix (M514): 触障检测用 high/low（而非 close-only）；障碍宽度按
    # 波动率动态设定（ATR 比例的 1.5 倍，clip 于 1.5%~8%），避免固定 ±2% 在
    # 低波动期过宽、高波动期过窄。
    atr_width = np.clip(1.5 * atr_pct / 100.0, 0.015, 0.08)
    tb_labels = label_triple_barrier(c, pt_sl=(1.02, 0.98), max_hold=20, high=h, low=l, width=atr_width)

    for i in range(LOOKBACK, n-6):
        if c[i] <= 1e-10:
            continue
        vec = [
            rsi6[i], rsi12[i], rsi24[i],
            kdj_k[i], kdj_d[i], kdj_j[i],
            wr10[i], wr20[i],
            cci20[i],
            roc5[i], roc10[i], roc20[i],
            bias5[i], bias10[i], bias20[i],
            psy12[i], psy24[i],
            adx14[i], pdi14[i], mdi14[i],
            dif[i], dea[i], macd_hist[i],
            # P2-Q8-fix (L527): 均线偏离统一加零值保护（原 ma20/ma60/ma144/ma300 无保护，
            # 价格异常为 0 时产生 inf 进入模型）
            (c[i]/ma5[i]-1)*100 if ma5[i]>1e-10 else 0,
            (c[i]/ma10[i]-1)*100 if ma10[i]>1e-10 else 0,
            (c[i]/ma20[i]-1)*100 if ma20[i]>1e-10 else 0,
            (c[i]/ma60[i]-1)*100 if ma60[i]>1e-10 else 0,
            (c[i]/ma144[i]-1)*100 if ma144[i]>1e-10 else 0,
            (c[i]/ma300[i]-1)*100 if ma300[i]>1e-10 else 0,
            cross_5_20[i], cross_20_60[i], cross_60_144[i],
            expma12_dist[i], expma50_dist[i],
            boll_pos[i], boll_width[i],
            atr_pct[i], atr_ratio[i],
            vol_ratio_5_20[i], vol_ratio_5_60[i],
            obv_slope[i], vr[i], mfi[i],
            range_pct[i], gap_pct[i], pos_20d[i], change_5d[i],
            vol_regime[i],
            rsi_x_macd[i], ma20_x_ma60[i], ma60_x_ma144[i],
            vol_x_change[i], rsi_x_cci[i],
            month_sin[i], month_cos[i], is_monday[i], is_friday[i],
            ret_skew[i], ret_kurt[i], ret_ac1[i],
        ]
        X_list.append(vec)
        fwd_ret = (c[i+5]/c[i]-1)*100
        yc_list.append(int(tb_labels[i]))
        yr_list.append(fwd_ret)

    return np.array(X_list), np.array(yc_list), np.array(yr_list)


# ── 训练 ─────────────────────────────────────────────

def train_model(train_symbols=None, force_shap=True):
    """
    训练 XGBoost + RandomForest 双模型，含 SHAP 和相关性分析。

    P2-Q8-fix (M519): 默认股票池从 2 只（600519/601288）改为与 v7 train_ensemble
    一致的 DEFAULT_TRAIN_SYMBOLS（50 只），避免两套模型能力与过拟合风险天差地别；
    force_shap 参数现真实生效：False 时跳过 SHAP 计算。
    """
    if train_symbols is None:
        train_symbols = DEFAULT_TRAIN_SYMBOLS
    t0 = _time.time()

    all_X, all_yc, all_yr = [], [], []
    _KL = Path(__file__).resolve().parent.parent / "data_warehouse" / "kline"
    for sym in train_symbols:
        try:
            # 优先本地kline(快), 其次akshare(慢, 本地/云端均可能失败)
            df = None
            kp = _KL / f"{sym}.parquet"
            if kp.exists():
                _raw = pd.read_parquet(kp, columns=["date", "open", "high", "low", "close", "volume"])
                # 本地kline英文列 → extract_features 需要的中文列(2026-08-22)
                df = pd.DataFrame({"日期": _raw["date"].astype(str).str[:10],
                                   "开盘": _raw["open"], "最高": _raw["high"],
                                   "最低": _raw["low"], "收盘": _raw["close"],
                                   "成交量": _raw["volume"]})
            if df is None:
                import akshare as ak
                try:
                    df = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                            start_date="20230101", adjust="qfq")
                except Exception:
                    df = None
            if df is None or len(df) < 300:
                continue
            X, yc, yr = extract_features(df.to_dict("records"))
            if len(X) > 50:
                all_X.append(X); all_yc.append(yc); all_yr.append(yr)
                logger.info(f"  ✅ {sym}: {len(X)}样本")
        except Exception as e:
            logger.warning(f"  ⚠️ {sym}: {e}")

    if not all_X:
        return {"error": "训练数据不足"}

    # ── Per-stock time-split: each stock split 80/20 individually ──
    # This prevents time-travel leakage across stock boundaries that would
    # occur when concatenating all stocks and doing a single global split.
    X_tr_parts, X_te_parts = [], []
    yc_tr_parts, yc_te_parts = [], []
    yr_tr_parts, yr_te_parts = [], []
    for stock_i, (X_i, yc_i, yr_i) in enumerate(zip(all_X, all_yc, all_yr)):
        # Q8-fix: 逐股票先清洗 NaN/Inf（ffill 只在单股票内进行，避免跨股票边界填充泄漏）；
        # 原实现 for arr in (X_tr, X_te) 只重绑局部变量，属死代码，从未生效。
        # P1-Q8-fix (H01): np.nan_to_num 就地清洗；ffill 限定在单股票内避免跨股票边界填充泄漏。
        # 审计 2026-08-16：剔除每只股票最后 14 行不可靠标签（extract_features 循环到
        # n-6，20 日三重障碍需要完整窗口，尾部 14 行标签不完整被当 0/未触障）。
        if len(X_i) > 14:
            X_i = X_i[:-14]
            yc_i = yc_i[:-14]
            yr_i = yr_i[:-14]
        X_i = np.asarray(X_i, dtype=float)
        if X_i.size > 0:
            X_i = pd.DataFrame(X_i).ffill().fillna(0.0).values.astype(np.float64)
            X_i = np.nan_to_num(X_i, nan=0.0, posinf=1e10, neginf=-1e10)
        all_X[stock_i] = X_i  # 同步更新，保证后续 X_all = vstack(all_X) 也是清洗后的
        s = max(1, int(len(X_i) * 0.8))
        if s >= len(X_i):
            s = len(X_i) - 1
        X_tr_parts.append(X_i[:s])
        X_te_parts.append(X_i[s:])
        yc_tr_parts.append(yc_i[:s])
        yc_te_parts.append(yc_i[s:])
        yr_tr_parts.append(yr_i[:s])
        yr_te_parts.append(yr_i[s:])

    X_tr = np.vstack(X_tr_parts)
    X_te = np.vstack(X_te_parts)
    yc_tr = np.concatenate(yc_tr_parts)
    yc_te = np.concatenate(yc_te_parts)
    yr_tr = np.concatenate(yr_tr_parts)
    yr_te = np.concatenate(yr_te_parts)
    yc = np.concatenate(all_yc)  # kept for full-array metrics below
    yr = np.concatenate(all_yr)
    # V4.1 fix: track stock origin IDs for grouped CV (cross-stock time alignment)
    stock_ids_tr = np.concatenate([np.full(len(p), i) for i, p in enumerate(X_tr_parts)])
    stock_ids_te = np.concatenate([np.full(len(p), i) for i, p in enumerate(X_te_parts)])
    t1 = _time.time()
    logger.info(f"\n[ml-v3] 总计 {len(X_tr)+len(X_te)}样本, {X_tr.shape[1]}特征 训练{len(X_tr)}测试{len(X_te)} ({t1-t0:.0f}s)")

    from sklearn.ensemble import RandomForestClassifier

    # ── Clean NaN/Inf from feature arrays ──
    # Q8-fix: 逐股票清洗已在拼接前完成；此处对拼接矩阵再做一次就地兜底
    # （不再 ffill，避免跨股票边界填充）。
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    y_model_tr, _, code_to_label = _encode_labels(yc_tr)
    y_model_te, _, _ = _encode_labels(yc_te)
    X_tr, X_te, y_tr, y_te = X_tr, X_te, y_model_tr, y_model_te
    # P1-Q8-fix (H07): 传入股票分组 → walk_forward_validate 改用 GroupKFold，
    # 不再在跨股票拼接矩阵上按行号做 TimeSeriesSplit（原实现对面板数据时间对齐声明不成立）。
    wf_results = walk_forward_validate(X_tr, yc_tr, n_train=min(252, len(X_tr) // 2), n_test=min(63, len(X_tr) // 4), groups=stock_ids_tr, n_splits=5)
    # Extract a time-series-aware validation split from training data
    # to avoid test data (X_te) leaking into hyperparameter tuning or calibration
    # P1-Q8-fix (H07): 验证集改为逐股票从训练部分切末 20% 拼接 ——
    # 原全局 80/20 在拼接矩阵上切分使验证集=最后一只股票的尾部，调参验证代表性差。
    X_fit_parts, X_val_parts = [], []
    y_fit_raw_parts, y_val_raw_parts = [], []
    for Xi, yi in zip(X_tr_parts, yc_tr_parts):
        ni = int(len(Xi))
        if ni < 2:
            continue
        vi = max(1, int(ni * 0.2))
        if vi >= ni:
            vi = ni - 1
        X_fit_parts.append(Xi[:ni - vi])
        X_val_parts.append(Xi[ni - vi:])
        y_fit_raw_parts.append(yi[:ni - vi])
        y_val_raw_parts.append(yi[ni - vi:])
    if X_fit_parts and X_val_parts and np.concatenate(y_fit_raw_parts).size > 5:
        X_tr_fit = np.vstack(X_fit_parts)
        X_val = np.vstack(X_val_parts)
        y_tr_fit, _, _ = _encode_labels(np.concatenate(y_fit_raw_parts))
        y_val, _, _ = _encode_labels(np.concatenate(y_val_raw_parts))
    else:
        # 兜底：样本过少时回退全局切分
        val_from_tr = max(1, int(len(X_tr) * 0.8)) if len(X_tr) > 50 else len(X_tr)
        X_tr_fit, y_tr_fit = X_tr[:val_from_tr], y_tr[:val_from_tr]
        X_val, y_val = X_tr[val_from_tr:], y_tr[val_from_tr:]
    tuned_params = tune_hyperparams(X_tr_fit, y_tr_fit, X_val, y_val, n_trials=100)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                 min_samples_leaf=10, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    # Q8-fix: 基模型只在 X_tr_fit 上训练（不含 X_val），使校准器在真正
    # held-out 的 X_val 上 fit/评估，避免样本内乐观（原 rf.fit(X_tr) 把
    # 校准集也喂给了基模型）。
    rf.fit(X_tr_fit, y_tr_fit)

    # XGBoost
    has_xgb = False; xgb = None
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(**tuned_params, random_state=42,
                            eval_metric="mlogloss",
                            verbosity=0)
        xgb.fit(X_tr_fit, y_tr_fit)
        # Clear feature names to avoid SHAP string conversion issues
        if hasattr(xgb, 'get_booster'):
            xgb.get_booster().feature_names = None
        has_xgb = True
    except Exception as e:
        logger.warning(f"  ⚠️ XGBoost不可用: {e}，仅用RandomForest")

    # 集成预测 (on test set for evaluation only)
    if has_xgb and xgb is not None:
        # Q8-fix: 原实现 pred = ((rp+xp)/2 > 0.5).astype(int) 输出二值 {0,1}，
        # 而 y_te 是三分类编码 {0,1,2}（0跌/1盘/2涨），pred 永不可能等于 2，
        # accuracy/precision/recall/f1 全部失真且静默上报。改为三分类 argmax。
        # P1-Q8-fix (H02): pred 用 argmax 对齐 y_te 的 {0,1,2} 标签空间，
        # 概率列经 _pad_proba_3class 按 classes_ 显式对齐，修复静默失真指标。
        rp_full = _pad_proba_3class(rf.predict_proba(X_te), rf)
        xp_full = _pad_proba_3class(xgb.predict_proba(X_te), xgb)
        pred = np.argmax((rp_full + xp_full) / 2.0, axis=1)
        imp = (rf.feature_importances_ + xgb.feature_importances_) / 2
    else:
        pred = rf.predict(X_te)
        imp = rf.feature_importances_[:N_FEATURES]  # clip in case dimension mismatch

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc  = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, pred, average="weighted", zero_division=0)
    # Calibration uses held-out validation split, NOT test data
    calibration_results = None
    try:
        if has_xgb and xgb is not None and len(X_val) > 10:
            # Q8-fix: positive_class=2（上涨）显式指定正类，原实现把编码类 1（盘整）当正类
            # P1-Q8-fix (H03): 校准概率经 _pad_proba_3class 显式对齐到 [跌,盘,涨] 三列，
            # 使 p_arr[:,-1]=P(上涨) 与 positive_class=2 一致（xgb 训练集缺类时 predict_proba
            # 列数 <3，原 p_arr[:,-1] 会取到 P(盘整)）；基模型只在 X_tr_fit 训练，
            # 校准器在 held-out X_val 上 fit/评估，避免样本内乐观。
            calibration_results = calibrate_proba(y_val, _pad_proba_3class(xgb.predict_proba(X_val), xgb), method="sigmoid", positive_class=2)
            if isinstance(calibration_results, dict) and "probabilities" in calibration_results:
                calibration_results["probabilities"] = [round(float(v), 6) for v in np.asarray(calibration_results["probabilities"]).ravel()[:200]]
    except Exception:
        calibration_results = None

    feat_imp = sorted(zip(FEATURES[:len(imp)], imp), key=lambda x: -x[1])

    # ── SHAP分析 ──
    # P2-Q8-fix (M519): force_shap=False 时跳过 SHAP（原实现无论 True/False 都跑）。
    # Use RF for SHAP (XGBoost 3.x has compat issue with shap 0.49)
    # Q8-fix: 原代码 _run_shap(rf, X_te, X.shape[1]) 中的 X 是 extract_features
    # 循环体最后一只股票的矩阵（循环变量残留），改用全样本矩阵维度。
    X_all = np.vstack(all_X)
    shap_results = None
    if force_shap:
        logger.info("\n[ml-v3] 📊 计算SHAP贡献度 (150样本)...")
        shap_t0 = _time.time()
        shap_results = _run_shap(rf, X_te, X_all.shape[1])
        if shap_results:
            dt = _time.time() - shap_t0
            logger.info(f"  ✅ SHAP完成 ({dt:.1f}s)")
            if "error" in shap_results:
                logger.warning(f"  ⚠️ {shap_results['error']}")
    else:
        logger.info("\n[ml-v3] force_shap=False，跳过SHAP计算")

    # ── 相关性分析 ──
    # Q8-fix: 原代码 _correlation_analysis(X, yc, yr) 的 X 是单股票矩阵（循环变量），
    # 而 yc/yr 是全市场拼接 → 长度不匹配必然 ValueError 崩溃；改用全样本 X_all。
    corr_results = _correlation_analysis(X_all, yc, yr)

    # ── 保存模型 ──
    cache_dir = _quant_cache_dir()
    # Invalidate stale ensemble cache when retraining v3 model
    old_ensemble = cache_dir / "ml_ensemble.pkl"
    if old_ensemble.exists():
        old_ensemble.unlink()
    old_ensemble_json = cache_dir / "ml_ensemble.json"
    if old_ensemble_json.exists():
        old_ensemble_json.unlink()
    cache = cache_dir / "ml_model.json"
    cache.write_text(json.dumps({
        "version": "v3",
        "training_date": datetime.now(CST).strftime("%Y-%m-%d %H:%M"),
        "samples": int(len(X_all)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "has_xgboost": has_xgb,
        "label_map": {str(k): int(v) for k, v in code_to_label.items()},
        "best_params": tuned_params,
        "walk_forward": wf_results,
        "calibration": calibration_results,
        "n_features": int(N_FEATURES),
        "feature_importance": [(n, round(v, 4)) for n, v in feat_imp],
        # P2-Q8-fix (M517): 移除从未被读取的 feature_stats —— 注释声称"用于预测归一化"，
        # 但 predict_stock/ensemble 预测均不读取（树模型 + LR 自带 scaler，无需 z-score），
        # 保留只会误导。
        "shap": shap_results,
        "correlation": corr_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "samples": int(len(X_all)), "has_xgboost": has_xgb,
        "best_params": tuned_params,
        "walk_forward": wf_results,
        "calibration": calibration_results,
        "feature_importance": feat_imp,
        "shap": shap_results,
        "correlation": corr_results,
    }


def _run_shap(model, X_te, n_features, max_samples=150):
    """Run SHAP TreeExplainer for feature contribution analysis.
    
    Uses signal-based timeout protection.
    """
    try:
        import signal

        class TimeoutError_(Exception):
            pass

        def _handler(signum, frame):
            raise TimeoutError_("SHAP超时")

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(60)  # 60s timeout

        try:
            import shap
            X_clean = np.asarray(X_te, dtype=np.float64)
            X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e10, neginf=-1e10)
            if len(X_clean) > max_samples:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_clean), max_samples, replace=False)
                X_sample = X_clean[idx]
            else:
                X_sample = X_clean

            explainer = shap.TreeExplainer(model, X_sample)
            shap_vals = explainer.shap_values(X_sample)

            # Handle multi-class output
            # Q8-fix: 原实现取 shap_vals[1]（编码类 1 = 盘整 FLAT），
            # "特征贡献度 Top" 实际描述的是盘整类；改为索引 2（编码类 2 = 上涨），
            # 与 predict_proba[:,2] 取上涨概率的语义一致。
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[2] if len(shap_vals) > 2 else shap_vals[-1]
            elif hasattr(shap_vals, 'shape') and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 2] if shap_vals.shape[2] > 2 else shap_vals[:, :, -1]

            shap_vals = np.asarray(shap_vals, dtype=np.float64)
            mean_shap = np.mean(np.abs(shap_vals), axis=0)
            n = min(n_features, len(mean_shap))
            feat_shap = sorted(zip(FEATURES[:n], mean_shap[:n]),
                              key=lambda x: -x[1])
            return {
                "top20": [(n, round(v, 6)) for n, v in feat_shap[:20]],
                "all": [(n, round(v, 6)) for n, v in feat_shap],
            }
        finally:
            signal.alarm(0)
    except Exception as e:
        return {"error": f"SHAP: {e}"}


def _correlation_analysis(X, yc, yr, max_feat=30):
    """Pearson correlation between features and target."""
    result = {"cls": [], "reg": []}
    n_features = min(N_FEATURES, X.shape[1])
    for j in range(n_features):
        feat = X[:, j]
        # Avoid all-NaN
        mask = ~(np.isnan(feat) | np.isinf(feat))
        if mask.sum() < 10:
            continue
        f = feat[mask]
        yc_f = yc[mask]
        yr_f = yr[mask]
        corr_cls = np.corrcoef(f, yc_f)[0, 1] if len(set(yc_f))>1 else 0
        corr_reg = np.corrcoef(f, yr_f)[0, 1] if np.std(yr_f)>1e-10 else 0
        if not np.isnan(corr_cls):
            result["cls"].append((FEATURES[j], round(corr_cls, 3)))
        if not np.isnan(corr_reg):
            result["reg"].append((FEATURES[j], round(corr_reg, 3)))
    result["cls"].sort(key=lambda x: -abs(x[1]))
    result["reg"].sort(key=lambda x: -abs(x[1]))
    return result


# ── 预测 ─────────────────────────────────────────────

def predict_stock(symbol):
    """个股预测：优先走训练好的集成模型（ensemble predict_proba 加权）。

    Q8-fix: 原实现 ml_score 完全由手工启发式规则累加（rsi<30、kdj_j<0、cci<-100…
    拍脑袋阈值），训练好的 RF/XGB 模型与校准概率完全未参与；
    现改为：模型路径（ml_ensemble.pkl 加权概率）计算分数，
    启发式规则仅保留为可解释性展示（signals 列表）。
    无集成模型时回退旧启发式并标注 score_source="heuristic"。
    """
    cache_file = Path.home() / ".quant_system" / "ml_model.json"
    ensemble_pkl = Path.home() / ".quant_system" / "ml_ensemble.pkl"
    if not cache_file.exists() and not ensemble_pkl.exists():
        return {"error": "请先运行 --train 或 --ensemble-train"}

    model_data = {}
    if cache_file.exists():
        try:
            model_data = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            model_data = {}
    imp_map = {n: v for n, v in model_data.get("feature_importance", [])}

    # 2026-08-22: 本地kline优先(免akshare网络失败), 与训练一致
    df = None
    _klp = Path(__file__).resolve().parent.parent / "data_warehouse" / "kline" / f"{symbol}.parquet"
    if _klp.exists():
        _raw = pd.read_parquet(_klp, columns=["date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame({"日期": _raw["date"].astype(str).str[:10], "开盘": _raw["open"],
                           "最高": _raw["high"], "最低": _raw["low"], "收盘": _raw["close"],
                           "成交量": _raw["volume"]})
    if df is None:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date="20230101", adjust="qfq")
    if df is None or len(df) < 300:
        return {"error": f"数据不足({len(df) if df is not None else 0})"}

    X, _, _ = extract_features(df.to_dict("records"))
    if len(X) == 0:
        return {"error": "特征提取失败"}

    latest = df.iloc[-1]
    x_latest = X[-1]
    nf = min(N_FEATURES, X.shape[1])

    # ── 模型路径（主）：ensemble 加权 predict_proba ──
    score_source = "heuristic"
    up_prob = None
    if ensemble_pkl.exists():
        try:
            with open(ensemble_pkl, "rb") as f:
                ensemble = pickle.load(f)
            emodels, eweights = ensemble["models"], ensemble["weights"]
            x_row = x_latest.reshape(1, -1)
            wp = np.zeros(3)
            tw = 0.0
            for mn in ["xgb", "rf", "lr"]:
                m = emodels.get(mn)
                w = eweights.get(mn, 0.0)
                if m is None or w == 0.0:
                    continue
                try:
                    proba_raw = m.predict_proba(x_row)[0]
                    pa = _pad_proba_3class(proba_raw, m)[0]
                    wp[:3] += pa * w
                    tw += w
                except Exception as e:
                    logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)
                    continue
            if tw > 0:
                wp /= tw
                up_prob = float(np.clip(wp[2], 0.0, 1.0))
                score_source = "ensemble"
        except Exception:
            up_prob = None

    # ── 启发式信号（仅解释展示；分数主路径为模型） ──
    signals = []
    weight_map = {}
    heuristic_score = 0.0

    for j in range(nf):
        name = FEATURES[j]
        w = imp_map.get(name, 0.01)
        weight_map[name] = w
        val = x_latest[j]
        contrib = 0.0

        # RSI超卖
        if "rsi" in name:
            n_period = int(''.join(filter(str.isdigit, name)) or 14)
            if val < 30:
                contrib = w * (30-val)/30 * 0.5
                signals.append(f"{name}={val:.1f}(超卖)")
            elif val > 70:
                contrib = -w * (val-70)/30 * 0.5
                signals.append(f"{name}={val:.1f}(超买)")
        # KDJ超卖
        elif name == "kdj_j" and val < 0:
            contrib = w * (-val)/100 * 0.5
            signals.append(f"KDJ-J={val:.1f}(超卖)")
        elif name == "kdj_j" and val > 100:
            contrib = -w * (val-100)/100 * 0.5
            signals.append(f"KDJ-J={val:.1f}(超买)")
        # CCI超卖
        elif name == "cci20" and val < -100:
            contrib = w * (-100-val)/200 * 0.5
            signals.append(f"CCI={val:.0f}(超卖)")
        elif name == "cci20" and val > 100:
            contrib = -w * (val-100)/200 * 0.5
            signals.append(f"CCI={val:.0f}(超买)")
        # Williams %R
        elif name == "wr10" and val < -80:
            contrib = w * (-80-val)/20 * 0.3
            signals.append(f"WR={val:.0f}(超卖)")
        # 均线附近
        elif "pct_ma" in name and abs(val) < 2:
            contrib = w * (1-abs(val)/2) * 0.5
            m = name.replace("pct_", "")
            signals.append(f"{m}={val:+.1f}%(附近)")
        # MACD金叉
        elif name == "macd_hist" and val > 0 and abs(val) < macd_threshold(x_latest, X):
            contrib = w * 0.3
        # BOLL下轨
        elif name == "boll_pos" and val < 0.1:
            contrib = w * (0.1-val)/0.1 * 0.5
            signals.append(f"BOLL下轨({val:.2f})")
        # MFI
        elif name == "mfi" and val < 20:
            contrib = w * (20-val)/20 * 0.3
            signals.append(f"MFI={val:.0f}(超卖)")
        # VR
        elif name == "vr" and val < 70:
            contrib = w * (70-val)/70 * 0.3
            signals.append(f"VR={val:.0f}(缩量)")
        # BIAS
        elif "bias" in name and val < -5:
            contrib = w * (-5-val)/10 * 0.3
        # ADX趋势
        elif name == "adx14" and val > 25:
            contrib = w * (val-25)/25 * 0.2
            signals.append(f"ADX={val:.1f}(趋势)")
        # 交互信号
        elif name == "rsi_x_macd" and val > 0:
            contrib = w * 0.2
        elif name == "rsi_x_cci" and val > 0:
            contrib = w * 0.2
        # OBV
        elif name == "obv_slope" and val > 5:
            contrib = w * val/20 * 0.2
        # Volume
        elif name == "vol_ratio_5_20" and val < 0.7:
            contrib = w * (0.7-val)/0.7 * 0.2
            signals.append(f"缩量({val:.2f}x)")

        heuristic_score += contrib

    # 分数：模型路径优先，启发式仅兜底
    if up_prob is not None:
        pct = round(up_prob * 100, 1)
    else:
        max_score = sum(weight_map.values()) * 1.5
        pct = min(100, max(0, heuristic_score/max_score*100)) if max_score>0 else 0

    # Top5 contributing features
    contribs = []
    for j in range(nf):
        name = FEATURES[j]
        w = weight_map.get(name, 0.01)
        val = x_latest[j]
        contribs.append((name, abs(w * val), w * val))

    contribs.sort(key=lambda x: -abs(x[1]))
    top_contrib = contribs[:5]

    return {
        "symbol": symbol,
        "price": float(latest["收盘"]),
        "date": str(latest.get("日期", "")),
        "ml_score": round(pct, 1),
        "score_source": score_source,
        "up_probability": round(up_prob, 4) if up_prob is not None else None,
        "recommendation": _get_rec(pct),
        "signals": signals[:8],
        "top_contributors": [(n, round(v, 4)) for n, _, v in top_contrib],
        "n_features": nf,
    }


def macd_threshold(val, X):
    """Estimate MACD histogram magnitude."""
    if X is None:
        return 5
    mh_idx = FEATURES.index("macd_hist") if "macd_hist" in FEATURES else -1
    if mh_idx >= 0 and hasattr(X, "shape") and len(X.shape) == 2 and mh_idx < X.shape[1]:
        return np.percentile(np.abs(X[:, mh_idx]), 75)
    return 5


def _get_rec(score):
    if score >= 50: return "🟢 积极关注"
    if score >= 30: return "🟡 适当关注"
    if score >= 15: return "⚪ 观望"
    return "🔴 回避"


# ── 报告格式化 ─────────────────────────────────────────

def format_importance(result):
    lines = ["═"*60,
             "🧠 **ML v3 — 全指标 + SHAP + 相关性分析**",
             "═"*60]
    model_type = "XGBoost+RF集成" if result.get("has_xgboost") else "RandomForest"
    lines.append(f"  模型: {model_type}")
    lines.append(f"  样本: {result['samples']} | "
                f"准确率: {result['accuracy']*100:.1f}% | "
                f"精准率: {result['precision']*100:.1f}%")
    lines.append(f"  F1: {result.get('f1', 0)*100:.1f}% | "
                f"召回率: {result.get('recall', 0)*100:.1f}%")
    lines.append(f"  特征数: {result.get('n_features', N_FEATURES)}")
    lines.append("")

    # ── 特征重要性 Top 15 ──
    lines.append("🏆 **Feature Importance (Top 15):**")
    lines.append(f"  {'':<3} {'特征':<26} {'权重':>7} {'贡献度分布':<25}")
    lines.append(f"  {'':-<3} {'-':-<26} {'-':->7} {'-':-<25}")
    for i, (n, v) in enumerate(result["feature_importance"][:15]):
        bar = "█" * max(1, int(v*300))
        lines.append(f"  {i+1:<3} {n:<26} {v*100:>6.2f}% {bar}")
    lines.append("")

    # ── SHAP Top 15 ──
    shap_data = result.get("shap")
    if shap_data and "top20" in shap_data:
        lines.append("📊 **SHAP 贡献度 (Top 15):**")
        for i, (n, v) in enumerate(shap_data["top20"][:15]):
            bar = "▓" * max(1, int(v*300))
            lines.append(f"  {i+1:<3} {n:<26} |SHAP|={v:.4f} {bar}")
        lines.append("")

    # ── 相关性 Top 10 (分类) ──
    corr = result.get("correlation", {})
    if corr.get("cls"):
        lines.append("🔗 **与涨跌方向相关性 Top 10 (Pearson):**")
        pos = [(n, v) for n, v in corr["cls"] if v > 0][:5]
        neg = [(n, v) for n, v in corr["cls"] if v < 0][:5]
        if pos:
            lines.append(f"  📈 正相关: " + ", ".join(f"{n}(+{v:.3f})" for n, v in pos))
        if neg:
            lines.append(f"  📉 负相关: " + ", ".join(f"{n}({v:.3f})" for n, v in neg))
        lines.append("")

    if corr.get("reg"):
        lines.append("🔗 **与涨跌幅%相关性 Top 5:**")
        pos_r = [(n,v) for n,v in corr["reg"] if v > 0][:3]
        neg_r = [(n,v) for n,v in corr["reg"] if v < 0][:3]
        parts = []
        if pos_r: parts.append("📈 " + ", ".join(f"{n}(+{v:.3f})" for n,v in pos_r))
        if neg_r: parts.append("📉 " + ", ".join(f"{n}({v:.3f})" for n,v in neg_r))
        if parts: lines.append("  " + " | ".join(parts))
        lines.append("")

    # ── 特征分类总结 ──
    lines.append("📋 **特征类别贡献汇总:**")
    categories = {
        "RSI/KDJ(超买超卖)": ["rsi6","rsi12","rsi24","kdj_k","kdj_d","kdj_j","wr10","wr20"],
        "CCI/WR(极端信号)": ["cci20"],
        "MACD/ADX(趋势动量)": ["macd_dif","macd_dea","macd_hist","adx14","pdi14","mdi14"],
        "均线(趋势位置)": ["pct_ma5","pct_ma10","pct_ma20","pct_ma60","pct_ma144","pct_ma300"],
        "均线交叉/EXPMA": ["cross_5_20","cross_20_60","cross_60_144","expma12_dist","expma50_dist"],
        # P2-Q8-fix (L528): 分类清单同步删除 boll_b / vol_ratio
        "BOLL(波动通道)": ["boll_pos","boll_width"],
        "ATR/波动率": ["atr_pct","atr_ratio","vol_regime"],
        "量能(VR/MFI/OBV)": ["vol_ratio_5_20","vol_ratio_5_60","obv_slope","vr","mfi"],
        "BIAS/ROC/PSY": ["bias5","bias10","bias20","roc5","roc10","roc20","psy12","psy24"],
        "价态/交互": ["range_pct","gap_pct","pos_20d","rsi_x_macd","rsi_x_cci"],
        "日历/统计": ["month_sin","month_cos","is_monday","is_friday","ret_skew","ret_kurt","ret_ac1"],
    }
    cat_scores = {}
    for cat, feats in categories.items():
        s = 0
        for n, v in result["feature_importance"]:
            if n in feats:
                s += v
        if s > 0:
            cat_scores[cat] = s
    cat_sorted = sorted(cat_scores.items(), key=lambda x: -x[1])
    for cat, s in cat_sorted:
        bar = "█" * max(1, int(s*400))
        lines.append(f"  {cat:<24} {s*100:>5.2f}% {bar}")

    lines.append("")
    lines.append("═"*60)
    return "\n".join(lines)


def format_prediction(p):
    if "error" in p:
        return f"⚠️ {p['error']}"
    lines = [
        "═"*56,
        f"🧠 **ML v3 — {p['symbol']} @¥{p['price']:.2f}**",
        f"  日期: {p['date']}  |  评分: {p['ml_score']}/100",
        f"  建议: {p['recommendation']}",
        "═"*56,
    ]
    if p.get("signals"):
        lines.append(f"\n📡 触发信号 ({len(p['signals'])}条):")
        for s in p["signals"]:
            lines.append(f"  • {s}")
    if p.get("top_contributors"):
        lines.append(f"\n🏆 Top5 贡献因子:")
        for n, v in p["top_contributors"]:
            arrow = "🟢" if v > 0 else "🔴"
            lines.append(f"  {arrow} {n:<26} {v:+.4f}")
    lines.append(f"\n📊 特征维度: {p.get('n_features', '?')}")
    lines.append("═"*56)
    return "\n".join(lines)


# ── V7 additions: label engineering, ensemble, auto-retrain ──────────
import pickle


def _create_labels(df, forward_days=None):
    """Generate multi-horizon 3-class + regression labels from DataFrame.

    P2-Q8-fix (M514): 原实现用 close 前向收益 ±2% 三分类，与主训练用的
    label_triple_barrier（high/low 触障 + 波动率缩放宽度）口径不一致；
    现统一委托 label_triple_barrier，分类标签映射为 {0:跌, 1:盘, 2:涨}。
    """
    if forward_days is None:
        forward_days = [1, 3, 5, 10, 20]
    close = df["收盘"].values.astype(np.float64)
    high = df["最高"].values.astype(np.float64) if "最高" in df.columns else close
    low = df["最低"].values.astype(np.float64) if "最低" in df.columns else close
    n = len(close)
    # 波动率缩放宽度（与 extract_features 口径一致）：1.5×ATR%，clip 于 1.5%~8%
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = _ema(tr, 14)
    atr_pct = np.where(close > 1e-10, atr / close * 100, 0)
    width = np.clip(1.5 * atr_pct / 100.0, 0.015, 0.08)
    labels = {}
    for fd in forward_days:
        cls_raw = label_triple_barrier(close, pt_sl=(1.02, 0.98), max_hold=fd,
                                       high=high, low=low, width=width)
        cls_arr = np.full(n, np.nan, dtype=np.float64)
        reg_arr = np.full(n, np.nan, dtype=np.float64)
        for i in range(n - fd):
            reg_arr[i] = (close[i + fd] / close[i] - 1.0) * 100.0
            cls_arr[i] = float(int(cls_raw[i]) + 1)  # {-1,0,1} → {0,1,2}
        labels[f"cls_{fd}d"] = cls_arr
        labels[f"reg_{fd}d"] = reg_arr
    return labels


def _timeseries_cv(X, y, n_splits=5, embargo=20):
    """Walk-forward expanding-window CV with embargo.

    ``embargo`` bars between train end and test start prevent label leakage
    when forward-return labels look ahead (e.g. 5-day return label at t
    uses t+5 prices which may overlap with the test window).

    Q8-fix: 默认 embargo 从 5 提到 20，覆盖三重障碍标签最大前视（max_hold=20），
    否则启用该函数即引入泄漏。
    # P1-Q8-fix (H04): 默认 embargo=20 >= 标签水平线 max_hold=20，杜绝训练样本
    # 标签使用测试窗价格造成的 CV 指标虚高。


    Yields (train_idx, test_idx) — expanding train, non-overlapping test blocks.
    """
    n = X.shape[0]
    block = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = i * block
        test_start = train_end + embargo  # embargo gap prevents label overlap
        if test_start >= n:
            break
        test_end = min(test_start + block, n)
        test_len = test_end - test_start
        if test_len < 1:
            break
        yield np.arange(train_end), np.arange(test_start, test_end)


def _record_top_features(models_dict, feature_names, out_path):
    """Record top-20 feature importance per model, with trend history."""
    top20 = {}
    full = {}
    for name, model in models_dict.items():
        if model is None:
            continue
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            imp = np.zeros(len(feature_names))
        paired = sorted(zip(feature_names[:len(imp)], imp), key=lambda x: -x[1])
        top20[name] = [{"feature": n, "importance": round(float(v), 4)} for n, v in paired[:20]]
        full[name] = [(n, round(float(v), 4)) for n, v in paired]
    record = {"timestamp": datetime.now(CST).isoformat(), "top20_per_model": top20}
    history = []
    if out_path.exists():
        try:
            history = json.loads(out_path.read_text("utf-8")).get("importance_trend", [])
        except Exception:
            history = []
    history.append({"timestamp": datetime.now(CST).isoformat(), "full_importance": full})
    record["importance_trend"] = history[-20:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    return record


def train_ensemble(train_symbols=None):
    """
    Train 3-model ensemble (XGB, RF, LR) with time series CV.
    Saves ensemble to ~/.quant_system/ml_ensemble.pkl + metadata JSON.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if train_symbols is None:
        # P2-Q8-fix (M519/M520): 与 train_model / auto_retrain 共用单一股票池常量
        train_symbols = DEFAULT_TRAIN_SYMBOLS

    cache_dir = _quant_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path(__file__).resolve().parent.parent / "tmp_tx" / "quant_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    logger.info(f"\n{'='*56}\n  🔄 V7 Ensemble Training — {len(train_symbols)} symbols\n{'='*56}")

    all_X, all_yc, all_yr = [], [], []
    _KL2 = Path(__file__).resolve().parent.parent / "data_warehouse" / "kline"
    for sym in train_symbols:
        try:
            df = None
            _kp2 = _KL2 / f"{sym}.parquet"
            if _kp2.exists():
                _rw = __import__("pandas").read_parquet(_kp2, columns=["date", "open", "high", "low", "close", "volume"])
                df = pd.DataFrame({"日期": _rw["date"].astype(str).str[:10], "开盘": _rw["open"],
                                   "最高": _rw["high"], "最低": _rw["low"], "收盘": _rw["close"],
                                   "成交量": _rw["volume"]})
            if df is None:
                import akshare as ak
                df = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date="20230101", adjust="qfq")
            if df is None or len(df) < 300:
                continue
            X, yc, yr = extract_features(df.to_dict("records"))
            if len(X) > 50:
                all_X.append(X)
                all_yc.append(yc)
                all_yr.append(yr)
                logger.info(f"  ✅ {sym}: {len(X)} samples")
        except Exception as e:
            logger.warning(f"  ⚠️ {sym}: {e}")
    if not all_X:
        return {"status": "error", "error": "训练数据不足"}

    # ── Per-stock 80/20 time-split to prevent cross-stock time travel ──
    X_tr_parts, X_te_parts = [], []
    y_tr_parts, y_te_parts = [], []
    for X_i, yc_i, yr_i in zip(all_X, all_yc, all_yr):
        # Q8-fix: 逐股票清洗 NaN/Inf（单股票内 ffill），避免拼接矩阵跨股票填充泄漏
        # 审计 2026-08-16：剔除每只股票最后 14 行不可靠标签（extract_features 循环到
        # n-6，20 日三重障碍需要完整窗口，尾部 14 行标签不完整被当 0/未触障）。
        if len(X_i) > 14:
            X_i = X_i[:-14]
            yc_i = yc_i[:-14]
            yr_i = yr_i[:-14]
        X_i = np.asarray(X_i, dtype=float)
        if X_i.size > 0:
            X_i = pd.DataFrame(X_i).ffill().fillna(0.0).values.astype(np.float64)
            X_i = np.nan_to_num(X_i, nan=0.0, posinf=1e10, neginf=-1e10)
        s = max(1, int(len(X_i) * 0.8))
        if s >= len(X_i):
            s = len(X_i) - 1
        X_tr_parts.append(X_i[:s])
        X_te_parts.append(X_i[s:])
        y_tr_parts.append(yc_i[:s].astype(int))
        y_te_parts.append(yc_i[s:].astype(int))

    X_tr = np.vstack(X_tr_parts)
    X_te = np.vstack(X_te_parts)
    y_tr = np.concatenate(y_tr_parts)
    y_te = np.concatenate(y_te_parts)
    # Q8-fix: 拼接后再兜底一次（不再 ffill，防跨股票填充）
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    # V4.1 fix: track stock group IDs for grouped CV (cross-stock time alignment)
    stock_groups_tr = np.concatenate([np.full(len(part), i) for i, part in enumerate(X_tr_parts)])
    logger.info(f"\n[ensemble] Total {len(X_tr)+len(X_te)} samples ({len(X_tr)} train, {len(X_te)} test), {X_tr.shape[1]} features ({_time.time()-t0:.0f}s)")

    # V4.1 fix: use GroupKFold with stock IDs to prevent cross-stock time travel leakage
    from sklearn.model_selection import GroupKFold as _GroupKFold
    cv_indices = list(_GroupKFold(n_splits=5).split(X_tr, y_tr, groups=stock_groups_tr))
    models = {"xgb": None, "rf": None, "lr": None}
    cv_scores = {"xgb": [], "rf": [], "lr": []}
    feature_names = FEATURES[:X_tr.shape[1]]

    # RandomForest
    logger.info("\n[ensemble] Training RandomForest...")
    final_rf = None
    for fold, (tr_idx, te_idx) in enumerate(cv_indices):
        X_fold_tr, X_fold_te = X_tr[tr_idx], X_tr[te_idx]
        y_fold_tr, y_fold_te = y_tr[tr_idx], y_tr[te_idx]
        if len(np.unique(y_fold_tr)) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_fold_tr, y_fold_tr)
        pred = rf.predict(X_fold_te)
        cv_scores["rf"].append({"fold": fold, "accuracy": round(accuracy_score(y_fold_te, pred), 4), "f1": round(f1_score(y_fold_te, pred, average="weighted", zero_division=0), 4)})
        final_rf = rf
    if final_rf is not None:
        models["rf"] = final_rf

    # XGBoost
    logger.info("[ensemble] Training XGBoost...")
    final_xgb = None
    for fold, (tr_idx, te_idx) in enumerate(cv_indices):
        X_fold_tr, X_fold_te = X_tr[tr_idx], X_tr[te_idx]
        y_fold_tr, y_fold_te = y_tr[tr_idx], y_tr[te_idx]
        if len(np.unique(y_fold_tr)) < 2:
            continue
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss", verbosity=0)
            xgb.fit(X_fold_tr, y_fold_tr)
            if hasattr(xgb, "get_booster"):
                xgb.get_booster().feature_names = None
            pred = xgb.predict(X_fold_te)
            cv_scores["xgb"].append({"fold": fold, "accuracy": round(accuracy_score(y_fold_te, pred), 4), "f1": round(f1_score(y_fold_te, pred, average="weighted", zero_division=0), 4)})
            final_xgb = xgb
        except Exception as e:
            logger.warning(f"    ⚠️ XGBoost fold {fold}: {e}")
    if final_xgb is not None:
        models["xgb"] = final_xgb

    # LogisticRegression (wrapped in Pipeline with StandardScaler)
    from sklearn.pipeline import Pipeline as _Pipeline
    from sklearn.preprocessing import StandardScaler as _Scaler
    logger.info("[ensemble] Training LogisticRegression (with StandardScaler)...")
    final_lr = None
    for fold, (tr_idx, te_idx) in enumerate(cv_indices):
        X_fold_tr, X_fold_te = X_tr[tr_idx], X_tr[te_idx]
        y_fold_tr, y_fold_te = y_tr[tr_idx], y_tr[te_idx]
        if len(np.unique(y_fold_tr)) < 2:
            continue
        lr_pipe = _Pipeline([
            ("scaler", _Scaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
        lr_pipe.fit(X_fold_tr, y_fold_tr)
        pred = lr_pipe.predict(X_fold_te)
        cv_scores["lr"].append({"fold": fold, "accuracy": round(accuracy_score(y_fold_te, pred), 4), "f1": round(f1_score(y_fold_te, pred, average="weighted", zero_division=0), 4)})
        final_lr = lr_pipe
    if final_lr is not None:
        models["lr"] = final_lr

    # Ensemble weights from CV F1
    weights = {}
    for mn in ["xgb", "rf", "lr"]:
        scores = cv_scores[mn]
        weights[mn] = max(np.mean([s["f1"] for s in scores]), 0.01) if scores else 0.0
    tw = sum(weights.values())
    if tw > 0:
        for k in weights:
            weights[k] /= tw
    logger.info(f"\n[ensemble] Weights: xgb={weights.get('xgb',0):.3f}, rf={weights.get('rf',0):.3f}, lr={weights.get('lr',0):.3f}")

    # Q8-fix: 上线模型必须在全量训练数据上重训 —— 原实现把最后一折
    # （GroupKFold 第 5 折，仅 ~80% 股票）的子集模型存为线上模型，
    # 20% 股票永不参与上线模型。CV 折仅用于评估权重，最终模型用 X_tr/y_tr 全量重训。
    # P1-Q8-fix (H05): 最终模型用全量 X_tr/y_tr（已拼接所有股票）重新 fit，
    # 不再依赖 CV 循环残留的最后一折子集切分（循环变量为 X_fold_tr/X_fold_te）。
    logger.info("\n[ensemble] Retraining final models on FULL training data...")
    if len(np.unique(y_tr)) >= 2:
        # RandomForest
        rf_final = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_final.fit(X_tr, y_tr)
        models["rf"] = rf_final
        # XGBoost
        try:
            from xgboost import XGBClassifier
            xgb_final = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss", verbosity=0)
            xgb_final.fit(X_tr, y_tr)
            if hasattr(xgb_final, "get_booster"):
                xgb_final.get_booster().feature_names = None
            models["xgb"] = xgb_final
        except Exception as e:
            logger.warning(f"    ⚠️ XGBoost final: {e}")
        # LogisticRegression
        lr_final = _Pipeline([
            ("scaler", _Scaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
        lr_final.fit(X_tr, y_tr)
        models["lr"] = lr_final
    else:
        logger.warning("    ⚠️ 训练标签不足 2 类，保留 CV 末折模型作为兜底")

    # Save pickle
    total_n = len(X_tr) + len(X_te)
    n_feat = X_tr.shape[1]
    ed = {"models": models, "weights": weights, "cv_scores": cv_scores, "feature_names": feature_names, "training_date": datetime.now(CST).isoformat(), "n_samples": int(total_n), "n_symbols": len(train_symbols)}
    with open(cache_dir / "ml_ensemble.pkl", "wb") as f:
        pickle.dump(ed, f)

    # Save metadata JSON
    meta = {"status": "trained", "training_date": datetime.now(CST).isoformat(), "n_samples": int(total_n), "n_symbols": len(train_symbols), "n_features": int(n_feat), "ensemble_weights": {k: round(v, 4) for k, v in weights.items()}, "cv_scores": cv_scores, "models_trained": [k for k, v in models.items() if v is not None]}
    (cache_dir / "ml_ensemble.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    # Invalidate old v3 model cache to prevent stale predict_stock usage
    old_v3 = cache_dir / "ml_model.json"
    if old_v3.exists():
        old_v3.unlink()
        logger.info("  🧹 Cleared stale v3 cache (ml_model.json)")

    # Feature importance tracking
    _record_top_features({k: v for k, v in models.items() if v is not None}, feature_names, cache_dir / "feature_importance.json")

    logger.info(f"\n[ensemble] ✅ Done in {_time.time()-t0:.0f}s")
    return meta


def auto_retrain(force=False):
    """Retrain ensemble if >24h old or force=True.
    Checks drift first; drift-triggered retrain overrides age check.
    """
    from quant_system.feature_store import compute_drift

    # 1. Check drift first
    drift = compute_drift("ensemble")
    drift_detected = drift.get("drift_detected", False)
    drift_note = f"(drift={drift_detected}, Δ={drift.get('delta', 0):+.2%})" if drift.get('n_samples', 0) >= 30 else "(insufficient samples)"

    # 2. Check age
    meta_path = Path.home() / ".quant_system" / "ml_ensemble.json"
    age_h = 99.0
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            last_ts = meta.get("training_date", "")
            if last_ts:
                age_h = (datetime.now(CST) - datetime.fromisoformat(last_ts)).total_seconds() / 3600
        except Exception as e:
            logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)

    # 3. Decision: retrain if forced, drift-triggered, or older than 24h
    if not force and not drift_detected and age_h < 24:
        msg = f"[auto_retrain] Recent ({age_h:.1f}h), no drift, skipping. {drift_note}"
        logger.info(msg)
        return {"status": "skipped", "age_hours": round(age_h, 1), "drift": drift}

    reason = "drift_detected" if drift_detected else ("force" if force else "age")
    logger.info(f"[auto_retrain] 🔄 Retraining ({reason}, age={age_h:.1f}h) {drift_note}...")
    # P2-Q8-fix (M520): 重训股票池与 train_ensemble 默认池统一为同一常量，
    # 避免自动重训后模型股票域漂移（原列表缺 002230、300274，且 300059 顺序移动）。
    r = train_ensemble(train_symbols=DEFAULT_TRAIN_SYMBOLS)
    return {"status": "retrained", "n_samples": r.get("n_samples"), "drift": drift, "reason": reason}


def predict_ensemble(symbol, horizon_days=20):
    """Ensemble prediction using weighted average of XGB, RF, LR.
    Caches extracted features into feature_store.

    Q8-fix: 默认 horizon 改为 20 —— 模型训练标签是 20 日三重障碍（±2%，max_hold=20），
    原默认报告 horizon_days=5（"5日预测"）与真实标签口径不符；
    概率列用 model.classes_ 显式对齐，不再依赖 sklearn 对类别 {-1,0,1} 排序的巧合。
    # P1-Q8-fix (H06): 对外口径统一为 20 日三重障碍（horizon_days 默认=20），
    # 不再向用户宣称"5日预测"；proba 列经 _pad_proba_3class + classes_ 显式对齐。
    """
    pkl_path = _quant_cache_dir() / "ml_ensemble.pkl"
    if not pkl_path.exists():
        return {"error": "Not trained. Run --ensemble-train first."}
    with open(pkl_path, "rb") as f:
        ensemble = pickle.load(f)
    models, weights = ensemble["models"], ensemble["weights"]

    # 2026-08-22: 本地kline优先(免akshare网络失败), 与训练一致
    df = None
    _klp3 = Path(__file__).resolve().parent.parent / "data_warehouse" / "kline" / f"{symbol}.parquet"
    if _klp3.exists():
        _raw3 = pd.read_parquet(_klp3, columns=["date", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame({"日期": _raw3["date"].astype(str).str[:10], "开盘": _raw3["open"],
                           "最高": _raw3["high"], "最低": _raw3["low"], "收盘": _raw3["close"],
                           "成交量": _raw3["volume"]})
    if df is None:
        import akshare as ak
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20230101", adjust="qfq")
        except Exception as e:
            return {"error": f"Fetch failed: {e}"}
    if df is None or len(df) < 300:
        return {"error": f"数据不足({len(df) if df is not None else 0})"}
    X, _, _ = extract_features(df.to_dict("records"))
    if len(X) == 0:
        return {"error": "特征提取失败"}
    x_latest = X[-1].reshape(1, -1)

    # ── 特征缓存 (最新一天) ──
    try:
        from quant_system.feature_store import cache_features as _cf
        last_date = str(df["日期"].iloc[-1])[:10]
        feat_dict = {FEATURES[i]: float(X[-1][i]) for i in range(min(len(FEATURES), X.shape[1]))}
        _cf(symbol, last_date, feat_dict)
    except Exception as e:
        logging.getLogger(__name__).error(f"[ml_signals] 操作失败: {e}", exc_info=True)

    model_results = {}
    for mn in ["xgb", "rf", "lr"]:
        m = models.get(mn)
        w = weights.get(mn, 0.0)
        if m is None or w == 0.0:
            model_results[mn] = {"status": "unavailable", "weight": w}
            continue
        try:
            pred_raw = int(m.predict(x_latest)[0])
            # Q8-fix: 把模型原始类标签（{-1,0,1} 或 {0,1,2}）映射到规范 {0,1,2}
            # （0跌/1盘/2涨），不再依赖 sklearn 类别排序巧合。
            pred = _canonical_class(pred_raw, m)
            # Q8-fix: 用 classes_ 把概率列显式对齐到 {0,1,2}（0跌/1盘/2涨）
            # P1-Q8-fix (H06): proba 列序经 model.classes_ 显式对齐，缺列补 0，
            # 避免某折只含 2 类时列序错位（如 P(上涨) 被当 P(盘整)）。
            proba_raw = m.predict_proba(x_latest)[0] if hasattr(m, "predict_proba") else None
            if proba_raw is not None:
                pa = _pad_proba_3class(proba_raw, m)[0]
                proba = [round(float(p), 4) for p in pa]
            else:
                proba = None
            model_results[mn] = {"status": "ok", "prediction": pred, "weight": round(w, 4), "probabilities": proba}
        except Exception as e:
            model_results[mn] = {"status": "error", "error": str(e), "weight": w}

    # Weighted ensemble probability
    wp = np.zeros(3)
    tw = 0.0
    for mn in ["xgb", "rf", "lr"]:
        mr = model_results.get(mn, {})
        w = weights.get(mn, 0.0)
        if mr.get("status") != "ok" or mr.get("probabilities") is None:
            continue
        pa = np.array(mr["probabilities"])
        if len(pa) >= 3:
            wp[:3] += pa[:3] * w
        else:
            wp[0] += pa[0] * w
            wp[2] += pa[-1] * w
            wp[1] += 0.5 * w
        tw += w
    if tw == 0:
        return {"error": "No valid predictions"}
    wp /= tw
    pred = int(np.argmax(wp))
    return {
        "symbol": symbol, "horizon_days": horizon_days,
        "prediction": pred, "prediction_label": {0: "DOWN", 1: "FLAT", 2: "UP"}.get(pred, "?"),
        "confidence": round(float(np.max(wp)), 4),
        "probs": {"down": round(float(wp[0]), 4), "flat": round(float(wp[1]), 4), "up": round(float(wp[2]), 4)},
        "models": model_results,
        "ensemble_weights": {k: round(v, 4) for k, v in weights.items()},
        "timestamp": datetime.now(CST).isoformat(),
    }


def _format_ensemble_prediction(result):
    if "error" in result:
        return f"⚠️ {result['error']}"
    lines = ["═" * 56,
             f"  🎯 Ensemble Prediction — {result['symbol']}",
             f"     Horizon: {result['horizon_days']}d",
             f"     Signal:  {result['prediction_label']} (class {result['prediction']})",
             f"     Confidence: {result['confidence']:.1%}",
             f"     Prob(DOWN)={result['probs']['down']:.1%}  FLAT={result['probs']['flat']:.1%}  UP={result['probs']['up']:.1%}",
             "─" * 56]
    for mn in ["xgb", "rf", "lr"]:
        mr = result["models"].get(mn, {})
        w = result["ensemble_weights"].get(mn, 0)
        lbl = {0: "DOWN", 1: "FLAT", 2: "UP"}.get(mr.get("prediction", -1), "N/A") if mr.get("status") == "ok" else "N/A"
        lines.append(f"  {mn.upper():>4} → {lbl}  (weight={w:.2f})")
    lines.append("═" * 56)
    return "\n".join(lines)


def _format_ensemble_train_result(result):
    if "error" in result or result.get("status") == "error":
        return f"⚠️ {result.get('error', 'Unknown error')}"
    lines = ["═" * 56,
             "  ✅ Ensemble Training Complete",
             f"     Samples:  {result.get('n_samples', '?')}",
             f"     Symbols:  {result.get('n_symbols', '?')}",
             f"     Features: {result.get('n_features', '?')}",
             f"     Date:     {result.get('training_date', '?')}"]
    weights = result.get("ensemble_weights", {})
    if weights:
        lines.append("─" * 56)
        lines.append("  Ensemble Weights (from CV F1):")
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            lines.append(f"    {name.upper():>4}: {w:.1%}")
    lines.append("═" * 56)
    return "\n".join(lines)


def _demo_ml_upgrade() -> dict[str, Any]:
    """Run a lightweight demonstration of the v3.1 ML upgrade helpers.

    P2-Q8-fix (M515): 原 _create_labels / _timeseries_cv 定义后从未被调用（断头功能），
    现接入本自检路径使其被真实执行（_timeseries_cv 的 embargo 默认已修正为 20）。
    """
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 420))
    X = rng.normal(size=(400, min(N_FEATURES, 12)))
    y = label_triple_barrier(close, max_hold=20)[:400]
    if len(np.unique(y)) < 2:
        y = np.where(rng.normal(size=400) > 0, 1, -1)
    wf = walk_forward_validate(X, y, n_train=252, n_test=63)
    drift = detect_feature_drift(X[-80:] + 0.05, X[:252])
    # Q8-fix: 显式正类 1（二值 demo 标签），与 calibrate_proba 默认 positive_class=2（上涨）区分
    cal = calibrate_proba((y[-100:] == 1).astype(int), rng.uniform(0.05, 0.95, 100), positive_class=1)
    # P2-Q8-fix (M515): 调用 _create_labels / _timeseries_cv，消除断头功能
    demo_df = pd.DataFrame({"收盘": close, "最高": close * 1.01, "最低": close * 0.99})
    multi_labels = _create_labels(demo_df, forward_days=[5, 20])
    tscv_splits = list(_timeseries_cv(X, y, n_splits=3))
    return {
        "triple_barrier_labels": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "multi_horizon_labels": {
            k: int(np.nansum(v)) for k, v in multi_labels.items() if k.startswith("cls_")
        },
        "timeseries_cv_splits": len(tscv_splits),
        "walk_forward": wf.get("summary", {}),
        "drift_detected": drift.get("drift_detected", False),
        "drift_alerts": drift.get("n_alerts", 0),
        "brier_before": cal.get("brier_before"),
        "brier_after": cal.get("brier_after"),
    }


# ── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--predict", type=str, default="")
    # P2-Q8-fix (L529): --shap/--correlation 当前与 --train 共用完整 train_model() 路径，
    # 帮助文本如实标注，不再声称"仅SHAP/仅相关性"。
    ap.add_argument("--shap", action="store_true",
                    help="触发完整 train_model()（含SHAP）—— 当前实现与 --train 共用同一路径")
    ap.add_argument("--correlation", action="store_true",
                    help="触发完整 train_model()（含相关性）—— 当前实现与 --train 共用同一路径")
    # ── V7 additions ──
    ap.add_argument("--ensemble-train", action="store_true")
    ap.add_argument("--ensemble-predict", type=str, default="")
    ap.add_argument("--auto-retrain", action="store_true")
    ap.add_argument("--demo-upgrade", action="store_true")
    args = ap.parse_args()

    if args.demo_upgrade:
        print(json.dumps(_demo_ml_upgrade(), ensure_ascii=False, indent=2))
    elif args.ensemble_train:
        r = train_ensemble()
        if "error" in r or r.get("status") == "error":
            print(f"⚠️ {r.get('error', 'Unknown error')}")
        else:
            print(_format_ensemble_train_result(r))
    elif args.ensemble_predict:
        r = predict_ensemble(args.ensemble_predict)
        if "error" in r:
            print(f"⚠️ {r['error']}")
        else:
            print(_format_ensemble_prediction(r))
    elif args.auto_retrain:
        r = auto_retrain()
        print(f"  ➡ {r.get('status', 'Done')}")
    elif args.train or args.importance or args.shap or args.correlation:
        r = train_model()
        if "error" in r:
            print(f"⚠️ {r['error']}")
        else:
            print(format_importance(r))
    elif args.predict:
        r = predict_stock(args.predict)
        print(format_prediction(r))
    else:
        ap.print_help()
