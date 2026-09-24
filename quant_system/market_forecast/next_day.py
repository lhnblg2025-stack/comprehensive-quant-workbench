"""
next_day.py — QuantV6 次日涨跌概率预测
ML 集成（逻辑回归+随机森林+梯度提升）+ 概率校准 + 降级路径（相似日统计）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.nextday")


@dataclass
class NextDayPrediction:
    """次日方向预测。"""
    p_up: float = 0.5
    p_down: float = 0.5
    expected_return_1d: float = 0.0
    confidence: float = 0.0        # 0~1，基于样本量与模型分歧
    method: str = "prior"          # ml / similarity / prior
    model_detail: dict = field(default_factory=dict)


def build_features(index_df: pd.DataFrame, for_training: bool = True,
                   label_mode: str = "close_close") -> pd.DataFrame:
    """从指数日线构建特征矩阵（技术特征）。

    P0-3 修复：for_training=True（训练）时丢弃无 label 的行；
    for_training=False（预测）时保留最后一行——其 label/fwd_ret 为 NaN 但不影响
    特征列，保证 predict_proba 的 p[-1] 对应含今日特征的 df.iloc[[-1]] 行。
    另外 label 用 np.where 生成，末行（无次日数据）为 NaN 而非 False。

    V6.1 新增 label_mode（剔除隔夜跳空）:
      - "close_close"（默认, 保持兼容）: 次日收盘涨跌, label = pct.shift(-1) > 0
      - "open_open": 次日开盘相对今日开盘涨跌, label = (next_open/open - 1) > 0。
        A股隔夜跳空占收益大头且由外盘/新闻驱动, open-to-open 剔除跳空后
        更能反映日内 alpha。指数日线需含 open 列。
    """
    if index_df is None or len(index_df) < 60:
        return pd.DataFrame()
    close = index_df["close"].astype(float)
    df = pd.DataFrame(index=index_df.index)
    df["pct"] = close.pct_change() * 100
    df["mom5"] = (close / close.shift(5) - 1) * 100
    df["mom10"] = (close / close.shift(10) - 1) * 100
    df["mom20"] = (close / close.shift(20) - 1) * 100
    df["ma5_gap"] = (close / close.rolling(5).mean() - 1) * 100
    df["ma20_gap"] = (close / close.rolling(20).mean() - 1) * 100
    df["ma60_gap"] = (close / close.rolling(60).mean() - 1) * 100
    df["std20"] = df["pct"].rolling(20).std()
    df["rsi14"] = _rsi(close)
    df["vol_ratio"] = df["pct"].rolling(5).mean().abs() / (df["pct"].rolling(20).std() + 1e-9)
    df["high_dist"] = (close / close.rolling(60).max() - 1) * 100
    df["low_dist"] = (close / close.rolling(60).min() - 1) * 100
    df["up3"] = (df["pct"] > 0).rolling(3).sum()   # 近3日上涨天数
    next_pct = df["pct"].shift(-1)
    # V6.1: label 构造支持剔除隔夜跳空（open-to-open）
    if label_mode == "open_open" and "open" in index_df.columns:
        next_open = index_df["open"].astype(float).shift(-1)
        cur_open = index_df["open"].astype(float)
        open_ret = (next_open / cur_open - 1) * 100
        df["fwd_ret"] = open_ret
        df["label"] = np.where(open_ret.notna(), (open_ret > 0).astype(int), np.nan)
    else:
        # 次日上涨=1；无次日数据的行 label 为 NaN（不能是 False/0，否则训练集混入假标签）
        df["label"] = np.where(next_pct.notna(), (next_pct > 0).astype(int), np.nan)
        df["fwd_ret"] = next_pct
    if for_training:
        df = df.dropna(subset=["label"])
    return df.dropna(thresh=len(df.columns) - 3)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


class NextDayModel:
    """次日方向 ML 模型（带训练/预测/降级）。"""

    FEATURES = ["pct", "mom5", "mom10", "mom20", "ma5_gap", "ma20_gap",
                "ma60_gap", "std20", "rsi14", "vol_ratio", "high_dist", "low_dist", "up3"]

    def __init__(self, min_train: int = 400, embargo: int = 5, label_mode: str = "close_close"):
        self.min_train = min_train
        self.embargo = embargo
        self.label_mode = label_mode
        self.model = None
        self.is_trained = False
        self.model_detail: dict = {}
        # P1-9：历史（预测概率, 实际结果）校准对；训练时由 OOS 折生成
        self.calibration_history = None

    def train(self, index_df: pd.DataFrame) -> bool:
        df = build_features(index_df, label_mode=self.label_mode)
        if df is None or len(df) < self.min_train:
            log.warning(f"训练数据不足: {len(df) if df is not None else 0} < {self.min_train}")
            return False
        X = df[self.FEATURES].fillna(0).values
        y = df["label"].values
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            models = [
                LogisticRegression(max_iter=1000, class_weight="balanced"),
                RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42),
                GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
            ]
            scores = self._select_scores(X, y, models)
            best_idx = int(np.argmax(scores))
            self.model = make_pipeline(StandardScaler(), models[best_idx])
            self.model.fit(X, y)
            self.is_trained = True
            self.model_detail = {"scores": [round(s, 3) for s in scores],
                                 "best": best_idx, "samples": len(X)}
            # P1-9：生成 OOS（样本外）预测-实际对，供概率校准使用
            self._build_calibration_history(X, y, models[best_idx])
            return True
        except Exception as e:
            log.warning(f"ML 训练失败，降级相似日: {str(e)[:100]}")
            return False

    def _select_scores(self, X: np.ndarray, y: np.ndarray, models: list):
        """P1-8：模型选择分数。

        优先用 Purged 时序交叉验证（TimeSeriesSplit(gap=embargo)）；
        sklearn 不可用/样本过小时回退到单次 80/20 切分，且切分时
        split_idx -= embargo 并在训练集末尾丢弃 embargo 行（gap 间隙）。
        """
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit

        emb = max(int(self.embargo or 0), 0)
        if len(X) >= 80:
            try:
                tscv = TimeSeriesSplit(n_splits=3, gap=emb)
                scores = []
                for m in models:
                    pipe = make_pipeline(StandardScaler(), m)
                    s = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy")
                    scores.append(float(np.mean(s)))
                return scores
            except Exception as e:
                log.warning(f"Purged CV 失败，回退单次切分: {str(e)[:80]}")

        # 回退：单次 80/20 时序切分 + embargo 间隙（丢弃训练集末尾 embargo 行）
        split = int(len(X) * 0.8)
        tr_end = max(split - emb, 0)
        X_tr, X_te, y_tr, y_te = X[:tr_end], X[split:], y[:tr_end], y[split:]
        scores = []
        for m in models:
            pipe = make_pipeline(StandardScaler(), m)
            pipe.fit(X_tr, y_tr)
            scores.append(float(pipe.score(X_te, y_te)))
        return scores

    def _build_calibration_history(self, X: np.ndarray, y: np.ndarray, best_est) -> None:
        """P1-9：用 best 模型在 Purged 时序折上生成 OOS (p, y) 校准对。

        失败或样本不足时保持 None（predict_proba 对无历史情况保持原值）。
        """
        try:
            from sklearn.base import clone
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            emb = max(int(self.embargo or 0), 0)
            if len(X) < 80:
                return
            tscv = TimeSeriesSplit(n_splits=3, gap=emb)
            hist_p: list[float] = []
            hist_y: list[float] = []
            for tr_idx, te_idx in tscv.split(X):
                pipe = make_pipeline(StandardScaler(), clone(best_est))
                pipe.fit(X[tr_idx], y[tr_idx])
                probs = pipe.predict_proba(X[te_idx])
                up_col = self._positive_class_column(probs, getattr(pipe, "classes_", None))
                hist_p.extend(probs[:, up_col].astype(float).tolist())
                hist_y.extend(y[te_idx].astype(float).tolist())
            if len(hist_p) >= 30:
                self.calibration_history = pd.DataFrame({"p": hist_p, "y": hist_y})
        except Exception as e:
            log.warning(f"校准历史生成失败: {str(e)[:80]}")
            self.calibration_history = None

    @staticmethod
    def _positive_class_column(probs: np.ndarray, classes) -> int:
        """Return the probability column representing the positive/up class.

        Degenerate models can expose a single class, or classes/probability columns
        can be inconsistent after fallback estimators. In those cases, use the most
        likely column of the latest row to keep prediction and calibration paths from
        failing on `classes_[1]` indexing.
        """
        if probs is None or getattr(probs, "ndim", 0) != 2 or probs.shape[1] == 0:
            return 0
        if classes is not None:
            try:
                class_list = list(classes)
                if 1 in class_list:
                    col = class_list.index(1)
                    if 0 <= col < probs.shape[1]:
                        return int(col)
            except Exception as e:
                log.error(f"[next_day] 操作失败: {e}", exc_info=True)
        return int(probs[-1].argmax())

    def predict_proba(self, index_df: pd.DataFrame) -> dict:
        """预测次日上涨概率。未训练返回 None。"""
        if not self.is_trained or self.model is None:
            return {}
        # P0-3：预测时保留最后一行（含今日特征，label 为 NaN），
        # p[-1] 因此对应 df.iloc[[-1]]（今日特征 → 明日涨跌），不再滞后一天。
        df = build_features(index_df, for_training=False, label_mode=self.label_mode)
        if df is None or len(df) == 0:
            return {}
        X = df[self.FEATURES].fillna(0).values
        p = self.model.predict_proba(X)
        # M5：防御单类别模型（训练集退化为单类时 classes_ 可能仅 1 类）。
        # 不做 try 保护的话 IndexError 会穿透 predict_next_day -> SignalCenter 全链路。
        up_col = self._positive_class_column(p, getattr(self.model, "classes_", None))
        latest_p = float(p[-1][up_col])
        recent_win = float((df["pct"].tail(20) > 0).mean()) if len(df) >= 20 else 0.5
        # 模型输出 + 近期胜率平滑
        final_p = 0.7 * latest_p + 0.3 * recent_win
        # P1-9：概率校准接线——用历史 OOS (预测, 实际) 对校准；无历史时保持原值
        from quant_system.market_forecast.calibration import calibrate_by_hit_rate
        final_p = calibrate_by_hit_rate(final_p, getattr(self, "calibration_history", None))
        # P2-3：expected_return_1d 明确为「历史基准期望」——最近 20 个有次日数据的
        # 交易日次日收益均值（无偏估计），并非模型回归输出；模型只输出方向概率。
        # 上层若同时有相似日统计，应优先用 similarity.expected_return_1d（真实分布期望）。
        hist_ret = df["fwd_ret"].tail(20).dropna()
        return {"p_up": round(final_p, 3), "p_down": round(1 - final_p, 3),
                # V6.1: 输出模型原始概率（未经平滑/校准），供置信度门槛判断
                "prob_raw": round(float(latest_p), 4),
                "expected_return_1d": round(float(hist_ret.mean()), 3) if len(hist_ret) else 0.0}


def predict_next_day(index_df: pd.DataFrame, similarity: dict | None = None,
                     model: NextDayModel | None = None) -> NextDayPrediction:
    """
    综合次日预测：
    1) ML 模型（已训练）→ 2) 相似日统计 → 3) 先验。
    similarity: SimilarDays 对象（可选）。
    """
    if model is not None and model.is_trained:
        ml = model.predict_proba(index_df)
        if ml:
            pred = NextDayPrediction(
                p_up=ml["p_up"], p_down=ml["p_down"],
                expected_return_1d=ml["expected_return_1d"],
                # P1-3：confidence 依据模型实际训练 scores 数量（原 ml dict 中无 model_detail 键，恒为 0.5）
                confidence=0.6 if len(getattr(model, "model_detail", {}).get("scores", [])) else 0.5,
                method="ml", model_detail=model.model_detail,
            )
            # 与相似日融合
            if similarity is not None and similarity.n_matches >= 5:
                p_up = 0.6 * pred.p_up + 0.4 * (1 - similarity.p_down_1d)
                pred.p_up = round(p_up, 3)
                pred.p_down = round(1 - p_up, 3)
                pred.expected_return_1d = round(
                    0.5 * pred.expected_return_1d + 0.5 * similarity.expected_return_1d, 3)
                # P1-2 标记：已融合相似日，上层 aggregate 不应再对 similarity 重复计权
                pred.model_detail = {**pred.model_detail, "fused_similarity": True}
            return pred

    if similarity is not None and similarity.n_matches >= 5:
        return NextDayPrediction(
            p_up=round(1 - similarity.p_down_1d, 3),
            p_down=similarity.p_down_1d,
            expected_return_1d=similarity.expected_return_1d,
            confidence=min(0.7, 0.3 + similarity.n_matches * 0.04),
            method="similarity",
            model_detail={"n_matches": similarity.n_matches},
        )

    # 先验：近期涨跌分布
    if index_df is not None and len(index_df) >= 60:
        ret = index_df["close"].pct_change().tail(60).dropna()
        p_up = float((ret > 0).mean())
        return NextDayPrediction(
            p_up=round(p_up, 3), p_down=round(1 - p_up, 3),
            expected_return_1d=round(float(ret.mean() * 100), 3),
            confidence=0.3, method="prior",
        )
    return NextDayPrediction()
