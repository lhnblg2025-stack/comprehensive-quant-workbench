"""
oos_pool_rule.py — W2.4 OOS 入池硬规则（样本外符号翻转 → 自动降权/退池）
============================================================================
问题背景：因子系统 299 注册 / 185 IC，但 OOS 符号保持率仅 34.9%（41 翻转）。
翻转因子若继续以原权重入池，等于把样本内过拟合的方向固化为组合暴露。

规则（硬约束，可直接单测）：
  1. 同号（sign(in_sample_ic) == sign(oos_ic)）          → keep,   ×1.0
  2. 翻转（符号不一致）                                  → downweight, ×downweight_multiplier(0.5)
  3. 严重翻转（翻转且 |OOS IC| >= severe_flip_abs_ic）   → retire, ×0.0
  4. 连续多期翻转（含当期连翻 >= retire_flip_streak）     → retire, ×0.0
  5. |OOS IC| < min_abs_oos_ic（方向近乎噪声）           → retire, ×0.0
  6. 缺失 OOS（oos_ic=None/NaN）                         → 保守 downweight（默认 ×0.5）

本模块只做纯计算与只读加载，不写任何状态、不吞异常：
  - 非法参数（权重越界 / 阈值非法 / 非法 action）→ raise ValueError。
  - 载入 OOS 报告失败 → 返回空 map（调用方据此"规则不生效"），并显式告警，
    绝不静默伪装成"已验证"。

接入点：quant_system/ic_factors/composite.py 的 DynamicFactorSelector.weights /
CompositeEngine.compute（因子入池/权重计算路径），见 apply_oos_rule_to_weights。
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

log = logging.getLogger(__name__)

# ── 默认阈值 ────────────────────────────────────────────────────────────
DEFAULT_MIN_ABS_OOS_IC = 0.005        # |OOS IC| 低于该值视为方向噪声 → 退池
DEFAULT_DOWNWEIGHT_MULTIPLIER = 0.5   # 单期翻转 → 降权 50%
DEFAULT_RETIRE_FLIP_STREAK = 2        # 连续翻转期数（含当期）>= 该值 → 退池
DEFAULT_SEVERE_FLIP_ABS_IC = 0.02     # 翻转且 |OOS IC| 达该阈值 → 严重翻转退池
DEFAULT_MISSING_OOS_ACTION = "downweight"   # 缺失 OOS 的保守动作
DEFAULT_MISSING_OOS_MULTIPLIER = 0.5        # 缺失 OOS 的保守降权系数

# 报告默认路径（与 composite.py 的 IC 报告同目录）
OOS_REPORT_DEFAULT = Path(__file__).resolve().parents[2] / "generated" / "ic_report" / "IC_OOS_REPORT.json"

def _num(x) -> float | None:
    """把标量收敛为 float；None/NaN/inf/不可解析 → None（视为缺失，走保守分支）。"""
    if x is None:
        return None
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.item() if getattr(x, "size", 1) == 1 else None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _validate_params(downweight_multiplier: float, retire_flip_streak: int,
                     severe_flip_abs_ic: float, min_abs_oos_ic: float,
                     missing_oos_action: str, missing_oos_multiplier: float) -> None:
    if not 0.0 <= downweight_multiplier <= 1.0:
        raise ValueError(f"downweight_multiplier 必须在 [0,1]，得到 {downweight_multiplier}")
    if not 0.0 <= missing_oos_multiplier <= 1.0:
        raise ValueError(f"missing_oos_multiplier 必须在 [0,1]，得到 {missing_oos_multiplier}")
    if int(retire_flip_streak) < 1:
        raise ValueError(f"retire_flip_streak 必须 >= 1，得到 {retire_flip_streak}")
    if min_abs_oos_ic < 0:
        raise ValueError(f"min_abs_oos_ic 必须 >= 0，得到 {min_abs_oos_ic}")
    if severe_flip_abs_ic < 0:
        raise ValueError(f"severe_flip_abs_ic 必须 >= 0，得到 {severe_flip_abs_ic}")
    if missing_oos_action not in ("keep", "downweight", "retire"):
        raise ValueError(f"missing_oos_action 非法: {missing_oos_action}")


def oos_flip_rule(
    in_sample_ic: float | None,
    oos_ic: float | None,
    *,
    flip_history: int = 0,
    min_abs_oos_ic: float = DEFAULT_MIN_ABS_OOS_IC,
    downweight_multiplier: float = DEFAULT_DOWNWEIGHT_MULTIPLIER,
    retire_flip_streak: int = DEFAULT_RETIRE_FLIP_STREAK,
    severe_flip_abs_ic: float = DEFAULT_SEVERE_FLIP_ABS_IC,
    missing_oos_action: str = DEFAULT_MISSING_OOS_ACTION,
    missing_oos_multiplier: float = DEFAULT_MISSING_OOS_MULTIPLIER,
) -> dict:
    """OOS 入池硬规则：样本内 IC 符号 vs 样本外(OOS) IC 符号。

    参数
    ----
    in_sample_ic : 样本内 IC 均值（如 OOS 报告的 ic_mean_train）。
    oos_ic       : 样本外 IC 均值（如 ic_mean_test）。None/NaN 视为缺失。
    flip_history : 之前连续翻转的期数（不含当期）。当期翻转时，
                   连续翻转期数 = flip_history + 1。
    其余阈值见模块常量。

    返回
    ----
    dict: {action: keep|downweight|retire, weight_multiplier: float,
           flip: bool, reason: str}
    """
    _validate_params(downweight_multiplier, retire_flip_streak, severe_flip_abs_ic,
                     min_abs_oos_ic, missing_oos_action, missing_oos_multiplier)
    in_sample_ic = _num(in_sample_ic)
    oos_ic = _num(oos_ic)
    flip_history = max(0, int(flip_history))

    # 缺失 OOS → 保守处理（默认降权，可配置 keep/retire）
    if oos_ic is None:
        if missing_oos_action == "retire":
            return {"action": "retire", "weight_multiplier": 0.0, "flip": False, "reason": "oos_missing"}
        if missing_oos_action == "keep":
            return {"action": "keep", "weight_multiplier": 1.0, "flip": False, "reason": "oos_missing"}
        return {"action": "downweight", "weight_multiplier": missing_oos_multiplier,
                "flip": False, "reason": "oos_missing"}

    # 样本内 IC 缺失 → 无法确立方向，保守降权（不吞异常，仅业务规则）
    if in_sample_ic is None:
        return {"action": "downweight", "weight_multiplier": missing_oos_multiplier,
                "flip": False, "reason": "in_sample_ic_missing"}

    # |OOS IC| 低于阈值 → 方向近乎噪声，退池
    if abs(oos_ic) < min_abs_oos_ic:
        return {"action": "retire", "weight_multiplier": 0.0, "flip": False,
                "reason": "oos_ic_below_threshold"}

    in_sgn = 1 if in_sample_ic > 0 else -1 if in_sample_ic < 0 else 0
    if in_sgn == 0:
        return {"action": "downweight", "weight_multiplier": downweight_multiplier,
                "flip": False, "reason": "in_sample_ic_zero"}

    oos_sgn = 1 if oos_ic > 0 else -1
    flip = in_sgn != oos_sgn
    if not flip:
        return {"action": "keep", "weight_multiplier": 1.0, "flip": False,
                "reason": "sign_consistent"}

    # 翻转 → 至少降权
    streak = int(flip_history) + 1
    if streak >= int(retire_flip_streak):
        return {"action": "retire", "weight_multiplier": 0.0, "flip": True,
                "reason": f"consecutive_flip_streak_{streak}"}
    if abs(oos_ic) >= severe_flip_abs_ic:
        return {"action": "retire", "weight_multiplier": 0.0, "flip": True,
                "reason": "severe_flip"}
    return {"action": "downweight", "weight_multiplier": downweight_multiplier,
            "flip": True, "reason": "flip"}


def load_oos_report(path: str | Path | None = None) -> dict[str, dict]:
    """只读加载 OOS 报告，返回 {因子名: {in_sample_ic, oos_ic, flip, flip_history}}。

    - 优先读 `factors` 全量表（V2.4+ 写入，含逐因子 ic_mean_train/ic_mean_test）。
    - 兼容旧报告：只有 flipped_factors 时，翻转因子记 flip=True 但 oos_ic=None
      （规则会按"缺失 OOS"保守降权，不会误判为 keep）。
    - 文件缺失 / 解析失败 → 返回 {} 并告警（调用方应视作"规则不生效"，不得伪装已验证）。
    """
    p = Path(path) if path else OOS_REPORT_DEFAULT
    if not p.exists():
        log.warning(f"[oos_pool_rule] OOS 报告不存在: {p}（OOS 入池规则不生效）")
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - 只读加载失败要可见，不能静默
        log.warning(f"[oos_pool_rule] OOS 报告读取失败: {e}")
        return {}

    out: dict[str, dict] = {}
    factors = data.get("factors")
    if isinstance(factors, list):
        for row in factors:
            if not isinstance(row, dict) or not row.get("factor"):
                continue
            name = str(row["factor"])
            train = _num(row.get("ic_mean_train"))
            test = _num(row.get("ic_mean_test"))
            out[name] = {
                "in_sample_ic": train,
                "oos_ic": test,
                "flip": bool(row.get("sign_keep") is False),
                "flip_history": 0,
            }
        if out:
            return out
    # 旧格式回退：翻转名单（无 OOS IC 数值 → 规则保守降权）
    for name in data.get("flipped_factors", []) or []:
        out.setdefault(str(name), {
            "in_sample_ic": None,
            "oos_ic": None,
            "flip": True,
            "flip_history": 0,
        })
    return out


def apply_oos_rule_to_weights(
    weights: pd.Series,
    oos_map: Mapping[str, Mapping],
    *,
    log_actions: bool = True,
    **rule_kwargs,
) -> pd.Series:
    """把 OOS 硬规则施加到因子权重上（乘 weight_multiplier，retire → 0）。

    weights : 因子权重 Series（未归一亦可，本函数只乘系数）。
    oos_map : load_oos_report() 返回的 {因子名: {...}}；不在 map 内的因子保持原权重。
    返回   : 与原 weights 对齐的新 Series（不归一，由调用方继续归一）。
    """
    out = weights.astype(float).copy()
    for f in out.index:
        info = oos_map.get(str(f))
        if not info:
            continue
        # 参数非法时 oos_flip_rule 会抛 ValueError，此处不吞异常
        rule = oos_flip_rule(
            info.get("in_sample_ic"),
            info.get("oos_ic"),
            flip_history=int(info.get("flip_history", 0) or 0),
            **rule_kwargs,
        )
        out.loc[f] = out.loc[f] * rule["weight_multiplier"]
        if log_actions and rule["action"] != "keep":
            log.info(
                "[oos_pool_rule] 因子 %s: action=%s mult=%.2f reason=%s",
                f, rule["action"], rule["weight_multiplier"], rule["reason"],
            )
    return out


def flip_streak(recent_flips: Sequence[bool]) -> int:
    """计算最近连续翻转期数（从序列尾部数 True 的连续长度）。

    用于"连续多期翻转 → 退池"：把各期 flip 结果按时间升序传入，
    得到尾部连续 True 的个数；当期再翻转时 flip_history 取该值。
    """
    n = 0
    for v in reversed(list(recent_flips)):
        if v:
            n += 1
        else:
            break
    return n


__all__ = [
    "oos_flip_rule",
    "load_oos_report",
    "apply_oos_rule_to_weights",
    "flip_streak",
    "DEFAULT_MIN_ABS_OOS_IC",
    "DEFAULT_DOWNWEIGHT_MULTIPLIER",
    "DEFAULT_RETIRE_FLIP_STREAK",
    "DEFAULT_SEVERE_FLIP_ABS_IC",
    "DEFAULT_MISSING_OOS_ACTION",
    "DEFAULT_MISSING_OOS_MULTIPLIER",
]
