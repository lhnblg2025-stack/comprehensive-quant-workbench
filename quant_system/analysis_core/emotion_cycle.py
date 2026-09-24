"""
emotion_cycle — 情绪周期六阶段机（V11 短线 M1）

六阶段: 冰点 → 修复 → 发酵 → 高潮 → 分歧 → 退潮

判定逻辑: 基于 zt_daily_stats 的各指标对每个阶段打分，
取得分最高者为当前阶段；置信度 = (最高分 - 次高分) 归一化。

转移概率: 用历史统计阶段转移矩阵（stage[t] → stage[t+1]），
样本不足的矩阵单元自动降权标注。

用法:
  python3 -m quant_system.analysis_core.emotion_cycle --today
  python3 -m quant_system.analysis_core.emotion_cycle --history
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import ZT_DAILY_STATS  # noqa: E402

STAGES = ["ice", "repair", "ferment", "climax", "divergence", "ebb"]
STAGE_CN = {
    "ice": "冰点", "repair": "修复", "ferment": "发酵",
    "climax": "高潮", "divergence": "分歧", "ebb": "退潮",
}

# 六阶段规则（v1.1 加权命中）: (特征列, 比较符, 阈值)
# 特征列: zt_cnt/max_board/premium/jr1/zb_rate + 派生 zt_chg/mb_chg(当日-前日)
STAGE_RULES: dict[str, list[tuple[str, str, float]]] = {
    "ice":        [("zt_cnt", "le", 45), ("max_board", "le", 2), ("premium", "le", 0.0)],
    "repair":     [("zt_cnt", "ge", 30), ("zt_cnt", "le", 70), ("max_board", "ge", 2), ("premium", "ge", 0.0)],
    "ferment":    [("zt_cnt", "ge", 55), ("max_board", "ge", 3), ("jr1", "ge", 0.25), ("premium", "ge", 0.5)],
    "climax":     [("zt_cnt", "ge", 100), ("max_board", "ge", 5)],
    "divergence": [("zb_rate", "ge", 0.35), ("mb_chg", "le", -3), ("zt_chg", "le", -10)],
    "ebb":        [("max_board", "le", 2), ("zt_cnt", "le", 45), ("premium", "le", -1.0)],
}


def _ok(row: pd.Series, col: str, op: str, bound: float) -> bool:
    val = row.get(col)
    if val is None or pd.isna(val):
        return False
    if op == "ge":
        return val >= bound
    if op == "le":
        return val <= bound
    return False


def stage_score(row: pd.Series, stage: str) -> tuple[int, int, list[str]]:
    """返回 (命中数, 条件总数, 证据列表)。"""
    rules = STAGE_RULES[stage]
    hits, evidence = 0, []
    for col, op, bound in rules:
        if _ok(row, col, op, bound):
            hits += 1
            cn = {"zt_cnt": "涨停家数", "max_board": "最高板", "premium": "昨日涨停溢价",
                  "jr1": "1进2晋级率", "zb_rate": "炸板率",
                  "zt_chg": "涨停家数环比", "mb_chg": "最高板环比"}[col]
            evidence.append(f"{cn}{'≥' if op=='ge' else '≤'}{bound}")
    return hits, len(rules), evidence


def classify(row: pd.Series) -> dict:
    """顺序判定（强特征优先，游资实战思路）。

    高潮→分歧→退潮→冰点→发酵→修复，前序命中即返回。
    """
    zt = row.get("zt_cnt")
    mb = row.get("max_board")
    zb = row.get("zb_rate")
    jr1 = row.get("jr1")
    prem = row.get("premium")
    zt_chg = row.get("zt_chg")
    mb_chg = row.get("mb_chg")
    big_loss = row.get("big_loss_cnt")  # 昨涨停今跌>5%家数（亏钱效应，心法: 大面多=氛围差）
    ev: list[str] = []

    def _nan(x):
        return x is None or pd.isna(x)

    if not _nan(zt) and not _nan(mb) and zt >= 100 and mb >= 5:
        stage = "climax"; conf = min(zt / 150, 1.0)
        ev = [f"涨停{zt}≥100", f"最高{mb}板≥5"]
    elif not _nan(zb) and zb >= 0.35:
        stage = "divergence"; conf = min(zb / 0.5, 1.0)
        ev = [f"炸板率{zb:.0%}≥35%"]
    elif (not _nan(mb_chg) and mb_chg <= -3 and not _nan(zt_chg) and zt_chg <= -10) or \
         (not _nan(mb_chg) and mb_chg <= -3 and not _nan(prem) and prem < 0) or \
         (not _nan(big_loss) and big_loss >= 15):
        stage = "divergence"; conf = 0.7
        ev = [f"最高板压缩{mb_chg}板", f"涨停环比{zt_chg}", f"大面{big_loss}家(亏钱效应)"]
    elif not _nan(mb) and not _nan(zt) and mb <= 2 and zt <= 45 and \
            ((not _nan(prem) and prem < 0) or (not _nan(big_loss) and big_loss >= 10)):
        stage = "ebb"; conf = 0.75
        ev = [f"最高仅{mb}板", f"涨停{zt}≤45", f"大面{big_loss}家"]
    # 冰点优先于宽松退潮（2026-08-10 审计: 原 ebb 宽松分支先判定，冰点日全部被归为退潮）
    elif not _nan(zt) and zt <= 35 and (not _nan(mb) and mb <= 2):
        stage = "ice"; conf = 0.7
        ev = [f"涨停{zt}≤35", f"最高仅{mb}板"]
    elif not _nan(mb) and not _nan(zt) and mb <= 2 and zt <= 45:
        stage = "ebb"; conf = 0.6
        ev = [f"最高仅{mb}板", f"涨停{zt}≤45"]
    elif not _nan(zt) and zt <= 30:
        stage = "ice"; conf = 0.6
        ev = [f"涨停{zt}≤30"]
    elif not _nan(zt) and not _nan(mb) and zt >= 55 and mb >= 3 and \
            ((not _nan(jr1) and jr1 >= 0.2) or (not _nan(prem) and prem >= 0.3)):
        stage = "ferment"; conf = 0.55 + min(zt / 200, 0.3)
        ev = [f"涨停{zt}≥55", f"最高{mb}板≥3", f"溢价{prem:.1f}%"]
    else:
        stage = "repair"; conf = 0.35
        ev = [f"涨停{zt}", f"最高{mb}板"]

    return {
        "stage": stage,
        "stage_cn": STAGE_CN[stage],
        "confidence": round(float(conf), 2),
        "evidence": ev,
        "score": round(float(conf), 2),
        "runner_up": "-",
    }


def transition_matrix(stages: pd.Series) -> pd.DataFrame:
    """阶段转移矩阵 + 每阶段样本数。"""
    pairs = list(zip(stages[:-1], stages[1:]))
    mat = pd.DataFrame(0.0, index=STAGES, columns=STAGES)
    counts = Counter(pairs)
    row_cnt = Counter(stages[:-1])
    for (a, b), c in counts.items():
        if row_cnt[a] > 0:
            mat.loc[a, b] = c / row_cnt[a]
    mat["样本数"] = [row_cnt[s] for s in STAGES]
    return mat


def run_history() -> pd.DataFrame:
    if not ZT_DAILY_STATS.exists():
        raise FileNotFoundError("先运行 ladder --build")
    df = pd.read_parquet(ZT_DAILY_STATS)
    # 派生特征: 环比变化 + premium 缺失时用 3日均值回退
    df["zt_chg"] = df["zt_cnt"].diff().fillna(0)
    df["mb_chg"] = df["max_board"].diff().fillna(0)
    df["premium"] = df["premium"].fillna(df["premium_t3"])
    df["jr1"] = df["jr1"].fillna(df["jr1_t3"])
    stages = df.apply(lambda r: classify(r)["stage"], axis=1)
    df["stage"] = stages
    df["stage_cn"] = df["stage"].map(STAGE_CN)
    return df


def run_today(as_of: str | None = None) -> dict:
    """返回最新情绪；传入 as_of 时严格截断到该交易日，禁止历史复盘混入未来数据。"""
    df = run_history()
    if as_of:
        cutoff = pd.Timestamp(as_of)
        df = df[pd.to_datetime(df["date"]) <= cutoff]
        if df.empty:
            raise ValueError(f"情绪数据早于请求日期: {as_of}")
    row = df.iloc[-1]
    mat = transition_matrix(df["stage"])
    nxt = mat.loc[row["stage"]]
    top3 = nxt.drop("样本数").sort_values(ascending=False).head(3)
    # 2026-08-10 审计：标注数据基准日与落后天数，下游（fusion）据此降级，
    # 禁止静默用旧涨停统计判定情绪阶段。
    _data_date = str(row["date"].date())
    _today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    return {
        "date": _data_date,
        "data_date": _data_date,
        "lag_days": max(0, (pd.Timestamp(_today) - pd.Timestamp(_data_date)).days),
        **classify(row),
        "transition_probs": {f"{STAGE_CN[k]}({k})": round(float(v), 2) for k, v in top3.items()},
        "matrix_samples": int(mat.loc[row["stage"], "样本数"]),
        "zt_cnt": int(row["zt_cnt"]),
        "max_board": int(row["max_board"]),
        "zb_rate": round(float(row["zb_rate"]), 3) if not pd.isna(row["zb_rate"]) else None,
        "premium": round(float(row["premium"]), 2) if not pd.isna(row["premium"]) else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="情绪周期六阶段机")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    if args.history:
        df = run_history()
        print(df[["date", "stage", "stage_cn"]].tail(20).to_string(index=False))
        print("\n转移矩阵:\n", transition_matrix(df["stage"]).round(2).to_string())
    else:
        import json
        print(json.dumps(run_today(), ensure_ascii=False, indent=2))
