"""Public strategy templates for Strategy Lab.

Only the three strategies selected for public release are exposed here:

* ``rsi_reversal``   — 14-day RSI mean reversion
* ``low_volatility`` — 60-day low-volatility cross-section
* ``momentum_12_1``  — classic 12-1 month cross-sectional momentum

The full private strategy library is not part of this release. See
``DECLASSIFICATION.md`` for the release scope. The execution contract is
unchanged: every template is a self-contained module source string exposing
``build_targets(panel, params) -> dict[Timestamp, dict[code, weight]]``.
"""
from __future__ import annotations

COMMON = '''import pandas as pd

def build_targets(panel, params):
    top_n = int(params.get("top_n", 20))
    rebalance = int(params.get("rebalance_sessions", 20))
    data = panel.copy().sort_values(["code", "date"])
{factor}
    dates = sorted(data["date"].dropna().unique())
    targets = {{}}
    for day in dates[::max(1, rebalance)]:
        frame = data[data["date"] == day].dropna(subset=["score"])
        picks = frame.sort_values("score", ascending=False).head(top_n)["code"].tolist()
        if picks:
            targets[pd.Timestamp(day)] = {{code: 1.0 / len(picks) for code in picks}}
    return targets
'''


def _template(title: str, description: str, factor: str, *, top_n: int, rebalance_sessions: int) -> dict:
    return {
        "title": title,
        "description": description,
        "source": COMMON.format(factor=factor),
        "parameters": {"top_n": top_n, "rebalance_sessions": rebalance_sessions},
    }


# Public release scope: exactly three strategies.
TEMPLATES = {
    "momentum_12_1": _template(
        "12-1月动量",
        "跳过最近一个月的经典横截面动量。",
        '    g = data.groupby("code")["raw_close"]\n    data["score"] = g.shift(21) / g.shift(252) - 1.0',
        top_n=15,
        rebalance_sessions=63,
    ),
    "low_volatility": _template(
        "低波动",
        "选择60日波动率较低的标的。",
        '    ret = data.groupby("code")["raw_close"].pct_change()\n    data["score"] = -ret.groupby(data["code"]).transform(lambda x: x.rolling(60, min_periods=30).std())',
        top_n=20,
        rebalance_sessions=20,
    ),
    "rsi_reversal": _template(
        "RSI反转",
        "选择14日RSI较低的超卖标的。",
        '    ret = data.groupby("code")["raw_close"].pct_change()\n    up = ret.clip(lower=0).groupby(data["code"]).transform(lambda x: x.rolling(14, min_periods=10).mean())\n    down = (-ret.clip(upper=0)).groupby(data["code"]).transform(lambda x: x.rolling(14, min_periods=10).mean())\n    data["score"] = -(100 - 100 / (1 + up / down))',
        top_n=20,
        rebalance_sessions=10,
    ),
}


def template_catalog() -> list[dict]:
    return [
        {
            "id": key,
            "title": value["title"],
            "description": value["description"],
            "parameters": value["parameters"],
        }
        for key, value in TEMPLATES.items()
    ]
