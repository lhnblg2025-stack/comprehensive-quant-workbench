"""Built-in low-frequency strategy templates for Strategy Lab."""
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


TEMPLATES = {
    "momentum_20": _template("20日动量", "按20日收益选强势标的。", '    data["score"] = data.groupby("code")["raw_close"].pct_change(20)', top_n=20, rebalance_sessions=20),
    "momentum_60": _template("60日动量", "按60日收益进行中期趋势排序。", '    data["score"] = data.groupby("code")["raw_close"].pct_change(60)', top_n=15, rebalance_sessions=30),
    "momentum_12_1": _template("12-1月动量", "跳过最近一个月的经典横截面动量。", '    g = data.groupby("code")["raw_close"]\n    data["score"] = g.shift(21) / g.shift(252) - 1.0', top_n=15, rebalance_sessions=63),
    "short_reversal": _template("短期反转", "买入最近5日跌幅靠前的标的。", '    data["score"] = -data.groupby("code")["raw_close"].pct_change(5)', top_n=20, rebalance_sessions=5),
    "low_volatility": _template("低波动", "选择60日波动率较低的标的。", '    ret = data.groupby("code")["raw_close"].pct_change()\n    data["score"] = -ret.groupby(data["code"]).transform(lambda x: x.rolling(60, min_periods=30).std())', top_n=20, rebalance_sessions=20),
    "trend_ma": _template("均线趋势", "价格相对20/60日均线的趋势强度。", '    g = data.groupby("code")["raw_close"]\n    ma20 = g.transform(lambda x: x.rolling(20, min_periods=15).mean())\n    ma60 = g.transform(lambda x: x.rolling(60, min_periods=40).mean())\n    data["score"] = data["raw_close"] / ma20 - 1.0 + ma20 / ma60 - 1.0', top_n=20, rebalance_sessions=20),
    "breakout_120": _template("120日突破", "选择接近或突破120日高点的标的。", '    high120 = data.groupby("code")["raw_high"].transform(lambda x: x.rolling(120, min_periods=60).max())\n    data["score"] = data["raw_close"] / high120', top_n=15, rebalance_sessions=20),
    "rsi_reversal": _template("RSI反转", "选择14日RSI较低的超卖标的。", '    ret = data.groupby("code")["raw_close"].pct_change()\n    up = ret.clip(lower=0).groupby(data["code"]).transform(lambda x: x.rolling(14, min_periods=10).mean())\n    down = (-ret.clip(upper=0)).groupby(data["code"]).transform(lambda x: x.rolling(14, min_periods=10).mean())\n    data["score"] = -(100 - 100 / (1 + up / down))', top_n=20, rebalance_sessions=10),
    "volume_breakout": _template("量价突破", "20日动量乘以成交量放大倍数。", '    g = data.groupby("code")\n    mom = g["raw_close"].pct_change(20)\n    vol_ma = g["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())\n    data["score"] = mom * data["volume"] / vol_ma', top_n=15, rebalance_sessions=20),
    "liquidity": _template("流动性", "按20日平均成交额选择高流动性标的。", '    data["score"] = data.groupby("code")["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())', top_n=30, rebalance_sessions=20),
    "quality_trend": _template("趋势质量", "60日收益除以日收益波动率。", '    ret = data.groupby("code")["raw_close"].pct_change()\n    mom = data.groupby("code")["raw_close"].pct_change(60)\n    vol = ret.groupby(data["code"]).transform(lambda x: x.rolling(60, min_periods=30).std())\n    data["score"] = mom / vol.replace(0, float("nan"))', top_n=15, rebalance_sessions=30),
    "multi_factor": _template("动量+低波+流动性", "三个截面排名等权组合。", '    g = data.groupby("code")\n    ret = g["raw_close"].pct_change()\n    data["mom"] = g["raw_close"].pct_change(60)\n    data["vol"] = -ret.groupby(data["code"]).transform(lambda x: x.rolling(60, min_periods=30).std())\n    data["liq"] = g["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())\n    data["score"] = data.groupby("date")[["mom", "vol", "liq"]].rank(pct=True).mean(axis=1)', top_n=20, rebalance_sessions=20),
    "dual_ma_filter": _template("双均线过滤", "仅保留20日均线高于60日均线的标的。", '    g = data.groupby("code")["raw_close"]\n    ma20 = g.transform(lambda x: x.rolling(20, min_periods=15).mean())\n    ma60 = g.transform(lambda x: x.rolling(60, min_periods=40).mean())\n    data["score"] = (ma20 / ma60 - 1.0).where(ma20 > ma60)', top_n=20, rebalance_sessions=20),
    "bollinger_reversion": _template("布林带均值回归", "选择偏离20日均线最深的标的。", '    g = data.groupby("code")["raw_close"]\n    ma = g.transform(lambda x: x.rolling(20, min_periods=15).mean())\n    std = g.transform(lambda x: x.rolling(20, min_periods=15).std())\n    data["score"] = -(data["raw_close"] - ma) / std.replace(0, float("nan"))', top_n=20, rebalance_sessions=5),
    "risk_adjusted_momentum": _template("风险调整动量", "20/60日动量除以60日波动率。", '    g = data.groupby("code")["raw_close"]\n    ret = g.pct_change()\n    mom = 0.5 * g.pct_change(20) + 0.5 * g.pct_change(60)\n    vol = ret.groupby(data["code"]).transform(lambda x: x.rolling(60, min_periods=30).std())\n    data["score"] = mom / vol.replace(0, float("nan"))', top_n=15, rebalance_sessions=30),
}


def template_catalog() -> list[dict]:
    return [{"id": key, "title": value["title"], "description": value["description"], "parameters": value["parameters"]} for key, value in TEMPLATES.items()]
