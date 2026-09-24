"""stock_lens 个股全景透视 — 单元+契约测试。

用 monkeypatch 注入合成数据验证评分逻辑与数据契约；
末尾一条真实数据集成用例验证端到端不抛异常。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import stock_lens  # noqa: E402


def _fake_zt() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-08-10", "2026-08-12", "2026-08-13", "2026-08-14"] * 2
                               + ["2026-08-14", "2026-08-14"]),
        "code": ["002172"] * 8 + ["600519", "000858"],
        "board_count": [1, 2, 2, 3] * 2 + [1, 1],
        "is_zt": [True] * 10,
    })


def _fake_lhb() -> pd.DataFrame:
    return pd.DataFrame({
        "代码": ["002172"] * 3,
        "上榜日": pd.to_datetime(["2026-08-12", "2026-08-13", "2026-08-14"]),
        "龙虎榜净买额": [1e8, 2e8, -5e7],
        "解读": ["游资买入", "游资买入", "1家机构卖出"],
        "上榜后1日": [3.2, -1.1, 2.5],
    })


@pytest.fixture()
def fake_market(monkeypatch, tmp_path):
    """把 stock_lens 的市场数据源替换为合成数据。"""
    monkeypatch.setattr(stock_lens, "MARKET_DIR", tmp_path / "market")
    monkeypatch.setattr(stock_lens, "CLASS_DIR", tmp_path / "classification")
    monkeypatch.setattr(stock_lens, "VALUATION_DIR", tmp_path / "valuation")
    monkeypatch.setattr(stock_lens, "FINANCIAL_DIR", tmp_path / "financial")
    (tmp_path / "market").mkdir(parents=True)
    (tmp_path / "classification").mkdir(parents=True)
    # 涨停池
    _fake_zt().to_parquet(tmp_path / "market" / "zt_pool_history.parquet", index=False)
    # 龙虎榜季度段: 文件命名 lhb_20260101_20260630.parquet
    _fake_lhb().to_parquet(tmp_path / "market" / "lhb_20260101_20261231.parquet", index=False)
    # 概念: 002172 属"白酒"(与600519/000858同概念, 概念内3只涨停→hot)
    pd.DataFrame({"concept": ["白酒"] * 3 + ["消费"],
                  "code": ["002172", "600519", "000858", "002172"]}).to_parquet(
        tmp_path / "classification" / "concept_member.parquet", index=False)
    # 全市场资金流
    pd.DataFrame({"日期": pd.to_datetime(["2026-08-14"]),
                  "主力净流入-净额": [5e9], "主力净流入-净占比": [1.2]}).to_parquet(
        tmp_path / "market" / "stock_market_fund_flow.parquet", index=False)
    # 融资明细(按日文件)
    md = tmp_path / "market" / "margin_detail_sh"
    md.mkdir(exist_ok=True)
    pd.DataFrame({"信用交易日期": pd.to_datetime(["2026-07-01", "2026-08-14"]),
                  "标的证券代码": ["002172"] * 2, "标的证券简称": ["X"] * 2,
                  "融资余额": [1e9, 1.2e9], "融资买入额": [1e8] * 2,
                  "融资偿还额": [8e7] * 2}).to_parquet(md / "20260814.parquet", index=False)
    # 股东户数
    pd.DataFrame({"代码": ["002172"], "股东户数-本次": [50000], "股东户数-上次": [60000],
                  "股东户数-增减比例": [-16.7], "股东户数统计截止日-本次": ["2026-06-30"]}).to_parquet(
        tmp_path / "market" / "gdhs_all.parquet", index=False)
    # 基金持仓
    pd.DataFrame({"股票代码": ["002172"], "持有基金家数": [15], "持股变化": ["增仓"],
                  "持股变动比例": [12.5]}).to_parquet(
        tmp_path / "market" / "fund_portfolio_hold.parquet", index=False)
    # 估值/财务
    (tmp_path / "valuation").mkdir(exist_ok=True)
    (tmp_path / "financial").mkdir(exist_ok=True)
    pd.DataFrame({"date": pd.to_datetime(["2026-08-14"]), "peTTM": [12.0], "pbMRQ": [1.2]}).to_parquet(
        tmp_path / "valuation" / "002172.parquet", index=False)
    pd.DataFrame({"加权净资产收益率(%)": [8.0, 8.5, 9.0, 9.5]}).to_parquet(
        tmp_path / "financial" / "002172.parquet", index=False)
    return tmp_path


def test_synthesize_quadrants():
    """四象限分类: 双强/短线强/中线强/双弱。"""
    assert stock_lens._synthesize({"score": 80}, {"score": 75})["verdict"] == "短中线共振(双强)"
    assert stock_lens._synthesize({"score": 80}, {"score": 30})["verdict"] == "短线驱动(情绪)"
    assert stock_lens._synthesize({"score": 30}, {"score": 80})["verdict"] == "中线驱动(价值)"
    assert stock_lens._synthesize({"score": 30}, {"score": 30})["verdict"] == "双弱(观望)"


def test_short_angle_scores(fake_market):
    """合成涨停+龙虎榜数据 → 短线分高且各要素计分正确。"""
    r = stock_lens._short_angle("002172", pd.Timestamp("2026-08-14"), name_hint="X")
    assert 0 <= r["score"] <= 100
    parts = {p["key"]: p for p in r["parts"]}
    assert parts["涨停基因"]["score"] >= 20  # 3次涨停+连板
    assert parts["龙虎榜"]["data"]["times_60d"] == 3
    assert parts["龙虎榜"]["data"]["net_buy_days"] == 2
    assert parts["龙虎榜"]["data"]["fwd1_winrate"] == pytest.approx(2 / 3, abs=0.01)
    assert parts["概念共振"]["data"]["hot"] is True  # 概念内2只涨停? ≥3 阈值需概念成员
    assert "data_sources" in r


def test_mid_angle_scores(fake_market):
    """合成融资加仓+筹码集中+基金增仓+低估 → 中线分高。"""
    r = stock_lens._mid_angle("002172", pd.Timestamp("2026-08-14"))
    assert 0 <= r["score"] <= 100
    parts = {p["key"]: p for p in r["parts"]}
    assert parts["融资盘"]["data"]["chg_40d"] == pytest.approx(20.0, abs=0.1)  # 1e9→1.2e9
    assert parts["筹码(股东户数)"]["score"] > 0
    assert parts["基金持仓"]["data"]["fund_count"] == 15
    assert parts["估值"]["data"]["pe"] == 12.0
    assert parts["估值"]["score"] >= 15  # PE12/PB1.2 → 低估


def test_analyze_contract(fake_market):
    """analyze 返回完整契约结构。"""
    r = stock_lens.analyze("002172", "2026-08-14", name_hint="X")
    assert r["code"] == "002172"
    assert set(r.keys()) >= {"synth", "short", "mid", "as_of"}
    assert set(r["synth"].keys()) >= {"verdict", "tone", "advice", "short_score", "mid_score"}
    assert 0 <= r["synth"]["short_score"] <= 100
    for sec in ("short", "mid"):
        assert isinstance(r[sec]["parts"], list) and r[sec]["parts"]
        assert isinstance(r[sec]["reasons"], list)
        assert isinstance(r[sec]["data_sources"], dict)


def test_analyze_unknown_code_degrades():
    """未知代码/数据缺失 → 不抛异常, 分数为0。"""
    r = stock_lens.analyze("999999", "2026-08-14")
    assert r["synth"]["short_score"] == 0.0
    assert r["synth"]["mid_score"] == 0.0


def test_margin_series_dedup(fake_market):
    """融资序列按日期去重排序。"""
    s = stock_lens._margin_series("002172")
    assert not s.empty
    assert s["信用交易日期"].is_monotonic_increasing
    assert s["信用交易日期"].nunique() == len(s)


# ── V12.3 二期: 三视角增强 ──

def test_synthesize3_quadrants():
    """三维合成: 三强共振/双强组合/单强/全弱。"""
    assert stock_lens._synthesize3({"score": 80}, {"score": 75}, {"score": 70})["verdict"] == "三线共振(主升)"
    assert stock_lens._synthesize3({"score": 80}, {"score": 75}, {"score": 20})["verdict"] == "情绪+波段共振"
    assert stock_lens._synthesize3({"score": 20}, {"score": 75}, {"score": 70})["verdict"] == "价值波段"
    assert stock_lens._synthesize3({"score": 80}, {"score": 20}, {"score": 70})["verdict"] == "题材价值"
    assert stock_lens._synthesize3({"score": 80}, {"score": 20}, {"score": 20})["verdict"] == "短线驱动(纯情绪)"
    assert stock_lens._synthesize3({"score": 20}, {"score": 75}, {"score": 20})["verdict"] == "中线驱动(筹码)"
    assert stock_lens._synthesize3({"score": 20}, {"score": 20}, {"score": 75})["verdict"] == "长线价值(左侧)"
    assert stock_lens._synthesize3({"score": 20}, {"score": 20}, {"score": 20})["verdict"] == "三线皆弱(回避)"
    r = stock_lens._synthesize3({"score": 80}, {"score": 75}, {"score": 70})
    assert r["strong_axes"] == ["短线", "中线", "长线"]


def test_announcement_signals_classify(fake_market):
    """公告利好/利空关键词分类: 业绩预增→利好, 减持→利空。"""
    monkeypatch = pytest.MonkeyPatch()
    fake = [
        {"date": "2026-08-14", "title": "002172:2026年半年度业绩预增公告", "type": "业绩预告"},
        {"date": "2026-08-13", "title": "002172:股东减持股份计划公告", "type": "减持"},
        {"date": "2026-08-12", "title": "002172:关于回购公司股份的进展公告", "type": "回购"},
    ]
    monkeypatch.setattr(stock_lens, "_load_cninfo", lambda code, end, days=30: fake)
    sig = stock_lens._announcement_signals("002172", pd.Timestamp("2026-08-14"))
    assert sig["ok"] is True
    assert sig["records"] == 3
    assert len(sig["bullish"]) == 2   # 业绩预增 + 回购
    assert len(sig["bearish"]) == 1   # 减持
    assert sig["score"] <= 0          # 2利好(+8) - 1利空(-8) = 0
    monkeypatch.undo()


def test_long_angle_contract(fake_market):
    """长线五要素结构与数据源标注。"""
    r = stock_lens._long_angle("002172")
    assert 0 <= r["score"] <= 100
    keys = {p["key"] for p in r["parts"]}
    assert keys >= {"宏观驱动", "深度价值", "质量成长", "市场水位", "事件底仓"}
    assert isinstance(r["reasons"], list)
    assert "data_sources" in r


def test_analyze_three_view(fake_market):
    """analyze 返回三视角 + synth 含 long_score。"""
    r = stock_lens.analyze("002172", "2026-08-14", name_hint="X")
    assert set(r.keys()) >= {"short", "mid", "long", "synth"}
    assert "long_score" in r["synth"]
    assert 0 <= r["synth"]["long_score"] <= 100
    assert isinstance(r["long"]["parts"], list)


def test_hot_rank_respects_end_no_lookahead(tmp_path, monkeypatch):
    """P2-9: _hot_rank 按 end 选最近一日 ≤ end 的热度文件，历史 date 不用当下人气。

    造 3 个带日期的人气文件，验证:
      - 无 end → 用最新(日期最大)文档;
      - end=中间日期 → 不读取更晚的文件(无前视)。
    """
    import pandas as pd
    # 造 3 个带日期的人气文件(日期递增, code 列带 SH 前缀), rank 随日期不同便于断言命中值
    dates = ["20260810", "20260812", "20260814"]
    for d, rank in zip(dates, [10, 77, 78]):
        df = pd.DataFrame({"code": ["SH600519", "SH000001"],
                           "rank": [rank, 1], "rank_change": [-1, +1], "pct_chg": [1.0, 2.0]})
        (tmp_path / f"hot_rank_{d}.parquet").write_bytes(df.to_parquet())
    monkeypatch.setattr(stock_lens, "HOT_DIR", tmp_path)
    # 无 end → 最新 0814(rank=78)
    r_now = stock_lens._hot_rank("600519")
    assert r_now.get("ok") and r_now["rank"] == 78
    # end=2026-08-12 → 命中 0812(rank=77), 不读 0814(rank=78)
    r_hist = stock_lens._hot_rank("600519", pd.Timestamp("2026-08-12"))
    assert r_hist.get("ok") and r_hist["rank"] == 77
    # end=2026-08-09 → 无 ≤ 该日文件(最老 0810) → ok=False 不误读
    r_none = stock_lens._hot_rank("600519", pd.Timestamp("2026-08-09"))
    assert not r_none.get("ok")
