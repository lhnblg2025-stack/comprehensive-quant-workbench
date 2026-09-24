"""V12.3 社媒情绪接入测试: 微博/B站/新闻/小红书适配器 + 情绪综合 + battle_map 集成。

用 monkeypatch 注入合成数据，覆盖:
  1. sentiment_score 词典分
  2. fetch_news 对 akshare 结果的聚合（多空计数 + 情绪分）
  3. fetch_xiaohongshu 无 cookie 时诚实降级(ok=False+note)，有 cookie 时才尝试
  4. market_sentiment_composite 多源加权 + 覆盖度
  5. _render_social_sentiment 渲染分支
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import social_sentiment as ss  # noqa: E402


# ── 1 词典分 ──
def test_sentiment_score():
    assert ss.sentiment_score("涨停 突破 主升 翻倍") > 0
    assert ss.sentiment_score("跌停 崩盘 套牢 割肉") < 0
    assert ss.sentiment_score("") == 0.0
    assert ss.sentiment_score("完全中性的句子没有倾向词") == 0.0
    # 复审A1/A2修复: 否定/中性短语不再误判, 单词不再打到±1.0极值
    assert ss.sentiment_score("利好出尽") < 0        # 实偏空, 不再+1.0
    assert ss.sentiment_score("风险控制") == 0.0     # 中性, 不再-1.0
    assert ss.sentiment_score("并非利好") < 0        # 否定
    assert ss.sentiment_score("不是利空") > 0        # 双重否定→偏多
    assert -0.95 < ss.sentiment_score("跌停") < 0    # 不极值
    assert 0 < ss.sentiment_score("涨停") < 0.95     # 不极值


# ── 1b 微博扩散式情绪 _weibo_diffusion ──
def test_weibo_diffusion():
    df = pd.DataFrame({"name": ["a"] * 10, "rate": [1.0] * 7 + [-1.0] * 3})
    d = ss._weibo_diffusion(df)
    assert d["bullish"] == 7 and d["bearish"] == 3
    assert d["sentiment"] == pytest.approx((7 - 3) / 10, abs=1e-6)  # 0.4 扩散偏多
    assert d["ratio"] == pytest.approx(7 / 3, abs=1e-3)
    # rate 缺失 → 回退空(中性)
    d2 = ss._weibo_diffusion(pd.DataFrame({"name": ["a"]}))
    assert d2["sentiment"] == 0.0
    d3 = ss._weibo_diffusion(pd.DataFrame())
    assert d3["sentiment"] == 0.0


# ── 2 fetch_news 聚合(monkeypatch akshare) ──
def test_fetch_news_aggregates(monkeypatch):
    import types

    class _FakeAk:
        def stock_news_em(self, symbol=""):
            return pd.DataFrame({"关键词": ["x"], "新闻标题": ["宏观利好 突破新高 大涨"]})

        def news_economic_baidu(self):
            return pd.DataFrame({"日期": ["2026-08-15"], "地区": ["中国"], "事件": ["央行降准 利好"]})

    fake = types.ModuleType("akshare")
    fake.stock_news_em = _FakeAk().stock_news_em
    fake.news_economic_baidu = _FakeAk().news_economic_baidu
    monkeypatch.setitem(sys.modules, "akshare", fake)
    r = ss.fetch_news()
    assert r["ok"] is True
    assert r["n"] >= 2
    assert r["bull"] >= 1      # 利好 关键词命中
    assert r["bull_bear_ratio"] >= 1
    assert r["sentiment"] > 0


# ── 3 fetch_xiaohongshu 诚实降级 ──
def test_fetch_xiaohongshu_no_cookie_degrades():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ss, "PLATFORM_COOKIES", {"xiaohongshu": "", "weibo": "", "bilibili": ""})
    r = ss.fetch_xiaohongshu()
    assert r["ok"] is False
    assert "cookie" in r.get("error", "").lower() or "签名" in r.get("error", "")
    monkeypatch.undo()


# ── 4 market_sentiment_composite 加权+覆盖 ──
def test_market_sentiment_composite(monkeypatch):
    live = [
        {"platform": "weibo", "ok": True, "value": '{"ok":true,"sentiment":0.1,"n":50}'},
        {"platform": "bilibili", "ok": True, "value": '{"ok":true,"sentiment":-0.05,"n":10}'},
        {"platform": "news", "ok": True, "value": '{"ok":true,"sentiment":0.0,"n":99}'},
        {"platform": "xiaohongshu", "ok": False, "value": '{"ok":false,"error":"需cookie"}',
         "error": "需cookie"},
    ]
    c = ss.market_sentiment_composite(live)
    # B1: 只对文本情绪源(weibo/bilibili/news/xiaohongshu)平均, 不含guba/baidu
    assert c["market_sentiment"] == pytest.approx((0.1 - 0.05 + 0.0) / 3, abs=1e-4)
    assert c["available"] == 3
    assert c["coverage"] == pytest.approx(3 / 4, abs=0.1)
    assert c["per_source"]["xiaohongshu"]["ok"] is False  # 失败源如实标记


# ── 5 render 分支 ──
def test_render_social_sentiment():
    from quant_system.analysis_core import battle_map as bm
    assert "偏多" in bm._render_social_sentiment({"value": 0.12, "coverage": 0.67})
    assert "覆盖不足" in bm._render_social_sentiment({"value": None, "coverage": 0.33})
    assert "偏空" in bm._render_social_sentiment({"value": -0.08, "coverage": 1.0})
