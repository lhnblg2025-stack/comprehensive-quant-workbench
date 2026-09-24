"""
social_sentiment — 社交情绪融合层（V11）

设计原则:
  - 适配器架构: 每平台一个 fetcher，独立 try/except，失败只记录不阻塞管线
  - 词典情感打标（免费快）: 看多/看空词库 → 多空比；LLM 精读留待语义层
  - 滚动存储: social_sentiment.parquet（date+platform+metric 长表），重复执行去重

V1 数据源分层:
  T1 确定性（本地/东财稳定接口）: 涨停池行业热度榜、千股千评(主力成本偏离)
  T2 东财股吧: 人气榜/热帖（接口可能风控，重试+降级）
  T3 微博/B站: 适配器占位，支持 config 注入 COOKIE 后启用

用法:
  python3 -m quant_system.analysis_core.social_sentiment --today
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.cpu_throttle import CpuThrottle  # noqa: E402

CST = timezone(timedelta(hours=8))
throttle = CpuThrottle()

SOCIAL_PARQUET = MARKET_DIR / "social_sentiment.parquet"
SOCIAL_DIR = ROOT / "data_warehouse" / "social"
ZT_EM_DAILY = MARKET_DIR / "zt_pool_em_daily.parquet"

# 平台 cookie 从环境/.env.secrets注入，禁止写入源码。
try:
    from secret_loader import get_secret as _get_secret
except Exception:
    _get_secret = lambda name, default="": default
PLATFORM_COOKIES = {
    "weibo": _get_secret("WEIBO_COOKIE", ""),
    "bilibili": _get_secret("BILIBILI_COOKIE", ""),
    "xiaohongshu": _get_secret("XIAOHONGSHU_COOKIE", ""),
}

# ── 金融情感词典（可扩充）──────────────────────────────
BULL_WORDS = ["涨停", "大涨", "暴涨", "主升", "突破", "利好", "龙头", "连板", "反包",
              "加速", "起飞", "打板", "超预期", "放量突破", "新高", "吃肉", "翻倍",
              "降准", "降息", "宽松", "放水", "利好政策", "提振", "稳增长", "回暖",
              "增长超预期", "盈利超预期", "上调评级", "增持", "回购", "扭亏", "订单",
              "中标大额", "扩产", "受益", "拐点", "景气"]
BEAR_WORDS = ["跌停", "大跌", "暴跌", "崩盘", "套牢", "利空", "退潮", "炸板", "割肉",
              "出货", "接盘", "跳水", "清仓", "腰斩", "核按钮", "天地板", "大面",
              "看空", "做空", "减持", "爆雷", "风险", "谨防", "警惕", "回调",
              "加息", "收紧", "缩表", "衰退", "通缩", "下滑", "亏损", "下调评级",
              "诉讼", "处罚", "违规", "退市", "质押爆仓", "债务违约", "裁员", "不可持续"]


# 结构式短语/否定（复审 A1 修复）：这些短语语义与词典词相反或中性化，命中则翻转/降权。
# 顺序重要：先查"翻转/中性化"整句模式，再查否定连词，再逐词计分。
_NEGATE_FLIP = [
    "利好出尽", "利空出尽", "风险释放", "利空落地", "靴子落地",  # 出尽→偏空/中性
]
_NEUTRALIZE = [
    "风险控制", "风险管理", "风险偏好中性", "不足为惧", "影响有限", "基本持平",
]
# 否定前缀（紧随的情感词取反）：并非/不是/未能/不会/不再 利好/利空/大涨/大跌...
_NEG_PREFIX = ("并非", "不是", "未", "没有", "不会", "不再", "不构成", "难言")

def sentiment_score(text: str) -> float:
    """词典情感分，归一化 [-1,1]。V12.3 复审(A1/A2)修复：
      - 识别"利好出尽/利空落地"等翻转短语（词面对利多实偏空）
      - 识别"风险控制/影响有限"等中性短语（不误判）
      - 识别否定前缀（"并非利好"→负；"不是利空"→正）
      - 按「命中情感词数/文本长度」平滑归一，避免单个情感词直接打到 ±1.0 极值
    docstring 口径: 分子=(看多-看空)，分母=命中词数 + 1（软饱和），与实现一致。"""
    if not text:
        return 0.0
    # 1) 中性短语：直接 0（"风险控制/风险管理/影响有限"非利空）
    if any(p in text for p in _NEUTRALIZE):
        return 0.0
    flip = -1.0 if any(p in text for p in _NEGATE_FLIP) else 1.0
    bull = bear = 0.0
    try:
        toks = [t for s in text.replace("，", " ").replace("、", " ").split() for t in [s]] if " " in text else [text]
    except Exception:  # noqa: BLE001
        toks = [text]
    # 对每个情感词：若恰好被否定前缀修饰则计反向，否则正向
    for w in BULL_WORDS:
        if w in text:
            if any(p + w in text for p in _NEG_PREFIX):
                bear += 1.0
            else:
                bull += 1.0
    for w in BEAR_WORDS:
        if w in text:
            if any(p + w in text for p in _NEG_PREFIX):
                bull += 1.0
            else:
                bear += 1.0
    # 未抽到词则截断空 ran。据此计分：
    if bull + bear == 0:
        return 0.0
    # 分子=flip*(bull-bear)；分母=命中词数+1（软饱和，不因单词直接±1）
    raw = flip * (bull - bear)
    denom = (bull + bear) + 1.0
    score = raw / denom
    # 再轻微压到 [-0.9,0.9] 防极值
    return round(float(max(-0.9, min(0.9, score))), 4)


# ────────────────────────────────────────────────────────────
# T1 确定性指标（本地/稳定）
# ────────────────────────────────────────────────────────────
def theme_heat_from_zt_pool() -> dict:
    """从 EM 涨停池算行业涨停家数 → 题材热度榜（当日）。"""
    if not ZT_EM_DAILY.exists():
        return {"ok": False, "error": "zt_pool_em_daily 不存在"}
    df = pd.read_parquet(ZT_EM_DAILY)
    last = df["date"].max()
    day = df[(df["date"] == last) & df["is_zt"]]
    if day.empty:
        return {"ok": False, "error": f"{last} 无涨停数据"}
    heat = day["industry"].value_counts().head(10)
    # 审计 2026-08-16：相对今天的天数滞后标记，避免陈旧涨停池被当"当日"可用
    lag_days = (datetime.now(CST).date() - last.date()).days if hasattr(last, "date") else None
    stale = lag_days is not None and lag_days > 3
    return {
        "ok": True, "date": str(last.date()), "source": "zt_pool_industry",
        "industries": [{"industry": k, "zt_cnt": int(v)} for k, v in heat.items()],
        "top": str(heat.index[0]) if len(heat) else "",
        "lag_days": lag_days, "stale": stale,
    }


def qianqian_comment() -> dict:
    """千股千评: 全市场个股主力成本/评价（东财稳定接口）。"""
    try:
        import akshare as ak
        df = ak.stock_comment_em()
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}
    if df is None or df.empty:
        return {"ok": False, "error": "空数据"}
    # 综合评分列存在时统计分布（列名随版本变化，防御）
    score_cols = [c for c in df.columns if "评分" in c or "综合" in c]
    out = {"ok": True, "source": "stock_comment_em", "rows": len(df)}
    if score_cols:
        s = df[score_cols[0]].dropna().astype(float)
        out["score_mean"] = round(float(s.mean()), 2)
        out["score_median"] = round(float(s.median()), 2)
    return out


# ────────────────────────────────────────────────────────────
# T2 东财股吧（腾讯云 hot_rank 快照优先 + 直连兜底）
# ────────────────────────────────────────────────────────────
def fetch_guba() -> dict:
    """东财股吧人气榜：优先读腾讯云快照（data_warehouse/hot_rank/），快照陈旧/缺失时直连。

    2026-08-11 改造: 东财 emappdata 接口对本机境外 IP 断连，akshare 在腾讯云上
    也因请求构造问题失败；但原生 requests + 完整 UA 在腾讯云(国内 IP)可用，
    已部署 C:\\quant\\crawler_hot_rank.py 每日 18:35 抓取 top100 → hot_rank_*.parquet。
    """
    import glob
    import akshare as ak

    HOT_DIR = ROOT / "data_warehouse" / "hot_rank"

    def _parse_parquet(path: Path):
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return None
        # code 列形如 SH600519 / SZ000001，无名称 → 用代码本身作为名称代理
        codes = df["code"].astype(str).tolist()
        ranks = df.get("rank")
        return {
            "ok": True, "source": "guba_hot_rank_snapshot", "n": len(df),
            "top": codes[:10],
            "top_rank": [int(r) if pd.notna(r) else None for r in ranks.head(10).tolist()] if ranks is not None else None,
            "as_of": str(Path(path).stem).replace("hot_rank_", ""),
            "sentiment": round(float(sentiment_score(" ".join(codes))), 3),
            "note": "东财人气榜 top100（腾讯云快照，代码级热度代理）",
        }

    # 1) 腾讯云快照优先（新鲜度 >2 日则降级直连）
    try:
        files = sorted(glob.glob(str(HOT_DIR / "hot_rank_*.parquet"))) if HOT_DIR.exists() else []
        if files:
            latest = Path(files[-1])
            as_of = latest.stem.replace("hot_rank_", "")
            try:
                snap_dt = datetime.strptime(as_of, "%Y%m%d").replace(tzinfo=CST)
                snap_stale = (datetime.now(CST) - snap_dt).days > 2
            except ValueError:
                snap_stale = True
            if not snap_stale:
                res = _parse_parquet(latest)
                if res:
                    return res
    except Exception as e:
        logging.getLogger(__name__).error(f"[social_sentiment] 操作失败: {e}", exc_info=True)

    # 2) 直连兜底（akshare 在腾讯云/本机可能失败，如实报错）
    for i in range(3):
        try:
            df = ak.stock_hot_rank_em()
            if df is not None and not df.empty:
                top = df.head(10)
                texts = " ".join(top["名称"].astype(str).tolist())
                return {
                    "ok": True, "source": "guba_hot_rank",
                    "n": len(df),
                    "top": top["名称"].head(10).tolist(),
                    "sentiment": round(float(sentiment_score(texts)), 3),
                    "note": "人气榜(名称级) 热度代理（直连）",
                }
        except Exception:
            throttle.sleep(default_ms=2000 * (i + 1))
    return {"ok": False, "error": "股吧人气榜快照缺失且接口不可用"}


# ────────────────────────────────────────────────────────────
# T3 微博/B站（Vultr 微博舆情快照优先 + 直连兜底）
# ────────────────────────────────────────────────────────────
def fetch_weibo(keyword: str = "") -> dict:
    """微博舆情：优先读 Vultr 快照（data_warehouse/social/weibo_*.parquet）。

    2026-08-11 改造: akshare 新版移除 stock_weibo_em（热搜），但
    stock_js_weibo_report(time_period="CNHOUR24")（新浪微博个股舆情涨跌率）在 Vultr
    境外 IP 可用，已部署 /root/quant-env/vultr_social.py 每日 18:40 抓取。
    """
    import glob
    import akshare as ak

    SOCIAL_DIR_LOCAL = ROOT / "data_warehouse" / "social"

    # 1) Vultr 快照优先
    try:
        files = sorted(glob.glob(str(SOCIAL_DIR_LOCAL / "weibo_*.parquet"))) if SOCIAL_DIR_LOCAL.exists() else []
        if files:
            latest = Path(files[-1])
            as_of = latest.stem.replace("weibo_", "")
            try:
                snap_dt = datetime.strptime(as_of, "%Y%m%d").replace(tzinfo=CST)
                snap_stale = (datetime.now(CST) - snap_dt).days > 2
            except ValueError:
                snap_stale = True
            if not snap_stale:
                df = pd.read_parquet(latest)
                if df is not None and not df.empty:
                    names = df["name"].astype(str).tolist()
                    rates = df.get("rate")
                    # V12.3: 加权个股舆情涨跌率 → 扩散式情绪(看多/看空广度)
                    diff = _weibo_diffusion(df)
                    return {
                        "ok": True, "source": "weibo_sentiment_snapshot", "n": len(df),
                        "top": names[:10], "as_of": as_of,
                        "sentiment": round(diff["sentiment"], 3),
                        "diffusion": diff,
                        "top_rates": [round(float(r), 2) if pd.notna(r) else None for r in rates.head(10).tolist()] if rates is not None else None,
                        "note": "微博个股舆情涨跌率（Vultr 快照，CNHOUR24）",
                    }
    except Exception as e:
        logging.getLogger(__name__).error(f"[social_sentiment] 操作失败: {e}", exc_info=True)

    # 2) 直连兜底（Vultr 上已验证可用）
    try:
        df = ak.stock_js_weibo_report(time_period="CNHOUR24")
        if df is not None and not df.empty:
            names = df["name"].astype(str).tolist()
            diff = _weibo_diffusion(df)
            return {
                "ok": True, "source": "weibo_sentiment_direct", "n": len(df),
                "top": names[:10],
                "sentiment": round(diff["sentiment"], 3),
                "diffusion": diff,
                "note": "微博个股舆情涨跌率（直连）",
            }
    except Exception as e:
        return {"ok": False, "error": f"微博舆情快照缺失且直连失败: {str(e)[:100]}"}
    return {"ok": False, "error": "微博舆情快照缺失且直连无数据"}


def _weibo_diffusion(df: pd.DataFrame) -> dict:
    """微博舆情扩散式情绪: 取 rate(舆情涨跌率)列, 正=偏多/负=偏空, 计算广度与均分。

    返回 {sentiment, bullish, bearish, neutral, ratio, mean_rate, plus_minus}。
    rate 缺失时回退为空(由调用方用 0)。
    """
    if df is None or df.empty or "rate" not in df.columns:
        return {"sentiment": 0.0, "bullish": 0, "bearish": 0, "neutral": len(df) if df is not None else 0,
                "ratio": None, "mean_rate": None, "plus_minus": 0}
    rates = pd.to_numeric(df["rate"], errors="coerce")
    pos = int((rates > 0).sum())
    neg = int((rates < 0).sum())
    neut = int((rates == 0).sum())
    denom = pos + neg + neut
    # sentiment = (pos-neg)/total ∈ [-1,1]（广度扩散）
    diffusion = (pos - neg) / max(denom, 1)
    return {"sentiment": round(float(diffusion), 3), "bullish": pos, "bearish": neg,
            "neutral": neut, "ratio": round(pos / max(neg, 1), 3),
            "mean_rate": round(float(rates.mean()), 4) if len(rates) else None,
            "plus_minus": pos - neg}


def fetch_bilibili(keyword: str = "A股") -> dict:
    """B站搜索适配器（旧版接口）。cookie 可选。"""
    import requests
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
         "Referer": "https://www.bilibili.com/"}
    if PLATFORM_COOKIES["bilibili"]:
        h["Cookie"] = PLATFORM_COOKIES["bilibili"]
    # V12.3 情绪接入(用户需求): 旧 search 接口在本机境外 IP 被风控(JSONDecodeError)。
    # 改走公开的「热门视频」接口 api.bilibili.com/x/web-interface/popular（无需登录，
    # 实测本机可用, code=0）。从中筛出与 A股/财经/基金相关视频做 词典情感 + 播放热度。
    try:
        r = requests.get("https://api.bilibili.com/x/web-interface/popular",
                         params={"ps": 50, "pn": 1}, headers=h, timeout=8)
        j = r.json()
        if j.get("code") != 0:
            return {"ok": False, "error": f"B站热门接口 code={j.get('code')} {str(j.get('message'))[:40]}"}
        items = (j.get("data") or {}).get("list") or []
        if not items:
            # 兜底: 老 search 接口
            r2 = requests.get("https://s.search.bilibili.com/cate/search",
                              params={"search_type": "video", "keyword": keyword,
                                      "page": 1, "order": "click"},
                              headers=h, timeout=8)
            res = (r2.json().get("data") or {}).get("result") or []
            if not res:
                return {"ok": False, "error": "B站热门+搜索都为空"}
            top = res[:10]
            texts = " ".join(str(x.get("title", "")) for x in top)
            plays = [int(x.get("play", 0)) for x in top]
            return {"ok": True, "source": "bilibili_search_fallback", "keyword": keyword,
                    "n": len(top), "plays": plays, "total_play": sum(plays),
                    "sentiment": round(sentiment_score(texts), 3)}
        # 筛财经/A股相关
        FIN_KEY = ("A股", "股票", "基金", "财经", "股市", "大盘", "股民", "上证", "炒股",
                   "ETF", "牛市", "熊市", "打板", "量化", "成交", "涨停")
        fin = [x for x in items if any(k in str(x.get("title", "")) for k in FIN_KEY)]
        pick = fin[:10] if fin else items[:10]
        texts = " ".join(str(x.get("title", "")) + " " + str(x.get("desc", "")) for x in pick)
        plays = [int(x.get("stat", {}).get("view", 0)) for x in pick]
        return {"ok": True, "source": "bilibili_popular", "keyword": "财经/热榜",
                "n": len(pick), "plays": plays, "total_play": sum(plays),
                "fin_count": len(fin), "sample_count": len(items),
                "sentiment": round(sentiment_score(texts), 3),
                "heat": round(min(1.0, sum(plays) / max(sum(plays), 1) / 1e6), 4)}
    except Exception as e:
        return {"ok": False, "error": f"B站采集失败: {str(e)[:120]}"}


def fetch_xiaohongshu(keyword: str = "") -> dict:
    """小红书笔记情绪（适配器占位 + 诚实降级）。

    小红书 web 接口需 x-s/x-s-common 签名 + cookie，本机境外 IP 无凭据无法直取。
    提供：
      1) 若在 PLATFORM_COOKIES 注入 cookie（在小红书登录后复制），走笔记搜索接口；
      2) 否则诚实返回 ok=False + note（不做假数据、不假装刷到）。
    用户的"爬小红书"需求受平台风控限制（无 cookie/签名拿不到），本适配器把这种
    限制如实暴露，而非静默给 0。
    """
    if PLATFORM_COOKIES.get("xiaohongshu"):
        try:
            import requests
            h = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
                 "Cookie": PLATFORM_COOKIES["xiaohongshu"], "Origin": "https://www.xiaohongshu.com"}
            r = requests.get(
                "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes",
                params={"keyword": keyword or "A股", "page": 1, "page_size": 20, "sort": "general"},
                headers=h, timeout=8)
            j = r.json()
            notes = ((j.get("data") or {}).get("items") or {}).get("notes") or []
            if not notes:
                return {"ok": False, "error": "小红书搜索为空/需更新cookie"}
            texts = " ".join(str(x.get("display_title", "")) for x in notes[:15])
            return {"ok": True, "source": "xiaohongshu_notes", "keyword": keyword,
                    "n": len(notes), "sentiment": round(sentiment_score(texts), 3),
                    "heat": round(min(1.0, len(notes) / 20.0), 4)}
        except Exception as e:
            return {"ok": False, "error": f"小红书采集失败: {str(e)[:120]}"}
    return {"ok": False, "error": "小红书需登录 cookie(x-s签名); 当前无凭据, 如实返回不可用",
            "note": "在 PLATFORM_COOKIES['xiaohongshu'] 注入 cookie 后启用"}


def _news_summary(titles: list[str], records: list[dict], *, source: str, coverage: str, source_symbol: str | None = None) -> dict:
    """Summarize already filtered news without changing its attribution."""
    bull = sum(1 for t in titles if sentiment_score(t) > 0)
    bear = sum(1 for t in titles if sentiment_score(t) < 0)
    neut = len(titles) - bull - bear
    score = round(float(sum(sentiment_score(t) for t in titles) / len(titles)), 4) if titles else 0.0
    return {
        "ok": bool(titles), "source": source, "coverage": coverage,
        "source_symbol": source_symbol, "n": len(titles), "bull": bull,
        "bear": bear, "neutral": neut,
        "bull_bear_ratio": round(bull / max(bear, 1), 3) if titles else None,
        "sentiment": score, "records": records,
        "top_bullish": [t for t in titles if sentiment_score(t) > 0][:5],
        "top_bearish": [t for t in titles if sentiment_score(t) < 0][:5],
    }


def fetch_news(symbol: str | None = None, limit: int = 30) -> dict:
    """Fetch either market news or strictly code-attributed stock news.

    ``symbol`` is intentionally optional for the existing market sentiment job.
    When supplied, only ``stock_news_em(symbol=...)`` is queried; macro events
    and other stocks are never mixed into that response.
    """
    import akshare as ak

    if symbol is not None:
        code = "".join(ch for ch in str(symbol).strip() if ch.isdigit())
        if len(code) != 6:
            return {"ok": False, "coverage": "stock_symbol", "source_symbol": code or str(symbol),
                    "error": "股票代码必须是6位数字"}
        try:
            df = ak.stock_news_em(symbol=code)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "coverage": "stock_symbol", "source_symbol": code,
                    "error": f"个股新闻接口失败: {str(exc)[:160]}"}
        if df is None or df.empty:
            return {"ok": False, "coverage": "stock_symbol", "source_symbol": code,
                    "error": "该股票暂无可用新闻"}
        title_col = next((c for c in ("新闻标题", "标题", "新闻内容", "摘要") if c in df.columns), None)
        if title_col is None:
            title_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        records = []
        titles = []
        for _, row in df.head(max(1, min(int(limit), 100))).iterrows():
            title = str(row.get(title_col) or "").strip()
            if not title or title.lower() == "nan":
                continue
            titles.append(title)
            records.append({
                "title": title,
                "date": str(next((row.get(c) for c in ("发布时间", "日期", "时间") if c in df.columns and pd.notna(row.get(c))), ""))[:32],
                "source": str(next((row.get(c) for c in ("文章来源", "来源", "新闻来源") if c in df.columns and pd.notna(row.get(c))), ""))[:80],
                "url": str(next((row.get(c) for c in ("新闻链接", "链接", "url") if c in df.columns and pd.notna(row.get(c))), ""))[:500],
            })
        result = _news_summary(titles, records, source="akshare.stock_news_em", coverage="stock_symbol", source_symbol=code)
        if not result["ok"]:
            result["error"] = "该股票暂无可用新闻标题"
        result["fetched"] = [f"stock_news_em({code}:{len(titles)})"]
        return result

    titles: list[str] = []
    fetched = []
    records: list[dict] = []
    # Market sentiment intentionally uses a broad sample and is never a stock claim.
    for sym in ("600519", "000001", "300750", "601318"):
        try:
            df = ak.stock_news_em(symbol=sym)
            if df is not None and not df.empty:
                c = "新闻标题" if "新闻标题" in df.columns else df.columns[1]
                for value in df[c].head(max(1, min(limit, 100))).tolist():
                    titles.append(str(value))
                fetched.append(f"stock_news_em({sym}:{len(df)})")
        except Exception:  # noqa: BLE001 - 单一标的风控不影响整体
            pass
    try:
        df2 = ak.news_economic_baidu()
        if df2 is not None and not df2.empty:
            c = "事件" if "事件" in df2.columns else df2.columns[2]
            titles += [str(x) for x in df2[c].head(max(1, min(limit, 100))).tolist()]
            fetched.append(f"news_economic_baidu({len(df2)})")
    except Exception:  # noqa: BLE001
        pass
    if not titles:
        return {"ok": False, "coverage": "market_aggregate", "source_symbol": None,
                "error": "新闻接口均不可用", "fetched": fetched}
    result = _news_summary(titles, records, source="news_akshare", coverage="market_aggregate")
    result["fetched"] = fetched
    return result


# ────────────────────────────────────────────────────────────
# 汇总 + 滚动存储
# ────────────────────────────────────────────────────────────
def fetch_baidu() -> dict:
    """百度股市通热搜（ak.stock_hot_search_baidu，Vultr 境外IP 已验证可用）。

    2026-08-10 审计修复:
      - akshare 该接口默认 date='20250616' 写死 → 必须显式传动态今天，禁止默认参数
      - 本地 data_warehouse/social/baidu_hot_*.json 快照（Vultr 爬虫产物）优先，
        但必须校验快照新鲜度：>2 日视为陈旧，先尝试直连今天，失败则如实降级，
        绝不静默用旧快照当"今日热搜"。
    """
    import glob
    import akshare as ak

    def _direct() -> dict:
        """直连百度股市通（显式动态日期）。"""
        try:
            df = ak.stock_hot_search_baidu(
                symbol="A股",
                date=datetime.now(CST).strftime("%Y%m%d"),
                time="今日",
            )
        except Exception as e:
            return {"ok": False, "error": f"百度热搜直连失败: {str(e)[:80]}"}
        if df is None or df.empty:
            return {"ok": False, "error": "百度热搜直连无数据"}
        names = df["名称/代码"].astype(str).tolist()
        heat = pd.to_numeric(df.get("综合热度"), errors="coerce")
        out = {
            "ok": True, "source": "baidu_hot_direct", "n": len(df),
            "top": names[:10], "as_of": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment": round(float(sentiment_score(" ".join(names))), 3),
            "note": "百度股市通热搜直连（今日动态）",
        }
        if heat is not None and heat.notna().any():
            out["heat_top"] = [round(float(h), 0) if pd.notna(h) else None for h in heat.head(10).tolist()]
        return out

    # 1) 本地 Vultr 快照优先，但校验新鲜度
    try:
        files = sorted(glob.glob(str(SOCIAL_DIR / "baidu_hot_*.json"))) if SOCIAL_DIR.exists() else []
        if files:
            with open(files[-1], encoding="utf-8") as f:
                d = json.load(f)
            items = d.get("items", [])
            as_of = str(d.get("date", ""))
            snap_stale = True
            _now_cst = datetime.now(CST)
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
                try:
                    _snap_dt = datetime.strptime(as_of, fmt).replace(tzinfo=CST)
                    snap_stale = (_now_cst - _snap_dt).days > 2
                    break
                except ValueError:
                    continue
            if snap_stale:
                # 快照陈旧 → 降级为直连今日；直连失败则如实报错
                direct = _direct()
                if direct.get("ok"):
                    return direct
                return {"ok": False,
                        "error": f"本地快照 {as_of} 陈旧且直连失败: {direct.get('error', '')[:80]}"}
            if items:
                names = []
                heat = []
                for it in items:
                    if not isinstance(it, dict) or not it:
                        continue
                    nm = it.get("名称/代码") or list(it.values())[0]
                    names.append(str(nm))
                    hv = it.get("综合热度")
                    try:
                        heat.append(float(hv) if hv is not None else None)
                    except (TypeError, ValueError):
                        heat.append(None)
                return {
                    "ok": True, "source": "baidu_hot", "n": len(items),
                    "top": names[:10], "as_of": as_of,
                    "sentiment": round(float(sentiment_score(" ".join(names))), 3),
                    "heat_top": heat[:10],
                    "note": "百度热搜(股票词) 散户关注度代理（Vultr 抓取）",
                }
    except Exception as e:
        logging.getLogger(__name__).error(f"[social_sentiment] 操作失败: {e}", exc_info=True)
    # 2) 直连兜底（显式动态日期）
    return _direct()


# ────────────────────────────────────────────────────────────
# V12.3 市场情绪综合（用户需求: 微博/小红书/B站/新闻 → 短线情绪指标）
# ────────────────────────────────────────────────────────────
def market_sentiment_composite(live: list[dict] | None = None) -> dict:
    """聚合多源社交/新闻情绪为「市场情绪综合指标」。

    输入: collect_all 的 rows（已含各平台结果）; 缺省则从缓存(≤10min)取, 无缓存才现场爬一次。
    逐源取 sentiment(词典分, [-1,1])/bull_bear_ratio……，做等权 + 可用性加权，
    输出:
      - market_sentiment: 综合 [-1,1]（>0 偏多, <0 偏空）
      - bull_bear_ratio:  看多/看空 比
      - available:        可用源数/总数（诚实标注数据覆盖）
      - per_source:       各源明细
    """
    import time as _t
    _ttl = getattr(market_sentiment_composite, "_cache", None)
    if live is None:
        if _ttl and (_t.time() - _ttl["ts"]) < 600:
            return _ttl["val"]  # 10min 缓存, 避免盘中/短线多路径重复爬网络
        live = collect_all()
    # B1 复审修复: 只对"产生真实文本情绪"的源做综合平均(guba/baidu 是对股票代码/名称
    # 打分恒≈0 的"常零源", 若计入会把综合情绪死死拉向中性→轴失效)。guba/baidu 只在
    # per_source 里如实展示, 不进 score 平均。这样情绪轴才有净信号压力。
    INFO_SOURCES = {"weibo", "bilibili", "news", "xiaohongshu"}
    sources = INFO_SOURCES
    per = {}
    score_sum, avail = 0.0, 0
    bull_sum, bear_sum = 0, 0
    for item in live:
        p = item.get("platform")
        if p not in sources:
            if p in {"guba", "baidu", "theme_heat", "qianqian"}:
                per[p] = {"ok": False, "error": "覆盖only(无文本情绪或代码名打分无意为)"}
            continue
        try:
            v = json.loads(item.get("value") or "{}")
        except Exception:  # noqa: BLE001
            v = {}
        if not v.get("ok"):
            per[p] = {"ok": False, "error": v.get("error", "")[:60]}
            continue
        s = float(v.get("sentiment", 0.0) or 0.0)
        score_sum += s
        avail += 1
        bull_sum += int(v.get("bull", 0) or 0)
        bear_sum += int(v.get("bear", 0) or 0)
        per[p] = {"ok": True, "sentiment": round(s, 3),
                  "n": v.get("n"), "source": v.get("source")}
    market = round(score_sum / avail, 4) if avail else None
    res = {
        "market_sentiment": market,
        "bull_bear_ratio": round(bull_sum / max(bear_sum, 1), 3) if (bull_sum or bear_sum) else None,
        "available": avail,
        "total": len(sources),
        "coverage": round(avail / len(sources), 2),
        "per_source": per,
        "updated": datetime.now(CST).strftime("%Y-%m-%d %H:%M"),
    }
    if live is None:
        market_sentiment_composite._cache = {"ts": _t.time(), "val": res}
    return res


def collect_all() -> list[dict]:
    """全平台采集，逐适配器 try/except 隔离（C1 复审修复）。任一适配器抛异常
    只记为 ok=False，不拖垮 collect_all / battle_map / daily_report。"""
    # 名 -> 无参调用方（懒执行，逐项 try/except）
    fetcher_calls = [
        ("theme_heat", lambda: theme_heat_from_zt_pool()),
        ("qianqian", lambda: qianqian_comment()),
        ("baidu", lambda: fetch_baidu()),
        ("guba", lambda: fetch_guba()),
        ("weibo", lambda: fetch_weibo()),
        ("bilibili", lambda: fetch_bilibili()),
        ("news", lambda: fetch_news()),
        ("xiaohongshu", lambda: fetch_xiaohongshu()),
    ]
    today = datetime.now(CST).strftime("%Y-%m-%d")
    rows = []
    for platform, fn in fetcher_calls:
        try:
            res = fn()
            res = res if isinstance(res, dict) else {"ok": False, "error": "非 dict 返回"}
        except Exception as e:  # noqa: BLE001 - 单适配器失败只记录
            res = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        rows.append({
            "date": today,
            "platform": platform,
            "ok": bool(res.get("ok")),
            "value": json.dumps(res, ensure_ascii=False),
            "error": res.get("error", "") if not res.get("ok") else "",
        })
    return rows


def store(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if SOCIAL_PARQUET.exists():
        old = pd.read_parquet(SOCIAL_PARQUET)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["date", "platform"], keep="last").reset_index(drop=True)
    df.to_parquet(SOCIAL_PARQUET, index=False)
    print(f"[social] {len(df)} 条记录（滚动累积）→ {SOCIAL_PARQUET}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="社交情绪融合层")
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args()
    rows = collect_all()
    for r in rows:
        status = "✅" if r["ok"] else f"⚠️ {r['error'][:60]}"
        print(f"[{r['platform']}] {status}")
        if r["ok"]:
            v = json.loads(r["value"])
            if r["platform"] == "theme_heat":
                print("   题材热度:", " > ".join(f"{x['industry']}涨停{x['zt_cnt']}" for x in v["industries"][:5]))
            elif "sentiment" in v:
                print(f"   情感分: {v['sentiment']} | Top: {v.get('top', [])[:5]}")
    store(rows)
