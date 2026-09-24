"""
quant_platform.alt_data — 另类数据融合层（V3）

来源：quant_v6/data/{esg,option,pledge,news,comment_sentiment,ttm_valuation}.py
设计：纯函数式实现（无 quant_v6 依赖），数据走 DataStore / akshare 直连，
      所有网络函数捕获异常并返回空容器，绝不崩溃。

能力：
  1. ESG 评级（东财 stock_esg_em）：另类质量因子——高分公司风险溢价与机构偏好
  2. 股权质押（东财 stock_gpzy_pledge_ratio_em）：暴雷预警——高质押率尾部风险过滤
  3. 期权（大商所商品期权 + 金融期权）：IV 与认购认沽比——市场恐慌/贪婪另类指标
  4. 新闻/评论情绪：舆情代理
  5. TTM 估值：滚动市盈率口径
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)

# 高质押风险阈值（%）
HIGH_PLEDGE_THRESHOLD = 50.0


# ══════════════════════════════════════════════════════════
# 1. ESG 评级
# ══════════════════════════════════════════════════════════

def get_esg_rating() -> pd.DataFrame:
    """拉取全市场 ESG 评级（回退链：新浪 ESG 评级 → 慧真 ESG → 东财旧接口）。

    注意: akshare 1.18.64 无 stock_esg_em（quant_v6 开发期版本有），
    用新浪/慧真评级接口替代；列名版本变化时模糊匹配。

    Returns:
        ESG 评级 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    for fn in ("stock_esg_rate_sina", "stock_esg_hz_sina", "stock_esg_em"):
        f = getattr(ak, fn, None)
        if f is None:
            continue
        try:
            df = f()
            if df is not None and not df.empty:
                return df.copy()
        except Exception as exc:
            logger.warning("拉取 ESG 评级(%s)失败: %s", fn, exc)
    return pd.DataFrame()


def esg_score_map() -> dict[str, dict]:
    """ESG 评级映射: {code: {"esg_rating": str, "esg_score": float|None}}。

    东财列名含"代码"/"ESG评级"/"ESG评分"（版本变化，模糊匹配）。
    """
    df = get_esg_rating()
    if df.empty:
        return {}
    code_col = next((c for c in df.columns if "代码" in str(c) or "code" in str(c).lower()), None)
    rating_col = next((c for c in df.columns
                       if "ESG" in str(c).upper() and ("评级" in str(c) or "等级" in str(c))), None)
    score_col = next((c for c in df.columns if "ESG" in str(c).upper() and "评分" in str(c)), None)
    if code_col is None:
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        code = str(row[code_col]).zfill(6)
        # V3 审计修复: 新浪源代码带 .SH/.SZ 后缀（600519.SH），剥离后才可单股查询
        code = code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        item: dict = {"esg_rating": str(row[rating_col]) if rating_col else None}
        if score_col:
            try:
                item["esg_score"] = float(row[score_col])
            except (TypeError, ValueError):
                item["esg_score"] = None
        else:
            item["esg_score"] = None
        out[code] = item
    return out


def esg_rating_of(symbol: str) -> Optional[dict]:
    """单只股票 ESG 评级。"""
    m = esg_score_map()
    return m.get(str(symbol).zfill(6))


def esg_quality_factor(top_pct: float = 0.2) -> list[str]:
    """ESG 质量因子: 评分最高前 top_pct 股票代码。"""
    df = get_esg_rating()
    if df.empty:
        return []
    code_col = next((c for c in df.columns if "代码" in str(c) or "code" in str(c).lower()), None)
    score_col = next((c for c in df.columns if "ESG" in str(c).upper() and "评分" in str(c)), None)
    if code_col is None or score_col is None:
        return []
    try:
        tmp = df[[code_col, score_col]].copy()
        tmp.columns = ["code", "score"]
        tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
        tmp = tmp.dropna(subset=["score"]).sort_values("score", ascending=False)
        n = max(1, int(len(tmp) * top_pct))
        return [str(c).zfill(6) for c in tmp["code"].head(n).tolist()]
    except Exception as exc:
        logger.warning("计算 ESG 质量因子失败: %s", exc)
        return []


# ══════════════════════════════════════════════════════════
# 2. 股权质押
# ══════════════════════════════════════════════════════════

def get_pledge_ratio(symbol: str = "全部股票", date: str = "") -> pd.DataFrame:
    """拉取股权质押比例（东财 stock_gpzy_pledge_ratio_em）。

    注意: akshare 1.18.64 签名是 (date)，quant_v6 开发期版本是 (symbol)。
    适配: 无 date 参数时用最近交易日；symbol 参数仅兼容旧调用。
    """
    if ak is None:
        return pd.DataFrame()
    try:
        import inspect
        sig = inspect.signature(ak.stock_gpzy_pledge_ratio_em)
        if "date" in sig.parameters and "symbol" not in sig.parameters:
            d = date or "20240906"
            df = ak.stock_gpzy_pledge_ratio_em(date=d)
        else:
            df = ak.stock_gpzy_pledge_ratio_em(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取质押比例失败 %s: %s", symbol, exc)
        return pd.DataFrame()


def pledge_ratio_of(symbol: str) -> Optional[float]:
    """单只股票质押比例(%)。"""
    df = get_pledge_ratio(symbol)
    if df.empty:
        return None
    try:
        col = next((c for c in df.columns if "质押比例" in str(c)), None)
        if col is None:
            return None
        return float(df.iloc[0][col])
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning("解析质押比例失败 %s: %s", symbol, exc)
        return None


def high_pledge_stocks(threshold: float = HIGH_PLEDGE_THRESHOLD, top_n: int = 50) -> list[dict]:
    """高质押比例股票列表（风险警示池）。"""
    df = get_pledge_ratio("全部股票")
    if df.empty:
        return []
    try:
        code_col = next((c for c in df.columns if "代码" in str(c) or "code" in str(c).lower()), df.columns[0])
        name_col = next((c for c in df.columns if "名称" in str(c) or "name" in str(c).lower()), None)
        ratio_col = next((c for c in df.columns if "质押比例" in str(c)), None)
        if ratio_col is None:
            return []
        out = []
        for _, row in df.iterrows():
            try:
                ratio = float(row[ratio_col])
            except (TypeError, ValueError):
                continue
            if ratio >= threshold:
                out.append({
                    "code": str(row[code_col]).zfill(6),
                    "name": str(row[name_col]) if name_col else "",
                    "pledge_ratio": ratio,
                })
        out.sort(key=lambda x: x["pledge_ratio"], reverse=True)
        return out[:top_n]
    except Exception as exc:
        logger.warning("筛选高质押股失败: %s", exc)
        return []


def pledge_risk_universe() -> set[str]:
    """高质押风险股票代码集合（供选股/风控过滤）。"""
    return {x["code"] for x in high_pledge_stocks()}


# ══════════════════════════════════════════════════════════
# 3. 期权（IV 与认购认沽比）
# ══════════════════════════════════════════════════════════

def get_dce_option_daily(date: str = "20240101") -> pd.DataFrame:
    """大商所商品期权日行情。"""
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.option_dce_daily(date=date)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取大商所期权行情失败 %s: %s", date, exc)
        return pd.DataFrame()


def get_finance_option_daily(
    symbol: str = "中证1000",
    exchange: str = "中国金融期货交易所",
    date: str = "20240101",
) -> pd.DataFrame:
    """金融期权日行情（含 ETF 期权）。"""
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.option_finance_board(symbol=symbol, exchange=exchange, date=date)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取金融期权行情失败 %s: %s", symbol, exc)
        return pd.DataFrame()


def _parse_iv(df: pd.DataFrame) -> Optional[dict]:
    """从期权行情 DataFrame 提取 IV 统计。"""
    if df.empty:
        return None
    try:
        iv_col = next((c for c in df.columns if "隐含波动率" in str(c) or "IV" in str(c).upper()), None)
        type_col = next((c for c in df.columns if "类型" in str(c) or "期权类型" in str(c)), None)
        if iv_col is None:
            return None
        vals = pd.to_numeric(df[iv_col], errors="coerce").dropna()
        if vals.empty:
            return None
        call_iv = put_iv = None
        if type_col is not None:
            call_mask = df[type_col].astype(str).str.contains("看涨|认购|Call", case=False)
            put_mask = df[type_col].astype(str).str.contains("看跌|认沽|Put", case=False)
            call_vals = pd.to_numeric(df.loc[call_mask, iv_col], errors="coerce").dropna()
            put_vals = pd.to_numeric(df.loc[put_mask, iv_col], errors="coerce").dropna()
            if not call_vals.empty:
                call_iv = round(float(call_vals.mean()), 4)
            if not put_vals.empty:
                put_iv = round(float(put_vals.mean()), 4)
        pcr = None
        if call_iv and put_iv and put_iv > 0:
            pcr = round(put_iv / call_iv, 4)
        return {
            "call_iv": call_iv,
            "put_iv": put_iv,
            "put_call_ratio": pcr,
            "avg_iv": round(float(vals.mean()), 4),
        }
    except Exception as exc:
        logger.warning("解析期权 IV 失败: %s", exc)
        return None


def option_sentiment(date: str = "20240101") -> Optional[dict]:
    """期权市场情绪快照（金融期权优先，回退商品期权）。"""
    df = get_finance_option_daily(date=date)
    source = "finance"
    if df.empty:
        df = get_dce_option_daily(date=date)
        source = "dce"
    if df.empty:
        return None
    stats = _parse_iv(df)
    if stats is None:
        return None
    return {"source": source, "date": date, **stats}


# ══════════════════════════════════════════════════════════
# 4. TTM 估值辅助
# ══════════════════════════════════════════════════════════

def ttm_valuation_of(symbol: str, days: int = 800) -> pd.DataFrame:
    """TTM 估值序列（走主栈 DataStore valuation，含 peTTM/pbMRQ/psTTM/pcfNcfTTM）。"""
    from quant_platform.data import valuation
    try:
        return valuation(symbol, days=days)
    except Exception as exc:
        logger.warning("TTM 估值拉取失败 %s: %s", symbol, exc)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# 5. 新闻/评论情绪（舆情代理）
# ══════════════════════════════════════════════════════════

def get_news_em(symbol: str = "600519") -> pd.DataFrame:
    """东财个股新闻（stock_news_em）。"""
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取新闻失败 %s: %s", symbol, exc)
        return pd.DataFrame()


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    print("=== 另类数据融合层调试 ===")
    print("ESG 行数:", len(get_esg_rating()))
    print("高质押 Top5:", high_pledge_stocks(50, 5))
    print("期权情绪:", option_sentiment("20240102"))
