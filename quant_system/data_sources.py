"""
Data source configuration — 干净分离的国内外 API 双栈。

国内栈 (CN):
  akshare 为主，腾讯/新浪为港股行情备选。

国际栈 (Global):
  Yahoo Finance 为主，腾讯 qt.gtimg.cn 为港美股实时行情。

快速切换: 在 create_api() 传入 override 字典即可热替换端点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── 国内数据源 ──────────────────────────────────────────────────────────────

@dataclass
class CNApi:
    """A 股国内数据源配置"""
    # 日 K 线
    hist: str = "akshare"                       # akshare (stock_zh_a_hist / stock_zh_a_hist_tx)
    # 分钟 K 线
    intraday: str = "akshare"                   # akshare (stock_zh_a_hist_min_em)
    # 板块
    sector: str = "akshare"                     # akshare (stock_board_industry_name_em)
    # 实时行情
    realtime: str = "akshare"                   # akshare / sina
    # 融资融券
    margin_sse: str = "sse_official"            # 上交所官网
    margin_szse: str = "szse_official"          # 深交所官网
    # 名称检索
    name_search: str = "akshare"                # akshare / local

    # 港股行情（国内栈内嵌港股）
    hk_quote: str = "tencent"                   # qt.gtimg.cn（海外可用）
    hk_quote_fallback: str = "sina"             # hq.sinajs.cn


@dataclass
class GlobalApi:
    """港美股/海外数据源配置"""
    # K 线
    kline: str = "yahoo"                        # query1.finance.yahoo.com
    # 实时行情
    realtime: str = "tencent"                   # qt.gtimg.cn（海外可访问）
    realtime_fallback: str | None = None
    # 名称解析 → 代码
    name_resolve_local: dict = field(default_factory=lambda: {
        "腾讯": "hk00700",
        "香港交易所": "hk00388",
        "苹果": "usAAPL",
        "特斯拉": "usTSLA",
        "微软": "usMSFT",
        "谷歌": "usGOOGL",
        "亚马逊": "usAMZN",
        "英伟达": "usNVDA",
        "Meta": "usMETA",
        "阿里巴巴": "usBABA",
    })


# ── 完整配置 ─────────────────────────────────────────────────────────────────

@dataclass
class ApiConfig:
    """全栈 API 配置"""
    cn: CNApi = field(default_factory=CNApi)
    global_: GlobalApi = field(default_factory=GlobalApi)

    def override(self, **overrides: Any) -> ApiConfig:
        """返回浅拷贝覆盖副本（仅顶层键覆盖）。"""
        import copy
        out = copy.copy(self)
        for k, v in overrides.items():
            if hasattr(out, k):
                setattr(out, k, v)
        return out


# ── 默认实例 ─────────────────────────────────────────────────────────────────

DEFAULT_API_CONFIG = ApiConfig()


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def get_api_config(config: ApiConfig | None = None) -> ApiConfig:
    """获取当前 API 配置。传入 None 返回默认。"""
    return config or DEFAULT_API_CONFIG


__all__ = [
    "ApiConfig", "CNApi", "GlobalApi",
    "DEFAULT_API_CONFIG", "get_api_config",
]
