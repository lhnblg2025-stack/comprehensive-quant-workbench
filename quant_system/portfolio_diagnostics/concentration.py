"""
concentration — 持仓集中度分析 (V5)

核心问题：我的组合是不是"看起来分散、实际上很集中"？
  1. 个股集中度 — HHI / Top1 / Top5 / Top10 权重占比
  2. 行业集中度 — 行业 HHI + 最大行业占比
  3. 风格集中度 — 大盘/中盘/小盘市值分布
  4. 风险预警 — 触发阈值时给出具体预警文案

对标：券商/公募机构持仓集中度合规检查（单票/单行业占比上限）。
"""

from __future__ import annotations
import logging

import time as _time
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


CST = timezone(timedelta(hours=8))

ROOT = Path(__file__).resolve().parent.parent

SECTOR_FALLBACK = "其他"

# ── 预警阈值 ──
SINGLE_STOCK_WARN = 0.15       # 单票 > 15%
INDUSTRY_WARN = 0.40           # 行业 > 40%
HHI_WARN = 0.20                # HHI > 0.2
TOP5_WARN = 0.70                # 前5大 > 70%

# ── 风格分位阈值 ──
LARGE_CAP_PCTL = 0.80           # >80分位 = 大盘
SMALL_CAP_PCTL = 0.20           # <20分位 = 小盘


class ConcentrationAnalysis:
    """持仓集中度分析引擎。

    核心假设：
      分散化不等于股票数量多，而是权重分布均匀 + 行业/风格不过度集中。
      HHI (Herfindahl-Hirschman Index) 是衡量集中度的标准指标：
        HHI = Σ(w_i)²，取值范围 [1/n, 1]，越接近1越集中。

    典型用法::

        ca = ConcentrationAnalysis()
        result = ca.compute(holdings)

    Attributes:
        cache: 上一次计算结果缓存
        last_fetch: 上次计算的时间戳
        cache_ttl: 缓存有效期（秒）
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0.0
        self.cache_ttl = cache_ttl
        self._sector_cache: dict[str, str] = {}
        self._market_cap_cache: dict[str, float] = {}
        self._spot_snapshot: Any = None
        self._spot_fetch_time: float = 0.0

    # ────────────────────────── 数据获取 ──────────────────────────

    def get_sector(self, code: str) -> str:
        """获取个股所属行业分类（带缓存，缺失时归类为"其他"）。

        Args:
            code: 股票代码

        Returns:
            str: 行业名称；获取失败或缺失时返回 "其他"。
        """
        if code in self._sector_cache:
            return self._sector_cache[code]

        sector = SECTOR_FALLBACK
        try:
            import akshare as ak
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and not info.empty:
                d = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
                candidate = d.get("行业")
                if candidate:
                    sector = str(candidate).strip() or SECTOR_FALLBACK
        except Exception as e:
            logging.getLogger(__name__).error(f"[concentration] 操作失败: {e}", exc_info=True)

        self._sector_cache[code] = sector
        return sector

    def _get_spot_snapshot(self):
        """获取全市场实时快照（含市值），带缓存以避免重复全量拉取。"""
        now = _time.time()
        if self._spot_snapshot is not None and now - self._spot_fetch_time < self.cache_ttl:
            return self._spot_snapshot
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            self._spot_snapshot = df
            self._spot_fetch_time = now
            return df
        except Exception:
            self._spot_snapshot = None
            return None

    def get_market_cap(self, code: str) -> float:
        """获取个股总市值（元）。

        Args:
            code: 股票代码

        Returns:
            float: 总市值（元），获取失败返回 NaN。
        """
        if code in self._market_cap_cache:
            return self._market_cap_cache[code]

        cap = float("nan")
        try:
            df = self._get_spot_snapshot()
            if df is not None and "代码" in df.columns:
                cap_col = None
                for c in df.columns:
                    if "总市值" in str(c):
                        cap_col = c
                        break
                row = df[df["代码"] == code]
                if not row.empty and cap_col is not None:
                    cap = float(row.iloc[0][cap_col])
        except Exception as e:
            logging.getLogger(__name__).error(f"[concentration] 操作失败: {e}", exc_info=True)

        self._market_cap_cache[code] = cap
        return cap

    # ────────────────────────── 核心计算 ──────────────────────────

    @staticmethod
    def compute_hhi(weights: list[float]) -> float:
        """计算 Herfindahl-Hirschman Index。

        Args:
            weights: 权重列表（应已归一化，和为1）

        Returns:
            float: HHI = Σ(w_i)²，范围 [1/n, 1]
        """
        if not weights:
            return 0.0
        arr = np.asarray(weights, dtype=float)
        return round(float(np.sum(arr ** 2)), 4)

    @staticmethod
    def _top_n_weight(sorted_weights: list[float], n: int) -> float:
        """前 n 大权重之和。"""
        return round(float(sum(sorted_weights[:n])), 4)

    def _compute_sector_concentration(
        self, codes: list[str], weights: dict[str, float]
    ) -> dict[str, Any]:
        """计算行业集中度。

        Args:
            codes: 股票代码列表
            weights: {code: 归一化权重}

        Returns:
            dict: {hhi, top_sector, sectors: [{name, weight, stock_count}, ...]}
        """
        sector_weight: dict[str, float] = {}
        sector_count: dict[str, int] = {}
        for code in codes:
            sector = self.get_sector(code)
            w = weights.get(code, 0.0)
            sector_weight[sector] = sector_weight.get(sector, 0.0) + w
            sector_count[sector] = sector_count.get(sector, 0) + 1

        sectors = sorted(
            [
                {"name": s, "weight": round(w, 4), "stock_count": sector_count[s]}
                for s, w in sector_weight.items()
            ],
            key=lambda x: x["weight"], reverse=True,
        )
        hhi = self.compute_hhi(list(sector_weight.values()))
        top_sector = sectors[0] if sectors else {"name": SECTOR_FALLBACK, "weight": 0.0}

        return {
            "hhi": hhi,
            "top_sector": {"name": top_sector["name"], "weight": top_sector["weight"]},
            "sectors": sectors,
        }

    def _full_market_caps(self) -> list[float]:
        """从全市场快照提取所有股票的总市值，作为风格分位基准。

        Returns:
            有效市值列表（正且有限）；快照不可得或解析失败返回空列表。
        """
        try:
            df = self._get_spot_snapshot()
            if df is None or df.empty:
                return []
            cap_col = None
            for c in df.columns:
                if "总市值" in str(c):
                    cap_col = c
                    break
            if cap_col is None:
                return []
            caps: list[float] = []
            for v in df[cap_col]:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(f) and f > 0:
                    caps.append(f)
            return caps
        except Exception:
            return []

    def _compute_style_concentration(
        self, codes: list[str], weights: dict[str, float]
    ) -> dict[str, float]:
        """计算风格集中度（大盘/中盘/小盘市值分布）。

        用全市场市值分位数划分（复用 _get_spot_snapshot 的一次全市场快照）：
          >80分位 = 大盘, 20~80分位 = 中盘, <20分位 = 小盘

        Args:
            codes: 股票代码列表
            weights: {code: 归一化权重}

        Returns:
            dict: {large_cap, mid_cap, small_cap} 权重占比之和应接近1
                  （无法获取市值的股票不计入，因此可能略小于1）；
                  另含 market_based 标记分位基准是否来自全市场。
        """
        caps = {code: self.get_market_cap(code) for code in codes}
        valid_caps = [c for c in caps.values() if np.isfinite(c) and c > 0]

        style_weight = {"large_cap": 0.0, "mid_cap": 0.0, "small_cap": 0.0}
        if not valid_caps:
            return {k: round(v, 4) for k, v in style_weight.items()}

        # P2-Q21-fix: 分位数基准改用全市场市值分布（原实现用组合持仓自身市值，
        # 10 只大盘股组合会把最小市值成员误判为"小盘"）。复用 _get_spot_snapshot
        # 已缓存的全市场快照，不额外增加网络请求。
        market_caps = self._full_market_caps()
        if market_caps:
            p80 = float(np.percentile(market_caps, 80))
            p20 = float(np.percentile(market_caps, 20))
        else:
            # 全市场快照不可得：降级用组合自身市值分位（可见降级，market_based=False）
            p80 = float(np.percentile(valid_caps, 80))
            p20 = float(np.percentile(valid_caps, 20))

        for code, cap in caps.items():
            if not np.isfinite(cap) or cap <= 0:
                continue
            w = weights.get(code, 0.0)
            if cap >= p80:
                style_weight["large_cap"] += w
            elif cap <= p20:
                style_weight["small_cap"] += w
            else:
                style_weight["mid_cap"] += w

        result = {k: round(v, 4) for k, v in style_weight.items()}
        result["market_based"] = bool(market_caps)
        return result

    def _generate_warnings(
        self,
        sorted_holdings: list[tuple[str, str, float]],
        hhi: float,
        top5_weight: float,
        sector_conc: dict[str, Any],
    ) -> list[str]:
        """根据阈值规则生成风险预警文案。

        规则:
          - 单票 > 15% → 预警
          - 行业 > 40% → 预警
          - HHI > 0.2 → 预警
          - 前5大 > 70% → 预警

        Args:
            sorted_holdings: [(code, name, weight), ...] 按权重降序
            hhi: 个股 HHI
            top5_weight: 前5大权重占比
            sector_conc: _compute_sector_concentration 的返回值

        Returns:
            list[str]: 预警文案列表
        """
        warnings: list[str] = []

        if sorted_holdings:
            top_code, top_name, top_w = sorted_holdings[0]
            if top_w > SINGLE_STOCK_WARN:
                warnings.append(
                    f"最大持仓 {top_name}({top_code}) 占比{top_w:.1%},"
                    f"超过单票风险阈值{SINGLE_STOCK_WARN:.0%}"
                )

        top_sector = sector_conc.get("top_sector", {})
        sector_w = top_sector.get("weight", 0.0)
        if sector_w > INDUSTRY_WARN:
            warnings.append(
                f"{top_sector.get('name', SECTOR_FALLBACK)}行业占比{sector_w:.1%},过于集中"
            )

        if hhi > HHI_WARN:
            warnings.append(f"个股HHI={hhi:.3f},超过{HHI_WARN}警戒线,持仓过度集中")

        if top5_weight > TOP5_WARN:
            warnings.append(f"前5大持仓占比{top5_weight:.1%},超过{TOP5_WARN:.0%}集中度上限")

        if sector_conc.get("hhi", 0) > HHI_WARN:
            warnings.append(
                f"行业HHI={sector_conc['hhi']:.3f},行业分布过度集中于少数板块"
            )

        return warnings

    # ────────────────────────── 主入口 ──────────────────────────

    def compute(self, holdings: list[dict[str, Any]]) -> dict[str, Any]:
        """计算组合的完整集中度诊断报告。

        Args:
            holdings: 持仓列表 [{"code": "600519", "weight": 0.15,
                      "name": "贵州茅台"}, ...]，权重会自动归一化。

        Returns:
            dict: 见模块文档规格，出错时返回 {"error": str}。
        """
        cache_key = str(sorted((h.get("code"), h.get("weight")) for h in holdings))
        now = _time.time()
        if (
            self.cache.get("_key") == cache_key
            and now - self.last_fetch < self.cache_ttl
        ):
            return self.cache

        try:
            if not holdings:
                return {"error": "无持仓数据"}

            codes = [h["code"] for h in holdings if h.get("code")]
            if not codes:
                return {"error": "持仓缺少有效股票代码"}

            raw_weights = {h["code"]: float(h.get("weight", 0) or 0) for h in holdings}
            total_w = sum(raw_weights.values())
            if total_w <= 0:
                return {"error": "持仓权重总和为0，无法归一化"}
            weights = {k: v / total_w for k, v in raw_weights.items()}

            name_map = {h.get("code"): h.get("name", h.get("code")) for h in holdings}
            sorted_holdings = sorted(
                [(c, name_map.get(c, c), weights[c]) for c in codes],
                key=lambda x: x[2], reverse=True,
            )
            sorted_weights = [w for _, _, w in sorted_holdings]

            hhi = self.compute_hhi(sorted_weights)
            top1 = self._top_n_weight(sorted_weights, 1)
            top5 = self._top_n_weight(sorted_weights, 5)
            top10 = self._top_n_weight(sorted_weights, 10)

            sector_conc = self._compute_sector_concentration(codes, weights)
            style_conc = self._compute_style_concentration(codes, weights)

            warnings = self._generate_warnings(sorted_holdings, hhi, top5, sector_conc)

            result: dict[str, Any] = {
                "_key": cache_key,
                "hhi": hhi,
                "top1_weight": top1,
                "top5_weight": top5,
                "top10_weight": top10,
                "stock_count": len(codes),
                "top_holdings": [
                    {"code": c, "name": n, "weight": round(w, 4)}
                    for c, n, w in sorted_holdings[:10]
                ],
                "sector_concentration": sector_conc,
                "style_concentration": style_conc,
                "risk_warnings": warnings,
            }
            self.cache = result
            self.last_fetch = now
            return result
        except Exception as e:
            return {"error": f"ConcentrationAnalysis.compute 失败: {e}"}


def main() -> None:
    """示例：计算模拟持仓的集中度诊断。"""
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.20},
        {"code": "000858", "name": "五粮液", "weight": 0.15},
        {"code": "300750", "name": "宁德时代", "weight": 0.12},
        {"code": "601318", "name": "中国平安", "weight": 0.10},
        {"code": "000333", "name": "美的集团", "weight": 0.08},
        {"code": "002415", "name": "海康威视", "weight": 0.07},
        {"code": "600036", "name": "招商银行", "weight": 0.06},
        {"code": "000002", "name": "万科A", "weight": 0.05},
        {"code": "601166", "name": "兴业银行", "weight": 0.04},
        {"code": "002304", "name": "洋河股份", "weight": 0.04},
    ]

    ca = ConcentrationAnalysis()
    result = ca.compute(holdings)

    logger.info("═" * 65)
    logger.info("  持仓集中度诊断")
    logger.info("═" * 65)

    if "error" in result:
        logger.warning(f"  ⚠️ 计算失败: {result['error']}")
        return

    logger.info(f"\n  持仓数量: {result['stock_count']}只")
    logger.info(f"  HHI      : {result['hhi']:.4f}")
    logger.info(f"  Top1权重 : {result['top1_weight']:.1%}")
    logger.info(f"  Top5权重 : {result['top5_weight']:.1%}")
    logger.info(f"  Top10权重: {result['top10_weight']:.1%}")

    logger.info()
    logger.info("  ── 行业集中度 ──")
    sc = result["sector_concentration"]
    logger.info(f"    行业HHI: {sc['hhi']:.4f}  最大行业: {sc['top_sector']['name']} "
          f"({sc['top_sector']['weight']:.1%})")
    for s in sc["sectors"][:5]:
        logger.info(f"    {s['name']:<10} {s['weight']:>7.1%}  ({s['stock_count']}只)")

    logger.info()
    logger.info("  ── 风格集中度 ──")
    style = result["style_concentration"]
    logger.info(f"    大盘: {style['large_cap']:.1%}  中盘: {style['mid_cap']:.1%}  "
          f"小盘: {style['small_cap']:.1%}")

    logger.info()
    if result["risk_warnings"]:
        logger.warning("  ⚠️ 风险预警:")
        for w in result["risk_warnings"]:
            logger.warning(f"    • {w}")
    else:
        logger.info("  ✓ 未触发集中度预警")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
