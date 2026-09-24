"""
leader_laggard — 领涨/领跌板块轮动追踪 (V5)

回答："当前市场的发动机在哪？"

不对比涨跌幅绝对值，比较：
1. 各行业的相对动量排名（20日 vs 60日）
2. 动量切换信号（之前领涨的现在滞涨了 → 风格切换）
3. 板块动量集中度（所有板块动量标准差 → 市场是否有主线）
4. 动量扩散/收敛（高动量板块数量变化 → 行情扩散还是收窄）

与"温度"的区别：温度看涨多少，这个看哪些板块在领涨/领跌/轮动。
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

CST = timezone(timedelta(hours=8))


class LeaderLaggard:
    """领涨/领跌板块轮动分析。"""

    def __init__(self, cache_ttl: int = 900) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """计算板块轮动状态。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "leading_sectors": [],
            "lagging_sectors": [],
            "rotation_signal": "无明显轮动",
            "momentum_concentration": 0.0,
            "score": 0,
        }

        try:
            import akshare as ak

            # Q22-fix: 原实现 ak.stock_board_industry_hist_em() 无参调用——该接口是
            # **单行业日线历史**（默认 symbol='小金属'），每行=交易日，被当作"板块"；
            # 行=交易日被当成"板块"，导致"领涨前10"实为某行业涨幅最大的10个交易日，
            # 且 row.get("板块名称") 列不存在→名称全空、单日"涨跌幅"冒充 20 日动量。
            # 改为：先用 stock_board_industry_name_em() 拿全行业列表，再逐行业
            # stock_board_industry_hist_em(symbol=行业名) 取日线算 20/60 日动量。
            board = ak.stock_board_industry_name_em()
            if board is None or board.empty:
                return result
            name_col = "板块名称" if "板块名称" in board.columns else board.columns[1]
            industry_names = [str(v) for v in board[name_col].tolist()]
            daily_change = {}
            if "涨跌幅" in board.columns:
                daily_change = dict(zip(industry_names, board["涨跌幅"].tolist()))

            # 逐行业取日线计算 20/60 日动量（最近 120 个自然日 ≈ 80+ 交易日）
            sectors = []
            start_dt = (datetime.now(CST) - timedelta(days=120)).strftime("%Y%m%d")
            end_dt = datetime.now(CST).strftime("%Y%m%d")
            for name in industry_names:
                try:
                    hist = ak.stock_board_industry_hist_em(symbol=name, start_date=start_dt, end_date=end_dt)
                    if hist is None or hist.empty:
                        continue
                    closes = hist["收盘"].astype(float).values
                    if len(closes) < 2:
                        continue
                    momentum = closes[-1] / closes[-21] - 1 if len(closes) > 21 else closes[-1] / closes[0] - 1
                    momentum_long = closes[-1] / closes[-61] - 1 if len(closes) > 61 else momentum
                    sectors.append({
                        "name": name,
                        "momentum_20d": round(float(momentum), 4),
                        "momentum_60d": round(float(momentum_long), 4),
                        "momentum_diff": round(float(momentum - momentum_long), 4),  # 近期加速/减速
                    })
                except Exception:
                    # 单行业取数失败：回退到当日涨跌幅作为 20d 代理，60d 置 0 并保留名称
                    chg = daily_change.get(name, 0)
                    try:
                        chg = float(chg) / 100.0
                    except Exception:
                        chg = 0.0
                    sectors.append({
                        "name": name,
                        "momentum_20d": round(chg, 4),
                        "momentum_60d": 0.0,
                        "momentum_diff": round(chg, 4),
                    })

            if not sectors:
                return result

            # 按20日动量排序
            sorted_by_mom = sorted(sectors, key=lambda x: x["momentum_20d"], reverse=True)

            # 领涨前10
            leading = sorted_by_mom[:10]
            result["leading_sectors"] = [
                {"name": s["name"], "momentum_20d": s["momentum_20d"],
                 "momentum_60d": s["momentum_60d"],
                 "accelerating": s["momentum_diff"] > 0}
                for s in leading
            ]

            # 领跌前10
            lagging = sorted_by_mom[-10:]
            result["lagging_sectors"] = [
                {"name": s["name"], "momentum_20d": s["momentum_20d"],
                 "momentum_60d": s["momentum_60d"],
                 "accelerating": s["momentum_diff"] > 0}
                for s in lagging
            ]

            # 动量集中度 (标准差)
            all_mom = [s["momentum_20d"] for s in sectors]
            concentration = float(np.std(all_mom)) if len(all_mom) > 1 else 0
            result["momentum_concentration"] = round(concentration, 4)

            # 加速/减速板块数量
            accelerating = sum(1 for s in sectors if s["momentum_diff"] > 0)
            decelerating = len(sectors) - accelerating
            result["accelerating_count"] = accelerating
            result["decelerating_count"] = decelerating

            # ── 轮动信号判断 ──

            # 1. 动量集中度（主线强度）
            if concentration > 0.03:
                result["rotation_signal"] = "主线清晰，动量集中在少数板块"
            elif concentration > 0.015:
                result["rotation_signal"] = "有主线但不强，板块间分化中"
            else:
                result["rotation_signal"] = "无明显主线，快速轮动市"

            # 2. 领先板块是否也在加速
            top5_momentum = [s["momentum_20d"] for s in leading[:5]]
            top5_60d = [s["momentum_60d"] for s in leading[:5]]
            if top5_momentum and top5_60d:
                avg_mom = np.mean(top5_momentum)
                avg_60m = np.mean(top5_60d)
                if avg_mom > avg_60m:
                    result["rotation_signal"] += "，领涨板块在加速"
                else:
                    result["rotation_signal"] += "，领涨板块动能衰减"

            # 3. 动量扩散指标
            high_mom = sum(1 for s in sectors if s["momentum_20d"] > 0.05)
            result["high_momentum_count"] = high_mom

            # ── 综合评分 ──
            score = 0.0
            if accelerating > decelerating * 1.5:
                score += 2
            if concentration > 0.02:
                score += 2
            if len(leading) > 0 and leading[0]["momentum_20d"] > 0.1:
                score += 1
            result["score"] = round(max(-10, min(10, score)), 1)
            # P2-Q22-fix(M243): 原 50+score*6.8 线性映射冒充"历史分位"输出。
            # 真实历史分位未实现，改输出字段标注为 score 映射值：
            #   - score_mapped_pctile 为新字段（明确为 score 映射）；
            #   - percentile 保留作兼容别名（值为同一映射值），
            #     并附 percentile_basis/percentile_note 说明其非历史分位。
            mapped = round(max(0, min(100, 50 + result["score"] * 6.8)), 1)
            result["score_mapped_pctile"] = mapped
            result["percentile"] = mapped
            result["percentile_basis"] = "score_mapping"
            result["percentile_note"] = "score线性映射值(50+score*6.8)，非真实历史分位；真实历史分位未实现"

        except Exception as e:
            result["error"] = str(e)

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    ll = LeaderLaggard()
    r = ll.compute()
    print("═" * 55)
    print(f"  板块轮动追踪")
    print("═" * 55)
    print(f"  轮动状态: {r.get('rotation_signal', 'N/A')}")
    print(f"  动量集中度: {r.get('momentum_concentration', 'N/A')}")
    print(f"  加速/减速板块: {r.get('accelerating_count', '?')}/{r.get('decelerating_count', '?')}")
    print(f"  高动量板块数(20日>5%): {r.get('high_momentum_count', '?')}")
    print(f"  综合评分: {r.get('score', 'N/A')}")

    print("\n  ── 领涨前10 ──")
    for s in r.get("leading_sectors", [])[:5]:
        acc = "🚀" if s.get("accelerating") else "⬇️"
        print(f"  {acc} {s['name']:<8} 20d:{s['momentum_20d']:>+7.2%}  60d:{s['momentum_60d']:>+7.2%}")

    print("\n  ── 领跌前5 ──")
    for s in r.get("lagging_sectors", [])[:5]:
        print(f"  {s['name']:<8} 20d:{s['momentum_20d']:>+7.2%}  60d:{s['momentum_60d']:>+7.2%}")


if __name__ == "__main__":
    main()
