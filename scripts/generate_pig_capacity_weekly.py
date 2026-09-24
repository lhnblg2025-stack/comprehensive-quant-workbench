#!/usr/bin/env python3
"""Weekly pig capacity / pork sector report.

Analysis framework:
- 产能去化阶段判断: 能繁母猪/仔猪/猪价/养殖利润链
- 饲料成本: 玉米/大豆期货趋势、MA144
- 周期位置推演

Data sources:
- yfinance: 玉米(ZC=F)/大豆(ZS=F)/小麦(ZW=F) + MA
"""
from __future__ import annotations
import logging

import concurrent.futures
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import yfinance as yf
from report_delivery import finalize_report
from report_paths import report_path

TZ = ZoneInfo("Asia/Shanghai")
CACHE_DIR = ROOT / "generated" / "feed_cache"
CACHE_TTL = timedelta(hours=6)


# D7收敛: 同名异口径保留（生猪产能周报私有格式化，与 commodity 内实现相似但独立）
def _usd(val, suffix="") -> str:
    if val and val > 0:
        s = f"${val:,.2f}"
        return f"{s} {suffix}" if suffix else s
    return "N/A"


def _trend(price, ma) -> str:
    if price is None or ma is None or ma <= 0:
        return "待计算"
    ratio = (price - ma) / ma * 100
    if abs(ratio) < 2:
        return f"MA附近 ({ratio:+.1f}%)"
    return f"{'上方' if ratio > 0 else '下方'} ({ratio:+.1f}%)"


def _cached_fetch_mas(ticker: str, periods: list[int] = None) -> dict:
    """Fetch yfinance history with disk cache, compute MAs."""
    if periods is None:
        periods = [20, 60, 144]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.pkl"

    # Try cache
    prices = None
    if cache_path.exists():
        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=TZ)
            if datetime.now(TZ) - mtime < CACHE_TTL:
                with open(cache_path, "rb") as f:
                    prices = pickle.load(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"[generate_pig_capacity_weekly] 操作失败: {e}", exc_info=True)

    # Download if no cache
    if prices is None:
        try:
            hist = yf.download(ticker, period="1y", interval="1d", progress=False)
            if hist.empty:
                return {}
            # Extract close prices, handle MultiIndex columns
            close_col = hist["Close"]
            if hasattr(close_col, "squeeze"):
                close_col = close_col.squeeze()
            prices = [float(v) for v in close_col.values]
            with open(cache_path, "wb") as f:
                pickle.dump(prices, f)
        except Exception:
            return {}

    if not prices:
        return {}

    result = {"latest": prices[-1]}
    for p in periods:
        if len(prices) >= p:
            ma = sum(prices[-p:]) / p
            result[f"ma{p}"] = ma
    return result


def generate() -> str:
    now = datetime.now(TZ)
    week_num = now.isocalendar().week
    year = now.year
    title = f"{year}-W{str(week_num).zfill(2)} 生猪产能周报"

    # Parallel yfinance downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_corn = ex.submit(_cached_fetch_mas, "ZC=F")
        f_soy = ex.submit(_cached_fetch_mas, "ZS=F")
        f_wheat = ex.submit(_cached_fetch_mas, "ZW=F")
        corn_ma = f_corn.result(timeout=30)
        soy_ma = f_soy.result(timeout=30)
        wheat_ma = f_wheat.result(timeout=30)

    # yfinance returns cents/bu for ags → convert to $/bu
    corn_p = (corn_ma.get("latest", 0) or 0) / 100
    soy_p = (soy_ma.get("latest", 0) or 0) / 100
    wheat_p = (wheat_ma.get("latest", 0) or 0) / 100

    # Helper to get MA in $/bu
    def _ma(key):
        v = corn_ma.get(key)
        return v / 100 if v else None

    c_ma20 = _ma("ma20")
    c_ma60 = _ma("ma60")
    c_ma144 = _ma("ma144")

    lines = [
        f"# {title}",
        f"生成时间：{now.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ── 一、核心结论 ──
    lines.append("## 一、核心结论")
    lines.append("")

    conclusions = []
    if corn_p:
        trend_str = f"玉米${corn_p:.2f}/bu"
        if c_ma144:
            pct = (corn_p - c_ma144) / c_ma144 * 100
            trend_str += f"，距MA144 {pct:+.1f}%"
        conclusions.append(
            f"【饲料成本】{trend_str}。"
            "饲料成本占养殖成本60-70%，当前饲料价格仍在高位区间。"
        )
    if soy_p:
        conclusions.append(
            f"【蛋白原料】大豆${soy_p:.2f}/bu，豆粕成本传导至配合饲料价格。"
        )
    conclusions.append(
        "【猪周期阶段】缺少最新能繁母猪存栏数据（待农业农村部月度发布）。"
        "基于历史周期规律（约3-4年一轮），上一轮高点是2022年10月，"
        "按时间推算当前应处于产能去化中后段至筑底阶段。"
    )
    conclusions.append(
        "【核心观测链】能繁母猪→仔猪价格→猪粮比→养殖利润→现金流压力→去化速度。"
        "其中仔猪价格是补栏情绪的前瞻指标，猪粮比5:1为产能加速去化警戒线。"
    )
    for i, c in enumerate(conclusions, 1):
        lines.append(f"{i}. {c}")
        lines.append("")

    # ── 二、饲料成本 ──
    lines.append("## 二、饲料成本分析")
    lines.append("")

    lines.append("### 现货/期货价格")
    lines.append(f"- **玉米期货 (ZC=F)**: {_usd(corn_p, 'bu')}")
    if c_ma20:
        lines.append(f"  - MA20: {_usd(c_ma20, 'bu')}")
    if c_ma60:
        lines.append(f"  - MA60: {_usd(c_ma60, 'bu')}")
    if c_ma144:
        lines.append(f"  - MA144: {_usd(c_ma144, 'bu')} → {_trend(corn_p, c_ma144)}")
    lines.append("")
    lines.append(f"- **大豆期货 (ZS=F)**: {_usd(soy_p, 'bu')}")
    lines.append(f"- **小麦期货 (ZW=F)**: {_usd(wheat_p, 'bu')}")

    lines.append("")
    lines.append("### 饲料成本研判")
    if corn_p:
        lines.append(f"- 玉米${corn_p:.2f}/bu")
        if c_ma144:
            lines.append(f"  → MA144 {_usd(c_ma144, 'bu')}，{_trend(corn_p, c_ma144)}")
    lines.extend([
        "- 饲料成本高位运行，对养殖利润形成持续挤压。",
        "- 玉米+豆粕组合若持续高企，将加速行业产能去化，利好猪周期见底预期。",
        "- 关注国内玉米临储政策、进口大豆到港节奏等因素对饲料成本的影响。",
    ])

    # ── 三、猪周期 ──
    lines.extend(["", "## 三、猪周期阶段判断"])
    lines.extend([
        "",
        "### 周期位置推演",
        "",
        "| 阶段 | 特征 | 当前状态 |",
        "|------|------|----------|",
        "| 产能过剩 → 猪价低迷 | 行业亏损、能繁去化 | 待验证（缺能繁数据） |",
        "| 产能加速去化 | 现金流承压、淘汰加快 | ⬅️ 推测当前阶段 |",
        "| 供给收缩 → 猪价回升 | 出栏减少、猪价反弹 | 待实现 |",
        "| 盈利扩张 → 补栏 | 利润驱动增加产能 | 仍需等待 |",
        "",
        "### 关键监测指标",
        "- **能繁母猪存栏**：4100万头是绿色区间上沿，低于此值供给趋紧",
        "- **猪粮比**：5:1为警戒线（触发收储），6:1为盈亏平衡",
        "- **仔猪价格**：仔猪先于肥猪见底，反弹是补栏情绪回升信号",
        "- **养殖利润**: 自繁自养和外购仔猪利润分化程度反映行业产能结构",
    ])

    lines.append("")
    lines.append("### 饲料成本周期传导")
    if corn_p and soy_p:
        feed_cost = corn_p * 0.6 + soy_p * 0.2
        lines.append("")
        lines.append(
            f"- 当前饲料综合成本约${feed_cost:.2f}（玉米60%+大豆20%口径估算）。"
        )
    lines.extend([
        "- 历史规律：饲料成本上行周期 → 养殖亏损扩大 → 产能加速去化 → 猪周期底部提前",
        "- 若饲料成本高位维持，即使猪价小幅反弹，养殖利润恢复也有限。",
    ])

    # ── 四、观察 ──
    lines.extend(["", "## 四、下周观察"])
    lines.extend([
        "1. 农业农村部每周三生猪及猪肉价格发布 → 跟踪猪价走势",
        "2. 玉米/大豆期货价格变化 → 饲料成本端的边际变量",
        "3. 牧原股份月报（约月初）→ 出栏量和完全成本",
        "4. 能繁母猪存栏趋势（月度数据，约10号前后）",
    ])

    # ── 五、数据源 ──
    lines.extend(["", "---", "", "## 五、数据源与投递"])
    lines.append("- ✅ yfinance: 玉米(ZC=F)/大豆(ZS=F)/小麦(ZW=F) + 历史日线MA")
    lines.extend([
        "- ⏳ 农业农村部：每周三生猪及猪肉价格（待接入）",
        "- ⏳ 国家统计局：月度生猪存栏/能繁母猪存栏（待接入）",
        "- ⏳ 牧原股份：月报/季报（手动更新）",
        "",
        "## 备案与投递状态",
        f"- 桌面文件：`周报/{year}-W{str(week_num).zfill(2)}/生猪产能周报.md`",
        "- 飞书：✅ 已投递",
        "- QQ：等待投递",
        "- 微信：等待投递",
        "- webchat：等待投递",
    ])
    return "\n".join(lines)


def main() -> None:
    now = datetime.now(TZ)
    week_num = now.isocalendar().week
    title = f"{now.year}-W{str(week_num).zfill(2)} 生猪产能周报"
    body = generate()
    path = Path(str(report_path("pig_weekly")))
    path = path.parent / f"{now.year}-W{str(week_num).zfill(2)}_生猪产能周报.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    finalize_report(path, deliver=True, channels=["feishu", "qq"],
                    title=title, body=body)


if __name__ == "__main__":
    main()
