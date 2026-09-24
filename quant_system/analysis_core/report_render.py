"""
report_render — 日报渲染（V11 交付层：PNG 长图）

把 V11 日报关键数据渲染成手机可读的 PNG 长图（对齐示例风格）。
字体: Noto Sans CJK（本机已装）。

用法:
  python3 -m quant_system.analysis_core.report_render --daily    # 生成短线日报长图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.analysis_core.emotion_cycle import run_today  # noqa: E402
from quant_system.analysis_core import decision_card, fusion  # noqa: E402

FONT = None
for f in fm.fontManager.ttflist:
    if "Noto Sans CJK" in f.name and "JP" not in f.name or f.name == "Noto Sans CJK SC":
        FONT = f.name
        break
if FONT is None:
    FONT = "Noto Sans CJK JP"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False


def render_daily_png(out: Path | None = None) -> Path:
    emo = run_today()
    card = decision_card.build_card()
    fu = fusion.fuse_today()
    stats = pd.read_parquet(MARKET_DIR / "zt_daily_stats.parquet").tail(10)

    fig, axs = plt.subplots(4, 1, figsize=(6.4, 14), facecolor="#0d1117")
    fig.suptitle(f"📊 V11 短线全景 {emo['date']}", color="#e6edf3", fontsize=16, fontweight="bold")

    # 1. 情绪周期 + 融合温度
    ax = axs[0]; ax.set_facecolor("#161b22")
    ax.text(0.02, 0.85, f"情绪周期: {emo['stage_cn']}  置信 {emo['confidence']}",
            transform=ax.transAxes, color="#58a6ff", fontsize=14)
    ax.text(0.02, 0.6, f"市场温度: {fu['temperature']}/100  {fu['tag']}",
            transform=ax.transAxes, color="#e6edf3", fontsize=13)
    ax.text(0.02, 0.35, f"涨停 {emo['zt_cnt']} | 最高 {emo['max_board']}板 | 炸板率 {emo['zb_rate']:.0%}",
            transform=ax.transAxes, color="#8b949e", fontsize=12)
    ax.axis("off")

    # 2. 连板天梯（近10日涨停家数 + 高度）
    ax = axs[1]; ax.set_facecolor("#161b22")
    ax.plot(stats["date"], stats["zt_cnt"], color="#3fb950", lw=2, label="涨停家数")
    ax.plot(stats["date"], stats["max_board"], color="#d29922", lw=2, label="最高板")
    ax.fill_between(stats["date"], stats["zt_cnt"], alpha=0.15, color="#3fb950")
    ax.set_title("连板天梯（近10日）", color="#e6edf3", fontsize=13)
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    ax.tick_params(colors="#8b949e"); ax.spines[:].set_color("#30363d")

    # 3. 次日推演三情景
    ax = axs[2]; ax.set_facecolor("#161b22")
    try:
        from quant_system.analysis_core.scenario import scenario_forecast
        sc = scenario_forecast()["scenarios"]
        labels = list(sc.keys()); probs = [sc[k]["prob"] for k in labels]
        colors = ["#3fb950", "#d29922", "#f85149"]
        bars = ax.bar(labels, probs, color=colors, alpha=0.85)
        for b, p in zip(bars, probs):
            ax.text(b.get_x() + b.get_width() / 2, p + 0.01, f"{p:.0%}",
                    ha="center", color="#e6edf3", fontsize=12)
        ax.set_ylim(0, 1); ax.set_title("次日推演（三情景概率）", color="#e6edf3", fontsize=13)
        ax.tick_params(colors="#8b949e"); ax.spines[:].set_color("#30363d")
    except Exception:
        ax.text(0.5, 0.5, "推演数据不可用", color="#f85149", ha="center", transform=ax.transAxes)

    # 4. 决策卡片
    ax = axs[3]; ax.set_facecolor("#161b22")
    ax.text(0.02, 0.82, f"总基调: {card['stance']}  仓位: {card['position_range']}",
            transform=ax.transAxes, color="#e6edf3", fontsize=13)
    opp = card["opportunities"][0] if card["opportunities"] else {}
    if opp:
        ax.text(0.02, 0.58, f"🎯 {opp['target']}", transform=ax.transAxes, color="#3fb950", fontsize=12)
        ax.text(0.02, 0.38, f"   买入: {opp['buy_cond']}", transform=ax.transAxes, color="#8b949e", fontsize=11)
        ax.text(0.02, 0.18, f"   止损: {opp['stop']}", transform=ax.transAxes, color="#f85149", fontsize=11)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if out is None:
        out = ROOT / "generated" / "daily_long.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="#0d1117")
    plt.close(fig)
    print(f"[render] PNG 已生成 → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="日报 PNG 渲染")
    ap.add_argument("--daily", action="store_true")
    args = ap.parse_args()
    if args.daily:
        render_daily_png()
    else:
        ap.print_help()
