#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日复盘多渠道推送（2026-08-21）

飞书（feishu_sender 独立机器人，已验证）+ 微信（openclaw-weixin）+ QQ（openclaw-qqbot）。

优先级: feishu 必达; weixin/qqbot 尽力而为 + 失败告警不静默。
用法:
  python3 scripts/push_review.py [--date 2026-08-21] [--md 复盘内容]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("push_review")

# 渠道配置：通过环境变量注入，仓库中不保存任何真实账号或接收方标识。
WECHAT_ACCOUNT = os.environ.get("WECHAT_ACCOUNT", "")
WECHAT_TARGET = os.environ.get("WECHAT_TARGET", "")


def _load_review_md(date: str | None) -> str:
    """读 review_{date}.md；date 缺省取最新。"""
    gen = ROOT / "generated"
    if date:
        p = gen / f"review_{date}.md"
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")
    files = sorted(gen.glob("review_*.md"))
    if not files:
        return ""
    return files[-1].read_text(encoding="utf-8")


def push_feishu(md: str, date: str) -> bool:
    """飞书（send_markdown，独立机器人直发，已验证 True）。"""
    try:
        from feishu_sender import send_markdown  # noqa: PLC0415
        ok = send_markdown(f"📊 每日A股复盘 {date}\n\n" + md[:3500],
                           title=f"📊 每日A股复盘 {date}")
        logger.info("飞书推送: %s", ok)
        return ok
    except Exception as e:  # noqa: BLE001
        logger.error("飞书推送异常: %s", str(e)[:100])
        return False


def _openclaw_send(channel: str, target: str, message: str,
                   account: str | None = None) -> tuple[bool, str]:
    """openclaw message send 通用。返回 (ok, error)。"""
    cmd = ["openclaw", "message", "send", "--channel", channel, "-t", target,
           "-m", message, "--json"]
    if account:
        cmd += ["--account", account]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = (r.stdout or "") + (r.stderr or "")
        ok = r.returncode == 0 and "Error" not in out and "error" not in out.lower()[:300]
        if r.returncode == 0 and "outbound-delivery-error" not in out.lower():
            return True, out[:80]
        return False, out[:120]
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:80]


def push_wechat(md: str, date: str) -> bool:
    """微信（openclaw-weixin）。im.bot 网关故障时记录失败。"""
    msg = f"📊 每日A股复盘 {date}\n\n" + md[:1500] + "\n…(详见 localhost:8600/review.html)"
    ok, err = _openclaw_send("openclaw-weixin", WECHAT_TARGET, msg, WECHAT_ACCOUNT)
    logger.info("微信推送: %s %s", ok, err[:60] if not ok else "")
    return ok


def push_qqbot(md: str, date: str) -> bool:
    """QQ（openclaw-qqbot）。插件 contract 状态决定可用性。"""
    msg = f"📊 每日A股复盘 {date}\n\n" + md[:1500]
    # qqbot target 需 QQ 账号/群；先用 account 默认
    ok, err = _openclaw_send("qqbot", "", msg)  # target 由 account 决定
    logger.info("QQ推送: %s %s", ok, err[:60] if not ok else "")
    return ok


def push_all(md: str, date: str) -> dict:
    """全渠道推送，返回各渠道结果。"""
    results = {
        "feishu": push_feishu(md, date),
        "wechat": push_wechat(md, date),
        "qqbot": push_qqbot(md, date),
    }
    summary = "、".join(f"{k}={'✅' if v else '❌'}" for k, v in results.items())
    logger.info("推送汇总: %s", summary)
    # 汇总结果落盘（供 cron 断言）
    (ROOT / "generated" / "push_result.json").write_text(
        json.dumps({"date": date, "results": results, "ts": __import__("time").time()},
                   ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--channel", default="feishu", choices=["feishu", "wechat", "qqbot", "all"],
                    help="推送渠道（默认 feishu；all=三渠道都试）")
    args = ap.parse_args()
    md = _load_review_md(args.date)
    if not md:
        logger.error("无复盘 md（需先跑 daily_review_chain.py）")
        sys.exit(1)
    date = args.date or ([f.name[7:-3] for f in sorted((ROOT / "generated").glob("review_*.md"))][-1] if list((ROOT / "generated").glob("review_*.md")) else "?")
    if args.channel == "all":
        res = push_all(md, date)
        sys.exit(0 if res.get("feishu") else 1)
    elif args.channel == "wechat":
        sys.exit(0 if push_wechat(md, date) else 1)
    elif args.channel == "qqbot":
        sys.exit(0 if push_qqbot(md, date) else 1)
    else:
        sys.exit(0 if push_feishu(md, date) else 1)