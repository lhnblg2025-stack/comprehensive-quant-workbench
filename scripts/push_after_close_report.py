#!/usr/bin/env python3
"""
push_after_close_report.py — 盘后飞书推送 (2026-08-14)
======================================================
用户硬诉求: 盘后必须含 龙虎榜短线 + 中长线低估提醒, 且要能看到(推送)。
pipeline --daily 生成产物但从不推送 → 用户"飞书都没看到了"。

本脚本组装盘后摘要推飞书(经 alert_dedup 每日一次防重复):
  1. after_close_extra_{date}.json  → 龙虎榜游资/个股净买 Top + 中长线低估池
  2. short_term_daily.md       → 决策卡摘要(总基调/仓位/机会榜)
  3. battle_map_{date}.md          → 行动卡(攻击方向/风险清单)

用法:
  python3 scripts/push_after_close_report.py [--date YYYY-MM-DD] [--dry-run]
  (云端 schtasks quant_push_after_close 19:05, 在 quant_pipeline_daily 18:50 之后)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CST = timezone(timedelta(hours=8))


def _today() -> str:
    """取最近交易日：优先统一交易日历（market_clock.latest_trading_day），失败回退日历日。

    审计修复(2026-08-24)：周末/非交易日运行不得用日历今天当交易日，避免把
    8/22、8/23 这种运行日伪装成新交易日。
    """
    try:
        sys.path.insert(0, str(ROOT / "quant_system"))
        from quant_system.market_clock import latest_trading_day  # noqa: PLC0415
        day = latest_trading_day()
        if day:
            return day
    except Exception:  # noqa: BLE001
        pass
    return datetime.now(CST).strftime("%Y-%m-%d")


def _read_json(name: str) -> dict:
    p = ROOT / "generated" / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _read_md(name: str) -> str:
    p = ROOT / "generated" / name
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _receipt_path(date: str) -> Path:
    out = ROOT / "generated" / "delivery_receipts"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"after_close_feishu_{date}.json"


def _funding_lines(snapshot: dict, date: str) -> list[str]:
    """Use the same dated decision bundle as the HTML, never legacy text fragments."""
    money = snapshot.get("money_flow") or {}
    etf = snapshot.get("etf_activity") or {}
    source_dates = snapshot.get("source_dates") or {}
    intraday = _read_json(f"intraday_flow_history_{date}.json")
    def item_name(item: dict) -> str:
        return str(item.get("name") or item.get("concept_name") or item.get("sector_name") or item.get("code") or "-")
    def item_value(item: dict) -> float | None:
        for key in ("net_yi", "main_net", "net", "net_amount", "amount"):
            try:
                if item.get(key) is not None:
                    return float(item[key])
            except (TypeError, ValueError):
                continue
        return None
    top_in = [f"{item_name(item)} {item_value(item):+.1f}亿" for item in (money.get("top_in") or [])[:5] if item_value(item) is not None]
    top_out = [f"{item_name(item)} {item_value(item):+.1f}亿" for item in (money.get("top_out") or [])[:5] if item_value(item) is not None]
    etf_top = [item_name(item) for item in (etf.get("top_amount") or [])[:4]]
    lines = ["## 资金与结构"]
    lines.append(f"- 日频行业/概念资金实际日期：{source_dates.get('sector_fund_flow') or '缺失'}；流入：{'、'.join(top_in) or '无可用记录'}")
    lines.append(f"- 流出：{'、'.join(top_out) or '无可用记录'}")
    lines.append(f"- ETF成交关注：{'、'.join(etf_top) or '无可用记录'}")
    if intraday:
        lines.append(f"- 盘中回放：{intraday.get('date')}，快照 {intraday.get('snapshot_count', 0)} 次；市场覆盖{'完整' if intraday.get('market_complete_day') else '部分'}，ETF覆盖{'完整' if intraday.get('etf_complete_day') else '部分/缺失'}；成交额方向代理不等于真实主力净流入。")
    else:
        lines.append("- 盘中回放：当日产物缺失，不以日频资金替代盘中结论。")
    return lines


def build_push_text(date: str | None = None) -> str:
    """Render the Feishu projection from one immutable, same-date decision bundle."""
    date = date or _today()
    snapshot = _read_json(f"decision_snapshot_after_close_{date}.json")
    if not snapshot or snapshot.get("as_of") != date or snapshot.get("mode") != "after_close":
        raise ValueError(f"统一盘后决策快照不可用或日期不一致: {date}")
    from sync_fusion_workbench import build_feishu  # noqa: PLC0415
    lines = [build_feishu(snapshot), "", *_funding_lines(snapshot, date), "", "## 产物与回放"]
    lines.append(f"- HTML研报：研报共享/{date}/综合详细研报_{date}.html")
    lines.append(f"- 资金回放：工作台 > 市场数据 > 资金与结构 > {date}")
    lines.append("- 本消息只投影同日统一决策包；数据缺失、日期错位或门控未通过会在正文显式呈现。")
    return "\n".join(lines)


def _read_receipt(date: str) -> dict:
    try:
        return json.loads(_receipt_path(date).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_receipt(date: str, text: str, status: str, error: str | None = None) -> None:
    payload = {
        "date": date, "channel": "feishu", "status": status,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "attempted_at": datetime.now(CST).isoformat(timespec="seconds"),
        "error": error,
    }
    _receipt_path(date).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 统一账本登记：盘后投递回执追加到 delivery_receipts/index.json。
    try:
        from execution_ledger import record_delivery  # noqa: PLC0415
        record_delivery(delivery_id=f"after_close_feishu:{date}", record=payload)
    except Exception:  # noqa: BLE001 - 账本失败不阻断推送主链
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="盘后飞书推送")
    ap.add_argument("--date", default=None, help="日期 YYYY-MM-DD（默认今天）")
    ap.add_argument("--dry-run", action="store_true", help="仅输出不推送")
    args = ap.parse_args()
    date = args.date or _today()
    try:
        text = build_push_text(date)
    except Exception as exc:
        _write_receipt(date, "", "blocked", str(exc))
        print(f"[push_after_close] 报告包不合格: {exc}", flush=True)
        return 2

    if args.dry_run:
        print(text)
        return 0
    existing = _read_receipt(date)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if existing.get("status") == "acknowledged" and existing.get("content_sha256") == content_hash:
        print("[push_after_close] 同一报告包已确认投递，跳过", flush=True)
        return 0

    from feishu_sender import send_markdown
    ok = send_markdown(text, title=f"📈 盘后决策日报 {date}")
    if ok:
        _write_receipt(date, text, "acknowledged")
        print("飞书推送: OK（receipt=acknowledged）", flush=True)
        return 0
    _write_receipt(date, text, "failed", "feishu_sender returned false")
    print("飞书推送: 失败（receipt=failed，允许下次重试）", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
