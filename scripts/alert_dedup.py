#!/usr/bin/env python3
"""
alert_dedup.py — 告警去重网关（飞书刷屏治理统一入口，2026-08-14）
================================================================
背景（审计结论）:
  - health_guardian 每5分钟无差别发"系统健康异常" → 持续故障时每5分钟刷屏
  - opportunity_monitor 盘中16轮，无机会也推"确认"消息
  - risk_monitor 每次运行无风险也推
  - intraday_monitor Tier2/Tier3 无冷却重复推
  - generated/alert_state.json 已无任何存活消费者（去重层随 quant_scan_alert 归档丢失）

设计:
  1. 状态文件 generated/alert_dedup_state.json（本机/云端各自独立落盘）
  2. should_send(channel, sig, level, same_sig_min_gap_s):
     - 新签名 → 立即放行
     - 同签名在 same_sig_min_gap_s 内 → 静默；到期重发一次（=仍在持续的提醒）
  3. 每 channel 每小时发送上限 rate_cap（默认10条）——硬顶
  4. 夜间静默: 23:00-08:00 只放行 level == "danger"
  5. 环境开关 QUANT_ALERT_SILENT=1 全局禁发（联调/测试用）

调用方约定: sig 必须剔除价格等易变字段（如 f"{sym}|{reason}" 而非含价格），
否则去重失效。本模块提供 normalize_price() 辅助。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))  # 全库统一 CST（Asia/Shanghai）
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except Exception:  # pragma: no cover
    ROOT = Path.cwd()

STATE_FILE = ROOT / "generated" / "alert_dedup_state.json"
NIGHT_START, NIGHT_END = 23, 8  # 夜间静默窗口 [23:00, 08:00)
DEFAULT_SAME_SIG_GAP_S = 2 * 3600  # 同签名默认最小重发间隔
DEFAULT_RATE_CAP = 10  # 每 channel 每小时上限

_PRICE_RE = re.compile(
    r"\d+\.\d+%?"                    # 带小数的价格/百分比（38.37 / 2.5%）
    r"|\d+(?:\.\d+)?(?:元|亿|%)"      # 带单位的金额/百分比（12.3亿 / 27元 / 3%）
    r"|\b\d{1,4}\b"                   # 1-4 位裸整数价格（27 / 921），不含 6 位代码和 8 位日期
)


def normalize_price(text: str) -> str:
    """剔除价格/百分比/金额等易变数字，让同信号不同价格归一为同一 sig。"""
    return _PRICE_RE.sub("#", text)


def make_sig(*parts: str) -> str:
    """用 md5 前 12 位构建去重签名（内容变化 → 签名变化）。"""
    joined = "|".join(parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


class AlertDeduper:
    """进程安全的告警去重器。单例由 get_deduper() 提供。"""

    def __init__(self, state_file: Path | None = None,
                 rate_cap: int = DEFAULT_RATE_CAP) -> None:
        self._lock = threading.Lock()
        self._state_file = state_file or STATE_FILE
        self._rate_cap = rate_cap
        self._mem: dict = {"channels": {}, "hourly": {}}

    # ---- 状态读写 ----
    def _load(self) -> dict:
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:  # noqa: BLE001 - 损坏则重建
                pass
        return {"version": 2, "channels": {}, "hourly": {}}

    def _save(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._mem, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self._state_file)
        except Exception as e:  # noqa: BLE001 - 状态落盘失败不阻塞告警
            print(f"[alert_dedup] 状态保存失败: {e}")

    def _now_hour(self, now: datetime) -> str:
        return now.strftime("%Y-%m-%dT%H")

    def _is_night(self, now: datetime) -> bool:
        return now.hour >= NIGHT_START or now.hour < NIGHT_END

    def should_send(self, channel: str, sig: str, level: str = "info",
                    same_sig_min_gap_s: int = DEFAULT_SAME_SIG_GAP_S,
                    now: datetime | None = None) -> bool:
        """判断是否允许发送。

        channel:            告警来源（health_guardian/opportunity/risk/intraday/...）
        sig:                内容签名（剔除价格等易变字段；make_sig 生成）
        level:              info/warn/danger —— 夜间只放行 danger
        same_sig_min_gap_s: 同一签名的最小重发间隔（新签名不受限，立即放行）

        规则: 新签名→立即发；同签名→间隔未到则静默（到点重发一次=仍在持续的提醒）；
              每 channel 每小时硬顶 rate_cap 条；夜间(23-8点)只放行 danger。
        """
        if os.environ.get("QUANT_ALERT_SILENT") == "1":
            return False
        now = now or datetime.now(CST)
        # 2026-08-14 修复: 调用方可能传 naive datetime(测试/旧代码),
        # 与状态里的 CST aware 时间相减会 TypeError。naive 统一按 CST 解释。
        if now.tzinfo is None:
            now = now.replace(tzinfo=CST)
        with self._lock:
            self._mem = self._load()
            channels = self._mem.setdefault("channels", {})
            hourly = self._mem.setdefault("hourly", {})

            hour_key = f"{channel}:{self._now_hour(now)}"
            sent_this_hour = hourly.get(hour_key, 0)
            if sent_this_hour >= self._rate_cap:
                return False

            # 夜间静默（danger 除外）
            if self._is_night(now) and level != "danger":
                return False

            prev = channels.get(channel, {})
            last_sig = prev.get("sig")
            last_ts = prev.get("ts", 0.0)
            # 2026-08-14 修复: fromtimestamp naive 与 now(CST aware) 相减
            # → TypeError(offset-naive vs aware) 导致去重失效直发兜底刷屏
            last_dt = (datetime.fromtimestamp(last_ts, CST)
                       if last_ts else None)

            if last_sig == sig and last_dt is not None:
                age = (now - last_dt).total_seconds()
                if age < max(same_sig_min_gap_s, 1):
                    return False

            channels[channel] = {"sig": sig, "ts": now.timestamp()}
            hourly[hour_key] = sent_this_hour + 1
            self._save()
            return True


_instance: AlertDeduper | None = None
_instance_lock = threading.Lock()


def get_deduper() -> AlertDeduper:
    """全局单例。"""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = AlertDeduper()
        return _instance


def reset_instance() -> None:
    """测试用：重置单例。"""
    global _instance
    with _instance_lock:
        _instance = None
