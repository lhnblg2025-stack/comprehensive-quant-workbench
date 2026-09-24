"""
alert_dispatcher.py — 预警分发器 (V5)

核心问题：预警产生后如何送达？

支持多渠道分发：
  - 飞书webhook（交互式卡片消息）
  - 本地日志文件（便于审计/回溯）
  - 控制台输出（开发调试/CLI场景）

设计上渠道彼此独立，任一渠道失败不影响其他渠道投递，
批量分发时也保证单条失败不阻断整体流程。

对标：券商风控系统的多渠道告警分发 / PagerDuty 简化版
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

logger = __import__("logging").getLogger(__name__)

LOG_DIR = ROOT / "alert_engine" / "logs"
LOG_FILE = LOG_DIR / "alerts.log"

SEVERITY_COLOR = {
    "high": "red",
    "medium": "orange",
    "low": "grey",
}

# P1-Q21-fix(H04): 分发层去重窗口 / 风暴聚合阈值 / 飞书重试次数的默认参数
DEFAULT_DEDUP_WINDOW_SECONDS = 1800  # 30 分钟
DEFAULT_STORM_LIMIT = 5              # 同规则+同标的 单批超过该阈值 -> 聚合为 1 条
DEFAULT_FEISHU_MAX_RETRIES = 3       # 飞书失败重试次数（指数退避）

SEVERITY_ICON = {
    "high": "🔴",
    "medium": "🟡",
    "low": "⚪",
}


class AlertDispatcher:
    """预警分发器——把预警字典投递到飞书/日志/控制台等多个渠道。

    预警字典的推荐结构（各渠道对字段的容忍度较高，缺失字段会用默认值填充）:
        {
            "rule": "single_stock_risk",
            "severity": "high",
            "message": "单票600519占比18%超限",
            "code": "600519", "name": "贵州茅台",
            "timestamp": "2026-07-31T10:00:00+08:00",
        }

    Attributes:
        log_file: 本地日志文件路径
    """

    def __init__(self, log_file: Path | str | None = None,
                 dedup_window: int = DEFAULT_DEDUP_WINDOW_SECONDS,
                 storm_limit: int = DEFAULT_STORM_LIMIT,
                 feishu_max_retries: int = DEFAULT_FEISHU_MAX_RETRIES) -> None:
        self.log_file = Path(log_file) if log_file else LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # P1-Q21-fix(H04): 分发级去重/风暴抑制/飞书重试参数
        self.dedup_window = dedup_window
        self.storm_limit = storm_limit
        self.feishu_max_retries = feishu_max_retries
        # 内容指纹 -> 上次分发时间戳（epoch 秒）
        self._dispatched_fingerprints: dict[str, float] = {}

    def reset_dedup(self) -> None:
        """清空分发去重状态（测试或切换告警周期时调用）。"""
        self._dispatched_fingerprints.clear()

    @staticmethod
    def _fingerprint(alert: dict[str, Any]) -> str:
        """生成预警内容指纹（规则 + 标的 + 消息），用于去重判断。"""
        rule = alert.get("rule", alert.get("metric", "unknown"))
        code = alert.get("code", "")
        message = alert.get("message", "")
        return f"{rule}|{code}|{message}"

    # ────────────────────────────────────────────────────────
    # 渠道: 飞书
    # ────────────────────────────────────────────────────────

    def send_feishu(self, webhook_url: str, message: dict[str, Any]) -> bool:
        """发送飞书交互式卡片消息。

        Args:
            webhook_url: 飞书自定义机器人 webhook 地址
            message: 预警字典，用于构造卡片内容

        Returns:
            是否发送成功（HTTP 200 且飞书返回 code=0）
        """
        if not webhook_url:
            logger.warning("飞书webhook地址为空，跳过发送")
            return False
        try:
            import requests
        except ImportError:
            logger.error("发送飞书消息需要 requests 库，请先安装: pip install requests")
            return False

        # P1-Q21-fix(H04): 飞书渠道加重试+指数退避（网络抖动/东财限流时提高送达率）
        last_exc: Exception | None = None
        for attempt in range(1, self.feishu_max_retries + 1):
            try:
                payload = self._build_feishu_card(message)
                resp = requests.post(webhook_url, json=payload, timeout=10)
                if resp.status_code != 200:
                    logger.error("飞书webhook返回非200状态: %s", resp.status_code)
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                else:
                    body = resp.json()
                    if body.get("code", body.get("StatusCode", 0)) != 0:
                        logger.error("飞书webhook返回业务错误: %s", body)
                        last_exc = RuntimeError(f"biz_code={body.get('code', body.get('StatusCode'))}")
                    else:
                        return True
            except Exception as exc:
                logger.error("发送飞书消息异常(第%d次): %s", attempt, exc)
                last_exc = exc
            if attempt < self.feishu_max_retries:
                backoff = 2 ** attempt  # 2s, 4s, ... 指数退避
                logger.info("飞书发送失败，%.1fs 后重试...", backoff)
                time.sleep(backoff)
        if last_exc is not None:
            logger.error("飞书发送重试%d次仍失败: %s", self.feishu_max_retries, last_exc)
        return False

    @staticmethod
    def _build_feishu_card(alert: dict[str, Any]) -> dict[str, Any]:
        """把预警字典渲染成飞书 interactive 卡片消息体。"""
        severity = alert.get("severity", "low")
        icon = SEVERITY_ICON.get(severity, "⚪")
        color = SEVERITY_COLOR.get(severity, "grey")
        rule = alert.get("rule", alert.get("metric", "unknown"))
        message = alert.get("message", "")
        code = alert.get("code", "")
        name = alert.get("name", "")
        timestamp = alert.get("timestamp", datetime.now(CST).isoformat())

        title = f"{icon} 量化预警 [{severity.upper()}] {rule}"
        content_lines = [message]
        if code:
            content_lines.append(f"**标的**: {code} {name}".strip())
        content_lines.append(f"**时间**: {timestamp}")

        return {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color,
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": "\n".join(content_lines)},
                    }
                ],
            },
        }

    # ────────────────────────────────────────────────────────
    # 渠道: 日志
    # ────────────────────────────────────────────────────────

    def send_log(self, alert: dict[str, Any]) -> bool:
        """把预警以 JSON Lines 格式追加写入本地日志文件。

        P2-Q21-fix: 原实现失败在函数内部吞掉并返回 None，dispatch 对 log 渠道
        一律视为成功（日志丢失静默）。现返回 bool；写入失败时降级 stderr 输出，
        保证失败可见（不静默吞异常）。

        Args:
            alert: 预警字典

        Returns:
            是否写入成功
        """
        try:
            record = dict(alert)
            record.setdefault("logged_at", datetime.now(CST).isoformat())
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return True
        except Exception as exc:
            logger.error("写入预警日志失败: %s", exc)
            # 降级：日志写入失败时输出到 stderr，保证不静默丢失
            try:
                sys.stderr.write(
                    "⚠️ 预警日志写入失败: " + json.dumps(alert, ensure_ascii=False, default=str) + "\n"
                )
            except Exception as e:
                logger.error(f"[alert_dispatcher] 操作失败: {e}", exc_info=True)
            return False

    # ────────────────────────────────────────────────────────
    # 渠道: 控制台
    # ────────────────────────────────────────────────────────

    def send_console(self, alert: dict[str, Any]) -> None:
        """把预警以易读格式打印到控制台。

        Args:
            alert: 预警字典
        """
        severity = alert.get("severity", "low")
        icon = SEVERITY_ICON.get(severity, "⚪")
        rule = alert.get("rule", alert.get("metric", "unknown"))
        message = alert.get("message", "")
        timestamp = alert.get("timestamp", datetime.now(CST).isoformat())
        print(f"{icon} [{severity.upper():<6s}] {timestamp} {rule}: {message}")

    # ────────────────────────────────────────────────────────
    # 统一分发
    # ────────────────────────────────────────────────────────

    def dispatch(self, alert: dict[str, Any], channels: list[str],
                 feishu_webhook: str | None = None) -> bool:
        """把单条预警分发到指定渠道列表。

        Args:
            alert: 预警字典
            channels: 渠道列表，支持 "feishu" / "log" / "console"
            feishu_webhook: 飞书webhook地址，channels含"feishu"时必须提供

        Returns:
            是否所有渠道均分发成功（任一渠道失败返回 False，
            但不会阻断其他渠道的投递）；被去重窗口抑制时视为已处理返回 True
        """
        # P1-Q21-fix(H04): 分发级内容指纹去重（同一预警在去重窗口内不重复投递）
        now_ts = time.time()
        fp = self._fingerprint(alert)
        if now_ts - self._dispatched_fingerprints.get(fp, 0.0) < self.dedup_window:
            logger.info("去重: 预警指纹 %s 在去重窗口内，跳过分发", fp)
            return True
        self._dispatched_fingerprints[fp] = now_ts

        all_ok = True
        for channel in channels:
            try:
                if channel == "feishu":
                    if not feishu_webhook:
                        logger.warning("channels包含feishu但未提供feishu_webhook，跳过")
                        all_ok = False
                        continue
                    ok = self.send_feishu(feishu_webhook, alert)
                    all_ok = all_ok and ok
                elif channel == "log":
                    # P2-Q21-fix: send_log 返回 bool，纳入 all_ok——日志丢失不再静默
                    all_ok = all_ok and self.send_log(alert)
                elif channel == "console":
                    self.send_console(alert)
                else:
                    logger.warning("未知分发渠道: %s", channel)
                    all_ok = False
            except Exception as exc:
                logger.error("渠道 %s 分发失败: %s", channel, exc)
                all_ok = False
        return all_ok

    def batch_dispatch(self, alerts: list[dict[str, Any]], channels: list[str],
                        feishu_webhook: str | None = None) -> list[bool]:
        """批量分发多条预警，单条失败不影响其余预警的分发。

        Args:
            alerts: 预警字典列表
            channels: 渠道列表
            feishu_webhook: 飞书webhook地址

        Returns:
            与 alerts 等长的 bool 列表，标记每条预警是否全渠道分发成功
            （被去重/聚合降级处理的条目视为成功，不破坏长度契约）
        """
        if not alerts:
            return []
        # P1-Q21-fix(H04): 风暴聚合降级——同规则+同标的 单批超过 storm_limit 条时
        # 合并为 1 条摘要后分发；分发级去重由 dispatch 内部完成。
        emit_plan = self._aggregate_storm(alerts)
        results = [True] * len(alerts)
        for alert, idxs in emit_plan:
            try:
                result = self.dispatch(alert, channels, feishu_webhook=feishu_webhook)
            except Exception as exc:
                logger.error("批量分发中单条预警异常: %s", exc)
                result = False
            for i in idxs:
                results[i] = result
        return results

    def _aggregate_storm(self, alerts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[int]]]:
        """把同规则+同标的 超阈值条数的预警聚合为 1 条摘要。

        Returns:
            [(待分发预警, 对应原始下标列表), ...]，待分发预警数 <= 原条数，
            且所有原始下标恰好被覆盖一次（保证 batch_dispatch 的返回长度契约）。
        """
        groups: dict[tuple[str, str], list[int]] = {}
        for i, a in enumerate(alerts):
            rule = str(a.get("rule", a.get("metric", "unknown")))
            code = str(a.get("code", ""))
            groups.setdefault((rule, code), []).append(i)

        plan: list[tuple[dict[str, Any], list[int]]] = []
        for (rule, code), idxs in groups.items():
            if self.storm_limit > 0 and len(idxs) > self.storm_limit:
                first = alerts[idxs[0]]
                summary = dict(first)
                summary["message"] = (
                    f"[风暴聚合] 规则 {rule} 对标的 {code} 连续触发 {len(idxs)} 次，"
                    f"已合并为1条；首条消息: {first.get('message', '')}"
                )
                summary["aggregated"] = True
                summary["aggregated_count"] = len(idxs)
                plan.append((summary, idxs))
            else:
                for i in idxs:
                    plan.append((alerts[i], [i]))
        return plan


# ════════════════════════════════════════════════════════════════
# main — 测试入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """独立运行测试：验证日志/控制台渠道正常工作，飞书渠道在无webhook时优雅降级。"""
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_alerts.log"
            dispatcher = AlertDispatcher(log_file=log_path)

            print("=" * 60)
            print("AlertDispatcher 预警分发器 — 测试")
            print("=" * 60)

            alert = {
                "rule": "single_stock_risk",
                "severity": "high",
                "message": "单票600519占比18%超限",
                "code": "600519",
                "name": "贵州茅台",
                "timestamp": datetime.now(CST).isoformat(),
            }

            print("\n[控制台渠道测试]")
            dispatcher.send_console(alert)

            print("\n[日志渠道测试]")
            dispatcher.send_log(alert)
            assert log_path.exists(), "日志文件应已创建"
            content = log_path.read_text(encoding="utf-8")
            assert "single_stock_risk" in content, "日志内容应包含规则名"
            print(f"  日志已写入: {log_path} ({len(content)} 字节)")

            print("\n[飞书卡片渲染测试]")
            card = dispatcher._build_feishu_card(alert)
            assert card["msg_type"] == "interactive"
            assert "card" in card
            print(f"  卡片标题: {card['card']['header']['title']['content']}")

            print("\n[飞书发送降级测试（无webhook）]")
            ok = dispatcher.send_feishu("", alert)
            assert ok is False, "空webhook应返回False"
            print("  ✓ 空webhook正确返回False")

            print("\n[统一dispatch测试]")
            result = dispatcher.dispatch(alert, channels=["log", "console"])
            assert result is True, "log+console渠道应全部成功"

            print("\n[批量分发测试]")
            alerts = [alert, dict(alert, code="000001", name="平安银行")]
            results = dispatcher.batch_dispatch(alerts, channels=["log", "console"])
            assert results == [True, True]

        print("\n✅ AlertDispatcher 测试通过")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ AlertDispatcher 测试失败: {exc}")
        raise


if __name__ == "__main__":
    main()
