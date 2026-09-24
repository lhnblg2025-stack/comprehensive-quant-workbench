#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chaos_drill.py — 三断混沌演练（dry-run 安全模式 + RTO/RPO 记录）

V12.1 阶段3：把"断链等告警"改成主动免疫。每周日低波动时段运行：
  1. 断源：模拟数据源不可用，验证 primary → fallback → unavailable 降级链。
  2. 断库：模拟核心 parquet 损坏，验证 read 降级返回 None/插补且不崩溃。
  3. 断网：模拟网络超时，验证 retry → backoff → degrade 处理链。

安全约束：
  - 默认且仅支持 dry-run，所有故障均通过 callable 注入，绝不真实断网/删文件。
  - 输出到 generated/chaos_drill/YYYYMMDD/chaos_{date}.json|md。
  - 全部通过打印"免疫正常"并退出 0；任一失败退出 1 并复用 feishu_sender 告警。

用法:
  python3 scripts/chaos_drill.py
  python3 scripts/chaos_drill.py --date 2026-08-16 --only db
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "generated" / "chaos_drill"
CST = timezone(timedelta(hours=8))

DRILL_LABELS = {
    "source": "断源",
    "db": "断库",
    "net": "断网",
}
VALID_DRILLS = tuple(DRILL_LABELS)
DRY_RUN_BACKOFF_SECONDS = (0.01, 0.02)
DRY_RUN_MAX_RETRIES = 3
_SENTINEL = object()


# ────────────────────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class DrillResult:
    """单项演练结果。rto_ms 即本次演练从注入到完成降级的耗时。"""

    key: str
    name: str
    passed: bool
    elapsed_ms: int
    rto_ms: int
    path: str
    rpo: dict[str, Any]
    detail: dict[str, Any]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────────────
# dry-run 默认 mock 故障
# ────────────────────────────────────────────────────────────────────────────


def _mock_source_primary_failure(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise ConnectionError("mock: primary data source unavailable")


def _mock_source_fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"source": "fallback", "rows": 42, "lag_seconds": 0}


def _mock_corrupt_parquet(*args: Any, **kwargs: Any) -> Any:
    raise OSError("mock: core parquet corruption")


def _mock_impute_frame(error: Exception) -> list[dict[str, Any]]:
    return [{"symbol": "MOCK", "close": None, "imputed": True}]


def _mock_network_timeout(*args: Any, **kwargs: Any) -> Any:
    raise TimeoutError("mock: network timeout")


def _mock_network_degraded(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"source": "cache", "rows": 7, "lag_seconds": 30}


def _mock_backoff(delay: float) -> None:
    time.sleep(delay)


def _noop_backoff(delay: float) -> None:
    return None


# ────────────────────────────────────────────────────────────────────────────
# 公共 helpers
# ────────────────────────────────────────────────────────────────────────────


def _pick(value: Any, default: Callable[..., Any]) -> Any:
    return default if value is _SENTINEL else value


def _usable(value: Any) -> bool:
    """None/空容器视为不可用，模拟降级判空。"""
    if value is None:
        return False
    empty = getattr(value, "empty", None)
    if empty is True:
        return False
    try:
        return len(value) > 0
    except Exception:
        return True


def _row_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except Exception:
        return 0


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _result(
    key: str,
    start: float,
    passed: bool,
    path: str,
    rpo: dict[str, Any],
    detail: dict[str, Any],
    error: str = "",
) -> DrillResult:
    elapsed = _elapsed_ms(start)
    return DrillResult(
        key=key,
        name=DRILL_LABELS[key],
        passed=passed,
        elapsed_ms=elapsed,
        rto_ms=elapsed,
        path=path,
        rpo=rpo,
        detail=detail,
        error=error,
    )


# ────────────────────────────────────────────────────────────────────────────
# 三项演练
# ────────────────────────────────────────────────────────────────────────────


def run_source_drill(
    primary_fetcher: Callable[..., Any] = _SENTINEL,
    fallback_fetcher: Callable[..., Any] = _SENTINEL,
) -> DrillResult:
    """断源演练：primary 不可用时切换 fallback，双双失败则 unavailable 标注。"""
    start = time.perf_counter()
    primary = _pick(primary_fetcher, _mock_source_primary_failure)
    fallback = _pick(fallback_fetcher, _mock_source_fallback)
    chain = ["primary", "fallback", "unavailable"]
    active: list[str] = []
    errors: list[str] = []

    data: Any = None
    try:
        data = primary()
    except Exception as exc:  # noqa: BLE001 — 演练必须吞掉注入异常并按降级链继续
        errors.append(f"primary: {exc}")

    if _usable(data):
        active = ["primary"]
        rpo = {"type": "source_ok", "value": 0, "note": "主源可用，无数据丢失"}
    else:
        active = ["primary", "fallback"]
        try:
            data = fallback()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"fallback: {exc}")

        if _usable(data):
            lag = data.get("lag_seconds", 0) if isinstance(data, dict) else 0
            rpo = {
                "type": "source_outage",
                "value": lag,
                "note": "主源不可用，已切 fallback；数据丢失窗口按 fallback 快照滞后记录",
            }
        else:
            active.append("unavailable")
            rpo = {
                "type": "full_outage",
                "value": None,
                "note": "主源与 fallback 均不可用，数据不可得",
            }

    detail = {
        "degradation_chain": chain,
        "active_chain": active,
        "errors": errors,
    }
    return _result(
        "source",
        start,
        passed=active[-1] != "unavailable",
        path=" -> ".join(active),
        rpo=rpo,
        detail=detail,
        error="; ".join(errors) if active[-1] == "unavailable" else "",
    )


def run_db_drill(
    read_frame: Callable[..., Any] = _SENTINEL,
    impute_frame: Callable[[Exception], Any] = _SENTINEL,
) -> DrillResult:
    """断库演练：parquet 损坏时 read 降级返回 None/插补，不允许进程崩溃。"""
    start = time.perf_counter()
    read_frame = _pick(read_frame, _mock_corrupt_parquet)
    impute_frame = _pick(impute_frame, _mock_impute_frame)
    chain = ["parquet", "read_degraded", "impute", "unavailable"]
    active: list[str] = []
    errors: list[str] = []

    read_error: Exception | None = None
    try:
        data = read_frame()
    except Exception as exc:  # noqa: BLE001
        read_error = exc
        errors.append(f"read: {exc}")
        data = None

    if _usable(data):
        active = ["parquet"]
        rpo = {"type": "none", "value": 0, "imputed_rows": 0, "note": "parquet 读取正常，无丢失"}
    else:
        active = ["parquet", "read_degraded"]
        imputed: Any = None
        impute_failed = False
        try:
            imputed = impute_frame(read_error if read_error is not None else "")
        except Exception as exc:  # noqa: BLE001
            impute_failed = True
            errors.append(f"impute: {exc}")

        if impute_failed:
            active.append("unavailable")
            rpo = {"type": "full_outage", "value": None, "imputed_rows": 0, "note": "读取与插补均不可用"}
        elif imputed is None:
            rpo = {
                "type": "none_fallback",
                "value": None,
                "imputed_rows": 0,
                "mode": "none",
                "note": "损坏读取已降级为 None 返回，进程未崩溃",
            }
        elif _usable(imputed):
            active.append("impute")
            rpo = {
                "type": "impute",
                "value": _row_count(imputed),
                "imputed_rows": _row_count(imputed),
                "mode": "impute",
                "note": "损坏读取已插补，不崩溃",
            }
        else:
            active.append("impute")
            rpo = {
                "type": "empty_impute",
                "value": 0,
                "imputed_rows": 0,
                "mode": "empty",
                "note": "插补返回空数据，未崩溃",
            }

    detail = {
        "degradation_chain": chain,
        "active_chain": active,
        "errors": errors,
    }
    return _result(
        "db",
        start,
        passed=active[-1] != "unavailable",
        path=" -> ".join(active),
        rpo=rpo,
        detail=detail,
        error="; ".join(errors) if active[-1] == "unavailable" else "",
    )


def run_net_drill(
    fetcher: Callable[..., Any] = _SENTINEL,
    degraded_fetcher: Callable[..., Any] = _SENTINEL,
    backoff: Callable[[float], None] = _SENTINEL,
    delays: Sequence[float] = DRY_RUN_BACKOFF_SECONDS,
    max_retries: int = DRY_RUN_MAX_RETRIES,
) -> DrillResult:
    """断网演练：超时后重试，每次失败退避；重试耗尽后走降级源。"""
    start = time.perf_counter()
    fetcher = _pick(fetcher, _mock_network_timeout)
    degraded_fetcher = _pick(degraded_fetcher, _mock_network_degraded)
    backoff_fn = _mock_backoff if backoff is _SENTINEL else (backoff or _noop_backoff)
    delays = tuple(delays or ())
    chain = ["request", "retry", "backoff", "degrade", "unavailable"]
    active: list[str] = []
    errors: list[str] = []
    scheduled_delays: list[float] = []
    retry_count = 0

    data: Any = None
    for attempt in range(max_retries):
        try:
            data = fetcher()
        except Exception as exc:  # noqa: BLE001
            retry_count += 1
            errors.append(f"request({attempt + 1}): {exc}")
            if attempt < max_retries - 1 and delays:
                delay = delays[min(attempt, len(delays) - 1)]
                scheduled_delays.append(delay)
                backoff_fn(delay)
            continue
        if _usable(data):
            break

    if _usable(data):
        active = ["request", "retry", "success"]
        path = f"request -> retry({retry_count}) -> success"
        rpo = {"type": "none", "value": 0, "note": "网络请求重试后成功，无数据丢失"}
    else:
        active = ["request", "retry", "backoff"]
        degraded: Any = None
        try:
            degraded = degraded_fetcher()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"degrade: {exc}")

        if _usable(degraded):
            active.append("degrade")
            lag = degraded.get("lag_seconds", 30) if isinstance(degraded, dict) else 30
            path = (
                f"request -> retry({retry_count}) -> "
                f"backoff({len(scheduled_delays)}) -> degrade"
            )
            rpo = {
                "type": "cache_lag",
                "value": lag,
                "note": "网络重试耗尽后使用降级缓存；RPO 按缓存滞后记录",
            }
        else:
            active.append("unavailable")
            path = " -> ".join(active)
            rpo = {"type": "full_outage", "value": None, "note": "网络与降级源均不可用"}

    detail = {
        "degradation_chain": chain,
        "active_chain": active,
        "retry_count": retry_count,
        "backoff_calls": scheduled_delays,
        "errors": errors,
    }
    return _result(
        "net",
        start,
        passed=active[-1] != "unavailable",
        path=path,
        rpo=rpo,
        detail=detail,
        error="; ".join(errors) if active[-1] == "unavailable" else "",
    )


# ────────────────────────────────────────────────────────────────────────────
# 汇总、落盘、告警、CLI
# ────────────────────────────────────────────────────────────────────────────


def select_drills(only: str | None = None) -> list[str]:
    if only is None:
        return list(VALID_DRILLS)
    if only not in VALID_DRILLS:
        raise ValueError(f"unknown drill: {only!r}; must be one of {VALID_DRILLS}")
    return [only]


def build_report(date_str: str, results: list[DrillResult]) -> dict[str, Any]:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    all_passed = failed == 0
    message = "免疫正常" if all_passed else f"存在失败演练: {failed}/{len(results)}"
    drills = [{**r.to_dict(), "dry_run": True} for r in results]
    return {
        "date": date_str,
        "generated_at": datetime.now(CST).isoformat(),
        "dry_run": True,
        "mode": "mock_fault_injection",
        "summary": {
            "all_passed": all_passed,
            "passed": passed,
            "failed": failed,
            "message": message,
        },
        "drills": drills,
    }


def run_all(
    date_str: str,
    only: str | None = None,
    output_root: Path | None = None,
    injections: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """运行选定演练；output_root 非 None 时同时写 JSON/MD 记录。"""
    datetime.strptime(date_str, "%Y-%m-%d")
    injections = injections or {}
    runners = {
        "source": run_source_drill,
        "db": run_db_drill,
        "net": run_net_drill,
    }
    results: list[DrillResult] = []
    for key in select_drills(only):
        kwargs = injections.get(key, {})
        results.append(runners[key](**kwargs))

    report = build_report(date_str, results)
    if output_root is not None:
        write_reports(report, output_root)
    return report


def write_reports(report: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    """写入 generated/chaos_drill/YYYYMMDD/chaos_{date}.json|md。"""
    date_str = report["date"]
    folder = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    out_dir = Path(output_root) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"chaos_{date_str}.json"
    md_path = out_dir / f"chaos_{date_str}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 混沌演练 {report['date']}",
        "",
        f"- 生成时间: {report.get('generated_at', '-')}",
        f"- 模式: {report.get('mode', '-')}（dry-run）",
        f"- 总结: {report['summary']['message']}",
        "",
        "## 结果",
        "",
        "| 项目 | 结果 | 耗时(ms) | 降级路径 | RTO(ms) | RPO | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for drill in report["drills"]:
        status = "通过" if drill["passed"] else "失败"
        rpo = json.dumps(drill["rpo"], ensure_ascii=False)
        error = drill.get("error") or "-"
        lines.append(
            f"| {drill['name']} | {status} | {drill['elapsed_ms']} | "
            f"{drill['path']} | {drill['rto_ms']} | {rpo} | {error} |"
        )
    return "\n".join(lines) + "\n"


def _send_feishu_text(text: str, title: str | None = None) -> bool:
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from feishu_sender import send_simple_text  # noqa: PLC0415

    return send_simple_text(text, title=title)


def send_failure_alert(report: dict[str, Any]) -> None:
    """任一演练失败时复用飞书 sender 发告警；发送失败不阻断退出码。"""
    failed = [d for d in report.get("drills", []) if not d.get("passed")]
    lines = [
        f"混沌演练失败: {report.get('date', '-')}",
        report.get("summary", {}).get("message", "存在失败演练"),
    ]
    for d in failed:
        lines.append(f"- {d.get('name', d.get('key'))}: {d.get('path', '-')}（{d.get('error') or '无错误详情'}）")
    text = "\n".join(lines)
    try:
        _send_feishu_text(text, title="⚠️ Chaos Drill 告警")
    except Exception as exc:  # noqa: BLE001
        print(f"[chaos-drill] 飞书告警失败: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三断混沌演练（dry-run 安全模式）")
    parser.add_argument("--date", help="演练日期 YYYY-MM-DD，默认今天(+08:00)")
    parser.add_argument("--only", choices=VALID_DRILLS, help="仅运行 source/db/net 之一")
    args = parser.parse_args(argv)
    date_str = args.date or datetime.now(CST).strftime("%Y-%m-%d")
    report = run_all(date_str, only=args.only, output_root=DEFAULT_OUTPUT_ROOT)

    print(f"[chaos-drill] {date_str} {report['summary']['message']}")
    for drill in report["drills"]:
        status = "通过" if drill["passed"] else "失败"
        print(f"  - {drill['name']}: {status} | {drill['path']} | RTO={drill['rto_ms']}ms")

    if not report["summary"]["all_passed"]:
        send_failure_alert(report)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
