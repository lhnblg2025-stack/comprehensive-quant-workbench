"""health_guardian 进程内存监控单元测试。

覆盖：
  - 同进程连续 3 次超阈值才触发告警
  - 单次超阈值不告警
  - 恢复正常后重置连续计数
  - psutil 缺失时降级 /proc 不崩溃
  - /proc VmRSS 解析与白名单匹配
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import health_guardian as hg


@pytest.fixture(autouse=True)
def fixed_mem_threshold(monkeypatch):
    monkeypatch.setattr(hg, "MEM_WARN_MB", 2048)


class FakeMemoryInfo:
    def __init__(self, rss_bytes: int):
        self.rss = rss_bytes


class FakeProcess:
    def __init__(self, pid: int, cmdline: list[str], rss_bytes: int):
        self.info = {
            "pid": pid,
            "cmdline": cmdline,
            "memory_info": FakeMemoryInfo(rss_bytes),
        }


def _rss_mb(proc_bytes: int) -> float:
    return proc_bytes / 1024 / 1024


def _set_artifact_context(monkeypatch, tmp_path: Path, day: str, hour: int = 21):
    """将产出物检查切换到"伪云端目录" tmp_path/generated 与固定日期。

    模拟腾讯云 ssh 可达，_cloud_artifact_latest 改为从伪云端目录 glob 解析，
    使产出物检查逻辑（交易日判断/after_hour/optional 门控）可在本机单测。
    """
    monkeypatch.setattr(hg, "ROOT", tmp_path)
    monkeypatch.setattr(
        hg,
        "_artifact_now",
        lambda: datetime(int(day[:4]), int(day[5:7]), int(day[8:10]), hour, 0),
    )
    monkeypatch.setattr(hg, "_load_trade_calendar", lambda: None)
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)
    cloud_dir = tmp_path / "generated"

    def _fake_cloud_artifact_latest(pattern: str, directory: str | None = None,
                                    recursive: bool = False) -> date | None:
        # 单测伪云端目录固定为 tmp_path/generated；目录参数为 None(默认) 时才取默认目录，
        # 与生产 _cloud_artifact_latest(pattern, directory=None) 语义一致。
        if directory is not None:
            return None
        dates: list[date] = []
        glob_root = cloud_dir / "**" if recursive else cloud_dir
        for path in (glob_root.glob(pattern) if recursive else cloud_dir.glob(pattern)):
            parsed = hg._parse_artifact_date(path)
            if parsed is not None:
                dates.append(parsed)
        return max(dates) if dates else None

    monkeypatch.setattr(hg, "_cloud_artifact_latest", _fake_cloud_artifact_latest)
    return cloud_dir


def _write_artifact(generated: Path, name: str) -> None:
    generated.mkdir(parents=True, exist_ok=True)
    (generated / name).write_text("{}\n".format(name), encoding="utf-8")


def test_three_consecutive_over_threshold_triggers(monkeypatch):
    proc = "realtime_snapshot.py(pid=10)"
    monkeypatch.setattr(
        hg,
        "_collect_process_rss",
        lambda: [(proc, _rss_mb(3 * 1024 * 1024 * 1024))],
    )
    state = {}

    assert hg._check_process_memory(state) == []
    assert hg._check_process_memory(state) == []
    alerts = hg._check_process_memory(state)

    assert len(alerts) == 1
    assert alerts[0].startswith("⚠️ OOM风险:")
    assert "RSS 3072MB > 2048MB" in alerts[0]
    assert state["last_mem"][proc]["rss_mb"] == pytest.approx(3072, abs=0.1)
    assert state[hg.MEM_OVER_COUNT_KEY][proc] == 3


def test_single_over_threshold_does_not_alert(monkeypatch):
    proc = "update_kline_tencent.py(pid=11)"
    monkeypatch.setattr(hg, "_collect_process_rss", lambda: [(proc, 2100.0)])
    state = {}

    assert hg._check_process_memory(state) == []
    assert state[hg.MEM_OVER_COUNT_KEY][proc] == 1
    assert state["last_mem"][proc]["rss_mb"] == 2100.0


def test_recovery_resets_consecutive_count(monkeypatch):
    proc = "temperature_replay.py(pid=12)"
    monkeypatch.setattr(hg, "_collect_process_rss", lambda: [(proc, 3000.0)])
    state = {}

    hg._check_process_memory(state)
    hg._check_process_memory(state)
    assert state[hg.MEM_OVER_COUNT_KEY][proc] == 2

    monkeypatch.setattr(hg, "_collect_process_rss", lambda: [(proc, 1024.0)])
    assert hg._check_process_memory(state) == []
    assert state[hg.MEM_OVER_COUNT_KEY][proc] == 0

    monkeypatch.setattr(hg, "_collect_process_rss", lambda: [(proc, 3000.0)])
    assert hg._check_process_memory(state) == []
    assert state[hg.MEM_OVER_COUNT_KEY][proc] == 1


def test_psutil_missing_fallback_does_not_crash(monkeypatch):
    monkeypatch.setattr(hg, "psutil", None)
    monkeypatch.setattr(
        hg,
        "_collect_process_rss_proc",
        lambda: [("quant_web(pid=13)", 2048.0)],
    )
    state = {}

    assert hg._check_process_memory(state) == []
    assert state["last_mem"]["quant_web(pid=13)"]["rss_mb"] == 2048.0


def test_proc_fallback_parses_vmrss_and_matches_cmdline(tmp_path):
    pid_dir = tmp_path / "123"
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"python\x00update_valuation_baostock.py\x00--day\x00")
    (pid_dir / "status").write_text("VmRSS:\t 2097152 kB\n", encoding="utf-8")

    results = hg._collect_process_rss_proc(tmp_path)

    assert results == [("update_valuation_baostock.py(pid=123)", 2048.0)]


def test_proc_fallback_ignores_non_whitelisted_process(tmp_path):
    pid_dir = tmp_path / "999"
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"python\x00other_script.py\x00")
    (pid_dir / "status").write_text("VmRSS:\t 2097152 kB\n", encoding="utf-8")

    assert hg._collect_process_rss_proc(tmp_path) == []


# ── P2b: 每日关键产出物账本 ──────────────────────────────────

def test_artifacts_fresh_no_alert(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    for name in (
        "battle_map_2026-08-12.md",
        "multi_agent_2026-08-12.json",
        "macro_veto_2026-08-12.json",
        "data_health_20260812.md",
    ):
        _write_artifact(generated, name)

    assert hg._check_artifacts({}) == []


def test_artifacts_missing_battle_map_alerts(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    for name in (
        "multi_agent_2026-08-12.json",
        "macro_veto_2026-08-12.json",
        "data_health_20260812.md",
    ):
        _write_artifact(generated, name)
    # battle_map 从未产出且宽限期已过（首次检查 2026-08-08，满 4 天）→ 严格告警
    state = {"artifact_grace": {"battle_map": {"first_seen": "2026-08-08"}}}

    alerts = hg._check_artifacts(state)

    battle_alerts = [a for a in alerts if "battle_map" in a]
    assert battle_alerts
    assert "产出物缺失" in battle_alerts[0]
    assert "应为 2026-08-12" in battle_alerts[0]


def test_artifacts_never_seen_grace_skips_then_alerts(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    _write_artifact(generated, "data_health_20260812.md")

    # 首次检查：记录 first_seen，不告警（宽限 2 天）
    state: dict = {}
    assert hg._check_artifacts(state) == []
    assert state["artifact_grace"]["battle_map"]["first_seen"] == "2026-08-12"

    # 宽限期内（首次检查 1 天前）仍不告警
    state = {"artifact_grace": {"battle_map": {"first_seen": "2026-08-11"},
                                "multi_agent": {"first_seen": "2026-08-11"},
                                "macro_veto": {"first_seen": "2026-08-11"}}}
    assert hg._check_artifacts(state) == []

    # 宽限期满（首次检查 2 天前）→ 告警
    state = {"artifact_grace": {"battle_map": {"first_seen": "2026-08-10"},
                                "multi_agent": {"first_seen": "2026-08-10"},
                                "macro_veto": {"first_seen": "2026-08-10"}}}
    alerts = hg._check_artifacts(state)
    assert any("battle_map" in a and "最新 无" in a for a in alerts)


def test_artifacts_stale_alerts(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    _write_artifact(generated, "battle_map_2026-08-05.md")

    alerts = hg._check_artifacts({})

    assert any("battle_map" in a and "产出物缺失" in a for a in alerts)


def test_artifacts_weekend_skips(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-16", hour=21)
    _write_artifact(generated, "battle_map_2026-08-11.md")

    assert hg._check_artifacts({}) == []


def test_artifacts_cloud_unreachable_skips(tmp_path, monkeypatch):
    _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: False)

    assert hg._check_artifacts({}) == []


def test_optional_artifact_stale_alerts_when_history_exists(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    _write_artifact(generated, "data_health_20260812.md")
    _write_artifact(generated, "rs_strength_2026-08-11.md")  # 有历史但已过期

    alerts = hg._check_artifacts({})

    assert any("rs_strength" in a and "产出物缺失" in a for a in alerts)
    assert not any("pattern_report" in a for a in alerts)  # 云端无历史 → 跳过


def test_optional_artifact_fresh_no_alert(tmp_path, monkeypatch):
    generated = _set_artifact_context(monkeypatch, tmp_path, "2026-08-12", hour=21)
    _write_artifact(generated, "data_health_20260812.md")
    _write_artifact(generated, "rs_strength_2026-08-12.md")
    _write_artifact(generated, "pattern_report_2026-08-12.md")

    assert hg._check_artifacts({}) == []


def test_parse_artifact_date_new_patterns():
    assert hg._parse_artifact_date(Path("rs_strength_2026-08-07.md")) == date(2026, 8, 7)
    assert hg._parse_artifact_date(Path("pattern_report_2026-08-11.md")) == date(2026, 8, 11)


# ── P1-3: 补录日频产物守护（html_report/daily_report/meta_review + mtime 型产物）──

def test_parse_artifact_date_p1_3_patterns():
    # 日期居中/后缀/前缀三种 P1-3 命名，含中文（GBK 往返乱码不影响 ASCII 日期片段）
    assert hg._parse_artifact_date(Path("A股多因子复盘日报_2026-08-14.html")) == date(2026, 8, 14)
    assert hg._parse_artifact_date(Path("2026-08-14-A股量化日报.md")) == date(2026, 8, 14)
    assert hg._parse_artifact_date(Path("meta_review_2026-08-14.json")) == date(2026, 8, 14)
    assert hg._parse_artifact_date(Path("meta_review_2026-08-14.md")) == date(2026, 8, 14)
    # 不能在无日期片段的对象乱取日期
    assert hg._parse_artifact_date(Path("noipraises.md")) is None
    assert hg._parse_artifact_date(Path("2026-13-01x.md")) is None


def test_mtime_product_stale_alerts(monkeypatch):
    # 交易日 21:00，产物最新 mtime 存于 3 天前 → 陈旧告警
    monkeypatch.setattr(hg, "_load_trade_calendar", lambda: None)
    monkeypatch.setattr(hg, "_artifact_now",
                        lambda: datetime(2026, 8, 12, 21, 0))
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)
    monkeypatch.setattr(hg, "_cloud_mtime_latest",
                        lambda d, p: datetime(2026, 8, 9, 18, 0, 0))

    alerts = hg._check_cloud_daily_products({})
    assert any("产物陈旧" in a and "msnapshot" in a for a in alerts)
    # after_min_hour 门控：未到点时不应检查
    monkeypatch.setattr(hg, "_artifact_now", lambda: datetime(2026, 8, 12, 10, 0))
    assert hg._check_cloud_daily_products({}) == []


def test_mtime_product_missing_alerts(monkeypatch):
    monkeypatch.setattr(hg, "_load_trade_calendar", lambda: None)
    monkeypatch.setattr(hg, "_artifact_now",
                        lambda: datetime(2026, 8, 12, 21, 0))
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)
    monkeypatch.setattr(hg, "_cloud_mtime_latest", lambda d, p: None)

    alerts = hg._check_cloud_daily_products({})
    assert any("产出物缺失" in a and "kline" in a for a in alerts)


def test_mtime_product_weekend_skips(monkeypatch):
    # 非交易日(周六) → 产物停更属正常，跳过
    monkeypatch.setattr(hg, "_load_trade_calendar", lambda: None)
    monkeypatch.setattr(hg, "_artifact_now",
                        lambda: datetime(2026, 8, 15, 21, 0))  # 2026-08-15 周六
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)

    assert hg._check_cloud_daily_products({}) == []


def test_mtime_product_unreachable_skips(monkeypatch):
    monkeypatch.setattr(hg, "_load_trade_calendar", lambda: None)
    monkeypatch.setattr(hg, "_artifact_now",
                        lambda: datetime(2026, 8, 12, 21, 0))
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: False)

    assert hg._check_cloud_daily_products({}) == []


# ── P1-4: 云端 config/.env.secrets 探针（fail-loud）──

def test_secrets_present_no_alert(monkeypatch):
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)
    monkeypatch.setattr(hg, "_cloud_ssh",
                        lambda cmd, timeout=20: (0, ["PRESENT"]) if "if exist" in cmd
                        else (0, ["line"]))
    assert hg._check_cloud_secrets() == []


def test_secrets_file_missing_alerts(monkeypatch):
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)
    monkeypatch.setattr(hg, "_cloud_ssh",
                        lambda cmd, timeout=20: (0, ["MISSING"]) if "if exist" in cmd
                        else (0, []))
    alerts = hg._check_cloud_secrets()
    assert alerts and "config/.env.secrets 不存在" in alerts[0]


def test_secrets_key_missing_alerts(monkeypatch):
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: True)
    calls = {}

    def fake_ssh(cmd, timeout=20):
        if "if exist" in cmd:
            return (0, ["PRESENT"])
        if "findstr" in cmd:
            # FEISHU_CHAT_ID 缺失 → 该 findstr 返回非 0
            return (1, []) if "FEISHU_CHAT_ID" in cmd else (0, ["line"])
        return (0, [])

    monkeypatch.setattr(hg, "_cloud_ssh", fake_ssh)
    alerts = hg._check_cloud_secrets()
    assert alerts and "缺必需 KEY: FEISHU_CHAT_ID" in alerts[0]


def test_secrets_unreachable_skips(monkeypatch):
    monkeypatch.setattr(hg, "_cloud_reachable", lambda: False)
    assert hg._check_cloud_secrets() == []


def test_cloud_artifact_latest_parses_names(monkeypatch):
    monkeypatch.setattr(
        hg,
        "_cloud_ssh",
        lambda cmd: (0, ["battle_map_2026-08-11.md", "battle_map_2026-08-12.md"]),
    )

    assert hg._cloud_artifact_latest("battle_map_*.md") == date(2026, 8, 12)


def test_cloud_artifact_latest_ssh_failure_returns_none(monkeypatch):
    monkeypatch.setattr(hg, "_cloud_ssh", lambda cmd: None)

    assert hg._cloud_artifact_latest("battle_map_*.md") is None


def test_cloud_artifact_latest_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(hg, "_cloud_ssh", lambda cmd: (1, ["File Not Found"]))

    assert hg._cloud_artifact_latest("battle_map_*.md") is None


def test_rt_snapshot_due_gate(monkeypatch):
    monkeypatch.setattr(hg, "_load_trade_calendar", lambda: None)

    assert hg._rt_snapshot_due(datetime(2026, 8, 13, 10, 0)) is True    # 周四盘中
    assert hg._rt_snapshot_due(datetime(2026, 8, 13, 9, 29)) is False   # 盘前
    assert hg._rt_snapshot_due(datetime(2026, 8, 13, 15, 6)) is False   # 盘后
    assert hg._rt_snapshot_due(datetime(2026, 8, 15, 10, 0)) is False   # 周六
