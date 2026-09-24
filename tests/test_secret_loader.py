"""secret_loader 密钥治理 — 单元测试。

验证 env:NAME 占位符解析: 环境变量优先 / .env.secrets 兜底 / 缺失保留占位符。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import secret_loader  # noqa: E402


@pytest.fixture()
def fresh_secrets(monkeypatch, tmp_path):
    """隔离 .env.secrets 读取: 指向 tmp 文件 + 清空环境变量缓存。"""
    sfile = tmp_path / ".env.secrets"
    sfile.write_text("# comment\nTEST_KEY=file_value\nEMPTY=\n", encoding="utf-8")
    monkeypatch.setattr(secret_loader, "SECRETS_FILE", sfile)
    monkeypatch.setattr(secret_loader, "_secrets_cache", None)
    return tmp_path


def test_get_secret_from_file(fresh_secrets):
    assert secret_loader.get_secret("TEST_KEY") == "file_value"


def test_get_secret_env_priority(fresh_secrets, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "env_value")
    assert secret_loader.get_secret("TEST_KEY") == "env_value"


def test_get_secret_missing_returns_default(fresh_secrets):
    assert secret_loader.get_secret("NOPE") is None
    assert secret_loader.get_secret("NOPE", "dflt") == "dflt"


def test_resolve_nested_dict(fresh_secrets, monkeypatch):
    monkeypatch.setenv("A", "a_val")
    data = {"k1": "env:A", "k2": {"k3": "env:TEST_KEY", "k4": "plain"}, "list": ["env:A", "x"]}
    out = secret_loader.resolve_secrets(data)
    assert out["k1"] == "a_val"
    assert out["k2"]["k3"] == "file_value"
    assert out["k2"]["k4"] == "plain"
    assert out["list"] == ["a_val", "x"]


def test_resolve_missing_keeps_placeholder(fresh_secrets):
    out = secret_loader.resolve_secrets({"k": "env:NOT_DEFINED_ANYWHERE"})
    assert out["k"] == "env:NOT_DEFINED_ANYWHERE"  # 保留占位符便于发现


def test_load_config_with_secrets(fresh_secrets, tmp_path, monkeypatch):
    monkeypatch.setenv("K1", "v1")
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"a": "env:K1", "b": {"c": "env:TEST_KEY"}}', encoding="utf-8")
    out = secret_loader.load_config_with_secrets(cfg)
    assert out == {"a": "v1", "b": {"c": "file_value"}}


def test_load_config_missing_file(tmp_path):
    assert secret_loader.load_config_with_secrets(tmp_path / "nope.json") == {}


def test_mask_secret():
    assert secret_loader.mask_secret("abcdefgh") == "abc***fgh"
    assert secret_loader.mask_secret("short") == "***"
    assert secret_loader.mask_secret("") == "***"


def test_real_config_has_no_plain_secrets():
    """真实配置文件不允许出现明文密钥(env: 占位符或非密钥字段除外)。"""
    import json
    import re

    for name in ("private_data_sources.json", "report_delivery.json"):
        p = ROOT / "config" / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        # 排除 env:占位符 / URL / ou_用户ID / qqbot: / 日期
        leaks = [l for l in re.findall(r"[A-Za-z0-9+/=]{16,}", txt)
                 if not l.startswith(("env:", "http", "qqbot:", "user:ou_"))
                 and "ou_" not in l and "token=" not in l
                 and not re.match(r"^20\d{6}", l)
                 and l not in ("com/api/ynote/mcp/sse",)]
        assert not leaks, f"{name} 残留明文密钥: {leaks[:5]}"


def test_secrets_file_gitignored():
    """.env.secrets 必须被 gitignore(防止提交真实密钥)。"""
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    assert ".env.secrets" in gi
