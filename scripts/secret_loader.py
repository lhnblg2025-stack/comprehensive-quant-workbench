#!/usr/bin/env python3
"""secret_loader.py — 密钥治理 (V12.3.0 安全收口)

原则: 配置文件(config/*.json)中不存放真实密钥, 只放 env:NAME 占位符。
真实密钥存放在:
  1. 环境变量(最高优先)
  2. config/.env.secrets 文件(KEY=VALUE 每行一个, 已 gitignore, 双机各自维护)

用法:
  from secret_loader import load_config_with_secrets, get_secret
  cfg = load_config_with_secrets("config/private_data_sources.json")  # 自动解析 env:NAME
  token = get_secret("TUSHARE_API_KEY")                              # 直接取
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = ROOT / "config" / ".env.secrets"

_ENV_PATTERN = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")

_secrets_cache: dict[str, str] | None = None


def _load_secrets_file() -> dict[str, str]:
    """读 config/.env.secrets (KEY=VALUE 每行, # 注释)。"""
    out: dict[str, str] = {}
    try:
        if SECRETS_FILE.exists():
            for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    except Exception:  # noqa: BLE001 - 读取失败返回空, 调用方自行降级
        pass
    return out


def get_secret(name: str, default: str | None = None) -> str | None:
    """取密钥: 环境变量优先, 其次 .env.secrets 文件。"""
    global _secrets_cache
    v = os.environ.get(name)
    if v:
        return v
    if _secrets_cache is None:
        _secrets_cache = _load_secrets_file()
    return _secrets_cache.get(name, default)


def resolve_secrets(obj):
    """递归替换 dict/list/str 中的 env:NAME 占位符为真实密钥。

    - 环境变量缺失且 .env.secrets 也缺失 → 保留原占位符(便于发现), 不抛异常
    - 字符串非 env: 前缀 → 原样返回
    """
    if isinstance(obj, dict):
        return {k: resolve_secrets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_secrets(v) for v in obj]
    if isinstance(obj, str):
        m = _ENV_PATTERN.match(obj.strip())
        if m:
            return get_secret(m.group(1)) or obj  # 缺失保留占位符
    return obj


def load_config_with_secrets(path) -> dict:
    """读取 json 配置并解析 env:NAME 占位符。文件缺失返回空 dict。"""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return resolve_secrets(data)


def mask_secret(value: str) -> str:
    """日志脱敏: 只保留前后 3 字符。"""
    if not value or len(value) < 8:
        return "***"
    return f"{value[:3]}***{value[-3:]}"
