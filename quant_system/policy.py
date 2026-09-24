"""政策引擎：集中读取量化策略中的魔法阈值。

设计目标:
  - 阈值只在 config/policy.yaml 中维护，代码只通过 get_policy 读取。
  - 配置缺失或文件不存在时，调用方传入的 default 原样返回，避免系统崩溃。
  - 用 lru_cache 缓存整份 policy 文件；修改文件后调用 clear_policy_cache。
  - 环境标签 ENV 支持 prod/backtest/all，默认 all。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - 仅在没有 PyYAML 的环境下降级
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "policy.yaml"

_ENV_KEYS = {"prod", "backtest", "all"}
_MISSING = object()


def _current_env() -> str:
    return os.environ.get("ENV", "all").strip().lower() or "all"


def _resolve_env_value(value: Any) -> Any:
    """若值是 {prod/backtest/all: ...} 形式，则按当前 ENV 解析；否则原样返回。"""
    if not isinstance(value, dict) or not value:
        return value
    if set(value).issubset(_ENV_KEYS):
        env = _current_env()
        if env in value:
            return value[env]
        if "all" in value:
            return value["all"]
        return _MISSING
    return value


def _read_policy_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    data: Any = None
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = None
    if data is None:
        try:
            data = json.loads(text)
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_policy(path: Path | str | None = None) -> dict:
    """读取并缓存整个 policy 文件。

    文件缺失、YAML/JSON 解析失败时返回空 dict；调用方通过 get_policy 的
    default 参数保持原行为。
    """
    target = Path(path) if path is not None else POLICY_PATH
    return _read_policy_file(target)


def _lookup(data: dict, key_path: str) -> Any:
    current: Any = data
    for part in key_path.split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def get_policy(key_path: str, default: Any = None) -> Any:
    """按点路径读取政策值，例如 get_policy("veto.hard_coef", 0.3)。"""
    data = load_policy(POLICY_PATH)
    value = _lookup(data, key_path)
    if value is _MISSING:
        return default
    value = _resolve_env_value(value)
    if value is _MISSING:
        return default
    return value


def clear_policy_cache() -> None:
    """清空 load_policy 的缓存；测试或热更新配置后调用。"""
    load_policy.cache_clear()


# 兼容部分测试/调用方直接使用 get_policy.cache_clear() 的写法。
get_policy.cache_clear = clear_policy_cache  # type: ignore[attr-defined]
