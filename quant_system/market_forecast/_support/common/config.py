"""
config.py — QuantV6 配置管理
YAML 加载（config/config.yaml），环境变量覆盖，mtime 热加载。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from quant_system.market_forecast._support.common.exceptions import ConfigError

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_DEFAULT_CONFIG_PATH = "/root/quant/config/config.yaml"
_LOCAL_OVERRIDE = "config.local.yaml"
_DEFAULT_DATA = {
    "trading": {"commission_min": 5.0, "stamp_tax_sell": 0.0005, "transfer_fee": 0.00001, "lot_size": 100},
    "data": {"northbound_intraday_net_buy_after": "2024-08-19"},
}  # P2-Q12-fix: 包内默认配置遵守 A 股成本/整手/北向披露规则，外部配置只做覆盖

_lock = threading.Lock()


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ConfigError("PyYAML 未安装")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置根必须是 dict: {path}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """配置单例，支持热加载。"""

    _instance: "Config | None" = None
    _lock = threading.Lock()

    def __init__(self, config_path: str | None = None, watch: bool = True):
        self.path = Path(config_path or os.environ.get("QV6_CONFIG", _DEFAULT_CONFIG_PATH))
        self.watch = watch
        self._mtime: float = -1
        self._data: dict = {}
        self._local_path = self.path.parent / _LOCAL_OVERRIDE
        self.reload()

    @classmethod
    def instance(cls, config_path: str | None = None) -> "Config":
        with cls._lock:
            if cls._instance is None:
                cls._instance = Config(config_path)
            elif config_path:
                cls._instance.path = Path(config_path)
                cls._instance.reload()
            return cls._instance

    def reload(self) -> None:
        if self.path.exists():
            data = _deep_merge(_DEFAULT_DATA, _load_yaml(self.path))
        else:
            data = dict(_DEFAULT_DATA)  # P2-Q12-fix: 部署机缺省配置文件时使用包内默认值，避免 Config() 直接抛错
        if self._local_path.exists():
            data = _deep_merge(data, _load_yaml(self.local_path))
        # 环境变量覆盖: QV6_<SECTION>_<KEY>
        for k, v in os.environ.items():
            if k.startswith("QV6_"):
                parts = k[4:].lower().split("_")
                node = data
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = v
        self._data = data
        self._mtime = self._stat_mtime()

    def _stat_mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except Exception:
            return 0.0

    def _maybe_reload(self) -> None:
        if self.watch and self._stat_mtime() != self._mtime:
            with _lock:
                if self._stat_mtime() != self._mtime:
                    self.reload()

    def get(self, key: str, default: Any = None) -> Any:
        """点路径取值，如 'trading.risk.max_drawdown'。"""
        self._maybe_reload()
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        self._maybe_reload()
        sec = self._data.get(name, {})
        return sec if isinstance(sec, dict) else {}

    def all(self) -> dict:
        self._maybe_reload()
        return self._data

    @property
    def local_path(self) -> Path:
        return self._local_path


def get_config(config_path: str | None = None) -> Config:
    return Config.instance(config_path)
