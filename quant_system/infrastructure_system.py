"""
infrastructure_system.py — 系统基础设施
V4.1 feature

提供：配置管理、告警系统、日志基础设施、性能基准测试
"""

import os
import json
import time
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)


# ══════════════════════════════════════
# 1. ConfigManager
# ══════════════════════════════════════

class ConfigManager:
    """配置管理"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
            cls._instance._config_dir = os.path.expanduser("~/.quant_system/config/")
            os.makedirs(cls._instance._config_dir, exist_ok=True)
        return cls._instance

    def load(self, name: str = "default") -> dict:
        """加载配置"""
        # P2-Q25-fix(L304): 删除未使用的 `path` 变量(后续实际使用 yaml_path/json_path)。
        yaml_path = os.path.join(self._config_dir, f"{name}.yaml")
        json_path = os.path.join(self._config_dir, f"{name}.json")

        for p in [yaml_path, json_path]:
            if os.path.exists(p):
                with open(p) as f:
                    if p.endswith(".yaml"):
                        self._config[name] = yaml.safe_load(f) or {}
                    else:
                        self._config[name] = json.load(f)
                return self._config[name]

        # 创建默认配置
        default = self._create_default()
        self._config[name] = default
        self.save(name, default)
        return default

    def save(self, name: str, config: dict):
        """保存配置"""
        path = os.path.join(self._config_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None, config_name: str = "default") -> Any:
        """获取配置值（点号分隔路径）"""
        cfg = self._config.get(config_name, {})
        parts = key.split(".")
        for part in parts:
            if isinstance(cfg, dict):
                cfg = cfg.get(part)
            else:
                return default
        return cfg if cfg is not None else default

    def set(self, key: str, value: Any, config_name: str = "default"):
        """设置配置值"""
        if config_name not in self._config:
            self._config[config_name] = {}
        cfg = self._config[config_name]
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cfg:
                cfg[part] = {}
            cfg = cfg[part]
        cfg[parts[-1]] = value
        self.save(config_name, self._config[config_name])

    def _create_default(self) -> dict:
        """创建默认配置"""
        return {
            "system": {
                "name": "quant_system",
                "version": "V4.1",
                "data_dir": os.path.expanduser("~/.quant_system/"),
            },
            "execution": {
                "broker": "sim",
                # P2-Q25-fix(M298): 佣金/印花税/过户费与 config.PortfolioConfig 统一,
                # 按 A股 契约: 最低佣金 5 元/笔、印花税卖出 0.05%、过户费 0.001% 双边。
                "commission_rate": 0.000085,    # 佣金率: 万0.85, 单笔最低见 min_commission
                "min_commission": 5.0,          # A股最低佣金: 5元/笔
                "stamp_tax_rate": 0.0005,       # 印花税: 仅卖出 0.05%
                "transfer_fee_rate": 0.00001,   # 过户费: 0.001% 双边
                "slippage_bp": 3,
            },
            "risk": {
                "max_position_pct": 0.1,
                "max_industry_pct": 0.3,
                "max_leverage": 1.0,
                "var_confidence": 0.95,
            },
            "data": {
                "cache_dir": os.path.expanduser("~/.quant_system/data/"),
                "factor_dir": os.path.expanduser("~/.quant_system/factors/"),
                "fundamental_dir": os.path.expanduser("~/.quant_system/fundamentals/"),
            },
            "monitoring": {
                "check_interval_seconds": 60,
                "alert_threshold_pct": 5,
            },
        }


# ══════════════════════════════════════
# 2. AlertSystem
# ══════════════════════════════════════

class Alert:
    """告警数据结构"""
    def __init__(self, level: str, source: str, message: str,
                 data: dict = None):
        self.level = level  # "info", "warning", "critical"
        self.source = source
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.id = f"{self.timestamp.strftime('%Y%m%d%H%M%S')}_{source}_{hash(message) % 10000}"

    def __repr__(self):
        return f"[{self.level.upper()}] {self.source}: {self.message}"


class AlertSystem:
    """告警系统"""

    def __init__(self, max_history: int = 1000):
        self._alerts: deque[Alert] = deque(maxlen=max_history)
        self._rules: list[dict] = []
        self._callbacks: list = []  # 告警回调函数列表

    def add_rule(self, name: str, condition_fn, level: str = "warning",
                 message_template: str = ""):
        """添加告警规则
        
        condition_fn(data) -> bool: 返回 True 时触发
        """
        self._rules.append({
            "name": name,
            "condition": condition_fn,
            "level": level,
            "message_template": message_template,
        })

    def add_callback(self, callback_fn):
        """添加告警回调（如发邮件、推送）"""
        self._callbacks.append(callback_fn)

    def check(self, data: dict) -> list[Alert]:
        """检查所有规则，返回新触发的告警"""
        triggered = []
        for rule in self._rules:
            try:
                if rule["condition"](data):
                    message = rule["message_template"] or f"条件触发: {rule['name']}"
                    # V11 审计修复（Medium）: 原实现同规则每轮命中即追加新告警无去重，
                    # 高频监控下同一条件可产生上千条重复告警。修正: 同规则+同消息去重
                    # （仅当与最近一条同规则告警消息不同时才新增）。
                    dedup_key = (rule["name"], message)
                    recent = [a for a in self._alerts if a.level == rule["level"]
                              and getattr(a, "dedup_key", None) == dedup_key]
                    if recent:
                        continue  # 已存在同规则同消息告警，跳过（不刷屏）
                    alert = Alert(rule["level"], rule["name"], message, data)
                    alert.dedup_key = dedup_key  # type: ignore[attr-defined]
                    self._alerts.append(alert)
                    triggered.append(alert)
                    for cb in self._callbacks:
                        try:
                            cb(alert)
                        except Exception as e:
                            logger.error(f"告警回调失败: {e}")
            except Exception as e:
                logger.error(f"告警规则 {rule['name']} 检查失败: {e}")
        return triggered

    def acknowledge(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_active(self, min_level: str = "warning") -> list[Alert]:
        """获取未确认且达到指定级别的告警"""
        levels = {"info": 0, "warning": 1, "critical": 2}
        min_lvl = levels.get(min_level, 0)
        return [a for a in self._alerts if not a.acknowledged
                and levels.get(a.level, 0) >= min_lvl]

    def report(self) -> str:
        """生成告警报告"""
        active = self.get_active("info")
        lines = [
            f"告警报告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            f"总告警数: {len(self._alerts)}",
            f"未确认: {len(active)}",
            "",
        ]
        for a in active[:20]:
            lines.append(f"  [{a.level.upper()}] {a.source}: {a.message}")
        return "\n".join(lines)

    def clear_all(self):
        """清除所有告警"""
        self._alerts.clear()


# ══════════════════════════════════════
# 3. LoggerInfra
# ══════════════════════════════════════

class LoggerInfra:
    """日志基础设施"""

    def __init__(self, log_dir: str = ""):
        self.log_dir = log_dir or os.path.expanduser("~/.quant_system/logs/")
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup()

    def _setup(self):
        """配置日志系统"""
        log_file = os.path.join(self.log_dir, f"quant_{datetime.now().strftime('%Y%m%d')}.log")

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # P2-Q25-fix(M299): 幂等配置 — 初始化前检查 root logger 是否已存在相同目标
        # 的 handler, 避免多次实例化 LoggerInfra 重复 addHandler 产生重复日志输出。
        abs_log_file = os.path.abspath(log_file)
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == abs_log_file
            for h in root_logger.handlers
        )
        has_console_handler = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in root_logger.handlers
        )

        if not has_file_handler:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
        if not has_console_handler:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            root_logger.addHandler(ch)

    def get_logger(self, name: str) -> logging.Logger:
        """获取命名 logger"""
        return logging.getLogger(name)

    def recent_logs(self, n: int = 50, level: str = "") -> list[dict]:
        """获取最近日志"""
        today = os.path.join(self.log_dir, f"quant_{datetime.now().strftime('%Y%m%d')}.log")
        if not os.path.exists(today):
            return []
        with open(today) as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            entries.append({
                "line": line.strip(),
                "timestamp": line[:19] if len(line) > 19 else "",
            })
        return entries

    def log_stats(self) -> dict:
        """日志统计"""
        today = os.path.join(self.log_dir, f"quant_{datetime.now().strftime('%Y%m%d')}.log")
        if not os.path.exists(today):
            return {"total_lines": 0}
        with open(today) as f:
            lines = f.readlines()
        errors = sum(1 for l in lines if "ERROR" in l)
        warnings = sum(1 for l in lines if "WARNING" in l)
        return {
            "total_lines": len(lines),
            "errors": errors,
            "warnings": warnings,
            "file_size_kb": os.path.getsize(today) // 1024,
        }


# ══════════════════════════════════════
# 4. PerformanceBenchmark
# ══════════════════════════════════════

class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(self, result_dir: str = ""):
        # P2-Q25-fix(L309): 结果持久化到磁盘(JSON), 进程重启后可恢复历史基准。
        self._results: list[dict] = []
        self._result_path = os.path.join(
            result_dir or os.path.expanduser("~/.quant_system/benchmarks/"),
            "benchmark_results.json",
        )
        os.makedirs(os.path.dirname(self._result_path), exist_ok=True)
        self._load()

    def _load(self):
        """从磁盘恢复历史结果; 文件缺失/损坏时保持空列表(降级可见: 打印告警)。"""
        if not os.path.exists(self._result_path):
            return
        try:
            with open(self._result_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                self._results = loaded
        except Exception as e:
            logger.warning(f"PerformanceBenchmark 历史结果加载失败(忽略): {e}")

    def _persist(self):
        """落盘当前全部结果; 写盘失败不中断调用(告警可见)。"""
        try:
            tmp = self._result_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._results, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._result_path)
        except Exception as e:
            logger.warning(f"PerformanceBenchmark 结果持久化失败: {e}")

    def _append(self, stats: dict):
        self._results.append(stats)
        self._persist()

    def timeit(self, func, *args, name: str = "", n_repeats: int = 3,
               **kwargs) -> dict:
        """计时测试"""
        times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        stats = {
            "name": name or func.__name__,
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "n_repeats": n_repeats,
            "timestamp": datetime.now().isoformat(),
        }
        self._append(stats)
        return stats

    def benchmark_data_pipeline(self, n_symbols: int = 100,
                                 n_days: int = 1000) -> dict:
        """基准测试：数据管道性能"""
        import time
        result = {"n_symbols": n_symbols, "n_days": n_days}
        start = time.time()

        # 模拟数据加载
        df = pd.DataFrame(
            np.random.randn(n_days, n_symbols),
            index=pd.date_range(end=datetime.now(), periods=n_days),
            columns=[f"s{i:06d}" for i in range(n_symbols)]
        )

        result["load_time"] = time.time() - start

        start = time.time()
        _ = df.rolling(20).mean()
        result["moving_avg_time"] = time.time() - start

        start = time.time()
        _ = df.rank(pct=True)
        result["rank_time"] = time.time() - start

        result["total_time"] = sum(v for k, v in result.items()
                                    if k.endswith("_time") and isinstance(v, (int, float)))
        self._append({"name": "data_pipeline", **result})
        return result

    def benchmark_factor_computation(self, n_factors: int = 50,
                                      n_stocks: int = 3000) -> dict:
        """基准测试：因子计算性能"""
        import time
        df = pd.DataFrame(
            np.random.randn(n_stocks, n_factors),
            columns=[f"factor_{i}" for i in range(n_factors)]
        )
        start = time.time()
        _ = df.apply(lambda x: (x - x.mean()) / x.std().clip(lower=1e-12))
        zscore_time = time.time() - start

        start = time.time()
        _ = df.rank(pct=True)
        rank_time = time.time() - start

        result = {
            "name": "factor_computation",
            "n_factors": n_factors,
            "n_stocks": n_stocks,
            "zscore_time": zscore_time,
            "rank_time": rank_time,
        }
        self._append(result)
        return result

    def report(self) -> str:
        """生成基准测试报告"""
        lines = [
            "=" * 55,
            "性能基准测试报告 (V4.1 feature)",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
            "",
        ]
        for r in self._results:
            lines.append(f"  [{r.get('name', 'test')}]")
            for k, v in r.items():
                if k != "name" and isinstance(v, (int, float)):
                    unit = "s" if "time" in k else ""
                    lines.append(f"    {k}: {v:.4f}{unit}")
            lines.append("")
        return "\n".join(lines)


# ══════════════════════════════════════
# 5. TimelineTracker
# ══════════════════════════════════════

class TimelineTracker:
    """时间线追踪器 - 记录系统关键事件"""

    def __init__(self):
        self._events: list[dict] = []

    def record(self, category: str, action: str, message: str = "",
               data: dict = None):
        """记录事件"""
        self._events.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "action": action,
            "message": message,
            "data": data or {},
        })

    def get_events(self, category: str = "", limit: int = 50) -> list[dict]:
        """获取事件"""
        events = self._events
        if category:
            events = [e for e in events if e["category"] == category]
        return events[-limit:]

    def report(self, hours: int = 24) -> str:
        """生成时间线报告"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [e for e in self._events
                  if datetime.fromisoformat(e["timestamp"]) > cutoff]
        lines = [
            f"事件时间线 (最近 {hours} 小时)",
            "=" * 55,
        ]
        for e in recent[-30:]:
            lines.append(f"  {e['timestamp'][:19]} [{e['category']}] {e['action']}")
        return "\n".join(lines)


__all__ = [
    "ConfigManager", "AlertSystem", "Alert",
    "LoggerInfra", "PerformanceBenchmark", "TimelineTracker",
]
