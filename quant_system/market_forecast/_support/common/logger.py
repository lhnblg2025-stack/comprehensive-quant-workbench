"""
logger.py — QuantV6 日志系统
控制台 + 文件双输出，按天滚动，级别可配。
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "quantv6", level: str = "INFO",
               log_file: str | None = None, console: bool = True) -> logging.Logger:
    """
    获取（或创建）命名 logger。
    - level: DEBUG/INFO/WARNING/ERROR
    - log_file: 日志文件路径，None 则默认 /root/quant/logs/quantv6.log（可被环境变量覆盖）
    - console: 是否同时输出控制台
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file is None:
        # 2026-08-22 稳定化: 默认回落路径从 /root/quant/logs(无权限刷屏) 改为仓库内可写目录
        _fallback = str(Path(__file__).resolve().parents[4] / "generated" / "logs" / "quantv6.log")
        log_file = os.environ.get("QV6_LOG_FILE", _fallback)
    if log_file:
        p = Path(log_file)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = TimedRotatingFileHandler(str(p), when="midnight", backupCount=30, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            # 2026-08-22 稳定化: 文件日志失败仅记一次 warning, 不再每次 exc_info 刷屏
            if not getattr(get_logger, "_warned", False):
                get_logger._warned = True
                logging.getLogger(__name__).warning(
                    f"[logger] 文件日志不可用({e}), 仅控制台输出")

    _LOGGERS[name] = logger
    return logger


def set_level(level: str) -> None:
    """全局调整日志级别。"""
    for lg in _LOGGERS.values():
        lg.setLevel(getattr(logging, level.upper(), logging.INFO))
