"""结构化日志配置（基于 loguru）。"""

from __future__ import annotations

import sys
from loguru import logger


_configured = False


def configure_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """初始化全局日志配置，幂等调用。"""
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level=log_level,
            rotation="100 MB",
            retention="7 days",
            encoding="utf-8",
        )

    _configured = True


def get_logger(name: str):
    """获取绑定了模块名的 logger。"""
    return logger.bind(name=name)
