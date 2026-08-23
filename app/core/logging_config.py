"""
app/core/logging_config.py — 统一日志配置

使用方式（在 main.py 或任何入口模块）:
    from app.core.logging_config import setup_logging
    setup_logging()

Author: 工藤
Date: 2026-08-05
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 默认日志目录
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# 日志格式
CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str | None = None,
    level: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB 单文件上限
    backup_count: int = 5,              # 保留最近 5 个备份
) -> None:
    """初始化全项目统一日志配置。

    输出:
        - 控制台: INFO 级别，简洁格式
        - 文件:   DEBUG 级别，含文件名+行号，自动轮转

    参数:
        log_dir:      日志目录，默认项目根目录下的 logs/
        level:        日志级别，默认读取 LOG_LEVEL 环境变量 → INFO
        max_bytes:    单个日志文件最大字节数
        backup_count: 保留的备份文件数量
    """
    # 解析日志级别
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level, logging.INFO)

    # 创建日志目录
    target_dir = Path(log_dir) if log_dir else LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 根 Logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # 根设最低，由 handler 各自控制

    # 清除已有的 handler（避免重复添加）
    root.handlers.clear()

    # ── 控制台 Handler ──────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    # ── 文件 Handler（带轮转） ────────────────────────
    today = datetime.now().strftime("%Y%m%d")
    log_file = target_dir / f"finace-agent-{today}.log"
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
    root.addHandler(file_handler)

    # ── 启动日志 ────────────────────────────────────
    logging.getLogger(__name__).info(
        "日志系统初始化完成: level=%s, dir=%s, max_bytes=%dMB",
        level,
        target_dir,
        max_bytes // (1024 * 1024),
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块 Logger（便捷函数）。

    等价于 logging.getLogger(name)，但确保日志系统已初始化。
    """
    return logging.getLogger(name)
