"""
日志工具

要点：
- 文件日志使用 RotatingFileHandler，避免单文件无限增长；
- 日志目录不可写时只降级到控制台并给出告警，**不会让服务启动失败**
  （之前的实现会在只读挂载 / 权限异常的目录上抛 PermissionError，直接带崩进程）；
- 重复调用 setup_logging() 是幂等的，不会重复挂载 handler。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import List, Optional

from app.config import settings

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_SETUP_FLAG = "._dsh_logging_configured"


def setup_logging() -> None:
    """配置根日志器（幂等，失败降级）"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root_logger = logging.getLogger()
    if getattr(root_logger, _SETUP_FLAG, False):
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
        return

    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if settings.LOG_TO_FILE:
        try:
            log_dir = os.path.dirname(settings.LOG_FILE_PATH)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                settings.LOG_FILE_PATH,
                maxBytes=settings.LOG_MAX_BYTES,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError as exc:
            # 关键：日志写入失败不能阻断服务启动
            root_logger.warning(
                "无法写入日志文件 %s（%s），已降级为仅控制台输出",
                settings.LOG_FILE_PATH,
                exc,
            )

    setattr(root_logger, _SETUP_FLAG, True)


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


class ProcessLogger:
    """处理过程日志记录器（用于把处理进度回传给前端 / WebSocket）"""

    def __init__(self) -> None:
        self.logs: List[dict] = []
        self.start_time: Optional[datetime] = None

    def start(self) -> None:
        self.logs = []
        self.start_time = datetime.now()

    def log(self, level: str, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": level,
            "step": step,
            "step_name": step_name,
            "progress": max(0, min(100, int(progress))),
            "message": message,
        }
        self.logs.append(entry)
        return entry

    def info(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        return self.log("info", step, message, progress, step_name)

    def warning(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        return self.log("warning", step, message, progress, step_name)

    def error(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        return self.log("error", step, message, progress, step_name)

    def success(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        return self.log("success", step, message, progress, step_name)

    def get_logs(self) -> List[dict]:
        return list(self.logs)

    def get_elapsed_time(self) -> float:
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
