"""
日志工具
"""
import os
import logging
import sys
from datetime import datetime
from typing import Optional

from app.config import settings


def setup_logging():
    """设置日志"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有的处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if settings.LOG_TO_FILE:
        # 确保日志目录存在
        log_dir = os.path.dirname(settings.LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(
            settings.LOG_FILE_PATH,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器

    Args:
        name: 日志器名称

    Returns:
        日志器实例
    """
    return logging.getLogger(name)


class ProcessLogger:
    """
    处理过程日志记录器
    用于记录处理过程中的日志，并支持实时推送
    """

    def __init__(self):
        self.logs: list = []
        self.start_time: Optional[datetime] = None

    def start(self):
        """开始记录"""
        self.logs = []
        self.start_time = datetime.now()

    def log(self, level: str, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        """
        记录日志

        Args:
            level: 日志级别 (info/warning/error/success)
            step: 步骤代码
            message: 日志消息
            progress: 进度百分比
            step_name: 步骤中文名

        Returns:
            日志条目
        """
        log_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": level,
            "step": step,
            "step_name": step_name,
            "progress": progress,
            "message": message
        }
        self.logs.append(log_entry)
        return log_entry

    def info(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        """记录信息日志"""
        return self.log("info", step, message, progress, step_name)

    def warning(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        """记录警告日志"""
        return self.log("warning", step, message, progress, step_name)

    def error(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        """记录错误日志"""
        return self.log("error", step, message, progress, step_name)

    def success(self, step: str, message: str, progress: int = 0, step_name: str = "") -> dict:
        """记录成功日志"""
        return self.log("success", step, message, progress, step_name)

    def get_logs(self) -> list:
        """获取所有日志"""
        return self.logs

    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
