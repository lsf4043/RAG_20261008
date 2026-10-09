"""
线程池管理 - 用于处理同步阻塞操作
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ThreadPoolManager:
    """线程池管理器"""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or settings.THREAD_POOL_SIZE
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"线程池初始化完成，最大工作线程数: {self.max_workers}")

    async def run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        在线程池中运行同步函数

        Args:
            func: 同步函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数执行结果
        """
        loop = asyncio.get_event_loop()

        # 包装函数以支持 kwargs
        def wrapper():
            return func(*args, **kwargs)

        return await loop.run_in_executor(self.executor, wrapper)

    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)
        logger.info("线程池已关闭")


# 全局线程池实例
thread_pool = ThreadPoolManager()
