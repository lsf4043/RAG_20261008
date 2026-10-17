"""
线程池管理 —— 用于把同步阻塞调用（Milvus / 模型推理）移出事件循环

要点：
- 使用 ``asyncio.get_running_loop()``（``get_event_loop()`` 在协程里已废弃）；
- ``run_sync`` 支持 ``timeout``，超时抛 ``asyncio.TimeoutError``，
  避免上游卡死时把工作线程永久占住；
- 进程退出时通过 ``shutdown_thread_pool()`` 释放线程。
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ThreadPoolManager:
    """线程池管理器"""

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers or settings.THREAD_POOL_SIZE
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="rag-io"
        )
        self._closed = False
        logger.info("线程池初始化完成，最大工作线程数: %s", self.max_workers)

    async def run_sync(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """在线程池中运行同步函数（可设置超时）"""
        if self._closed:
            raise RuntimeError("线程池已关闭，拒绝新的任务")
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        future = loop.run_in_executor(self.executor, call)
        return await asyncio.wait_for(future, timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=wait)
        logger.info("线程池已关闭")


thread_pool = ThreadPoolManager()


def shutdown_thread_pool() -> None:
    thread_pool.shutdown()
