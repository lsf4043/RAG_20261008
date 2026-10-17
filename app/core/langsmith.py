"""
LangSmith 可观测性

两种方式：
1. 环境变量自动追踪（推荐）：``LANGCHAIN_TRACING_V2=true`` + API Key，LangChain /
   LangGraph 的所有调用自动上报，零侵入；
2. ``@traceable`` 装饰业务方法（query / stream_query），得到业务级 run。

本次修复：删掉了手写 ``create_run`` / ``update_run`` 的客户端——那条路径在异常时
不会结束 run，会在 LangSmith 里留下一堆悬挂记录，而且新版 SDK 已不推荐直接调用。
"""
from __future__ import annotations

import os
from typing import Callable

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith() -> bool:
    """把 LANGSMITH_* 配置映射到 SDK 读取的环境变量"""
    if not settings.langsmith_enabled:
        logger.info("LangSmith 追踪未启用（未配置 LANGSMITH_API_KEY 或已关闭）")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    logger.info("LangSmith 追踪已启用，项目: %s", settings.LANGSMITH_PROJECT)
    return True


def traceable(name: str, run_type: str = "chain") -> Callable[[Callable], Callable]:
    """
    业务级追踪装饰器：未启用 LangSmith 时原样返回函数（零开销、零依赖）
    """

    def decorator(func: Callable) -> Callable:
        if not settings.langsmith_enabled:
            return func
        try:
            from langsmith import traceable as _ls_traceable

            return _ls_traceable(name=name, run_type=run_type)(func)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangSmith traceable 不可用，跳过追踪: %s", exc)
            return func

    return decorator


def langsmith_status() -> str:
    return "enabled" if settings.langsmith_enabled else "disabled"
