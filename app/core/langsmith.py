"""
LangSmith 可观测性模块（升级版新增，P0 优化）
集成 LangChain 官方可观测性平台 LangSmith，追踪 RAG 全链路。

两种追踪方式：
- 方案A（推荐）：环境变量自动追踪。设置 LANGCHAIN_TRACING_V2=true 后，
  LangChain/LangGraph 的所有调用自动上报，零代码侵入。
- 方案B：手动埋点。使用 langsmith.Client 创建 run，精细控制追踪范围。

本模块负责初始化 LangSmith 环境，并提供手动埋点的便捷工具。
"""
import os
from typing import Optional, Any, Dict

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith():
    """
    初始化 LangSmith 环境（在应用启动时调用）

    通过设置环境变量启用 LangChain 自动追踪：
    - LANGCHAIN_TRACING_V2: 开启追踪
    - LANGCHAIN_API_KEY: API Key（ls__ 开头）
    - LANGCHAIN_PROJECT: 项目名
    - LANGCHAIN_ENDPOINT: 端点（默认官方 SaaS）
    """
    if not settings.langsmith_enabled:
        logger.info("LangSmith 追踪未启用（未配置 LANGSMITH_API_KEY 或 LANGSMITH_TRACING=false）")
        return

    # 将 .env 中的 LANGSMITH_* 配置映射写入 SDK 实际读取的 LANGCHAIN_* 环境变量
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    logger.info(f"LangSmith 追踪已启用，项目: {settings.LANGSMITH_PROJECT}")


class LangSmithClient:
    """LangSmith 手动埋点客户端（方案B）"""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """懒加载 langsmith.Client"""
        if self._client is None:
            if not settings.langsmith_enabled:
                return None
            try:
                from langsmith import Client
                self._client = Client()
            except Exception as e:
                logger.warning(f"LangSmith Client 初始化失败: {str(e)}")
                return None
        return self._client

    def trace(self, name: str, inputs: Dict[str, Any], tags: Optional[list] = None):
        """
        创建一次追踪 run

        Args:
            name: run 名称
            inputs: 输入
            tags: 标签

        Returns:
            run 对象（或 None）
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            run = client.create_run(
                name=name,
                run_type="chain",
                inputs=inputs,
                tags=tags or [],
            )
            return run
        except Exception as e:
            logger.warning(f"创建 LangSmith trace 失败: {str(e)}")
            return None

    def end_trace(self, run, outputs: Dict[str, Any]):
        """结束追踪 run，记录输出"""
        client = self._get_client()
        if client is None or run is None:
            return
        try:
            client.update_run(run.id, outputs=outputs)
        except Exception as e:
            logger.warning(f"更新 LangSmith trace 失败: {str(e)}")


# 全局 LangSmith 客户端实例
langsmith_client = LangSmithClient()