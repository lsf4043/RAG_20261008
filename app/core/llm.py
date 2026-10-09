"""
大模型服务（LangChain 版）
使用 LangChain 的 ChatDashScope 替代旧版手写 dashscope 调用。

优化点：
- 使用 LangChain 标准 ChatModel 接口（invoke / stream / bind_tools），与 LangGraph 无缝集成
- 支持结构化输出（with_structured_output），用于 CRAG 文档评分等需要稳定格式的场景
- 通过 LangChain 自动上报 LangSmith（无需额外埋点）
"""
from typing import List, Dict, Optional, AsyncGenerator, Any

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """阿里百炼大模型服务（基于 LangChain ChatDashScope）"""

    def __init__(self):
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE

        # 初始化 LangChain ChatTongyi（DashScope 通义千问）
        self._llm = ChatTongyi(
            model=self.model,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            model_kwargs={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )
        logger.info(f"LangChain 大模型初始化完成，模型: {self.model}")

    def get_llm(self) -> ChatTongyi:
        """获取底层 ChatDashScope 实例（供 LangChain 链 / LangGraph 使用）"""
        return self._llm

    def _to_messages(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """
        将字典格式消息转换为 LangChain 消息对象

        Args:
            messages: 消息列表 [{role: user/assistant/system, content: ...}]
            system_prompt: 系统提示（可选，放在最前）

        Returns:
            LangChain 消息列表
        """
        result: List[BaseMessage] = []
        if system_prompt:
            result.append(SystemMessage(content=system_prompt))

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        return result

    def chat(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        同步对话生成

        Args:
            messages: 消息列表
            system_prompt: 系统提示
            temperature: 温度（可选，覆盖默认）
            max_tokens: 最大 token 数（可选，覆盖默认）

        Returns:
            生成的回答
        """
        langchain_messages = self._to_messages(messages, system_prompt)
        llm = self._llm
        # 如果指定了温度/token，则临时构建一个覆盖参数的实例
        if temperature is not None or max_tokens is not None:
            llm = ChatTongyi(
                model=self.model,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                model_kwargs={
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                },
            )

        response = llm.invoke(langchain_messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"大模型响应完成，生成 {len(answer)} 字符")
        return answer

    async def achat(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        异步对话生成

        Args:
            messages: 消息列表
            system_prompt: 系统提示
            temperature: 温度
            max_tokens: 最大 token 数

        Returns:
            生成的回答
        """
        langchain_messages = self._to_messages(messages, system_prompt)
        response = await self._llm.ainvoke(langchain_messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"大模型（异步）响应完成，生成 {len(answer)} 字符")
        return answer

    async def astream(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        异步流式对话生成

        Args:
            messages: 消息列表
            system_prompt: 系统提示
            temperature: 温度
            max_tokens: 最大 token 数

        Yields:
            生成的文本片段
        """
        langchain_messages = self._to_messages(messages, system_prompt)
        llm = self._llm
        if temperature is not None or max_tokens is not None:
            llm = ChatTongyi(
                model=self.model,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                model_kwargs={
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                },
            )

        async for chunk in llm.astream(langchain_messages):
            content = chunk.content
            if content:
                yield content if isinstance(content, str) else str(content)

    def invoke_messages(self, messages: List[BaseMessage]) -> str:
        """
        直接调用（接收 LangChain 消息对象，供 LangGraph 节点使用）

        Args:
            messages: LangChain 消息列表

        Returns:
            生成的回答
        """
        response = self._llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return answer

    async def ainvoke_messages(self, messages: List[BaseMessage]) -> str:
        """
        异步直接调用（接收 LangChain 消息对象，供 LangGraph 节点使用）

        Args:
            messages: LangChain 消息列表

        Returns:
            生成的回答
        """
        response = await self._llm.ainvoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return answer


# 全局大模型服务实例
llm_service = LLMService()