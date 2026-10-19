"""
大模型服务（DeepSeek，OpenAI 兼容接口）

相对旧版的关键修复：
- **不再在 import 阶段构造客户端**：改为首次使用时懒加载，单元测试 / /docs /
  健康检查不需要真实密钥；
- **不再每次调用都新建 ChatOpenAI**：按 (temperature, max_tokens) 复用
  ``model_copy`` 出来的轻量变体，底层 httpx 连接池共用；
- 统一超时与重试（LLM_TIMEOUT / LLM_MAX_RETRIES）；
- 提供 ``structured()`` 供 CRAG 批评分、Query 改写等场景做结构化输出。
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _to_text(content: Any) -> str:
    """兼容 OpenAI 返回 str 或 content blocks 两种情况"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


class LLMService:
    """DeepSeek 大模型服务"""

    def __init__(self) -> None:
        self.model = settings.DEEPSEEK_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        self._base: Any = None
        self._variants: Dict[tuple, Any] = {}
        # 可重入：即使将来有人在持锁路径里再取一次锁也不会自死锁
        self._lock = threading.RLock()

    # ---------- 客户端 ----------
    def is_configured(self) -> bool:
        return bool(settings.DEEPSEEK_API_KEY)

    def _create_base(self) -> Any:
        settings.require_deepseek_key()
        return init_chat_model(
            model=self.model,
            model_provider="deepseek",
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    @property
    def base(self) -> Any:
        """底层 ChatModel（懒加载 + 线程安全）"""
        if self._base is None:
            with self._lock:
                if self._base is None:
                    self._base = self._create_base()
                    logger.info("LangChain 大模型初始化完成，模型: %s", self.model)
        return self._base

    def get_llm(self) -> Any:
        return self.base

    def _variant(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Any:
        """按参数复用模型变体（共享底层连接池，不重复创建客户端）"""
        # 注意：必须先解析 base 再进入临界区。
        # 旧实现先 `with self._lock:` 再调 `self.base`，而 base 内部又要拿同一把锁，
        # 普通 Lock 不可重入 → 首次调用走 structured()/指定 temperature 时会自死锁
        # （表现为请求 120s 超时且没有任何日志）。
        base = self.base

        updates: Dict[str, Any] = {}
        if temperature is not None:
            updates["temperature"] = temperature
        if max_tokens is not None:
            updates["max_tokens"] = max_tokens
        if not updates:
            return base

        key = tuple(sorted(updates.items()))
        with self._lock:
            variant = self._variants.get(key)
            if variant is None:
                variant = base.model_copy(update=updates)
                self._variants[key] = variant
            return variant

    def structured(self, schema: Any, temperature: float = 0.0, max_tokens: Optional[int] = None) -> Any:
        """返回绑定了结构化输出 schema 的 runnable

        DeepSeek 的 deepseek-flash 默认开启思考模式，而思考模式既不支持强制
        ``tool_choice``（function_calling 路径会 400），也会忽略 temperature。
        因此结构化调用默认关闭思考，并显式指定 function_calling，避免
        json_schema 在 DeepSeek 上不可用。
        """
        llm = self._variant(temperature=temperature, max_tokens=max_tokens)
        if settings.LLM_STRUCTURED_DISABLE_THINKING:
            extra_body = dict(getattr(llm, "extra_body", None) or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            llm = llm.model_copy(update={"extra_body": extra_body})
        return llm.with_structured_output(schema, method="function_calling")

    # ---------- 消息转换 ----------
    def _to_messages(
        self, messages: Sequence[Dict[str, str]], system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        result: List[BaseMessage] = []
        if system_prompt:
            result.append(SystemMessage(content=system_prompt))
        for msg in messages:
            role = (msg or {}).get("role", "user")
            content = (msg or {}).get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        return result

    # ---------- 同步接口 ----------
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self._variant(temperature, max_tokens).invoke(
            self._to_messages(messages, system_prompt)
        )
        return _to_text(response.content)

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self._variant(temperature, max_tokens).invoke(list(messages))
        return _to_text(response.content)

    # ---------- 异步接口 ----------
    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self._variant(temperature, max_tokens).ainvoke(
            self._to_messages(messages, system_prompt)
        )
        return _to_text(response.content)

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self._variant(temperature, max_tokens).ainvoke(list(messages))
        return _to_text(response.content)

    async def astream(
        self,
        messages: Sequence[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._variant(temperature, max_tokens).astream(
            self._to_messages(messages, system_prompt)
        ):
            text = _to_text(chunk.content)
            if text:
                yield text

    async def astream_messages(
        self,
        messages: Sequence[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """对已构造好的 BaseMessage 列表做 token 级流式输出"""
        async for chunk in self._variant(temperature, max_tokens).astream(list(messages)):
            text = _to_text(chunk.content)
            if text:
                yield text


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


def __getattr__(name: str) -> Any:
    """兼容旧的 ``from app.core.llm import llm_service`` 写法（懒加载）"""
    if name == "llm_service":
        return get_llm_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
