"""
Embedding 服务（阿里百炼 DashScope / text-embedding-v3、v4）

为什么不再用 ``langchain_community.DashScopeEmbeddings``：
该类 ``model_config = ConfigDict(extra="forbid")``，只向 SDK 传 ``input`` /
``text_type`` / ``model``，**无法传 ``dimension`` / ``output_type`` / ``instruct``**。
而 text-embedding-v4 的默认维度是 1024，如果只改模型名不改封装，就会拿到 1024 维向量，
与 ``EMBEDDING_DIMENSION``（如 1536）和 Milvus 集合 schema 不一致，写入直接失败。

因此这里直接调用 ``dashscope.TextEmbedding.call``，以便：
- 指定输出维度（v4 支持 2048/1536/1024/768/512/256/128/64）；
- 区分 ``text_type``（query / document）；
- 查询侧附加 ``instruct`` 任务指令（Qwen3-Embedding 检索增强用法）；
- 按模型上限分批（v3/v4 每批 10 行，v1/v2 每批 25 行）；
- 复用项目统一的 Embedding 缓存，并做维度与数量校验。
"""
from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.embeddings import Embeddings

from app.config import EMBEDDING_MODEL_SPECS, settings
from app.core.cache import embedding_cache
from app.core.exceptions import EmbeddingException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class _EmbeddingCallError(EmbeddingException):
    """内部使用：携带「是否可重试」标记"""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class DashScopeEmbeddingClient:
    """DashScope 文本向量客户端（薄封装，便于单测替换）"""

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = int(dimension or settings.EMBEDDING_DIMENSION)
        spec = EMBEDDING_MODEL_SPECS.get(self.model, {})
        self.max_batch = int(spec.get("max_batch", 10))
        self.supports_instruct = bool(spec.get("instruct", False))
        self._api_key = api_key or settings.require_dashscope_key()
        self.calls = 0  # 观测/测试用：实际发出的请求次数

    # ---------- 分批 ----------
    @property
    def batch_size(self) -> int:
        """实际使用的批大小（配置值不超过模型上限）"""
        return max(1, min(settings.EMBEDDING_BATCH_SIZE, self.max_batch))

    # ---------- 单次请求 ----------
    def _call_once(
        self,
        texts: Sequence[str],
        text_type: str,
        instruct: Optional[str] = None,
    ) -> List[List[float]]:
        import dashscope  # 延迟导入：未配置 Embedding 时不影响服务启动

        try:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=list(texts),
                api_key=self._api_key,
                text_type=text_type,
                dimension=self.dimension,
                output_type="dense",
                **({"instruct": instruct} if instruct else {}),
            )
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权异常统一转业务异常
            raise _EmbeddingCallError(f"Embedding 请求异常: {exc}", retryable=True) from exc

        self.calls += 1
        status = getattr(response, "status_code", None)
        if status != 200:
            message = (
                f"DashScope Embedding 调用失败（HTTP {status}）："
                f"{getattr(response, 'code', '')} {getattr(response, 'message', '')}"
            ).strip()
            # 429 限流与 5xx 可重试；4xx 参数/鉴权错误重试无意义
            retryable = status == 429 or (isinstance(status, int) and status >= 500)
            raise _EmbeddingCallError(message, retryable=retryable)

        output = getattr(response, "output", None) or {}
        items = output.get("embeddings") or []
        if items and isinstance(items[0], dict) and "text_index" in items[0]:
            items = sorted(items, key=lambda item: item.get("text_index", 0))

        vectors = [item.get("embedding") for item in items if isinstance(item, dict)]
        if len(vectors) != len(texts):
            raise _EmbeddingCallError(
                f"Embedding 返回数量不一致：期望 {len(texts)}，实际 {len(vectors)}",
                retryable=False,
            )
        for vector in vectors:
            if vector is None:
                raise _EmbeddingCallError("Embedding 返回结构异常：缺少 embedding 字段", retryable=False)
            if len(vector) != self.dimension:
                raise _EmbeddingCallError(
                    f"Embedding 维度不符：期望 {self.dimension}，实际 {len(vector)}。"
                    f"请检查 EMBEDDING_DIMENSION 是否与 {self.model} 及 Milvus 集合一致",
                    retryable=False,
                )
        return vectors  # type: ignore[return-value]

    # ---------- 带重试的分批调用 ----------
    def _with_retry(self, func, batch_len: int):
        last_error: Optional[_EmbeddingCallError] = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                return func()
            except _EmbeddingCallError as exc:
                last_error = exc
                if not exc.retryable or attempt >= settings.MAX_RETRIES:
                    raise
                delay = settings.RETRY_DELAY * attempt
                logger.warning(
                    "Embedding 批次（%s 条）第 %s 次失败：%s，%.1fs 后重试",
                    batch_len,
                    attempt,
                    str(exc)[:200],
                    delay,
                )
                time.sleep(delay)
        raise last_error or EmbeddingException("Embedding 调用失败")

    def embed(
        self,
        texts: Sequence[str],
        text_type: str = "document",
        instruct: Optional[str] = None,
    ) -> List[List[float]]:
        """分批向量化，返回顺序与输入一致"""
        if not texts:
            return []

        vectors: List[List[float]] = []
        batch_size = self.batch_size
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            vectors.extend(
                self._with_retry(
                    lambda batch=batch: self._call_once(batch, text_type, instruct),
                    batch_len=len(batch),
                )
            )
        return vectors


class EmbeddingService(Embeddings):
    """Embedding 服务：缓存 + 校验 + 批处理"""

    def __init__(self) -> None:
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._client: Optional[DashScopeEmbeddingClient] = None
        self._lock = threading.Lock()

    # ---------- 懒加载 ----------
    @property
    def client(self) -> DashScopeEmbeddingClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = DashScopeEmbeddingClient()
                    logger.info(
                        "Embedding 客户端初始化完成，模型: %s, 维度: %s, 批大小: %s, instruct: %s",
                        self._client.model,
                        self._client.dimension,
                        self._client.batch_size,
                        self._client.supports_instruct and bool(settings.EMBEDDING_QUERY_INSTRUCT),
                    )
        return self._client

    @property
    def embeddings(self) -> "EmbeddingService":
        """兼容旧属性（RAGAS 的 LangchainEmbeddingsWrapper 需要 embed_documents/embed_query）"""
        return self

    def is_configured(self) -> bool:
        return bool(settings.DASHSCOPE_API_KEY)

    # ---------- LangChain 标准接口 ----------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        # 同一批次内的重复文本（页眉页脚、模板句很常见）只向量化一次
        unique_texts: List[str] = []
        unique_index: Dict[str, int] = {}
        targets: List[tuple] = []

        for index, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                results[index] = cached
                continue
            if text not in unique_index:
                unique_index[text] = len(unique_texts)
                unique_texts.append(text)
            targets.append((index, text))

        if unique_texts:
            logger.info(
                "向量化：%s/%s 条未命中缓存（去重后 %s 条），模型 %s，批大小 %s",
                len(targets),
                len(texts),
                len(unique_texts),
                self.client.model,
                self.client.batch_size,
            )
            vectors = self.client.embed(unique_texts, text_type="document")
            if len(vectors) != len(unique_texts):
                raise EmbeddingException(
                    f"Embedding 返回数量不一致：期望 {len(unique_texts)}，实际 {len(vectors)}"
                )
            by_text = dict(zip(unique_texts, vectors))
            for text, vector in by_text.items():
                embedding_cache.set(text, vector)
            for index, text in targets:
                results[index] = by_text[text]
        else:
            logger.info("向量化：全部 %s 条命中缓存，跳过 API 调用", len(texts))

        missing = [index for index, item in enumerate(results) if item is None]
        if missing:
            raise EmbeddingException(f"Embedding 结果缺失，索引: {missing[:10]}")
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        cached = embedding_cache.get(text)
        if cached is not None:
            logger.debug("查询向量命中缓存: %s", text[:30])
            return cached

        instruct = None
        if self.client.supports_instruct and settings.EMBEDDING_QUERY_INSTRUCT:
            instruct = settings.EMBEDDING_QUERY_INSTRUCT

        vectors = self.client.embed([text], text_type="query", instruct=instruct)
        if not vectors:
            raise EmbeddingException("查询向量化失败：返回为空")
        vector = vectors[0]
        embedding_cache.set(text, vector)
        return vector

    # ---------- 兼容旧接口 ----------
    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def encode_single(self, text: str) -> List[float]:
        return self.embed_query(text)

    def get_dimension(self) -> int:
        return self.dimension

    def get_stats(self) -> Dict[str, Any]:
        stats = dict(embedding_cache.get_stats())
        stats.update({"model": self.model, "dimension": self.dimension})
        return stats


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.embeddings import embedding_service``（懒加载）"""
    if name == "embedding_service":
        return get_embedding_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
