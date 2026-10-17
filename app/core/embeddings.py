"""
Embedding 服务（DashScope / text-embedding-*）

相对旧版的关键修复：
- 懒加载：缺少 DASHSCOPE_API_KEY 时不在 import 阶段崩溃；
- 批量结果数量校验：API 返回条数与输入不一致时直接报错，避免
  “缓存 key 与向量错位”这种极难排查的问题；
- 缓存命中统计与清空接口。
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Dict, List, Optional

from langchain_community.embeddings import DashScopeEmbeddings

from app.config import settings
from app.core.cache import embedding_cache
from app.core.exceptions import EmbeddingException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """阿里百炼 Embedding 服务"""

    def __init__(self) -> None:
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._embeddings: Optional[DashScopeEmbeddings] = None
        self._lock = threading.Lock()

    # ---------- 懒加载 ----------
    @property
    def embeddings(self) -> DashScopeEmbeddings:
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    self._embeddings = DashScopeEmbeddings(
                        model=self.model,
                        dashscope_api_key=settings.require_dashscope_key(),
                    )
                    logger.info(
                        "LangChain Embedding 初始化完成，模型: %s, 维度: %s",
                        self.model,
                        self.dimension,
                    )
        return self._embeddings

    def is_configured(self) -> bool:
        return bool(settings.DASHSCOPE_API_KEY)

    # ---------- LangChain 标准接口 ----------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            logger.info(
                "向量化：%s/%s 条未命中缓存，调用 Embedding API",
                len(uncached_texts),
                len(texts),
            )
            try:
                embeddings = self.embeddings.embed_documents(uncached_texts)
            except Exception as exc:  # noqa: BLE001 - 统一转成业务异常
                raise EmbeddingException(f"Embedding 调用失败: {exc}") from exc

            if len(embeddings) != len(uncached_texts):
                raise EmbeddingException(
                    f"Embedding 返回数量不一致：期望 {len(uncached_texts)}，实际 {len(embeddings)}"
                )

            for original_index, emb in zip(uncached_indices, embeddings):
                results[original_index] = emb
                embedding_cache.set(texts[original_index], emb)
        else:
            logger.info("向量化：全部 %s 条命中缓存，跳过 API 调用", len(texts))

        missing = [i for i, item in enumerate(results) if item is None]
        if missing:
            raise EmbeddingException(f"Embedding 结果缺失，索引: {missing[:10]}")
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        cached = embedding_cache.get(text)
        if cached is not None:
            logger.debug("查询向量命中缓存: %s", text[:30])
            return cached

        try:
            embedding = self.embeddings.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingException(f"Embedding 调用失败: {exc}") from exc
        embedding_cache.set(text, embedding)
        return embedding

    # ---------- 兼容旧接口 ----------
    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def encode_single(self, text: str) -> List[float]:
        return self.embed_query(text)

    def get_dimension(self) -> int:
        return self.dimension

    def get_stats(self) -> Dict[str, object]:
        return embedding_cache.get_stats()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.embeddings import embedding_service``（懒加载）"""
    if name == "embedding_service":
        return get_embedding_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
