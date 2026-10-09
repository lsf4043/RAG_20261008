"""
Embedding 服务（LangChain 版）
使用 LangChain 的 DashScopeEmbeddings 替代旧版手写 dashscope 调用。

优化点：
- 使用 LangChain 标准 Embedding 接口（embed_documents / embed_query），便于与 VectorStore 无缝集成
- 集成 EmbeddingCache，相同文本不重复调用 API，降低成本
- 保留标题权重增强能力（由上层 document_service 传入拼接文本）
"""
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings

from app.config import settings
from app.core.cache import embedding_cache
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """阿里百炼 Embedding 服务（基于 LangChain DashScopeEmbeddings）"""

    def __init__(self):
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        # 初始化 LangChain DashScopeEmbeddings，后续 embed_documents/embed_query 均可直接调用
        self._embeddings = DashScopeEmbeddings(
            model=self.model,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            # text-embedding-v2 固定 1536 维，无需指定 dims
        )
        logger.info(f"LangChain Embedding 初始化完成，模型: {self.model}, 维度: {self.dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化（LangChain 标准接口，带缓存）

        Args:
            texts: 文本列表

        Returns:
            向量列表（顺序与输入一致）
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts = []       # 未命中缓存的文本
        uncached_indices = []     # 未命中缓存的文本在原列表中的下标

        # 第一遍：逐条查询缓存
        for i, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 第二遍：对未命中的文本批量调用 API
        if uncached_texts:
            logger.info(f"向量化：{len(uncached_texts)}/{len(texts)} 条未命中缓存，调用 Embedding API")
            embeddings = self._embeddings.embed_documents(uncached_texts)
            for pos, (original_index, emb) in enumerate(zip(uncached_indices, embeddings)):
                results[original_index] = emb
                # 写入缓存（uncached_texts 与 uncached_indices 顺序一致）
                embedding_cache.set(uncached_texts[pos], emb)
        else:
            logger.info(f"向量化：全部 {len(texts)} 条命中缓存，跳过 API 调用")

        return results

    def embed_query(self, text: str) -> List[float]:
        """
        单条查询向量化（LangChain 标准接口，带缓存）

        Args:
            text: 查询文本

        Returns:
            向量
        """
        cached = embedding_cache.get(text)
        if cached is not None:
            logger.debug(f"查询向量命中缓存: {text[:30]}")
            return cached

        embedding = self._embeddings.embed_query(text)
        embedding_cache.set(text, embedding)
        return embedding

    # ---- 兼容旧接口的便捷方法 ----
    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量向量化（兼容旧版 encode 命名）"""
        return self.embed_documents(texts)

    def encode_single(self, text: str) -> List[float]:
        """单条向量化（兼容旧版 encode_single 命名）"""
        return self.embed_query(text)

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension


# 全局 Embedding 服务实例
embedding_service = EmbeddingService()