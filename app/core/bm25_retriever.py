"""
BM25 稀疏检索服务

关键修复：**索引只构建一次**。旧实现每次 ``search()`` 都调用
``BM25Retriever.from_documents`` 重建 BM25Okapi，而一次问答有 3 个 query 变体，
等于每个请求做 3 次全量索引构建（文档上千时直接秒级卡顿）。
现在在 ``rebuild()`` 时构建一次，检索时只做 ``model_copy(update={"k": ...})``
（共享同一份倒排结构，零拷贝），并用锁保护索引替换。
"""
from __future__ import annotations

import threading
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.documents import copy_document
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BM25RetrieverService:
    """BM25 稀疏检索服务"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._retriever: Optional[BM25Retriever] = None
        self._count = 0
        self._lock = threading.RLock()
        logger.info("BM25 检索服务初始化完成 (k1=%s, b=%s)", k1, b)

    # ---------- 分词 ----------
    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词（中文按字、英文/数字按词）"""
        tokens: List[str] = []
        buffer: List[str] = []

        def flush() -> None:
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()

        for char in text or "":
            if "\u4e00" <= char <= "\u9fff":
                flush()
                tokens.append(char)
            elif char.isalnum():
                buffer.append(char)
            else:
                flush()
        flush()
        return tokens

    # ---------- 索引 ----------
    def rebuild(self, documents: List[Document]) -> None:
        """重建索引（文档库变化时调用一次）"""
        with self._lock:
            if not documents:
                self._retriever = None
                self._count = 0
                logger.info("BM25 索引已清空")
                return
            self._retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=self._tokenize,
                bm25_params={"k1": self.k1, "b": self.b, "epsilon": self.epsilon},
            )
            self._count = len(documents)
        logger.info("BM25 索引重建完成，文档数: %s", len(documents))

    def clear(self) -> None:
        self.rebuild([])

    @property
    def count(self) -> int:
        return self._count

    def is_ready(self) -> bool:
        return self._retriever is not None and self._count > 0

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        with self._lock:
            retriever = self._retriever
            count = self._count
        if retriever is None or count == 0:
            return []

        k = max(1, min(top_k, count))
        # model_copy 只换 k，不重建倒排索引
        snapshot = retriever.model_copy(update={"k": k})
        results = snapshot.invoke(query)
        logger.info("BM25 检索完成，返回 %s 条结果", len(results))
        return [copy_document(doc) for doc in results]

    def get_retriever(self, k: int = 5) -> Optional[BaseRetriever]:
        """获取 LangChain Retriever 视图（供需要 Retriever 的组合场景使用）"""
        with self._lock:
            if self._retriever is None:
                return None
            return self._retriever.model_copy(update={"k": max(1, min(k, self._count))})


_bm25_service: Optional[BM25RetrieverService] = None
_bm25_lock = threading.Lock()


def get_bm25_service() -> BM25RetrieverService:
    global _bm25_service
    if _bm25_service is None:
        with _bm25_lock:
            if _bm25_service is None:
                _bm25_service = BM25RetrieverService()
    return _bm25_service


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.bm25_retriever import bm25_retriever_service``"""
    if name == "bm25_retriever_service":
        return get_bm25_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
