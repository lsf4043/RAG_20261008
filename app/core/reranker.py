"""
重排序服务（Cross-Encoder）

修复点：
- 懒加载 + 线程安全初始化（失败后可重试，不会永久判死）；
- 支持 ``RERANKER_DEVICE``（cpu / cuda / cuda:N / mps），不再固定吃 CPU；
- 打分写入**文档副本**的 metadata，避免污染 BM25 索引里共享的 Document 对象；
- 分数异常（NaN / 维度不符）时整体降级到原顺序，而不是抛给上层。
"""
from __future__ import annotations

import math
import threading
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.core.documents import copy_document
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RerankerService:
    """Cross-Encoder 重排序服务"""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self._cross_encoder: Any = None
        self._initialized = False
        self._available = False
        self._lock = threading.Lock()

    # ---------- 初始化 ----------
    def initialize(self) -> bool:
        with self._lock:
            if self._initialized:
                return self._available
            try:
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder

                model_kwargs = {}
                device = (settings.RERANKER_DEVICE or "cpu").strip()
                if device and device != "auto":
                    model_kwargs["device"] = device

                logger.info("正在加载重排序模型: %s (device=%s)", self.model_name, device)
                self._cross_encoder = HuggingFaceCrossEncoder(
                    model_name=self.model_name,
                    model_kwargs=model_kwargs,
                )
                self._available = True
                logger.info("重排序模型加载成功")
            except Exception as exc:  # noqa: BLE001
                logger.error("重排序模型加载失败: %s，将降级为原始检索顺序", exc)
                self._cross_encoder = None
                self._available = False
            finally:
                self._initialized = True
            return self._available

    def is_available(self) -> bool:
        return self._available

    def is_initialized(self) -> bool:
        return self._initialized

    # ---------- 重排 ----------
    def rerank(
        self,
        documents: List[Document],
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return []

        limit = top_n or settings.RERANK_TOP_K
        if not self._initialized:
            self.initialize()

        if not self._available or self._cross_encoder is None:
            logger.warning("重排序模型不可用，返回原始顺序")
            return [copy_document(doc) for doc in documents[:limit]]

        try:
            pairs: List[Tuple[str, str]] = [(query, doc.page_content) for doc in documents]
            raw_scores = list(self._cross_encoder.score(pairs))
            if len(raw_scores) != len(documents):
                raise ValueError(f"重排分数数量不匹配: {len(raw_scores)} != {len(documents)}")

            scored: List[Tuple[Document, float]] = []
            for doc, score in zip(documents, raw_scores):
                value = float(score)
                if not math.isfinite(value):
                    raise ValueError("重排分数包含 NaN/Inf")
                copied = copy_document(doc)
                copied.metadata["rerank_score"] = value
                scored.append((copied, value))

            scored.sort(key=lambda item: item[1], reverse=True)
            results = [doc for doc, _ in scored[:limit]]
            logger.info("重排序完成，返回 %s 条结果", len(results))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("重排序失败，降级为原顺序: %s", exc)
            return [copy_document(doc) for doc in documents[:limit]]


_reranker_service: Optional[RerankerService] = None
_reranker_lock = threading.Lock()


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        with _reranker_lock:
            if _reranker_service is None:
                _reranker_service = RerankerService()
    return _reranker_service


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.reranker import reranker_service``"""
    if name == "reranker_service":
        return get_reranker_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
