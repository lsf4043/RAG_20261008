"""
缓存模块

- ``EmbeddingCache``：文本 → 向量 的 LRU 缓存，减少 DashScope 调用；
- ``QueryCache``：问答结果缓存，key 覆盖问题 + 会话历史 + 检索参数，避免
  “不同会话的同一个问题互相串答案”；
- 知识库发生变化（上传 / 删除）时必须调用 ``invalidate_query_cache()``，
  这由 document_service 负责，避免答案与知识库不一致。
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any, List, Optional

from cachetools import LRUCache, TTLCache

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Embedding 结果缓存（LRU，key 为文本哈希）"""

    def __init__(self, maxsize: Optional[int] = None) -> None:
        self.maxsize = maxsize or settings.EMBEDDING_CACHE_SIZE
        self._cache: LRUCache = LRUCache(maxsize=self.maxsize)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        logger.info("Embedding 缓存初始化完成，容量: %s", self.maxsize)

    @staticmethod
    def _key(text: str) -> str:
        return _md5(text)

    def get(self, text: str) -> Optional[List[float]]:
        if not settings.CACHE_ENABLED:
            return None
        key = self._key(text)
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self.hits += 1
                return value
            self.misses += 1
            return None

    def set(self, text: str, embedding: List[float]) -> None:
        if not settings.CACHE_ENABLED:
            return
        with self._lock:
            self._cache[self._key(text)] = embedding

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{(self.hits / total * 100) if total else 0:.2f}%",
                "size": len(self._cache),
                "maxsize": self.maxsize,
            }


class QueryCache:
    """问答结果缓存（TTL + LRU）"""

    def __init__(self, maxsize: Optional[int] = None, ttl: Optional[int] = None) -> None:
        self.maxsize = maxsize or settings.QUERY_CACHE_SIZE
        self.ttl = ttl or settings.QUERY_CACHE_TTL
        self._cache: TTLCache = TTLCache(maxsize=self.maxsize, ttl=self.ttl)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        logger.info("查询缓存初始化完成，容量: %s, TTL: %ss", self.maxsize, self.ttl)

    @staticmethod
    def _history_digest(history: Optional[List[dict]]) -> str:
        """把会话历史折叠成摘要，保证多轮上下文不同则缓存不命中"""
        if not history:
            return ""
        parts = [f"{m.get('role', '')}:{m.get('content', '')}" for m in history]
        return _md5("\n".join(parts))[:16]

    def _key(self, question: str, top_k: int, history: Optional[List[dict]]) -> str:
        return _md5(f"{question}::{top_k}::{self._history_digest(history)}")

    def get(self, question: str, top_k: int, history: Optional[List[dict]] = None) -> Optional[Any]:
        if not settings.CACHE_ENABLED:
            return None
        key = self._key(question, top_k, history)
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self.hits += 1
                logger.debug("查询缓存命中: %s", question[:30])
                return value
            self.misses += 1
            return None

    def set(self, question: str, top_k: int, result: Any, history: Optional[List[dict]] = None) -> None:
        if not settings.CACHE_ENABLED:
            return
        with self._lock:
            self._cache[self._key(question, top_k, history)] = result

    def clear(self) -> None:
        """知识库变更后必须调用，否则会返回旧知识下的答案"""
        with self._lock:
            self._cache.clear()
            logger.info("查询缓存已清空（知识库变更）")

    def get_stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{(self.hits / total * 100) if total else 0:.2f}%",
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
            }


embedding_cache = EmbeddingCache()
query_cache = QueryCache()


def invalidate_query_cache() -> None:
    """知识库变更（上传/删除）后调用"""
    query_cache.clear()
