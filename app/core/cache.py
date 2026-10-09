"""
缓存模块（升级版新增）
提供 Embedding 缓存 和 查询结果缓存，避免重复计算、降低 API 成本。

- EmbeddingCache：对相同文本的向量化结果做缓存（LRU），减少 DashScope Embedding API 调用
- QueryCache：对相同 query 的检索+生成结果做缓存（TTL + LRU），降低响应延迟
"""
import hashlib
from typing import List, Optional, Any
from threading import Lock

from cachetools import TTLCache, LRUCache

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Embedding 结果缓存（LRU，按文本内容哈希作为 key）"""

    def __init__(self, maxsize: Optional[int] = None):
        self.maxsize = maxsize or settings.EMBEDDING_CACHE_SIZE
        self._cache: LRUCache = LRUCache(maxsize=self.maxsize)
        self._lock = Lock()
        self.hits = 0       # 缓存命中次数（统计用）
        self.misses = 0     # 缓存未命中次数（统计用）
        logger.info(f"Embedding 缓存初始化完成，容量: {self.maxsize}")

    def _key(self, text: str) -> str:
        """根据文本内容生成缓存 key（MD5 哈希）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """查询缓存，命中返回向量，未命中返回 None"""
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

    def set(self, text: str, embedding: List[float]):
        """写入缓存"""
        if not settings.CACHE_ENABLED:
            return
        key = self._key(text)
        with self._lock:
            self._cache[key] = embedding

    def get_stats(self) -> dict:
        """获取缓存命中率统计"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self._cache),
        }


class QueryCache:
    """查询结果缓存（TTL + LRU，按 query+参数 哈希作为 key）"""

    def __init__(
        self,
        maxsize: Optional[int] = None,
        ttl: Optional[int] = None
    ):
        self.maxsize = maxsize or settings.QUERY_CACHE_SIZE
        self.ttl = ttl or settings.QUERY_CACHE_TTL
        self._cache: TTLCache = TTLCache(maxsize=self.maxsize, ttl=self.ttl)
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        logger.info(f"查询缓存初始化完成，容量: {self.maxsize}, TTL: {self.ttl}s")

    def _key(self, question: str, top_k: int) -> str:
        """根据问题 + top_k 生成缓存 key"""
        raw = f"{question}::{top_k}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def get(self, question: str, top_k: int) -> Optional[Any]:
        """查询缓存"""
        if not settings.CACHE_ENABLED:
            return None
        key = self._key(question, top_k)
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self.hits += 1
                logger.debug(f"查询缓存命中: {question[:30]}")
                return value
            self.misses += 1
            return None

    def set(self, question: str, top_k: int, result: Any):
        """写入缓存"""
        if not settings.CACHE_ENABLED:
            return
        key = self._key(question, top_k)
        with self._lock:
            self._cache[key] = result

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self._cache),
        }


# 全局缓存实例
embedding_cache = EmbeddingCache()
query_cache = QueryCache()