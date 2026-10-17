"""缓存：key 覆盖会话历史、知识库变更后失效"""
from app.core.cache import EmbeddingCache, QueryCache, invalidate_query_cache, query_cache


def test_query_cache_misses_when_history_differs():
    cache = QueryCache(maxsize=10, ttl=60)
    history_a = [{"role": "user", "content": "什么是RAG？"}, {"role": "assistant", "content": "检索增强生成"}]
    history_b = [{"role": "user", "content": "什么是BM25？"}, {"role": "assistant", "content": "稀疏检索"}]

    cache.set("它有什么优点", 5, {"answer": "A"}, history_a)

    assert cache.get("它有什么优点", 5, history_a) == {"answer": "A"}
    assert cache.get("它有什么优点", 5, history_b) is None, "相同问题在不同会话上下文下不能串答案"


def test_query_cache_distinguishes_top_k():
    cache = QueryCache(maxsize=10, ttl=60)
    cache.set("问题", 5, {"answer": "top5"})
    assert cache.get("问题", 10) is None
    assert cache.get("问题", 5) == {"answer": "top5"}


def test_invalidate_query_cache_clears_global_cache():
    query_cache.set("某个问题", 5, {"answer": "x"}, [])
    assert query_cache.get("某个问题", 5, []) is not None

    invalidate_query_cache()
    assert query_cache.get("某个问题", 5, []) is None


def test_embedding_cache_roundtrip_and_clear():
    cache = EmbeddingCache(maxsize=2)
    cache.set("文本", [0.1, 0.2])
    assert cache.get("文本") == [0.1, 0.2]
    assert cache.get("另一个文本") is None

    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1

    cache.clear()
    assert cache.get("文本") is None
