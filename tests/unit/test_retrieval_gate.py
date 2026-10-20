"""检索门控：向量相似度短路 + 重排分数下限"""
from app.config import settings
from app.core.documents import copy_document
from app.graph.nodes import rerank_node, retrieve_node


class _FakeVectorStore:
    def __init__(self, scored):
        self._scored = scored
        self.calls = 0

    def similarity_search_with_score(self, query, k=5):
        self.calls += 1
        return self._scored(query, k)


class _FakeBM25:
    def __init__(self, ready=False, hits=None):
        self._ready = ready
        self._hits = hits or []

    def is_ready(self):
        return self._ready

    def search(self, query, top_k=5):
        return list(self._hits)

    @property
    def count(self):
        return len(self._hits)


def _patch(monkeypatch, vector_store, bm25):
    monkeypatch.setattr("app.graph.nodes.get_vector_store_service", lambda: vector_store)
    monkeypatch.setattr("app.graph.nodes.get_bm25_service", lambda: bm25)


def test_low_similarity_is_treated_as_no_knowledge(monkeypatch, make_document):
    doc = make_document("无关内容", document_id="d", page_num=1, chunk_id=0)
    _patch(monkeypatch, _FakeVectorStore(lambda q, k: [(doc, 0.16)]), _FakeBM25())
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_SCORE", 0.35)

    result = retrieve_node({"question": "重庆有啥吃的美食", "queries": ["重庆有啥吃的美食"], "notes": []})

    assert result["documents"] == []
    assert any("低于阈值" in note for note in result["notes"])


def test_high_similarity_keeps_candidates(monkeypatch, make_document):
    doc = make_document("Milvus 常用 HNSW", document_id="d", page_num=1, chunk_id=0)
    _patch(monkeypatch, _FakeVectorStore(lambda q, k: [(doc, 0.58)]), _FakeBM25())
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_SCORE", 0.35)

    result = retrieve_node({"question": "Milvus 用什么索引", "queries": ["Milvus 用什么索引"], "notes": []})

    assert result["documents"]
    assert not any("低于阈值" in note for note in result["notes"])


def test_gate_can_be_disabled(monkeypatch, make_document):
    doc = make_document("无关内容", document_id="d", page_num=1, chunk_id=0)
    _patch(monkeypatch, _FakeVectorStore(lambda q, k: [(doc, 0.05)]), _FakeBM25())
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_SCORE", 0.0)

    result = retrieve_node({"question": "问题", "queries": ["问题"], "notes": []})

    assert result["documents"], "阈值为 0 时应关闭门控"


def test_gate_does_not_drop_bm25_only_results(monkeypatch, make_document):
    """向量库返回为空（例如故障/无向量）时，不能用 0 分把 BM25 结果误杀"""
    doc = make_document("BM25 命中", document_id="d", page_num=1, chunk_id=0)
    _patch(monkeypatch, _FakeVectorStore(lambda q, k: []), _FakeBM25(ready=True, hits=[doc]))
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_SCORE", 0.35)

    result = retrieve_node({"question": "问题", "queries": ["问题"], "notes": []})

    assert result["documents"], "仅 BM25 有结果时不应被向量门控清空"


def test_gate_uses_best_score_across_query_variants(monkeypatch, make_document):
    """多 query 变体里只要有一个足够相似，就不能判为无关"""
    weak = make_document("弱相关", document_id="w", page_num=1, chunk_id=0)
    strong = make_document("强相关", document_id="s", page_num=1, chunk_id=0)

    def scored(query, k):
        return [(weak, 0.10)] if "弱" in query else [(strong, 0.52)]

    _patch(monkeypatch, _FakeVectorStore(scored), _FakeBM25())
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_SCORE", 0.35)

    result = retrieve_node(
        {"question": "问", "queries": ["弱说法", "强说法"], "notes": []}
    )

    assert result["documents"]


# ---------- 重排分数下限：判断“库里到底有没有答案”的主力信号 ----------
class _FakeReranker:
    """按预设分数返回副本，模拟 Cross-Encoder"""

    def __init__(self, scores):
        self._scores = scores

    def rerank(self, documents, query, top_n=None):
        ranked = []
        for doc, score in zip(documents, self._scores):
            copied = copy_document(doc)
            copied.metadata["rerank_score"] = score
            ranked.append(copied)
        ranked.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return ranked[: top_n or len(ranked)]


def test_rerank_filters_docs_below_min_score(monkeypatch, make_document):
    """“只提到实体但没有答案”的片段（如简历里的籍贯 宝鸡）必须被过滤"""
    docs = [
        make_document("简历提到籍贯陕西宝鸡", document_id="d", page_num=1, chunk_id=0),
        make_document("与问题无关的论文参考文献", document_id="d", page_num=2, chunk_id=1),
    ]
    monkeypatch.setattr("app.graph.nodes.get_reranker_service", lambda: _FakeReranker([0.39, 0.21]))
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_MIN_SCORE", 0.40)

    result = rerank_node(
        {"graded_documents": docs, "rewritten_query": "宝鸡有啥好吃的", "top_k": 5, "notes": []}
    )

    assert result["documents"] == [], "全部低于下限时应清空，交由生成节点走通用知识"
    assert any("RERANK_MIN_SCORE" in note for note in result["notes"])


def test_rerank_keeps_docs_above_min_score(monkeypatch, make_document):
    docs = [make_document("简历个人资料", document_id="d", page_num=1, chunk_id=0)]
    monkeypatch.setattr("app.graph.nodes.get_reranker_service", lambda: _FakeReranker([0.90]))
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_MIN_SCORE", 0.40)

    result = rerank_node(
        {"graded_documents": docs, "rewritten_query": "我的籍贯是哪里", "top_k": 5, "notes": []}
    )

    assert len(result["documents"]) == 1


def test_rerank_gate_disabled_when_zero(monkeypatch, make_document):
    docs = [make_document("弱相关", document_id="d", page_num=1, chunk_id=0)]
    monkeypatch.setattr("app.graph.nodes.get_reranker_service", lambda: _FakeReranker([0.05]))
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_MIN_SCORE", 0.0)

    result = rerank_node(
        {"graded_documents": docs, "rewritten_query": "问题", "top_k": 5, "notes": []}
    )

    assert len(result["documents"]) == 1


# ---------- 本地兜底通道的分数归一化 ----------
def test_local_logits_are_normalized_to_probability():
    """本地 CrossEncoder 输出 logits；必须过 sigmoid，否则 RERANK_MIN_SCORE 会误滤"""
    from app.core.reranker import sigmoid_to_probability

    assert abs(sigmoid_to_probability(0.0) - 0.5) < 1e-6, "logit 0 对应 p=0.5"
    assert sigmoid_to_probability(5.0) > 0.99
    assert sigmoid_to_probability(-5.0) < 0.01
    # 单调性：不改变排序
    assert sigmoid_to_probability(-3.0) < sigmoid_to_probability(1.0) < sigmoid_to_probability(3.0)
    # 极端值不溢出
    assert sigmoid_to_probability(10_000) == 1.0
    assert sigmoid_to_probability(-10_000) == 0.0
