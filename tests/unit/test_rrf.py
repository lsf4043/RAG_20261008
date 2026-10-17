"""混合检索的加权 RRF 融合"""
from app.graph.nodes import _rrf_fusion


def test_rrf_dedupes_across_channels(make_document):
    shared = make_document("共享文档", document_id="d1", chunk_id=1, page_num=1)
    only_vec = make_document("仅向量命中", document_id="d2", chunk_id=1, page_num=1)
    only_bm25 = make_document("仅BM25命中", document_id="d3", chunk_id=1, page_num=1)

    fused = _rrf_fusion(
        [("vector", [shared, only_vec]), ("bm25", [shared, only_bm25])],
        top_k=10,
        weights={"vector": 1.0, "bm25": 1.0},
        k=60,
    )

    assert len(fused) == 3, "同一文档在多路命中应被合并，而不是重复出现"
    top = fused[0]
    assert "vector" in top.metadata["retrieval_channels"]
    assert "bm25" in top.metadata["retrieval_channels"]
    assert top.metadata["rrf_score"] > 0


def test_rrf_respects_channel_weights(make_document):
    doc_vec = make_document("向量第一名", document_id="v", chunk_id=1, page_num=1)
    doc_bm25 = make_document("BM25第一名", document_id="b", chunk_id=1, page_num=1)

    # 只保留 BM25 通道 → BM25 的文档排第一
    fused = _rrf_fusion(
        [("vector", [doc_vec]), ("bm25", [doc_bm25])],
        top_k=5,
        weights={"vector": 0.0, "bm25": 5.0},
        k=60,
    )
    assert fused[0].metadata["document_id"] == "b"

    # 反向
    fused2 = _rrf_fusion(
        [("vector", [doc_vec]), ("bm25", [doc_bm25])],
        top_k=5,
        weights={"vector": 5.0, "bm25": 0.0},
        k=60,
    )
    assert fused2[0].metadata["document_id"] == "v"


def test_rrf_truncates_to_top_k(make_document):
    docs = [make_document(f"doc-{i}", document_id=f"d{i}", chunk_id=1, page_num=1) for i in range(10)]
    fused = _rrf_fusion([("vector", docs)], top_k=3, weights={"vector": 1.0}, k=60)
    assert len(fused) == 3
    assert fused[0].metadata["document_id"] == "d0"


def test_rrf_does_not_mutate_input_metadata(make_document):
    doc = make_document("正文", document_id="d", chunk_id=1, page_num=1)
    _rrf_fusion([("vector", [doc])], top_k=1, weights={"vector": 1.0}, k=60)
    assert "rrf_score" not in doc.metadata, "融合分数只能写在副本上"
