"""
Milvus 集成测试（真实读写，验证 P0 级修复）

覆盖：
- 动态字段写入（metadata 不再被首个文档的 schema 锁死）
- ``raw_content`` 与 text_field(``content``) 不再冲突
- 按 document_id 删除**真的生效**（旧实现用不存在的 JSON 路径，静默失败）
- 文档列表聚合
- 重启后从向量库重建 BM25

需要可用的 Milvus；连不上时自动 skip：
    MILVUS_HOST=... MILVUS_PORT=19530 pytest -m integration

说明：Embedding 用确定性假实现（不消耗 DashScope 配额，也不依赖有效 Key）。
"""
from __future__ import annotations

import hashlib
import random

import pytest
from langchain_core.documents import Document

from app.config import settings

pytestmark = pytest.mark.integration

TEST_COLLECTION = "rag_integration_test"
DIM = settings.EMBEDDING_DIMENSION


class FakeEmbeddings:
    """确定性假 Embedding：同样的文本得到同样的向量"""

    def _vec(self, text: str):
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]

    def embed_documents(self, texts):
        return [self._vec(text) for text in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture()
def milvus_service(monkeypatch):
    """构造一个指向测试集合并使用假 Embedding 的 VectorStoreService"""
    from app.core.vector_store import VectorStoreService

    monkeypatch.setattr("app.core.vector_store.get_embedding_service", lambda: FakeEmbeddings())
    monkeypatch.setattr(settings, "MILVUS_COLLECTION", TEST_COLLECTION)

    service = VectorStoreService()
    service.collection_name = TEST_COLLECTION
    if not service.connect():
        pytest.skip("Milvus 不可用，跳过集成测试")

    # 清理上一次失败残留
    try:
        if service._store.client.has_collection(TEST_COLLECTION):
            service._store.client.drop_collection(TEST_COLLECTION)
    except Exception:  # noqa: BLE001
        pass

    # 集合已被删掉：清掉刚才按残留集合算出的 schema 缓存
    service._existing_fields = None
    service._incompatible_reason = None
    assert service.connect()

    yield service

    try:
        service._store.client.drop_collection(TEST_COLLECTION)
    except Exception:  # noqa: BLE001
        pass


def _docs(document_id: str, count: int):
    return [
        Document(
            page_content=f"Milvus 向量检索测试片段 {i}",
            metadata={
                "raw_content": f"Milvus 向量检索测试片段 {i}",
                "title": "Milvus 说明",
                "document_id": document_id,
                "document_name": "集成测试文档",
                "file_type": "txt",
                "page_num": 1,
                "chunk_id": i,
            },
        )
        for i in range(count)
    ]


def test_add_search_delete_roundtrip(milvus_service):
    document_id = "doc-integration-1"
    milvus_service.add_documents(_docs(document_id, 3))

    # metadata 没有被丢弃（动态字段生效）
    documents = milvus_service.iter_documents()
    assert len(documents) == 3
    assert all(doc.metadata.get("document_id") == document_id for doc in documents)
    assert all(doc.metadata.get("raw_content") for doc in documents)
    assert all(doc.metadata.get("chunk_id") is not None for doc in documents)

    # 检索
    results = milvus_service.similarity_search_with_score("Milvus 向量检索测试片段 1", k=2)
    assert results, "检索不到刚写入的数据"

    # 文档列表聚合
    listed = milvus_service.get_document_list()
    assert len(listed) == 1
    assert listed[0]["chunk_count"] == 3

    # 删除（旧实现在这里是静默失败的）
    deleted = milvus_service.delete_by_document_id(document_id)
    assert deleted == 3

    assert milvus_service.get_document_list() == []
    assert milvus_service.iter_documents() == []


def test_delete_is_idempotent(milvus_service):
    assert milvus_service.delete_by_document_id("not-exist-doc") == 0


def test_rejects_unsafe_document_id(milvus_service):
    from app.core.exceptions import MilvusException

    with pytest.raises(MilvusException):
        milvus_service.delete_by_document_id('x" or id != "')


def test_bm25_can_be_rebuilt_from_store(milvus_service, monkeypatch):
    """模拟“服务重启”：从向量库重建 BM25"""
    from app.core.bm25_retriever import BM25RetrieverService

    milvus_service.add_documents(_docs("doc-integration-2", 2))

    service = BM25RetrieverService()
    service.rebuild(milvus_service.iter_documents())

    assert service.count == 2
    hits = service.search("Milvus 向量检索", top_k=2)
    assert hits, "重建后的 BM25 索引应能检索到内容"
    # 返回的是副本，改 metadata 不会污染索引
    hits[0].metadata["x"] = 1
    assert "x" not in service._retriever.docs[0].metadata
