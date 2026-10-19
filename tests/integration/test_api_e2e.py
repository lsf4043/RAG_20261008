"""
端到端集成测试：上传 → 解析 → 切片 → 向量化 → 入库 → BM25 → 问答 → 删除

用确定性假 Embedding / 假 LLM，但 Milvus 是真实的，因此可以验证：
- 上传接口整条链路（含 chunk_id 编号、metadata 动态字段）
- 混合检索（向量 + BM25）与 CRAG 评分
- 问答接口返回引用来源、used_fallback=false
- 删除接口真的把向量删掉

需要可用的 Milvus；连不上自动 skip。运行：
    pytest -m integration
"""
from __future__ import annotations

import hashlib
import random

import pytest

from app.config import settings

pytestmark = pytest.mark.integration

TEST_COLLECTION = "rag_e2e_test"
DIM = settings.EMBEDDING_DIMENSION


class FakeEmbeddings:
    def _vec(self, text: str):
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """按 schema 名字返回不同结构；生成阶段返回固定答案"""

    def structured(self, schema, temperature=0.0, max_tokens=None):
        name = getattr(schema, "__name__", "")

        class _Runner:
            def invoke(self, messages):
                if name == "QueryPlan":
                    return {"rewritten_query": "Milvus 是什么", "variants": ["Milvus 向量数据库介绍"]}
                # 评分：前 10 篇都算相关（测试集很小）
                return {"grades": [{"index": i + 1, "relevant": True} for i in range(10)]}

        return _Runner()

    def invoke_messages(self, messages, temperature=None, max_tokens=None):
        return "根据知识库，Milvus 是一个开源向量数据库，常用于 RAG 的向量检索。"

    def chat(self, messages, system_prompt=None, temperature=None, max_tokens=None):
        return self.invoke_messages(messages)

    async def astream_messages(self, messages, temperature=None, max_tokens=None):
        for piece in ("根据知识库，", "Milvus 是一个开源向量数据库。"):
            yield piece


@pytest.fixture()
def e2e_env(monkeypatch):
    """准备一套指向测试集合的服务，并把 app 内部引用替换掉"""
    from app.core.bm25_retriever import BM25RetrieverService
    from app.core.vector_store import VectorStoreService
    import app.services.document_service as doc_module

    monkeypatch.setattr("app.core.vector_store.get_embedding_service", lambda: FakeEmbeddings())
    monkeypatch.setattr(settings, "MILVUS_COLLECTION", TEST_COLLECTION)
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(settings, "CRAG_MAX_GRADE_DOCS", 10)

    fake_llm = FakeLLM()
    monkeypatch.setattr("app.graph.nodes.get_llm_service", lambda: fake_llm)
    monkeypatch.setattr("app.core.query_processor.get_llm_service", lambda: fake_llm)
    # 流式生成走 chat_service 里的引用，也要替换
    monkeypatch.setattr("app.services.chat_service.get_llm_service", lambda: fake_llm)

    service = VectorStoreService()
    service.collection_name = TEST_COLLECTION
    if not service.connect():
        pytest.skip("Milvus 不可用，跳过端到端集成测试")

    try:
        if service._store.client.has_collection(TEST_COLLECTION):
            service._store.client.drop_collection(TEST_COLLECTION)
    except Exception:  # noqa: BLE001
        pass
    service._existing_fields = None
    service._incompatible_reason = None
    assert service.connect()

    bm25 = BM25RetrieverService()
    doc_module._registry.clear()

    monkeypatch.setattr("app.services.document_service.get_vector_store_service", lambda: service)
    monkeypatch.setattr("app.services.document_service.get_bm25_service", lambda: bm25)
    monkeypatch.setattr("app.graph.nodes.get_vector_store_service", lambda: service)
    monkeypatch.setattr("app.graph.nodes.get_bm25_service", lambda: bm25)

    yield {"service": service, "bm25": bm25, "doc_module": doc_module}

    try:
        service._store.client.drop_collection(TEST_COLLECTION)
    except Exception:  # noqa: BLE001
        pass


def test_full_rag_flow(e2e_env):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    content = (
        "Milvus 是一个开源的向量数据库，专为大规模向量相似度检索设计。\n"
        "它支持 HNSW、IVF 等索引类型，常用度量方式包括 COSINE、L2 和 IP。\n"
        "RAG 系统通常用 Milvus 存放文档切片的向量，并在查询时做近邻搜索。\n"
    ).encode("utf-8")

    # 1) 上传
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("milvus.txt", content, "text/plain")},
        data={"document_name": "Milvus 说明"},
    )
    assert response.status_code == 200, response.text
    upload = response.json()["data"]
    document_id = upload["document_id"]
    assert upload["total_chunks"] >= 1
    assert upload["total_characters"] > 0

    # 2) BM25 已同步
    assert e2e_env["bm25"].count >= 1

    # 3) 文档列表
    listed = client.get("/api/v1/documents/list").json()["data"]
    assert listed["total"] == 1
    assert listed["documents"][0]["document_id"] == document_id

    # 4) 问答：应命中知识库并返回来源
    answer = client.post("/api/v1/chat/query", json={"question": "Milvus 是什么？", "top_k": 3})
    assert answer.status_code == 200, answer.text
    payload = answer.json()["data"]
    assert payload["route"] == "rag", "知识类问题应走检索"
    assert payload["used_fallback"] is False
    assert payload["sources"], "应该返回引用来源"
    assert payload["sources"][0]["document_name"] == "Milvus 说明"
    assert "Milvus" in payload["answer"]

    # 4.1) 闲聊：不应走检索，也不应返回“未命中知识库”的拒答
    greeting = client.post("/api/v1/chat/query", json={"question": "你好", "top_k": 3})
    assert greeting.status_code == 200, greeting.text
    greeting_data = greeting.json()["data"]
    assert greeting_data["route"] == "direct", "闲聊应走通用直答"
    assert greeting_data["sources"] == []
    assert "知识库中没有检索到" not in greeting_data["answer"]

    # 5) 再问一次：应命中查询缓存
    cached = client.post("/api/v1/chat/query", json={"question": "Milvus 是什么？", "top_k": 3})
    assert cached.json()["data"]["cached"] is True

    # 6) 删除（同时应让查询缓存失效，避免继续返回旧答案）
    deleted = client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["data"]["deleted_chunks"] >= 1

    assert client.get("/api/v1/documents/list").json()["data"]["total"] == 0
    assert e2e_env["service"].iter_documents() == []

    after_delete = client.post("/api/v1/chat/query", json={"question": "Milvus 是什么？", "top_k": 3})
    assert after_delete.json()["data"]["cached"] is False, "知识库变了，不能继续命中旧缓存"
    assert after_delete.json()["data"]["used_fallback"] is True


def test_stream_endpoint(e2e_env):
    """SSE 流式接口：复用检索子图 + token 级流式"""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    content = "Milvus 支持 HNSW 索引与 COSINE 相似度。".encode("utf-8")
    uploaded = client.post(
        "/api/v1/documents/upload",
        files={"file": ("stream.txt", content, "text/plain")},
        data={"document_name": "流式测试"},
    )
    assert uploaded.status_code == 200, uploaded.text

    response = client.post(
        "/api/v1/chat/stream", json={"question": "Milvus 支持什么索引？", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.text

    assert "event: start" in body
    assert "event: progress" in body
    assert "event: sources" in body
    assert "event: complete" in body
    assert "Milvus" in body

    # 闲聊走流式直答：不应出现检索进度，complete 里 route=direct
    greeting = client.post("/api/v1/chat/stream", json={"question": "你好", "top_k": 3})
    assert greeting.status_code == 200
    greeting_body = greeting.text
    assert '"route": "direct"' in greeting_body
    assert "event: answer" in greeting_body
    assert "正在检索知识库" not in greeting_body
    # 生成阶段是 token 级流式：至少出现两次 answer 事件
    assert body.count("event: answer") >= 2
