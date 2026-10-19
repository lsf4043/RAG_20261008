"""重排序：qwen3.7-text-rerank 优先、异常降级本地、熔断与参数校验"""
from langchain_core.documents import Document

import pytest

from app.config import settings
from app.core import reranker as reranker_module
from app.core.reranker import DashScopeReRankClient, ReRankAPIError, RerankerService


class FakeRerankResponse:
    def __init__(self, results, status_code=200, code="", message=""):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.output = {"results": results} if status_code == 200 else None


class FakeDashScope:
    """记录调用参数，按给定顺序返回 relevance_score"""

    def __init__(self, order=None, exc=None, status_code=200):
        self.calls = []
        self.order = order
        self.exc = exc
        self.status_code = status_code

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        if self.status_code != 200:
            return FakeRerankResponse([], status_code=self.status_code, code="Err", message="boom")
        documents = kwargs["documents"]
        order = self.order or list(range(len(documents)))
        results = [
            {"index": index, "relevance_score": 1.0 - position * 0.1}
            for position, index in enumerate(order)
        ]
        return FakeRerankResponse(results)


class FakeAPIClient:
    def __init__(self, pairs=None, exc=None):
        self.pairs = pairs or []
        self.exc = exc
        self.calls = []

    def rerank(self, query, documents, top_n=None):
        self.calls.append({"query": query, "documents": list(documents), "top_n": top_n})
        if self.exc is not None:
            raise self.exc
        return list(self.pairs)


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def score(self, pairs):
        self.calls += 1
        return self.scores[: len(pairs)]


def _documents(count=3):
    return [
        Document(
            page_content=f"文档内容 {i}",
            metadata={"document_id": "doc-1", "chunk_id": i},
        )
        for i in range(count)
    ]


def _local_service(scores=None, api_client=None):
    service = RerankerService(api_client=api_client)
    service._initialized = True
    service._available = True
    service._cross_encoder = FakeCrossEncoder(scores or [0.1, 0.9, 0.5])
    return service


@pytest.fixture()
def fake_dashscope(monkeypatch):
    import dashscope

    def install(**kwargs):
        fake = FakeDashScope(**kwargs)
        monkeypatch.setattr(dashscope.TextReRank, "call", fake)
        return fake

    return install


def test_api_success_sorts_and_writes_score(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dashscope")
    api = FakeAPIClient([(2, 0.9), (0, 0.5), (1, 0.1)])
    service = _local_service(api_client=api)
    docs = _documents()

    results = service.rerank(docs, "问题", top_n=2)

    assert [doc.metadata["chunk_id"] for doc in results] == [2, 0]
    assert results[0].metadata["rerank_score"] == 0.9
    assert api.calls[0]["top_n"] == 3, "应请求全部候选的分数后再截断"
    assert "rerank_score" not in docs[0].metadata, "不能污染原始 Document"
    assert service._cross_encoder.calls == 0, "API 正常时不应触发本地模型"


def test_api_exception_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dashscope")
    api = FakeAPIClient(exc=ReRankAPIError("boom"))
    service = _local_service(scores=[0.2, 0.9, 0.4], api_client=api)

    results = service.rerank(_documents(), "问题")

    assert [doc.metadata["chunk_id"] for doc in results] == [1, 2, 0]
    assert service._cross_encoder.calls == 1
    assert service._api_failed_until > 0, "失败后应进入熔断冷却"


def test_cooldown_skips_api_after_failure(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dashscope")
    monkeypatch.setattr(settings, "RERANK_API_COOLDOWN", 60)
    api = FakeAPIClient(exc=ReRankAPIError("boom"))
    service = _local_service(api_client=api)

    service.rerank(_documents(), "问题")
    service.rerank(_documents(), "问题")

    assert len(api.calls) == 1, "冷却期内应直接走本地，避免反复等待超时"
    assert service._cross_encoder.calls == 2


def test_provider_local_never_calls_api(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "local")
    api = FakeAPIClient(exc=ReRankAPIError("should not be called"))
    service = _local_service(api_client=api)

    service.rerank(_documents(), "问题")

    assert api.calls == []
    assert service._cross_encoder.calls == 1


def test_http_error_falls_back_to_local(monkeypatch, fake_dashscope):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dashscope")
    fake_dashscope(status_code=400)
    service = _local_service(scores=[0.1, 0.9, 0.5])

    results = service.rerank(_documents(), "问题")

    assert [doc.metadata["chunk_id"] for doc in results] == [1, 2, 0], "400 时必须降级本地"
    assert service._api_failed_until > 0


def test_api_client_parses_and_validates(monkeypatch, fake_dashscope):
    monkeypatch.setattr(settings, "RERANK_API_TIMEOUT", 3)
    monkeypatch.setattr(settings, "RERANK_API_INSTRUCT", "custom instruction")
    fake = fake_dashscope(order=[1, 0])

    client = DashScopeReRankClient()
    scored = client.rerank("问题", ["a", "b"], top_n=2)

    assert scored == [(1, 1.0), (0, 0.9)]
    call = fake.calls[0]
    assert call["model"] == settings.RERANK_API_MODEL
    assert call["top_n"] == 2
    assert call["request_timeout"] == 3
    assert call["instruct"] == "custom instruction"


def test_api_response_count_mismatch_is_rejected(monkeypatch, fake_dashscope):
    fake_dashscope(order=[0])  # 3 篇文档只返回 1 条
    client = DashScopeReRankClient()

    with pytest.raises(ReRankAPIError) as exc:
        client.rerank("问题", ["a", "b", "c"], top_n=3)
    assert "数量不一致" in str(exc.value)


def test_api_documents_are_truncated(monkeypatch, fake_dashscope):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dashscope")
    monkeypatch.setattr(settings, "RERANK_API_MAX_CHARS", 100)
    fake = fake_dashscope(order=[0])
    service = _local_service()
    service._api_client = DashScopeReRankClient()

    service.rerank([Document(page_content="长" * 500, metadata={"chunk_id": 0})], "问题")

    assert len(fake.calls[0]["documents"][0]) == 100


def test_api_max_docs_limits_request_size(monkeypatch, fake_dashscope):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "dashscope")
    monkeypatch.setattr(settings, "RERANK_API_MAX_DOCS", 2)
    fake = fake_dashscope(order=[0, 1])
    service = _local_service()
    service._api_client = DashScopeReRankClient()

    results = service.rerank(_documents(5), "问题", top_n=5)

    assert len(fake.calls[0]["documents"]) == 2
    assert len(results) == 2, "未送入 API 的候选不参与本轮重排"
