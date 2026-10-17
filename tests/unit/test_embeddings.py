"""Embedding：v4 维度控制、批处理、text_type/instruct、缓存与错误处理"""
import pytest
from pydantic import ValidationError

from app.config import Settings, settings
from app.core import embeddings as embeddings_module
from app.core.cache import embedding_cache
from app.core.exceptions import EmbeddingException


class FakeResponse:
    def __init__(self, vectors, status_code=200, code="", message=""):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.output = {
            "embeddings": [
                {"text_index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ]
        }


class FakeDashScope:
    """记录每次调用的参数，并按 dimension 返回等长向量"""

    def __init__(self, fail_status=None, fail_times=0):
        self.calls = []
        self.fail_status = fail_status
        self.fail_times = fail_times

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_status and self.fail_times > 0:
            self.fail_times -= 1
            return FakeResponse([], status_code=self.fail_status, code="Err", message="boom")
        dimension = kwargs.get("dimension") or 1024
        texts = kwargs["input"]
        vectors = [[float(len(text) % 7)] * dimension for text in texts]
        return FakeResponse(vectors)

    @property
    def batch_sizes(self):
        return [len(call["input"]) for call in self.calls]


@pytest.fixture()
def fake_dashscope(monkeypatch):
    import dashscope

    def install(**kwargs):
        fake = FakeDashScope(**kwargs)
        monkeypatch.setattr(dashscope.TextEmbedding, "call", fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    embedding_cache.clear()
    yield
    embedding_cache.clear()


def _service(monkeypatch, dimension=None, batch=None):
    if dimension is not None:
        monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", dimension)
    if batch is not None:
        monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", batch)
    return embeddings_module.EmbeddingService()


def test_batches_respect_v4_limit(fake_dashscope, monkeypatch):
    fake = fake_dashscope()
    service = _service(monkeypatch, dimension=1536, batch=10)

    texts = [f"文本-{i}" for i in range(25)]
    vectors = service.embed_documents(texts)

    assert len(vectors) == 25
    assert all(len(vector) == 1536 for vector in vectors), "维度必须与 EMBEDDING_DIMENSION 一致"
    assert fake.batch_sizes == [10, 10, 5], "v4 单请求上限 10 行，必须自动分批"
    assert all(call["model"] == "text-embedding-v4" for call in fake.calls)
    assert all(call["dimension"] == 1536 for call in fake.calls)
    assert all(call["text_type"] == "document" for call in fake.calls)
    assert all("instruct" not in call for call in fake.calls), "文档侧不应发送 instruct"


def test_query_uses_query_type_and_instruct(fake_dashscope, monkeypatch):
    fake = fake_dashscope()
    service = _service(monkeypatch, dimension=1024, batch=10)

    vector = service.embed_query("Milvus 支持哪些索引？")

    assert len(vector) == 1024
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["text_type"] == "query"
    assert call["dimension"] == 1024
    assert call["instruct"] == settings.EMBEDDING_QUERY_INSTRUCT


def test_instruct_can_be_disabled(fake_dashscope, monkeypatch):
    fake = fake_dashscope()
    monkeypatch.setattr(settings, "EMBEDDING_QUERY_INSTRUCT", "")
    service = _service(monkeypatch, dimension=1536)

    service.embed_query("问题")

    assert "instruct" not in fake.calls[0]


def test_cache_prevents_duplicate_calls(fake_dashscope, monkeypatch):
    fake = fake_dashscope()
    service = _service(monkeypatch, dimension=1536)

    service.embed_documents(["同一段文本", "同一段文本"])
    assert fake.batch_sizes == [1], "重复文本应只请求一次"

    service.embed_documents(["同一段文本"])
    assert len(fake.calls) == 1, "第二次应完全命中缓存"


def test_dimension_mismatch_is_reported(fake_dashscope, monkeypatch):
    """服务端返回 1024 维但配置是 1536：必须明确报错，而不是写坏 Milvus"""
    import dashscope

    def wrong_dimension(**kwargs):
        return FakeResponse([[0.1] * 1024])

    monkeypatch.setattr(dashscope.TextEmbedding, "call", wrong_dimension)
    service = _service(monkeypatch, dimension=1536)

    with pytest.raises(EmbeddingException) as exc:
        service.embed_documents(["文本"])
    assert "维度不符" in str(exc.value)
    assert "EMBEDDING_DIMENSION" in str(exc.value)


def test_client_error_is_not_retried(fake_dashscope, monkeypatch):
    fake = fake_dashscope(fail_status=401, fail_times=5)
    monkeypatch.setattr(settings, "MAX_RETRIES", 3)
    service = _service(monkeypatch, dimension=1536)

    with pytest.raises(EmbeddingException) as exc:
        service.embed_documents(["文本"])

    assert len(fake.calls) == 1, "鉴权/参数错误重试没有意义"
    assert "401" in str(exc.value)


def test_server_error_is_retried(fake_dashscope, monkeypatch):
    fake = fake_dashscope(fail_status=500, fail_times=5)
    monkeypatch.setattr(settings, "MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "RETRY_DELAY", 0.01)
    service = _service(monkeypatch, dimension=1536)

    with pytest.raises(EmbeddingException):
        service.embed_documents(["文本"])

    assert len(fake.calls) == 3, "5xx 应按 MAX_RETRIES 重试后才失败"


def test_config_rejects_dimension_not_supported_by_model():
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            EMBEDDING_MODEL="text-embedding-v4",
            EMBEDDING_DIMENSION=999,
            EMBEDDING_BATCH_SIZE=10,
        )
    assert "EMBEDDING_DIMENSION" in str(exc.value)


def test_config_rejects_batch_over_model_limit():
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            EMBEDDING_MODEL="text-embedding-v4",
            EMBEDDING_DIMENSION=1536,
            EMBEDDING_BATCH_SIZE=25,
        )
    assert "EMBEDDING_BATCH_SIZE" in str(exc.value)


def test_v4_dimension_options_are_all_accepted():
    for dimension in (2048, 1536, 1024, 768, 512, 256, 128, 64):
        parsed = Settings(
            _env_file=None,
            EMBEDDING_MODEL="text-embedding-v4",
            EMBEDDING_DIMENSION=dimension,
            EMBEDDING_BATCH_SIZE=10,
        )
        assert parsed.EMBEDDING_DIMENSION == dimension


def test_v2_rejects_v4_only_dimension():
    """回退到 v2 时，2048 这种 v4 专属维度必须被拒绝"""
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            EMBEDDING_MODEL="text-embedding-v2",
            EMBEDDING_DIMENSION=2048,
            EMBEDDING_BATCH_SIZE=10,
        )
