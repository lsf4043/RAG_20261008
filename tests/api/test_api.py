"""API 层：状态码语义、参数校验、不泄露内部异常"""
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.services.chat_service import chat_service
from app.services.document_service import get_document_service


@pytest.fixture()
def client():
    """不进入 lifespan（避免测试时去连 Milvus / 下载重排模型）

    raise_server_exceptions=False：Starlette 在把 500 响应发给客户端后仍会向上抛出
    异常（用于让服务器日志记录栈），关掉它才能像真实客户端那样断言响应体。
    """
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_health_liveness(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["status"] in {"healthy", "degraded", "unhealthy"}
    assert "milvus" in body["data"]["services"]


def test_healthz_is_minimal(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_chat_rejects_invalid_top_k(client):
    response = client.post("/api/v1/chat/query", json={"question": "hi", "top_k": 999})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 422
    assert body["error"] == "VALIDATION_ERROR"


def test_chat_rejects_empty_question(client):
    response = client.post("/api/v1/chat/query", json={"question": ""})
    assert response.status_code == 422


def test_chat_success_envelope(client, monkeypatch):
    async def fake_query(**kwargs):
        return {
            "answer": "答案",
            "question": kwargs["question"],
            "sources": [],
            "model": "deepseek-chat",
            "process_time": 0.1,
            "logs": [],
            "session_id": None,
            "used_fallback": False,
            "degraded": False,
            "notes": [],
            "crag_iterations": 0,
            "rewritten_query": kwargs["question"],
            "cached": False,
        }

    monkeypatch.setattr(chat_service, "query", fake_query)
    response = client.post("/api/v1/chat/query", json={"question": "什么是 RAG"})

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["answer"] == "答案"
    assert body["data"]["used_fallback"] is False


def test_unhandled_error_does_not_leak_details(client, monkeypatch):
    async def boom(**kwargs):
        raise ValueError("database password is hunter2")

    monkeypatch.setattr(chat_service, "query", boom)
    response = client.post("/api/v1/chat/query", json={"question": "hi"})

    assert response.status_code == 500
    assert "hunter2" not in response.text


def test_delete_missing_document_returns_404(client, monkeypatch):
    async def fake_delete(document_id):
        return {"document_id": document_id, "deleted_chunks": 0, "deleted_in_registry": 0}

    monkeypatch.setattr(get_document_service(), "delete_document", fake_delete)
    response = client.delete("/api/v1/documents/not-exist")

    assert response.status_code == 404
    assert response.json()["error"] == "NOT_FOUND"


def test_delete_existing_document_returns_count(client, monkeypatch):
    async def fake_delete(document_id):
        return {"document_id": document_id, "deleted_chunks": 3, "deleted_in_registry": 3}

    monkeypatch.setattr(get_document_service(), "delete_document", fake_delete)
    response = client.delete("/api/v1/documents/doc-1")

    assert response.status_code == 200
    assert response.json()["data"]["deleted_chunks"] == 3


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert response.json()["error"] == "UNSUPPORTED_FILE_TYPE"


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE", 1)  # 1 MB
    payload = b"a" * (2 * 1024 * 1024)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.txt", payload, "text/plain")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "FILE_TOO_LARGE"


def test_upload_rejects_overlap_not_less_than_chunk_size(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"chunk_size": "200", "overlap": "200"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "VALIDATION_FAILED"


def test_documents_list_envelope(client, monkeypatch):
    async def fake_list():
        return [
            {"document_id": "d1", "document_name": "手册", "file_type": "pdf", "chunk_count": 4}
        ]

    monkeypatch.setattr(get_document_service(), "get_document_list", fake_list)
    response = client.get("/api/v1/documents/list")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["total"] == 1
    assert body["data"]["documents"][0]["chunk_count"] == 4
