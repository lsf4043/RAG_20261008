"""向量库的字段策略：动态字段模式不能过滤 metadata，静态 schema 才需要过滤"""
from types import SimpleNamespace

from langchain_core.documents import Document

from app.core.vector_store import VectorStoreService


def _service(dynamic: bool, fields):
    service = VectorStoreService()
    service._store = SimpleNamespace(enable_dynamic_field=dynamic)
    service._existing_fields = set(fields)
    return service


DYNAMIC_SCHEMA = {"id", "content", "embedding"}


def test_dynamic_fields_keep_all_metadata():
    service = _service(True, DYNAMIC_SCHEMA)
    doc = Document(
        page_content="正文",
        metadata={"document_id": "d1", "raw_content": "正文", "chunk_id": 0, "page_num": 0},
    )

    kept = service._fit_document(doc)
    assert kept.metadata == doc.metadata, "动态字段模式下不能丢弃任何 metadata"
    assert "document_id" in service._available_fields(["document_id", "document_name"])


def test_static_schema_filters_unknown_fields():
    service = _service(False, {"id", "content", "embedding", "document_id"})
    doc = Document(page_content="正文", metadata={"document_id": "d1", "raw_content": "正文"})

    kept = service._fit_document(doc)
    assert kept.metadata == {"document_id": "d1"}
    assert service._available_fields(["document_id", "raw_content"]) == ["document_id"]


def test_unknown_schema_does_not_filter():
    service = VectorStoreService()
    service._store = SimpleNamespace(enable_dynamic_field=False)
    service._existing_fields = None

    doc = Document(page_content="正文", metadata={"anything": 1})
    assert service._fit_document(doc).metadata == {"anything": 1}
    assert service._available_fields(["a", "b"]) == ["a", "b"]
