"""向量库的字段策略：静态 schema 需过滤 metadata，动态字段（历史集合）不能过滤"""
from types import SimpleNamespace

from langchain_core.documents import Document

from app.core.vector_store import SCALAR_FIELDS, VectorStoreService


def _service(dynamic: bool, fields):
    service = VectorStoreService()
    service._store = SimpleNamespace(enable_dynamic_field=dynamic)
    service._existing_fields = set(fields)
    return service


DYNAMIC_SCHEMA = {"id", "content", "embedding"}

# document_service 写入的 metadata 键：必须全部出现在静态 schema 声明里，
# 否则 langchain-milvus 在静态模式下会静默丢弃（不报错、不落库）
DOCUMENT_METADATA_KEYS = {
    "title",
    "raw_content",
    "document_id",
    "document_name",
    "file_type",
    "page_num",
    "chunk_id",
}


def test_scalar_fields_cover_all_document_metadata():
    assert set(SCALAR_FIELDS) == DOCUMENT_METADATA_KEYS


def test_indexed_fields_are_filter_targets_only():
    """只有真正参与 filter 的字段才建索引，避免白占内存"""
    indexed = {name for name, spec in SCALAR_FIELDS.items() if spec.get("index_type")}
    assert indexed == {"document_id", "file_type"}
    assert SCALAR_FIELDS["document_id"]["index_type"] == "INVERTED"
    assert SCALAR_FIELDS["file_type"]["index_type"] == "INVERTED"


def test_var_char_fields_have_enough_max_length():
    # Milvus 的 VARCHAR 超长是硬报错，不是截断，长度必须按最坏情况估
    assert SCALAR_FIELDS["document_id"]["max_length"] >= 36  # uuid4 带连字符
    assert SCALAR_FIELDS["document_name"]["max_length"] >= 255
    assert SCALAR_FIELDS["raw_content"]["max_length"] == 65535
    assert SCALAR_FIELDS["title"]["max_length"] >= 80


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
