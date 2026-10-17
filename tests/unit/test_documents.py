"""Document 工具：去重 key、副本隔离、正文取值"""
from app.core.documents import copy_document, doc_key, doc_text


def test_doc_key_distinguishes_same_chunk_id_on_different_pages(make_document):
    a = make_document("第一页内容", document_id="doc-1", chunk_id=0, page_num=1)
    b = make_document("第二页内容", document_id="doc-1", chunk_id=0, page_num=2)

    assert doc_key(a) != doc_key(b), "chunk_id 相同但页码不同的两篇文档不能被当成同一篇"


def test_doc_key_is_stable_across_processes(make_document):
    doc = make_document("没有元数据的正文")
    # 使用 sha1 而不是内置 hash()，保证跨进程稳定
    assert doc_key(doc) == doc_key(copy_document(doc))
    assert doc_key(doc).startswith("sha1:")


def test_copy_document_isolates_metadata(make_document):
    original = make_document("正文", document_id="d", chunk_id=1, page_num=1)
    copied = copy_document(original)
    copied.metadata["rerank_score"] = 0.9

    assert "rerank_score" not in original.metadata, "写入副本不能污染原始文档"
    assert copied.page_content == original.page_content


def test_doc_text_prefers_raw_content(make_document):
    doc = make_document("标题\n标题\n原始正文", raw_content="原始正文", title="标题")
    assert doc_text(doc) == "原始正文"


def test_doc_text_falls_back_to_page_content(make_document):
    doc = make_document("只有 page_content")
    assert doc_text(doc) == "只有 page_content"
