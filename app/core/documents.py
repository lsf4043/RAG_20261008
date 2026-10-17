"""
Document 相关的公共工具

集中解决三件事：
1. **稳定的去重 key**：旧实现用 ``document_id + chunk_id``，而 chunk_id 是按页从 0
   重新编号的，跨页必然重复 → 不同内容被 RRF 当成同一篇合并。现在把 page_num
   一起纳入 key，并用 sha1 兜底（``hash()`` 在不同进程间不稳定）。
2. **不可变复制**：BM25 检索器返回的是索引里持有的原始 Document 对象，节点若直接
   往 metadata 写分数，会污染索引本身并在并发请求间串数据。统一 copy 后再写。
3. **正文取值**：兼容 ``raw_content``（新）/ ``content``（旧）/ ``page_content``。
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from langchain_core.documents import Document

RAW_CONTENT_FIELD = "raw_content"


def copy_document(doc: Document) -> Document:
    """浅层复制一个 Document（metadata 也复制一份）"""
    return Document(
        page_content=doc.page_content,
        metadata=dict(doc.metadata or {}),
        id=getattr(doc, "id", None),
    )


def doc_key(doc: Document) -> str:
    """生成稳定的去重 key"""
    metadata: Dict[str, Any] = doc.metadata or {}
    document_id = metadata.get("document_id")
    chunk_id = metadata.get("chunk_id")
    if document_id is not None and chunk_id is not None:
        page = metadata.get("page_num", 0)
        return f"{document_id}|{page}|{chunk_id}"
    if getattr(doc, "id", None):
        return f"id:{doc.id}"
    digest = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()
    return f"sha1:{digest}"


def doc_text(doc: Document, fallback: str = "") -> str:
    """取文档正文（优先原始正文，其次加权文本）"""
    metadata: Dict[str, Any] = doc.metadata or {}
    return (
        metadata.get(RAW_CONTENT_FIELD)
        or metadata.get("content")
        or doc.page_content
        or fallback
    )


def doc_title(doc: Document) -> str:
    return str((doc.metadata or {}).get("title") or "")


def doc_name(doc: Document) -> str:
    return str((doc.metadata or {}).get("document_name") or "未知文档")


def doc_page(doc: Document) -> Optional[int]:
    page = (doc.metadata or {}).get("page_num")
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None
