"""
文档处理服务

本次修复：
1. **chunk_id 全局连续**：跨页调用 split_text 时传入 start_chunk_id，解决
   “不同页 chunk 撞号 → RRF 把不同内容当成同一篇”的问题；
2. **元数据字段改名** ``raw_content``：避免与 Milvus 的 text_field(``content``) 同名
   （同名会被 langchain-milvus 判定为保留字段，直接抛异常）；
3. **BM25 启动自愈**：``bootstrap_bm25()`` 在应用启动时从 Milvus 重建索引，
   重启后混合检索不再静默退化为纯向量；
4. **知识库变更即失效查询缓存**：上传 / 删除后清空 query_cache；
5. **删除不再假成功**：向量库删除失败会抛异常，返回真实删除条数；
6. **解析器懒加载**：没有安装 OCR 依赖时，服务仍可启动并提供问答/文本类文档。
"""
from __future__ import annotations

import os
import tempfile
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

from app.config import settings
from app.core.bm25_retriever import get_bm25_service
from app.core.cache import invalidate_query_cache
from app.core.exceptions import DocumentParseException, MilvusException, UnsupportedFileTypeException
from app.core.thread_pool import thread_pool
from app.core.vector_store import RAW_CONTENT_FIELD, get_vector_store_service
from app.parsers.base import ParseResult
from app.utils.logger import ProcessLogger, get_logger
from app.utils.text_splitter import TextSplitter

logger = get_logger(__name__)

# 内存文档注册表（与 Milvus 同步，用于 BM25 与文档列表兜底）
_registry: List[Document] = []
_registry_lock = threading.RLock()


def _sync_bm25(documents: Optional[List[Document]] = None) -> None:
    """把注册表同步到 BM25 检索器"""
    with _registry_lock:
        snapshot = list(documents if documents is not None else _registry)
    get_bm25_service().rebuild(snapshot)
    logger.info("BM25 索引已同步，文档数: %s", len(snapshot))


class DocumentService:
    """文档处理服务"""

    def __init__(self) -> None:
        self._unstructured_parser = None
        self._word_parser = None

    # ---------- 懒加载解析器 ----------
    @property
    def unstructured_parser(self):
        if self._unstructured_parser is None:
            from app.parsers.unstructured_parser import UnstructuredParser

            self._unstructured_parser = UnstructuredParser()
        return self._unstructured_parser

    @property
    def word_parser(self):
        if self._word_parser is None:
            from app.parsers.word_parser import WordParser

            self._word_parser = WordParser()
        return self._word_parser

    # ==================== 启动自愈 ====================
    async def bootstrap_bm25(self) -> int:
        """从向量库重建 BM25 索引（应用启动时调用）"""
        try:
            documents = await thread_pool.run_sync(
                get_vector_store_service().iter_documents,
                timeout=settings.MILVUS_TIMEOUT * 6,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("启动时读取向量库失败，BM25 索引保持为空（仅向量检索）: %s", exc)
            return 0

        with _registry_lock:
            _registry.clear()
            _registry.extend(documents)

        await thread_pool.run_sync(_sync_bm25, documents)
        return len(documents)

    # ==================== 处理文档 ====================
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        document_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_ocr: bool = False,
        skip_index: bool = False,
        log_callback: Optional[Callable[[dict], None]] = None,
    ) -> Dict[str, Any]:
        process_logger = ProcessLogger()
        process_logger.start()

        def log(level: str, step: str, message: str, progress: int = 0, step_name: str = "") -> None:
            entry = getattr(process_logger, level)(step, message, progress, step_name)
            if log_callback:
                try:
                    log_callback(entry)
                except Exception:  # noqa: BLE001
                    logger.debug("log_callback 执行失败", exc_info=True)

        file_ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
        if file_ext not in settings.allowed_extensions_list:
            raise UnsupportedFileTypeException(f"不支持的文件类型: {file_ext or '(无扩展名)'}")

        document_id = str(uuid.uuid4())
        doc_name = document_name or os.path.splitext(filename)[0]
        tmp_path: Optional[str] = None

        log("info", "receiving", f"开始接收文件: {filename} (大小: {len(file_content)/1024/1024:.2f}MB)", 5, "接收文件")

        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            log("info", "parsing", "正在解析文档...", 15, "文档解析")
            parse_results = await self._parse_document(tmp_path, file_ext, use_ocr, log)

            if not parse_results:
                raise DocumentParseException("文档解析失败或内容为空")

            total_chars = sum(len(r.text or "") for r in parse_results)
            log("success", "parsing", f"解析完成，共提取 {total_chars} 字符 / {len(parse_results)} 页", 40, "文档解析")

            log("info", "splitting", "正在分割文本...", 45, "文本分割")
            splitter = TextSplitter(
                chunk_size=chunk_size or settings.CHUNK_SIZE,
                chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
            )

            all_chunks = []
            for result in parse_results:
                # 关键：start_chunk_id 让编号跨页连续
                chunks = splitter.split_text(
                    result.text, result.page_num, start_chunk_id=len(all_chunks)
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                raise DocumentParseException("分割后无有效文本块")
            log("success", "splitting", f"分割完成，生成 {len(all_chunks)} 个文本块", 55, "文本分割")

            documents: List[Document] = []
            for chunk in all_chunks:
                # 标题重复两次做权重增强（仅在有真实标题时）
                embed_text = f"{chunk.title}\n{chunk.title}\n{chunk.content}" if chunk.title else chunk.content
                documents.append(
                    Document(
                        page_content=embed_text,
                        metadata={
                            "title": chunk.title,
                            RAW_CONTENT_FIELD: chunk.content,
                            "document_id": document_id,
                            "document_name": doc_name,
                            "file_type": file_ext,
                            "page_num": chunk.page_num,
                            "chunk_id": chunk.chunk_id,
                        },
                    )
                )

            log("info", "storing", f"正在写入向量库（{len(documents)} 条）...", 75, "存储向量")
            await thread_pool.run_sync(
                get_vector_store_service().add_documents,
                documents,
                timeout=settings.REQUEST_TIMEOUT,
            )
            log("success", "storing", "存储完成", 85, "存储向量")

            log("info", "indexing", "正在更新 BM25 索引...", 90, "构建索引")
            if skip_index:
                # 批量上传场景：先只登记，最后统一重建一次（避免 O(n²)）
                with _registry_lock:
                    _registry.extend(documents)
            else:
                await self._append_to_registry(documents)
            log("success", "indexing", "BM25 索引更新完成", 95, "构建索引")

            # 知识库变化 → 查询缓存必须失效
            invalidate_query_cache()

            elapsed = process_logger.get_elapsed_time()
            log("success", "complete", f"处理完成，总耗时 {elapsed:.2f}s", 100, "完成")

            return {
                "success": True,
                "document_id": document_id,
                "document_name": doc_name,
                "file_type": file_ext,
                "total_chunks": len(all_chunks),
                "total_characters": total_chars,
                "process_time": elapsed,
                "logs": process_logger.get_logs(),
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as exc:
                    logger.warning("删除临时文件失败: %s", exc)

    async def _append_to_registry(self, documents: List[Document]) -> None:
        with _registry_lock:
            _registry.extend(documents)
            snapshot = list(_registry)
        await thread_pool.run_sync(get_bm25_service().rebuild, snapshot)

    async def reindex(self) -> int:
        """按当前注册表重建 BM25 索引（批量上传结束后调用一次）"""
        with _registry_lock:
            snapshot = list(_registry)
        await thread_pool.run_sync(get_bm25_service().rebuild, snapshot)
        invalidate_query_cache()
        return len(snapshot)

    # ---------- 解析调度 ----------
    async def _parse_document(
        self, file_path: str, file_ext: str, use_ocr: bool, log: Callable
    ) -> List[ParseResult]:
        if file_ext in {"png", "jpg", "jpeg", "bmp", "tiff"}:
            log("info", "parsing", "图片文件，使用 OCR 解析", 20, "文档解析")
            from app.parsers.ocr_parser import get_ocr_parser

            parser = get_ocr_parser()
            return await thread_pool.run_sync(parser.parse, file_path, timeout=settings.REQUEST_TIMEOUT)

        if file_ext == "pdf":
            if use_ocr:
                log("info", "parsing", "强制 OCR 解析 PDF", 20, "文档解析")
                from app.parsers.ocr_parser import get_ocr_parser

                parser = get_ocr_parser()
                return await thread_pool.run_sync(parser.parse, file_path, timeout=settings.REQUEST_TIMEOUT)

            log("info", "parsing", "使用复杂 PDF 解析器（文本 + 表格 + 图片）", 20, "文档解析")
            from app.parsers.complex_pdf_parser import get_complex_pdf_parser

            parser = get_complex_pdf_parser()
            return await thread_pool.run_sync(parser.parse, file_path, timeout=settings.REQUEST_TIMEOUT)

        if file_ext in {"doc", "docx"}:
            log("info", "parsing", f"使用 Word 解析器解析 {file_ext.upper()}", 20, "文档解析")
            return await thread_pool.run_sync(self.word_parser.parse, file_path, timeout=settings.REQUEST_TIMEOUT)

        log("info", "parsing", f"使用 Unstructured 解析 {file_ext.upper()}", 20, "文档解析")
        return await thread_pool.run_sync(
            self.unstructured_parser.parse, file_path, timeout=settings.REQUEST_TIMEOUT
        )

    # ==================== 查询 / 删除 ====================
    async def get_document_list(self) -> List[Dict[str, Any]]:
        """优先从 Milvus 聚合（多进程/重启后仍准确），失败则回退内存注册表"""
        try:
            return await thread_pool.run_sync(
                get_vector_store_service().get_document_list,
                timeout=settings.MILVUS_TIMEOUT * 3,
            )
        except MilvusException as exc:
            logger.warning("从向量库获取文档列表失败，回退内存注册表: %s", exc)

        doc_dict: Dict[str, Dict[str, Any]] = {}
        with _registry_lock:
            for doc in _registry:
                metadata = doc.metadata or {}
                doc_id = metadata.get("document_id")
                if not doc_id:
                    continue
                item = doc_dict.get(doc_id)
                if item is None:
                    doc_dict[doc_id] = {
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name", "未知文档"),
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunk_count": 1,
                    }
                else:
                    item["chunk_count"] += 1
        return list(doc_dict.values())

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """删除文档（向量库 + 内存注册表 + 缓存）"""
        deleted_in_store = await thread_pool.run_sync(
            get_vector_store_service().delete_by_document_id,
            document_id,
            timeout=settings.MILVUS_TIMEOUT * 3,
        )

        with _registry_lock:
            before = len(_registry)
            kept = [d for d in _registry if (d.metadata or {}).get("document_id") != document_id]
            _registry[:] = kept
            deleted_in_registry = before - len(kept)

        await thread_pool.run_sync(get_bm25_service().rebuild, list(kept))
        invalidate_query_cache()

        return {
            "document_id": document_id,
            "deleted_chunks": deleted_in_store or deleted_in_registry,
            "deleted_in_registry": deleted_in_registry,
        }


_document_service: Optional[DocumentService] = None
_document_service_lock = threading.Lock()


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        with _document_service_lock:
            if _document_service is None:
                _document_service = DocumentService()
    return _document_service


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.services.document_service import document_service``"""
    if name == "document_service":
        return get_document_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
