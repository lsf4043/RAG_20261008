"""
文档处理服务（LangChain 版）
保留原有文档解析层（复杂 PDF 三层解析、OCR、多格式），改造向量化与入库为 LangChain Document。

优化点（相对旧版）：
- 使用 LangChain Document + VectorStore 统一入库
- 保留标题权重增强（page_content 为标题加权拼接文本）
- 维护内存文档注册表，供 BM25 检索器同步
"""
import os
import uuid
import tempfile
import threading
from typing import List, Dict, Any, Optional, Callable

from langchain_core.documents import Document

from app.config import settings
from app.core.vector_store import vector_store_service
from app.core.bm25_retriever import bm25_retriever_service
from app.core.thread_pool import thread_pool
from app.parsers.unstructured_parser import UnstructuredParser
from app.parsers.ocr_parser import get_ocr_parser
from app.parsers.complex_pdf_parser import get_complex_pdf_parser
from app.parsers.word_parser import WordParser
from app.parsers.base import ParseResult
from app.utils.text_splitter import TextSplitter
from app.utils.logger import get_logger, ProcessLogger

logger = get_logger(__name__)

# 全局文档注册表（供 BM25 索引构建，与 Milvus 向量库同步）
_bm25_documents: List[Document] = []
_bm25_lock = threading.Lock()


def _sync_bm25_index():
    """将内存文档注册表同步到 BM25 检索器（重建索引）"""
    with _bm25_lock:
        bm25_retriever_service.rebuild(list(_bm25_documents))
    logger.info(f"BM25 索引已同步，文档数: {len(_bm25_documents)}")


class DocumentService:
    """文档处理服务"""

    def __init__(self):
        self.unstructured_parser = UnstructuredParser()
        self.word_parser = WordParser()

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        document_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_ocr: bool = False,
        log_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        处理文档（解析 → 分割 → 构造 LangChain Document → 向量化入库 → 更新 BM25）

        Args:
            file_content: 文件内容
            filename: 文件名
            document_name: 文档名称（可选）
            chunk_size: 分块大小（可选）
            chunk_overlap: 分块重叠（可选）
            use_ocr: 是否强制 OCR
            log_callback: 日志回调

        Returns:
            处理结果
        """
        process_logger = ProcessLogger()
        process_logger.start()

        def log(level, step, message, progress=0, step_name=""):
            entry = getattr(process_logger, level)(step, message, progress, step_name)
            if log_callback:
                log_callback(entry)

        try:
            document_id = str(uuid.uuid4())
            doc_name = document_name or os.path.splitext(filename)[0]
            file_ext = os.path.splitext(filename)[1].lower().replace('.', '')

            log("info", "receiving", f"开始接收文件: {filename} (大小: {len(file_content)/1024/1024:.2f}MB)", 5, "接收文件")

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name

                log("info", "detecting", f"检测到文件类型: {file_ext.upper()}", 10, "类型检测")

                # 解析文档
                log("info", "parsing", "正在解析文档...", 15, "文档解析")
                parse_results = await self._parse_document(tmp_path, file_ext, use_ocr, log)

                if not parse_results:
                    log("error", "parsing", "文档解析失败或内容为空", 20, "文档解析")
                    return {"success": False, "error": "文档解析失败或内容为空", "logs": process_logger.get_logs()}

                total_chars = sum(len(r.text) for r in parse_results)
                log("success", "parsing", f"解析完成，共提取 {total_chars} 字符", 40, "文档解析")

                # 文本分割
                log("info", "splitting", "正在分割文本...", 45, "文本分割")
                splitter = TextSplitter(
                    chunk_size=chunk_size or settings.CHUNK_SIZE,
                    chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
                )

                all_chunks = []
                for result in parse_results:
                    chunks = splitter.split_text(result.text, result.page_num)
                    all_chunks.extend(chunks)

                log("success", "splitting", f"分割完成，生成 {len(all_chunks)} 个文本块", 55, "文本分割")

                if not all_chunks:
                    return {"success": False, "error": "分割后无有效文本块", "logs": process_logger.get_logs()}

                # 构造 LangChain Document（page_content 为标题加权文本，正文存 metadata）
                log("info", "embedding", f"正在构造文档并向量化 {len(all_chunks)} 个文本块...", 60, "向量化")

                documents: List[Document] = []
                for chunk in all_chunks:
                    # 标题重复 2 次做权重增强（用于向量化与 BM25）
                    combined_text = f"{chunk.title}\n{chunk.title}\n{chunk.content}"
                    documents.append(Document(
                        page_content=combined_text,
                        metadata={
                            "title": chunk.title,
                            "content": chunk.content,      # 原始正文
                            "document_id": document_id,
                            "document_name": doc_name,
                            "file_type": file_ext,
                            "page_num": chunk.page_num,
                            "chunk_id": chunk.chunk_id,
                        },
                    ))

                # 向量化 + 入库（LangChain VectorStore 自动调用 embedding_service）
                log("info", "storing", "正在写入向量库...", 75, "存储向量")
                await thread_pool.run_sync(vector_store_service.add_documents, documents)
                log("success", "storing", f"存储完成，共 {len(documents)} 条数据", 85, "存储向量")

                # 同步 BM25 索引
                log("info", "indexing", "正在更新 BM25 索引...", 90, "构建索引")
                with _bm25_lock:
                    _bm25_documents.extend(documents)
                await thread_pool.run_sync(_sync_bm25_index)
                log("success", "indexing", "BM25 索引更新完成", 95, "构建索引")

                elapsed_time = process_logger.get_elapsed_time()
                log("success", "complete", f"处理完成! 总耗时: {elapsed_time:.2f}s", 100, "完成")

                return {
                    "success": True,
                    "document_id": document_id,
                    "document_name": doc_name,
                    "file_type": file_ext,
                    "total_chunks": len(all_chunks),
                    "total_characters": total_chars,
                    "process_time": elapsed_time,
                    "logs": process_logger.get_logs(),
                }

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {str(e)}")

        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}")
            log("error", "error", f"处理失败: {str(e)}", 0, "错误")
            return {"success": False, "error": str(e), "logs": process_logger.get_logs()}

    async def _parse_document(self, file_path, file_ext, use_ocr, log) -> List[ParseResult]:
        """解析文档（保留原有解析器调度逻辑）"""
        if file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
            log("info", "parsing", "检测到图片文件，使用 OCR 解析", 20, "文档解析")
            ocr_parser = get_ocr_parser()
            return await thread_pool.run_sync(ocr_parser.parse, file_path)

        if file_ext == 'pdf':
            if use_ocr:
                log("info", "parsing", "强制使用 OCR 解析 PDF", 20, "文档解析")
                ocr_parser = get_ocr_parser()
                return await thread_pool.run_sync(ocr_parser.parse, file_path)
            else:
                log("info", "parsing", "使用复杂 PDF 解析器（支持表格、图片、文本）", 20, "文档解析")
                complex_parser = get_complex_pdf_parser()
                return await thread_pool.run_sync(complex_parser.parse, file_path)

        if file_ext in ['doc', 'docx']:
            log("info", "parsing", f"使用 Word 解析器解析 {file_ext.upper()} 文件", 20, "文档解析")
            return await thread_pool.run_sync(self.word_parser.parse, file_path)

        log("info", "parsing", f"使用 Unstructured 解析 {file_ext.upper()} 文件", 20, "文档解析")
        return await thread_pool.run_sync(self.unstructured_parser.parse, file_path)

    async def get_document_list(self) -> List[Dict]:
        """获取文档列表"""
        # 从内存注册表聚合统计（避免依赖 Milvus 查询）
        doc_dict: Dict[str, Dict] = {}
        with _bm25_lock:
            for doc in _bm25_documents:
                metadata = doc.metadata or {}
                doc_id = metadata.get("document_id", "")
                if not doc_id:
                    continue
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = {
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name", "未知文档"),
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunk_count": 1,
                    }
                else:
                    doc_dict[doc_id]["chunk_count"] += 1
        # 若内存为空，回退到向量库查询
        if not doc_dict:
            return await thread_pool.run_sync(vector_store_service.get_document_list)
        return list(doc_dict.values())

    async def delete_document(self, document_id: str) -> Dict:
        """
        删除文档（同时清理向量库与内存 BM25 注册表）

        Args:
            document_id: 文档ID

        Returns:
            删除结果
        """
        # 从向量库删除
        await thread_pool.run_sync(vector_store_service.delete_by_document_id, document_id)

        # 从内存注册表删除并重建 BM25
        global _bm25_documents
        with _bm25_lock:
            before = len(_bm25_documents)
            _bm25_documents = [
                d for d in _bm25_documents
                if (d.metadata or {}).get("document_id") != document_id
            ]
            deleted = before - len(_bm25_documents)
        bm25_retriever_service.rebuild(list(_bm25_documents))

        return {"success": True, "deleted_chunks": deleted}


# 全局文档服务实例
document_service = DocumentService()