"""
文档接口

修复点：
- 上传**流式读取 + 边读边校验大小**，不再把超大请求整包读进内存；
- 错误一律抛 RAGException（HTTP 状态码正确），不再 200 + body.code；
- 补齐 response_model，/docs 有真实 schema；
- 批量上传只在最后重建一次 BM25 索引（旧实现每个文件重建一次，O(n²)）；
- 删除不存在的文档返回 404，而不是“删除成功”。
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import require_admin_token
from app.config import settings
from app.core.exceptions import (
    FileTooLargeException,
    NotFoundException,
    UnsupportedFileTypeException,
    ValidationException,
)
from app.models.common import ApiResponse, success_response
from app.models.document import (
    BatchUploadData,
    BatchUploadItem,
    DeleteResultData,
    DocumentListData,
    UploadResultData,
)
from app.services.document_service import get_document_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["文档管理"])

_READ_CHUNK = 1024 * 1024


async def _read_upload_limited(file: UploadFile) -> bytes:
    """流式读取上传文件，超过 MAX_FILE_SIZE 立即中断"""
    max_bytes = settings.MAX_FILE_SIZE * 1024 * 1024
    buffer = bytearray()
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise FileTooLargeException(
                f"文件超过大小限制 {settings.MAX_FILE_SIZE}MB（已读取 {len(buffer) / 1024 / 1024:.1f}MB）"
            )
    if not buffer:
        raise ValidationException("上传文件为空")
    return bytes(buffer)


def _validate_extension(filename: Optional[str]) -> str:
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    if not ext or ext not in settings.allowed_extensions_list:
        raise UnsupportedFileTypeException(
            f"不支持的文件类型: {ext or '(无扩展名)'}；支持: {settings.ALLOWED_EXTENSIONS}"
        )
    return ext


@router.post(
    "/upload",
    response_model=ApiResponse[UploadResultData],
    dependencies=[Depends(require_admin_token)],
    summary="上传单个文档",
)
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    document_name: Optional[str] = Form(None, description="文档名称，默认取文件名"),
    chunk_size: Optional[int] = Form(None, description=f"分块大小，默认 {settings.CHUNK_SIZE}"),
    overlap: Optional[int] = Form(None, description=f"分块重叠，默认 {settings.CHUNK_OVERLAP}"),
    use_ocr: bool = Form(False, description="强制使用 OCR（扫描版 PDF）"),
):
    _validate_extension(file.filename)

    if chunk_size is not None and not (100 <= chunk_size <= 4000):
        raise ValidationException("chunk_size 需在 100~4000 之间")
    if overlap is not None and overlap < 0:
        raise ValidationException("overlap 不能为负数")
    if chunk_size is not None and overlap is not None and overlap >= chunk_size:
        raise ValidationException("overlap 必须小于 chunk_size")

    content = await _read_upload_limited(file)

    result = await get_document_service().process_document(
        file_content=content,
        filename=file.filename,
        document_name=document_name,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        use_ocr=use_ocr,
    )
    return success_response(data=result, message="文档上传成功")


@router.post(
    "/upload/batch",
    response_model=ApiResponse[BatchUploadData],
    dependencies=[Depends(require_admin_token)],
    summary="批量上传文档",
)
async def upload_documents_batch(
    files: List[UploadFile] = File(..., description="上传的多个文件"),
    chunk_size: Optional[int] = Form(None, description="分块大小"),
    overlap: Optional[int] = Form(None, description="分块重叠"),
    use_ocr: bool = Form(False, description="强制使用 OCR"),
):
    service = get_document_service()
    results: List[BatchUploadItem] = []
    success_count = 0
    failed_count = 0

    for file in files:
        try:
            _validate_extension(file.filename)
            content = await _read_upload_limited(file)
            result = await service.process_document(
                file_content=content,
                filename=file.filename,
                document_name=None,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                use_ocr=use_ocr,
                skip_index=True,  # 批量场景最后统一重建一次索引
            )
            payload = {
                key: result[key]
                for key in (
                    "document_id",
                    "document_name",
                    "file_type",
                    "total_chunks",
                    "total_characters",
                    "process_time",
                )
            }
            results.append(BatchUploadItem(filename=file.filename, success=True, **payload))
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("文件 %s 处理失败: %s", file.filename, exc)
            results.append(
                BatchUploadItem(filename=file.filename or "unknown", success=False, error=str(exc))
            )
            failed_count += 1

    await service.reindex()

    return success_response(
        data={
            "total": len(files),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results,
        },
        message=f"批量上传完成：成功 {success_count}，失败 {failed_count}",
    )


@router.get("/list", response_model=ApiResponse[DocumentListData], summary="文档列表")
async def get_document_list():
    documents = await get_document_service().get_document_list()
    return success_response(data={"total": len(documents), "documents": documents})


@router.delete(
    "/{document_id}",
    response_model=ApiResponse[DeleteResultData],
    dependencies=[Depends(require_admin_token)],
    summary="删除文档",
)
async def delete_document(document_id: str):
    result = await get_document_service().delete_document(document_id)
    if not result["deleted_chunks"]:
        raise NotFoundException(f"文档不存在或已被删除: {document_id}")
    return success_response(data=result, message="文档删除成功")
