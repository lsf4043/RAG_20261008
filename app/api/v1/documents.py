"""
文档上传接口
"""
import os
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException

from app.config import settings
from app.services.document_service import document_service
from app.models.common import success_response, error_response
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    document_name: Optional[str] = Form(None, description="文档名称"),
    chunk_size: Optional[int] = Form(500, description="分块大小"),
    overlap: Optional[int] = Form(50, description="重叠大小"),
    use_ocr: Optional[bool] = Form(False, description="强制使用OCR")
):
    """
    上传单个文档到知识库

    支持的文件格式: PDF, DOC, DOCX, TXT, MD, XLSX, PPTX, HTML, 图片等
    """
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower().replace('.', '')
        if file_ext not in settings.allowed_extensions_list:
            return error_response(
                f"不支持的文件类型: {file_ext}",
                code=415
            )

        # 检查文件大小
        content = await file.read()
        file_size_mb = len(content) / 1024 / 1024
        if file_size_mb > settings.MAX_FILE_SIZE:
            return error_response(
                f"文件大小超过限制: {file_size_mb:.2f}MB > {settings.MAX_FILE_SIZE}MB",
                code=400
            )

        # 处理文档
        result = await document_service.process_document(
            file_content=content,
            filename=file.filename,
            document_name=document_name,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            use_ocr=use_ocr
        )

        if result.get("success"):
            return success_response(
                data={
                    "document_id": result["document_id"],
                    "document_name": result["document_name"],
                    "file_type": result["file_type"],
                    "total_chunks": result["total_chunks"],
                    "total_characters": result["total_characters"],
                    "process_time": result["process_time"],
                    "logs": result["logs"]
                },
                message="文档上传成功"
            )
        else:
            return error_response(
                result.get("error", "处理失败"),
                code=500,
                data={"logs": result.get("logs", [])}
            )

    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        return error_response(f"上传失败: {str(e)}", code=500)


@router.post("/upload/batch")
async def upload_documents_batch(
    files: List[UploadFile] = File(..., description="上传的多个文件"),
    chunk_size: Optional[int] = Form(500, description="分块大小"),
    overlap: Optional[int] = Form(50, description="重叠大小"),
    use_ocr: Optional[bool] = Form(False, description="强制使用OCR")
):
    """
    批量上传多个文档到知识库

    支持的文件格式: PDF, DOC, DOCX, TXT, MD, XLSX, PPTX, HTML, 图片等
    """
    try:
        results = []
        success_count = 0
        failed_count = 0

        for file in files:
            try:
                # 检查文件扩展名
                file_ext = os.path.splitext(file.filename)[1].lower().replace('.', '')
                if file_ext not in settings.allowed_extensions_list:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": f"不支持的文件类型: {file_ext}"
                    })
                    failed_count += 1
                    continue

                # 检查文件大小
                content = await file.read()
                file_size_mb = len(content) / 1024 / 1024
                if file_size_mb > settings.MAX_FILE_SIZE:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": f"文件大小超过限制: {file_size_mb:.2f}MB > {settings.MAX_FILE_SIZE}MB"
                    })
                    failed_count += 1
                    continue

                # 处理文档
                result = await document_service.process_document(
                    file_content=content,
                    filename=file.filename,
                    document_name=None,  # 批量上传使用文件名作为文档名
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    use_ocr=use_ocr
                )

                if result.get("success"):
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "document_id": result["document_id"],
                        "document_name": result["document_name"],
                        "file_type": result["file_type"],
                        "total_chunks": result["total_chunks"],
                        "total_characters": result["total_characters"],
                        "process_time": result["process_time"]
                    })
                    success_count += 1
                else:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": result.get("error", "处理失败")
                    })
                    failed_count += 1

            except Exception as e:
                logger.error(f"文件 {file.filename} 处理失败: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
                failed_count += 1

        return success_response(
            data={
                "total": len(files),
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results
            },
            message=f"批量上传完成: 成功 {success_count} 个, 失败 {failed_count} 个"
        )

    except Exception as e:
        logger.error(f"批量上传失败: {str(e)}")
        return error_response(f"批量上传失败: {str(e)}", code=500)


@router.websocket("/upload/logs")
async def upload_document_logs(websocket: WebSocket):
    """
    文档上传实时日志 WebSocket
    """
    await websocket.accept()

    def log_callback(entry):
        """日志回调，发送到 WebSocket"""
        import asyncio
        asyncio.create_task(websocket.send_json(entry))

    try:
        # 等待接收文件信息
        data = await websocket.receive_json()

        # 这里简化处理，实际应用中需要更复杂的文件传输逻辑
        # 可以通过 base64 编码传输文件内容，或者使用单独的 HTTP 上传接口

        await websocket.send_json({
            "level": "info",
            "message": "WebSocket 连接已建立，请使用 HTTP POST 接口上传文件"
        })

    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {str(e)}")
        await websocket.close()


@router.get("/list")
async def get_document_list():
    """
    获取已上传文档列表
    """
    try:
        documents = await document_service.get_document_list()
        return success_response(data={
            "total": len(documents),
            "documents": documents
        })
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        return error_response(f"获取失败: {str(e)}", code=500)


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    删除指定文档
    """
    try:
        result = await document_service.delete_document(document_id)
        return success_response(
            data={"deleted_chunks": result["deleted_chunks"]},
            message="文档删除成功"
        )
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        return error_response(f"删除失败: {str(e)}", code=500)
