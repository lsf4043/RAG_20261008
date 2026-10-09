"""
文档相关数据模型
"""
from typing import Optional, List, Any
from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    document_name: Optional[str] = Field(None, description="文档名称")
    chunk_size: Optional[int] = Field(300, description="分块大小")
    overlap: Optional[int] = Field(50, description="重叠大小")
    use_ocr: Optional[bool] = Field(False, description="强制使用OCR")


class LogEntry(BaseModel):
    """日志条目"""
    time: str
    level: str
    step: str
    step_name: str = ""
    progress: int = 0
    message: str


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    file_type: Optional[str] = None
    parse_method: Optional[str] = None
    total_chunks: Optional[int] = None
    total_characters: Optional[int] = None
    process_time: Optional[float] = None
    logs: List[LogEntry] = []


class DocumentInfo(BaseModel):
    """文档信息"""
    document_id: str
    document_name: str
    file_type: str
    chunk_count: int = 0
    create_time: Optional[str] = None


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    documents: List[DocumentInfo]


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""
    deleted_chunks: int
