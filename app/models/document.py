"""
文档相关数据模型
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.chat import LogEntry


class DocumentInfo(BaseModel):
    """文档信息"""

    document_id: str
    document_name: str
    file_type: str
    chunk_count: int = 0
    create_time: Optional[str] = None


class DocumentListData(BaseModel):
    total: int
    documents: List[DocumentInfo] = Field(default_factory=list)


class UploadResultData(BaseModel):
    """单个文档上传结果"""

    document_id: str
    document_name: str
    file_type: str
    total_chunks: int
    total_characters: int
    process_time: float
    logs: List[LogEntry] = Field(default_factory=list)


class BatchUploadItem(BaseModel):
    filename: str
    success: bool
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    file_type: Optional[str] = None
    total_chunks: Optional[int] = None
    total_characters: Optional[int] = None
    process_time: Optional[float] = None
    error: Optional[str] = None


class BatchUploadData(BaseModel):
    total: int
    success_count: int
    failed_count: int
    results: List[BatchUploadItem] = Field(default_factory=list)


class DeleteResultData(BaseModel):
    document_id: str
    deleted_chunks: int
    deleted_in_registry: int = 0


class HealthData(BaseModel):
    """健康检查结果"""

    status: str
    version: str
    uptime_seconds: int
    services: Dict[str, Any] = Field(default_factory=dict)


class StatsData(BaseModel):
    """运行统计"""

    embedding_cache: Dict[str, Any] = Field(default_factory=dict)
    query_cache: Dict[str, Any] = Field(default_factory=dict)
    bm25_documents: int = 0
    sessions: int = 0
    collection: Dict[str, Any] = Field(default_factory=dict)
