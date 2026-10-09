"""
问答相关数据模型
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    top_k: Optional[int] = Field(5, description="检索知识数量")
    temperature: Optional[float] = Field(0.7, description="模型温度")
    max_tokens: Optional[int] = Field(2000, description="最大生成token数")
    return_sources: Optional[bool] = Field(True, description="是否返回知识来源")


class ClearSessionRequest(BaseModel):
    """清空会话请求"""
    session_id: str = Field(..., description="会话ID")


class SourceInfo(BaseModel):
    """知识来源信息"""
    text: str
    score: float
    document_name: str
    page_num: int


class LogEntry(BaseModel):
    """日志条目"""
    time: str
    level: str
    step: str
    step_name: str = ""
    progress: int = 0
    message: str


class ChatQueryResponse(BaseModel):
    """问答响应"""
    answer: str
    question: str
    sources: List[SourceInfo] = []
    model: str
    process_time: float
    logs: List[LogEntry] = []
