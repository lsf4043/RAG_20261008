"""
问答相关数据模型

请求模型带完整边界校验（旧版 top_k / temperature 没有任何约束，
一个 top_k=100000 就能把 Milvus 打满）。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    """问答请求"""

    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    session_id: Optional[str] = Field(None, max_length=128, description="会话 ID（多轮对话）")
    top_k: int = Field(5, ge=1, le=20, description="进入生成的文档数量")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="模型温度，留空使用服务端默认值"
    )
    max_tokens: Optional[int] = Field(
        None, ge=100, le=8000, description="最大生成 token 数，留空使用服务端默认值"
    )
    return_sources: bool = Field(True, description="是否返回引用来源")


class ClearSessionRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., min_length=1, max_length=128)


class SourceInfo(BaseModel):
    """知识来源"""

    text: str = Field(..., description="片段原文（截断）")
    score: float = Field(0.0, description="相关性分数（优先 rerank_score）")
    rrf_score: Optional[float] = Field(None, description="RRF 融合分")
    rerank_score: Optional[float] = Field(None, description="Cross-Encoder 分数")
    document_id: Optional[str] = None
    document_name: str = "未知文档"
    page_num: int = 0
    title: str = ""
    chunk_id: Optional[int] = None


class LogEntry(BaseModel):
    """处理过程日志条目"""

    time: str
    level: str
    step: str
    step_name: str = ""
    progress: int = 0
    message: str


class ChatData(BaseModel):
    """问答结果"""

    answer: str
    question: str
    sources: List[SourceInfo] = Field(default_factory=list)
    model: str
    process_time: float
    logs: List[LogEntry] = Field(default_factory=list)
    session_id: Optional[str] = None
    used_fallback: bool = Field(False, description="是否脱离知识库作答（True 表示答案无知识库依据）")
    route: str = Field("rag", description="意图路由结果：rag=知识库检索，direct=通用直答（未检索知识库）")
    degraded: bool = Field(False, description="是否发生降级（BM25 缺失 / 评分失败等）")
    notes: List[str] = Field(default_factory=list, description="降级与策略说明")
    crag_iterations: int = 0
    rewritten_query: Optional[str] = None
    cached: bool = False


class ClearSessionData(BaseModel):
    success: bool
    message: str
