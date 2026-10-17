"""
LangGraph 状态定义

字段说明：
- 输入：question / history
- Query 处理：rewritten_query / queries
- 检索：documents（RRF 融合后）/ graded_documents（CRAG 通过）/ is_relevant
- 控制：iteration（CRAG 轮次）/ candidate_k（每轮可动态放宽）
- 生成：answer / sources / used_fallback（是否脱离知识库作答）
- 诊断：degraded / notes（降级原因，直接暴露给前端，避免“静默降级”）
- 请求级参数覆盖：top_k / temperature / max_tokens（让 API 参数真正生效）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """RAG 流程状态（total=False：所有字段可选）"""

    # ---------- 输入 ----------
    question: str
    history: List[Dict[str, str]]

    # ---------- 请求级参数 ----------
    top_k: int
    temperature: Optional[float]
    max_tokens: Optional[int]

    # ---------- Query 处理 ----------
    rewritten_query: str
    queries: List[str]

    # ---------- 检索 ----------
    documents: List[Document]
    graded_documents: List[Document]
    is_relevant: bool
    candidate_k: int

    # ---------- CRAG 控制 ----------
    iteration: int

    # ---------- 生成 ----------
    answer: str
    sources: List[Dict[str, Any]]
    used_fallback: bool

    # ---------- 诊断 ----------
    degraded: bool
    notes: List[str]


def initial_state(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> RAGState:
    """构造初始状态（请求级参数优先于全局配置）"""
    from app.config import settings

    return {
        "question": question,
        "history": history or [],
        "top_k": int(top_k) if top_k else settings.RETRIEVAL_TOP_K,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "rewritten_query": question,
        "queries": [question],
        "documents": [],
        "graded_documents": [],
        "is_relevant": False,
        "candidate_k": settings.RETRIEVAL_CANDIDATE_K,
        "iteration": 0,
        "answer": "",
        "sources": [],
        "used_fallback": False,
        "degraded": False,
        "notes": [],
    }
