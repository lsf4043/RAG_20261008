"""
LangGraph 状态机组装

拆成两张图：
- ``build_retrieval_graph()``：query_rewrite → retrieve → grade →（CRAG 循环）→ rerank
- ``build_graph()``：retrieval 子图 → generate

为什么要拆：流式问答需要“先完成检索，再对生成阶段做 token 级流式”。
旧实现为了流式把整条检索链在 chat_service 里手写了一遍，与状态机形成两份
会各自漂移的实现。现在两条入口共用同一张检索子图，逻辑只有一份。
"""
from __future__ import annotations

import threading

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    decide_after_grade,
    generate_node,
    grade_documents_node,
    increment_iteration_node,
    query_rewrite_node,
    rerank_node,
    retrieve_node,
)
from app.graph.state import RAGState
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_retrieval_graph():
    """检索阶段子图（含 CRAG 条件路由）"""
    workflow = StateGraph(RAGState)

    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("increment_iteration", increment_iteration_node)

    workflow.add_edge(START, "query_rewrite")
    workflow.add_edge("query_rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        decide_after_grade,
        {
            "rewrite_retrieve": "increment_iteration",
            "rerank": "rerank",
        },
    )
    workflow.add_edge("increment_iteration", "query_rewrite")
    workflow.add_edge("rerank", END)

    compiled = workflow.compile()
    logger.debug("检索子图编译完成（CRAG）")
    return compiled


def build_graph():
    """完整状态机：检索子图 + 生成"""
    retrieval = build_retrieval_graph()

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieval", retrieval)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieval")
    workflow.add_edge("retrieval", "generate")
    workflow.add_edge("generate", END)

    compiled = workflow.compile()
    logger.info("RAG 状态机编译完成（Agentic RAG + CRAG）")
    return compiled


_graph = None
_retrieval_graph = None
_graph_lock = threading.Lock()


def get_graph():
    """完整状态机单例"""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


def get_retrieval_graph():
    """检索子图单例（流式问答复用）"""
    global _retrieval_graph
    if _retrieval_graph is None:
        with _graph_lock:
            if _retrieval_graph is None:
                _retrieval_graph = build_retrieval_graph()
    return _retrieval_graph


def reset_graphs() -> None:
    """清空缓存（测试用）"""
    global _graph, _retrieval_graph
    with _graph_lock:
        _graph = None
        _retrieval_graph = None
