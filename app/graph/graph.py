"""
LangGraph 状态机组装（升级版核心）
将 RAG 各节点组装为 Agentic RAG 状态机，支持 CRAG 条件路由与多步检索。

流程：
    START
      ↓
  query_rewrite (Query 改写/扩展)
      ↓
  retrieve (混合检索)
      ↓
  grade (CRAG 文档评分)
      ↓  ←── 条件路由 ──┐
      │                  │ (rewrite_retrieve: 全部无关且未达上限)
  rerank (重排序)         │
      ↓                  │
  generate (生成)   increment_iteration (迭代+1) → 回到 query_rewrite
      ↓
    END
"""
from langgraph.graph import StateGraph, START, END

from app.graph.state import RAGState
from app.graph.nodes import (
    query_rewrite_node,
    retrieve_node,
    grade_documents_node,
    decide_after_grade,
    rerank_node,
    generate_node,
    increment_iteration_node,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """
    构建并编译 RAG 状态机

    Returns:
        编译后的 StateGraph（可直接 invoke/ainvoke/stream）
    """
    # 实例化状态图
    workflow = StateGraph(RAGState)

    # 注册节点
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("increment_iteration", increment_iteration_node)

    # 入口
    workflow.add_edge(START, "query_rewrite")

    # 直线流程
    workflow.add_edge("query_rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # CRAG 条件路由
    workflow.add_conditional_edges(
        "grade",
        decide_after_grade,
        {
            "rewrite_retrieve": "increment_iteration",  # 重检索路径
            "rerank": "rerank",                         # 正常路径
        },
    )

    # 重检索路径：迭代计数后回到 query 改写
    workflow.add_edge("increment_iteration", "query_rewrite")

    # 正常路径：重排序 → 生成 → 结束
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    # 编译
    compiled = workflow.compile()
    logger.info("RAG 状态机编译完成（Agentic RAG + CRAG）")
    return compiled


# 全局编译后的状态机实例（懒加载，在应用启动后构建）
_graph = None


def get_graph():
    """获取编译后的状态机单例"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph