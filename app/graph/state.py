"""
LangGraph 状态定义（升级版核心）
定义 RAG 流程的状态结构，作为状态机各节点之间传递的数据载体。

状态字段说明：
- question：用户原始问题
- history：多轮对话历史
- rewritten_query：指代消解后的查询
- queries：用于检索的 query 列表（含 Multi-Query 变体、HyDE 假设答案）
- documents：检索到的候选文档（去重合并后）
- graded_documents：经过 CRAG 相关性评分后的文档
- is_relevant：是否检索到相关内容
- iteration：CRAG 重检索次数
- answer：最终生成的答案
- sources：答案引用来源
"""
from typing import TypedDict, List, Optional, Dict, Any

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """RAG 流程状态（total=False 表示所有字段可选）"""

    # 输入
    question: str                              # 原始问题
    history: List[Dict[str, str]]              # 对话历史 [{role, content}]

    # Query 处理
    rewritten_query: str                       # 指代消解后的查询
    queries: List[str]                         # 检索 query 列表

    # 检索
    documents: List[Document]                  # 检索候选文档（去重合并）
    graded_documents: List[Document]           # CRAG 评分后的文档
    is_relevant: bool                          # 是否存在相关文档

    # CRAG 控制
    iteration: int                             # 重检索迭代次数

    # 生成
    answer: str                                # 最终答案
    sources: List[Dict[str, Any]]              # 引用来源
    used_fallback: bool                        # 是否走了无上下文 fallback


def initial_state(question: str, history: Optional[List[Dict[str, str]]] = None) -> RAGState:
    """
    构造初始状态

    Args:
        question: 用户问题
        history: 对话历史

    Returns:
        初始 RAGState
    """
    return {
        "question": question,
        "history": history or [],
        "rewritten_query": question,
        "queries": [question],
        "documents": [],
        "graded_documents": [],
        "is_relevant": False,
        "iteration": 0,
        "answer": "",
        "sources": [],
        "used_fallback": False,
    }