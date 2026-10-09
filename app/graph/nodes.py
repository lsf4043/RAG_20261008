"""
LangGraph RAG 节点定义（升级版核心）
实现 Agentic RAG 的五个核心节点 + 条件路由：

1. query_rewrite_node    - Query 改写/扩展（P1 优化）
2. retrieve_node         - 混合检索（向量 + BM25 + RRF 融合，多 query 并行）
3. grade_documents_node  - CRAG 文档相关性评分（P1 优化）
4. rerank_node           - Cross-Encoder 重排序
5. generate_node         - 生成答案（带引用，Self-RAG 基础）

条件路由：
- decide_after_grade     - CRAG 决策：全部无关且未达迭代上限 → 重检索，否则 → 重排序
"""
from typing import List, Dict, Tuple, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.graph.state import RAGState
from app.core.llm import llm_service
from app.core.query_processor import query_processor
from app.core.vector_store import vector_store_service
from app.core.bm25_retriever import bm25_retriever_service
from app.core.reranker import reranker_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 辅助函数
# ============================================================

def _doc_key(doc: Document) -> str:
    """
    生成文档唯一标识（用于去重）

    Args:
        doc: 文档

    Returns:
        唯一标识字符串
    """
    metadata = doc.metadata or {}
    doc_id = metadata.get("document_id", "")
    chunk_id = str(metadata.get("chunk_id", ""))
    if doc_id and chunk_id:
        return f"{doc_id}_{chunk_id}"
    # 无标识时回退到内容哈希
    return str(hash(doc.page_content))


def _doc_text(doc: Document) -> str:
    """
    提取文档正文（优先 metadata 中的原始正文，否则用 page_content）

    Args:
        doc: 文档

    Returns:
        正文文本
    """
    metadata = doc.metadata or {}
    return metadata.get("content") or doc.page_content or ""


def _rrf_fusion(
    ranked_lists: List[List[Document]],
    top_k: int,
    k: int = 60
) -> List[Document]:
    """
    RRF（Reciprocal Rank Fusion）融合多路检索结果

    Args:
        ranked_lists: 多个已排序的文档列表（每路检索结果）
        top_k: 返回文档数量
        k: RRF 参数

    Returns:
        融合后的文档列表
    """
    score_map: Dict[str, Tuple[Document, float]] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = _doc_key(doc)
            rrf_score = 1.0 / (k + rank + 1)
            if key not in score_map:
                score_map[key] = (doc, rrf_score)
            else:
                # 同一文档出现在多路结果中，累加分数
                existing_doc, existing_score = score_map[key]
                score_map[key] = (existing_doc, existing_score + rrf_score)

    # 按融合分数降序
    sorted_items = sorted(score_map.values(), key=lambda x: x[1], reverse=True)

    # 返回 top_k 文档，并把 RRF 分数写入 metadata
    results = []
    for doc, score in sorted_items[:top_k]:
        doc.metadata["rrf_score"] = score
        results.append(doc)

    return results


# ============================================================
# 节点 1：Query 改写
# ============================================================

def query_rewrite_node(state: RAGState) -> RAGState:
    """
    Query 改写节点：结合历史做指代消解，并做 Multi-Query 扩展 / HyDE

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    question = state["question"]
    history = state.get("history", [])

    logger.info(f"[query_rewrite] 开始处理问题: {question[:50]}")

    # 生成检索 query 列表（含改写 + 扩展 + HyDE）
    queries = query_processor.build_retrieval_queries(question, history)

    state["rewritten_query"] = queries[0]  # 第一个是改写后的主 query
    state["queries"] = queries

    logger.info(f"[query_rewrite] 生成 {len(queries)} 个检索 query")
    return state


# ============================================================
# 节点 2：混合检索
# ============================================================

def retrieve_node(state: RAGState) -> RAGState:
    """
    混合检索节点：对每个 query 做向量 + BM25 双路召回，RRF 融合去重

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    queries = state.get("queries", [state["question"]])
    candidate_k = settings.RETRIEVAL_CANDIDATE_K

    logger.info(f"[retrieve] 开始混合检索，共 {len(queries)} 个 query")

    all_ranked_lists: List[List[Document]] = []

    for query in queries:
        try:
            # 1. 向量检索
            vector_results = vector_store_service.similarity_search_with_score(
                query, k=candidate_k
            )
            vector_docs = [doc for doc, _ in vector_results]
            all_ranked_lists.append(vector_docs)
            logger.info(f"[retrieve] 向量检索 '{query[:30]}' 返回 {len(vector_docs)} 条")

            # 2. BM25 检索
            if bm25_retriever_service.count > 0:
                bm25_docs = bm25_retriever_service.search(query, top_k=candidate_k)
                all_ranked_lists.append(bm25_docs)
                logger.info(f"[retrieve] BM25 检索 '{query[:30]}' 返回 {len(bm25_docs)} 条")
        except Exception as e:
            logger.error(f"[retrieve] 检索异常: {str(e)}")

    # 3. RRF 融合去重
    fused = _rrf_fusion(all_ranked_lists, top_k=candidate_k)
    state["documents"] = fused

    logger.info(f"[retrieve] RRF 融合后得到 {len(fused)} 条候选文档")
    return state


# ============================================================
# 节点 3：CRAG 文档评分
# ============================================================

def _grade_single_document(question: str, doc_text: str) -> bool:
    """
    用 LLM 判断单篇文档与问题的相关性

    Args:
        question: 问题
        doc_text: 文档正文

    Returns:
        是否相关
    """
    prompt = f"""你是一个文档相关性评估专家。请判断以下文档内容是否与用户问题相关，能否用于回答该问题。

只回答 "yes" 或 "no"，不要输出任何其他内容。

用户问题：{question}

文档内容：
{doc_text[:1000]}

相关（yes/no）："""

    try:
        answer = llm_service.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        logger.warning(f"文档评分失败，默认为相关: {str(e)}")
        return True


def grade_documents_node(state: RAGState) -> RAGState:
    """
    CRAG 文档评分节点：用 LLM 评估每篇候选文档与问题的相关性

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    documents = state.get("documents", [])
    question = state["rewritten_query"] or state["question"]

    # 未启用 CRAG 时跳过评分，全部视为相关
    if not settings.CRAG_ENABLED:
        state["graded_documents"] = documents
        state["is_relevant"] = len(documents) > 0
        return state

    if not documents:
        state["graded_documents"] = []
        state["is_relevant"] = False
        return state

    logger.info(f"[grade] CRAG 开始评分 {len(documents)} 篇文档")

    graded = []
    for doc in documents:
        text = _doc_text(doc)
        is_relevant = _grade_single_document(question, text)
        if is_relevant:
            doc.metadata["crag_relevant"] = True
            graded.append(doc)
        else:
            doc.metadata["crag_relevant"] = False

    state["graded_documents"] = graded
    state["is_relevant"] = len(graded) > 0

    logger.info(f"[grade] CRAG 评分完成：{len(graded)}/{len(documents)} 篇相关")
    return state


# ============================================================
# 条件路由：CRAG 决策
# ============================================================

def decide_after_grade(state: RAGState) -> str:
    """
    CRAG 决策函数：根据评分结果决定下一步

    - 全部无关 且 未达迭代上限 → "rewrite_retrieve"（重检索）
    - 否则 → "rerank"（重排序）

    Args:
        state: 当前状态

    Returns:
        目标节点名称
    """
    if not state["is_relevant"] and state.get("iteration", 0) < settings.CRAG_MAX_ITERATIONS:
        logger.info("[route] 检索结果全部无关，触发重检索")
        return "rewrite_retrieve"
    logger.info("[route] 检索结果相关，进入重排序")
    return "rerank"


# ============================================================
# 节点 4：重排序
# ============================================================

def rerank_node(state: RAGState) -> RAGState:
    """
    重排序节点：用 Cross-Encoder 对相关文档重排序

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    documents = state.get("graded_documents", [])
    question = state["rewritten_query"] or state["question"]

    if not documents:
        state["documents"] = []
        return state

    # 未启用重排序时直接保留原顺序
    if not settings.RERANK_ENABLED:
        state["documents"] = documents[:settings.RETRIEVAL_TOP_K]
        return state

    logger.info(f"[rerank] 开始重排序 {len(documents)} 篇文档")

    reranked = reranker_service.rerank(
        documents, question, top_n=settings.RETRIEVAL_TOP_K
    )

    state["documents"] = reranked
    logger.info(f"[rerank] 重排序完成，保留 {len(reranked)} 篇")
    return state


# ============================================================
# 节点 5：生成
# ============================================================

def generate_node(state: RAGState) -> RAGState:
    """
    生成节点：基于检索内容生成答案，并附带引用来源

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    question = state["question"]
    history = state.get("history", [])
    documents = state.get("documents", [])

    # 组装上下文
    context_parts = []
    for i, doc in enumerate(documents):
        text = _doc_text(doc)
        title = doc.metadata.get("title", "") if doc.metadata else ""
        doc_name = doc.metadata.get("document_name", "未知文档") if doc.metadata else "未知文档"
        prefix = f"【知识片段 {i + 1}】（来源: {doc_name}"
        if title:
            prefix += f"，标题: {title}"
        prefix += f"）"
        context_parts.append(f"{prefix}\n{text}")

    context_text = "\n\n".join(context_parts)

    # 构建系统提示
    if context_text:
        system_prompt = f"""你是一个智能问答助手。请根据以下知识库内容回答用户问题。

## 知识库内容：
{context_text}

## 回答要求：
1. 严格基于上述知识库内容回答用户问题
2. 回答要简洁、准确、有条理
3. 如果知识库内容与问题相关，综合运用相关知识回答
4. 如果知识库内容不足以回答，请明确说明"根据现有知识库无法准确回答"
5. 如果用户问题涉及之前的对话内容，请结合上下文回答"""
        state["used_fallback"] = False
    else:
        system_prompt = """你是一个智能问答助手。知识库中没有检索到与问题相关的内容，请基于你的通用知识回答。

## 回答要求：
1. 回答要简洁、准确、有条理
2. 如果涉及专业领域且不确定，请说明不确定性
3. 如果用户问题涉及之前的对话内容，请结合上下文回答"""
        state["used_fallback"] = True

    # 组装消息
    messages: List[Any] = [SystemMessage(content=system_prompt)]
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=question))

    logger.info(f"[generate] 开始生成答案（{'带回知识库' if not state['used_fallback'] else '无上下文 fallback'}）")

    try:
        answer = llm_service.invoke_messages(messages)
    except Exception as e:
        logger.error(f"[generate] 生成失败: {str(e)}")
        answer = f"抱歉，生成回答时出现错误: {str(e)}"

    state["answer"] = answer

    # 格式化引用来源
    sources = []
    for doc in documents:
        metadata = doc.metadata or {}
        sources.append({
            "text": _doc_text(doc),
            "score": metadata.get("rrf_score", metadata.get("score", 0)),
            "document_name": metadata.get("document_name", "未知文档"),
            "page_num": metadata.get("page_num", 0),
            "title": metadata.get("title", ""),
        })
    state["sources"] = sources

    logger.info(f"[generate] 生成完成，答案长度 {len(answer)} 字符，来源 {len(sources)} 条")
    return state


# ============================================================
# 重检索预处理：迭代计数递增（供条件边跳转回 query_rewrite 前使用）
# ============================================================

def increment_iteration_node(state: RAGState) -> RAGState:
    """
    递增 CRAG 迭代计数（在重检索前调用）

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    state["iteration"] = state.get("iteration", 0) + 1
    logger.info(f"[crag] 开始第 {state['iteration']} 次重检索")
    return state