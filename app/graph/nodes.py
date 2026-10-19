"""
LangGraph RAG 节点

本次修复的核心问题：
1. **CRAG 重试不再空转**：重检索时会把上一轮的 query 作为“失败提示”回灌给改写器，
   同时放宽候选数（candidate_k），因此第二轮真的会换一批文档；
2. **CRAG 评分批量化**：从“每篇文档一次 LLM 调用”改为“一次调用判 N 篇”，
   单问题 LLM 调用数从 20~60 次降到 2~3 次；
3. **节点不修改入参**：统一返回局部状态更新，并只写入 Document 副本，
   避免 BM25 索引里的对象被污染、并发请求互相串分数；
4. **无依据时不再假装有依据**：检索不到内容时默认拒答，``used_fallback`` 显式暴露；
5. **配置真正生效**：RRF 按 VECTOR_WEIGHT / BM25_WEIGHT 加权，top_k / temperature /
   max_tokens 从请求状态读取。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.core.documents import copy_document, doc_key, doc_name, doc_text, doc_title
from app.core.exceptions import LLMException
from app.core.intent_router import ROUTE_DIRECT, ROUTE_RAG, classify_intent
from app.core.llm import get_llm_service
from app.core.query_processor import QueryProcessor, get_query_processor
from app.core.vector_store import get_vector_store_service
from app.core.bm25_retriever import get_bm25_service
from app.core.reranker import get_reranker_service
from app.graph.state import RAGState
from app.utils.logger import get_logger

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "抱歉，知识库中没有检索到与该问题相关的内容。\n\n"
    "建议：\n"
    "1. 换一种问法或补充更具体的信息；\n"
    "2. 确认相关文档已经上传到知识库；\n"
    "3. 如果只想基于通用知识闲聊，可以让管理员开启 ALLOW_LLM_FALLBACK。"
)

MAX_SOURCE_TEXT = 800
MAX_GRADE_EXCERPT = 600


# ============================================================
# 辅助：RRF 融合（支持通道权重）
# ============================================================
def _rrf_fusion(
    ranked_lists: Sequence[Tuple[str, List[Document]]],
    top_k: int,
    weights: Dict[str, float],
    k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion

    Args:
        ranked_lists: [(通道名, 已排序文档列表)]
        top_k: 返回数量
        weights: 通道权重（VECTOR_WEIGHT / BM25_WEIGHT）
        k: RRF 平滑常数
    """
    score_map: Dict[str, List[Any]] = {}

    for channel, docs in ranked_lists:
        weight = float(weights.get(channel, 1.0))
        if weight <= 0:
            continue
        for rank, doc in enumerate(docs):
            key = doc_key(doc)
            contribution = weight / (k + rank + 1)
            entry = score_map.get(key)
            if entry is None:
                copied = copy_document(doc)
                copied.metadata["retrieval_channels"] = [channel]
                score_map[key] = [copied, contribution]
            else:
                entry[1] += contribution
                channels = entry[0].metadata.setdefault("retrieval_channels", [])
                if channel not in channels:
                    channels.append(channel)

    ordered = sorted(score_map.values(), key=lambda item: item[1], reverse=True)

    results: List[Document] = []
    for doc, score in ordered[:top_k]:
        doc.metadata["rrf_score"] = round(float(score), 6)
        results.append(doc)
    return results


# ============================================================
# 节点 0：意图路由（闲聊直答 / 知识库检索）
# ============================================================
def route_intent_node(state: RAGState) -> Dict[str, Any]:
    """
    判断本次请求是否需要检索知识库

    - direct：闲聊、寒暄、通用写作/翻译 → 不检索，直接让模型回答
    - rag   ：知识类问题 → 走完整 RAG 链路

    规则判定为 0 成本；只有规则无法判定的模糊短句才会多花一次极小 LLM 调用。
    """
    question = state["question"]
    history = state.get("history") or []
    notes = list(state.get("notes") or [])

    decision = classify_intent(question, history)
    label = "直接作答" if decision.is_direct else "知识库检索"
    logger.info(
        "[route_intent] %s（来源=%s，理由=%s）", label, decision.source, decision.reason
    )
    notes.append(f"意图路由：{label}（{decision.reason}）")

    return {"route": decision.route, "notes": notes}


def decide_after_route(state: RAGState) -> str:
    """条件边：direct → 直答节点；其他 → 检索子图"""
    route = state.get("route") or ROUTE_RAG
    return ROUTE_DIRECT if route == ROUTE_DIRECT else "rag"


# ============================================================
# 节点 1：Query 改写 / 扩展
# ============================================================
def query_rewrite_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    history = state.get("history") or []
    iteration = int(state.get("iteration", 0) or 0)
    notes = list(state.get("notes") or [])

    retry_hint = ""
    if iteration > 0:
        retry_hint = QueryProcessor.build_retry_hint(list(state.get("queries") or []))
        notes.append(f"CRAG 第 {iteration} 轮重检索：已回灌失败查询并放宽候选数")

    processor = get_query_processor()
    try:
        queries = processor.build_retrieval_queries(question, history, retry_hint=retry_hint or None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[query_rewrite] 规划失败，使用原问题: %s", exc)
        queries = [question]
        notes.append(f"Query 规划失败，降级为原问题：{exc}")

    # 重检索时逐轮放宽候选数（默认 20 → 30 → 40），避免反复捞同一批文档
    base_k = settings.RETRIEVAL_CANDIDATE_K
    candidate_k = base_k if iteration == 0 else min(base_k + base_k * iteration // 2, 200)

    logger.info(
        "[query_rewrite] %s 个检索 query（iteration=%s, candidate_k=%s）",
        len(queries),
        iteration,
        candidate_k,
    )
    return {
        "rewritten_query": queries[0] if queries else question,
        "queries": queries or [question],
        "candidate_k": candidate_k,
        "notes": notes,
    }


# ============================================================
# 节点 2：混合检索
# ============================================================
def retrieve_node(state: RAGState) -> Dict[str, Any]:
    queries = list(state.get("queries") or [state["question"]])
    candidate_k = int(state.get("candidate_k") or settings.RETRIEVAL_CANDIDATE_K)
    notes = list(state.get("notes") or [])
    degraded = bool(state.get("degraded"))

    vector_service = get_vector_store_service()
    bm25_service = get_bm25_service()
    bm25_ready = bm25_service.is_ready()

    ranked: List[Tuple[str, List[Document]]] = []
    vector_errors = 0

    for query in queries:
        try:
            vector_hits = [doc for doc, _ in vector_service.similarity_search_with_score(query, k=candidate_k)]
            if vector_hits:
                ranked.append(("vector", vector_hits))
            logger.info("[retrieve] 向量召回 %s 条（query=%s）", len(vector_hits), query[:30])
        except Exception as exc:  # noqa: BLE001
            vector_errors += 1
            logger.error("[retrieve] 向量检索异常: %s", exc)

        if bm25_ready:
            try:
                bm25_hits = bm25_service.search(query, top_k=candidate_k)
                if bm25_hits:
                    ranked.append(("bm25", bm25_hits))
                logger.info("[retrieve] BM25 召回 %s 条（query=%s）", len(bm25_hits), query[:30])
            except Exception as exc:  # noqa: BLE001
                logger.error("[retrieve] BM25 检索异常: %s", exc)

    if not bm25_ready:
        degraded = True
        notes.append("BM25 索引为空：当前仅使用向量检索（知识库为空或索引尚未构建）")

    if vector_errors and vector_errors == len(queries):
        degraded = True
        notes.append("向量检索全部失败：请检查 Milvus 连接")

    fused = _rrf_fusion(
        ranked,
        top_k=candidate_k,
        weights={"vector": settings.VECTOR_WEIGHT, "bm25": settings.BM25_WEIGHT},
        k=settings.RRF_K,
    )
    logger.info("[retrieve] RRF 融合后 %s 条候选（通道数=%s）", len(fused), len(ranked))

    return {
        "documents": fused,
        "degraded": degraded,
        "notes": notes,
    }


# ============================================================
# 节点 3：CRAG 批量化相关性评分
# ============================================================
class _DocGrade(BaseModel):
    index: int = Field(description="文档编号（从 1 开始）")
    relevant: bool = Field(description="该文档能否用于回答用户问题")


class _GradeBatch(BaseModel):
    grades: List[_DocGrade] = Field(description="每篇文档的相关性判断")


_GRADE_PROMPT = """你是检索结果相关性评估专家。判断下面的文档片段能否用于回答用户问题。

判定标准：
- 只要文档中包含了回答该问题所需的信息（哪怕只是部分信息），就算相关；
- 仅仅出现相同的词但语义无关，算不相关；
- 不要因为文档不完整而判为不相关。

用户问题：{question}

文档片段：
{docs}

请对每一篇文档给出 relevant=true/false。"""


def _format_grading_docs(docs: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        text = doc_text(doc)[:MAX_GRADE_EXCERPT].replace("\n", " ")
        blocks.append(f"[{i}] {text}")
    return "\n".join(blocks)


def _grade_batch(question: str, docs: List[Document]) -> Optional[List[bool]]:
    """
    一次性判断多篇文档的相关性

    Returns:
        与 docs 等长的布尔列表；LLM 不可用时返回 None（调用方按“全相关”处理，宁可多留不可错杀）
    """
    if not docs:
        return []

    prompt = _GRADE_PROMPT.format(question=question, docs=_format_grading_docs(docs))
    try:
        grader = get_llm_service().structured(_GradeBatch, temperature=0.0, max_tokens=800)
        result = grader.invoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[grade] 批量评分失败，按全部相关处理: %s", exc)
        return None

    if isinstance(result, dict):
        try:
            result = _GradeBatch(**result)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(result, _GradeBatch):
        return None

    flags = [True] * len(docs)  # 未返回的文档默认保留（fail-open）
    for item in result.grades:
        index = item.index - 1
        if 0 <= index < len(docs):
            flags[index] = bool(item.relevant)
    return flags


def grade_documents_node(state: RAGState) -> Dict[str, Any]:
    """
    CRAG 相关性评分节点：一次 LLM 调用批量判断候选文档能否回答用户问题

    行为要点：
    - 只评分前 CRAG_MAX_GRADE_DOCS 篇（后面的文档本来也会被 rerank 截断）；
    - 评分失败时 fail-open（全部保留）并标记 degraded，宁可多留不可错杀；
    - 只在 Document 副本上写 crag_relevant 标记，不污染检索阶段的原始对象；
    - 相关文档数 >= CRAG_MIN_RELEVANT 时 is_relevant 为 True，
      供 decide_after_grade 决定是否需要重检索。

    Returns:
        状态更新：graded_documents / is_relevant / degraded / notes
    """
    documents = list(state.get("documents") or [])
    question = state.get("rewritten_query") or state["question"]
    notes = list(state.get("notes") or [])
    degraded = bool(state.get("degraded"))

    if not documents:
        return {"graded_documents": [], "is_relevant": False, "degraded": True,
                "notes": notes + ["未检索到任何候选文档"]}

    if not settings.CRAG_ENABLED:
        graded = [copy_document(doc) for doc in documents]
        return {"graded_documents": graded, "is_relevant": bool(graded)}

    # 只对前 N 篇做 LLM 评分（后面的文档本来也会被 rerank 截断）
    to_grade = documents[: settings.CRAG_MAX_GRADE_DOCS]
    logger.info("[grade] CRAG 批量评分 %s 篇文档（一次 LLM 调用）", len(to_grade))

    flags = _grade_batch(question, to_grade)
    if flags is None:
        degraded = True
        notes.append("CRAG 评分失败：已跳过过滤（保留全部候选）")
        flags = [True] * len(to_grade)

    graded: List[Document] = []
    for doc, relevant in zip(to_grade, flags):
        copied = copy_document(doc)
        copied.metadata["crag_relevant"] = bool(relevant)
        if relevant:
            graded.append(copied)

    is_relevant = len(graded) >= settings.CRAG_MIN_RELEVANT
    logger.info("[grade] CRAG 评分完成：%s/%s 篇相关（is_relevant=%s）", len(graded), len(to_grade), is_relevant)

    return {
        "graded_documents": graded,
        "is_relevant": is_relevant,
        "degraded": degraded,
        "notes": notes,
    }


# ============================================================
# 条件路由：CRAG 决策
# ============================================================
def decide_after_grade(state: RAGState) -> str:
    """全部无关且未达迭代上限 → 重检索；否则 → 重排"""
    iteration = int(state.get("iteration", 0))

    # CRAG 关闭时不重试
    if not settings.CRAG_ENABLED:
        return "rerank"

    # 本轮一条候选都没召回：知识库为空 / 向量库不可用，
    # 换个问法也不可能变出内容来，重试只是白等两轮（空库时能省 8s+）
    if not state.get("documents"):
        logger.info("[route] 本轮没有召回任何候选文档，跳过重检索")
        return "rerank"

    # 没有任何改写能力时，重检索只会拿到一模一样的 query，属于纯空转
    can_rephrase = (
        settings.QUERY_REWRITE_ENABLED or settings.MULTI_QUERY_ENABLED or settings.HYDE_ENABLED
    )
    if not can_rephrase:
        if not state.get("is_relevant"):
            logger.info("[route] 无改写能力，跳过重检索")
        return "rerank"

    if not state.get("is_relevant") and iteration < settings.CRAG_MAX_ITERATIONS:
        logger.info("[route] 检索结果全部无关，触发第 %s 次重检索", iteration + 1)
        return "rewrite_retrieve"
    logger.info("[route] 进入重排序")
    return "rerank"


# ============================================================
# 节点 4：重排序
# ============================================================
def rerank_node(state: RAGState) -> Dict[str, Any]:
    """
    重排序节点：对 CRAG 过滤后的文档按与问题的相关性重新排序并截断

    行为要点：
    - 输入取 graded_documents（CRAG 评分输出），top_k 优先读请求状态，
      缺省回退到 RETRIEVAL_TOP_K；
    - RERANK_ENABLED 关闭时不排序，直接截断前 top_k 篇；
    - RERANK_MIN_SCORE > 0 时按 rerank_score 过滤弱相关文档，过滤数量记入 notes；
    - 输出 documents 供生成节点构造上下文使用。

    Returns:
        状态更新：documents（最终入选文档）/ notes
    """
    documents = list(state.get("graded_documents") or [])
    question = state.get("rewritten_query") or state["question"]
    top_k = int(state.get("top_k") or settings.RETRIEVAL_TOP_K)
    notes = list(state.get("notes") or [])

    if not documents:
        return {"documents": []}

    if not settings.RERANK_ENABLED:
        return {"documents": [copy_document(doc) for doc in documents[:top_k]]}

    logger.info("[rerank] 重排序 %s 篇文档", len(documents))
    reranked = get_reranker_service().rerank(documents, question, top_n=top_k)

    if settings.RERANK_MIN_SCORE > 0:
        kept = [
            doc
            for doc in reranked
            if float(doc.metadata.get("rerank_score", 0.0)) >= settings.RERANK_MIN_SCORE
        ]
        if len(kept) != len(reranked):
            notes.append(
                f"按 RERANK_MIN_SCORE={settings.RERANK_MIN_SCORE} 过滤掉 {len(reranked) - len(kept)} 篇弱相关文档"
            )
        reranked = kept

    logger.info("[rerank] 重排序完成，保留 %s 篇", len(reranked))
    return {"documents": reranked, "notes": notes}


# ============================================================
# 节点 5：生成
# ============================================================
def _build_context(documents: List[Document]) -> str:
    parts: List[str] = []
    used = 0
    for i, doc in enumerate(documents, start=1):
        text = doc_text(doc)
        title = doc_title(doc)
        name = doc_name(doc)
        header = f"【知识片段 {i}】（来源: {name}"
        if title:
            header += f"，标题: {title}"
        page = doc.metadata.get("page_num")
        if page:
            header += f"，页码: {page}"
        header += "）"
        block = f"{header}\n{text}"

        if used + len(block) > settings.CONTEXT_MAX_CHARS and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _build_messages(question: str, history: List[Dict[str, str]], context: str) -> List[BaseMessage]:
    if context:
        system_prompt = f"""你是一个严谨的知识库问答助手。请严格依据下面的知识库内容回答用户问题。

## 知识库内容
{context}

## 回答要求
1. 只使用上述知识库内容，不要引入外部知识或臆造事实
2. 引用信息时用【知识片段 n】标注来源
3. 如果知识库内容不足以回答，直接说明"根据现有知识库无法准确回答"，不要猜测
4. 回答简洁、条理清晰，涉及多轮对话时结合上下文
5. 使用与用户问题相同的语言作答"""
    else:
        system_prompt = """你是一个知识库问答助手。当前知识库中没有检索到与问题相关的内容。

## 回答要求
1. 明确告知用户"知识库中没有找到相关内容"
2. 不要编造事实
3. 可以提示用户换一种问法或补充文档"""

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    for msg in history or []:
        role = (msg or {}).get("role", "user")
        content = str((msg or {}).get("content", ""))
        messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
    messages.append(HumanMessage(content=question))
    return messages


def build_generation_messages(
    question: str,
    history: List[Dict[str, str]],
    documents: List[Document],
) -> Tuple[List[BaseMessage], bool]:
    """
    构造生成阶段的 messages（供 generate_node 与流式问答复用，保证两条路径提示词一致）

    Returns:
        (messages, has_context)
    """
    context = _build_context(documents) if documents else ""
    return _build_messages(question, history, context), bool(context)


def generate_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    history = list(state.get("history") or [])
    documents = list(state.get("documents") or [])
    notes = list(state.get("notes") or [])
    degraded = bool(state.get("degraded"))

    # 无依据：默认拒答，避免“模型自由发挥被当成知识库答案”
    if not documents and not settings.ALLOW_LLM_FALLBACK:
        logger.info("[generate] 无检索结果且未开启 fallback，直接拒答")
        return {
            "answer": NO_CONTEXT_ANSWER,
            "sources": [],
            "used_fallback": True,
            "degraded": True,
            "notes": notes + ["知识库无相关依据，已按策略拒答"],
        }

    messages, has_context = build_generation_messages(question, history, documents)

    logger.info("[generate] 生成答案（依据 %s 篇文档，fallback=%s）", len(documents), not has_context)
    try:
        answer = get_llm_service().invoke_messages(
            messages,
            temperature=state.get("temperature"),
            max_tokens=state.get("max_tokens"),
        )
    except Exception as exc:  # noqa: BLE001
        # 详细错误只写日志：上游 4xx/5xx 的原文可能包含账号、配额等内部信息
        logger.error("[generate] 生成失败: %s", exc, exc_info=True)
        raise LLMException("生成回答失败：模型调用异常，请稍后重试（详情见服务端日志）") from exc

    sources = []
    for doc in documents:
        metadata = doc.metadata or {}
        sources.append(
            {
                "text": doc_text(doc)[:MAX_SOURCE_TEXT],
                "score": metadata.get("rerank_score", metadata.get("rrf_score", 0.0)),
                "rrf_score": metadata.get("rrf_score", 0.0),
                "rerank_score": metadata.get("rerank_score"),
                "document_id": metadata.get("document_id"),
                "document_name": metadata.get("document_name", "未知文档"),
                "page_num": metadata.get("page_num", 0),
                "title": metadata.get("title", ""),
                "chunk_id": metadata.get("chunk_id"),
            }
        )

    logger.info("[generate] 完成，答案 %s 字符，来源 %s 条", len(answer), len(sources))
    return {
        "answer": answer,
        "sources": sources,
        "used_fallback": not has_context,
        "degraded": degraded,
        "notes": notes,
    }


# ============================================================
# 节点 5：通用直答（不检索知识库）
# ============================================================
DIRECT_SYSTEM_PROMPT = """你是一个友好、可靠的通用助手。
本轮对话**没有**检索知识库，你的回答来自自身的通用知识与推理能力。

回答要求：
1. 寒暄/闲聊：简短自然，不要长篇大论，也不要刻意卖弄；
2. 写作、翻译、改写、计算、解释概念等任务：直接完成，必要时给出步骤；
3. 如果问题涉及具体产品文档、内部资料、专有数据（即需要查知识库才能准确回答），
   请明确说明“这需要查询知识库才能准确回答”，不要编造事实；
4. 不确定时就说不确定，使用与用户相同的语言作答。"""


def build_direct_messages(
    question: str,
    history: List[Dict[str, str]],
) -> List[BaseMessage]:
    """构造通用直答的 messages（与 RAG 生成共用 history 处理方式）"""
    messages: List[BaseMessage] = [SystemMessage(content=DIRECT_SYSTEM_PROMPT)]
    for msg in history or []:
        role = (msg or {}).get("role", "user")
        content = str((msg or {}).get("content", ""))
        if not content:
            continue
        messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
    messages.append(HumanMessage(content=question))
    return messages


def generate_direct_node(state: RAGState) -> Dict[str, Any]:
    """闲聊 / 通用任务：不检索知识库，直接生成"""
    question = state["question"]
    history = list(state.get("history") or [])
    notes = list(state.get("notes") or [])

    messages = build_direct_messages(question, history)
    logger.info("[generate_direct] 通用直答（未检索知识库）")

    try:
        answer = get_llm_service().invoke_messages(
            messages,
            temperature=state.get("temperature"),
            max_tokens=state.get("max_tokens"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[generate_direct] 生成失败: %s", exc, exc_info=True)
        raise LLMException("生成回答失败：模型调用异常，请稍后重试（详情见服务端日志）") from exc

    return {
        "answer": answer,
        "sources": [],
        # used_fallback=True 表示“本次回答没有知识库依据”，前端据此提示
        "used_fallback": True,
        "degraded": False,
        "route": ROUTE_DIRECT,
        "notes": notes + ["通用对话直接作答（未检索知识库）"],
    }


# ============================================================
# CRAG 迭代计数
# ============================================================
def increment_iteration_node(state: RAGState) -> Dict[str, Any]:
    iteration = int(state.get("iteration", 0) or 0) + 1
    logger.info("[crag] 开始第 %s 次重检索", iteration)
    return {"iteration": iteration}
