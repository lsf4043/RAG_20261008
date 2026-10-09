"""
问答服务（LangGraph 版）
基于 LangGraph 状态机实现 Agentic RAG 问答流程。

流程说明：
- query()       ：走完整 LangGraph 状态机（含 CRAG 自动重检索），非流式返回
- stream_query()：复用检索节点做检索，生成阶段用 LLM token 级流式输出（优化体验）

优化点（相对旧版）：
- 用 LangGraph 状态机替代线性硬编码流程
- 支持 Query 改写 / Multi-Query / HyDE（P1）
- 支持 CRAG 检索质量评估 + 自动重检索（P1）
- 集成查询缓存（P1）
"""
import asyncio
from typing import Dict, Any, Optional, Callable, AsyncGenerator

from app.config import settings
from app.core.cache import query_cache
from app.core.llm import llm_service
from app.core.langsmith import langsmith_client
from app.graph.state import initial_state
from app.graph import nodes as graph_nodes
from app.graph.graph import get_graph
from app.services.session_service import session_service
from app.utils.logger import get_logger, ProcessLogger

logger = get_logger(__name__)


class ChatService:
    """问答服务（LangGraph 状态机编排）"""

    # ============ 检索阶段（复用 LangGraph 节点函数） ============

    def _run_retrieval(self, question: str, history: list) -> Dict[str, Any]:
        """
        执行检索阶段（query 改写 + 混合检索 + CRAG + 重排序）

        手动串联 LangGraph 节点函数，实现与状态机一致的检索逻辑，
        供流式接口复用（流式接口需在生成阶段做 token 级流式）。

        Args:
            question: 用户问题
            history: 对话历史

        Returns:
            包含 documents / rewritten_query / used_fallback 的字典
        """
        state = initial_state(question, history)

        # 1. Query 改写
        state = graph_nodes.query_rewrite_node(state)
        # 2. 混合检索
        state = graph_nodes.retrieve_node(state)
        # 3. CRAG 评分
        state = graph_nodes.grade_documents_node(state)

        # 4. CRAG 路由：全部无关时重检索（最多 CRAG_MAX_ITERATIONS 次）
        iteration = 0
        while (not state["is_relevant"]) and iteration < settings.CRAG_MAX_ITERATIONS:
            iteration += 1
            logger.info(f"CRAG 重检索第 {iteration} 次")
            state["iteration"] = iteration
            state = graph_nodes.query_rewrite_node(state)
            state = graph_nodes.retrieve_node(state)
            state = graph_nodes.grade_documents_node(state)

        # 5. 重排序
        state = graph_nodes.rerank_node(state)

        return {
            "documents": state.get("documents", []),
            "rewritten_query": state.get("rewritten_query", question),
            "used_fallback": len(state.get("documents", [])) == 0,
            "iteration": iteration,
        }

    # ============ 非流式查询 ============

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_sources: bool = True,
        log_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        问答查询（LangGraph 状态机，支持多轮对话）

        Args:
            question: 用户问题
            session_id: 会话ID
            top_k: 检索知识数量
            temperature: 模型温度
            max_tokens: 最大 token 数
            return_sources: 是否返回知识来源
            log_callback: 日志回调

        Returns:
            回答结果
        """
        process_logger = ProcessLogger()
        process_logger.start()

        def log(level, step, message, progress=0, step_name=""):
            entry = getattr(process_logger, level)(step, message, progress, step_name)
            if log_callback:
                log_callback(entry)

        # LangSmith 手动 trace（方案B）
        trace = langsmith_client.trace("rag_query", {"question": question, "session_id": session_id})

        try:
            log("info", "receiving", f"收到问题: {question[:50]}", 5, "接收问题")

            history = []
            if session_id:
                history = session_service.get_history(session_id)

            # 查询缓存
            cached = query_cache.get(question, top_k)
            if cached:
                log("success", "cache", "命中查询缓存", 100, "完成")
                return {**cached, "logs": process_logger.get_logs(), "session_id": session_id}

            # 构造初始状态，用 LangGraph 状态机执行
            state = initial_state(question, history)

            log("info", "graph", "启动 LangGraph 状态机...", 10, "状态机")
            # 同步状态机放到线程池执行，避免阻塞事件循环
            # 注：top_k 等检索参数由 settings 统一管理，状态机内部读取
            result_state = await asyncio.to_thread(get_graph().invoke, state)

            answer = result_state.get("answer", "")
            sources = result_state.get("sources", [])

            if not return_sources:
                sources = []

            # 保存对话历史
            if session_id:
                session_service.add_message(session_id, "user", question)
                session_service.add_message(session_id, "assistant", answer)

            elapsed_time = process_logger.get_elapsed_time()
            log("success", "complete", f"处理完成! 总耗时: {elapsed_time:.2f}s", 100, "完成")

            result = {
                "success": True,
                "answer": answer,
                "question": question,
                "sources": sources,
                "model": settings.LLM_MODEL,
                "process_time": elapsed_time,
                "logs": process_logger.get_logs(),
                "session_id": session_id,
            }

            # 写入查询缓存
            query_cache.set(question, top_k, result)

            langsmith_client.end_trace(trace, {"answer": answer, "sources_count": len(sources)})
            return result

        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            log("error", "error", f"处理失败: {str(e)}", 0, "错误")
            return {
                "success": False,
                "error": str(e),
                "logs": process_logger.get_logs(),
            }

    # ============ 流式查询 ============

    async def stream_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式问答查询（SSE）
        检索阶段复用 LangGraph 节点，生成阶段 token 级流式

        Args:
            question: 用户问题
            session_id: 会话ID
            top_k: 检索数量
            temperature: 温度
            max_tokens: 最大 token 数

        Yields:
            流式事件
        """
        try:
            yield {"event": "start", "data": {"question": question}}

            history = []
            if session_id:
                history = session_service.get_history(session_id)

            # 检索阶段（在线程池执行）
            yield {"event": "progress", "data": {"step": "retrieving", "message": "正在检索知识库...", "progress": 20}}
            retrieval = await asyncio.to_thread(self._run_retrieval, question, history)
            documents = retrieval["documents"]

            # 发送来源
            yield {"event": "sources", "data": {"sources": [
                {
                    "text": (d.metadata.get("content", d.page_content) or "")[:100] + "...",
                    "score": d.metadata.get("rrf_score", 0),
                    "document_name": d.metadata.get("document_name", "未知文档"),
                }
                for d in documents[:3]
            ]}}

            yield {"event": "progress", "data": {"step": "generating", "message": "正在生成回答...", "progress": 60}}

            # 组装生成上下文与提示词（复用 generate 节点的 prompt 逻辑）
            context_parts = []
            for i, doc in enumerate(documents):
                text = doc.metadata.get("content", doc.page_content) or doc.page_content
                doc_name = doc.metadata.get("document_name", "未知文档")
                context_parts.append(f"【知识片段 {i + 1}】（来源: {doc_name}）\n{text}")
            context_text = "\n\n".join(context_parts)

            if context_text:
                system_prompt = f"""你是一个智能问答助手。请根据以下知识库内容回答用户问题。

## 知识库内容：
{context_text}

## 回答要求：
1. 严格基于上述知识库内容回答
2. 回答简洁、准确、有条理
3. 内容不足时说明"根据现有知识库无法准确回答"
4. 涉及之前对话内容时结合上下文回答"""
            else:
                system_prompt = """你是一个智能问答助手。知识库中没有检索到相关内容，请基于通用知识回答，保持简洁准确。"""

            # 组装对话消息（dict 格式，供 llm_service.astream 使用）
            messages_dict = list(history) + [{"role": "user", "content": question}]

            # token 级流式生成
            full_answer = ""
            async for chunk in llm_service.astream(messages_dict, system_prompt=system_prompt):
                full_answer += chunk
                yield {"event": "answer", "data": {"text": chunk, "done": False}}

            yield {"event": "answer", "data": {"text": "", "done": True}}

            # 保存对话历史
            if session_id:
                session_service.add_message(session_id, "user", question)
                session_service.add_message(session_id, "assistant", full_answer)

            yield {"event": "complete", "data": {"message": "处理完成"}}

        except Exception as e:
            logger.error(f"流式问答处理失败: {str(e)}")
            yield {"event": "error", "data": {"error": str(e)}}

    # ============ 清空会话 ============

    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        """清空会话历史"""
        session_service.clear_session(session_id)
        return {"success": True, "message": f"会话 {session_id} 已清空"}


# 全局问答服务实例
chat_service = ChatService()