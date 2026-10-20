"""
问答服务

本次修复：
1. **流式不再复制检索逻辑**：非流式走完整状态机，流式复用同一张*检索子图*再对生成做
   token 级流式，两份实现合二为一；
2. **请求级参数真正生效**：top_k / temperature / max_tokens 写进 state；
3. **缓存 key 含会话历史**，且知识库变更时由 document_service 主动失效；
4. **不再吞异常返回 200**：失败抛 RAGException，由全局异常处理器给出正确 HTTP 状态码；
5. 所有状态机调用都有超时（REQUEST_TIMEOUT），并暴露 used_fallback / degraded / notes。
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from app.config import settings
from app.core.cache import query_cache
from app.core.exceptions import RAGException
from app.core.intent_router import (
    ROUTE_DIRECT,
    ROUTE_DIRECT_FALLBACK,
    ROUTE_RAG,
    ROUTE_WEB,
    classify_intent,
)
from app.core.langsmith import traceable
from app.core.llm import get_llm_service
from app.graph import nodes as graph_nodes
from app.graph.graph import get_graph, get_retrieval_graph
from app.graph.state import initial_state
from app.services.session_service import session_service
from app.utils.logger import ProcessLogger, get_logger

logger = get_logger(__name__)

GRAPH_CONFIG = {"recursion_limit": 50}


class ChatService:
    """问答服务（LangGraph 状态机编排）"""

    # ==================== 非流式 ====================
    @traceable(name="rag_query", run_type="chain")
    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_sources: bool = True,
        log_callback: Optional[Callable[[dict], None]] = None,
    ) -> Dict[str, Any]:
        process_logger = ProcessLogger()
        process_logger.start()

        def log(level: str, step: str, message: str, progress: int = 0, step_name: str = "") -> None:
            entry = getattr(process_logger, level)(step, message, progress, step_name)
            if log_callback:
                try:
                    log_callback(entry)
                except Exception:  # noqa: BLE001
                    logger.debug("log_callback 执行失败", exc_info=True)

        log("info", "receiving", f"收到问题: {question[:50]}", 5, "接收问题")

        history = session_service.get_history(session_id) if session_id else []

        cached = query_cache.get(question, top_k, history)
        if cached:
            log("success", "cache", "命中查询缓存", 100, "完成")
            result = dict(cached)
            result["logs"] = process_logger.get_logs()
            result["session_id"] = session_id
            result["cached"] = True
            return result

        state = initial_state(
            question,
            history=history,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        log("info", "graph", "启动 LangGraph 状态机...", 20, "状态机")

        try:
            result_state = await asyncio.wait_for(
                asyncio.to_thread(get_graph().invoke, state, GRAPH_CONFIG),
                timeout=settings.REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            msg = f"问答超时（>{settings.REQUEST_TIMEOUT}s），请稍后重试或缩小检索范围"
            log("error", "timeout", msg, 0, "超时")
            raise RAGException(msg, code=504, error="TIMEOUT") from exc
        except RAGException:
            raise
        except Exception as exc:  # noqa: BLE001
            log("error", "error", f"问答失败: {exc}", 0, "错误")
            raise RAGException(f"问答处理失败: {exc}", code=500, error="RAG_FAILED") from exc

        answer = result_state.get("answer", "")
        sources = result_state.get("sources", []) if return_sources else []
        used_fallback = bool(result_state.get("used_fallback"))
        degraded = bool(result_state.get("degraded"))
        notes = list(result_state.get("notes") or [])

        if session_id and answer:
            session_service.add_message(session_id, "user", question)
            session_service.add_message(session_id, "assistant", answer)

        elapsed = process_logger.get_elapsed_time()
        log("success", "complete", f"处理完成，总耗时 {elapsed:.2f}s", 100, "完成")

        result = {
            "answer": answer,
            "question": question,
            "sources": sources,
            "model": settings.DEEPSEEK_MODEL,
            "process_time": elapsed,
            "logs": process_logger.get_logs(),
            "session_id": session_id,
            "used_fallback": used_fallback,
            "degraded": degraded,
            "notes": notes,
            "crag_iterations": int(result_state.get("iteration", 0) or 0),
            "rewritten_query": result_state.get("rewritten_query", question),
            "route": result_state.get("route", "rag"),
            "cached": False,
        }

        # 只在结果可用、且不是降级/无依据时写缓存，避免把“拒答”固化在缓存里
        if answer and not degraded and not used_fallback:
            query_cache.set(question, top_k, result, history)

        return result

    # ==================== 流式 ====================
    @traceable(name="rag_stream", run_type="chain")
    async def stream_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式问答：检索子图 → token 级流式生成"""
        history = session_service.get_history(session_id) if session_id else []
        state = initial_state(
            question,
            history=history,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        full_answer = ""
        completed = False

        try:
            yield {"event": "start", "data": {"question": question}}

            # 意图路由：闲聊/通用任务直接作答，避免无意义的检索开销
            decision = await asyncio.to_thread(classify_intent, question, history)
            if decision.is_direct:
                yield {
                    "event": "progress",
                    "data": {"step": "direct", "message": "通用对话，直接作答...", "progress": 20},
                }
                yield {"event": "sources", "data": {"sources": []}}

                messages = graph_nodes.build_direct_messages(question, history)
                async for chunk in get_llm_service().astream_messages(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    full_answer += chunk
                    yield {"event": "answer", "data": {"text": chunk, "done": False}}

                yield {"event": "answer", "data": {"text": "", "done": True}}
                yield {
                    "event": "complete",
                    "data": {
                        "message": "处理完成",
                        "route": ROUTE_DIRECT,
                        "used_fallback": True,
                        "degraded": False,
                        "notes": [f"意图路由：直接作答（{decision.reason}）"],
                    },
                }
                completed = True
                return

            yield {
                "event": "progress",
                "data": {"step": "retrieving", "message": "正在检索知识库...", "progress": 20},
            }

            try:
                retrieval_state = await asyncio.wait_for(
                    asyncio.to_thread(get_retrieval_graph().invoke, state, GRAPH_CONFIG),
                    timeout=settings.REQUEST_TIMEOUT,
                )
            except asyncio.TimeoutError as exc:
                raise RAGException(
                    f"检索超时（>{settings.REQUEST_TIMEOUT}s）", code=504, error="TIMEOUT"
                ) from exc

            documents = list(retrieval_state.get("documents") or [])
            notes = list(retrieval_state.get("notes") or [])
            degraded = bool(retrieval_state.get("degraded"))
            merged_state = {**state, **retrieval_state}

            route = "rag"
            used_fallback = False
            kb_answered = False

            def _kb_source_list(items):
                return [
                    {
                        "text": graph_nodes.doc_text(doc)[:200],
                        "score": (doc.metadata or {}).get(
                            "rerank_score", (doc.metadata or {}).get("rrf_score", 0.0)
                        ),
                        "document_name": (doc.metadata or {}).get("document_name", "未知文档"),
                        "page_num": (doc.metadata or {}).get("page_num", 0),
                        "title": (doc.metadata or {}).get("title", ""),
                    }
                    for doc in items[:3]
                ]

            # ---------- 第一级：知识库（模型声明 [[KB]] / [[GENERAL]]）----------
            if documents:
                yield {
                    "event": "progress",
                    "data": {"step": "generating", "message": "正在生成回答...", "progress": 60},
                }
                messages, _ = graph_nodes.build_generation_messages(question, history, documents)
                kb_sources = _kb_source_list(documents)
                buffer = ""
                decided = False
                general_declared = False

                async for chunk in get_llm_service().astream_messages(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    if decided:
                        full_answer += chunk
                        yield {"event": "answer", "data": {"text": chunk, "done": False}}
                        continue

                    buffer += chunk
                    stripped = buffer.lstrip()
                    ready = (
                        stripped.startswith(graph_nodes.PROVENANCE_KB)
                        or stripped.startswith(graph_nodes.PROVENANCE_GENERAL)
                        or "\n" in buffer
                        or len(buffer) >= 40
                    )
                    if not ready:
                        continue

                    decided = True
                    tag, body = graph_nodes.split_provenance(buffer)
                    if tag == graph_nodes.PROVENANCE_GENERAL:
                        general_declared = True
                        notes = notes + ["模型判定知识库内容不足以回答"]
                        break  # 放弃这次生成，进入下一级兜底

                    yield {"event": "sources", "data": {"sources": kb_sources}}
                    if body:
                        full_answer += body
                        yield {"event": "answer", "data": {"text": body, "done": False}}

                if not general_declared:
                    if not decided:
                        tag, body = graph_nodes.split_provenance(buffer)
                        if tag == graph_nodes.PROVENANCE_GENERAL:
                            notes = notes + ["模型判定知识库内容不足以回答"]
                        else:
                            yield {"event": "sources", "data": {"sources": kb_sources}}
                            if body:
                                full_answer += body
                                yield {"event": "answer", "data": {"text": body, "done": False}}
                            kb_answered = True
                    else:
                        kb_answered = True
                    if kb_answered:
                        route = ROUTE_RAG
                        used_fallback = False

            # ---------- 第二级：联网 ----------
            answered = kb_answered
            if not answered and settings.WEB_SEARCH_ENABLED:
                yield {
                    "event": "progress",
                    "data": {"step": "web", "message": "知识库不足，正在联网检索...", "progress": 50},
                }
                outcome = await asyncio.wait_for(
                    asyncio.to_thread(
                        graph_nodes.web_search, {**merged_state, "notes": notes}
                    ),
                    timeout=settings.REQUEST_TIMEOUT,
                )
                notes = list(outcome.get("notes") or notes)
                web_documents = list(outcome.get("documents") or [])
                if web_documents:
                    route = ROUTE_WEB
                    used_fallback = False
                    yield {
                        "event": "sources",
                        "data": {"sources": graph_nodes.web_sources(web_documents)[:3]},
                    }
                    messages = graph_nodes.build_web_messages(question, history, web_documents)
                    async for chunk in get_llm_service().astream_messages(
                        messages, temperature=temperature, max_tokens=max_tokens
                    ):
                        full_answer += chunk
                        yield {"event": "answer", "data": {"text": chunk, "done": False}}
                    answered = True

            # ---------- 第三级：通用知识 / 拒答 ----------
            if not answered:
                if not settings.ALLOW_LLM_FALLBACK:
                    full_answer = graph_nodes.NO_CONTEXT_ANSWER
                    yield {"event": "sources", "data": {"sources": []}}
                    yield {"event": "answer", "data": {"text": full_answer, "done": False}}
                    yield {
                        "event": "complete",
                        "data": {
                            "message": "知识库中未找到相关内容",
                            "route": ROUTE_RAG,
                            "used_fallback": True,
                            "degraded": degraded,
                            "notes": notes + ["知识库无相关依据，已按策略拒答"],
                        },
                    }
                    completed = True
                    return

                route = ROUTE_DIRECT_FALLBACK
                used_fallback = True
                notes = notes + ["知识库未命中，改用通用知识作答（非知识库依据）"]
                yield {"event": "sources", "data": {"sources": []}}
                messages = graph_nodes.build_direct_messages(question, history)
                async for chunk in get_llm_service().astream_messages(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    full_answer += chunk
                    yield {"event": "answer", "data": {"text": chunk, "done": False}}

            yield {"event": "answer", "data": {"text": "", "done": True}}
            yield {
                "event": "complete",
                "data": {
                    "message": "处理完成",
                    "route": route,
                    "used_fallback": used_fallback,
                    "degraded": degraded,
                    "notes": notes,
                },
            }
            completed = True

        except RAGException as exc:
            logger.error("流式问答失败: %s", exc.message)
            yield {"event": "error", "data": {"error": exc.message, "code": exc.code}}
        except Exception as exc:  # noqa: BLE001
            logger.error("流式问答失败: %s", exc, exc_info=True)
            yield {"event": "error", "data": {"error": "生成回答失败，请稍后重试", "code": 500}}
        finally:
            if session_id and full_answer:
                suffix = "" if completed else "（回答中断）"
                session_service.add_message(session_id, "user", question)
                session_service.add_message(session_id, "assistant", full_answer + suffix)

    # ==================== 会话 ====================
    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        session_service.clear_session(session_id)
        return {"success": True, "message": f"会话 {session_id} 已清空"}


chat_service = ChatService()
