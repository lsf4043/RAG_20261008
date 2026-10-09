"""
问答接口（优化版：支持流式输出）
"""
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import json

from app.models.chat import ChatQueryRequest, ClearSessionRequest
from app.services.chat_service import chat_service
from app.models.common import success_response, error_response
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["智能问答"])


@router.post("/query")
async def query_chat(request: ChatQueryRequest):
    """
    智能问答接口

    用户提出问题，系统检索知识库并生成回答
    支持多轮对话（通过 session_id）
    """
    try:
        if not request.question or not request.question.strip():
            return error_response("问题不能为空", code=400)

        result = await chat_service.query(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k or 5,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            return_sources=request.return_sources if request.return_sources is not None else True
        )

        if result.get("success"):
            return success_response(data={
                "answer": result["answer"],
                "question": result["question"],
                "sources": result["sources"],
                "model": result["model"],
                "process_time": result["process_time"],
                "logs": result["logs"],
                "session_id": result.get("session_id")
            })
        else:
            return error_response(
                result.get("error", "处理失败"),
                code=500,
                data={"logs": result.get("logs", [])}
            )

    except Exception as e:
        logger.error(f"问答处理失败: {str(e)}")
        return error_response(f"处理失败: {str(e)}", code=500)


@router.post("/session/clear")
async def clear_session(request: ClearSessionRequest):
    """
    清空会话历史

    清空指定会话的对话历史，开始新的对话
    """
    try:
        result = await chat_service.clear_session(request.session_id)
        return success_response(data=result)
    except Exception as e:
        logger.error(f"清空会话失败: {str(e)}")
        return error_response(f"清空会话失败: {str(e)}", code=500)


@router.websocket("/query/logs")
async def query_chat_logs(websocket: WebSocket):
    """
    问答实时日志 WebSocket
    """
    await websocket.accept()

    def log_callback(entry):
        """日志回调，发送到 WebSocket"""
        import asyncio
        asyncio.create_task(websocket.send_json(entry))

    try:
        # 等待接收问题
        data = await websocket.receive_json()
        question = data.get("question", "")
        session_id = data.get("session_id")  # 支持多轮对话
        top_k = data.get("top_k", 5)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 2000)

        if not question:
            await websocket.send_json({
                "level": "error",
                "message": "问题不能为空"
            })
            await websocket.close()
            return

        # 处理问答
        result = await chat_service.query(
            question=question,
            session_id=session_id,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            log_callback=log_callback
        )

        # 发送最终结果
        await websocket.send_json({
            "level": "result",
            "data": result
        })

    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {str(e)}")
        await websocket.send_json({
            "level": "error",
            "message": str(e)
        })
        await websocket.close()


@router.post("/stream")
async def stream_chat(request: ChatQueryRequest):
    """
    流式问答接口（SSE）
    
    返回Server-Sent Events流，实时推送处理进度和答案
    """
    async def generate():
        try:
            async for chunk in chat_service.stream_query(
                question=request.question,
                session_id=request.session_id,
                top_k=request.top_k or 5,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                # 格式化为SSE格式
                event = chunk.get("event", "message")
                data = chunk.get("data", {})
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"流式问答失败: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
