"""
问答接口

修复点：
- 统一使用 HTTP 状态码（异常由全局处理器转换），并补齐 response_model；
- 请求参数全部带边界校验（见 models/chat.py）；
- WebSocket 进度推送改为「队列 + 线程安全投递」，不再从线程里裸调
  ``asyncio.create_task``（那会在工作线程里抛 "no running event loop"）。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.api.deps import require_admin_token, require_ws_token
from app.core.exceptions import AuthenticationError, RAGException
from app.models.chat import ChatData, ChatQueryRequest, ClearSessionData, ClearSessionRequest
from app.models.common import ApiResponse, success_response
from app.services.chat_service import chat_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["智能问答"])


@router.post("/query", response_model=ApiResponse[ChatData], summary="智能问答")
async def query_chat(request: ChatQueryRequest):
    result = await chat_service.query(
        question=request.question.strip(),
        session_id=request.session_id,
        top_k=request.top_k,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        return_sources=request.return_sources,
    )
    return success_response(data=result)


@router.post(
    "/session/clear",
    response_model=ApiResponse[ClearSessionData],
    dependencies=[Depends(require_admin_token)],
    summary="清空会话历史",
)
async def clear_session(request: ClearSessionRequest):
    result = await chat_service.clear_session(request.session_id)
    return success_response(data=result)


@router.post("/stream", summary="流式问答（SSE）")
async def stream_chat(request: ChatQueryRequest):
    """Server-Sent Events 流式问答"""

    async def generate():
        try:
            async for chunk in chat_service.stream_query(
                question=request.question.strip(),
                session_id=request.session_id,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                event = chunk.get("event", "message")
                payload = chunk.get("data", {})
                yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            logger.info("客户端断开 SSE 连接")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("流式问答失败: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': '生成回答失败'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.websocket("/query/logs")
async def query_chat_logs(websocket: WebSocket):
    """问答实时日志（WebSocket）：推送处理进度 + 最终结果"""
    await websocket.accept()

    try:
        await require_ws_token(websocket)
    except AuthenticationError as exc:
        await websocket.send_json({"level": "error", "message": exc.message})
        await websocket.close(code=1008)
        return

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        logger.info("WebSocket 在接收请求前断开")
        return

    question = str(data.get("question") or "").strip()
    if not question:
        await websocket.send_json({"level": "error", "message": "问题不能为空"})
        await websocket.close(code=1008)
        return

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def log_callback(entry: Dict[str, Any]) -> None:
        # 可能从工作线程调用，必须线程安全地投递
        loop.call_soon_threadsafe(queue.put_nowait, entry)

    task = asyncio.create_task(
        chat_service.query(
            question=question,
            session_id=data.get("session_id"),
            top_k=_as_int(data.get("top_k"), 5),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            log_callback=log_callback,
        )
    )

    try:
        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=0.1)
                await websocket.send_json(entry)
            except asyncio.TimeoutError:
                if task.done():
                    break

        result = await task
        await websocket.send_json({"level": "result", "data": result})
    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
        task.cancel()
    except RAGException as exc:
        await websocket.send_json({"level": "error", "code": exc.code, "message": exc.message})
    except Exception as exc:  # noqa: BLE001
        logger.error("WebSocket 处理失败: %s", exc, exc_info=True)
        await websocket.send_json({"level": "error", "message": "服务器内部错误"})
    finally:
        if not task.done():
            task.cancel()
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
