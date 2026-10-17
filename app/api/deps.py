"""
API 依赖：可选的令牌鉴权

设置 ``API_AUTH_TOKEN`` 后，写操作（上传 / 删除 / 清空会话）必须携带
``X-API-Token`` 头（WebSocket 用 ``?token=`` 查询参数）。
留空表示不开启鉴权（本地演示默认）。
"""
from __future__ import annotations

from typing import Optional

from fastapi import Header, WebSocket

from app.config import settings
from app.core.exceptions import AuthenticationError


async def require_admin_token(x_api_token: Optional[str] = Header(None, alias="X-API-Token")) -> None:
    """写操作鉴权"""
    if not settings.API_AUTH_TOKEN:
        return
    if not x_api_token or x_api_token != settings.API_AUTH_TOKEN:
        raise AuthenticationError("缺少或错误的 X-API-Token")


async def require_ws_token(websocket: WebSocket) -> None:
    """WebSocket 鉴权（token 走查询参数）"""
    if not settings.API_AUTH_TOKEN:
        return
    token = websocket.query_params.get("token")
    if not token or token != settings.API_AUTH_TOKEN:
        raise AuthenticationError("缺少或错误的 token")
