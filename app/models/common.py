"""
通用请求 / 响应模型

响应体保持 ``{code, message, data}``（前端依赖该结构），同时通过
``response_model=ApiResponse[T]`` 让 /docs 具备真实的 schema。
"""
from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应信封"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: int = Field(200, description="业务状态码，与 HTTP 状态码保持一致")
    message: str = Field("success", description="提示信息")
    error: Optional[str] = Field(None, description="机器可读的错误码（成功时为 null）")
    data: Optional[T] = Field(None, description="业务数据")


def success_response(data: Any = None, message: str = "success") -> dict:
    return {"code": 200, "message": message, "data": data}


def error_response(message: str, code: int = 500, error: str = "ERROR", data: Any = None) -> dict:
    return {"code": code, "message": message, "error": error, "data": data}
