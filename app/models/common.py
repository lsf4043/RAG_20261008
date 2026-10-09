"""
通用响应模型
"""
from typing import Optional, Any, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    """通用API响应"""
    code: int = 200
    message: str = "success"
    data: Optional[T] = None

    class Config:
        arbitrary_types_allowed = True


def success_response(data: Any = None, message: str = "success") -> dict:
    """成功响应"""
    return {
        "code": 200,
        "message": message,
        "data": data
    }


def error_response(message: str, code: int = 500, data: Any = None) -> dict:
    """错误响应"""
    return {
        "code": code,
        "message": message,
        "data": data
    }
