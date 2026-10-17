"""
自定义异常与全局异常处理

统一约定：
- 业务异常继承 ``RAGException``，携带 HTTP 状态码 + 机器可读错误码；
- 响应体保持 ``{"code", "message", "data"}`` 结构（前端依赖该结构），
  但 **HTTP 状态码与 code 字段保持一致**，不再一律返回 200。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class RAGException(Exception):
    """RAG 系统基础异常"""

    code: int = 500
    error: str = "RAG_ERROR"

    def __init__(self, message: str, code: Optional[int] = None, error: Optional[str] = None):
        self.message = message
        if code is not None:
            self.code = code
        if error is not None:
            self.error = error
        super().__init__(message)


class MilvusException(RAGException):
    code = 503
    error = "MILVUS_UNAVAILABLE"


class EmbeddingException(RAGException):
    code = 502
    error = "EMBEDDING_FAILED"


class LLMException(RAGException):
    code = 502
    error = "LLM_FAILED"


class DocumentParseException(RAGException):
    code = 400
    error = "DOCUMENT_PARSE_FAILED"


class ValidationException(RAGException):
    code = 400
    error = "VALIDATION_FAILED"


class FileTooLargeException(RAGException):
    code = 413
    error = "FILE_TOO_LARGE"


class UnsupportedFileTypeException(RAGException):
    code = 415
    error = "UNSUPPORTED_FILE_TYPE"


class AuthenticationError(RAGException):
    code = 401
    error = "UNAUTHORIZED"


class NotFoundException(RAGException):
    code = 404
    error = "NOT_FOUND"


def error_body(message: str, code: int, error: str = "ERROR", data: Any = None) -> Dict[str, Any]:
    return {"code": code, "message": message, "error": error, "data": data}


def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """业务异常 → 对应 HTTP 状态码"""
    if exc.code >= 500:
        logger.error("业务异常 %s: %s", exc.error, exc.message, exc_info=True)
    else:
        logger.warning("请求被拒绝 %s: %s", exc.error, exc.message)
    return JSONResponse(status_code=exc.code, content=error_body(exc.message, exc.code, exc.error))


def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """参数校验失败 → 422（保持可读的错误详情）"""
    details = []
    for item in exc.errors():
        loc = ".".join(str(part) for part in item.get("loc", []))
        details.append({"field": loc, "message": item.get("msg", "")})
    message = "; ".join(f"{d['field']}: {d['message']}" for d in details) or "请求参数校验失败"
    return JSONResponse(
        status_code=422,
        content=error_body(message, 422, "VALIDATION_ERROR", {"errors": details}),
    )


def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Starlette/FastAPI 的 HTTPException → 统一响应体"""
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(str(exc.detail), exc.status_code, "HTTP_ERROR"),
    )


def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：不把内部堆栈细节直接暴露给客户端"""
    logger.error("未处理的异常: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_body("服务器内部错误，请联系管理员（详情见服务端日志）", 500, "INTERNAL_ERROR"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """注册全部异常处理器（注意：必须注册具体异常类，不能直接注册 Exception 后手写分发）"""
    app.add_exception_handler(RAGException, rag_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


# ---- 兼容旧调用（app.main 之前导入过 exception_handler） ----
def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RAGException):
        return rag_exception_handler(request, exc)
    return unhandled_exception_handler(request, exc)
