"""
自定义异常类
"""
from fastapi import HTTPException
from fastapi.responses import JSONResponse


class RAGException(Exception):
    """RAG 系统基础异常"""
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(message)


class MilvusException(RAGException):
    """Milvus 相关异常"""
    def __init__(self, message: str):
        super().__init__(message, 503)


class EmbeddingException(RAGException):
    """向量化相关异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)


class LLMException(RAGException):
    """大模型相关异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)


class DocumentParseException(RAGException):
    """文档解析异常"""
    def __init__(self, message: str):
        super().__init__(message, 400)


class ValidationException(RAGException):
    """参数验证异常"""
    def __init__(self, message: str):
        super().__init__(message, 400)


class FileTooLargeException(RAGException):
    """文件过大异常"""
    def __init__(self, message: str):
        super().__init__(message, 413)


class UnsupportedFileTypeException(RAGException):
    """不支持的文件类型异常"""
    def __init__(self, message: str):
        super().__init__(message, 415)


def exception_handler(request, exc):
    """全局异常处理器"""
    if isinstance(exc, RAGException):
        return JSONResponse(
            status_code=exc.code,
            content={
                "code": exc.code,
                "message": exc.message,
                "data": None
            }
        )
    # 其他异常
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": f"服务器内部错误: {str(exc)}",
            "data": None
        }
    )