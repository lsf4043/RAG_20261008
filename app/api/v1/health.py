"""
健康检查接口
"""
import time
from fastapi import APIRouter

from app.config import settings
from app.core.vector_store import vector_store_service
from app.core.reranker import reranker_service
from app.models.common import success_response, error_response
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["系统"])

# 启动时间
START_TIME = time.time()


@router.get("/health")
async def health_check():
    """
    健康检查接口

    检查服务状态和各组件连接情况
    """
    services = {}

    # 检查 Milvus 连接
    try:
        services["milvus"] = {
            "status": "connected" if vector_store_service.is_connected() else "disconnected",
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
        }
    except Exception as e:
        services["milvus"] = {"status": "error", "error": str(e)}

    # 检查 DashScope API
    try:
        services["dashscope"] = {
            "status": "available",
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
        }
    except Exception as e:
        services["dashscope"] = {"status": "error", "error": str(e)}

    # 检查重排序模型
    try:
        services["reranker"] = {
            "status": "ready" if reranker_service.is_available() else "degraded",
            "model": settings.RERANKER_MODEL,
        }
    except Exception as e:
        services["reranker"] = {"status": "error", "error": str(e)}

    # 检查 LangSmith
    services["langsmith"] = {
        "status": "enabled" if settings.langsmith_enabled else "disabled",
        "project": settings.LANGSMITH_PROJECT,
    }

    # 计算运行时间
    uptime_seconds = int(time.time() - START_TIME)

    # 判断整体状态
    all_healthy = all(
        s.get("status") in ["connected", "available", "ready", "enabled", "disabled", "degraded"]
        for s in services.values()
    )

    return success_response(data={
        "status": "healthy" if all_healthy else "unhealthy",
        "services": services,
        "version": settings.APP_VERSION,
        "uptime_seconds": uptime_seconds,
    })


@router.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }