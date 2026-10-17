"""
健康检查接口

修复点：
- ``/health`` 只做**轻量存活检查**（不产生外部调用），但状态是真实计算的：
  缺失 DeepSeek/DashScope Key、Milvus 未连接都会如实反映；
- ``/health/ready`` 做**就绪检查**：真正 ping Milvus，未就绪返回 503，
  Docker / K8s 的健康检查挂这个接口才有意义；
- ``/stats`` 暴露缓存命中率、BM25 文档数、会话数等运行指标。
"""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.bm25_retriever import get_bm25_service
from app.core.cache import embedding_cache, query_cache
from app.core.embeddings import get_embedding_service
from app.core.llm import get_llm_service
from app.core.reranker import get_reranker_service
from app.core.vector_store import get_vector_store_service
from app.models.common import ApiResponse, success_response
from app.models.document import HealthData, StatsData
from app.services.session_service import session_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["系统"])

START_TIME = time.time()


def _collect_services() -> Dict[str, Any]:
    store = get_vector_store_service()
    return {
        "milvus": {
            "status": "connected" if store.is_connected() else "disconnected",
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
            "collection": settings.MILVUS_COLLECTION,
            # ok / incompatible：集合存在但 schema 与本应用不匹配（上传会失败）
            "schema": "incompatible" if store.schema_warning else "ok",
            "schema_warning": store.schema_warning,
        },
        "llm": {
            "status": "configured" if get_llm_service().is_configured() else "missing_api_key",
            "model": settings.DEEPSEEK_MODEL,
        },
        "embedding": {
            "status": "configured" if get_embedding_service().is_configured() else "missing_api_key",
            "model": settings.EMBEDDING_MODEL,
        },
        "bm25": {
            "status": "ready" if get_bm25_service().is_ready() else "empty",
            "documents": get_bm25_service().count,
        },
        "reranker": {
            "status": "ready" if get_reranker_service().is_available() else "degraded",
            "model": settings.RERANKER_MODEL,
        },
        "langsmith": {"status": "enabled" if settings.langsmith_enabled else "disabled"},
    }


def _overall_status(services: Dict[str, Any]) -> str:
    if services["milvus"]["status"] != "connected":
        return "unhealthy"
    if services["llm"]["status"] != "configured" or services["embedding"]["status"] != "configured":
        return "unhealthy"
    if services["milvus"]["schema"] != "ok":
        return "degraded"
    if services["reranker"]["status"] != "ready" or services["bm25"]["status"] != "ready":
        return "degraded"
    return "healthy"


@router.get("/health", response_model=ApiResponse[HealthData], summary="存活检查")
async def health_check():
    services = _collect_services()
    return success_response(
        data={
            "status": _overall_status(services),
            "version": settings.APP_VERSION,
            "uptime_seconds": int(time.time() - START_TIME),
            "services": services,
        }
    )


@router.get("/health/ready", summary="就绪检查（真正 ping Milvus）")
async def readiness_check():
    health = get_vector_store_service().health()
    ready = (
        health.get("status") == "connected"
        and health.get("schema") == "ok"
        and get_llm_service().is_configured()
        and get_embedding_service().is_configured()
    )
    payload = {
        "status": "ready" if ready else "not_ready",
        "version": settings.APP_VERSION,
        "milvus": health,
        "llm_configured": get_llm_service().is_configured(),
        "embedding_configured": get_embedding_service().is_configured(),
        "bm25_documents": get_bm25_service().count,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@router.get("/stats", response_model=ApiResponse[StatsData], summary="运行统计")
async def stats():
    return success_response(
        data={
            "embedding_cache": embedding_cache.get_stats(),
            "query_cache": query_cache.get_stats(),
            "bm25_documents": get_bm25_service().count,
            "sessions": session_service.get_session_count(),
            "collection": get_vector_store_service().get_collection_stats(),
        }
    )
