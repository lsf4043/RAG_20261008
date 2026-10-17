"""
RAG 智能问答系统（LangGraph 版）- FastAPI 应用入口

P0 修复：``lifespan`` 必须接收 app 参数。Starlette 的调用方式是
``async with self.lifespan_context(app)``（见 starlette/routing.py），
旧签名 ``async def lifespan():`` 会让服务在启动阶段直接抛 TypeError。
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.v1.router import router as api_v1_router
from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.langsmith import setup_langsmith
from app.core.reranker import get_reranker_service
from app.core.thread_pool import shutdown_thread_pool, thread_pool
from app.core.vector_store import get_vector_store_service
from app.graph.graph import get_graph
from app.services.document_service import get_document_service
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_index_cache: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    setup_logging()
    logger.info("=" * 60)
    logger.info("启动 %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("技术栈: LangChain + LangGraph + LangSmith + RAGAS")
    logger.info("=" * 60)

    if not settings.API_AUTH_TOKEN:
        logger.warning("未配置 API_AUTH_TOKEN：上传/删除接口当前无鉴权，请仅在内网使用")

    # 1. LangSmith 可观测性
    setup_langsmith()

    # 2. Milvus 连接
    logger.info("正在连接 Milvus 向量库...")
    connected = await thread_pool.run_sync(
        get_vector_store_service().connect, timeout=settings.MILVUS_TIMEOUT * 2
    )
    if connected:
        logger.info("Milvus 向量库连接成功")
        # 3. 从向量库重建 BM25（解决重启后混合检索静默退化）
        try:
            count = await get_document_service().bootstrap_bm25()
            logger.info("BM25 索引重建完成，共 %s 条 chunk", count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 索引重建失败（将仅使用向量检索）: %s", exc)
    else:
        logger.warning("Milvus 向量库连接失败：向量检索不可用，BM25 索引将为空")

    # 4. 预加载重排序模型（失败不影响启动；关闭重排时不加载，省内存和时间）
    if settings.RERANK_ENABLED:
        logger.info("正在预加载重排序模型...")
        await thread_pool.run_sync(get_reranker_service().initialize)
        logger.info("重排序模型预加载完成，可用=%s", get_reranker_service().is_available())
    else:
        logger.info("RERANK_ENABLED=false，跳过重排序模型加载")

    # 5. 编译状态机（同时校验图结构）
    get_graph()
    logger.info("LangGraph 状态机编译完成")
    logger.info("应用启动完成")

    try:
        yield
    finally:
        logger.info("正在关闭应用...")
        shutdown_thread_pool()
        logger.info("应用已关闭")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "基于 LangChain + LangGraph 的 Agentic RAG 系统：混合检索（向量 + BM25 + 加权 RRF）、"
        "Query 改写/扩展、CRAG 批量相关性评估与重检索、Cross-Encoder 重排、引用溯源。"
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS：`*` 与 credentials 不能同时开启（浏览器会拒绝），config 已自动降级
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(api_v1_router)


def _load_index_html() -> str:
    """读取前端页面（带内存缓存，避免每次请求都读盘）"""
    global _index_cache
    if _index_cache is None:
        path = os.path.join(_STATIC_DIR, "index.html")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _index_cache = handle.read()
        except OSError as exc:  # noqa: BLE001
            logger.error("读取前端页面失败: %s", exc)
            _index_cache = "<h1>RAG 服务已启动</h1><p>前端页面缺失，请访问 /docs 使用 API。</p>"
    return _index_cache


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return _load_index_html()


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """极简存活探针（不依赖任何外部服务，供反向代理/容器使用）"""
    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
