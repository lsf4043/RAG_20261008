"""
RAG 智能问答系统（LangGraph 版）- FastAPI 应用入口
基于 LangChain + LangGraph 的 Agentic RAG 系统。
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

from app.config import settings
from app.api.v1.router import router as api_v1_router
from app.core.vector_store import vector_store_service
from app.core.reranker import reranker_service
from app.core.langsmith import setup_langsmith
from app.core.thread_pool import thread_pool
from app.graph.graph import get_graph
from app.utils.logger import setup_logging, get_logger

# 设置日志
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan():
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info(f"启动 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("技术栈: LangChain + LangGraph + LangSmith + RAGAS")
    logger.info("=" * 60)

    # 1. 初始化 LangSmith 可观测性（通过环境变量启用自动追踪）
    setup_langsmith()

    # 2. 连接 Milvus 向量库（同步阻塞操作，卸载到线程池避免卡住事件循环）
    logger.info("正在连接 Milvus 向量库...")
    connected = await thread_pool.run_sync(vector_store_service.connect)
    if connected:
        logger.info("Milvus 向量库连接成功")
    else:
        logger.warning("Milvus 向量库连接失败，向量检索功能不可用")

    # 3. 预加载重排序模型
    logger.info("正在预加载重排序模型...")
    await thread_pool.run_sync(reranker_service.initialize)
    logger.info("重排序模型预加载完成")

    # 4. 构建并编译 LangGraph 状态机
    logger.info("正在编译 LangGraph 状态机...")
    get_graph()
    logger.info("LangGraph 状态机编译完成")

    logger.info("应用启动完成")

    yield

    # 关闭时
    logger.info("正在关闭应用...")
    logger.info("应用已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    description="基于 LangChain + LangGraph 的 Agentic RAG 智能问答系统，支持混合检索、Query 改写、CRAG、重排序",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
from app.core.exceptions import exception_handler as custom_exception_handler

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return custom_exception_handler(request, exc)


# 注册路由
app.include_router(api_v1_router)


# 首页路由
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


# 启动入口
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )