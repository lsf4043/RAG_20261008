"""
配置管理模块（升级版）
基于 pydantic-settings，统一管理应用、LangChain、LangGraph、LangSmith、检索等全部配置。

相比旧版新增：
- LangSmith 可观测性配置（通过环境变量自动追踪 LangChain/LangGraph 调用）
- RAG 检索策略配置（Query 改写、CRAG、重排序、混合检索开关）
- 缓存配置（Embedding 缓存 + 查询缓存）
- 会话配置（超时、历史长度）
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置（单例，通过 get_settings 获取）"""

    # ============ 应用信息 ============
    APP_NAME: str = "RAG 智能问答系统（LangGraph 版）"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ============ API 配置 ============
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(8000, ge=1, le=65535)

    # ============ 阿里百炼 API ============
    DASHSCOPE_API_KEY: str = Field(..., min_length=32, description="阿里百炼 API Key")

    # ============ Milvus 向量库 ============
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = Field(19530, ge=1, le=65535)
    MILVUS_COLLECTION: str = "RAG_20261008"
    MILVUS_DROP_OLD: bool = False  # 是否在启动时删除已存在的同名集合（危险，默认关闭）

    # ============ Embedding 配置 ============
    EMBEDDING_MODEL: str = "text-embedding-v2"
    EMBEDDING_DIMENSION: int = Field(1536, ge=1)

    # ============ 大模型配置 ============
    LLM_MODEL: str = "qwen-turbo"
    LLM_MAX_TOKENS: int = Field(2000, ge=100, le=8000)
    LLM_TEMPERATURE: float = Field(0.7, ge=0.0, le=1.0)

    # ============ 文本分割配置 ============
    CHUNK_SIZE: int = Field(800, ge=100, le=4000)
    CHUNK_OVERLAP: int = Field(100, ge=0, le=500)

    # ============ 文件上传配置 ============
    MAX_FILE_SIZE: int = Field(50, ge=1, le=100)  # MB
    ALLOWED_EXTENSIONS: str = "pdf,doc,docx,txt,md,rtf,csv,xls,xlsx,ppt,pptx,eml,msg,html,htm,xml,epub,png,jpg,jpeg,bmp,tiff,json,yaml,yml"

    # ============ 线程池配置 ============
    THREAD_POOL_SIZE: int = Field(4, ge=1, le=32)

    # ============ 日志配置 ============
    LOG_LEVEL: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "logs/app.log"

    # ============ 重试配置 ============
    MAX_RETRIES: int = Field(3, ge=1, le=10)
    RETRY_DELAY: float = Field(1.0, ge=0.1, le=10.0)

    # ============ LangSmith 可观测性配置 ============
    # 字段名对齐 .env 中的 LANGSMITH_*；启动时由 setup_langsmith() 映射写入
    # SDK 实际读取的 LANGCHAIN_* 环境变量（如 LANGCHAIN_TRACING_V2=true）
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str = Field("", description="LangSmith API Key，请通过环境变量 / .env 提供，切勿硬编码到源码中")
    LANGSMITH_PROJECT: str = "RAG_20261008"
    LANGSMITH_ENDPOINT: str = "https://apac.api.smith.langchain.com"

    # ============ RAG 检索策略配置 ============
    RETRIEVAL_TOP_K: int = Field(5, ge=1, le=50)            # 最终返回的文档数量
    RETRIEVAL_CANDIDATE_K: int = Field(20, ge=1, le=100)    # 召回候选数量（重排前）
    VECTOR_WEIGHT: float = Field(0.5, ge=0.0, le=1.0)       # 混合检索中向量权重
    BM25_WEIGHT: float = Field(0.5, ge=0.0, le=1.0)         # 混合检索中 BM25 权重

    # Query 改写（P1 优化）
    QUERY_REWRITE_ENABLED: bool = True          # 是否启用多轮指代消解
    MULTI_QUERY_ENABLED: bool = True            # 是否启用 Multi-Query 扩展
    MULTI_QUERY_NUM: int = Field(3, ge=1, le=8)  # Multi-Query 生成的变体数量
    HYDE_ENABLED: bool = False                  # 是否启用 HyDE（假设答案嵌入，需额外 LLM 调用）

    # CRAG（Corrective RAG，P1 优化）
    CRAG_ENABLED: bool = True                   # 是否启用检索质量评估
    CRAG_MAX_ITERATIONS: int = Field(2, ge=1, le=5)  # CRAG 最大重检索次数

    # 重排序
    RERANK_ENABLED: bool = True                 # 是否启用重排序
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_TOP_K: int = Field(5, ge=1, le=20)  # 重排序后保留的数量

    # 缓存
    CACHE_ENABLED: bool = True                  # 是否启用缓存
    EMBEDDING_CACHE_SIZE: int = Field(10000, ge=1)      # Embedding 缓存条数
    QUERY_CACHE_SIZE: int = Field(1000, ge=1)           # 查询结果缓存条数
    QUERY_CACHE_TTL: int = Field(3600, ge=1)            # 查询缓存过期时间（秒）

    # ============ 会话配置 ============
    SESSION_TIMEOUT: int = Field(3600, ge=60)         # 会话超时时间（秒）
    SESSION_MAX_HISTORY: int = Field(10, ge=1)        # 每个会话保留的最大历史轮数

    # ============ 属性 ============
    @property
    def allowed_extensions_list(self) -> List[str]:
        """获取允许的文件扩展名列表"""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def langsmith_enabled(self) -> bool:
        """判断 LangSmith 追踪是否可用（需要显式配置 API Key）"""
        return self.LANGSMITH_TRACING and bool(self.LANGSMITH_API_KEY)

    @field_validator('DASHSCOPE_API_KEY')
    @classmethod
    def validate_api_key(cls, v):
        """校验阿里百炼 API Key 长度（至少 32 字符）"""
        if not v or len(v) < 32:
            raise ValueError('DASHSCOPE_API_KEY 必须至少 32 字符')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # 允许读取的环境变量（大小写不敏感）
        case_sensitive = False
        # 忽略 .env 中 Settings 未定义的变量（兼容原项目 .env、全局环境变量）
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例（带缓存，避免重复读取 .env）"""
    return Settings()


# 全局配置实例（供各模块直接 import）
settings = get_settings()