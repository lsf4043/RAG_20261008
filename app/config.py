"""
配置管理模块
基于 pydantic-settings 统一管理应用、LangChain/LangGraph、LangSmith、检索与缓存配置。

约定：
- 所有密钥只能来自环境变量 / .env，源码中不出现任何真实密钥；
- 缺失必需密钥时不在 import 阶段崩溃，而是在真正使用该能力时抛出带修复指引的
  ``ConfigurationError``，因此单元测试、/docs、健康检查等场景可以正常启动；
- 非法组合（如 chunk_overlap >= chunk_size、权重全为 0）在启动时即校验失败。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """配置缺失或不合法（消息面向使用者，包含修复指引）"""


class Settings(BaseSettings):
    """应用配置（单例，通过 get_settings 获取）"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ============ 应用信息 ============
    APP_NAME: str = "RAG 智能问答系统（LangGraph 版）"
    APP_VERSION: str = "2.1.0"
    DEBUG: bool = False

    # ============ API 配置 ============
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(8000, ge=1, le=65535)
    # 可选的简单鉴权：非空时，写操作（上传/删除/清空会话）必须带 X-API-Token
    API_AUTH_TOKEN: str = Field("", description="管理接口令牌，留空表示不鉴权")
    # 逗号分隔；"*" 表示允许所有来源（此时自动关闭 credentials，符合 CORS 规范）
    CORS_ORIGINS: str = "*"

    # ============ 阿里百炼（Embedding）============
    DASHSCOPE_API_KEY: str = Field("", description="阿里百炼 API Key，仅 Embedding 使用")

    # ============ Milvus 向量库 ============
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = Field(19530, ge=1, le=65535)
    MILVUS_COLLECTION: str = "RAG_20261008"
    MILVUS_DROP_OLD: bool = False  # 启动时删除同名集合（危险，默认关闭）
    MILVUS_TIMEOUT: float = Field(10.0, gt=0, le=120)
    MILVUS_QUERY_BATCH: int = Field(500, ge=1, le=16384)

    # ============ Embedding 配置 ============
    EMBEDDING_MODEL: str = "text-embedding-v2"
    EMBEDDING_DIMENSION: int = Field(1536, ge=1)

    # ============ 大模型配置（DeepSeek，OpenAI 兼容接口）============
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_API_KEY: str = Field("", description="DeepSeek API Key")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_MAX_TOKENS: int = Field(2000, ge=100, le=8000)
    LLM_TEMPERATURE: float = Field(0.7, ge=0.0, le=2.0)
    LLM_TIMEOUT: float = Field(60.0, gt=0, le=600)
    LLM_MAX_RETRIES: int = Field(2, ge=0, le=10)

    # ============ 文本分割配置 ============
    CHUNK_SIZE: int = Field(800, ge=100, le=4000)
    CHUNK_OVERLAP: int = Field(100, ge=0, le=500)

    # ============ 文件上传配置 ============
    MAX_FILE_SIZE: int = Field(50, ge=1, le=200)  # MB
    ALLOWED_EXTENSIONS: str = (
        "pdf,doc,docx,txt,md,rtf,csv,xls,xlsx,ppt,pptx,eml,msg,html,htm,xml,"
        "epub,png,jpg,jpeg,bmp,tiff,json,yaml,yml"
    )

    # ============ 线程池 / 超时 ============
    THREAD_POOL_SIZE: int = Field(8, ge=1, le=64)
    REQUEST_TIMEOUT: float = Field(120.0, gt=0, le=900)

    # ============ 日志配置 ============
    LOG_LEVEL: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_MAX_BYTES: int = Field(10 * 1024 * 1024, ge=1024)
    LOG_BACKUP_COUNT: int = Field(3, ge=0, le=20)

    # ============ 重试配置（Milvus / 检索类操作）============
    MAX_RETRIES: int = Field(3, ge=1, le=10)
    RETRY_DELAY: float = Field(1.0, ge=0.1, le=10.0)

    # ============ LangSmith 可观测性 ============
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str = Field("", description="LangSmith API Key")
    LANGSMITH_PROJECT: str = "RAG_20261008"
    LANGSMITH_ENDPOINT: str = "https://apac.api.smith.langchain.com"

    # ============ RAG 检索策略 ============
    RETRIEVAL_TOP_K: int = Field(5, ge=1, le=50)            # 最终进入生成的文档数
    RETRIEVAL_CANDIDATE_K: int = Field(20, ge=1, le=200)    # 召回候选数（重排前）
    RRF_K: int = Field(60, ge=1, le=200)                    # RRF 平滑常数
    VECTOR_WEIGHT: float = Field(1.0, ge=0.0, le=10.0)      # 混合检索中向量通道权重
    BM25_WEIGHT: float = Field(1.0, ge=0.0, le=10.0)        # 混合检索中 BM25 通道权重
    CONTEXT_MAX_CHARS: int = Field(12000, ge=500)           # 拼进 prompt 的上下文字符上限

    # Query 改写 / 扩展
    QUERY_REWRITE_ENABLED: bool = True
    MULTI_QUERY_ENABLED: bool = True
    MULTI_QUERY_NUM: int = Field(3, ge=1, le=8)
    HYDE_ENABLED: bool = False

    # CRAG
    CRAG_ENABLED: bool = True
    CRAG_MAX_ITERATIONS: int = Field(2, ge=0, le=5)
    CRAG_MAX_GRADE_DOCS: int = Field(20, ge=1, le=100)   # 单次评分最多送进 LLM 的文档数
    CRAG_MIN_RELEVANT: int = Field(1, ge=1)               # 至少命中几篇才算“相关”

    # 重排序
    RERANK_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "cpu"
    RERANK_TOP_K: int = Field(5, ge=1, le=20)
    RERANK_MIN_SCORE: float = Field(0.0, ge=-100.0, le=100.0)  # 低于该分数直接丢弃

    # 生成
    ALLOW_LLM_FALLBACK: bool = False  # 无检索结果时是否允许脱离知识库作答

    # 缓存
    CACHE_ENABLED: bool = True
    EMBEDDING_CACHE_SIZE: int = Field(10000, ge=1)
    QUERY_CACHE_SIZE: int = Field(1000, ge=1)
    QUERY_CACHE_TTL: int = Field(3600, ge=1)

    # ============ 会话配置 ============
    SESSION_TIMEOUT: int = Field(3600, ge=60)
    SESSION_MAX_HISTORY: int = Field(10, ge=1)
    SESSION_MAX_COUNT: int = Field(10000, ge=1)

    # ============ 文档解析 ============
    OCR_TEXT_THRESHOLD: int = Field(200, ge=0)  # 页面可提取文本低于该长度才走整页 OCR
    OCR_MIN_CONFIDENCE: float = Field(0.5, ge=0.0, le=1.0)

    # ============ 派生属性 ============
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def cors_allow_credentials(self) -> bool:
        """`*` 与 credentials 不能同时生效（浏览器会拒绝），按规范自动降级"""
        return "*" not in self.cors_origins_list

    @property
    def langsmith_enabled(self) -> bool:
        return self.LANGSMITH_TRACING and bool(self.LANGSMITH_API_KEY)

    # ============ 校验 ============
    @model_validator(mode="after")
    def _validate_combinations(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE，否则文本分割会原地死循环")
        if self.VECTOR_WEIGHT <= 0 and self.BM25_WEIGHT <= 0:
            raise ValueError("VECTOR_WEIGHT 与 BM25_WEIGHT 不能同时为 0")
        if self.RERANKER_DEVICE not in {"cpu", "cuda", "mps", "auto"} and not self.RERANKER_DEVICE.startswith("cuda:"):
            raise ValueError("RERANKER_DEVICE 仅支持 cpu / cuda / cuda:N / mps / auto")
        return self

    # ============ 必需密钥的延迟校验 ============
    def require_dashscope_key(self) -> str:
        if not self.DASHSCOPE_API_KEY:
            raise ConfigurationError(
                "缺少 DASHSCOPE_API_KEY：Embedding 依赖阿里百炼。"
                "请在 .env 或环境变量中配置后重启。"
            )
        return self.DASHSCOPE_API_KEY

    def require_deepseek_key(self) -> str:
        if not self.DEEPSEEK_API_KEY:
            raise ConfigurationError(
                "缺少 DEEPSEEK_API_KEY：生成与查询改写依赖 DeepSeek。"
                "请在 .env 或环境变量中配置后重启。"
            )
        return self.DEEPSEEK_API_KEY


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
