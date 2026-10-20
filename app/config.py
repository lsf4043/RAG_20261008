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

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/config.py 的上一级）：.env 用绝对路径定位，
# 避免从 app/ 等子目录启动时相对路径找不到 .env 而静默退回默认值
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


# 阿里百炼通用文本向量模型的规格表（来源：百炼官方「同步接口API详情」）
# - dimensions：允许的输出维度，**必须与 EMBEDDING_DIMENSION 一致**
# - max_batch：单请求最大行数（v3/v4 为 10，v1/v2 为 25）
# - max_tokens：单行最大 token 数（v3/v4 为 8192，v1/v2 为 2048）
# - instruct：是否支持查询侧任务指令（Qwen3-Embedding 系列的检索增强用法）
EMBEDDING_MODEL_SPECS: dict[str, dict] = {
    "text-embedding-v4": {
        "dimensions": {2048, 1536, 1024, 768, 512, 256, 128, 64},
        "default_dimension": 1024,
        "max_batch": 10,
        "max_tokens": 8192,
        "instruct": True,
    },
    "text-embedding-v3": {
        "dimensions": {1024, 768, 512, 256, 128, 64},
        "default_dimension": 1024,
        "max_batch": 10,
        "max_tokens": 8192,
        "instruct": True,
    },
    "text-embedding-v2": {
        "dimensions": {1536},
        "default_dimension": 1536,
        "max_batch": 25,
        "max_tokens": 2048,
        "instruct": False,
    },
    "text-embedding-v1": {
        "dimensions": {1536},
        "default_dimension": 1536,
        "max_batch": 25,
        "max_tokens": 2048,
        "instruct": False,
    },
}


class ConfigurationError(RuntimeError):
    """配置缺失或不合法（消息面向使用者，包含修复指引）"""


class Settings(BaseSettings):
    """应用配置（单例，通过 get_settings 获取）"""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
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
    # 模型规格见 EMBEDDING_MODEL_SPECS；切换模型时 **必须** 同步调整 EMBEDDING_DIMENSION，
    # 否则 Milvus 集合维度与向量维度不一致，写入会失败（启动时会校验并明确报错）。
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_DIMENSION: int = Field(1536, ge=1)
    # 单请求行数上限：v3/v4 为 10，v1/v2 为 25；超过模型上限时会被自动收敛
    EMBEDDING_BATCH_SIZE: int = Field(10, ge=1, le=25)
    # 查询侧任务指令（仅 v3/v4 生效；留空表示不发送 instruct）
    # Qwen3-Embedding 官方建议用一句英文描述检索任务，可提升召回
    EMBEDDING_QUERY_INSTRUCT: str = (
        "Given a user question, retrieve the passages from the knowledge base that answer it"
    )

    # ============ 大模型配置（DeepSeek，OpenAI 兼容接口）============
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_API_KEY: str = Field("", description="DeepSeek API Key")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_MAX_TOKENS: int = Field(2000, ge=100, le=8000)
    LLM_TEMPERATURE: float = Field(0.7, ge=0.0, le=2.0)
    LLM_TIMEOUT: float = Field(60.0, gt=0, le=600)
    LLM_MAX_RETRIES: int = Field(2, ge=0, le=10)
    # 结构化输出（Query 规划 / CRAG 评分）是否关闭 DeepSeek 思考模式：
    # deepseek-flash 默认开启思考，而思考模式不支持强制 tool_choice（function_calling 会 400），
    # 且会忽略 temperature；结构化任务本身也不需要思维链
    LLM_STRUCTURED_DISABLE_THINKING: bool = True

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
    # 相似度门控：向量检索最高分低于该值时，判定“知识库无相关内容”，
    # 直接走通用知识作答（跳过 CRAG 评分与重排，常识问题可从 ~17s 降到 ~3s）。
    # 该阈值与 Embedding 模型强相关，必须按自己的语料标定；设为 0 表示关闭。
    #
    # 标定依据（text-embedding-v4 + 简历/论文语料，cos 最高分）：
    #   相关问题：0.34(我的籍贯) 0.51(论文结论) 0.55(简历技能) 0.56(论文方法) 0.58(Milvus)
    #   无关问题：0.21(重庆火锅) 0.25(推荐小说) 0.30(咸阳美食) 0.31(天气) 0.36(宝鸡美食)
    # 注意：两组有重叠（相关最低 0.34 < 无关最高 0.36），**余弦无法单独判定“有没有答案”**。
    # 所以这里只用来短路“明显无关”的查询，不承担“能不能答”的判断——
    # 后者由生成模型读证据后声明 [[KB]] / [[GENERAL]] 决定（见 nodes.py）。
    RETRIEVAL_MIN_SCORE: float = Field(0.25, ge=0.0, le=1.0)
    RRF_K: int = Field(60, ge=1, le=200)                    # RRF 平滑常数
    VECTOR_WEIGHT: float = Field(1.0, ge=0.0, le=10.0)      # 混合检索中向量通道权重
    BM25_WEIGHT: float = Field(1.0, ge=0.0, le=10.0)        # 混合检索中 BM25 通道权重
    CONTEXT_MAX_CHARS: int = Field(12000, ge=500)           # 拼进 prompt 的上下文字符上限

    # Query 改写 / 扩展
    QUERY_REWRITE_ENABLED: bool = True
    MULTI_QUERY_ENABLED: bool = True
    MULTI_QUERY_NUM: int = Field(3, ge=1, le=8)
    HYDE_ENABLED: bool = False

    # CRAG（Corrective RAG）
    # 默认关闭：它的“逐篇相关性过滤”与 Cross-Encoder 重排 + RETRIEVAL_MIN_SCORE 高度重叠，
    # 而“检索失败→重写再查”在单域知识库里基本无效（库里没有的内容重写也查不到），
    # 常态成本却是每个问题 1~3 次额外 LLM 调用、2~8s 延迟。
    # 何时开启：语料嘈杂、top-k 经常混入大量无关片段，或将来接入了联网兜底
    # （CRAG 原论文的“纠正”是转去 web search）。开启前建议先跑 RAGAS 消融确认收益。
    CRAG_ENABLED: bool = False
    CRAG_MAX_ITERATIONS: int = Field(2, ge=0, le=5)
    CRAG_MAX_GRADE_DOCS: int = Field(20, ge=1, le=100)   # 单次评分最多送进 LLM 的文档数
    CRAG_MIN_RELEVANT: int = Field(1, ge=1)               # 至少命中几篇才算“相关”

    # 重排序
    RERANK_ENABLED: bool = True
    # 主通道：dashscope = 阿里百炼 qwen3.7-text-rerank（异常自动降级本地）；local = 仅本地 Cross-Encoder
    RERANK_PROVIDER: str = "dashscope"
    # 商业 API 重排（复用 DASHSCOPE_API_KEY）
    RERANK_API_MODEL: str = "qwen3.7-text-rerank"
    RERANK_API_TIMEOUT: float = Field(10.0, gt=0, le=120)
    # 自定义排序任务说明（英文效果更好；留空则使用模型默认的问答检索策略）
    RERANK_API_INSTRUCT: str = (
        "Given a user question, retrieve the passages from the knowledge base that answer it"
    )
    # 送进 API 的单条文档截断字符数（模型单条上限 30000 token，留足余量）
    RERANK_API_MAX_CHARS: int = Field(4000, ge=100, le=30000)
    # 单次请求最多送进 API 的文档数（模型上限 500；默认值同时控制请求 token 总量）
    RERANK_API_MAX_DOCS: int = Field(50, ge=1, le=500)
    # API 连续失败后的冷却时间（秒），期间直接走本地，避免反复白等超时
    RERANK_API_COOLDOWN: float = Field(60.0, ge=0.0, le=3600.0)
    # 本地兜底通道
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "cpu"
    RERANK_TOP_K: int = Field(5, ge=1, le=20)
    # 重排分数下限：**默认关闭（0）**。
    # 曾经用它做“库里有没有答案”的判定，但实测相关最低 0.43 / 无关最高 0.39，
    # 只差 0.04，属于“拟合出来的阈值”，会出现 0.401 走知识库、0.399 走通用知识的悬崖。
    # 现在改由生成模型读证据后声明来源（[[KB]] / [[GENERAL]]），这个值只作为可选的
    # 噪声预过滤保留：设成 >0 时才会丢弃低分文档。
    RERANK_MIN_SCORE: float = Field(0.0, ge=-100.0, le=100.0)

    # HuggingFace 模型下载源（国内网络直连 huggingface.co 常因证书校验失败，
    # 默认走镜像；海外环境可在 .env 中设为 https://huggingface.co）
    HF_ENDPOINT: str = "https://hf-mirror.com"

    # 时区：用于把“当前时间”注入提示词（模型自己不知道今天是几号）
    APP_TIMEZONE: str = "Asia/Shanghai"

    # 生成策略：知识库检索不到内容时，是否允许用大模型的通用知识作答
    # 产品决策：默认开启（闲聊直答 / 知识类先查库 / 查不到→通用知识作答，并在响应里标注）
    # 严格合规场景可设为 false：查不到就明确拒答
    ALLOW_LLM_FALLBACK: bool = True

    # ============ 联网检索（可选，默认关闭）============
    # 触发条件：知识库不足以回答（无候选，或最高重排分 < WEB_SEARCH_MIN_SCORE）
    # 且 WEB_SEARCH_ENABLED=true。联网也失败时回落到通用知识作答。
    WEB_SEARCH_ENABLED: bool = True
    # deepseek：官方 Search API（POST /v1/tools/search，复用 DEEPSEEK_API_KEY）
    #     **推荐**：结构化结果（rerank_score 0~3 / authority / date / summary 600~800 字），
    #     比抓 HTML 稳定；按成功请求计费（$0.75 / 1000 次）
    # so：360 搜索，免 Key，中文长尾不错（但要解析 HTML）
    # bing：免 Key，中文长尾很差（会把“周原镇党委书记”拆成“周”字）
    # duckduckgo：免 Key，需要能访问外网
    # tavily：需要自己的 API Key，结果稳定
    WEB_SEARCH_PROVIDER: str = "deepseek"
    WEB_SEARCH_API_KEY: str = ""
    # 取多少条结果。DeepSeek Search 按“请求”计费，与 count 无关，取 5 条覆盖更好
    WEB_SEARCH_MAX_RESULTS: int = Field(5, ge=1, le=10)
    WEB_SEARCH_TIMEOUT: float = Field(10.0, gt=0, le=60)
    # 知识库最高重排分低于该值时才去联网（高于它就认为库里有答案）
    WEB_SEARCH_MIN_SCORE: float = Field(0.35, ge=0.0, le=1.0)
    # 拼进 prompt 的网页摘要总字符上限
    WEB_SEARCH_MAX_CHARS: int = Field(4000, ge=200)

    # ============ 意图路由 ============
    # 只做一件事：识别“闲聊/寒暄”走直答（闭集、零成本、误判代价极低）。
    # 其余一切输入都先检索知识库，检索不到再由生成节点用通用知识作答。
    # 不再用规则判断“是不是知识问题”——那类语义判断必然漏（见 README 2.8）。
    ROUTER_ENABLED: bool = True

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
    def _validate_embeddings(self) -> "Settings":
        """Embedding 模型与维度必须匹配，否则写入 Milvus 时才报错就太晚了"""
        spec = EMBEDDING_MODEL_SPECS.get(self.EMBEDDING_MODEL)
        if spec is None:
            # 未知模型（例如私有部署的自定义模型）：不做维度假设，仅提示
            return self
        allowed = sorted(spec["dimensions"])
        if self.EMBEDDING_DIMENSION not in spec["dimensions"]:
            raise ValueError(
                f"EMBEDDING_DIMENSION={self.EMBEDDING_DIMENSION} 不被 {self.EMBEDDING_MODEL} 支持；"
                f"可选值: {allowed}（默认 {spec['default_dimension']}）"
            )
        if self.EMBEDDING_BATCH_SIZE > spec["max_batch"]:
            raise ValueError(
                f"EMBEDDING_BATCH_SIZE={self.EMBEDDING_BATCH_SIZE} 超过 {self.EMBEDDING_MODEL} "
                f"单请求上限 {spec['max_batch']} 行"
            )
        return self

    @model_validator(mode="after")
    def _validate_combinations(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE，否则文本分割会原地死循环")
        if self.VECTOR_WEIGHT <= 0 and self.BM25_WEIGHT <= 0:
            raise ValueError("VECTOR_WEIGHT 与 BM25_WEIGHT 不能同时为 0")
        if self.RERANKER_DEVICE not in {"cpu", "cuda", "mps", "auto"} and not self.RERANKER_DEVICE.startswith("cuda:"):
            raise ValueError("RERANKER_DEVICE 仅支持 cpu / cuda / cuda:N / mps / auto")
        if (self.RERANK_PROVIDER or "").strip().lower() not in {"dashscope", "local"}:
            raise ValueError("RERANK_PROVIDER 仅支持 dashscope / local")
        if (self.WEB_SEARCH_PROVIDER or "").strip().lower() not in {
            "deepseek",
            "so",
            "bing",
            "duckduckgo",
            "tavily",
        }:
            raise ValueError("WEB_SEARCH_PROVIDER 仅支持 deepseek / so / bing / duckduckgo / tavily")
        if self.WEB_SEARCH_ENABLED and self.WEB_SEARCH_PROVIDER.strip().lower() == "tavily" and not self.WEB_SEARCH_API_KEY:
            raise ValueError("WEB_SEARCH_PROVIDER=tavily 需要配置 WEB_SEARCH_API_KEY")
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

# huggingface_hub 在 import 时读取 HF_ENDPOINT，必须在首次导入前写入环境变量；
# 用 setdefault 尊重用户已显式设置的系统环境变量
if settings.HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)
