# RAG 智能问答系统（LangGraph 版）v2.1

基于 **LangChain + LangGraph** 的 Agentic RAG 系统：混合检索（向量 + BM25 + 加权 RRF）、
Query 改写/扩展、CRAG 批量相关性评估与重检索、Cross-Encoder 重排、引用溯源、可选拒答策略。

> v2.1 是一次**以「可运行、可验证、不骗人」为目标的修复版本**。v2.0 的架构方向正确，
> 但存在多处“代码存在但行为不成立”的问题（服务起不来、删除不生效、重检索空转、
> 评估跑不通）。下面第 2 节逐条列出改了什么、为什么改。

---

## 1. 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 编排 | LangGraph 1.1.x（检索子图 + 生成节点） |
| 组件库 | LangChain 1.2.x（ChatModel / Embeddings / Retriever / CrossEncoder） |
| 向量库 | Milvus（langchain-milvus 0.4.x / pymilvus 3.x，HNSW + COSINE） |
| LLM | DeepSeek（OpenAI 兼容接口） |
| Embedding | 阿里百炼 DashScope `text-embedding-v4`（可配 2048/1536/1024… + 查询指令） |
| 重排 | 阿里百炼 `qwen3.7-text-rerank`（商业 API，复用 `DASHSCOPE_API_KEY`；异常自动降级 `BAAI/bge-reranker-base` 本地 CrossEncoder） |
| 可观测 | LangSmith（环境变量自动追踪 + 业务级 `@traceable`） |
| 评估 | RAGAS（含消融对比） |
| 解析 | PyMuPDF + pdfplumber + PaddleOCR + python-docx + unstructured |

---

## 2. v2.1 修复清单

### 2.1 阻断级（不修就跑不起来 / 数据悄悄错）

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 服务启动即崩 | `lifespan` 缺 `app` 参数，Starlette 调用 `lifespan_context(app)` 抛 `TypeError` | 签名改为 `async def lifespan(app: FastAPI)`，并把日志配置移进 lifespan |
| 2 | 上传后元数据丢失 | langchain-milvus 默认 `enable_dynamic_field=False`，集合 schema 由**第一个文档**的 metadata 键固化，后续新键被静默丢弃；metadata 里的 `content` 与 text_field 同名还会触发 `Metadata key is reserved` | 显式 `enable_dynamic_field=True`，metadata 键改名 `raw_content`，并显式传主键 id |
| 3 | 删除文档不生效 | 删除表达式写成 `metadata["document_id"] == ...`，但集合里没有名为 `metadata` 的 JSON 字段 | 改为顶层字段 `document_id == "..."`，并对 id 做白名单校验（防表达式注入）；删除失败不再返回成功 |
| 4 | 重启后混合检索静默退化 | BM25 索引只存在进程内存里，进程重启即空，且代码只判断 `count > 0` 不报错 | 新增 `bootstrap_bm25()`：启动时从 Milvus 全量读回 chunk 重建索引；索引不可用时在响应里标 `degraded` |
| 5 | 知识库更新后答案仍旧 | 查询缓存 key 只有 `question + top_k`，上传/删除后从不失效 | key 增加会话历史摘要；上传/删除/批量重建后调用 `invalidate_query_cache()` |

### 2.2 正确性

| # | 问题 | 修复 |
|---|------|------|
| 6 | **CRAG 重检索是空转**：`query_rewrite_node` 输入不变，重试必然拿到同一批 query | 重试时回灌“上一轮失败的 query”作为提示、逐轮放宽 `candidate_k`；无任何改写能力时直接跳过重试 |
| 7 | CRAG 每篇文档一次 LLM 调用（20~60 次/问题） | 改为**一次调用批量判定**（`_GradeBatch` 结构化输出），单问题 LLM 往返从 60+ 降到 3~4 次 |
| 8 | 跨页 `chunk_id` 重号 → 不同内容被 RRF 合并成一类 | `split_text(..., start_chunk_id=...)` 全局连续编号；去重 key 加入 `page_num`；`hash()` 换成 sha1 |
| 9 | 节点就地修改共享 Document（BM25 索引里的对象），并发会串分数 | 所有分数写入 Document **副本**，节点只返回局部状态更新 |
| 10 | 检索不到内容时仍让 LLM 自由作答，用户以为答案来自知识库 | 新增 `ALLOW_LLM_FALLBACK`（默认 false）→ 直接拒答；API 返回 `used_fallback / degraded / notes`，前端显式提示 |
| 11 | `VECTOR_WEIGHT` / `BM25_WEIGHT` / `MAX_RETRIES` / 接口的 `top_k`、`temperature` 全是“假旋钮” | RRF 支持通道加权；重试接入 Milvus 操作；请求参数写入 state 并在生成阶段生效 |
| 12 | 章节识别首行标题被丢弃（`pos > last_pos + 10` 且 `last_pos=-1`） | 修正边界判断（这是原代码里就存在的 bug，被新增单测抓出来） |
| 13 | 健康检查永远返回 healthy | `/health` 如实计算状态；新增 `/health/ready`（真正 ping Milvus + 校验 schema，未就绪返回 503） |
| 14 | 流式问答手写了一遍检索链，与状态机双份实现 | 拆出**检索子图**，非流式（完整图）与流式（检索子图 + token 流）共用同一份检索逻辑 |

### 2.3 性能与成本

| # | 问题 | 修复 |
|---|------|------|
| 15 | `BM25Retriever.from_documents` 在每次 `search()` 里重建索引（3 个 query = 3 次全量构建） | 只在 `rebuild()` 时构建一次，检索时 `model_copy(update={"k": ...})` 复用倒排结构 |
| 16 | 每次调温就新建一个 OpenAI client（评分场景每篇文档一个） | 按 `(temperature, max_tokens)` 复用 `model_copy` 变体，共享连接池 |
| 17 | 改写 + Multi-Query 两次串行 LLM 往返 | 合并为一次结构化调用（`QueryPlan`），失败再退化到纯文本解析 |
| 18 | 复杂 PDF“有图就整页 2× 渲染 + OCR”，且与已提取文本重复入库 | 仅当页面可提取文本低于 `OCR_TEXT_THRESHOLD` 才 OCR，并剔除重复行；表格用 bbox 从正文中裁掉，不会一份内容出现三次 |
| 19 | 无任何超时，上游卡死会占满线程池 | `thread_pool.run_sync(..., timeout=)`、请求级 `REQUEST_TIMEOUT`、LLM `timeout/max_retries` |

### 2.4 工程 / 部署 / 安全

- **HTTP 语义**：所有错误走异常处理器，HTTP 状态码与 `code` 字段一致（旧版一律 200）；
  500 不再把内部异常文本回显给客户端。
- **参数校验**：请求模型补齐 `min_length/ge/le`，并注册 `RequestValidationError` 处理器（422）。
- **响应模型**：全部接口有 `response_model`，`/docs` 出现真实 schema。
- **上传**：边读边校验大小（不再整包读进内存）；扩展名白名单；`overlap >= chunk_size` 直接拒绝。
- **鉴权**：新增可选 `API_AUTH_TOKEN`（写操作校验 `X-API-Token`，WS 用 `?token=`）。
- **CORS**：`*` 与 `credentials` 不再同时开启（浏览器规范要求）。
- **WebSocket**：进度推送改为「队列 + `call_soon_threadsafe`」，不再从工作线程裸调 `create_task`；
  删掉只能提示“请用 HTTP 上传”的空壳 WS 接口。
- **依赖**：`pyproject.toml` 与 `requirements.txt` 对齐（补 `python-multipart`、`websockets`、`python-dotenv` 等；
  RAGAS 移出运行依赖；`paddleocr` 锁 `<3`）；解析器全部**延迟导入**，没装 OCR 也能启动。
- **PaddleOCR 3.x 兼容层**：`app/parsers/ocr_engine.py` 统一 2.x/3.x 的构造参数与返回结构差异。
- **日志**：`RotatingFileHandler` + 目录不可写时降级为仅控制台（不再因日志目录权限导致服务起不来）。
- **Docker**：新增 `.dockerignore`（避免把 `.venv` 打进构建上下文）、非 root 运行、健康检查改用 `/health/ready`、
  compose 补 `DEEPSEEK_*` / `LANGSMITH_*` / `host.docker.internal`、模型缓存卷。
- **部署文档**：补 `.env.example`（README 一直在让你 `cp .env.example .env`，但文件不存在）。

### 2.5 测试与评估

- 新增 **62 个单元测试**（分割器、去重 key、加权 RRF、缓存失效、CRAG 决策/批量评分、配置校验、
  Embedding v4 维度/批处理/指令、LLM 自死锁回归、API 状态码与安全）。
- 新增 **7 个 Milvus 集成测试**（真实读写，含完整 E2E：上传 → 检索 → 问答 → 删除 → 缓存失效，
  以及“重启后元数据不丢”的回归）。
  `pytest` 默认跳过 `integration`，`pytest -m integration` 单独运行。
- RAGAS 脚本重写：接上 DeepSeek + DashScope（旧版默认要 `OPENAI_API_KEY`，从未跑通）、
  用 `to_pandas()` 取指标（旧版读 `_scores` 得到空报告）、支持 `--ingest` 灌语料、
  支持 `--variants` 跑**消融对比**（纯向量 / 混合 / +重排 / 全量）。

### 2.6 Embedding 升级到 text-embedding-v4

| 项目 | 说明 |
|---|---|
| 维度 | v4 支持 2048 / 1536 / 1024（默认） / 768 / 512 / 256 / 128 / 64；本项目用 **1536**（与 Milvus 集合一致） |
| 上下文 | 单行 8192 token（v2 只有 2048，长 chunk 会被静默截断） |
| 批大小 | v3/v4 单请求上限 **10 行**（v1/v2 为 25），客户端自动分批 |
| 查询指令 | v3/v4 支持 `instruct`，查询侧附带任务描述可提升召回（`EMBEDDING_QUERY_INSTRUCT`） |
| 价格 | v4 更低（0.0005 元/千 token vs v2 的 0.0007） |

**为什么不再用 `langchain_community.DashScopeEmbeddings`**：该类
`model_config = ConfigDict(extra="forbid")`，只向 SDK 传 `input` / `text_type` / `model`，
无法传 `dimension` / `output_type` / `instruct`。而 v4 默认输出 **1024 维**，
只改模型名会导致向量维度与 `EMBEDDING_DIMENSION` / Milvus 集合不一致，写入直接失败。
因此 `app/core/embeddings.py` 改为直接调用 `dashscope.TextEmbedding.call`（保留缓存、
批处理、重试与维度校验），并新增 `EMBEDDING_MODEL_SPECS` 做“模型 ↔ 维度”启动校验。

**迁移规则（重要）**：

1. `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` / Milvus 集合维度三者必须一致，否则启动即报错；
2. 换模型 = 换向量空间，**旧向量不可复用**：需新建集合（或 `MILVUS_DROP_OLD=true`）并重新入库；
3. 回退到 v2 时，2048/1536 之外的维度会被配置校验拒绝。

### 2.7 两个运行期缺陷（真实链路测试暴露）

- **LLM 自死锁**：`LLMService._variant()` 先持有 `threading.Lock` 再调 `self.base`，
  而 `base` 内部又要拿同一把锁 → 首次调用走 `structured()`/指定 `temperature` 时永久卡住，
  表现为“问答请求 120s 超时且没有任何节点日志”。已改为先解析 `base` 再进临界区，并用 `RLock` 兜底
  （`tests/unit/test_llm.py` 有专门的死锁回归测试）。
- **动态字段被“字段裁剪”误杀**：集合重启后 schema 已知（只有 id/content/embedding），
  旧逻辑按 schema 过滤 metadata，导致 `document_id` 等动态字段被丢弃 →
  `documents/list` 返回空、来源显示“未知文档”、BM25 与向量结果无法去重。
  已改为“动态字段模式不过滤”，并加集成回归测试。

### 2.8 意图路由（Adaptive RAG）

**问题**：以前任何输入（包括“你好”“谢谢”）都会走完整 RAG 链路：Query 改写 → 混合检索 →
CRAG 评分 → 生成。知识库为空或问的是闲聊时，还会直接拒答——既慢，体验也差。

**现在**：状态机入口先做一次意图判定，**先便宜后昂贵**：

| 判定层 | 输入示例 | 结果 | 额外开销 |
|---|---|---|---|
| 规则：寒暄/闲聊 | 你好、谢谢、再见、你是谁、在吗 | 直接作答 | 0 |
| 规则：通用任务 | 帮我写祝福语、翻译成英文、总结这段话、讲个笑话 | 直接作答 | 0 |
| 规则：知识特征 | 什么/怎么/为什么/配置/报错…，或含问号 | 走 RAG | 0 |
| 规则：多轮指代 | 有历史时的“它呢”“那个怎么配” | 走 RAG | 0 |
| 规则：长句 | 长度 ≥ `ROUTER_LONG_TEXT_THRESHOLD` | 走 RAG | 0 |
| LLM 兜底 | 极短且模糊，如“帮我看看” | A/B 单字母分类（`max_tokens=4`） | 1 次极小调用 |

- 判定失败/无法解析 → **保守回退 RAG**（把知识问题误判成闲聊会产生幻觉，代价更大）。
- 直答走独立的 `generate_direct` 节点，系统提示明确写“本轮未检索知识库，不要编造”，
  响应里返回 `route="direct"`、`used_fallback=true`、`sources=[]`，前端显示“💬 通用回答”。
- 同时修掉一个空转：**本轮一条候选都没召回时不再触发 CRAG 重检索**
  （空知识库下能省 8s+，`decide_after_grade` 里 2 行守卫 + 单测）。

实测（真实 DeepSeek + v4 Embedding）：

| 输入 | 路由 | 耗时 | 结果 |
|---|---|---|---|
| 你好 | direct | **0.77s** | “你好！有什么可以帮你的吗？” |
| 谢谢你！ | direct | 0.65s | 自然回复 |
| 翻译成英文：今天天气不错 | direct | 0.62s | The weather is nice today. |
| Milvus 常用什么索引和度量？（有文档） | rag | 4.24s | 引用【知识片段 1】，答出 HNSW + COSINE |

关闭方式：`ROUTER_ENABLED=false` 即退回“全部走 RAG”的旧行为。

---

## 3. 架构

```
START → route（意图路由）
          ├── direct → generate_direct → END          # 闲聊/通用任务：不检索，直接答
          └── rag ──┐
                    ▼
     ┌──────────────── 检索子图（非流式与流式共用）────────────────┐
     │ query_rewrite → retrieve → grade ──┬→ increment_iteration ─┤
     │                                    └→ rerank → END          │
     └─────────────────────────────────────────────────────────────┘
                    ▼
              generate → END
```

- `GET  /api/v1/health`       轻量存活检查（无外部调用，但状态真实计算）
- `GET  /api/v1/health/ready` 就绪检查（ping Milvus + schema 校验，未就绪 503）
- 流式接口复用**同一套路由 + 检索子图**，生成阶段走 LLM token 流，**不重复实现检索逻辑**

### 目录结构

```
app/
├── main.py                 # FastAPI 入口（lifespan / CORS / 异常处理器 / 启动自愈）
├── config.py               # pydantic-settings 配置 + 组合校验
├── api/
│   ├── deps.py             # 可选令牌鉴权
│   └── v1/{chat,documents,health,router}.py
├── core/
│   ├── llm.py              # DeepSeek 客户端（复用 + 结构化输出）
│   ├── embeddings.py       # DashScope Embedding（懒加载 + 缓存）
│   ├── vector_store.py     # Milvus（动态字段 / 真删除 / 全量遍历 / schema 兼容检查）
│   ├── bm25_retriever.py   # BM25（构建一次，复用）
│   ├── reranker.py         # qwen3.7-text-rerank 优先 + 本地 Cross-Encoder 兜底
│   ├── query_processor.py  # 一次调用完成改写 + 多查询扩展
│   ├── documents.py        # Document 副本 / 去重 key / 正文取值
│   ├── cache.py            # Embedding LRU + 查询 TTL（可失效）
│   ├── exceptions.py       # 业务异常 + 全局处理器
│   └── thread_pool.py      # 带超时的线程池
├── graph/                  # state / nodes / graph（检索子图 + 完整图）
├── services/               # chat / document / session
├── parsers/                # pdf / ocr(2.x+3.x 兼容) / word(含表格) / unstructured(含页码)
├── models/                 # 请求/响应模型（带校验）
└── utils/                  # 分割器（chunk_id 全局唯一）/ 日志
tests/
├── unit/                   # 44 个单测
├── api/                    # 接口层测试
├── integration/            # 真实 Milvus 集成测试（含 E2E）
└── evaluation/             # RAGAS 消融评估
```

---

## 4. 快速开始

```bash
# 1. 依赖（Python >= 3.11）
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt                     # 含文档解析能力
# 或者只要问答能力：pip install -e ".[dev]"

# 2. 配置
cp .env.example .env    # 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / MILVUS_*
# 若仓库未包含 .env.example，请按第 5 节配置表自行创建 .env

# 3. 启动
python -m app.main
# 或：uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 前端页面：http://localhost:8000/
- 接口文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/v1/health ｜ 就绪：`/api/v1/health/ready`

### Milvus 集合说明（重要）

- 应用会在首次写入时创建集合，字段：`id`(VARCHAR 主键)、`content`(参与向量化的文本)、
  `embedding`(FLOAT_VECTOR)、以及 metadata 动态字段（`raw_content / title / document_id /
  document_name / file_type / page_num / chunk_id`）。
- 若 `MILVUS_COLLECTION` 指向了**其它项目遗留的集合**，启动时会打印明确的 `schema 不兼容` 错误，
  `/api/v1/documents/list` 仍可读，但上传会被拒绝并提示修复方式。推荐：给本项目一个专用集合名。
- `MILVUS_DROP_OLD=true` 会在启动时删除同名集合重建（**会丢数据**，仅在确认可弃时使用）。

---

## 5. 关键配置

| 变量 | 默认 | 说明 |
|------|------|------|
| `EMBEDDING_MODEL` | `text-embedding-v4` | 向量模型；切换需同步改维度并重建集合 |
| `EMBEDDING_DIMENSION` | `1536` | 必须落在模型允许集合内（v4: 2048/1536/1024/768/512/256/128/64） |
| `EMBEDDING_BATCH_SIZE` | `10` | 单请求行数，不得超过模型上限（v4=10，v2=25） |
| `EMBEDDING_QUERY_INSTRUCT` | 一句检索指令 | 查询侧 `instruct`（仅 v3/v4；留空则不发） |
| `ROUTER_ENABLED` | `true` | 意图路由总开关；false = 所有请求都走知识库检索（旧行为） |
| `ROUTER_LLM_ENABLED` | `true` | 仅对规则判不了的“模糊短句”允许一次极小 LLM 分类 |
| `ROUTER_LONG_TEXT_THRESHOLD` | `12` | 超过该长度直接按“实质性提问”走检索 |
| `ALLOW_LLM_FALLBACK` | `false` | 仅约束 RAG 路径：检索不到内容时是否允许脱离知识库作答 |
| `LLM_STRUCTURED_DISABLE_THINKING` | `true` | 结构化调用（Query 规划 / CRAG 评分）关闭 DeepSeek 思考模式：思考模式不支持强制 `tool_choice`，且会忽略 `temperature` |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` | `1.0` / `1.0` | 加权 RRF 的通道权重 |
| `RETRIEVAL_CANDIDATE_K` | `20` | 召回候选数；CRAG 重试时自动放宽 |
| `CRAG_MAX_ITERATIONS` | `2` | 最大重检索轮数 |
| `CRAG_MAX_GRADE_DOCS` | `20` | 单次批量评分的文档数上限 |
| `CRAG_MIN_RELEVANT` | `1` | 至少命中几篇才算“相关” |
| `RERANK_PROVIDER` | `dashscope` | `dashscope` = 走 `qwen3.7-text-rerank`（异常自动降级本地）；`local` = 只用本地模型 |
| `RERANK_API_MODEL` | `qwen3.7-text-rerank` | 商业重排模型（复用 `DASHSCOPE_API_KEY`） |
| `RERANK_API_TIMEOUT` / `RERANK_API_COOLDOWN` | `10` / `60` | API 超时与失败后的熔断冷却（秒，冷却期内直接走本地） |
| `RERANK_API_MAX_CHARS` / `RERANK_API_MAX_DOCS` | `4000` / `50` | 单条文档截断字符数 / 单次请求最多送排文档数 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | 本地兜底 CrossEncoder（API 异常时使用） |
| `RERANKER_DEVICE` | `cpu` | `cpu` / `cuda` / `cuda:0` / `mps` / `auto` |
| `RERANK_MIN_SCORE` | `0` | 低于该分数直接丢弃（>0 才生效；API 分数范围 0~1） |
| `QUERY_REWRITE_ENABLED` / `MULTI_QUERY_ENABLED` | `true` | 关闭后检索 query 不再改写，CRAG 也不会重试 |
| `CONTEXT_MAX_CHARS` | `12000` | 拼进 prompt 的上下文字符上限 |
| `REQUEST_TIMEOUT` | `120` | 单次问答总超时（秒） |
| `API_AUTH_TOKEN` | 空 | 非空时写操作需要 `X-API-Token` |
| `CORS_ORIGINS` | `*` | 逗号分隔；为 `*` 时自动关闭 credentials |
| `OCR_TEXT_THRESHOLD` | `200` | 页面可提取文本低于该长度才整页 OCR |

完整列表见 `.env.example`。

---

## 6. API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传单个文档（multipart） |
| POST | `/api/v1/documents/upload/batch` | 批量上传（最后统一重建一次索引） |
| GET | `/api/v1/documents/list` | 文档列表（按 document_id 聚合） |
| DELETE | `/api/v1/documents/{document_id}` | 删除文档（向量库 + 内存索引 + 缓存） |
| POST | `/api/v1/chat/query` | 问答 |
| POST | `/api/v1/chat/stream` | 流式问答（SSE） |
| WS | `/api/v1/chat/query/logs` | 处理进度推送（可选） |
| POST | `/api/v1/chat/session/clear` | 清空会话 |
| GET | `/api/v1/health` | 存活检查 |
| GET | `/api/v1/health/ready` | 就绪检查（503 表示未就绪） |
| GET | `/api/v1/stats` | 缓存命中率 / BM25 文档数 / 会话数 |

响应统一为 `{"code", "message", "error", "data"}`，**HTTP 状态码与 `code` 一致**。

问答响应里额外给出：

```json
{
  "route": "rag",           // rag = 走了知识库检索；direct = 通用直答（未检索知识库）
  "used_fallback": false,   // true = 本次回答没有知识库依据（direct 时恒为 true）
  "degraded": false,        // true = 发生了降级（BM25 空 / 评分失败等）
  "notes": [],              // 降级原因与策略说明（含意图路由理由）
  "crag_iterations": 0,
  "cached": false
}
```

---

## 7. 测试

```bash
pip install -e ".[dev]"

pytest                     # 单元 + 接口测试（默认跳过 integration）
pytest -m integration      # 需要可用的 Milvus，会真实建/删测试集合
```

- 单元测试不访问网络：`conftest.py` 用假 Key，任何真实外部调用都会立即失败。
- 集成测试使用**确定性假 Embedding + 假 LLM**，只依赖真实 Milvus，不消耗模型配额。

---

## 8. 评估（RAGAS 消融）

```bash
pip install -e ".[evaluation]"

# 先把语料灌进知识库，再跑全部变体的对比
python -m tests.evaluation.run_ragas_eval --ingest --corpus ./docs

# 只跑某一个变体
python -m tests.evaluation.run_ragas_eval --variants hybrid_rerank
```

输出对比表（faithfulness / answer_relevancy / context_precision / context_recall）并写入
`tests/evaluation/eval_report.json`。变体：`vector_only` / `hybrid` / `hybrid_rerank` / `full`。

> 只有这张表能证明“混合检索/重排/CRAG 到底有没有用”。建议在 README 里贴一次结果。

---

## 9. 部署

```bash
cp .env.example .env      # 填好密钥（无模板时按第 5 节配置表创建）
docker compose -f docker/docker-compose.yml up -d --build

# 需要本地起 Milvus 时
docker compose -f docker/docker-compose.yml --profile milvus up -d
```

**单 worker 运行**：BM25 索引与会话都存在进程内。多 worker / 多副本需要：
- BM25 → 换成 Milvus 的 `BM25BuiltInFunction` 稀疏检索，或独立 ES；
- 会话 → Redis；
- 查询缓存 → Redis（且按知识库版本失效）。

---

## 10. 已知限制（有意保留，写在这里而不是藏着）

1. BM25 索引在进程内存中：启动时重建（可接受），但多副本会各建一份。
2. 会话存储为进程内 dict：重启即丢，`SESSION_MAX_COUNT` 防内存无限增长。
3. 文档解析不支持扫描版 PDF 的版面还原（表格是 Markdown 近似）。
4. `.doc` 老格式依赖 antiword / textract / Word COM 之一，缺失时给出明确提示。
5. 本地重排模型首次启动会下载（约 300MB）；它现在是 `qwen3.7-text-rerank` 的兜底通道，建议挂载 HF 缓存目录（`RERANK_PROVIDER=local` 时完全不走 API）。
6. 没有内置限流/配额：`API_AUTH_TOKEN` 只是最简鉴权，公网部署请放在网关后面。

## 11. 排错

| 现象 | 原因与处理 |
|------|-----------|
| 启动报 `TypeError: lifespan() takes 0 positional arguments` | v2.0 的 bug，v2.1 已修复；确认拉到的代码是 2.1.0 |
| 上传返回 `Embedding 调用失败: 401 InvalidApiKey` | `DASHSCOPE_API_KEY` 无效或过期，去百炼控制台重新生成 |
| 上传返回 `schema 不兼容` | `MILVUS_COLLECTION` 指向了其它项目的集合，换一个专用集合名（或 `MILVUS_DROP_OLD=true` 重建） |
| `/health/ready` 返回 503 | 看响应里的 `milvus.status / schema / llm_configured`；`row_count` 需要 flush 才准确，正常现象 |
| 问答一直 `used_fallback=true` | 知识库为空或检索不到；先 `/documents/list` 确认数据，再看 `notes` |
| 说“你好”也被拒答 / 闲聊很慢 | 检查 `ROUTER_ENABLED` 是否为 `true`；若为 false 则所有请求都会走 RAG |
| 闲聊被误判成知识问题 | 在 `app/core/intent_router.py` 的 `_CHITCHAT` / `_TASK_VERBS` 里补规则，或调大 `ROUTER_LLM_ENABLED` 让模型兜底 |
| 问答 120s 超时且日志里没有任何节点输出 | ①`LLMService._variant` 自死锁（已在 2.7 修复）；②DeepSeek 不支持 `json_schema`，`with_structured_output` 已显式改用 `function_calling`；若仍 400 看下一条 |
| 日志出现 `Thinking mode does not support this tool_choice` | `deepseek-flash` 默认开启思考模式，思考模式不允许强制指定 `tool_choice`；`LLM_STRUCTURED_DISABLE_THINKING=true`（默认）会在结构化调用中关闭思考 |
| `documents/list` 返回 0 但问答有来源 | 动态字段被字段裁剪误杀（已在 2.7 修复）；确认 `_fit_document` 不过滤动态字段 |
| 上传报 `Embedding 维度不符` | `EMBEDDING_DIMENSION` 与模型默认维度或 Milvus 集合维度不一致 |
| 日志出现 `[rerank] qwen3.7-text-rerank 调用失败，降级本地 Cross-Encoder` | 检查 `DASHSCOPE_API_KEY`、百炼限流与网络；`RERANK_API_COOLDOWN` 内自动走本地，冷却结束后自动重试 API |
| 重排结果与旧版排序差异较大 | API 的 `relevance_score` 为 0~1 的相对分数，本地 CrossEncoder 是 logits；使用 `RERANK_MIN_SCORE` 时注意两种分数量纲不同 |
| `paddleocr` 报参数错误 | v3 改了 API，已用 `app/parsers/ocr_engine.py` 兼容；若仍失败请锁 `paddleocr<3` |

## License

MIT
