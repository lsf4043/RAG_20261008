# RAG 智能问答系统（LangGraph 版）

基于 **LangChain + LangGraph** 的 Agentic RAG（检索增强生成）系统，在原有 RAG 项目基础上升级而来，实现了从 **Advanced RAG** 到 **Modular/Agentic RAG** 的架构跃迁。

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步 API 服务 |
| 编排框架 | LangGraph 1.1.10 | 状态机编排 RAG 流程 |
| 组件库 | LangChain 1.2.15 | 标准 Embedding/LLM/Retriever/Reranker 接口 |
| 向量库 | Milvus（langchain-milvus） | HNSW 索引向量检索 |
| LLM / Embedding | 阿里百炼（DashScope） | qwen-turbo / text-embedding-v2 |
| 重排序 | BAAI/bge-reranker-base | Cross-Encoder 重排序 |
| 可观测性 | LangSmith | LLM 调用全链路追踪 |
| 评估 | RAGAS | 检索/生成质量量化评估 |
| 文档解析 | PaddleOCR + pdfplumber + PyMuPDF | 多格式文档解析 |

## 核心特性（相对旧版升级）

### ✅ P0 优化（已实现）
- **LangGraph 状态机**：RAG 流程由线性硬编码重构为可编排的状态机
- **LangChain 组件化**：Embedding/LLM/VectorStore/BM25/Reranker 全部迁移到 LangChain 标准接口
- **RAGAS 评估**：`tests/evaluation/` 提供自动化质量评估
- **LangSmith 可观测性**：环境变量自动追踪（`LANGCHAIN_TRACING_V2=true`）

### ✅ P1 优化（已实现）
- **Query 改写/扩展**：多轮指代消解 + Multi-Query 扩展 + HyDE
- **Corrective RAG（CRAG）**：检索质量评估 + 自动重检索
- **缓存机制**：Embedding 缓存 + 查询结果缓存

## 架构流程（LangGraph 状态机）

```
START
  → query_rewrite   (Query 改写/扩展)
  → retrieve        (混合检索：向量 + BM25 + RRF 融合)
  → grade           (CRAG 文档相关性评分)
  → ┌─ 全部无关且未达上限 → increment_iteration → 回到 query_rewrite
    └─ 相关 → rerank (Cross-Encoder 重排序)
  → generate        (生成答案 + 引用来源)
  → END
```

## 项目结构

```
app/
├── main.py                   # FastAPI 入口
├── config.py                 # 配置管理（pydantic-settings）
├── api/v1/                   # API 接口层
├── core/                     # LangChain 组件层
│   ├── embeddings.py         # DashScopeEmbeddings 封装
│   ├── llm.py                # ChatDashScope 封装
│   ├── vector_store.py       # Milvus VectorStore 封装
│   ├── bm25_retriever.py     # BM25 检索器（LangChain）
│   ├── reranker.py           # Cross-Encoder 重排序
│   ├── query_processor.py    # Query 改写/扩展/HyDE（新增）
│   ├── cache.py              # 缓存（新增）
│   ├── langsmith.py          # LangSmith 可观测性（新增）
│   └── ...
├── graph/                    # LangGraph 状态机（新增）
│   ├── state.py              # RAGState 定义
│   ├── nodes.py              # 节点函数
│   └── graph.py              # 状态机组装
├── services/                 # 业务服务层
│   ├── chat_service.py       # 问答服务（调用 LangGraph）
│   ├── document_service.py   # 文档处理服务
│   └── session_service.py    # 会话管理
├── parsers/                  # 文档解析层（保留）
├── models/                   # 数据模型
└── utils/                    # 工具（切分、日志）
tests/
└── evaluation/               # RAGAS 评估（新增）
    ├── eval_dataset.json
    └── run_ragas_eval.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或安装 RAGAS 评估依赖
pip install -e ".[dev,evaluation]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY 等配置
```

### 3. 启动服务

```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. 访问

- API 文档：http://localhost:8000/docs
- 前端页面：http://localhost:8000/
- 健康检查：http://localhost:8000/api/v1/health

## 运行评估

```bash
python -m tests.evaluation.run_ragas_eval
```

评估指标：
- **faithfulness**（忠实度）：答案是否基于检索内容
- **answer_relevancy**（答案相关性）：答案是否回答了对的问题
- **context_precision**（上下文精确率）：检索内容是否相关
- **context_recall**（上下文召回率）：需要的检索内容是否都检索到

## 启用 LangSmith 追踪

在 `.env` 中配置：

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxx
LANGCHAIN_PROJECT=rag-langgraph-v2
```

启动后，LangChain/LangGraph 的所有调用会自动上报到 LangSmith 控制台。

## 主要 API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/documents/upload` | POST | 上传文档 |
| `/api/v1/documents/list` | GET | 文档列表 |
| `/api/v1/documents/{id}` | DELETE | 删除文档 |
| `/api/v1/chat/query` | POST | 智能问答 |
| `/api/v1/chat/stream` | POST | 流式问答（SSE） |
| `/api/v1/health` | GET | 健康检查 |