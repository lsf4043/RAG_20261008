# RAG 智能问答系统（LangGraph 版）
#
# 说明：
# - 依赖统一从 requirements.txt 安装（与 pyproject.toml 保持一致）；
# - 以非 root 用户运行；
# - 健康检查挂 /api/v1/health/ready（真正 ping Milvus，不是"进程还活着"）；
# - 单 worker 运行：BM25 索引与会话都在进程内，多 worker 需要换成外部存储（见 README）。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    LOG_TO_FILE=false

WORKDIR /app

# 系统依赖：PyMuPDF / pdfplumber 需要 libgl、libglib；unstructured 需要 libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，充分利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码（.dockerignore 已排除 .venv / logs / tests 等）
COPY app ./app
COPY pyproject.toml ./

# 非 root 运行
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready', timeout=5).status == 200 else sys.exit(1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
