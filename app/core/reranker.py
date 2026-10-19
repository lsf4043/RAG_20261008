"""
重排序服务（商业 API 优先 + 本地 Cross-Encoder 兜底）

策略：
- 默认走阿里百炼 ``qwen3.7-text-rerank``（复用 ``DASHSCOPE_API_KEY``）；
- 商业 API 的任何异常（网络 / 鉴权 / 限流 / 返回结构异常）都会降级到本地
  Cross-Encoder，并在 ``RERANK_API_COOLDOWN`` 秒内不再尝试 API，避免每次请求都白等超时；
- ``RERANK_PROVIDER=local`` 可完全关闭 API 通道；
- 本地模型仍支持后台预热；API 冷却期间若本地尚未就绪，按原始顺序返回；
- 打分写入**文档副本**的 metadata，避免污染 BM25 索引里共享的 Document 对象。
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.core.documents import copy_document, doc_text
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReRankAPIError(RuntimeError):
    """商业重排 API 调用异常（统一触发本地降级）"""


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """兼容 dashscope 返回的 dict / DictMixin / dataclass 三种形态"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except Exception:  # noqa: BLE001
            pass
    return getattr(obj, name, default)


class DashScopeReRankClient:
    """DashScope 文本重排序客户端（薄封装，便于单测替换）"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or settings.RERANK_API_MODEL
        self._api_key = api_key or settings.require_dashscope_key()

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        调用 qwen3.7-text-rerank

        Returns:
            ``(原始文档下标, relevance_score)`` 列表，数量与输入文档一致
        """
        import dashscope  # 延迟导入：未启用 API 重排时不影响服务启动

        texts = list(documents)
        try:
            response = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=texts,
                top_n=top_n,
                api_key=self._api_key,
                request_timeout=int(settings.RERANK_API_TIMEOUT),
                **({"instruct": settings.RERANK_API_INSTRUCT} if settings.RERANK_API_INSTRUCT else {}),
            )
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权异常统一转业务异常
            raise ReRankAPIError(f"重排序请求异常: {exc}") from exc

        status = _field(response, "status_code")
        if status != 200:
            detail = f"{_field(response, 'code', '')} {_field(response, 'message', '')}".strip()
            raise ReRankAPIError(f"DashScope 重排序失败（HTTP {status}）：{detail}")

        results = _field(_field(response, "output"), "results")
        if not isinstance(results, list) or not results:
            raise ReRankAPIError("DashScope 重排序返回结构异常：缺少 results")

        scored: List[Tuple[int, float]] = []
        seen = set()
        for item in results:
            index = _field(item, "index")
            score = _field(item, "relevance_score")
            try:
                index = int(index)
                score = float(score)
            except (TypeError, ValueError) as exc:
                raise ReRankAPIError(f"DashScope 重排序返回字段非法: {item!r}") from exc
            if not 0 <= index < len(texts):
                raise ReRankAPIError(f"DashScope 重排序返回越界索引: {index}")
            if index in seen:
                raise ReRankAPIError(f"DashScope 重排序返回重复索引: {index}")
            if not math.isfinite(score):
                raise ReRankAPIError("DashScope 重排序返回 NaN/Inf 分数")
            seen.add(index)
            scored.append((index, score))

        if len(scored) != len(texts):
            raise ReRankAPIError(
                f"DashScope 重排序返回数量不一致：期望 {len(texts)}，实际 {len(scored)}"
            )
        return scored


class RerankerService:
    """重排序服务：DashScope API 优先，异常自动降级本地 Cross-Encoder"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_client: Optional[DashScopeReRankClient] = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self._api_client = api_client
        self._api_client_lock = threading.Lock()
        # API 熔断截止时间（monotonic）：写入是原子赋值，读取无需加锁
        self._api_failed_until = 0.0
        self._cross_encoder: Any = None
        self._initialized = False
        self._available = False
        # 后台加载进行中（用于请求侧非阻塞降级，避免死等初始化锁）
        self._initializing = False
        self._lock = threading.Lock()

    # ---------- 通道状态 ----------
    @property
    def provider(self) -> str:
        return (settings.RERANK_PROVIDER or "local").strip().lower()

    @property
    def active_model(self) -> str:
        """当前主通道使用的模型名（健康检查展示用）"""
        return settings.RERANK_API_MODEL if self.provider == "dashscope" else self.model_name

    def _api_enabled(self) -> bool:
        return (
            self.provider == "dashscope"
            and bool(settings.DASHSCOPE_API_KEY)
            and bool(settings.RERANK_API_MODEL)
        )

    @property
    def api_client(self) -> DashScopeReRankClient:
        if self._api_client is None:
            with self._api_client_lock:
                if self._api_client is None:
                    self._api_client = DashScopeReRankClient()
                    logger.info(
                        "重排序 API 客户端初始化完成，模型: %s, 超时: %ss",
                        self._api_client.model,
                        settings.RERANK_API_TIMEOUT,
                    )
        return self._api_client

    # ---------- 初始化 ----------
    def initialize(self) -> bool:
        with self._lock:
            if self._initialized:
                return self._available
            self._initializing = True
            try:
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder

                model_kwargs = {}
                device = (settings.RERANKER_DEVICE or "cpu").strip()
                if device and device != "auto":
                    model_kwargs["device"] = device

                logger.info("正在加载本地重排序模型: %s (device=%s)", self.model_name, device)
                self._cross_encoder = HuggingFaceCrossEncoder(
                    model_name=self.model_name,
                    model_kwargs=model_kwargs,
                )
                self._available = True
                logger.info("本地重排序模型加载成功")
            except Exception as exc:  # noqa: BLE001
                logger.error("本地重排序模型加载失败: %s，降级为原始检索顺序", exc)
                self._cross_encoder = None
                self._available = False
            finally:
                self._initializing = False
                self._initialized = True
            return self._available

    def warm_up_async(self) -> bool:
        """
        在后台守护线程中加载本地模型：立即返回，不阻塞启动与进程退出

        用 daemon 线程而不是线程池，避免退出时被
        ``ThreadPoolExecutor.shutdown(wait=True)`` 卡在下载上。

        Returns:
            True 表示已启动后台加载；False 表示已在加载中或已初始化
        """
        with self._lock:
            if self._initialized or self._initializing:
                return False
            # 先在锁内置位，堵住“请求在后台线程拿到锁之前进来”的竞态窗口
            self._initializing = True

        def _run() -> None:
            try:
                self.initialize()
            except Exception as exc:  # noqa: BLE001
                logger.warning("本地重排序模型后台加载异常: %s", exc)
                with self._lock:
                    self._initializing = False

        try:
            threading.Thread(target=_run, name="reranker-warmup", daemon=True).start()
        except RuntimeError as exc:
            with self._lock:
                self._initializing = False
            logger.warning("本地重排序模型后台线程启动失败: %s", exc)
            return False
        return True

    def is_available(self) -> bool:
        """API 通道已配置即视为就绪；API 实际失败会自动降级本地"""
        if self._api_enabled():
            return True
        return self._available

    def is_local_available(self) -> bool:
        return self._available

    def is_initializing(self) -> bool:
        return self._initializing

    def is_initialized(self) -> bool:
        return self._initialized

    # ---------- 重排 ----------
    def rerank(
        self,
        documents: List[Document],
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return []

        limit = top_n or settings.RERANK_TOP_K

        if self._api_enabled() and time.monotonic() >= self._api_failed_until:
            try:
                results = self._rerank_with_api(documents, query, limit)
                logger.info(
                    "[rerank] %s 重排完成，返回 %s 条", settings.RERANK_API_MODEL, len(results)
                )
                return results
            except Exception as exc:  # noqa: BLE001 - 任何异常都必须降级，不能影响问答
                cooldown = max(float(settings.RERANK_API_COOLDOWN or 0.0), 0.0)
                self._api_failed_until = time.monotonic() + cooldown
                logger.warning(
                    "[rerank] %s 调用失败，降级本地 Cross-Encoder（%.0fs 内不再尝试 API）: %s",
                    settings.RERANK_API_MODEL,
                    cooldown,
                    exc,
                )

        return self._rerank_local(documents, query, limit)

    def _rerank_with_api(
        self,
        documents: List[Document],
        query: str,
        limit: int,
    ) -> List[Document]:
        max_docs = max(int(settings.RERANK_API_MAX_DOCS), 1)
        candidates = documents[: min(len(documents), max_docs)]
        if len(candidates) < len(documents):
            logger.info(
                "[rerank] 候选 %s 篇，仅前 %s 篇送入 API（RERANK_API_MAX_DOCS=%s）",
                len(documents),
                len(candidates),
                settings.RERANK_API_MAX_DOCS,
            )

        max_chars = max(int(settings.RERANK_API_MAX_CHARS), 1)
        texts = [
            doc_text(doc)[:max_chars] or doc.page_content[:max_chars] for doc in candidates
        ]
        scored_pairs = self.api_client.rerank(query, texts, top_n=len(texts))

        scored: List[Tuple[Document, float]] = []
        for index, score in scored_pairs:
            copied = copy_document(candidates[index])
            copied.metadata["rerank_score"] = float(score)
            scored.append((copied, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored[:limit]]

    def _rerank_local(
        self,
        documents: List[Document],
        query: str,
        limit: int,
    ) -> List[Document]:
        if not self._initialized:
            if self._initializing:
                # 后台预热进行中：不抢初始化锁，直接按原始顺序返回，避免请求被下载卡住
                logger.info("本地重排序模型正在后台加载，本次按原始顺序返回")
                return [copy_document(doc) for doc in documents[:limit]]
            self.initialize()

        if not self._available or self._cross_encoder is None:
            logger.warning("本地重排序模型不可用，返回原始顺序")
            return [copy_document(doc) for doc in documents[:limit]]

        try:
            pairs: List[Tuple[str, str]] = [(query, doc.page_content) for doc in documents]
            raw_scores = list(self._cross_encoder.score(pairs))
            if len(raw_scores) != len(documents):
                raise ValueError(f"重排分数数量不匹配: {len(raw_scores)} != {len(documents)}")

            scored: List[Tuple[Document, float]] = []
            for doc, score in zip(documents, raw_scores):
                value = float(score)
                if not math.isfinite(value):
                    raise ValueError("重排分数包含 NaN/Inf")
                copied = copy_document(doc)
                copied.metadata["rerank_score"] = value
                scored.append((copied, value))

            scored.sort(key=lambda item: item[1], reverse=True)
            results = [doc for doc, _ in scored[:limit]]
            logger.info("本地重排序完成，返回 %s 条结果", len(results))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("本地重排序失败，降级为原顺序: %s", exc)
            return [copy_document(doc) for doc in documents[:limit]]


_reranker_service: Optional[RerankerService] = None
_reranker_lock = threading.Lock()


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        with _reranker_lock:
            if _reranker_service is None:
                _reranker_service = RerankerService()
    return _reranker_service


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.reranker import reranker_service``"""
    if name == "reranker_service":
        return get_reranker_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
