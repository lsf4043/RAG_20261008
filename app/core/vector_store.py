"""
向量数据库服务（langchain-milvus 0.4.x / pymilvus 3.x）

本次修复的关键点：
1. **显式开启 ``enable_dynamic_field=True``**。旧实现依赖默认值 False，导致集合
   schema 由“第一个上传的文档”的 metadata 键固化，后续新增字段被静默丢弃，
   且 metadata 的 ``content`` 与 text_field 同名会触发 “Metadata key is reserved”。
   现在 metadata 全部作为动态字段落库，字段可自由演进。
2. **删除使用顶层字段表达式** ``document_id == "..."``（旧实现用 JSON 路径去匹配
   一个并不存在的 ``metadata`` 字段，删除永远不生效）。同时对 document_id 做
   白名单校验，避免表达式注入。
3. 连接使用 ``uri`` 形式（pymilvus 3.x 的 MilvusClient 只认 uri），并真正发起一次
   远端调用验证连通性，而不是“构造成功就算连上”。
4. 提供 ``iter_documents()`` 供启动时重建 BM25 索引，解决“重启后混合检索静默退化”。
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.config import settings
from app.core.embeddings import get_embedding_service
from app.core.exceptions import MilvusException
from app.utils.logger import get_logger

logger = get_logger(__name__)

TEXT_FIELD = "content"          # 参与向量化的文本（标题加权后的 chunk 文本）
VECTOR_FIELD = "embedding"
PRIMARY_FIELD = "id"
RAW_CONTENT_FIELD = "raw_content"   # 原始正文（与 TEXT_FIELD 区分开，避免字段名冲突）

_SAFE_ID = re.compile(r"^[0-9a-zA-Z_\-]{1,64}$")


def _escape_expr_value(value: str) -> str:
    """转义 Milvus 表达式中的字符串字面量"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class VectorStoreService:
    """Milvus 向量库服务"""

    def __init__(self) -> None:
        self.collection_name = settings.MILVUS_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION
        self._store: Optional[Milvus] = None
        self._connected = False
        self._lock = threading.RLock()
        # 已存在集合的实际字段（用于兼容旧集合 / 过滤未知 metadata）
        self._existing_fields: Optional[set] = None
        # 与当前应用 schema 不兼容的原因（None 表示兼容或集合尚不存在）
        self._incompatible_reason: Optional[str] = None

    # ==================== 连接 ====================
    def connect(self) -> bool:
        with self._lock:
            if self._connected and self._store is not None:
                return True
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            try:
                logger.info("正在连接 Milvus: %s (collection=%s)", uri, self.collection_name)
                store = Milvus(
                    embedding_function=get_embedding_service(),
                    collection_name=self.collection_name,
                    connection_args={"uri": uri},
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 128},
                    },
                    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                    text_field=TEXT_FIELD,
                    vector_field=VECTOR_FIELD,
                    primary_field=PRIMARY_FIELD,
                    # 关键修复：metadata 走动态字段，schema 不再被首个文档锁死
                    enable_dynamic_field=True,
                    drop_old=settings.MILVUS_DROP_OLD,
                    # 注意：这里刻意不传 timeout。
                    # langchain-milvus 0.4.0 的 add_embeddings() 在“集合首次创建”分支里把
                    # timeout 写进了 kwargs 且未清理，随后 insert(..., timeout=timeout, **kwargs)
                    # 会抛 "got multiple values for keyword argument 'timeout'"，导致首次写入
                    # 必然失败。超时统一由 thread_pool.run_sync(timeout=...) 在外层控制。
                )
                # 真正验证连通性
                store.client.list_collections()
                self._store = store
                self._connected = True
                self._inspect_schema(store)
                logger.info("Milvus 连接成功，集合: %s", self.collection_name)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error("Milvus 连接失败: %s", exc)
                self._store = None
                self._connected = False
                return False

    def _inspect_schema(self, store: Milvus) -> None:
        """
        检查已存在集合的 schema 是否与当前应用兼容

        典型场景：MILVUS_COLLECTION 指向了另一个项目遗留的集合（主键 INT64、
        没有 raw_content 等）。此时读可以降级，写会失败 —— 必须提前说清楚，
        而不是等用户上传时报一句 Milvus 底层错误。
        """
        self._existing_fields = None
        self._incompatible_reason = None

        try:
            client = store.client
            if not client.has_collection(self.collection_name):
                return  # 集合尚不存在，首次写入时按本应用的 schema 创建
            info = client.describe_collection(self.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取集合 schema 失败: %s", exc)
            return

        from pymilvus import DataType

        fields = {field["name"]: field for field in info.get("fields", [])}
        self._existing_fields = set(fields)

        problems: List[str] = []
        primary = fields.get(PRIMARY_FIELD)
        if primary is None:
            problems.append(f"缺少主键字段 {PRIMARY_FIELD}")
        elif primary.get("type") not in (DataType.VARCHAR, DataType.INT64):
            # VARCHAR（显式传入 uuid）与 INT64（langchain-milvus 自动 auto_id）都是合法的
            problems.append(f"主键 {PRIMARY_FIELD} 类型不受支持（实际 {primary.get('type')}）")

        text = fields.get(TEXT_FIELD)
        if text is None:
            problems.append(f"缺少文本字段 {TEXT_FIELD}")
        elif text.get("type") != DataType.VARCHAR:
            problems.append(f"文本字段 {TEXT_FIELD} 不是 VARCHAR")

        vector = fields.get(VECTOR_FIELD)
        if vector is None:
            problems.append(f"缺少向量字段 {VECTOR_FIELD}")
        else:
            params = vector.get("params") or {}
            dim = params.get("dim")
            if dim is not None and int(dim) != self.dimension:
                problems.append(f"向量维度不匹配：集合 {dim} vs 配置 {self.dimension}")

        if problems:
            self._incompatible_reason = "；".join(problems)
            logger.error(
                "集合 %s 与当前应用不兼容：%s\n"
                "修复方式（二选一）：\n"
                "  1) 在 .env 里把 MILVUS_COLLECTION 改成一个新名字（推荐，旧的别动）；\n"
                "  2) 确认数据可丢弃后设置 MILVUS_DROP_OLD=true 让应用重建该集合。",
                self.collection_name,
                self._incompatible_reason,
            )
        else:
            logger.info("集合 schema 校验通过，字段: %s", sorted(self._existing_fields))

    @property
    def schema_warning(self) -> Optional[str]:
        return self._incompatible_reason

    def is_connected(self) -> bool:
        return self._connected and self._store is not None

    def _ensure_store(self) -> Milvus:
        if not self.is_connected():
            # 允许运行期重连（Milvus 重启后不必重启应用）
            if not self.connect():
                raise MilvusException("Milvus 向量库不可用，请检查 MILVUS_HOST / MILVUS_PORT")
        return self._store  # type: ignore[return-value]

    def _retry(self, operation: str, func, *args, **kwargs):
        """对 Milvus 瞬时错误做有限重试（复用 MAX_RETRIES / RETRY_DELAY 配置）"""
        last_exc: Optional[Exception] = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except MilvusException:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= settings.MAX_RETRIES:
                    break
                delay = settings.RETRY_DELAY * attempt
                logger.warning("%s 第 %s 次失败: %s，%.1fs 后重试", operation, attempt, exc, delay)
                time.sleep(delay)
        raise MilvusException(f"{operation}失败（已重试 {settings.MAX_RETRIES} 次）: {last_exc}")

    def _incompatible_hint(self) -> str:
        return (
            f"集合 {self.collection_name} 与当前应用 schema 不兼容（{self._incompatible_reason}）。"
            "请把 MILVUS_COLLECTION 改成新的集合名，或确认可丢弃后设置 MILVUS_DROP_OLD=true 重建。"
        )

    def _uses_dynamic_fields(self) -> bool:
        """集合是否为动态字段模式（metadata 不在 schema 里，但仍可查询）"""
        store = self._store
        return bool(getattr(store, "enable_dynamic_field", False))

    def _available_fields(self, requested: List[str]) -> List[str]:
        """
        计算可安全查询的字段

        - 动态字段模式：schema 里只有 id/content/embedding，metadata 是动态字段，
          **不能**按 schema 过滤，否则 document_id 等字段会被误判为不存在；
        - 静态 schema 模式（历史遗留集合）：与真实字段取交集，避免查询报错。
        """
        if self._existing_fields is None or self._uses_dynamic_fields():
            return list(requested)
        return [name for name in requested if name in self._existing_fields]

    def _fit_document(self, document: Document) -> Document:
        """把 metadata 裁剪到集合已有字段，避免“未知字段”写入失败"""
        if self._existing_fields is None or self._uses_dynamic_fields():
            # 动态字段模式下任何 metadata 键都合法，绝不能过滤（否则元数据会被静默丢弃）
            return document
        metadata = {
            key: value
            for key, value in (document.metadata or {}).items()
            if key in self._existing_fields
        }
        dropped = set(document.metadata or {}) - set(metadata)
        if dropped:
            logger.debug("集合缺少字段 %s，写入时已丢弃", sorted(dropped))
        return Document(page_content=document.page_content, metadata=metadata)

    def _count_rows(self, store: Milvus) -> Optional[int]:
        """
        精确统计行数

        ``get_collection_stats`` 返回的 row_count 依赖 flush/compaction，删除后可能滞后；
        这里优先用 ``count(*)``（Milvus 2.3+ 支持），失败再回退。
        """
        try:
            rows = store.client.query(
                self.collection_name, filter="", output_fields=["count(*)"]
            )
            if rows and isinstance(rows[0], dict):
                for key, value in rows[0].items():
                    if key.startswith("count"):
                        return int(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("count(*) 失败，回退 row_count: %s", exc)

        try:
            stats = store.client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0)) if isinstance(stats, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取集合统计失败: %s", exc)
            return None

    def health(self) -> Dict[str, Any]:
        """健康检查：真正探活（集合尚未创建不算不健康）"""
        base = {
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
            "collection": self.collection_name,
            "schema": "incompatible" if self._incompatible_reason else "ok",
            "schema_warning": self._incompatible_reason,
        }
        try:
            store = self._ensure_store()
            if not store.client.has_collection(self.collection_name):
                # 服务可达，但还没上传过任何文档 → 集合尚未创建，这是正常状态
                return {**base, "status": "connected", "row_count": 0, "collection_exists": False}
            return {
                **base,
                "status": "connected",
                "row_count": self._count_rows(store),
                "collection_exists": True,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                **base,
                "status": "disconnected",
                "error": str(exc),
                "schema": "unsupported" if self._incompatible_reason else "unknown",
            }

    # ==================== 写入 ====================
    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        store = self._ensure_store()

        if self._incompatible_reason:
            # 与其让 Milvus 抛一句 “field raw_content not exist”，不如直接说清楚怎么修
            raise MilvusException(self._incompatible_hint())

        prepared = [self._fit_document(doc) for doc in documents]
        # 显式给出主键：否则 langchain-milvus 会把 auto_id 翻成 True 并自动建 INT64 主键，
        # 导致“同一份配置两次启动看到的 schema 不一样”。
        ids = [uuid.uuid4().hex for _ in prepared]

        def _write() -> List[str]:
            return store.add_documents(prepared, ids=ids)

        try:
            ids = self._retry("写入向量库", _write)
            logger.info("成功写入 %s 条文档到向量库", len(ids))
            return ids
        except MilvusException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("写入向量库失败: %s", exc)
            raise MilvusException(f"写入向量库失败: {exc}") from exc

    def flush(self) -> None:
        """把内存中的写入刷到存储（row_count 统计、外部工具读取前需要）

        注意：部分 Milvus 服务端对 flush 有速率限制，失败只告警、不影响业务。
        """
        store = self._ensure_store()
        try:
            store.client.flush(self.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("flush 集合失败: %s", exc)

    # ==================== 检索 ====================
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        store = self._ensure_store()
        try:
            results = store.similarity_search_with_score(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.error("向量检索失败: %s", exc)
            return []

        if score_threshold is not None:
            # COSINE 相似度：越大越相关
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        logger.info("向量检索完成，返回 %s 条结果", len(results))
        return list(results)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, score_threshold)]

    # ==================== 删除 ====================
    def delete_by_document_id(self, document_id: str) -> int:
        """
        按 document_id 删除该文档的全部 chunk

        Returns:
            实际删除的 chunk 数（删除失败抛 MilvusException，不再“假成功”）
        """
        if not _SAFE_ID.match(document_id or ""):
            raise MilvusException(f"非法的 document_id: {document_id!r}")

        store = self._ensure_store()
        expr = f'document_id == "{_escape_expr_value(document_id)}"'

        try:
            rows = store.client.query(
                self.collection_name,
                filter=expr,
                output_fields=[PRIMARY_FIELD],
                limit=16384,
            )
        except Exception as exc:  # noqa: BLE001
            # 集合为空 / 尚未创建时 query 会报错，这里视为 0 条
            logger.warning("查询待删除数据失败（按 0 条处理）: %s", exc)
            rows = []

        count = len(rows or [])
        if count == 0:
            logger.info("向量库中没有 document_id=%s 的数据", document_id)
            return 0

        ok = self._retry("删除向量数据", lambda: store.delete(expr=expr))
        if not ok:
            raise MilvusException(f"删除 document_id={document_id} 失败")
        logger.info("已删除文档 %s 的 %s 条向量数据", document_id, count)
        return count

    # ==================== 全量读取 / 列表 ====================
    def _iter_query_rows(
        self,
        output_fields: List[str],
        filter_expr: str = "",
        batch_size: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """稳定的全量遍历（query_iterator 优先，失败回退 offset 分页）"""
        store = self._ensure_store()
        client = store.client
        batch = batch_size or settings.MILVUS_QUERY_BATCH

        try:
            if not client.has_collection(self.collection_name):
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("检查集合是否存在失败: %s", exc)
            return

        def _iterator() -> Iterator[Dict[str, Any]]:
            iterator = client.query_iterator(
                self.collection_name,
                batch_size=batch,
                filter=filter_expr,
                output_fields=output_fields,
            )
            try:
                while True:
                    rows = iterator.next()
                    if not rows:
                        break
                    yield from rows
            finally:
                try:
                    iterator.close()
                except Exception:  # noqa: BLE001
                    pass

        try:
            yield from _iterator()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_iterator 失败，回退到 offset 分页: %s", exc)

        offset = 0
        while True:
            rows = client.query(
                self.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=batch,
                offset=offset,
            )
            if not rows:
                break
            yield from rows
            if len(rows) < batch:
                break
            offset += len(rows)

    def get_document_list(self) -> List[Dict[str, Any]]:
        """按 document_id 聚合文档列表（只取必要字段，避免把正文全拉回来）"""
        fields = self._available_fields(["document_id", "document_name", "file_type"])
        if "document_id" not in fields:
            logger.warning("集合缺少 document_id 字段，无法列文档（集合: %s）", self.collection_name)
            return []

        doc_dict: Dict[str, Dict[str, Any]] = {}
        try:
            for row in self._iter_query_rows(fields):
                doc_id = row.get("document_id")
                if not doc_id:
                    continue
                item = doc_dict.get(doc_id)
                if item is None:
                    doc_dict[doc_id] = {
                        "document_id": doc_id,
                        "document_name": row.get("document_name") or "未知文档",
                        "file_type": row.get("file_type") or "unknown",
                        "chunk_count": 1,
                    }
                else:
                    item["chunk_count"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("获取文档列表失败: %s", exc)
            raise MilvusException(f"获取文档列表失败: {exc}") from exc
        return list(doc_dict.values())

    def iter_documents(self) -> List[Document]:
        """读取全部 chunk（用于启动时重建 BM25 索引）"""
        requested = [
            TEXT_FIELD,
            RAW_CONTENT_FIELD,
            "title",
            "document_id",
            "document_name",
            "file_type",
            "page_num",
            "chunk_id",
        ]
        fields = self._available_fields(requested)
        if TEXT_FIELD not in fields:
            logger.warning("集合缺少文本字段 %s，无法重建 BM25 索引", TEXT_FIELD)
            return []

        documents: List[Document] = []
        for row in self._iter_query_rows(fields):
            text = row.get(TEXT_FIELD) or row.get(RAW_CONTENT_FIELD)
            if not text:
                continue
            metadata = {
                key: row.get(key)
                for key in fields
                if key != TEXT_FIELD and row.get(key) is not None
            }
            metadata.setdefault(RAW_CONTENT_FIELD, text)
            documents.append(Document(page_content=text, metadata=metadata))
        logger.info("从 Milvus 读取 %s 条 chunk 用于重建 BM25", len(documents))
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            store = self._ensure_store()
            return {
                "collection_name": self.collection_name,
                "total_vectors": self._count_rows(store),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("获取集合统计失败: %s", exc)
            return {}


_vector_store_service: Optional[VectorStoreService] = None
_vector_store_lock = threading.Lock()


def get_vector_store_service() -> VectorStoreService:
    global _vector_store_service
    if _vector_store_service is None:
        with _vector_store_lock:
            if _vector_store_service is None:
                _vector_store_service = VectorStoreService()
    return _vector_store_service


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.vector_store import vector_store_service``"""
    if name == "vector_store_service":
        return get_vector_store_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
