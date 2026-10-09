"""
向量数据库服务（LangChain 版）
使用 LangChain 的 langchain-milvus 集成替代旧版手写 pymilvus 封装。

优化点：
- 使用 LangChain 标准 VectorStore 接口，便于与 Retriever / LangGraph 无缝集成
- 入库时自动向量化（调用 embedding_service），保留标题权重增强（page_content 为标题加权拼接文本）
- 文档元数据（title/content/document_id 等）通过 Milvus JSON 字段存储，支持过滤与删除
- 底层保留 pymilvus Collection 访问能力，用于文档列表统计等高级操作
"""
from typing import List, Dict, Optional, Tuple, Any

from langchain_milvus import Milvus
from langchain_core.documents import Document

from app.config import settings
from app.core.embeddings import embedding_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """Milvus 向量库服务（基于 langchain-milvus）"""

    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION
        self._store: Optional[Milvus] = None
        self._connected = False

    def connect(self) -> bool:
        """
        连接并初始化 Milvus 向量库（同步）

        Returns:
            是否连接成功
        """
        try:
            logger.info(f"正在连接 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

            # 初始化 langchain-milvus 向量库（会自动创建集合与索引）
            self._store = Milvus(
                embedding_function=embedding_service,   # 传入带 embed_documents/embed_query 的对象
                collection_name=self.collection_name,
                connection_args={
                    "host": settings.MILVUS_HOST,
                    "port": settings.MILVUS_PORT,
                },
                # 向量索引：HNSW + 余弦相似度
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 128},
                },
                # 搜索参数
                search_params={
                    "metric_type": "COSINE",
                    "params": {"ef": 64},
                },
                # 文本字段名（LangChain 会将 Document.page_content 存入该字段）
                text_field="content",
                vector_field="embedding",
                primary_field="id",
                # 是否删除同名旧集合（由环境变量控制，默认关闭，避免误删）
                drop_old=settings.MILVUS_DROP_OLD,
            )

            self._connected = True
            logger.info(f"Milvus 连接成功，集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Milvus 连接失败: {str(e)}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected and self._store is not None

    def _ensure_store(self) -> Milvus:
        """确保向量库已初始化，否则抛出异常"""
        if not self.is_connected():
            raise RuntimeError("Milvus 向量库尚未连接，请先调用 connect()")
        return self._store

    # ==================== 写入 ====================

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量写入文档到向量库

        Args:
            documents: LangChain Document 列表。
                       page_content 为实际向量化的文本（由上层构造，可含标题加权），
                       metadata 保存 title/content/document_id 等元数据。

        Returns:
            写入的文档 ID 列表
        """
        store = self._ensure_store()
        try:
            ids = store.add_documents(documents)
            logger.info(f"成功写入 {len(ids)} 条文档到向量库")
            return ids
        except Exception as e:
            logger.error(f"写入向量库失败: {str(e)}")
            raise

    # ==================== 检索 ====================

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        向量检索（返回带分数的文档）

        Args:
            query: 查询文本
            k: 返回数量
            score_threshold: 相似度阈值，低于此值的结果被过滤（可选）

        Returns:
            [(Document, score), ...] 按分数降序排列
        """
        store = self._ensure_store()
        try:
            results = store.similarity_search_with_score(query, k=k)

            # 过滤低于阈值的结果
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]

            logger.info(f"向量检索完成，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        向量检索（仅返回文档，不带分数）

        Args:
            query: 查询文本
            k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            文档列表
        """
        scored = self.similarity_search_with_score(query, k, score_threshold)
        return [doc for doc, _ in scored]

    # ==================== 删除 ====================

    def delete_by_document_id(self, document_id: str) -> int:
        """
        根据 document_id 删除文档（metadata JSON 字段过滤）

        Args:
            document_id: 文档 ID

        Returns:
            删除的数据条数（Milvus 不直接返回，返回 0 表示执行成功）
        """
        store = self._ensure_store()
        try:
            # metadata 存于 JSON 字段，用 JSON 表达式过滤
            expr = f'metadata["document_id"] == "{document_id}"'
            store.delete(expr=expr)
            logger.info(f"已删除文档 '{document_id}' 的所有向量数据")
            return 0
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return 0

    # ==================== 列表/统计 ====================

    def get_document_list(self) -> List[Dict]:
        """
        获取文档列表（按 document_id 聚合统计）

        Returns:
            文档信息列表
        """
        store = self._ensure_store()
        try:
            # 访问底层 pymilvus Collection，查询 metadata JSON 字段
            collection = store.col
            # 确保集合已加载到内存
            if not collection.has_index() or collection.num_entities == 0:
                return []

            # 查询所有实体的 metadata
            results = collection.query(
                expr="",
                output_fields=["metadata"],
                limit=10000,
            )

            # 按 document_id 聚合
            doc_dict: Dict[str, Dict] = {}
            for item in results:
                metadata = item.get("metadata", {})
                # metadata 可能是 JSON 字符串或 dict，做兼容处理
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)

                doc_id = metadata.get("document_id", "")
                if not doc_id:
                    continue

                if doc_id not in doc_dict:
                    doc_dict[doc_id] = {
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name", "未知文档"),
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunk_count": 1,
                    }
                else:
                    doc_dict[doc_id]["chunk_count"] += 1

            return list(doc_dict.values())
        except Exception as e:
            logger.error(f"获取文档列表失败: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息
        """
        store = self._ensure_store()
        try:
            collection = store.col
            return {
                "collection_name": self.collection_name,
                "total_vectors": collection.num_entities,
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {str(e)}")
            return {}


# 全局向量库服务实例
vector_store_service = VectorStoreService()