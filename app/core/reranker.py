"""
重排序服务（LangChain 版）
使用 LangChain 的 HuggingFaceCrossEncoder 对候选文档做 Cross-Encoder 重排序。

优化点：
- 使用 HuggingFaceCrossEncoder.score 计算相关性分数，按分数降序排列
- 保留失败降级逻辑（模型加载失败时回退到原始顺序）
"""
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RerankerService:
    """重排序服务（基于 HuggingFaceCrossEncoder）"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._cross_encoder = None
        self._initialized = False
        self._available = False  # 模型是否可用（加载失败时为 False）

    def initialize(self):
        """
        初始化重排序模型（同步，建议在应用启动时调用）
        """
        if self._initialized:
            return

        try:
            logger.info(f"正在加载重排序模型: {self.model_name}")

            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            self._cross_encoder = HuggingFaceCrossEncoder(model_name=self.model_name)

            self._available = True
            self._initialized = True
            logger.info("重排序模型加载成功")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {str(e)}")
            logger.warning("将使用原始检索顺序作为降级方案")
            self._cross_encoder = None
            self._available = False
            self._initialized = True

    def is_available(self) -> bool:
        """检查重排序模型是否可用"""
        return self._available

    def rerank(self, documents: List[Document], query: str, top_n: Optional[int] = None) -> List[Document]:
        """
        对候选文档重排序（Cross-Encoder 打分后按分数降序截取）

        Args:
            documents: 候选文档列表
            query: 查询文本
            top_n: 保留的文档数量（可选，默认用配置值）

        Returns:
            重排序后的文档列表
        """
        if not self._initialized:
            self.initialize()

        if not documents:
            return []

        limit = top_n or settings.RERANK_TOP_K

        # 降级：模型不可用，返回原始文档（截取 top_n）
        if not self._available or self._cross_encoder is None:
            logger.warning("重排序模型不可用，返回原始顺序")
            return documents[:limit]

        try:
            # 构造 (query, document_text) 对，调用 cross_encoder.score 批量打分
            text_pairs: List[Tuple[str, str]] = [
                (query, doc.page_content) for doc in documents
            ]
            scores = self._cross_encoder.score(text_pairs)

            # 将分数写入 metadata，按分数降序排列
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = score
                scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = [doc for doc, _ in scored_docs[:limit]]

            logger.info(f"重排序完成，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return documents[:limit]


# 全局重排序服务实例
reranker_service = RerankerService()