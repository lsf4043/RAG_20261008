"""
BM25 检索器（LangChain 版）
使用 LangChain 的 BM25Retriever 替代旧版手写 BM25，并保留中英文混合分词优化。

优化点：
- 使用 LangChain 标准 Retriever 接口，便于与向量检索器组成混合检索（EnsembleRetriever / RRF）
- 保留中英文混合分词（中文按字、英文按词），通过 preprocess_func 注入
- 维护内存文档索引，与 Milvus 向量库保持同步
"""
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.utils.logger import get_logger

logger = get_logger(__name__)


class BM25RetrieverService:
    """BM25 稀疏检索服务（基于 LangChain BM25Retriever）"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        初始化 BM25 检索服务

        Args:
            k1: BM25 参数 k1
            b: BM25 参数 b
            epsilon: IDF 下界
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        # 内存文档索引（与向量库同步）
        self._documents: List[Document] = []
        logger.info(f"BM25 检索服务初始化完成 (k1={k1}, b={b})")

    def _tokenize(self, text: str) -> List[str]:
        """
        中英文混合分词（中文按字，英文按词，保留原项目优化）

        Args:
            text: 待分词文本

        Returns:
            词元列表
        """
        tokens: List[str] = []
        english_word = ""

        for char in text:
            # 中文字符：中文按单字切分
            if '\u4e00' <= char <= '\u9fff':
                if english_word:
                    tokens.append(english_word.lower())
                    english_word = ""
                tokens.append(char)
            # 英文字母和数字：累计为英文单词
            elif char.isalnum():
                english_word += char
            # 其他字符（标点、空格等）：作为英文单词的分隔符
            else:
                if english_word:
                    tokens.append(english_word.lower())
                    english_word = ""

        # 处理末尾遗留的英文单词
        if english_word:
            tokens.append(english_word.lower())

        return tokens

    def rebuild(self, documents: List[Document]):
        """
        重建 BM25 索引（文档库变化时调用）

        Args:
            documents: 全部文档列表（与向量库保持一致）
        """
        self._documents = documents
        logger.info(f"BM25 索引重建完成，文档数: {len(documents)}")

    @property
    def count(self) -> int:
        """当前索引的文档数量"""
        return len(self._documents)

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """
        获取 BM25 检索器（LangChain 标准 Retriever）

        Args:
            k: 返回文档数量

        Returns:
            BM25Retriever 实例
        """
        if not self._documents:
            # 空索引时返回一个空检索器（避免报错）
            retriever = BM25Retriever.from_documents(
                [Document(page_content="__empty__")],
                preprocess_func=self._tokenize,
                k=k
            )
        else:
            # bm25_params 传递 BM25 算法参数（k1/b/epsilon）
            retriever = BM25Retriever.from_documents(
                self._documents,
                preprocess_func=self._tokenize,
                k=k,
                bm25_params={
                    "k1": self.k1,
                    "b": self.b,
                    "epsilon": self.epsilon,
                },
            )
        return retriever

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        BM25 检索（返回文档列表，带排名信息用于混合检索）

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            文档列表（按 BM25 相关度降序）
        """
        if not self._documents:
            return []

        retriever = self.get_retriever(k=top_k)
        results = retriever.invoke(query)
        logger.info(f"BM25 检索完成，返回 {len(results)} 条结果")
        return list(results)


# 全局 BM25 检索服务实例
bm25_retriever_service = BM25RetrieverService()