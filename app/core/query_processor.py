"""
Query 处理服务（升级版新增，P1 优化）
实现 Query 改写 / Multi-Query 扩展 / HyDE，提升复杂问题与多轮对话的检索召回率。

方法说明：
- rewrite_query：结合对话历史改写 query，消解指代词（它/这个/那个等）
- multi_query_expand：生成多个语义等价的 query 变体，用于多路检索
- hyde：让 LLM 先生成假设答案，用假设答案做向量检索（提升召回）
"""
import re
from typing import List, Optional, Dict

from app.config import settings
from app.core.llm import llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QueryProcessor:
    """Query 改写/扩展处理器"""

    # ---- Query 改写（指代消解） ----

    def rewrite_query(self, question: str, history: Optional[List[Dict]] = None) -> str:
        """
        结合对话历史改写 query，消解指代词

        Args:
            question: 用户最新问题
            history: 对话历史 [{role: user/assistant, content: ...}]

        Returns:
            改写后的完整问题
        """
        if not settings.QUERY_REWRITE_ENABLED:
            return question

        # 无历史时直接返回原问题，无需改写
        if not history:
            return question

        # 将历史格式化为文本
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-6:]]  # 只取最近 6 条
        )

        prompt = f"""你是一个对话改写助手。请根据对话历史，将用户的最新问题改写为一个独立、完整、无需上下文即可理解的问题。

要求：
1. 消解指代词（如"它"、"这个"、"那个"、"上述"等），替换为具体的实体
2. 补充历史中提到的关键信息
3. 只输出改写后的问题，不要输出任何解释或额外内容

对话历史：
{history_text}

用户最新问题：{question}

改写后的问题："""

        try:
            rewritten = llm_service.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # 改写任务用低温，保证稳定
            ).strip()
            # 清理可能的多余引号/换行
            rewritten = rewritten.strip('"\' \n')
            logger.info(f"Query 改写: '{question[:30]}' -> '{rewritten[:30]}'")
            return rewritten or question
        except Exception as e:
            logger.warning(f"Query 改写失败，使用原问题: {str(e)}")
            return question

    # ---- Multi-Query 扩展 ----

    def multi_query_expand(self, question: str) -> List[str]:
        """
        生成多个语义等价的 query 变体（Multi-Query）

        Args:
            question: 原始问题

        Returns:
            query 变体列表（含原问题）
        """
        if not settings.MULTI_QUERY_ENABLED:
            return [question]

        n = settings.MULTI_QUERY_NUM
        prompt = f"""你是一个查询扩展助手。请将用户的问题改写为 {n} 个语义相同但表述不同的变体，用于多路检索以提升召回率。

要求：
1. 每个变体从不同角度表述同一个问题
2. 保持语义一致，不要改变问题的含义
3. 每行一个变体，不要编号、不要任何解释

用户问题：{question}

{n} 个变体："""

        try:
            response = llm_service.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            ).strip()

            # 按行解析变体，过滤空行和编号前缀
            variants: List[str] = []
            for line in response.split("\n"):
                line = re.sub(r'^\s*\d+[.、)）]\s*', '', line).strip()  # 去掉编号
                if line and line.lower() not in [v.lower() for v in variants]:
                    variants.append(line)

            # 确保原问题在列表中（去重后）
            result = [question] + variants[:n]
            logger.info(f"Multi-Query 扩展: 生成 {len(result)} 个查询变体")
            return result
        except Exception as e:
            logger.warning(f"Multi-Query 扩展失败，仅使用原问题: {str(e)}")
            return [question]

    # ---- HyDE（假设答案） ----

    def hyde(self, question: str) -> Optional[str]:
        """
        生成假设答案（HyDE），用于假设答案向量化后检索

        Args:
            question: 用户问题

        Returns:
            假设答案文本（若禁用则返回 None）
        """
        if not settings.HYDE_ENABLED:
            return None

        prompt = f"""请简要回答以下问题，生成一个假设性答案（不需要完全准确，只需覆盖可能的答案要点）：

问题：{question}

假设答案："""

        try:
            answer = llm_service.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            ).strip()
            logger.info(f"HyDE 生成假设答案: {answer[:50]}...")
            return answer
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {str(e)}")
            return None

    # ---- 综合处理：生成用于检索的 query 列表 ----

    def build_retrieval_queries(
        self,
        question: str,
        history: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        生成用于检索的 query 列表（整合改写 + 扩展 + HyDE）

        Args:
            question: 原始问题
            history: 对话历史

        Returns:
            检索 query 列表（原始问题 + 改写问题 + 变体 + 假设答案）
        """
        queries: List[str] = []

        # 1. Query 改写（多轮指代消解）
        rewritten = self.rewrite_query(question, history)

        # 2. Multi-Query 扩展
        expanded = self.multi_query_expand(rewritten)

        queries.extend(expanded)

        # 3. HyDE 假设答案（可选）
        hyde_answer = self.hyde(rewritten)
        if hyde_answer:
            queries.append(hyde_answer)

        # 去重
        seen = set()
        deduped = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                deduped.append(q)

        logger.info(f"生成检索 query 列表：共 {len(deduped)} 个")
        return deduped


# 全局 Query 处理器实例
query_processor = QueryProcessor()