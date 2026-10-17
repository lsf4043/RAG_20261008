"""
Query 处理服务：改写（指代消解）+ 多查询扩展 + HyDE

关键修复：
- **一次 LLM 调用同时完成“改写 + 变体生成”**。旧实现是串行两次往返
  （rewrite → multi_query），每次问答平白多 1~2 秒；
- 支持 CRAG 重试提示（retry_hint）：第二轮必须真的换一种检索视角，
  否则“重检索”就是原地空转；
- 全部失败时退化为原始问题，绝不阻塞主流程。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QueryPlan(BaseModel):
    """结构化输出：改写后的主查询 + 检索变体"""

    rewritten_query: str = Field(description="消解指代后、可独立理解的主查询")
    variants: List[str] = Field(default_factory=list, description="语义等价但表述不同的检索变体")


_PLAN_PROMPT = """你是检索查询规划器，服务于一个中文知识库问答系统。

任务：
1. 结合对话历史，把用户最新问题改写为「独立、完整、无需上下文即可理解」的查询；
   消解“它 / 这个 / 那个 / 上述 / 前面提到的”等指代词，补全历史中出现的关键实体。
2. 再生成 {n} 条语义等价但表述不同的检索变体，用于提升召回率。

要求：
- 保持原问题语义，不要臆造新的约束条件；
- 变体要覆盖不同说法/同义词/可能的知识库用词，不要只是换标点；
- 如果问题很短或已经是独立问题，改写结果可以与原问题相同；
- 只输出结构化结果，不要解释。
{retry_hint}
对话历史：
{history}

用户最新问题：{question}"""

_RETRY_HINT_TEMPLATE = """
注意：上一轮用下列查询检索后没有找到相关内容，请换一个角度（更换同义词、上位词、
拆解子问题、补充可能的文档用词），不要重复上一轮的查询：
{previous}
"""


class QueryProcessor:
    """Query 改写 / 扩展 / HyDE"""

    # ---------- 内部工具 ----------
    @staticmethod
    def _format_history(history: Optional[List[Dict[str, str]]], limit: int = 6) -> str:
        if not history:
            return "（无）"
        recent = history[-limit:]
        lines = []
        for msg in recent:
            role = "用户" if (msg or {}).get("role") == "user" else "助手"
            content = str((msg or {}).get("content", ""))[:500]
            lines.append(f"{role}: {content}")
        return "\n".join(lines) or "（无）"

    @staticmethod
    def _dedupe(queries: List[str], limit: int) -> List[str]:
        seen = set()
        result: List[str] = []
        for query in queries:
            text = (query or "").strip().strip("\"'“”")
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    # ---------- 组合调用 ----------
    def plan_queries(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        retry_hint: Optional[str] = None,
    ) -> List[str]:
        """
        生成用于检索的 query 列表（第一个元素为主查询）

        Args:
            question: 用户问题
            history: 对话历史
            retry_hint: 上一轮失败的查询描述（CRAG 重检索时传入）
        """
        rewrite_enabled = settings.QUERY_REWRITE_ENABLED
        expand_enabled = settings.MULTI_QUERY_ENABLED

        # 都关掉：直接用原问题
        if not rewrite_enabled and not expand_enabled:
            queries = [question]
            hyde_answer = self.hyde(question)
            if hyde_answer:
                queries.append(hyde_answer)
            return queries

        n = settings.MULTI_QUERY_NUM if expand_enabled else 0
        prompt = _PLAN_PROMPT.format(
            n=n or 1,
            retry_hint=retry_hint or "",
            history=self._format_history(history),
            question=question,
        )

        plan: Optional[QueryPlan] = None
        try:
            planner = get_llm_service().structured(QueryPlan, temperature=0.0)
            result = planner.invoke([HumanMessage(content=prompt)])
            if isinstance(result, QueryPlan):
                plan = result
            elif isinstance(result, dict):
                plan = QueryPlan(**result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("结构化 Query 规划失败，退化为纯文本解析: %s", exc)
            plan = self._plan_by_text(prompt)

        base = question
        variants: List[str] = []
        if plan is not None:
            base = (plan.rewritten_query or question).strip() or question
            if expand_enabled:
                variants = [v for v in plan.variants if v and v.strip()]

        if not rewrite_enabled:
            base = question
        if not expand_enabled:
            variants = []

        queries = self._dedupe([base] + variants, limit=1 + max(n, 0))

        hyde_answer = self.hyde(base)
        if hyde_answer:
            queries.append(hyde_answer)

        logger.info("检索 query 规划完成：%s 条（retry=%s）", len(queries), bool(retry_hint))
        return queries or [question]

    def _plan_by_text(self, prompt: str) -> Optional[QueryPlan]:
        """结构化输出不可用时的兜底：纯文本 + 行解析"""
        try:
            text = get_llm_service().chat(
                [{"role": "user", "content": prompt + "\n\n请按以下格式输出：\n第一行：改写后的查询\n其余每行：一个检索变体"}],
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query 规划完全失败，使用原问题: %s", exc)
            return None

        lines = [re.sub(r"^\s*[-*\d.、)）]+\s*", "", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return None
        return QueryPlan(rewritten_query=lines[0], variants=lines[1:])

    # ---------- 向后兼容的细粒度接口 ----------
    def rewrite_query(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        if not settings.QUERY_REWRITE_ENABLED or not history:
            return question
        return self.plan_queries(question, history)[0]

    def multi_query_expand(self, question: str) -> List[str]:
        if not settings.MULTI_QUERY_ENABLED:
            return [question]
        return self.plan_queries(question, None)

    def hyde(self, question: str) -> Optional[str]:
        if not settings.HYDE_ENABLED:
            return None
        prompt = (
            "请针对下面的问题写一段简短的假设性答案（不需要准确，覆盖可能的答案要点即可），"
            "用于向量检索：\n\n问题："
            f"{question}\n\n假设答案："
        )
        try:
            answer = get_llm_service().chat(
                [{"role": "user", "content": prompt}], temperature=0.5, max_tokens=300
            ).strip()
            return answer or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("HyDE 生成失败: %s", exc)
            return None

    def build_retrieval_queries(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        retry_hint: Optional[str] = None,
    ) -> List[str]:
        return self.plan_queries(question, history, retry_hint=retry_hint)

    @staticmethod
    def build_retry_hint(previous_queries: List[str]) -> str:
        if not previous_queries:
            return ""
        listed = "\n".join(f"- {q[:120]}" for q in previous_queries[:5])
        return _RETRY_HINT_TEMPLATE.format(previous=listed)


_query_processor: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor


def __getattr__(name: str) -> object:
    """兼容旧的 ``from app.core.query_processor import query_processor``"""
    if name == "query_processor":
        return get_query_processor()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
