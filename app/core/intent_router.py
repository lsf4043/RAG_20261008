"""
意图路由（Adaptive RAG）

解决的问题：以前任何输入（包括“你好”“谢谢”）都会走完整 RAG 链路——
Query 改写 → 混合检索 → CRAG 评分 → 生成，检索不到时还会直接拒答。
对闲聊/通用任务来说这是纯浪费，而且体验很差。

现在的判定顺序（**先便宜后昂贵**，绝大多数请求零额外开销）：

1. 规则层（纯字符串匹配，0 成本）
   - 寒暄/闲聊（你好、谢谢、再见、你是谁…） → direct
   - 命中知识类特征（什么/怎么/为什么/配置/报错…，或含问号） → rag
   - 通用任务（写文案、翻译、总结、讲笑话…）且不含知识特征 → direct
   - 多轮指代（它/这个/上面… 且已有历史） → rag
   - 长句（≥ ROUTER_LONG_TEXT_THRESHOLD） → rag
2. 少量模糊短句（如“帮我优化下”）才调用一次极小 LLM 分类（max_tokens=4）
   - LLM 不可用/输出异常 → 保守回退 rag

为什么要“保守回退 rag”：走 RAG 最坏是拒答（有 `ALLOW_LLM_FALLBACK` 兜底策略），
而把知识问题误判为闲聊则会直接产生幻觉，代价更大。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.config import settings
from app.core.llm import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

ROUTE_DIRECT = "direct"
ROUTE_RAG = "rag"

# 去掉首尾标点 / 空白 / 表情符号后再判断
_TRIM_CHARS = " \t\r\n!！。.,，、?？~～;；:：\"'“”‘’()（）[]【】…—-"
_GREETING = re.compile(
    r"^(你好|您好|hi|hello|hey|哈喽|在吗|在么|早上好|中午好|下午好|晚上好|早安|晚安)(呀|啊|吗|呢|哈|嘛)?$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(谢谢|多谢|感谢|谢了|thanks|thank you|thx)(你|啦|了)?$", re.IGNORECASE)
_BYE = re.compile(r"^(再见|拜拜|拜|bye|goodbye|see you)(啦|了)?$", re.IGNORECASE)
_ACK = re.compile(r"^(好的?|好滴|嗯+|哦+|收到|ok|okay|了解|明白|知道了)$", re.IGNORECASE)
_LAUGH = re.compile(r"^(哈+|呵+|嘿+|呵呵|哈哈)$")
_SELF_INTRO = re.compile(
    r"^(你是谁|你叫什么|你是什么|你能做什么|你会什么|你能干什么|你能帮我做什么|介绍一下你自己|自我介绍)$"
)

_CHITCHAT: Sequence[re.Pattern] = (_GREETING, _THANKS, _BYE, _ACK, _LAUGH, _SELF_INTRO)

# 知识类特征：出现这些词基本可以确定需要查知识库
_KNOWLEDGE_HINTS = re.compile(
    r"(什么|啥|怎么|怎样|如何|为什么|为啥|哪些|哪个|哪里|哪儿|是否|多少|几|"
    r"区别|对比|比较|原理|定义|概念|含义|作用|"
    r"配置|部署|安装|启动|报错|异常|错误|失败|参数|用法|示例|接口|文档|版本|支持|"
    r"步骤|流程|规范|标准|限制|性能|优化|架构|实现)"
)

# 通用任务：出现这些动词（且不含知识特征）时，模型自己就能完成
_TASK_VERBS = re.compile(
    r"(写|起草|生成|创作|编|讲|翻译|润色|改写|总结|概括|提炼|列|计算|算|校对|"
    r"(起|取)(个|一个)?名(字|儿)?)"
)

# 多轮指代：需要结合历史 + 知识库
_FOLLOW_UP = re.compile(r"^(它|他|她|这个|那个|这些|那些|上面|刚才|前面|上述|该|此)|(呢|吗)$")


@dataclass
class IntentDecision:
    """路由结果"""

    route: str          # ROUTE_DIRECT / ROUTE_RAG
    reason: str         # 人类可读的判定理由（会写进 notes）
    source: str         # rule / llm / default

    @property
    def is_direct(self) -> bool:
        return self.route == ROUTE_DIRECT


_INTENT_PROMPT = """判断用户这句话是否需要查询【特定知识库】才能回答。

A = 不需要：闲聊、问候、通用写作、翻译、计算、常识问答
B = 需要：涉及具体产品文档、内部资料、专有数据、事实性查询

只回答一个字母（A 或 B），不要解释。

用户：{question}
"""


class IntentRouter:
    """意图路由器"""

    # ---------- 规则层 ----------
    @staticmethod
    def _normalize(question: str) -> str:
        return (question or "").strip().strip(_TRIM_CHARS).strip()

    def classify(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentDecision:
        text = (question or "").strip()
        if not text:
            return IntentDecision(ROUTE_RAG, "空问题，默认走检索", "default")

        if not settings.ROUTER_ENABLED:
            return IntentDecision(ROUTE_RAG, "意图路由已关闭（ROUTER_ENABLED=false）", "default")

        normalized = self._normalize(text)
        if not normalized:
            return IntentDecision(ROUTE_DIRECT, "只有标点或表情，按闲聊处理", "rule")

        for pattern in _CHITCHAT:
            if pattern.match(normalized):
                return IntentDecision(ROUTE_DIRECT, "寒暄/闲聊", "rule")

        has_question_mark = any(mark in text for mark in ("?", "？"))
        if _KNOWLEDGE_HINTS.search(normalized) or has_question_mark:
            return IntentDecision(ROUTE_RAG, "命中知识类特征", "rule")

        if _TASK_VERBS.search(normalized):
            return IntentDecision(ROUTE_DIRECT, "通用任务（写作/翻译/计算等）", "rule")

        if history and _FOLLOW_UP.search(normalized):
            return IntentDecision(ROUTE_RAG, "多轮指代，需要检索知识库", "rule")

        if len(normalized) >= settings.ROUTER_LONG_TEXT_THRESHOLD:
            return IntentDecision(ROUTE_RAG, "长句默认走检索", "rule")

        # ---------- 少量模糊短句才交给 LLM ----------
        if settings.ROUTER_LLM_ENABLED:
            verdict = self._llm_decide(text, history)
            if verdict is not None:
                return IntentDecision(verdict, "LLM 意图分类", "llm")

        return IntentDecision(ROUTE_RAG, "无法判定，保守默认走检索", "default")

    # ---------- LLM 兜底分类（只在模糊短句上触发） ----------
    def _llm_decide(self, question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        try:
            # 刻意不用 structured 输出（DeepSeek 不支持 json_schema），
            # 只要一个字母，纯文本足够稳
            answer = get_llm_service().chat(
                [{"role": "user", "content": _INTENT_PROMPT.format(question=question)}],
                temperature=0.0,
                max_tokens=4,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[intent] LLM 分类失败，回退到检索: %s", exc)
            return None

        verdict = (answer or "").strip().upper()[:1]
        if verdict == "A":
            return ROUTE_DIRECT
        if verdict == "B":
            return ROUTE_RAG
        logger.debug("[intent] LLM 返回无法解析: %r，回退到检索", answer)
        return None


_router: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


def classify_intent(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentDecision:
    """供节点 / 服务层调用的便捷入口"""
    return get_intent_router().classify(question, history)
