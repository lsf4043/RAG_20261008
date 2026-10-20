"""
意图路由（极简版）

产品策略（已与需求方确认）：

| 输入 | 处理 |
|---|---|
| 闲聊 / 寒暄 / 致谢 / 告别 | 直接作答（不检索） |
| 其余一切 | **先检索知识库**；检索不到再由生成节点用通用知识作答，并在响应里标注 |

为什么这里只保留“闲聊”一条规则：

1. 闲聊是一个**闭集**（你好、谢谢、再见、在吗、你是谁、哈哈…），词表判定既准又零成本；
2. 误判代价极低：漏判只是多走一次检索，最终仍会走“检索不到 → 通用知识作答”，
   答案一样，只多 ~0.3s；
3. 反过来，用规则判断“这是不是知识问题”一定漏——之前用 `什么/怎么/啥/几` 当特征，
   8 条自然提问里 7 条被误判成知识问题（“重庆有啥吃的美食”命中 `啥`），
   导致常识问题被送进空知识库并拒答。**语义判断交给检索结果，不交给正则。**

“检索不到怎么办”不在这里决定，而是在 `generate_node` / 流式生成里，依据
`ALLOW_LLM_FALLBACK` 决定“通用知识作答”还是“拒答”。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

ROUTE_DIRECT = "direct"          # 闲聊直答
ROUTE_RAG = "rag"                # 检索知识库
ROUTE_WEB = "web"                # 知识库不足 → 联网检索
ROUTE_DIRECT_FALLBACK = "direct_fallback"  # 检索未命中 → 通用知识作答

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

# 纯时间类问题：**闭集精确匹配**。
# 为什么要单独处理：模型没有时钟，问“今天几号”时如果走检索 → 必然没有相关文档 →
# 还会白跑一趟联网搜索（搜“今天几号”毫无意义）。这类问题用精确词表拦下最划算，
# 且不会误伤“今天天气怎么样”（那是需要联网的真实查询）。
_TIME_QUERIES = frozenset(
    {
        "今天几号",
        "今天几号了",
        "今天是几号",
        "今天是几号了",
        "今天几号啊",
        "今天日期",
        "今天的日期",
        "今天几月几号",
        "今天是几月几号",
        "现在几点",
        "现在几点了",
        "现在几点钟",
        "现在时间",
        "当前时间",
        "今天星期几",
        "今天星期几了",
        "今天周几",
        "今天礼拜几",
        "今年是哪一年",
        "今年是哪年",
        "现在是几月",
        "这个月是几月",
        "现在是什么时候",
        "现在是白天还是晚上",
    }
)


@dataclass
class IntentDecision:
    """路由结果"""

    route: str   # ROUTE_DIRECT / ROUTE_RAG
    reason: str  # 人类可读的判定理由（写进 notes）
    source: str  # rule / default

    @property
    def is_direct(self) -> bool:
        return self.route == ROUTE_DIRECT


class IntentRouter:
    """意图路由器：闲聊直答，其余一律先检索"""

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
            return IntentDecision(ROUTE_RAG, "空问题，走检索", "default")

        if not settings.ROUTER_ENABLED:
            return IntentDecision(ROUTE_RAG, "意图路由已关闭（ROUTER_ENABLED=false）", "default")

        normalized = self._normalize(text)
        if not normalized:
            return IntentDecision(ROUTE_DIRECT, "只有标点或表情，按闲聊处理", "rule")

        for pattern in _CHITCHAT:
            if pattern.match(normalized):
                return IntentDecision(ROUTE_DIRECT, "寒暄/闲聊", "rule")

        if normalized in _TIME_QUERIES:
            return IntentDecision(ROUTE_DIRECT, "询问当前时间（时间已注入提示词）", "rule")

        # 其余全部先查知识库；查不到由生成节点决定“通用知识作答”还是“拒答”
        return IntentDecision(ROUTE_RAG, "非闲聊，先检索知识库", "rule")


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
