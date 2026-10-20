"""
意图路由与回答策略

策略（产品决策）：
- 闲聊/寒暄 → 直接作答
- 其余一切 → 先检索知识库
- 检索不到 → 用通用知识作答并标注（ALLOW_LLM_FALLBACK=true）；关闭时明确拒答

这个文件同时守住一个回归点：**不要再用“疑问词”判断是不是知识问题**。
之前用 `什么/啥/怎么/几` 当特征，导致“重庆有啥吃的美食”被判成知识问题，
送进空知识库后直接拒答。
"""
import pytest

from app.config import settings
from app.core import intent_router
from app.core.intent_router import ROUTE_DIRECT, ROUTE_DIRECT_FALLBACK, ROUTE_RAG, IntentRouter
from app.graph.nodes import (
    NO_CONTEXT_ANSWER,
    decide_after_route,
    generate_direct_node,
    generate_node,
    route_intent_node,
)


@pytest.fixture()
def router():
    return IntentRouter()


# ---------- 闲聊 → 直答 ----------
@pytest.mark.parametrize(
    "text",
    ["你好", "您好！", "hi", "hello~", "谢谢你", "多谢啦", "再见", "好的", "嗯嗯",
     "你是谁", "在吗", "哈哈", "早上好", "你好吗"],
)
def test_chitchat_goes_direct_without_llm(router, text, monkeypatch):
    """寒暄走规则层，零成本；绝不能为了一句“你好”再调一次模型"""

    def boom():
        raise AssertionError("闲聊判定不应触达 LLM")

    monkeypatch.setattr(intent_router, "get_llm_service", boom, raising=False)

    decision = router.classify(text)
    assert decision.route == ROUTE_DIRECT
    assert decision.source == "rule"


# ---------- 纯时间问题 → 直答（不检索、不联网） ----------
@pytest.mark.parametrize(
    "text",
    ["今天几号", "今天是几号？", "现在几点了", "今天星期几", "当前时间", "今天日期"],
)
def test_time_questions_go_direct(router, text):
    decision = router.classify(text)

    assert decision.route == ROUTE_DIRECT, "问日期时不该白跑检索和联网"
    assert "时间" in decision.reason


@pytest.mark.parametrize(
    "text",
    ["今天天气怎么样", "今天有什么新闻", "今天的会议纪要说了什么"],
)
def test_questions_containing_today_are_not_time_questions(router, text):
    """“今天…怎么样”是要检索/联网的真实查询，不能被时间词表吞掉"""
    assert router.classify(text).route == ROUTE_RAG


# ---------- 其余一律先检索（这正是本次修复的核心）----------
@pytest.mark.parametrize(
    "text",
    [
        "重庆有啥吃的美食",          # 之前被 `啥` 误判成知识问题 → 现在也应先检索，但不会再拒答
        "重庆有什么好吃的",
        "推荐几个重庆的火锅店",
        "北京天气怎么样",
        "Milvus 支持哪些索引？",
        "帮我写一段祝福语",
        "翻译成英文：今天天气不错",
        "帮我看看",
        "它呢",
    ],
)
def test_everything_else_goes_rag(router, text):
    decision = router.classify(text)
    assert decision.route == ROUTE_RAG
    assert decision.source == "rule"


def test_history_does_not_change_routing(router):
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
    ]
    # 多轮里每一轮都独立判定：上一轮是闲聊，不影响这一轮
    assert router.classify("重庆有啥吃的美食", history).route == ROUTE_RAG
    assert router.classify("谢谢", history).route == ROUTE_DIRECT


def test_router_disabled_always_rag(router, monkeypatch):
    monkeypatch.setattr(settings, "ROUTER_ENABLED", False)
    assert router.classify("你好").route == ROUTE_RAG


def test_empty_question_defaults_to_rag(router):
    assert router.classify("   ").route == ROUTE_RAG


def test_punctuation_only_is_treated_as_chitchat(router):
    assert router.classify("。。。").route == ROUTE_DIRECT


# ---------- 图路由 ----------
def test_decide_after_route():
    assert decide_after_route({"route": ROUTE_DIRECT}) == "direct"
    assert decide_after_route({"route": ROUTE_RAG}) == "rag"
    assert decide_after_route({"route": ROUTE_DIRECT_FALLBACK}) == "rag"
    assert decide_after_route({}) == "rag"


def test_route_intent_node_marks_direct():
    result = route_intent_node({"question": "你好", "history": [], "notes": []})
    assert result["route"] == ROUTE_DIRECT
    assert any("意图路由" in note for note in result["notes"])


def test_route_intent_node_marks_rag():
    result = route_intent_node({"question": "Milvus 支持哪些索引？", "history": [], "notes": []})
    assert result["route"] == ROUTE_RAG


# ---------- 闲聊直答节点 ----------
def test_generate_direct_node_has_no_sources(fake_llm):
    fake_llm(chat="你好！有什么可以帮你？")

    result = generate_direct_node(
        {"question": "你好", "history": [], "notes": [], "temperature": None, "max_tokens": None}
    )

    assert result["answer"] == "你好！有什么可以帮你？"
    assert result["route"] == ROUTE_DIRECT
    assert result["sources"] == []
    assert result["used_fallback"] is True
    assert any("未检索知识库" in note for note in result["notes"])


def test_generate_direct_node_passes_history():
    """多轮闲聊也要带上历史"""
    captured = {}

    class _LLM:
        def invoke_messages(self, messages, temperature=None, max_tokens=None):
            captured["count"] = len(messages)
            captured["roles"] = [type(m).__name__ for m in messages]
            return "好的"

    import app.graph.nodes as nodes_module

    original = nodes_module.get_llm_service
    nodes_module.get_llm_service = lambda: _LLM()
    try:
        generate_direct_node(
            {
                "question": "谢谢",
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ],
                "notes": [],
            }
        )
    finally:
        nodes_module.get_llm_service = original

    assert captured["count"] == 4  # system + 2 条历史 + 当前问题
    assert captured["roles"][0] == "SystemMessage"
    assert captured["roles"][-1] == "HumanMessage"


# ---------- 检索未命中 → 通用知识作答 ----------
def test_generate_node_falls_back_to_general_answer(fake_llm, monkeypatch):
    """“重庆有啥吃的美食”在空知识库下的正确行为：用通用知识答，而不是拒答"""
    monkeypatch.setattr(settings, "ALLOW_LLM_FALLBACK", True)
    fake_llm(chat="重庆的火锅和小面都很有名，还有酸辣粉、毛血旺。")

    result = generate_node(
        {"question": "重庆有啥吃的美食", "history": [], "documents": [], "notes": []}
    )

    assert result["route"] == ROUTE_DIRECT_FALLBACK
    assert result["sources"] == []
    assert result["used_fallback"] is True
    assert "火锅" in result["answer"]
    assert any("通用知识" in note for note in result["notes"])


def test_generate_node_refuses_when_fallback_disabled(fake_llm, monkeypatch):
    """严格合规场景：关闭 fallback 后仍然是明确拒答"""
    monkeypatch.setattr(settings, "ALLOW_LLM_FALLBACK", False)

    def boom():
        raise AssertionError("拒答时不应该调用 LLM")

    import app.graph.nodes as nodes_module

    original = nodes_module.get_llm_service
    nodes_module.get_llm_service = boom
    try:
        result = generate_node(
            {"question": "重庆有啥吃的美食", "history": [], "documents": [], "notes": []}
        )
    finally:
        nodes_module.get_llm_service = original

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["route"] == ROUTE_RAG
    assert result["used_fallback"] is True


def test_generate_node_uses_kb_prompt_when_documents_exist(fake_llm, make_document):
    captured = {}

    class _LLM:
        def invoke_messages(self, messages, temperature=None, max_tokens=None):
            captured["system"] = messages[0].content
            return "根据知识库……【知识片段 1】"

    import app.graph.nodes as nodes_module

    original = nodes_module.get_llm_service
    nodes_module.get_llm_service = lambda: _LLM()
    try:
        result = generate_node(
            {
                "question": "Milvus 用什么索引？",
                "history": [],
                "documents": [make_document("Milvus 常用 HNSW 索引", document_id="d", chunk_id=0, page_num=1)],
                "notes": [],
            }
        )
    finally:
        nodes_module.get_llm_service = original

    assert result["route"] == ROUTE_RAG
    assert result["used_fallback"] is False
    assert "知识片段 1" in captured["system"]
    assert result["sources"]
