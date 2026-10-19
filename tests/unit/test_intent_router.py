"""意图路由：闲聊直答 vs 知识库检索"""
import pytest

from app.config import settings
from app.core import intent_router
from app.core.intent_router import ROUTE_DIRECT, ROUTE_RAG, IntentRouter


@pytest.fixture()
def router():
    return IntentRouter()


class _RecordingLLM:
    def __init__(self, answer="A"):
        self.answer = answer
        self.calls = 0

    def chat(self, messages, system_prompt=None, temperature=None, max_tokens=None):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.parametrize(
    "text",
    ["你好", "您好！", "hi", "hello~", "谢谢你", "多谢啦", "再见", "好的", "嗯嗯",
     "你是谁", "在吗", "哈哈", "早上好", "你好吗"],
)
def test_chitchat_goes_direct_without_llm(router, text, monkeypatch):
    """寒暄必须走规则层，绝不能为了一句“你好”再花一次 LLM 调用"""

    def boom():
        raise AssertionError("闲聊不应调用 LLM 分类")

    monkeypatch.setattr(intent_router, "get_llm_service", boom)

    decision = router.classify(text)
    assert decision.route == ROUTE_DIRECT
    assert decision.source == "rule"


@pytest.mark.parametrize(
    "text",
    [
        "Milvus 支持哪些索引？",
        "怎么配置 HNSW 参数",
        "为什么上传失败",
        "RAG 是什么",
        "部署步骤",
        "这个报错怎么解决",
        "Milvus 和 FAISS 有什么区别",
    ],
)
def test_knowledge_questions_go_rag(router, text):
    assert router.classify(text).route == ROUTE_RAG


@pytest.mark.parametrize(
    "text",
    ["帮我写一段祝福语", "翻译成英文：今天天气不错", "总结一下我这段话", "讲个笑话", "帮我起个名字"],
)
def test_general_tasks_go_direct(router, text):
    assert router.classify(text).route == ROUTE_DIRECT


def test_follow_up_with_history_goes_rag(router):
    history = [{"role": "user", "content": "Milvus 是什么"}, {"role": "assistant", "content": "向量库"}]
    assert router.classify("它呢", history).route == ROUTE_RAG


def test_long_sentence_goes_rag(router):
    text = "我想了解一下这个系统内部的检索流程大概是怎么串起来的"
    decision = router.classify(text)
    assert decision.route == ROUTE_RAG
    assert decision.source == "rule"


def test_ambiguous_short_text_uses_llm(router, monkeypatch):
    fake = _RecordingLLM("A")
    monkeypatch.setattr(intent_router, "get_llm_service", lambda: fake)

    decision = router.classify("帮我看看")

    assert fake.calls == 1
    assert decision.route == ROUTE_DIRECT
    assert decision.source == "llm"


def test_llm_can_decide_rag(router, monkeypatch):
    fake = _RecordingLLM("B")
    monkeypatch.setattr(intent_router, "get_llm_service", lambda: fake)

    decision = router.classify("帮我看看")

    assert decision.route == ROUTE_RAG
    assert decision.source == "llm"


def test_llm_garbage_falls_back_to_rag(router, monkeypatch):
    monkeypatch.setattr(intent_router, "get_llm_service", lambda: _RecordingLLM("不知道"))
    assert router.classify("帮我看看").route == ROUTE_RAG


def test_llm_failure_falls_back_to_rag(router, monkeypatch):
    monkeypatch.setattr(
        intent_router, "get_llm_service", lambda: _RecordingLLM(RuntimeError("超时"))
    )
    decision = router.classify("帮我看看")
    assert decision.route == ROUTE_RAG
    assert decision.source == "default"


def test_router_disabled_always_rag(router, monkeypatch):
    monkeypatch.setattr(settings, "ROUTER_ENABLED", False)
    assert router.classify("你好").route == ROUTE_RAG


def test_router_llm_disabled_skips_llm(router, monkeypatch):
    monkeypatch.setattr(settings, "ROUTER_LLM_ENABLED", False)

    def boom():
        raise AssertionError("ROUTER_LLM_ENABLED=false 时不应调用 LLM")

    monkeypatch.setattr(intent_router, "get_llm_service", boom)
    assert router.classify("帮我看看").route == ROUTE_RAG


def test_empty_question_defaults_to_rag(router):
    assert router.classify("   ").route == ROUTE_RAG


def test_punctuation_only_is_treated_as_chitchat(router):
    assert router.classify("。。。").route == ROUTE_DIRECT


def test_decide_after_route():
    from app.graph.nodes import decide_after_route

    assert decide_after_route({"route": ROUTE_DIRECT}) == "direct"
    assert decide_after_route({"route": ROUTE_RAG}) == "rag"
    assert decide_after_route({}) == "rag"


# ---------- 节点级 ----------
def test_route_intent_node_marks_direct():
    from app.graph.nodes import route_intent_node

    result = route_intent_node({"question": "你好", "history": [], "notes": []})

    assert result["route"] == ROUTE_DIRECT
    assert any("意图路由" in note for note in result["notes"])


def test_route_intent_node_marks_rag():
    from app.graph.nodes import route_intent_node

    result = route_intent_node({"question": "Milvus 支持哪些索引？", "history": [], "notes": []})

    assert result["route"] == ROUTE_RAG


def test_generate_direct_node_has_no_sources(fake_llm):
    from app.graph.nodes import generate_direct_node

    fake_llm(chat="你好！有什么可以帮你？")

    result = generate_direct_node(
        {"question": "你好", "history": [], "notes": [], "temperature": None, "max_tokens": None}
    )

    assert result["answer"] == "你好！有什么可以帮你？"
    assert result["route"] == ROUTE_DIRECT
    assert result["sources"] == []
    assert result["used_fallback"] is True
    assert any("未检索知识库" in note for note in result["notes"])


def test_generate_direct_node_passes_history(fake_llm):
    """多轮闲聊也要带上历史（否则模型会忘掉上文）"""
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
        from app.graph.nodes import generate_direct_node

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

    # system + 2 条历史 + 当前问题
    assert captured["count"] == 4
    assert captured["roles"][0] == "SystemMessage"
    assert captured["roles"][-1] == "HumanMessage"
