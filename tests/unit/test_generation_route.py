"""
生成阶段的来源判定与三级兜底：知识库 → 联网 → 通用知识

设计要点：
- “库里到底有没有答案”由**读证据的模型**声明 [[KB]] / [[GENERAL]]，不靠分数阈值；
- 声明 GENERAL 时丢掉引用；开启联网则先联网，联网无果才用通用知识；
- 联网失败绝不阻塞主流程。
"""
import pytest

from app.config import settings
from app.core.intent_router import ROUTE_DIRECT_FALLBACK, ROUTE_RAG, ROUTE_WEB
from app.core.web_search import WebResult
from app.graph.nodes import (
    PROVENANCE_GENERAL,
    PROVENANCE_KB,
    generate_node,
    split_provenance,
    web_search,
)


class _FakeLLM:
    """按顺序返回预设回复，并记录每次的 system prompt"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.system_prompts = []

    def invoke_messages(self, messages, temperature=None, max_tokens=None):
        self.system_prompts.append(messages[0].content)
        if not self.replies:
            raise AssertionError("LLM 被调用次数超出预期")
        return self.replies.pop(0)


def _install_llm(monkeypatch, *replies):
    import app.graph.nodes as nodes_module

    fake = _FakeLLM(replies)
    monkeypatch.setattr(nodes_module, "get_llm_service", lambda: fake)
    return fake


class _FakeWebService:
    provider = "bing"

    def __init__(self, results, enabled=True):
        self.results = results
        self.enabled = enabled

    def is_enabled(self):
        return self.enabled

    def search(self, query, max_results=None):
        return list(self.results)


def _install_web(monkeypatch, results, enabled=True):
    monkeypatch.setattr(
        "app.graph.nodes.get_web_search_service",
        lambda: _FakeWebService(results, enabled=enabled),
    )


# ---------- 来源声明解析 ----------
def test_split_provenance_kb():
    tag, body = split_provenance("[[KB]]\n答案正文")
    assert tag == PROVENANCE_KB
    assert body == "答案正文"


def test_split_provenance_general_with_colon_and_spaces():
    tag, body = split_provenance("   [[GENERAL]]：用通用知识答")
    assert tag == PROVENANCE_GENERAL
    assert body == "用通用知识答"


def test_split_provenance_absent_defaults_to_none():
    tag, body = split_provenance("普通答案，没有标记")
    assert tag is None
    assert body == "普通答案，没有标记"


# ---------- 第一级：知识库 ----------
def test_generate_keeps_sources_when_model_declares_kb(monkeypatch, make_document):
    doc = make_document("简历写着籍贯陕西宝鸡", document_id="d", page_num=1, chunk_id=0)
    fake = _install_llm(monkeypatch, "[[KB]]\n根据【知识片段 1】，籍贯是陕西宝鸡。")

    result = generate_node(
        {"question": "我的籍贯是哪里", "history": [], "documents": [doc], "notes": []}
    )

    assert result["route"] == ROUTE_RAG
    assert result["answer"].startswith("根据【知识片段 1】")
    assert result["used_fallback"] is False
    assert result["sources"]
    assert PROVENANCE_KB in fake.system_prompts[0]


def test_generate_without_tag_treats_as_kb(monkeypatch, make_document):
    doc = make_document("Milvus 常用 HNSW", document_id="d", page_num=1, chunk_id=0)
    _install_llm(monkeypatch, "根据知识库，Milvus 常用 HNSW。")

    result = generate_node(
        {"question": "Milvus 用什么索引", "history": [], "documents": [doc], "notes": []}
    )

    assert result["route"] == ROUTE_RAG
    assert result["sources"]


# ---------- 第二级：联网 ----------
def test_general_declared_then_web(monkeypatch, make_document):
    """“宝鸡有啥好吃的”：简历只提到籍贯 → 模型声明 GENERAL → 联网"""
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    _install_web(
        monkeypatch,
        [WebResult("宝鸡美食", "https://example.com/baoji", "臊子面、擀面皮")],
    )
    doc = make_document("简历：籍贯陕西宝鸡", document_id="d", page_num=1, chunk_id=0)
    fake = _install_llm(
        monkeypatch,
        "[[GENERAL]]\n（这里本可以用通用知识）",
        "根据网上资料，宝鸡的臊子面很有名 [1]。",
    )

    result = generate_node(
        {"question": "宝鸡有啥好吃的", "history": [], "documents": [doc], "notes": []}
    )

    assert result["route"] == ROUTE_WEB
    assert result["used_fallback"] is False
    assert result["sources"][0]["url"] == "https://example.com/baoji"
    assert "网上资料" in result["answer"]
    assert len(fake.system_prompts) == 2
    assert "检索结果" in fake.system_prompts[1]


def test_no_documents_goes_web_first(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    _install_web(monkeypatch, [WebResult("T", "https://example.com/x", "摘要")])
    _install_llm(monkeypatch, "根据网上资料，答案是……[1]")

    result = generate_node({"question": "问题", "history": [], "documents": [], "notes": []})

    assert result["route"] == ROUTE_WEB
    assert result["sources"][0]["url"] == "https://example.com/x"


# ---------- 第三级：通用知识 ----------
def test_general_declared_and_web_disabled_reuses_general_answer(monkeypatch, make_document):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", False)
    doc = make_document("简历：籍贯陕西宝鸡", document_id="d", page_num=1, chunk_id=0)
    fake = _install_llm(monkeypatch, "[[GENERAL]]\n宝鸡好吃的有岐山臊子面。")

    result = generate_node(
        {"question": "宝鸡有啥好吃的", "history": [], "documents": [doc], "notes": []}
    )

    assert result["route"] == ROUTE_DIRECT_FALLBACK
    assert result["used_fallback"] is True
    assert result["sources"] == []
    assert result["answer"] == "宝鸡好吃的有岐山臊子面。"
    assert len(fake.system_prompts) == 1, "已有通用答案时应复用，不必再调一次"


def test_web_no_results_falls_back_to_general(monkeypatch, make_document):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    _install_web(monkeypatch, [])  # 联网无结果
    doc = make_document("简历：籍贯陕西宝鸡", document_id="d", page_num=1, chunk_id=0)
    _install_llm(monkeypatch, "[[GENERAL]]\n宝鸡好吃的有擀面皮。")

    result = generate_node(
        {"question": "宝鸡有啥好吃的", "history": [], "documents": [doc], "notes": []}
    )

    assert result["route"] == ROUTE_DIRECT_FALLBACK
    assert result["sources"] == []
    assert any("联网" in note for note in result["notes"])


def test_web_failure_does_not_block(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    _install_web(monkeypatch, [], enabled=False)
    _install_llm(monkeypatch, "通用回答")

    result = generate_node({"question": "问题", "history": [], "documents": [], "notes": []})

    assert result["route"] == ROUTE_DIRECT_FALLBACK
    assert result["answer"] == "通用回答"


def test_refuses_when_everything_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "ALLOW_LLM_FALLBACK", False)
    _install_web(monkeypatch, [])

    result = generate_node({"question": "问题", "history": [], "documents": [], "notes": []})

    assert result["route"] == ROUTE_RAG
    assert result["used_fallback"] is True
    assert "知识库" in result["answer"]


# ---------- web_search 辅助函数 ----------
def test_web_search_helper_builds_documents(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    _install_web(monkeypatch, [WebResult("标题", "https://example.com/a", "摘要")])

    outcome = web_search({"question": "问题", "notes": []})

    assert outcome["web_found"] is True
    document = outcome["documents"][0]
    assert document.metadata["url"] == "https://example.com/a"
    assert document.metadata["source"] == "web"


def test_web_search_helper_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    _install_web(monkeypatch, [], enabled=False)

    outcome = web_search({"question": "问题", "notes": []})

    assert outcome["web_found"] is False
    assert outcome["documents"] == []
    assert any("未配置" in note or "未启用" in note for note in outcome["notes"])
