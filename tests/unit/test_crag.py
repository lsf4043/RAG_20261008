"""CRAG 决策、批量评分、无依据拒答、重检索真的换 query

注意：CRAG 现在**默认关闭**（见 config.py 的说明）。本文件验证的是“开启 CRAG 时”的
行为，因此统一用 fixture 强制打开；关闭时的默认行为由 test_config 与
test_intent_router 覆盖。
"""
import pytest

from app.config import settings
from app.graph.nodes import (
    decide_after_grade,
    generate_node,
    grade_documents_node,
    query_rewrite_node,
)


@pytest.fixture(autouse=True)
def crag_enabled(monkeypatch):
    monkeypatch.setattr(settings, "CRAG_ENABLED", True)


def _make_docs(make_document, count):
    return [
        make_document(f"文档内容 {i}", document_id="doc", chunk_id=i, page_num=1)
        for i in range(count)
    ]


def test_route_to_retry_when_nothing_relevant_and_iterations_left(make_document):
    state = {
        "is_relevant": False,
        "iteration": 0,
        "documents": [make_document("召回到了但都判为不相关")],
    }
    assert decide_after_grade(state) == "rewrite_retrieve"


def test_route_skips_retry_when_nothing_retrieved():
    """一条候选都没有（知识库为空/向量库不可用）时，重检索毫无意义"""
    state = {"is_relevant": False, "iteration": 0, "documents": []}
    assert decide_after_grade(state) == "rerank"


def test_route_to_rerank_when_relevant():
    assert decide_after_grade({"is_relevant": True, "iteration": 0}) == "rerank"


def test_route_stops_retrying_at_max_iterations():
    state = {"is_relevant": False, "iteration": settings.CRAG_MAX_ITERATIONS}
    assert decide_after_grade(state) == "rerank"


def test_grade_uses_single_batched_call(fake_llm, make_document):
    fake_llm(
        grader={
            "grades": [
                {"index": 1, "relevant": True},
                {"index": 2, "relevant": False},
                {"index": 3, "relevant": True},
            ]
        }
    )
    docs = _make_docs(make_document, 3)

    result = grade_documents_node(
        {"question": "q", "rewritten_query": "q", "documents": docs, "notes": []}
    )

    graded_ids = [doc.metadata["chunk_id"] for doc in result["graded_documents"]]
    assert graded_ids == [0, 2]
    assert result["is_relevant"] is True
    # 输入文档不被就地修改
    assert "crag_relevant" not in docs[0].metadata


def test_grade_fails_open_when_llm_errors(fake_llm, make_document):
    fake_llm(grader=RuntimeError("模型超时"))
    docs = _make_docs(make_document, 3)

    result = grade_documents_node(
        {"question": "q", "rewritten_query": "q", "documents": docs, "notes": []}
    )

    assert len(result["graded_documents"]) == 3, "评分失败时应保留候选，而不是清空"
    assert result["degraded"] is True
    assert any("CRAG" in note for note in result["notes"])


def test_generate_refuses_without_context(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_LLM_FALLBACK", False)

    result = generate_node({"question": "q", "history": [], "documents": [], "notes": []})

    assert result["used_fallback"] is True
    assert result["sources"] == []
    assert "知识库" in result["answer"]


def test_retry_round_widens_candidates_and_feeds_back_failures(fake_llm):
    # structured 抛错 → 走纯文本兜底，保证不依赖真实模型
    fake_llm(grader=RuntimeError("no structured"), chat="改写后的问题\n变体一\n变体二")

    state = {
        "question": "RAG 是什么",
        "history": [],
        "iteration": 1,
        "queries": ["RAG 是什么", "旧变体"],
        "notes": ["已有说明"],
    }
    result = query_rewrite_node(state)

    assert result["candidate_k"] > settings.RETRIEVAL_CANDIDATE_K
    assert result["rewritten_query"] == "改写后的问题"
    assert any("重检索" in note for note in result["notes"])
    assert any("已有说明" == note for note in result["notes"])
