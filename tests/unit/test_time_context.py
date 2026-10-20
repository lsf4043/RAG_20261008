"""当前时间注入：模型没有时钟，必须把“现在”写进提示词"""
import re

from app.config import settings
from app.graph.nodes import (
    build_direct_messages,
    build_kb_messages,
    build_web_messages,
)
from app.utils.time_context import now_text, resolve_timezone, time_context

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} 星期[一二三四五六日]（.+）$")


def test_now_text_format():
    text = now_text("Asia/Shanghai")
    assert _DATE_RE.match(text), text
    assert "Asia/Shanghai" in text


def test_now_text_falls_back_on_bad_timezone():
    text = now_text("Not/AZone")
    assert "UTC+8" in text
    _, label = resolve_timezone("Not/AZone")
    assert label == "UTC+8"


def test_time_context_tells_model_not_to_claim_ignorance():
    context = time_context("Asia/Shanghai")
    assert "现在" in context
    assert "无法获取" in context, "必须明确禁止“我无法获取实时时间”这类回答"


def test_direct_messages_include_current_date():
    messages = build_direct_messages("今天几号", [])

    system = messages[0].content
    assert "当前时间" in system
    assert now_text(settings.APP_TIMEZONE)[:10] in system, "system prompt 里要有今天的日期"


def test_kb_messages_include_current_date():
    messages = build_kb_messages("最近有什么变化", [], "【知识片段 1】内容")
    assert "当前时间" in messages[0].content


def test_web_messages_include_current_date():
    from langchain_core.documents import Document

    documents = [Document(page_content="摘要", metadata={"title": "标题", "url": "https://x"})]
    messages = build_web_messages("今天有什么新闻", [], documents)
    assert "当前时间" in messages[0].content
