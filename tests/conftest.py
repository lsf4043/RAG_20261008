"""
pytest 全局配置

**必须在导入 app 之前设置环境变量**：
- 用假的 API Key，确保任何“真调用外部模型”的代码路径都会快速失败而不是烧钱；
- 关闭文件日志，避免测试往 logs/ 里写文件；
- 关闭 LangSmith 追踪，避免测试数据上报。
"""
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-test-dashscope-000000000000000000"
os.environ["DEEPSEEK_API_KEY"] = "sk-test-deepseek-0000000000000000000"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LOG_TO_FILE"] = "0"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["REQUEST_TIMEOUT"] = "30"
os.environ["MILVUS_TIMEOUT"] = "1"
os.environ["LLM_MAX_RETRIES"] = "0"

import pytest  # noqa: E402


@pytest.fixture()
def make_document():
    """构造测试用 Document 的便捷工厂"""

    def _make(content: str, **metadata):
        from langchain_core.documents import Document

        return Document(page_content=content, metadata=metadata)

    return _make


@pytest.fixture()
def fake_llm(monkeypatch):
    """
    把 app.graph.nodes / app.core.query_processor 里的 LLM 换成假实现

    用法: fake_llm(grader=..., chat=...)
    """

    def _install(grader=None, chat=None):
        class _Structured:
            def __init__(self, result):
                self._result = result

            def invoke(self, messages):
                if isinstance(self._result, Exception):
                    raise self._result
                return self._result

        class _FakeLLM:
            def structured(self, schema, temperature=0.0, max_tokens=None):
                return _Structured(grader() if callable(grader) else grader)

            def chat(self, messages, system_prompt=None, temperature=None, max_tokens=None):
                return chat() if callable(chat) else (chat or "")

            def invoke_messages(self, messages, temperature=None, max_tokens=None):
                return chat() if callable(chat) else (chat or "")

        fake = _FakeLLM()
        monkeypatch.setattr("app.graph.nodes.get_llm_service", lambda: fake)
        monkeypatch.setattr("app.core.query_processor.get_llm_service", lambda: fake)
        # 意图路由在“模糊短句”上也会调 LLM，测试里必须一并替换，避免真实请求
        monkeypatch.setattr("app.core.intent_router.get_llm_service", lambda: fake)
        return fake

    return _install
