"""
LLM 服务的回归测试

重点是一个曾经把整个问答接口拖成 120s 超时的真实 bug：
``LLMService._variant()`` 先持有 `self._lock`，再调用 `self.base`，而 `base`
内部又要获取同一把锁。`threading.Lock` 不可重入 → 自死锁。
只有“首次调用就走 _variant（如 structured / 指定 temperature）”才会触发，
所以单测里一般不会发现，必须显式回归。
"""
import threading

from app.core import llm as llm_module


class StubModel:
    """假的 ChatModel：只需要支持 model_copy / with_structured_output"""

    def __init__(self) -> None:
        self.copies = 0

    def model_copy(self, update=None):
        self.copies += 1
        return StubModel()

    def with_structured_output(self, schema, **kwargs):
        return "structured-runnable"


def _run_with_timeout(func, timeout=5.0):
    """在线程里执行，超时即判定为死锁（daemon 线程不会阻塞进程退出）"""
    result = {}

    def runner():
        result["value"] = func()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    return thread, result


def test_structured_first_call_does_not_deadlock(monkeypatch):
    monkeypatch.setattr(llm_module, "init_chat_model", lambda **kwargs: StubModel())
    service = llm_module.LLMService()

    thread, result = _run_with_timeout(lambda: service.structured(dict, temperature=0.0))

    assert not thread.is_alive(), "structured() 首次调用发生自死锁（_variant 持锁后又取 self.base）"
    assert result["value"] == "structured-runnable"


def test_temperature_variant_first_call_does_not_deadlock(monkeypatch):
    monkeypatch.setattr(llm_module, "init_chat_model", lambda **kwargs: StubModel())
    service = llm_module.LLMService()

    thread, result = _run_with_timeout(lambda: service._variant(temperature=0.3))

    assert not thread.is_alive(), "_variant() 首次调用发生自死锁"


def test_base_is_created_once_and_variants_are_cached(monkeypatch):
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return StubModel()

    monkeypatch.setattr(llm_module, "init_chat_model", factory)
    service = llm_module.LLMService()

    first = service._variant(temperature=0.0)
    second = service._variant(temperature=0.0)
    third = service._variant(temperature=0.5)

    assert len(created) == 1, "底层 ChatModel 只应构造一次"
    assert first is second, "相同参数应复用同一个变体"
    assert third is not first


def test_lock_is_reentrant():
    """即使将来又有人在持锁路径里取锁，也不该自死锁"""
    service = llm_module.LLMService()
    assert hasattr(service._lock, "acquire") and getattr(service._lock, "_is_owned", None) is not None
    service._lock.acquire()
    try:
        assert service._lock.acquire(blocking=False), "锁必须可重入"
        service._lock.release()
    finally:
        service._lock.release()


class StructuredStub:
    """记录 structured() 最终绑定参数的假模型"""

    def __init__(self, **kwargs) -> None:
        self.extra_body = kwargs.get("extra_body")
        self.structured_kwargs = None

    def model_copy(self, update=None):
        data = {"extra_body": getattr(self, "extra_body", None)}
        data.update(update or {})
        return StructuredStub(**data)

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = kwargs
        return self


def test_structured_disables_thinking_and_uses_function_calling(monkeypatch):
    """DeepSeek 思考模式不支持强制 tool_choice，结构化调用必须关闭思考"""
    monkeypatch.setattr(llm_module, "init_chat_model", lambda **kwargs: StructuredStub())
    monkeypatch.setattr(llm_module.settings, "LLM_STRUCTURED_DISABLE_THINKING", True)
    service = llm_module.LLMService()

    runnable = service.structured(dict)

    assert runnable.extra_body == {"thinking": {"type": "disabled"}}
    assert runnable.structured_kwargs["method"] == "function_calling"


def test_structured_keeps_thinking_when_switch_off(monkeypatch):
    """关掉开关时不应注入 thinking 参数，但 function_calling 仍须显式指定"""
    monkeypatch.setattr(llm_module, "init_chat_model", lambda **kwargs: StructuredStub())
    monkeypatch.setattr(llm_module.settings, "LLM_STRUCTURED_DISABLE_THINKING", False)
    service = llm_module.LLMService()

    runnable = service.structured(dict)

    assert runnable.extra_body is None
    assert runnable.structured_kwargs["method"] == "function_calling"
