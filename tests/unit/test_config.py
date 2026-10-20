"""配置校验"""
import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None, DASHSCOPE_API_KEY="x", DEEPSEEK_API_KEY="y")
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
    assert settings.cors_allow_credentials is False  # "*" 时自动关闭 credentials
    assert settings.langsmith_enabled is False


def test_crag_is_disabled_by_default():
    """CRAG 默认关闭：它的过滤与重排/相似度门控重叠，纠正分支在单域库里无效"""
    assert Settings.model_fields["CRAG_ENABLED"].default is False


def test_llm_fallback_is_enabled_by_default():
    """默认策略：检索不到 → 用通用知识作答（并在响应里标注）"""
    assert Settings.model_fields["ALLOW_LLM_FALLBACK"].default is True


def test_router_is_enabled_by_default():
    assert Settings.model_fields["ROUTER_ENABLED"].default is True


def test_retrieval_thresholds_are_calibrated_defaults():
    """余弦只负责短路“明显无关”；“有没有答案”交给模型声明，不再靠重排阈值"""
    assert Settings.model_fields["RETRIEVAL_MIN_SCORE"].default == 0.25
    assert Settings.model_fields["RERANK_MIN_SCORE"].default == 0.0


def test_web_search_defaults_to_deepseek():
    """联网默认走 DeepSeek 官方 Search API（同一把 Key，结构化结果）"""
    assert Settings.model_fields["WEB_SEARCH_ENABLED"].default is True
    assert Settings.model_fields["WEB_SEARCH_PROVIDER"].default == "deepseek"
    assert Settings.model_fields["WEB_SEARCH_MAX_RESULTS"].default == 5


def test_web_search_provider_is_validated():
    import pytest

    with pytest.raises(ValidationError):
        Settings(_env_file=None, WEB_SEARCH_PROVIDER="google")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, WEB_SEARCH_ENABLED=True, WEB_SEARCH_PROVIDER="tavily")


def test_overlap_not_less_than_chunk_size_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, CHUNK_SIZE=100, CHUNK_OVERLAP=100)


def test_both_retrieval_weights_zero_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, VECTOR_WEIGHT=0, BM25_WEIGHT=0)


def test_missing_keys_raise_actionable_error_on_demand():
    from app.config import ConfigurationError

    settings = Settings(_env_file=None, DASHSCOPE_API_KEY="", DEEPSEEK_API_KEY="")
    with pytest.raises(ConfigurationError):
        settings.require_dashscope_key()
    with pytest.raises(ConfigurationError):
        settings.require_deepseek_key()


def test_cors_origins_parsed():
    settings = Settings(_env_file=None, CORS_ORIGINS="http://a.com,http://b.com")
    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]
    assert settings.cors_allow_credentials is True
