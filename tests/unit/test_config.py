"""配置校验"""
import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None, DASHSCOPE_API_KEY="x", DEEPSEEK_API_KEY="y")
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
    assert settings.cors_allow_credentials is False  # "*" 时自动关闭 credentials
    assert settings.langsmith_enabled is False


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
