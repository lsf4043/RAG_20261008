"""联网检索：查询清洗、解析、相关性过滤、去重、失败降级、开关"""
import pytest

from app.config import settings
from app.core import web_search
from app.core.web_search import (
    WebResult,
    WebSearchService,
    _clean_html,
    query_ngrams,
    rank_results,
    sanitize_query,
)


def test_sanitize_query_strips_question_words():
    """实测：Bing 搜“宝鸡市市长是谁”返回城市概况，搜“宝鸡市市长”才能命中政府网站"""
    assert sanitize_query("宝鸡市市长是谁") == "宝鸡市市长"
    assert sanitize_query("周原镇党委书记是谁？") == "周原镇党委书记"
    assert sanitize_query("请问宝鸡有啥好吃的呢") == "宝鸡有啥好吃的"
    assert sanitize_query("Milvus 是什么") == "Milvus"


def test_sanitize_query_keeps_original_when_too_short():
    assert sanitize_query("为什么") == "为什么"  # 清洗后为空，退回原句
    assert sanitize_query("") == ""


def test_ngrams_cover_chinese_and_latin():
    grams = query_ngrams("宝鸡市市长")
    assert "宝鸡" in grams and "市长" in grams
    assert query_ngrams("Milvus 索引")[0] == "milvus"


def test_rank_results_drops_dictionary_noise():
    """Bing 把“周原镇党委书记”拆成“周”字去查汉字，这类结果必须被过滤"""
    results = [
        WebResult("周（汉语汉字）_百度百科", "https://baike.baidu.com/zhou", "周，读音zhōu"),
        WebResult("周原镇党委书记冯永平主持会议", "https://example.com/a", "周原镇党委书记 冯永平"),
    ]

    ranked = rank_results(results, "周原镇党委书记")

    assert ranked[0].url == "https://example.com/a"
    assert all("汉语汉字" not in item.title for item in ranked)


def test_rank_results_keeps_all_when_nothing_matches():
    results = [WebResult("别的东西", "https://example.com/x", "无关")]
    assert rank_results(results, "宝鸡市市长") == results, "全都沾不上边时不要清空"


DDG_HTML = """
<div class="result results_links">
  <h2><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">宝鸡美食<b>推荐</b></a></h2>
  <a class="result__snippet" href="#">宝鸡的岐山臊子面 &amp; 擀面皮都很有名</a>
</div>
<div class="result results_links">
  <h2><a rel="nofollow" class="result__a" href="https://example.com/b">第二篇</a></h2>
  <a class="result__snippet" href="#">第二段摘要</a>
</div>
"""


@pytest.fixture(autouse=True)
def reset_service():
    web_search.reset_web_search_service()
    yield
    web_search.reset_web_search_service()


def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", False)
    service = WebSearchService(provider="duckduckgo")

    assert service.is_enabled() is False
    assert service.search("宝鸡美食") == []


def test_tavily_without_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    service = WebSearchService(provider="tavily", api_key="")

    assert service.is_enabled() is False
    assert service.search("x") == []


def test_parse_duckduckgo_html_unwraps_urls_and_cleans_text():
    service = WebSearchService(provider="duckduckgo")

    results = service._parse_ddg_html(DDG_HTML, limit=5)

    assert len(results) == 2
    assert results[0].url == "https://example.com/a?x=1", "DDG 的跳转链接需要还原"
    assert results[0].title == "宝鸡美食 推荐", "HTML 标签要被清掉"
    assert "岐山臊子面" in results[0].snippet and "&" in results[0].snippet
    assert results[1].url == "https://example.com/b"


def test_parse_so_html_extracts_title_url_snippet():
    html = """
    <li class="res-list">
      <style>.x{color:red}</style>
      <h3 class="res-title"><a href="https://example.com/a" target="_blank">周原镇党委书记<b>会议</b></a></h3>
      <p class="res-desc">周原镇党委书记冯永平主持会议</p>
    </li>
    <li class="res-list">
      <h3 class="res-title"><a href="https://example.com/b">第二条</a></h3>
      <div class="res-rich">没有标准摘要块时的整块文本</div>
    </li>
    """
    service = WebSearchService(provider="so")

    results = service._parse_so_html(html, limit=5)

    assert len(results) == 2
    assert results[0].title == "周原镇党委书记 会议", "标签要清掉"
    assert results[0].url == "https://example.com/a"
    assert "冯永平" in results[0].snippet
    assert results[1].url == "https://example.com/b"
    assert results[1].snippet, "没有 res-desc 时应回退到整块文本"


def test_image_aggregator_results_are_filtered(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    service = WebSearchService(provider="so")
    service._search_so = lambda q, n: [
        WebResult("宝鸡市市长_360图片", "https://image.so.com/i?q=x", "图片"),
        WebResult("宝鸡市人民政府 市长", "https://example.com/a", "市长 牛恺"),
    ]

    results = service.search("宝鸡市市长是谁", max_results=3)

    assert [r.url for r in results] == ["https://example.com/a"]


def test_search_retries_with_raw_query_when_sanitized_empty(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    service = WebSearchService(provider="so")
    calls = []

    def fake(q, n):
        calls.append(q)
        return [] if q == "宝鸡市市长" else [WebResult("T", "https://example.com/x", "s")]

    service._search_so = fake

    results = service.search("宝鸡市市长是谁", max_results=3)

    assert calls == ["宝鸡市市长", "宝鸡市市长是谁"]
    assert results


def test_deepseek_search_builds_payload_and_maps_summary(monkeypatch):
    """DeepSeek 官方 Search API：ori_question 传原问句，snippet 用 600~800 字的 summary"""
    from app.core import web_search as ws_module

    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    def fake_post(url, headers, payload, timeout):
        captured.update(url=url, headers=headers, payload=payload)
        return {
            "results": [
                {
                    "title": "牛恺当选宝鸡市市长",
                    "url": "https://example.com/a",
                    "snippet": "短摘要",
                    "summary": "长摘要内容" * 20,
                    "content": "正文内容",
                    "rerank_score": 2.4,
                    "authority": 3,
                },
                {"title": "无链接", "url": "", "snippet": "应被丢弃"},
            ]
        }

    monkeypatch.setattr(ws_module, "_post_json", fake_post)
    service = ws_module.WebSearchService(provider="deepseek")

    results = service.search("宝鸡市市长是谁", max_results=5)

    assert captured["url"].endswith("/v1/tools/search")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["ori_question"] == "宝鸡市市长是谁", "原问句要一并传给搜索引擎"
    assert captured["payload"]["query"] == "宝鸡市市长", "query 用清洗后的关键词"
    assert captured["payload"]["count"] == 5
    assert captured["payload"]["need_content"] is True

    assert [r.url for r in results] == ["https://example.com/a"]
    assert results[0].snippet.startswith("长摘要内容"), "优先用 summary（给模型读的长摘要）"


def test_deepseek_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")

    service = WebSearchService(provider="deepseek")

    assert service.is_enabled() is False
    assert service.search("问题") == []


def test_deepseek_http_error_returns_empty(monkeypatch):
    from app.core import web_search as ws_module

    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "sk-test")

    def boom(url, headers, payload, timeout):
        raise RuntimeError("HTTP 402: insufficient balance")

    monkeypatch.setattr(ws_module, "_post_json", boom)
    service = ws_module.WebSearchService(provider="deepseek")

    assert service.search("问题") == []


@pytest.mark.parametrize("q", ["宝鸡有啥好吃的", "周原镇党委书记是谁"])
def test_search_dedupes_and_limits(monkeypatch, q):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    service = WebSearchService(provider="so")
    service._search_so = lambda query, n: [
        WebResult("A", "https://a.com", q),
        WebResult("A2", "https://a.com", "重复链接"),
        WebResult("B", "https://b.com", q),
    ]

    results = service.search(q, max_results=2)

    assert [r.url for r in results] == ["https://a.com", "https://b.com"]


def test_network_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    service = WebSearchService(provider="duckduckgo")

    def boom(q, n):
        raise RuntimeError("网络不可用")

    service._search_duckduckgo = boom

    assert service.search("问题") == [], "任何异常都不能抛给问答主流程"


def test_empty_query_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
    service = WebSearchService(provider="duckduckgo")
    assert service.search("   ") == []


def test_clean_html_handles_entities():
    assert _clean_html("<b>a</b> &amp; <i>b</i>\n\n c") == "a & b c"
