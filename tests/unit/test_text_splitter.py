"""文本分割：chunk_id 唯一性、overlap 防御、章节切分"""
from app.utils.text_splitter import TextSplitter


def test_chunk_ids_are_globally_unique_across_pages():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
    pages = ["第一页内容。" * 40, "第二页内容。" * 40, "第三页内容。" * 40]

    chunks = []
    for page_no, text in enumerate(pages, start=1):
        # 关键回归点：跨页时 start_chunk_id 必须接续
        page_chunks = splitter.split_text(text, page_no, start_chunk_id=len(chunks))
        chunks.extend(page_chunks)

    assert len(chunks) > 6
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 必须全局唯一（旧实现按页从 0 开始会撞号）"
    assert ids == sorted(ids)


def test_split_texts_numbers_chunks_continuously():
    splitter = TextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_texts(["甲" * 500, "乙" * 500], page_nums=[1, 2])
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert {c.page_num for c in chunks} == {1, 2}


def test_overlap_ge_chunk_size_is_clamped():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=500)
    assert splitter.chunk_overlap < splitter.chunk_size

    chunks = splitter.split_text("内容。" * 500, page_num=1)
    assert chunks
    for chunk in chunks:
        assert len(chunk.content) <= splitter.chunk_size * 2


def test_empty_text_returns_no_chunks():
    splitter = TextSplitter()
    assert splitter.split_text("   \n  ", 1) == []


def test_markdown_title_is_detected_and_not_duplicated_in_content():
    splitter = TextSplitter(chunk_size=200, chunk_overlap=20, enable_chapter_split=False)
    chunks = splitter.split_text("# 安装说明\n\n先安装依赖，然后启动服务。", page_num=1)

    assert len(chunks) == 1
    assert chunks[0].title == "安装说明"
    assert "安装说明" not in chunks[0].content


def test_chapter_splitter_recognizes_chinese_headings():
    from app.utils.text_splitter import ChapterTextSplitter

    splitter = ChapterTextSplitter(min_chunk_size=10)
    splits = splitter.split_text("第一章 概述\n内容A\n第二章 部署\n内容B")

    titles = [title for title, _ in splits]
    assert titles == ["第一章 概述", "第二章 部署"]


def test_leading_heading_becomes_chunk_title():
    splitter = TextSplitter(chunk_size=300, chunk_overlap=20, enable_chapter_split=False)
    chunks = splitter.split_text("第一章 部署说明\n\n先安装依赖，再启动服务。", page_num=1)

    assert chunks[0].title == "第一章 部署说明"
