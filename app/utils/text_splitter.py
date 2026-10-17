"""
文本分割工具

本次修复：
1. **chunk_id 全局唯一**：旧实现每页调用 ``split_text`` 都从 0 开始编号，
   而文档去重 key 用的是 ``document_id + chunk_id``，于是不同页的 chunk 撞号，
   RRF 融合时会把不同内容当成同一篇。现在由调用方传入 ``start_chunk_id`` 连续编号。
2. **递归分割交给 LangChain**：删掉手写的 RecursiveCharacterTextSplitter（其中
   ``range(0, n, chunk_size - overlap)`` 在 overlap>=chunk_size 时会踩零步长死循环），
   直接复用经过充分测试的 ``langchain_text_splitters.RecursiveCharacterTextSplitter``。
3. 标题识别收紧：只有真正的标题（Markdown / 编号标题 / 短行）才参与“标题重复两次”
   的权重增强，避免把正文首句复制进向量文本制造噪声。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter as LangChainRecursiveSplitter

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 换行
    "。", "！", "？", "；",
    ". ", "! ", "? ", "; ",
    "，", ", ",
    " ",
    "",
]


@dataclass
class TextChunk:
    """文本块"""

    title: str
    content: str
    chunk_id: int
    page_num: int = 0


class ChapterTextSplitter:
    """按章节标题切分（识别不到章节时返回整段）"""

    CHAPTER_PATTERNS = [
        r"^第[一二三四五六七八九十百千万零\d]+[章节篇部]\s*.+",
        r"^[一二三四五六七八九十百千万]+[、.．]\s*.+",
        r"^\d+[、.．]\s*.+",
        r"^#+\s+.+",
        r"^[A-Z][、.．]\s*.+",
        r"^[（(][一二三四五六七八九十\d]+[）)]\s*.+",
    ]

    def __init__(self, min_chunk_size: int = 100) -> None:
        self.min_chunk_size = min_chunk_size
        self.chapter_patterns = [re.compile(p, re.MULTILINE) for p in self.CHAPTER_PATTERNS]

    def _find_chapters(self, text: str) -> List[Tuple[int, str]]:
        chapters: List[Tuple[int, str]] = []
        for pattern in self.chapter_patterns:
            for match in pattern.finditer(text):
                chapters.append((match.start(), match.group().strip()))
        chapters.sort(key=lambda item: item[0])

        unique: List[Tuple[int, str]] = []
        last_pos = -1
        for pos, title in chapters:
            # 注意：不能写成 pos > last_pos + 10 —— last_pos 初值为 -1，
            # 会让“位于正文开头（pos=0）的章节标题”被判为重复而丢弃。
            if last_pos < 0 or pos > last_pos + 10:
                unique.append((pos, title))
                last_pos = pos
        return unique

    def split_text(self, text: str) -> List[Tuple[str, str]]:
        if not text or not text.strip():
            return []

        chapters = self._find_chapters(text)
        if not chapters:
            return [("", text)]

        result: List[Tuple[str, str]] = []
        for i, (pos, title) in enumerate(chapters):
            end = chapters[i + 1][0] if i + 1 < len(chapters) else len(text)
            content = text[pos:end].strip()

            lines = content.split("\n", 1)
            content = lines[1].strip() if len(lines) > 1 else ""
            if content:
                result.append((title, content))

        if chapters and chapters[0][0] > 0:
            intro = text[: chapters[0][0]].strip()
            if len(intro) >= self.min_chunk_size:
                result.insert(0, ("引言", intro))
        return result


class TextSplitter:
    """文本分割器（章节切分 + 递归字符切分）"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_chapter_split: bool = True,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # 防御：overlap 必须小于 chunk_size
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(
                "chunk_overlap(%s) >= chunk_size(%s)，已自动下调 overlap",
                self.chunk_overlap,
                self.chunk_size,
            )
            self.chunk_overlap = max(0, self.chunk_size // 10)

        self.enable_chapter_split = enable_chapter_split
        self.recursive_splitter = LangChainRecursiveSplitter(
            separators=DEFAULT_SEPARATORS,
            keep_separator=True,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        self.chapter_splitter = ChapterTextSplitter(min_chunk_size=max(50, self.chunk_size // 2))

    # ---------- 标题识别 ----------
    def _extract_title(self, text: str) -> Tuple[str, str]:
        lines = text.strip().split("\n")
        if not lines:
            return "", ""

        title = ""
        content_start = 0
        for i, raw_line in enumerate(lines[:3]):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                content_start = i + 1
                break
            if len(line) < 80 and not line.endswith(("。", "！", "？", "；", "：", ".", "!", "?", ";", ":")):
                if ChapterTextSplitter.CHAPTER_PATTERNS and any(
                    re.match(p, line) for p in ChapterTextSplitter.CHAPTER_PATTERNS
                ):
                    title = line
                    content_start = i + 1
                    break
                if i == 0 and len(line) < 40:
                    title = line
                    content_start = i + 1
                    break

        content = "\n".join(lines[content_start:]).strip()
        return title, content

    # ---------- 分割 ----------
    def split_text(self, text: str, page_num: int = 0, start_chunk_id: int = 0) -> List[TextChunk]:
        """
        分割单个页面的文本

        Args:
            text: 文本
            page_num: 页码
            start_chunk_id: 起始 chunk 编号（跨页连续，保证全局唯一）
        """
        if not text or not text.strip():
            return []

        title, content = self._extract_title(text)
        if not content:
            return []

        chunks: List[TextChunk] = []
        chunk_id = start_chunk_id

        # 1) 章节切分（能识别到多个章节时）
        if self.enable_chapter_split:
            chapter_splits = self.chapter_splitter.split_text(content)
            if len(chapter_splits) > 1:
                for chapter_title, chapter_content in chapter_splits:
                    if len(chapter_content) > self.chunk_size:
                        for sub in self.recursive_splitter.split_text(chapter_content):
                            chunks.append(
                                TextChunk(
                                    title=chapter_title or title,
                                    content=sub,
                                    chunk_id=chunk_id,
                                    page_num=page_num,
                                )
                            )
                            chunk_id += 1
                    else:
                        chunks.append(
                            TextChunk(
                                title=chapter_title or title,
                                content=chapter_content,
                                chunk_id=chunk_id,
                                page_num=page_num,
                            )
                        )
                        chunk_id += 1
                logger.info("章节分割完成，生成 %s 个文本块", len(chunks))
                return chunks

        # 2) 递归字符切分
        for sub in self.recursive_splitter.split_text(content):
            sub = sub.strip()
            if not sub:
                continue
            chunks.append(
                TextChunk(title=title, content=sub, chunk_id=chunk_id, page_num=page_num)
            )
            chunk_id += 1

        logger.info("递归字符分割完成，生成 %s 个文本块（标题: %s）", len(chunks), title[:30])
        return chunks

    def split_texts(
        self, texts: List[str], page_nums: Optional[List[int]] = None
    ) -> List[TextChunk]:
        """分割多个页面，chunk_id 全局连续（推荐 document_service 使用本方法）"""
        all_chunks: List[TextChunk] = []
        for i, text in enumerate(texts):
            page_num = page_nums[i] if page_nums and i < len(page_nums) else 0
            page_chunks = self.split_text(text, page_num, start_chunk_id=len(all_chunks))
            all_chunks.extend(page_chunks)
        logger.info("共生成 %s 个文本块", len(all_chunks))
        return all_chunks


text_splitter = TextSplitter()
