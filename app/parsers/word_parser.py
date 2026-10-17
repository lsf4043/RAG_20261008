"""
Word 文档解析器

修复点：
- **.docx 现在会解析表格**（旧实现只读段落，表格内容全部丢失），并保持
  段落 / 表格在正文中的原始顺序；
- **.doc 的 win32com 分支调用 CoInitialize**：解析发生在线程池里，COM 未初始化会
  直接抛 "CoInitialize has not been called"；
- 全部重依赖延迟导入，缺库时给出可执行的修复提示而不是在 import 阶段崩溃。
"""
from __future__ import annotations

import os
from typing import List, Optional

from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WordParser(BaseParser):
    """Word 解析器（.doc / .docx，不依赖 LibreOffice）"""

    SUPPORTED_EXTENSIONS = ["doc", "docx"]

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        logger.info("解析 Word 文档: %s", file_path)

        try:
            if ext == "docx":
                text = self._parse_docx(file_path)
            elif ext == "doc":
                text = self._parse_doc(file_path)
            else:
                logger.error("不支持的 Word 格式: %s", ext)
                return []
        except Exception as exc:  # noqa: BLE001
            logger.error("Word 文档解析失败: %s", exc)
            return []

        if not text.strip():
            logger.warning("Word 文档解析结果为空: %s", file_path)
            return []

        return [
            ParseResult(
                text=text,
                page_num=0,
                metadata={"parser": "word", "format": ext},
            )
        ]

    # ---------- .docx ----------
    def _parse_docx(self, file_path: str) -> str:
        try:
            import docx  # python-docx
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise RuntimeError("需要安装 python-docx：pip install python-docx") from exc

        document = docx.Document(file_path)
        parts: List[str] = []

        # 按文档流顺序遍历段落与表格
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if not text:
                    continue
                # 保留标题层级，便于后续章节分割
                style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
                if style.startswith("heading") or style.startswith("标题"):
                    parts.append(f"# {text}")
                else:
                    parts.append(text)
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)
                markdown = self._table_to_markdown(table)
                if markdown:
                    parts.append(markdown)

        return "\n\n".join(parts)

    @staticmethod
    def _table_to_markdown(table) -> str:
        rows: List[List[str]] = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cells.append(" ".join(cell.text.split()))
            if any(cells):
                rows.append(cells)
        if not rows:
            return ""

        width = max(len(row) for row in rows)
        rows = [(row + [""] * width)[:width] for row in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # ---------- .doc ----------
    def _parse_doc(self, file_path: str) -> str:
        for loader in (self._parse_doc_with_antiword, self._parse_doc_with_textract, self._parse_doc_with_word):
            try:
                text = loader(file_path)
            except ImportError:
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s 解析 .doc 失败: %s", loader.__name__, exc)
                continue
            if text and text.strip():
                return text

        raise RuntimeError(
            "无法解析 .doc 老格式文件，请任选一种方案：\n"
            "1) pip install antiword-python\n"
            "2) 用 Word/WPS 另存为 .docx 后上传\n"
            "3) 安装 LibreOffice 后由 unstructured 解析"
        )

    @staticmethod
    def _parse_doc_with_antiword(file_path: str) -> str:
        import antiword  # type: ignore[import-not-found]

        return antiword.read_doc(file_path)

    @staticmethod
    def _parse_doc_with_textract(file_path: str) -> str:
        import textract  # type: ignore[import-not-found]

        return textract.process(file_path).decode("utf-8", errors="ignore")

    @staticmethod
    def _parse_doc_with_word(file_path: str) -> str:
        if os.name != "nt":
            raise ImportError("win32com 仅在 Windows 可用")
        import pythoncom  # type: ignore[import-not-found]
        import win32com.client  # type: ignore[import-not-found]

        pythoncom.CoInitialize()  # 线程池里必须显式初始化 COM
        word = None
        document = None
        try:
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            document = word.Documents.Open(file_path, ReadOnly=True)
            return str(document.Content.Text or "")
        finally:
            try:
                if document is not None:
                    document.Close(False)
            finally:
                if word is not None:
                    word.Quit()
                pythoncom.CoUninitialize()

    def supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


_word_parser: Optional[WordParser] = None


def get_word_parser() -> WordParser:
    global _word_parser
    if _word_parser is None:
        _word_parser = WordParser()
    return _word_parser
