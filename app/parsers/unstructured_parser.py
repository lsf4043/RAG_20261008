"""
Unstructured 通用解析器

修复点：
- **保留页码**：旧实现把所有页塞进一个 ``page_num=0`` 的结果里，导致
  “第 N 页”的引用信息完全丢失；现在按 element.metadata.page_number 聚合。
- **不再硬编码 ``skip_infer_table_types=[]``**（原注释说“跳过表格推理以加快速度”，
  实际传空列表等于全都推理，语义相反）。
- 延迟导入：未安装 unstructured 时不影响服务启动。
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional

from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class UnstructuredParser(BaseParser):
    """Unstructured 通用解析器"""

    SUPPORTED_EXTENSIONS = [
        "pdf", "doc", "docx", "txt", "md", "rtf", "csv", "xls", "xlsx",
        "ppt", "pptx", "eml", "msg", "html", "htm", "xml", "epub", "json", "yaml", "yml",
    ]

    # 纯文本类格式直接读文件：更快、更稳定，且不依赖 unstructured 的临时目录操作
    TEXT_EXTENSIONS = {"txt", "md", "markdown", "csv", "tsv", "json", "yaml", "yml", "log"}

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        logger.info("Unstructured 解析: %s", file_path)
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")

        if ext in self.TEXT_EXTENSIONS:
            return self._read_plain_text(file_path)

        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            logger.error("未安装 unstructured，尝试按纯文本读取: %s", exc)
            return self._read_plain_text(file_path, required=False)

        elements = self._partition(partition, file_path)
        if elements is None:
            # unstructured 失败时再试一次纯文本（例如依赖缺失 / 临时目录不可写）
            return self._read_plain_text(file_path, required=False)

        # 按页码聚合（没有页码的归到第 0 页）
        by_page: Dict[int, List[str]] = OrderedDict()
        for element in elements:
            text = getattr(element, "text", None) or str(element) or ""
            if not text.strip():
                continue
            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None) if metadata else None
            try:
                page = int(page_number) if page_number else 0
            except (TypeError, ValueError):
                page = 0
            by_page.setdefault(page, []).append(text.strip())

        if not by_page:
            logger.warning("Unstructured 解析结果为空: %s", file_path)
            return []

        results: List[ParseResult] = []
        for page in sorted(by_page):
            content = "\n\n".join(by_page[page])
            if content:
                results.append(
                    ParseResult(
                        text=content,
                        page_num=page,
                        metadata={"parser": "unstructured", "element_count": len(by_page[page])},
                    )
                )

        logger.info(
            "Unstructured 解析完成，%s 页 / %s 字符",
            len(results),
            sum(len(item.text) for item in results),
        )
        return results

    def _read_plain_text(self, file_path: str, required: bool = True) -> List[ParseResult]:
        """直接读取纯文本（多编码兜底）"""
        try:
            with open(file_path, "rb") as handle:
                raw = handle.read()
        except OSError as exc:
            logger.error("读取文件失败: %s", exc)
            return []

        text = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            logger.warning("无法识别文件编码: %s", file_path)
            return []

        if not text.strip():
            logger.warning("文件内容为空: %s", file_path)
            return []

        return [
            ParseResult(
                text=text,
                page_num=0,
                metadata={"parser": "plain_text", "encoding": encoding if text else None},
            )
        ]

    def _partition(self, partition, file_path: str):
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        try:
            return partition(filename=file_path, strategy="fast")
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "soffice" in message.lower() or "libreoffice" in message.lower():
                logger.error(
                    "解析 .%s 需要 LibreOffice：请安装 LibreOffice，或先把文件另存为 .docx/.txt",
                    ext,
                )
                return None
            logger.warning("fast 策略失败（%s），改用 auto 策略重试", exc)

        try:
            # auto 可能触发版面/表格模型下载，务必兜底
            return partition(filename=file_path, strategy="auto")
        except Exception as exc:  # noqa: BLE001
            logger.error("Unstructured 解析失败: %s", exc)
            return None

    def supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


_unstructured_parser: Optional[UnstructuredParser] = None


def get_unstructured_parser() -> UnstructuredParser:
    global _unstructured_parser
    if _unstructured_parser is None:
        _unstructured_parser = UnstructuredParser()
    return _unstructured_parser
