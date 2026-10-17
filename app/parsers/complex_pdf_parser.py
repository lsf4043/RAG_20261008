"""
复杂 PDF 解析器（文本 + 表格 + 图片 OCR）

相对旧实现的三处关键修复：

1. **不再“有图就整页 OCR”**。旧逻辑对“含任意图片的每一页”都按 2 倍渲染整页再 OCR，
   一份带页眉 logo 的 100 页 PDF 会做 100 次整页 OCR（分钟级），而且 OCR 文本会和
   ``page.get_text()`` 已提取的文本重复入库，直接拉低 context_precision。
   现在改为：**只有该页可提取文本量低于阈值（疑似扫描页）时才整页 OCR**，并把与已有
   文本重复的行剔除。
2. **表格不再重复三遍**。用 pdfplumber 的 ``table.bbox`` 把表格区域从 PyMuPDF 的文本块中
   裁掉，表格只以 Markdown 形式出现一次。
3. **依赖延迟导入**：没装 OCR / pdfplumber 时，服务仍能启动并处理其它格式。
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings
from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

BBox = Tuple[float, float, float, float]


def _import_fitz():
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 推荐名称
    except ImportError:  # pragma: no cover - 向后兼容
        import fitz  # type: ignore[no-redef]
    return fitz


def _overlaps(a: BBox, b: BBox, threshold: float = 0.5) -> bool:
    """a 与 b 的交集是否占 a 面积的一半以上"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    return inter / area >= threshold


class ComplexPDFParser(BaseParser):
    """复杂 PDF 解析器"""

    SUPPORTED_EXTENSIONS = ["pdf"]

    def __init__(self) -> None:
        logger.info("复杂 PDF 解析器就绪（OCR 模型按需加载）")

    # ==================== 主流程 ====================
    def parse(self, file_path: str, force_ocr: bool = False, **kwargs) -> List[ParseResult]:
        fitz = _import_fitz()
        logger.info("开始解析 PDF: %s (force_ocr=%s)", file_path, force_ocr)

        tables_by_page = self._extract_tables(file_path)
        results: List[ParseResult] = []

        try:
            document = fitz.open(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("打开 PDF 失败: %s", exc)
            return []

        try:
            for page_index in range(len(document)):
                page = document[page_index]
                page_no = page_index + 1
                tables = tables_by_page.get(page_no, [])
                table_boxes = [bbox for bbox, _ in tables]

                text = self._extract_page_text(page, table_boxes)

                if force_ocr or self._needs_ocr(page, text):
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        text = self._merge_ocr(text, ocr_text)
                        logger.info("第 %s 页使用 OCR 补充文本（%s 字符）", page_no, len(ocr_text))

                parts: List[str] = []
                if text.strip():
                    parts.append(text.strip())
                for _, markdown in tables:
                    parts.append(f"[表格内容]\n{markdown}")

                if not parts:
                    continue

                results.append(
                    ParseResult(
                        text="\n\n".join(parts),
                        page_num=page_no,
                        metadata={
                            "parser": "complex_pdf",
                            "has_table": bool(tables),
                            "used_ocr": bool(force_ocr or self._needs_ocr(page, text)),
                        },
                    )
                )
        finally:
            document.close()

        logger.info("PDF 解析完成，共 %s 页", len(results))
        return results

    # ==================== 文本 / 表格 ====================
    def _extract_page_text(self, page, table_boxes: Sequence[BBox]) -> str:
        """提取正文块（排除表格区域）"""
        try:
            blocks = page.get_text("blocks")
        except Exception as exc:  # noqa: BLE001
            logger.warning("提取页面文本失败: %s", exc)
            return ""

        kept: List[Tuple[float, float, str]] = []
        for block in blocks:
            try:
                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            except (IndexError, TypeError):
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            box = (float(x0), float(y0), float(x1), float(y1))
            if any(_overlaps(box, table_box) for table_box in table_boxes):
                continue
            kept.append((float(y0), float(x0), text.strip()))

        kept.sort(key=lambda item: (item[0], item[1]))
        return "\n".join(item[2] for item in kept)

    def _extract_tables(self, file_path: str) -> Dict[int, List[Tuple[BBox, str]]]:
        """用 pdfplumber 提取表格，并返回 bbox 以便从正文中裁掉"""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("未安装 pdfplumber，跳过表格提取")
            return {}

        tables: Dict[int, List[Tuple[BBox, str]]] = {}
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    if hasattr(page, "find_tables"):
                        found = page.find_tables()
                        for table in found:
                            markdown = self._table_to_markdown(table.extract())
                            if markdown:
                                bbox = tuple(float(v) for v in table.bbox)  # type: ignore[assignment]
                                tables.setdefault(page_index, []).append((bbox, markdown))
                    else:  # 老版本 pdfplumber
                        for raw in page.extract_tables():
                            markdown = self._table_to_markdown(raw)
                            if markdown:
                                tables.setdefault(page_index, []).append(
                                    ((0.0, 0.0, 0.0, 0.0), markdown)
                                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("表格提取失败（忽略表格）: %s", exc)
        return tables

    @staticmethod
    def _table_to_markdown(table: Optional[List[List[Any]]]) -> str:
        if not table or not table[0]:
            return ""
        rows = [[("" if cell is None else str(cell).replace("\n", " ").strip()) for cell in row] for row in table]
        header = rows[0]
        width = len(header)
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for row in rows[1:]:
            row = (row + [""] * width)[:width]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # ==================== OCR ====================
    @staticmethod
    def _needs_ocr(page, extracted_text: str) -> bool:
        """仅在“疑似扫描页”时 OCR：可提取文本很少"""
        if len(extracted_text.strip()) >= settings.OCR_TEXT_THRESHOLD:
            return False
        try:
            has_images = bool(page.get_images(full=True))
        except Exception:  # noqa: BLE001
            has_images = False
        # 文本极少：无论有没有图片都值得 OCR 一次（矢量文字也可能提取失败）
        return has_images or not extracted_text.strip()

    def _ocr_page(self, page) -> str:
        fitz = _import_fitz()
        tmp_path: Optional[str] = None
        try:
            matrix = fitz.Matrix(2, 2)
            pixmap = page.get_pixmap(matrix=matrix)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(pixmap.tobytes("png"))
                tmp_path = tmp.name

            from app.parsers.ocr_engine import ocr_image

            lines = ocr_image(tmp_path)
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            logger.warning("页面 OCR 失败: %s", exc)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _merge_ocr(extracted_text: str, ocr_text: str) -> str:
        """把 OCR 结果合并进正文，剔除与已有文本重复的行"""
        existing = {line.strip() for line in extracted_text.splitlines() if line.strip()}
        existing_blob = extracted_text.replace(" ", "")
        extra: List[str] = []
        for line in ocr_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in existing:
                continue
            if stripped.replace(" ", "") and stripped.replace(" ", "") in existing_blob:
                continue
            extra.append(stripped)
        if not extra:
            return extracted_text
        return (extracted_text + "\n" + "\n".join(extra)).strip()

    def supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


_complex_pdf_parser: Optional[ComplexPDFParser] = None


def get_complex_pdf_parser() -> ComplexPDFParser:
    global _complex_pdf_parser
    if _complex_pdf_parser is None:
        _complex_pdf_parser = ComplexPDFParser()
    return _complex_pdf_parser
