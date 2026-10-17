"""
OCR 解析器（扫描版 PDF / 图片）

修复点：与 PaddleOCR 3.x 的 API 差异统一交给 ``ocr_engine`` 处理；
重依赖（PyMuPDF / PIL / paddleocr）全部延迟导入，避免“没装 OCR 就起不了服务”。
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

from app.parsers.base import BaseParser, ParseResult
from app.parsers.ocr_engine import ocr_image
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OCRParser(BaseParser):
    """OCR 解析器"""

    SUPPORTED_EXTENSIONS = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff"]
    IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext in self.IMAGE_EXTENSIONS:
            return self._parse_image(file_path)
        if ext == "pdf":
            return self._parse_pdf(file_path)
        logger.error("OCR 不支持的文件类型: %s", ext)
        return []

    def _parse_image(self, image_path: str) -> List[ParseResult]:
        logger.info("OCR 解析图片: %s", image_path)
        try:
            lines = ocr_image(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR 解析图片失败: %s", exc)
            return []

        text = "\n".join(lines).strip()
        if not text:
            logger.warning("OCR 结果为空: %s", image_path)
            return []
        return [ParseResult(text=text, page_num=0, metadata={"parser": "ocr", "type": "image"})]

    def _parse_pdf(self, pdf_path: str) -> List[ParseResult]:
        logger.info("OCR 解析 PDF: %s", pdf_path)
        try:
            from app.parsers.complex_pdf_parser import _import_fitz

            fitz = _import_fitz()
        except Exception as exc:  # noqa: BLE001
            logger.error("无法加载 PyMuPDF: %s", exc)
            return []

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("打开 PDF 失败: %s", exc)
            return []

        results: List[ParseResult] = []
        try:
            total = len(document)
            logger.info("PDF 共 %s 页，开始逐页 OCR", total)
            for page_index in range(total):
                page = document[page_index]
                tmp_path: Optional[str] = None
                try:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(pixmap.tobytes("png"))
                        tmp_path = tmp.name
                    text = "\n".join(ocr_image(tmp_path)).strip()
                    if text:
                        results.append(
                            ParseResult(
                                text=text,
                                page_num=page_index + 1,
                                metadata={"parser": "ocr", "type": "pdf"},
                            )
                        )
                        logger.info("第 %s/%s 页 OCR 完成（%s 字符）", page_index + 1, total, len(text))
                except Exception as exc:  # noqa: BLE001
                    logger.error("第 %s 页 OCR 失败: %s", page_index + 1, exc)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
        finally:
            document.close()

        return results

    def supported_extensions(self) -> List[str]:
        return self.SUPPORTED_EXTENSIONS


_ocr_parser: Optional[OCRParser] = None


def get_ocr_parser() -> OCRParser:
    global _ocr_parser
    if _ocr_parser is None:
        _ocr_parser = OCRParser()
    return _ocr_parser
