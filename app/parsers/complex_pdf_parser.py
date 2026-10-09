"""
复杂 PDF 解析器
支持文本、表格、图片混合的 PDF 文档
"""
import os
import io
import tempfile
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from paddleocr import PaddleOCR

from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ComplexPDFParser(BaseParser):
    """复杂 PDF 解析器，支持文本+表格+图片混合"""

    SUPPORTED_EXTENSIONS = ['pdf']

    def __init__(self):
        """初始化解析器"""
        logger.info("正在初始化复杂 PDF 解析器...")
        self.ocr = PaddleOCR(lang="ch", use_angle_cls=True, show_log=False)
        logger.info("复杂 PDF 解析器初始化完成")

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        """
        解析复杂 PDF 文档

        Args:
            file_path: 文件路径

        Returns:
            解析结果列表
        """
        logger.info(f"开始解析复杂 PDF: {file_path}")

        try:
            results = []

            # 1. 使用 pdfplumber 提取表格
            logger.info("正在提取表格...")
            tables_content = self._extract_tables(file_path)

            # 2. 使用 PyMuPDF 检测并 OCR 图片区域
            logger.info("正在提取图片中的文字...")
            images_content = self._extract_images_text(file_path)

            # 3. 使用 PyMuPDF 提取普通文本
            logger.info("正在提取文本...")
            text_content = self._extract_text(file_path)

            # 4. 合并所有内容
            all_content = []

            for item in tables_content:
                all_content.append({
                    "type": "table",
                    "page": item["page"],
                    "text": item["text"],
                    "priority": 1  # 表格优先级高
                })

            for item in images_content:
                all_content.append({
                    "type": "image",
                    "page": item["page"],
                    "text": item["text"],
                    "priority": 2
                })

            for item in text_content:
                all_content.append({
                    "type": "text",
                    "page": item["page"],
                    "text": item["text"],
                    "priority": 3
                })

            # 5. 按页面分组，合并内容
            page_contents = {}
            for item in all_content:
                page_num = item["page"]
                if page_num not in page_contents:
                    page_contents[page_num] = []
                page_contents[page_num].append(item)

            # 6. 组装最终结果
            for page_num in sorted(page_contents.keys()):
                items = page_contents[page_num]
                # 按优先级排序
                items.sort(key=lambda x: x["priority"])

                # 合并该页所有内容
                page_text_parts = []
                for item in items:
                    if item["text"].strip():
                        if item["type"] == "table":
                            page_text_parts.append(f"[表格内容]\n{item['text']}")
                        elif item["type"] == "image":
                            page_text_parts.append(f"[图片文字]\n{item['text']}")
                        else:
                            page_text_parts.append(item["text"])

                page_text = "\n\n".join(page_text_parts)

                if page_text.strip():
                    results.append(ParseResult(
                        text=page_text,
                        page_num=page_num,
                        metadata={
                            "parser": "complex_pdf",
                            "has_table": any(i["type"] == "table" for i in items),
                            "has_image": any(i["type"] == "image" for i in items)
                        }
                    ))

            logger.info(f"复杂 PDF 解析完成，共 {len(results)} 页")
            return results

        except Exception as e:
            logger.error(f"复杂 PDF 解析失败: {str(e)}")
            return []

    def _extract_tables(self, file_path: str) -> List[Dict]:
        """使用 pdfplumber 提取表格"""
        tables_content = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # 将表格转换为 Markdown 格式
                            table_md = self._table_to_markdown(table)
                            if table_md.strip():
                                tables_content.append({
                                    "page": page_num + 1,
                                    "text": table_md
                                })
                                logger.info(f"第 {page_num + 1} 页提取到表格")

        except Exception as e:
            logger.error(f"表格提取失败: {str(e)}")

        return tables_content

    def _table_to_markdown(self, table: List[List]) -> str:
        """将表格转换为 Markdown 格式"""
        if not table:
            return ""

        lines = []

        # 处理表头
        header = table[0]
        header = [str(cell) if cell else "" for cell in header]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # 处理数据行
        for row in table[1:]:
            row = [str(cell) if cell else "" for cell in row]
            # 确保列数一致
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[:len(header)]) + " |")

        return "\n".join(lines)

    def _extract_images_text(self, file_path: str) -> List[Dict]:
        """提取 PDF 中图片的文字"""
        images_content = []

        try:
            pdf_document = fitz.open(file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # 获取页面中的图片
                image_list = page.get_images(full=True)

                if not image_list:
                    continue

                # 渲染页面为图片进行 OCR（检测是否有图片中的文字）
                mat = fitz.Matrix(2, 2)  # 放大2倍
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # 保存临时图片
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp.write(img_data)

                try:
                    # OCR 识别
                    ocr_result = self.ocr.ocr(tmp_path, cls=True)
                    text_lines = self._extract_ocr_text(ocr_result)

                    if text_lines:
                        full_text = '\n'.join(text_lines)
                        # 过滤掉已经通过普通文本提取的内容（避免重复）
                        # 这里简单处理，实际可以通过位置对比来去重
                        images_content.append({
                            "page": page_num + 1,
                            "text": full_text
                        })
                        logger.info(f"第 {page_num + 1} 页提取到图片文字")

                finally:
                    os.unlink(tmp_path)

            pdf_document.close()

        except Exception as e:
            logger.error(f"图片文字提取失败: {str(e)}")

        return images_content

    def _extract_ocr_text(self, result) -> List[str]:
        """从 OCR 结果提取文本"""
        text_lines = []

        for line in result:
            if line:
                for word_info in line:
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    if confidence > 0.5:  # 置信度过滤
                        text_lines.append(text)

        return text_lines

    def _extract_text(self, file_path: str) -> List[Dict]:
        """使用 PyMuPDF 提取普通文本"""
        text_content = []

        try:
            pdf_document = fitz.open(file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text("text")

                if text.strip():
                    text_content.append({
                        "page": page_num + 1,
                        "text": text.strip()
                    })

            pdf_document.close()

        except Exception as e:
            logger.error(f"文本提取失败: {str(e)}")

        return text_content

    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        return self.SUPPORTED_EXTENSIONS


# 全局解析器实例（延迟初始化）
_complex_pdf_parser = None


def get_complex_pdf_parser() -> ComplexPDFParser:
    """获取复杂 PDF 解析器实例"""
    global _complex_pdf_parser
    if _complex_pdf_parser is None:
        _complex_pdf_parser = ComplexPDFParser()
    return _complex_pdf_parser
