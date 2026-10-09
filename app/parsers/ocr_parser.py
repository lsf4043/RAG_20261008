"""
OCR 文档解析器
使用 PaddleOCR 处理扫描版 PDF 和图片
"""
import os
import io
import tempfile
from typing import List

import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR

from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OCRParser(BaseParser):
    """OCR 文档解析器"""

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = [
        'pdf',      # PDF (扫描型)
        'png',      # PNG 图片
        'jpg',      # JPG 图片
        'jpeg',     # JPEG 图片
        'bmp',      # BMP 图片
        'tiff',     # TIFF 图片
    ]

    # 图片扩展名
    IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']

    def __init__(self):
        """初始化 OCR 引擎"""
        logger.info("正在初始化 PaddleOCR...")
        self.ocr = PaddleOCR(lang="ch", use_angle_cls=True, show_log=False)
        logger.info("PaddleOCR 初始化完成")

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        """
        解析文档

        Args:
            file_path: 文件路径

        Returns:
            解析结果列表
        """
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')

        if ext in self.IMAGE_EXTENSIONS:
            return self._parse_image(file_path)
        elif ext == 'pdf':
            return self._parse_pdf(file_path)
        else:
            logger.error(f"不支持的文件类型: {ext}")
            return []

    def _parse_image(self, image_path: str) -> List[ParseResult]:
        """解析图片"""
        logger.info(f"开始 OCR 解析图片: {image_path}")

        try:
            result = self.ocr.ocr(image_path, cls=True)

            text_lines = self._extract_text_from_ocr_result(result)
            full_text = '\n'.join(text_lines)

            if not full_text.strip():
                logger.warning(f"OCR 解析结果为空: {image_path}")
                return []

            parse_result = ParseResult(
                text=full_text,
                page_num=0,
                metadata={
                    "parser": "ocr",
                    "type": "image"
                }
            )

            logger.info(f"OCR 解析完成，提取 {len(full_text)} 字符")
            return [parse_result]

        except Exception as e:
            logger.error(f"OCR 解析图片失败: {str(e)}")
            return []

    def _parse_pdf(self, pdf_path: str) -> List[ParseResult]:
        """解析 PDF"""
        logger.info(f"开始 OCR 解析 PDF: {pdf_path}")

        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            logger.info(f"PDF 共 {total_pages} 页")

            results = []

            for page_num in range(total_pages):
                logger.info(f"正在处理第 {page_num + 1}/{total_pages} 页")

                page = pdf_document[page_num]

                # 渲染页面为图片
                mat = fitz.Matrix(3, 3)  # 放大3倍提高OCR精度
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # 保存临时图片
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp_path)

                try:
                    # OCR 识别
                    ocr_result = self.ocr.ocr(tmp_path, cls=True)
                    text_lines = self._extract_text_from_ocr_result(ocr_result)
                    full_text = '\n'.join(text_lines)

                    if full_text.strip():
                        results.append(ParseResult(
                            text=full_text,
                            page_num=page_num + 1,
                            metadata={
                                "parser": "ocr",
                                "type": "pdf"
                            }
                        ))
                        logger.info(f"第 {page_num + 1} 页提取完成，共 {len(full_text)} 字符")

                finally:
                    # 删除临时文件
                    os.unlink(tmp_path)

            pdf_document.close()
            logger.info(f"PDF OCR 解析完成，共处理 {len(results)} 页")
            return results

        except Exception as e:
            logger.error(f"OCR 解析 PDF 失败: {str(e)}")
            return []

    def _extract_text_from_ocr_result(self, result) -> List[str]:
        """
        从 OCR 结果中提取文本

        Args:
            result: PaddleOCR 返回的结果

        Returns:
            文本行列表
        """
        text_boxes = []

        for line in result:
            if line:
                for word_info in line:
                    box = word_info[0]
                    text = word_info[1][0]
                    confidence = word_info[1][1]

                    # 计算文本框中心坐标
                    y_center = (box[0][1] + box[2][1]) / 2
                    x_center = (box[0][0] + box[2][0]) / 2

                    text_boxes.append({
                        'text': text,
                        'confidence': confidence,
                        'y_center': y_center,
                        'x_center': x_center
                    })

        if not text_boxes:
            return []

        # 按 Y 坐标排序，组织成行
        text_boxes.sort(key=lambda x: x['y_center'])

        # 将 Y 坐标相近的文本框合并为一行
        y_threshold = 20
        lines = []
        current_line = [text_boxes[0]]

        for i in range(1, len(text_boxes)):
            if abs(text_boxes[i]['y_center'] - text_boxes[i-1]['y_center']) < y_threshold:
                current_line.append(text_boxes[i])
            else:
                lines.append(current_line)
                current_line = [text_boxes[i]]

        if current_line:
            lines.append(current_line)

        # 每行内按 X 坐标排序
        text_lines = []
        for line in lines:
            line.sort(key=lambda x: x['x_center'])
            line_text = ' '.join([item['text'] for item in line])
            text_lines.append(line_text)

        return text_lines

    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        return self.SUPPORTED_EXTENSIONS


# 全局 OCR 解析器实例 (延迟初始化)
_ocr_parser = None


def get_ocr_parser() -> OCRParser:
    """获取 OCR 解析器实例"""
    global _ocr_parser
    if _ocr_parser is None:
        _ocr_parser = OCRParser()
    return _ocr_parser
