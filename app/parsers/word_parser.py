"""
Word 文档解析器
支持 .doc 和 .docx 格式,无需 LibreOffice
"""
import os
from typing import List

from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WordParser(BaseParser):
    """Word 文档解析器"""

    SUPPORTED_EXTENSIONS = ['doc', 'docx']

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        """
        解析 Word 文档

        Args:
            file_path: 文件路径

        Returns:
            解析结果列表
        """
        logger.info(f"开始解析 Word 文档: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().replace('.', '')

        try:
            if ext == 'docx':
                text_content = self._parse_docx(file_path)
            elif ext == 'doc':
                text_content = self._parse_doc(file_path)
            else:
                logger.error(f"不支持的文件格式: {ext}")
                return []

            if not text_content.strip():
                logger.warning(f"Word 文档解析结果为空: {file_path}")
                return []

            result = ParseResult(
                text=text_content,
                page_num=0,
                metadata={
                    "parser": "word_parser",
                    "format": ext
                }
            )

            logger.info(f"Word 文档解析完成，提取 {len(text_content)} 字符")
            return [result]

        except Exception as e:
            logger.error(f"Word 文档解析失败: {str(e)}")
            return []

    def _parse_docx(self, file_path: str) -> str:
        """解析 .docx 文件"""
        try:
            # 优先使用 python-docx
            from docx import Document
            doc = Document(file_path)
            text_content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text_content
        except ImportError:
            logger.warning("python-docx 未安装,尝试使用 docx2txt")
            try:
                # 降级使用 docx2txt
                import docx2txt
                return docx2txt.process(file_path)
            except ImportError:
                logger.error("python-docx 和 docx2txt 都未安装,无法解析 .docx 文件")
                raise

    def _parse_doc(self, file_path: str) -> str:
        """解析 .doc 文件"""
        try:
            # 尝试使用 antiword-python
            try:
                import antiword
                return antiword.read_doc(file_path)
            except ImportError:
                pass

            # 尝试使用 textract (需要安装 textract)
            try:
                import textract
                text = textract.process(file_path)
                return text.decode('utf-8')
            except ImportError:
                pass

            # 最后尝试使用 win32com (仅 Windows)
            if os.name == 'nt':
                try:
                    import win32com.client
                    word = win32com.client.Dispatch("Word.Application")
                    word.Visible = False
                    doc = word.Documents.Open(file_path)
                    text_content = doc.Content.Text
                    doc.Close()
                    word.Quit()
                    return text_content
                except Exception as e:
                    logger.warning(f"win32com 解析失败: {str(e)}")

            logger.error(
                "无法解析 .doc 文件。请安装以下任一工具:\n"
                "1. pip install antiword-python\n"
                "2. pip install textract\n"
                "3. 安装 LibreOffice: https://www.libreoffice.org/\n"
                "4. (Windows) 安装 Microsoft Word 并使用 win32com"
            )
            raise Exception("缺少 .doc 文件解析工具")

        except Exception as e:
            logger.error(f".doc 文件解析失败: {str(e)}")
            raise

    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        return self.SUPPORTED_EXTENSIONS
