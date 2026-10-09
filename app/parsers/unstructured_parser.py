"""
Unstructured 文档解析器
支持多种常见文档格式
"""
import os
from typing import List
import warnings

# 禁用 unstructured 的自动模型下载提示
os.environ['UNSTRUCTURED_DISABLE_DOWNLOADS'] = 'true'

from unstructured.partition.auto import partition

from app.parsers.base import BaseParser, ParseResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class UnstructuredParser(BaseParser):
    """Unstructured 文档解析器"""

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = [
        'pdf',      # PDF (文本型)
        'doc',      # Word 97-2003
        'docx',     # Word 2007+
        'txt',      # 纯文本
        'md',       # Markdown
        'rtf',      # 富文本
        'csv',      # CSV
        'xls',      # Excel 97-2003
        'xlsx',     # Excel 2007+
        'ppt',      # PowerPoint 97-2003
        'pptx',     # PowerPoint 2007+
        'eml',      # 邮件
        'msg',      # Outlook 邮件
        'html',     # HTML
        'htm',      # HTML
        'xml',      # XML
        'epub',     # 电子书
        'json',     # JSON
        'yaml',     # YAML
        'yml',      # YAML
    ]

    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        """
        解析文档

        Args:
            file_path: 文件路径

        Returns:
            解析结果列表
        """
        logger.info(f"开始使用 Unstructured 解析: {file_path}")

        try:
            # 获取文件扩展名
            ext = os.path.splitext(file_path)[1].lower()

            # 根据文件类型选择分区方法
            # 使用 'fast' 策略避免下载额外的模型
            # 支持中英文混合文档
            elements = partition(
                filename=file_path,
                strategy="fast",  # 使用快速策略,不依赖深度学习模型
                skip_infer_table_types=[],  # 跳过表格推理以加快速度
                languages=["eng", "chi_sim"]  # 支持英文和中文
            )

            # 提取文本
            text_content = "\n\n".join([str(el) for el in elements if str(el).strip()])

            if not text_content.strip():
                logger.warning(f"Unstructured 解析结果为空: {file_path}")
                return []

            result = ParseResult(
                text=text_content,
                page_num=0,
                metadata={
                    "parser": "unstructured",
                    "element_count": len(elements),
                    "strategy": "fast"
                }
            )

            logger.info(f"Unstructured 解析完成，提取 {len(text_content)} 字符")
            return [result]

        except Exception as e:
            error_msg = str(e)

            # 检查是否是 LibreOffice 缺失错误
            if "soffice command was not found" in error_msg or "libreoffice" in error_msg.lower():
                logger.error(
                    f"解析 {ext} 文件需要 LibreOffice。请选择以下任一方案:\n"
                    "1. 安装 LibreOffice: https://www.libreoffice.org/get-help/install-howto/\n"
                    "2. 将 .doc 文件转换为 .docx 格式\n"
                    "3. 使用专门的 Word 解析器 (word_parser.py)"
                )
                return []

            logger.error(f"Unstructured 解析失败: {error_msg}")
            # 如果 fast 策略失败,尝试使用 auto 策略
            try:
                logger.info("尝试使用 auto 策略重新解析...")
                elements = partition(filename=file_path, strategy="auto")
                text_content = "\n\n".join([str(el) for el in elements if str(el).strip()])

                if text_content.strip():
                    result = ParseResult(
                        text=text_content,
                        page_num=0,
                        metadata={
                            "parser": "unstructured",
                            "element_count": len(elements),
                            "strategy": "auto_fallback"
                        }
                    )
                    logger.info(f"auto 策略解析成功，提取 {len(text_content)} 字符")
                    return [result]
            except Exception as e2:
                logger.error(f"auto 策略也失败: {str(e2)}")

            return []

    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        return self.SUPPORTED_EXTENSIONS
