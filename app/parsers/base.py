"""
文档解析器基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ParseResult:
    """解析结果"""
    text: str  # 文本内容
    page_num: int = 0  # 页码
    metadata: Dict[str, Any] = None  # 元数据


class BaseParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        """
        解析文档

        Args:
            file_path: 文件路径
            **kwargs: 其他参数

        Returns:
            解析结果列表
        """
        pass

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        支持的文件扩展名

        Returns:
            扩展名列表 (小写)
        """
        pass

    def can_parse(self, file_extension: str) -> bool:
        """
        检查是否支持该文件类型

        Args:
            file_extension: 文件扩展名

        Returns:
            是否支持
        """
        return file_extension.lower() in self.supported_extensions()
