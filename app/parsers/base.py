"""
文档解析器基类
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ParseResult:
    """解析结果（按页返回）"""

    text: str
    page_num: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> List[ParseResult]:
        """解析文档"""

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名（小写、不带点）"""

    def can_parse(self, file_extension: str) -> bool:
        return file_extension.lower().lstrip(".") in self.supported_extensions()
