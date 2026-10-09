"""
文本分割工具（优化版：支持递归字符分割和章节分割）
"""
from typing import List, Optional
from dataclasses import dataclass
import re

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """文本块（优化后：支持标题提取）"""
    title: str          # 标题
    content: str        # 正文内容
    chunk_id: int
    page_num: int = 0


class RecursiveCharacterTextSplitter:
    """
    递归字符文本分割器
    按照分隔符优先级递归分割文本，保持语义完整性
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        
        # 默认分隔符优先级（从大到小）
        # 优先按段落、句子分割，最后才按字符分割
        if separators is None:
            self.separators = [
                "\n\n",      # 段落分隔（优先级最高）
                "\n",        # 换行
                "。",        # 中文句号
                "！",        # 中文感叹号
                "？",        # 中文问号
                "；",        # 中文分号
                ".",         # 英文句号
                "!",         # 英文感叹号
                "?",         # 英文问号
                ";",         # 英文分号
                "，",        # 中文逗号
                ",",         # 英文逗号
                " ",         # 空格
                ""           # 字符级别（最后手段）
            ]
        else:
            self.separators = separators
    
    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """使用指定分隔符分割文本"""
        if separator:
            if self.keep_separator:
                # 保留分隔符在分割结果中
                splits = text.split(separator)
                return [s + separator for s in splits[:-1]] + [splits[-1]]
            else:
                return text.split(separator)
        else:
            # 按字符分割
            return list(text)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并分割片段，确保每个块不超过chunk_size，优先保持句子完整性"""
        merged_chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # 如果单个split就超过chunk_size，需要进一步分割
            if split_length > self.chunk_size:
                # 先保存当前chunk
                if current_chunk:
                    merged_chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 递归分割这个过大的split
                if separator:
                    # 找到下一个优先级的分隔符
                    try:
                        next_sep_index = self.separators.index(separator) + 1
                        if next_sep_index < len(self.separators):
                            next_separator = self.separators[next_sep_index]
                        else:
                            next_separator = ""
                    except ValueError:
                        next_separator = ""
                    
                    # 递归分割
                    sub_chunks = self._split_text_recursive(split, next_separator)
                    merged_chunks.extend(sub_chunks)
                else:
                    # 已经是字符级别，强制分割
                    for i in range(0, split_length, self.chunk_size - self.chunk_overlap):
                        merged_chunks.append(split[i:i + self.chunk_size])
                continue
            
            # 检查是否需要开始新的chunk
            if current_length + split_length > self.chunk_size and current_chunk:
                # 优先保持句子完整性：如果当前split是句子结尾，将其加入当前chunk
                is_sentence_end = any(split.endswith(punct) for punct in ["。", "！", "？", ".", "!", "?"])
                
                if is_sentence_end:
                    # 允许稍微超过chunk_size以保持句子完整
                    if current_length + split_length <= self.chunk_size * 1.2:  # 20% 容差
                        current_chunk.append(split)
                        current_length += split_length
                        merged_chunks.append(''.join(current_chunk))
                    else:
                        # 即使是句子结尾也超过太多，需要分割
                        merged_chunks.append(''.join(current_chunk))
                else:
                    # 不是句子结尾，正常分割
                    merged_chunks.append(''.join(current_chunk))
                
                # 处理重叠：保留最后几个split作为重叠
                overlap_chunks = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_chunks.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunks
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_length
        
        # 保存最后一个chunk
        if current_chunk:
            merged_chunks.append(''.join(current_chunk))
        
        return merged_chunks
    
    def _split_text_recursive(self, text: str, separator: str) -> List[str]:
        """递归分割文本"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # 使用当前分隔符分割
        splits = self._split_text_with_separator(text, separator)
        
        # 过滤空字符串
        splits = [s for s in splits if s.strip()]
        
        if not splits:
            return []
        
        # 检查是否所有split都小于chunk_size
        if all(len(s) <= self.chunk_size for s in splits):
            # 合并相邻的split
            return self._merge_splits(splits, separator)
        
        # 如果还有过大的split，使用下一个分隔符递归分割
        try:
            next_sep_index = self.separators.index(separator) + 1
            if next_sep_index < len(self.separators):
                next_separator = self.separators[next_sep_index]
            else:
                next_separator = ""
        except ValueError:
            next_separator = ""
        
        # 递归处理每个split
        result = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                result.extend(self._split_text_recursive(split, next_separator))
        
        # 最后合并一次
        return self._merge_splits(result, separator)
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        if not text or not text.strip():
            return []
        
        # 从第一个分隔符开始递归分割
        return self._split_text_recursive(text, self.separators[0])


class ChapterTextSplitter:
    """
    章节文本分割器
    识别文档中的章节结构，按章节分割
    """
    
    # 章节标题的正则模式
    CHAPTER_PATTERNS = [
        r'^第[一二三四五六七八九十百千万零\d]+[章节篇部]\s*.+',  # 第一章、第1节等
        r'^[一二三四五六七八九十百千万]+[、.．]\s*.+',  # 一、二、等
        r'^\d+[、.．]\s*.+',  # 1、2、等
        r'^#+\s+.+',  # Markdown标题
        r'^[A-Z][、.．]\s*.+',  # A、B、等
        r'^\([一二三四五六七八九十\d]+\)\s*.+',  # （一）、（1）等
    ]
    
    def __init__(self, min_chunk_size: int = 100):
        self.min_chunk_size = min_chunk_size
        self.chapter_patterns = [re.compile(p, re.MULTILINE) for p in self.CHAPTER_PATTERNS]
    
    def _find_chapters(self, text: str) -> List[tuple]:
        """查找文档中的章节位置"""
        chapters = []
        
        for pattern in self.chapter_patterns:
            matches = list(pattern.finditer(text))
            if matches:
                for match in matches:
                    chapters.append((match.start(), match.group().strip()))
        
        # 按位置排序
        chapters.sort(key=lambda x: x[0])
        
        # 去重（如果多个模式匹配到同一位置）
        unique_chapters = []
        last_pos = -1
        for pos, title in chapters:
            if pos > last_pos + 10:  # 至少间隔10个字符
                unique_chapters.append((pos, title))
                last_pos = pos
        
        return unique_chapters
    
    def split_text(self, text: str) -> List[tuple]:
        """
        按章节分割文本
        
        Returns:
            List of (title, content) tuples
        """
        if not text or not text.strip():
            return []
        
        chapters = self._find_chapters(text)
        
        if not chapters:
            # 没有找到章节，返回整个文本
            return [("", text)]
        
        result = []
        for i, (pos, title) in enumerate(chapters):
            # 章节内容从当前位置到下一个章节开始
            if i + 1 < len(chapters):
                content = text[pos:chapters[i+1][0]].strip()
            else:
                content = text[pos:].strip()
            
            # 移除标题行
            lines = content.split('\n', 1)
            if len(lines) > 1:
                content = lines[1].strip()
            else:
                content = ""
            
            if content:
                result.append((title, content))
        
        # 处理第一个章节之前的内容
        if chapters and chapters[0][0] > 0:
            intro_content = text[:chapters[0][0]].strip()
            if len(intro_content) >= self.min_chunk_size:
                result.insert(0, ("引言", intro_content))
        
        return result


class TextSplitter:
    """文本分割器（整合递归字符分割和章节分割）"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_chapter_split: bool = True
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.enable_chapter_split = enable_chapter_split
        
        # 初始化递归字符分割器
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # 初始化章节分割器
        self.chapter_splitter = ChapterTextSplitter(
            min_chunk_size=self.chunk_size // 2
        )

    def _extract_title(self, text: str) -> tuple:
        """
        提取标题和正文
        
        Args:
            text: 文本内容
            
        Returns:
            (title, content) 元组
        """
        lines = text.strip().split('\n')
        
        if not lines:
            return ("", "")
        
        # 尝试识别标题（通常在第一行或前几行）
        title = ""
        content_start = 0
        
        # 标题识别规则：
        # 1. 第一行较短（<100字符）且不以标点结尾
        # 2. 包含标题关键词（第、章、节等）
        # 3. Markdown标题格式（#、##、###）
        for i, line in enumerate(lines[:3]):  # 只检查前3行
            line = line.strip()
            if not line:
                continue
            
            # Markdown标题
            if line.startswith('#'):
                title = line.lstrip('#').strip()
                content_start = i + 1
                break
            
            # 中文标题特征
            if len(line) < 100 and not line.endswith(('。', '！', '？', '；', '：', '.', '!', '?', ';', ':')):
                if any(keyword in line for keyword in ['第', '章', '节', '部分', '一、', '二、', '三、', '1.', '2.', '3.']):
                    title = line
                    content_start = i + 1
                    break
                # 如果第一行较短且没有标点，可能是标题
                if i == 0 and len(line) < 50:
                    title = line
                    content_start = i + 1
                    break
        
        # 提取正文
        content = '\n'.join(lines[content_start:]).strip()
        
        # 如果没有识别到标题，使用第一句话作为标题
        if not title and content:
            # 提取第一句话（最多50字符）
            first_sentence = ""
            for char in content[:100]:
                first_sentence += char
                if char in ['。', '！', '？', '.', '!', '?']:
                    break
            title = first_sentence[:50] if len(first_sentence) > 10 else content[:30]
        
        return title, content

    def split_text(
        self,
        text: str,
        page_num: int = 0
    ) -> List[TextChunk]:
        """
        分割文本（使用递归字符分割和章节分割）

        Args:
            text: 待分割的文本
            page_num: 页码

        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []

        # 提取标题和正文
        title, content = self._extract_title(text)
        
        if not content:
            return []

        chunks = []
        chunk_id = 0
        
        # 尝试章节分割
        if self.enable_chapter_split:
            chapter_splits = self.chapter_splitter.split_text(content)
            
            # 如果成功识别到章节
            if len(chapter_splits) > 1:
                logger.info(f"识别到 {len(chapter_splits)} 个章节，按章节分割")
                
                for chapter_title, chapter_content in chapter_splits:
                    # 对每个章节内容使用递归字符分割
                    if len(chapter_content) > self.chunk_size:
                        sub_chunks = self.recursive_splitter.split_text(chapter_content)
                        
                        for i, sub_chunk in enumerate(sub_chunks):
                            chunk_title = chapter_title if i == 0 else f"{chapter_title} (续{i})"
                            chunks.append(TextChunk(
                                title=chunk_title,
                                content=sub_chunk,
                                chunk_id=chunk_id,
                                page_num=page_num
                            ))
                            chunk_id += 1
                    else:
                        # 章节内容较小，直接作为一个chunk
                        chunks.append(TextChunk(
                            title=chapter_title or title,
                            content=chapter_content,
                            chunk_id=chunk_id,
                            page_num=page_num
                        ))
                        chunk_id += 1
                
                logger.info(f"章节分割完成，生成 {len(chunks)} 个文本块")
                return chunks
        
        # 如果没有章节或章节分割未启用，使用递归字符分割
        if len(content) <= self.chunk_size:
            return [TextChunk(title=title, content=content, chunk_id=0, page_num=page_num)]
        
        # 使用递归字符分割
        text_chunks = self.recursive_splitter.split_text(content)
        
        for i, chunk_content in enumerate(text_chunks):
            chunk_title = title if i == 0 else f"{title} (续{i})"
            chunks.append(TextChunk(
                title=chunk_title,
                content=chunk_content,
                chunk_id=chunk_id,
                page_num=page_num
            ))
            chunk_id += 1

        logger.info(f"递归字符分割完成，生成 {len(chunks)} 个文本块（标题: {title[:30]}...）")
        return chunks

    def split_texts(
        self,
        texts: List[str],
        page_nums: List[int] = None
    ) -> List[TextChunk]:
        """
        分割多个文本

        Args:
            texts: 文本列表
            page_nums: 页码列表

        Returns:
            文本块列表
        """
        all_chunks = []
        chunk_id = 0

        for i, text in enumerate(texts):
            page_num = page_nums[i] if page_nums and i < len(page_nums) else 0
            text_chunks = self.split_text(text, page_num)

            # 重新编号
            for tc in text_chunks:
                tc.chunk_id = chunk_id
                chunk_id += 1

            all_chunks.extend(text_chunks)

        logger.info(f"总共生成 {len(all_chunks)} 个文本块")
        return all_chunks


# 全局分割器实例
text_splitter = TextSplitter()
