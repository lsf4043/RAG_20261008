"""
会话管理服务（保留旧版逻辑，改用配置驱动）
管理用户对话历史，支持多轮对话。
"""
from typing import List, Dict, Optional
from datetime import datetime
import threading

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Message:
    """消息类"""
    def __init__(self, role: str, content: str):
        self.role = role  # "user" 或 "assistant"
        self.content = content
        self.timestamp = datetime.now()


class Session:
    """会话类"""
    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.messages: List[Message] = []
        self.max_history = max_history  # 最多保留的历史轮数
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append(Message(role, content))
        self.updated_at = datetime.now()

        # 限制历史消息数量（user + assistant = 2 条/轮）
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]

    def get_history(self) -> List[Dict]:
        """获取对话历史（用于大模型调用）"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def clear(self):
        """清空历史"""
        self.messages = []
        self.updated_at = datetime.now()


class SessionService:
    """会话管理服务"""

    def __init__(self, session_timeout: Optional[int] = None):
        self.session_timeout = session_timeout or settings.SESSION_TIMEOUT
        self.max_history = settings.SESSION_MAX_HISTORY
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        logger.info(f"会话管理服务初始化完成，超时: {self.session_timeout}s, 最大历史: {self.max_history} 轮")

    def get_or_create_session(self, session_id: str) -> Session:
        """获取或创建会话"""
        with self._lock:
            self._clean_expired_sessions()
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id, self.max_history)
                logger.info(f"创建新会话: {session_id}")
            return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        session = self.get_or_create_session(session_id)
        session.add_message(role, content)
        logger.debug(f"会话 {session_id} 添加消息: {role}")

    def get_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        with self._lock:
            self._clean_expired_sessions()
            if session_id in self.sessions:
                return self.sessions[session_id].get_history()
            return []

    def clear_session(self, session_id: str):
        """清空会话历史"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                logger.info(f"清空会话: {session_id}")

    def delete_session(self, session_id: str):
        """删除会话"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"删除会话: {session_id}")

    def _clean_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if (now - session.updated_at).total_seconds() > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"清理过期会话: {len(expired)} 个")

    def get_session_count(self) -> int:
        """获取当前会话数量"""
        return len(self.sessions)


# 全局会话服务实例
session_service = SessionService()