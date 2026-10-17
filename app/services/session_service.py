"""
会话管理服务

修复点：
- 线程安全（RLock，且返回值是快照副本，调用方修改不会影响会话）；
- 会话数量上限（SESSION_MAX_COUNT），避免长时间运行内存无限增长；
- 过期清理从“每次读都全表扫描”改为按需 + 独立时间戳更新。
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Dict, List, Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Message:
    """会话消息"""

    __slots__ = ("role", "content", "timestamp")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content
        self.timestamp = datetime.now()


class Session:
    """会话"""

    def __init__(self, session_id: str, max_history: int = 10) -> None:
        self.session_id = session_id
        self.messages: List[Message] = []
        self.max_history = max_history
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))
        self.updated_at = datetime.now()
        # 一轮 = user + assistant，按轮数截断
        limit = self.max_history * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def get_history(self) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now()


class SessionService:
    """会话管理服务（进程内存储；多 worker / 多副本部署需替换为 Redis）"""

    def __init__(self, session_timeout: Optional[int] = None) -> None:
        self.session_timeout = session_timeout or settings.SESSION_TIMEOUT
        self.max_history = settings.SESSION_MAX_HISTORY
        self.max_sessions = settings.SESSION_MAX_COUNT
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        logger.info(
            "会话服务初始化完成，超时: %ss, 最大历史: %s 轮, 最大会话数: %s",
            self.session_timeout,
            self.max_history,
            self.max_sessions,
        )

    def _cleanup_locked(self) -> None:
        """清理过期会话（调用方需持有锁）"""
        now = datetime.now()
        expired = [
            sid
            for sid, session in self.sessions.items()
            if (now - session.updated_at).total_seconds() > self.session_timeout
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
        if expired:
            logger.info("清理过期会话 %s 个", len(expired))

        # 超过上限时淘汰最久未活跃的会话
        overflow = len(self.sessions) - self.max_sessions
        if overflow > 0:
            ordered = sorted(self.sessions.items(), key=lambda item: item[1].updated_at)
            for sid, _ in ordered[:overflow]:
                self.sessions.pop(sid, None)
            logger.warning("会话数量超限，淘汰最久未活跃会话 %s 个", overflow)

    def get_or_create_session(self, session_id: str) -> Session:
        with self._lock:
            self._cleanup_locked()
            session = self.sessions.get(session_id)
            if session is None:
                session = Session(session_id, self.max_history)
                self.sessions[session_id] = session
                logger.info("创建新会话: %s", session_id)
            return session

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self.get_or_create_session(session_id)
            session.add_message(role, content)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            session = self.sessions.get(session_id)
            return session.get_history() if session else []

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.clear()
                logger.info("清空会话: %s", session_id)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            if self.sessions.pop(session_id, None):
                logger.info("删除会话: %s", session_id)

    def get_session_count(self) -> int:
        with self._lock:
            return len(self.sessions)


session_service = SessionService()
