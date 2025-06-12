"""
对话记忆存储实现
管理对话历史和会话状态
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Deque
from collections import deque

from core.exceptions.errors import ConversationException

logger = logging.getLogger(__name__)


class ConversationMemory:
    """对话记忆存储 - 管理对话历史和会话状态"""

    def __init__(self,
                 max_conversation_turns: int = 10,
                 max_tokens_per_message: int = 1000,
                 session_timeout_hours: int = 24):
        """
        初始化对话记忆存储

        Args:
            max_conversation_turns: 最大对话轮次
            max_tokens_per_message: 每条消息的最大token数
            session_timeout_hours: 会话超时时间（小时）
        """
        self.max_turns = max_conversation_turns
        self.max_tokens = max_tokens_per_message
        self.session_timeout = session_timeout_hours * 3600  # 转换为秒
        self.sessions = {}  # 会话ID到对话历史的映射

        logger.info(f"对话记忆存储初始化完成，最大轮次: {max_conversation_turns}, 超时: {session_timeout_hours}小时")

    def create_session(self) -> str:
        """
        创建新会话并返回会话ID

        Returns:
            新创建的会话ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "messages": deque(maxlen=self.max_turns),
            "created_at": time.time(),
            "last_active": time.time(),
            "metadata": {},
            "turn_count": 0
        }

        logger.debug(f"创建新会话: {session_id}")
        return session_id

    def add_message(self,
                   session_id: str,
                   role: str,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加消息到会话历史

        Args:
            session_id: 会话ID
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            metadata: 附加元数据

        Returns:
            是否成功添加

        Raises:
            ConversationException: 当会话不存在或添加失败时
        """
        if session_id not in self.sessions:
            raise ConversationException(
                f"会话ID不存在: {session_id}",
                error_code="SESSION_NOT_FOUND"
            )

        try:
            # 截断过长消息
            truncated_content = self._truncate_content(content)

            message = {
                "role": role,
                "content": truncated_content,
                "timestamp": time.time(),
                "turn_id": self.sessions[session_id]["turn_count"]
            }

            if metadata:
                message["metadata"] = metadata

            self.sessions[session_id]["messages"].append(message)
            self.sessions[session_id]["last_active"] = time.time()
            self.sessions[session_id]["turn_count"] += 1

            logger.debug(f"添加消息到会话 {session_id}，角色: {role}，长度: {len(content)}")
            return True

        except Exception as e:
            error_msg = f"添加消息失败: {str(e)}"
            logger.error(error_msg)
            raise ConversationException(error_msg, error_code="ADD_MESSAGE_ERROR")

    def get_messages(self,
                    session_id: str,
                    max_turns: Optional[int] = None,
                    include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        获取指定会话的消息历史

        Args:
            session_id: 会话ID
            max_turns: 最大返回轮次，None表示返回所有
            include_metadata: 是否包含元数据

        Returns:
            消息历史列表

        Raises:
            ConversationException: 当会话不存在时
        """
        if session_id not in self.sessions:
            raise ConversationException(
                f"会话ID不存在: {session_id}",
                error_code="SESSION_NOT_FOUND"
            )

        messages = list(self.sessions[session_id]["messages"])

        # 限制返回的消息数量
        if max_turns and max_turns < len(messages):
            messages = messages[-max_turns:]

        # 根据需要包含或排除元数据
        if not include_metadata:
            messages = [
                {k: v for k, v in msg.items() if k != "metadata"}
                for msg in messages
            ]

        return messages

    def get_conversation_history(self,
                                session_id: str,
                                max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取格式化的对话历史，适用于LLM上下文

        Args:
            session_id: 会话ID
            max_turns: 最大返回轮次

        Returns:
            格式化的对话历史
        """
        raw_messages = self.get_messages(session_id, max_turns, include_metadata=False)

        formatted_history = []
        for msg in raw_messages:
            formatted_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return formatted_history

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典

        Raises:
            ConversationException: 当会话不存在时
        """
        if session_id not in self.sessions:
            raise ConversationException(
                f"会话ID不存在: {session_id}",
                error_code="SESSION_NOT_FOUND"
            )

        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_active": session["last_active"],
            "message_count": len(session["messages"]),
            "turn_count": session["turn_count"],
            "metadata": session.get("metadata", {})
        }

    def update_session_metadata(self,
                              session_id: str,
                              metadata: Dict[str, Any]) -> bool:
        """
        更新会话元数据

        Args:
            session_id: 会话ID
            metadata: 要更新的元数据

        Returns:
            是否成功更新

        Raises:
            ConversationException: 当会话不存在时
        """
        if session_id not in self.sessions:
            raise ConversationException(
                f"会话ID不存在: {session_id}",
                error_code="SESSION_NOT_FOUND"
            )

        if "metadata" not in self.sessions[session_id]:
            self.sessions[session_id]["metadata"] = {}

        self.sessions[session_id]["metadata"].update(metadata)
        self.sessions[session_id]["last_active"] = time.time()

        logger.debug(f"更新会话元数据: {session_id}")
        return True

    def clear_session(self, session_id: str) -> bool:
        """
        清除会话历史但保留会话

        Args:
            session_id: 会话ID

        Returns:
            是否成功清除

        Raises:
            ConversationException: 当会话不存在时
        """
        if session_id not in self.sessions:
            raise ConversationException(
                f"会话ID不存在: {session_id}",
                error_code="SESSION_NOT_FOUND"
            )

        self.sessions[session_id]["messages"].clear()
        self.sessions[session_id]["turn_count"] = 0
        self.sessions[session_id]["last_active"] = time.time()

        logger.info(f"清除会话历史: {session_id}")
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否成功删除

        Raises:
            ConversationException: 当会话不存在时
        """
        if session_id not in self.sessions:
            raise ConversationException(
                f"会话ID不存在: {session_id}",
                error_code="SESSION_NOT_FOUND"
            )

        del self.sessions[session_id]
        logger.info(f"删除会话: {session_id}")
        return True

    def cleanup_expired_sessions(self) -> int:
        """
        清理过期的会话

        Returns:
            清理的会话数量
        """
        now = time.time()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if now - session["last_active"] > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"清理过期会话: {len(expired_sessions)} 个")

        return len(expired_sessions)

    def get_active_sessions(self) -> List[str]:
        """
        获取活跃会话列表

        Returns:
            活跃会话ID列表
        """
        now = time.time()
        active_sessions = []

        for session_id, session in self.sessions.items():
            if now - session["last_active"] <= self.session_timeout:
                active_sessions.append(session_id)

        return active_sessions

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        获取会话统计信息

        Returns:
            统计信息字典
        """
        total_sessions = len(self.sessions)
        active_sessions = len(self.get_active_sessions())
        total_messages = sum(len(session["messages"]) for session in self.sessions.values())

        avg_messages_per_session = total_messages / total_sessions if total_sessions > 0 else 0

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "expired_sessions": total_sessions - active_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": avg_messages_per_session
        }

    def _truncate_content(self, content: str) -> str:
        """
        截断过长的内容

        Args:
            content: 原始内容

        Returns:
            截断后的内容
        """
        # 粗略估计token数（中文按字符数，英文按单词数）
        estimated_tokens = len(content)

        if estimated_tokens > self.max_tokens * 4:
            truncated = content[:self.max_tokens * 4]
            logger.warning(f"消息内容过长，已截断: {estimated_tokens} -> {len(truncated)}")
            return truncated + "...(已截断)"

        return content

    def session_exists(self, session_id: str) -> bool:
        """
        检查会话是否存在

        Args:
            session_id: 会话ID

        Returns:
            会话是否存在
        """
        return session_id in self.sessions