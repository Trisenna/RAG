"""
对话管理器实现
管理对话流程和状态，整合记忆存储和上下文分析
"""

import logging
from typing import List, Dict, Any, Optional

from .memory import ConversationMemory
from components.query.analyzers import ContextAnalyzer
from core.exceptions.errors import ConversationException

logger = logging.getLogger(__name__)


class ConversationManager:
    """对话管理器 - 管理对话流程和状态"""

    def __init__(self,
                 memory: ConversationMemory,
                 context_analyzer: ContextAnalyzer,
                 max_history_turns: int = 5,
                 enable_context_analysis: bool = True):
        """
        初始化对话管理器

        Args:
            memory: 对话记忆存储实例
            context_analyzer: 上下文分析器实例
            max_history_turns: 最大历史轮次
            enable_context_analysis: 是否启用上下文分析
        """
        self.memory = memory
        self.context_analyzer = context_analyzer
        self.max_history_turns = max_history_turns
        self.enable_context_analysis = enable_context_analysis

        logger.info(f"对话管理器初始化完成，最大历史轮次: {max_history_turns}, 上下文分析: {enable_context_analysis}")

    def create_session(self) -> str:
        """
        创建新的对话会话

        Returns:
            新创建的会话ID
        """
        session_id = self.memory.create_session()
        logger.info(f"创建新对话会话: {session_id}")
        return session_id

    def process_user_query(self,
                          session_id: str,
                          query: str) -> Dict[str, Any]:
        """
        处理用户查询，添加到历史并分析上下文

        Args:
            session_id: 会话ID
            query: 用户查询

        Returns:
            处理结果包含上下文信息

        Raises:
            ConversationException: 当处理失败时
        """
        try:
            # 确保会话存在
            if not self.memory.session_exists(session_id):
                session_id = self.create_session()
                logger.info(f"会话不存在，创建新会话: {session_id}")

            # 将查询添加到历史
            self.memory.add_message(session_id, "user", query)

            # 获取对话历史
            history = self.memory.get_conversation_history(session_id, self.max_history_turns)

            # 分析上下文信息
            context_data = self._analyze_context(query, history) if self.enable_context_analysis else {}

            # 构建返回结果
            result = {
                "session_id": session_id,
                "query": query,
                "history": history,
                "context_data": context_data,
                "has_context": self._has_meaningful_context(context_data),
                "turn_count": len(history)
            }

            logger.debug(f"处理用户查询完成，会话: {session_id}, 历史轮次: {len(history)}")
            return result

        except ConversationException:
            raise
        except Exception as e:
            error_msg = f"处理用户查询失败: {str(e)}"
            logger.error(error_msg)
            raise ConversationException(error_msg, error_code="PROCESS_QUERY_ERROR")

    def add_assistant_response(self,
                             session_id: str,
                             response: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加助手回复到对话历史

        Args:
            session_id: 会话ID
            response: 助手回复
            metadata: 回复元数据

        Returns:
            是否成功添加

        Raises:
            ConversationException: 当添加失败时
        """
        try:
            success = self.memory.add_message(session_id, "assistant", response, metadata)

            if success:
                logger.debug(f"添加助手回复到会话 {session_id}，长度: {len(response)}")

            return success

        except ConversationException:
            raise
        except Exception as e:
            error_msg = f"添加助手回复失败: {str(e)}"
            logger.error(error_msg)
            raise ConversationException(error_msg, error_code="ADD_RESPONSE_ERROR")

    def get_conversation_summary(self, session_id: str) -> str:
        """
        生成对话历史摘要，用于提示词构建

        Args:
            session_id: 会话ID

        Returns:
            对话历史摘要
        """
        try:
            history = self.memory.get_conversation_history(session_id, self.max_history_turns)

            if not history:
                return ""

            summary = "以下是之前的对话内容：\n\n"

            for msg in history:
                role = "用户" if msg["role"] == "user" else "助手"
                content = msg["content"]

                # 限制每条消息的长度
                if len(content) > 150:
                    content = content[:150] + "..."

                summary += f"{role}: {content}\n\n"

            return summary.strip()

        except Exception as e:
            logger.error(f"生成对话摘要失败: {str(e)}")
            return ""

    def clear_session_history(self, session_id: str) -> bool:
        """
        清除会话历史

        Args:
            session_id: 会话ID

        Returns:
            是否成功清除
        """
        try:
            success = self.memory.clear_session(session_id)
            if success:
                logger.info(f"清除会话历史: {session_id}")
            return success

        except ConversationException:
            raise
        except Exception as e:
            error_msg = f"清除会话历史失败: {str(e)}"
            logger.error(error_msg)
            raise ConversationException(error_msg, error_code="CLEAR_HISTORY_ERROR")

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否成功删除
        """
        try:
            success = self.memory.delete_session(session_id)
            if success:
                logger.info(f"删除会话: {session_id}")
            return success

        except ConversationException:
            raise
        except Exception as e:
            error_msg = f"删除会话失败: {str(e)}"
            logger.error(error_msg)
            raise ConversationException(error_msg, error_code="DELETE_SESSION_ERROR")

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息
        """
        try:
            session_info = self.memory.get_session_info(session_id)

            # 添加管理器级别的信息
            session_info.update({
                "max_history_turns": self.max_history_turns,
                "context_analysis_enabled": self.enable_context_analysis
            })

            return session_info

        except ConversationException:
            raise
        except Exception as e:
            error_msg = f"获取会话信息失败: {str(e)}"
            logger.error(error_msg)
            raise ConversationException(error_msg, error_code="GET_SESSION_INFO_ERROR")

    def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            清理的会话数量
        """
        try:
            cleaned_count = self.memory.cleanup_expired_sessions()
            if cleaned_count > 0:
                logger.info(f"清理过期会话: {cleaned_count} 个")
            return cleaned_count

        except Exception as e:
            logger.error(f"清理过期会话失败: {str(e)}")
            return 0

    def get_conversation_statistics(self) -> Dict[str, Any]:
        """
        获取对话统计信息

        Returns:
            统计信息
        """
        try:
            stats = self.memory.get_session_statistics()

            # 添加管理器级别的统计信息
            stats.update({
                "max_history_turns": self.max_history_turns,
                "context_analysis_enabled": self.enable_context_analysis
            })

            return stats

        except Exception as e:
            logger.error(f"获取对话统计信息失败: {str(e)}")
            return {}

    def _analyze_context(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        分析查询和历史的上下文信息

        Args:
            query: 当前查询
            history: 对话历史

        Returns:
            上下文分析结果
        """
        try:
            context_data = {}

            # 分析对话上下文
            if history:
                conversation_context = self.context_analyzer.extract_conversation_context(history, query)
                context_data["conversation_context"] = conversation_context

            # 分析查询中的引用
            reference_info = self.context_analyzer.detect_references(query)
            context_data["reference_info"] = reference_info

            # 分析查询意图
            intent_info = self.context_analyzer.analyze_query_intent(query)
            context_data["intent_info"] = intent_info

            return context_data

        except Exception as e:
            logger.error(f"上下文分析失败: {str(e)}")
            return {}

    def _has_meaningful_context(self, context_data: Dict[str, Any]) -> bool:
        """
        判断是否有有意义的上下文

        Args:
            context_data: 上下文数据

        Returns:
            是否有有意义的上下文
        """
        if not context_data:
            return False

        # 检查对话上下文
        conv_context = context_data.get("conversation_context", {})
        if conv_context.get("has_context", False):
            return True

        # 检查引用信息
        ref_info = context_data.get("reference_info", {})
        if ref_info.get("has_references", False):
            return True

        return False

    def set_context_analysis(self, enabled: bool):
        """
        设置是否启用上下文分析

        Args:
            enabled: 是否启用
        """
        self.enable_context_analysis = enabled
        logger.info(f"设置上下文分析: {'启用' if enabled else '禁用'}")

    def set_max_history_turns(self, max_turns: int):
        """
        设置最大历史轮次

        Args:
            max_turns: 最大轮次
        """
        if max_turns < 1:
            raise ConversationException("最大历史轮次必须大于0", error_code="INVALID_MAX_TURNS")

        self.max_history_turns = max_turns
        logger.info(f"设置最大历史轮次: {max_turns}")