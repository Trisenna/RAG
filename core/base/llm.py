"""
LLM基类接口定义
提供统一的语言模型接口规范
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """语言模型基类接口"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本响应

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数

        Returns:
            生成的文本响应
        """
        pass

    @abstractmethod
    def generate_with_context(self,
                              query: str,
                              context: List[str],
                              **kwargs) -> str:
        """
        基于上下文生成响应

        Args:
            query: 用户查询
            context: 上下文信息列表
            **kwargs: 其他参数

        Returns:
            基于上下文生成的响应
        """
        pass

    @abstractmethod
    def chat(self,
             messages: List[Dict[str, str]],
             **kwargs) -> Dict[str, Any]:
        """
        聊天模式对话

        Args:
            messages: 对话消息列表
            **kwargs: 其他参数

        Returns:
            聊天响应结果
        """
        pass