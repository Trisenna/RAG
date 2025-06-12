"""
嵌入模型基类接口定义
提供统一的嵌入模型接口规范
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """嵌入模型基类接口"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入向量

        Args:
            texts: 文档文本列表

        Returns:
            嵌入向量列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入向量

        Args:
            text: 查询文本

        Returns:
            查询的嵌入向量
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        返回嵌入向量的维度

        Returns:
            向量维度
        """
        pass