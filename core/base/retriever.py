"""
检索器基类接口定义
提供统一的检索器接口规范
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.schema import Document


class BaseRetriever(ABC):
    """检索器基类接口"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            检索到的文档列表
        """
        pass

    def retrieve_with_filters(self,
                             query: str,
                             filters: Dict[str, Any],
                             top_k: int = 5) -> List[Document]:
        """
        带过滤条件的检索

        Args:
            query: 查询文本
            filters: 过滤条件
            top_k: 返回的最大文档数量

        Returns:
            检索到的文档列表
        """
        # 默认实现，子类可以重写
        return self.retrieve(query, top_k)


class BaseReranker(ABC):
    """重排序器基类接口"""

    @abstractmethod
    def rerank(self,
               query: str,
               documents: List[Document],
               top_k: int = 5) -> List[Document]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的最大文档数量

        Returns:
            重排序后的文档列表
        """
        pass