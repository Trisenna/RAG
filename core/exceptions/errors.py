"""
自定义异常类定义
统一管理项目中的异常类型
"""


class RAGException(Exception):
    """RAG系统基础异常类"""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class LLMException(RAGException):
    """LLM相关异常"""
    pass


class EmbeddingException(RAGException):
    """嵌入模型相关异常"""
    pass


class VectorStoreException(RAGException):
    """向量存储相关异常"""
    pass


class DocumentProcessingException(RAGException):
    """文档处理相关异常"""
    pass


class RetrievalException(RAGException):
    """检索相关异常"""
    pass


class ConfigurationException(RAGException):
    """配置相关异常"""
    pass


class ConversationException(RAGException):
    """对话管理相关异常"""
    pass


class IndexingException(RAGException):
    """索引相关异常"""
    pass


class QueryException(RAGException):
    """查询相关异常"""
    pass