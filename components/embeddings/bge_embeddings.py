"""
BGE嵌入模型实现
基于BGE的中文嵌入模型
"""

import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from core.base.embeddings import BaseEmbeddings
from core.config.settings import settings
from core.config.llm_config import llm_config_manager
from core.exceptions.errors import EmbeddingException

logger = logging.getLogger(__name__)


class BGEEmbeddings(BaseEmbeddings):
    """BGE嵌入模型实现，使用单例模式避免重复加载"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("创建BGE嵌入模型实例...")
            cls._instance = super(BGEEmbeddings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        try:
            embedding_config = llm_config_manager.get_embedding_config()

            # 初始化HuggingFace嵌入模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_config.bge_model_name,
                model_kwargs={
                    'device': settings.EMBEDDING_MODEL_DEVICE
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': settings.EMBEDDING_BATCH_SIZE
                }
            )

            self._initialized = True
            logger.info(f"BGE嵌入模型加载完成！模型: {embedding_config.bge_model_name}")

        except Exception as e:
            error_msg = f"BGE嵌入模型初始化失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingException(error_msg, error_code="BGE_INIT_ERROR")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档生成嵌入向量

        Args:
            texts: 文档文本列表

        Returns:
            嵌入向量列表

        Raises:
            EmbeddingException: 当嵌入生成失败时
        """
        try:
            if not texts:
                return []

            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"成功为 {len(texts)} 个文档生成嵌入向量")
            return embeddings

        except Exception as e:
            error_msg = f"文档嵌入生成失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingException(error_msg, error_code="BGE_EMBED_DOCS_ERROR")

    def embed_query(self, text: str) -> List[float]:
        """
        为查询生成嵌入向量
        BGE模型推荐在查询前添加前缀以提高检索效果

        Args:
            text: 查询文本

        Returns:
            查询的嵌入向量

        Raises:
            EmbeddingException: 当嵌入生成失败时
        """
        try:
            if not text:
                raise EmbeddingException("查询文本不能为空", error_code="EMPTY_QUERY")

            # BGE模型推荐为查询添加前缀
            query_text = "查询：" + text
            embedding = self.embeddings.embed_query(query_text)
            logger.debug(f"成功为查询生成嵌入向量，查询长度: {len(text)}")
            return embedding

        except Exception as e:
            error_msg = f"查询嵌入生成失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingException(error_msg, error_code="BGE_EMBED_QUERY_ERROR")

    @property
    def dimension(self) -> int:
        """
        返回嵌入向量的维度
        BGE-small-zh模型的维度为512

        Returns:
            向量维度
        """
        return 512