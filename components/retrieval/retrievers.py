"""
检索器实现
包含基础检索器、混合检索器等
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from core.base.retriever import BaseRetriever
from core.exceptions.errors import RetrievalException
from components.vectorstore.elasticsearch_store import ElasticsearchVectorStore

logger = logging.getLogger(__name__)


class BasicRetriever(BaseRetriever):
    """基础文档检索器"""

    def __init__(self, vector_store: ElasticsearchVectorStore):
        """
        初始化基础检索器

        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        logger.info("基础检索器初始化完成")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            检索到的文档列表

        Raises:
            RetrievalException: 当检索失败时
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            documents = self.vector_store.search(query, k=top_k)
            logger.debug(f"基础检索完成，查询: {query[:50]}..., 结果数量: {len(documents)}")
            return documents

        except Exception as e:
            error_msg = f"基础检索失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="BASIC_RETRIEVAL_ERROR")

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
        # TODO: 实现基于元数据的过滤
        logger.warning("过滤检索功能待实现，使用基础检索")
        return self.retrieve(query, top_k)


class HybridRetriever(BaseRetriever):
    """混合检索器 - 结合语义搜索和关键词搜索"""

    def __init__(self,
                 vector_store: ElasticsearchVectorStore,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        初始化混合检索器

        Args:
            vector_store: 向量存储实例
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
        """
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        # 验证权重
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            logger.warning(f"权重和不等于1.0: {semantic_weight + keyword_weight}")

        logger.info(f"混合检索器初始化完成，语义权重: {semantic_weight}, 关键词权重: {keyword_weight}")

    def retrieve(self, query: str, top_k: int = 8) -> List[Document]:
        """
        执行混合检索 - 结合向量搜索和BM25搜索

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            融合后的文档列表

        Raises:
            RetrievalException: 当检索失败时
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            # 计算各搜索方式的候选数量
            semantic_k = min(top_k * 2, 20)
            keyword_k = min(top_k * 2, 20)

            # 并行执行两种搜索
            semantic_results = self._semantic_search(query, k=semantic_k)
            keyword_results = self._keyword_search(query, k=keyword_k)

            # 融合结果
            merged_results = self._fuse_results(
                semantic_results,
                keyword_results,
                top_k=top_k
            )

            logger.debug(f"混合检索完成，语义: {len(semantic_results)}, 关键词: {len(keyword_results)}, 融合后: {len(merged_results)}")
            return merged_results

        except Exception as e:
            error_msg = f"混合检索失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="HYBRID_RETRIEVAL_ERROR")

    def _semantic_search(self, query: str, k: int = 10) -> List[Document]:
        """执行语义向量搜索"""
        try:
            return self.vector_store.search(query, k=k)
        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            return []

    def _keyword_search(self, query: str, k: int = 10) -> List[Document]:
        """执行BM25关键词搜索"""
        try:
            return self.vector_store.keyword_search(query, k=k)
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}")
            return []

    def _fuse_results(self,
                     semantic_docs: List[Document],
                     keyword_docs: List[Document],
                     top_k: int = 5) -> List[Document]:
        """
        使用RRF(Reciprocal Rank Fusion)算法融合两种检索结果

        Args:
            semantic_docs: 语义搜索结果
            keyword_docs: 关键词搜索结果
            top_k: 返回的最大文档数量

        Returns:
            融合后的文档列表
        """
        # 创建文档到索引的映射
        id_to_doc = {}
        semantic_ranks = {}
        keyword_ranks = {}

        # 处理语义搜索结果
        for i, doc in enumerate(semantic_docs):
            doc_id = self._get_doc_id(doc)
            semantic_ranks[doc_id] = i + 1
            id_to_doc[doc_id] = doc

        # 处理关键词搜索结果
        for i, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            keyword_ranks[doc_id] = i + 1
            id_to_doc[doc_id] = doc

        # 计算RRF分数 (k=60是常用默认值)
        k = 60
        rrf_scores = {}

        for doc_id in id_to_doc:
            semantic_rank = semantic_ranks.get(doc_id, len(semantic_docs) + 1)
            keyword_rank = keyword_ranks.get(doc_id, len(keyword_docs) + 1)

            # RRF公式: score = w1 * 1/(rank1 + k) + w2 * 1/(rank2 + k)
            semantic_score = self.semantic_weight * (1.0 / (semantic_rank + k))
            keyword_score = self.keyword_weight * (1.0 / (keyword_rank + k))

            rrf_scores[doc_id] = semantic_score + keyword_score

        # 按RRF分数排序并取top_k
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True
        )[:top_k]

        # 构建结果列表
        results = []
        for doc_id in sorted_doc_ids:
            doc = id_to_doc[doc_id]
            # 在元数据中添加RRF分数
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["rrf_score"] = rrf_scores[doc_id]
            doc.metadata["semantic_rank"] = semantic_ranks.get(doc_id, -1)
            doc.metadata["keyword_rank"] = keyword_ranks.get(doc_id, -1)
            results.append(doc)

        return results

    def _get_doc_id(self, doc: Document) -> str:
        """
        获取文档的唯一ID

        Args:
            doc: 文档对象

        Returns:
            文档唯一ID
        """
        if hasattr(doc, 'metadata') and doc.metadata:
            # 优先使用chunk_id
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                return chunk_id

            # 回退到source + chunk_index
            source = doc.metadata.get("source", "")
            chunk_index = doc.metadata.get("chunk_index", 0)
            return f"{source}_{chunk_index}"

        # 最后使用内容哈希
        return str(hash(doc.page_content))

    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """
        更新检索权重

        Args:
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
        """
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            logger.warning(f"权重和不等于1.0: {semantic_weight + keyword_weight}")

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        logger.info(f"更新检索权重，语义: {semantic_weight}, 关键词: {keyword_weight}")