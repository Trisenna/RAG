"""
搜索服务实现
提供基础的文档搜索功能
"""

import logging
from typing import List, Dict, Any

from core.exceptions.errors import RetrievalException
from components.vectorstore.elasticsearch_store import ElasticsearchVectorStore
from components.retrieval.retrievers import BasicRetriever, HybridRetriever

logger = logging.getLogger(__name__)


class SearchService:
    """搜索服务 - 负责文档检索和搜索"""

    def __init__(self,
                 use_hybrid_retrieval: bool = True,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        初始化搜索服务

        Args:
            use_hybrid_retrieval: 是否使用混合检索
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
        """
        try:
            # 初始化向量存储
            self.vector_store = ElasticsearchVectorStore()

            # 初始化检索器
            if use_hybrid_retrieval:
                self.retriever = HybridRetriever(
                    self.vector_store,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight
                )
                retriever_type = "混合检索"
            else:
                self.retriever = BasicRetriever(self.vector_store)
                retriever_type = "基础检索"

            self.use_hybrid_retrieval = use_hybrid_retrieval

            logger.info(f"搜索服务初始化完成，检索器类型: {retriever_type}")

        except Exception as e:
            error_msg = f"搜索服务初始化失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="SEARCH_SERVICE_INIT_ERROR")

    def search_documents(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            搜索结果
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            logger.debug(f"执行文档搜索，查询: {query[:50]}..., top_k: {top_k}")

            # 执行检索
            documents = self.retriever.retrieve(query, top_k=top_k)

            # 转换为响应格式
            results = []
            for i, doc in enumerate(documents):
                result_item = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": self._clean_metadata(doc.metadata) if doc.metadata else {},
                    "score": self._extract_score(doc)
                }
                results.append(result_item)

            response = {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "retrieval_type": "hybrid" if self.use_hybrid_retrieval else "basic",
                "total_available": self.vector_store.get_document_count()
            }

            logger.debug(f"文档搜索完成，返回 {len(results)} 个结果")
            return response

        except RetrievalException:
            raise
        except Exception as e:
            error_msg = f"搜索文档时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "query": query,
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def search_with_filters(self,
                           query: str,
                           filters: Dict[str, Any],
                           top_k: int = 4) -> Dict[str, Any]:
        """
        带过滤条件的搜索

        Args:
            query: 查询文本
            filters: 过滤条件
            top_k: 返回的最大文档数量

        Returns:
            搜索结果
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            logger.debug(f"执行过滤搜索，查询: {query[:50]}..., 过滤器: {filters}")

            # 执行带过滤条件的检索
            documents = self.retriever.retrieve_with_filters(query, filters, top_k=top_k)

            # 转换为响应格式
            results = []
            for i, doc in enumerate(documents):
                result_item = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": self._clean_metadata(doc.metadata) if doc.metadata else {},
                    "score": self._extract_score(doc)
                }
                results.append(result_item)

            response = {
                "status": "success",
                "query": query,
                "filters": filters,
                "results": results,
                "count": len(results),
                "retrieval_type": "filtered_" + ("hybrid" if self.use_hybrid_retrieval else "basic")
            }

            logger.debug(f"过滤搜索完成，返回 {len(results)} 个结果")
            return response

        except RetrievalException:
            raise
        except Exception as e:
            error_msg = f"过滤搜索时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "query": query,
                "filters": filters,
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def semantic_search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        纯语义搜索

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            搜索结果
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            logger.debug(f"执行语义搜索，查询: {query[:50]}..., top_k: {top_k}")

            # 直接使用向量存储的语义搜索
            documents = self.vector_store.search(query, k=top_k)

            # 转换为响应格式
            results = []
            for i, doc in enumerate(documents):
                result_item = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": self._clean_metadata(doc.metadata) if doc.metadata else {},
                    "score": self._extract_score(doc)
                }
                results.append(result_item)

            response = {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "retrieval_type": "semantic"
            }

            logger.debug(f"语义搜索完成，返回 {len(results)} 个结果")
            return response

        except RetrievalException:
            raise
        except Exception as e:
            error_msg = f"语义搜索时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "query": query,
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def keyword_search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        纯关键词搜索

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            搜索结果
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            logger.debug(f"执行关键词搜索，查询: {query[:50]}..., top_k: {top_k}")

            # 直接使用向量存储的关键词搜索
            documents = self.vector_store.keyword_search(query, k=top_k)

            # 转换为响应格式
            results = []
            for i, doc in enumerate(documents):
                result_item = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": self._clean_metadata(doc.metadata) if doc.metadata else {},
                    "score": self._extract_score(doc)
                }
                results.append(result_item)

            response = {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "retrieval_type": "keyword"
            }

            logger.debug(f"关键词搜索完成，返回 {len(results)} 个结果")
            return response

        except RetrievalException:
            raise
        except Exception as e:
            error_msg = f"关键词搜索时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "query": query,
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索相关统计信息

        Returns:
            统计信息
        """
        try:
            stats = {
                "total_documents": self.vector_store.get_document_count(),
                "index_name": self.vector_store.index_name,
                "retrieval_type": "hybrid" if self.use_hybrid_retrieval else "basic",
                "vector_store_type": "elasticsearch"
            }

            # 如果是混合检索，添加权重信息
            if self.use_hybrid_retrieval and hasattr(self.retriever, 'semantic_weight'):
                stats.update({
                    "semantic_weight": self.retriever.semantic_weight,
                    "keyword_weight": self.retriever.keyword_weight
                })

            return stats

        except Exception as e:
            logger.error(f"获取搜索统计信息失败: {str(e)}")
            return {
                "total_documents": 0,
                "error": str(e)
            }

    def update_retrieval_weights(self, semantic_weight: float, keyword_weight: float) -> Dict[str, Any]:
        """
        更新检索权重（仅适用于混合检索）

        Args:
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重

        Returns:
            更新结果
        """
        try:
            if not self.use_hybrid_retrieval:
                return {
                    "status": "error",
                    "message": "当前使用的不是混合检索，无法更新权重"
                }

            if not hasattr(self.retriever, 'set_weights'):
                return {
                    "status": "error",
                    "message": "当前检索器不支持权重设置"
                }

            # 验证权重
            if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
                return {
                    "status": "error",
                    "message": f"权重之和必须等于1.0，当前为: {semantic_weight + keyword_weight}"
                }

            # 更新权重
            self.retriever.set_weights(semantic_weight, keyword_weight)

            logger.info(f"更新检索权重，语义: {semantic_weight}, 关键词: {keyword_weight}")
            return {
                "status": "success",
                "message": "检索权重更新成功",
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight
            }

        except Exception as e:
            error_msg = f"更新检索权重失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "error": str(e)
            }

    def _clean_metadata(self, metadata: dict) -> dict:
        """清理元数据，移除敏感或冗余信息"""
        if not metadata:
            return {}

        # 保留的重要字段
        important_fields = [
            "filename", "source", "chunk_id", "chunk_index",
            "total_chunks", "document_id", "is_proposition",
            "semantic_level", "file_type", "score", "rrf_score"
        ]

        cleaned = {}
        for field in important_fields:
            if field in metadata:
                cleaned[field] = metadata[field]

        return cleaned

    def _extract_score(self, doc) -> float:
        """提取文档的相关性分数"""
        if not hasattr(doc, 'metadata') or not doc.metadata:
            return 0.0

        # 优先使用RRF分数，其次使用其他分数
        for score_field in ['rrf_score', 'score', 'search_score']:
            score = doc.metadata.get(score_field)
            if score is not None:
                try:
                    return float(score)
                except (ValueError, TypeError):
                    continue

        return 0.0

    def test_connection(self) -> Dict[str, Any]:
        """
        测试搜索服务连接状态

        Returns:
            连接状态信息
        """
        try:
            # 测试向量存储连接
            doc_count = self.vector_store.get_document_count()

            return {
                "status": "connected",
                "vector_store": "elasticsearch",
                "document_count": doc_count,
                "retriever_type": "hybrid" if self.use_hybrid_retrieval else "basic"
            }

        except Exception as e:
            error_msg = f"连接测试失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "disconnected",
                "error": error_msg
            }