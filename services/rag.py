"""
基础RAG服务实现
不考虑对话历史的RAG服务
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from core.config.llm_config import llm_config_manager
from core.exceptions.errors import LLMException, RetrievalException
from components.llm.openai_llm import OpenAILLM
from components.vectorstore.elasticsearch_store import ElasticsearchVectorStore
from components.retrieval.retrievers import HybridRetriever
from components.retrieval.rerankers import LLMReranker, ScoreBasedReranker, HybridReranker
from components.query.transformers import QueryRewriter, QueryDecomposer
from components.response.synthesizers import BaseResponseSynthesizer

logger = logging.getLogger(__name__)


class RAGService:
    """基础RAG服务 - 集成所有高级RAG特性的完整服务"""

    def __init__(self,
                 temperature: float = 0.7,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3,
                 use_query_rewriting: bool = True,
                 use_query_decomposition: bool = True,
                 use_reranking: bool = True,
                 use_citation: bool = True,
                 reranking_strategy: str = "hybrid"):
        """
        初始化RAG服务

        Args:
            temperature: LLM温度参数
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
            use_query_rewriting: 是否使用查询重写
            use_query_decomposition: 是否使用查询分解
            use_reranking: 是否使用重排序
            use_citation: 是否使用引用
            reranking_strategy: 重排序策略 ("llm", "score", "hybrid")
        """
        try:
            # 初始化LLM
            self.llm = OpenAILLM(temperature=temperature)

            # 初始化向量存储和检索器
            self.vector_store = ElasticsearchVectorStore()
            self.retriever = HybridRetriever(
                self.vector_store,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )

            # 初始化查询转换组件
            self.query_rewriter = QueryRewriter(self.llm) if use_query_rewriting else None
            self.query_decomposer = QueryDecomposer(self.llm) if use_query_decomposition else None

            # 初始化重排序器
            self.reranker = self._initialize_reranker(reranking_strategy) if use_reranking else None

            # 初始化响应合成器
            self.response_synthesizer = BaseResponseSynthesizer(self.llm)

            # 保存配置
            self.use_query_rewriting = use_query_rewriting
            self.use_query_decomposition = use_query_decomposition
            self.use_reranking = use_reranking
            self.use_citation = use_citation
            self.reranking_strategy = reranking_strategy

            # 线程池执行器用于异步调用
            self.executor = ThreadPoolExecutor(max_workers=5)

            logger.info(f"RAG服务初始化完成，查询重写: {use_query_rewriting}, 查询分解: {use_query_decomposition}, 重排序: {use_reranking}")

        except Exception as e:
            error_msg = f"RAG服务初始化失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg, error_code="RAG_SERVICE_INIT_ERROR")

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        执行RAG查询

        Args:
            query: 用户查询
            top_k: 返回的最大文档数量

        Returns:
            查询结果
        """
        start_time = time.time()

        try:
            if not query.strip():
                raise LLMException("查询文本不能为空", error_code="EMPTY_QUERY")

            logger.debug(f"开始RAG查询: {query[:50]}...")

            # 判断是否需要分解查询
            if (self.use_query_decomposition and
                self.query_decomposer and
                self.query_decomposer.should_decompose(query)):

                logger.debug("使用复杂查询处理流程")
                result = self._process_complex_query(query, top_k)
            else:
                logger.debug("使用简单查询处理流程")
                result = self._process_simple_query(query, top_k)

            # 添加处理时间
            result["processing_time"] = time.time() - start_time
            result["query"] = query

            logger.info(f"RAG查询完成，耗时: {result['processing_time']:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"RAG查询失败: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": f"处理查询时出错: {str(e)}",
                "query": query,
                "processing_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }

    async def query_async(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        异步执行RAG查询

        Args:
            query: 用户查询
            top_k: 返回的最大文档数量

        Returns:
            查询结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.query(query, top_k)
        )

    def _process_simple_query(self, query: str, top_k: int) -> Dict[str, Any]:
        """处理简单查询"""
        processing_info = {}

        try:
            # 1. 查询重写
            if self.use_query_rewriting and self.query_rewriter:
                rewritten_query = self.query_rewriter.rewrite(query)
                processing_info["rewritten_query"] = rewritten_query
                processing_info["original_query"] = query
            else:
                rewritten_query = query
                processing_info["rewritten_query"] = query

            # 2. 检索文档
            retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=top_k * 2)
            processing_info["retrieved_count"] = len(retrieved_docs)

            # 3. 重排序（如果启用）
            if self.use_reranking and self.reranker and retrieved_docs:
                ranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)
                processing_info["reranked"] = True
                processing_info["rerank_strategy"] = self.reranking_strategy
            else:
                ranked_docs = retrieved_docs[:top_k]
                processing_info["reranked"] = False

            # 4. 生成回答
            result = self.response_synthesizer.synthesize(
                query,
                ranked_docs,
                use_citation=self.use_citation
            )

            # 添加处理信息
            result["processing_info"] = processing_info
            result["status"] = "success"

            return result

        except Exception as e:
            logger.error(f"简单查询处理失败: {str(e)}")
            return {
                "answer": f"处理查询时出错: {str(e)}",
                "sources": [],
                "processing_info": processing_info,
                "status": "error",
                "error": str(e)
            }

    def _process_complex_query(self, query: str, top_k: int) -> Dict[str, Any]:
        """处理复杂查询（查询分解）"""
        processing_info = {}

        try:
            # 1. 分解查询
            sub_queries = self.query_decomposer.decompose(query)
            processing_info["sub_queries"] = sub_queries
            processing_info["sub_query_count"] = len(sub_queries)

            # 2. 处理子查询
            sub_results = []
            for sub_query in sub_queries:
                # 使用较小的top_k处理子查询
                sub_result = self._process_simple_query(sub_query, top_k=3)
                sub_results.append(sub_result)

            # 3. 合成最终回答
            result = self.response_synthesizer.synthesize_from_multiple_queries(
                query,
                sub_queries,
                sub_results
            )

            # 添加处理信息
            result["processing_info"] = processing_info
            result["sub_results"] = sub_results
            result["status"] = "success"

            return result

        except Exception as e:
            logger.error(f"复杂查询处理失败: {str(e)}")
            return {
                "answer": f"处理复杂查询时出错: {str(e)}",
                "sources": [],
                "processing_info": processing_info,
                "status": "error",
                "error": str(e)
            }

    def _initialize_reranker(self, strategy: str):
        """初始化重排序器"""
        try:
            if strategy == "llm":
                return LLMReranker(self.llm)
            elif strategy == "score":
                return ScoreBasedReranker()
            elif strategy == "hybrid":
                llm_reranker = LLMReranker(self.llm)
                score_reranker = ScoreBasedReranker()
                return HybridReranker(llm_reranker, score_reranker)
            else:
                logger.warning(f"未知的重排序策略: {strategy}，使用默认的混合策略")
                llm_reranker = LLMReranker(self.llm)
                score_reranker = ScoreBasedReranker()
                return HybridReranker(llm_reranker, score_reranker)

        except Exception as e:
            logger.error(f"初始化重排序器失败: {str(e)}")
            # 回退到基于分数的重排序器
            return ScoreBasedReranker()

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            服务配置信息
        """
        try:
            return {
                "service_type": "basic_rag",
                "llm_model": self.llm.config.model if hasattr(self.llm, 'config') else "unknown",
                "vector_store": "elasticsearch",
                "retriever_type": "hybrid",
                "features": {
                    "query_rewriting": self.use_query_rewriting,
                    "query_decomposition": self.use_query_decomposition,
                    "reranking": self.use_reranking,
                    "citation": self.use_citation
                },
                "reranking_strategy": self.reranking_strategy,
                "retrieval_weights": {
                    "semantic": self.retriever.semantic_weight,
                    "keyword": self.retriever.keyword_weight
                },
                "document_count": self.vector_store.get_document_count()
            }

        except Exception as e:
            logger.error(f"获取服务信息失败: {str(e)}")
            return {
                "service_type": "basic_rag",
                "status": "error",
                "error": str(e)
            }

    def update_configuration(self, **kwargs) -> Dict[str, Any]:
        """
        更新服务配置

        Args:
            **kwargs: 配置参数

        Returns:
            更新结果
        """
        try:
            updated_params = []

            # 更新检索权重
            if "semantic_weight" in kwargs and "keyword_weight" in kwargs:
                semantic_weight = kwargs["semantic_weight"]
                keyword_weight = kwargs["keyword_weight"]

                if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
                    return {
                        "status": "error",
                        "message": f"权重之和必须等于1.0，当前为: {semantic_weight + keyword_weight}"
                    }

                self.retriever.set_weights(semantic_weight, keyword_weight)
                updated_params.extend(["semantic_weight", "keyword_weight"])

            # 更新功能开关
            feature_flags = ["use_query_rewriting", "use_query_decomposition", "use_reranking", "use_citation"]
            for flag in feature_flags:
                if flag in kwargs:
                    setattr(self, flag, kwargs[flag])
                    updated_params.append(flag)

            # 更新重排序策略
            if "reranking_strategy" in kwargs:
                new_strategy = kwargs["reranking_strategy"]
                if new_strategy in ["llm", "score", "hybrid"]:
                    self.reranking_strategy = new_strategy
                    self.reranker = self._initialize_reranker(new_strategy)
                    updated_params.append("reranking_strategy")
                else:
                    return {
                        "status": "error",
                        "message": f"不支持的重排序策略: {new_strategy}"
                    }

            logger.info(f"更新RAG服务配置: {updated_params}")
            return {
                "status": "success",
                "message": f"成功更新配置: {', '.join(updated_params)}",
                "updated_parameters": updated_params
            }

        except Exception as e:
            error_msg = f"更新配置失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "error": str(e)
            }

    def test_query_pipeline(self, test_query: str = "测试查询") -> Dict[str, Any]:
        """
        测试查询流水线

        Args:
            test_query: 测试查询文本

        Returns:
            测试结果
        """
        try:
            # 测试各个组件
            pipeline_status = {}

            # 测试检索器
            try:
                docs = self.retriever.retrieve(test_query, top_k=1)
                pipeline_status["retriever"] = "ok"
                pipeline_status["retriever_result_count"] = len(docs)
            except Exception as e:
                pipeline_status["retriever"] = f"error: {str(e)}"

            # 测试查询重写器
            if self.query_rewriter:
                try:
                    rewritten = self.query_rewriter.rewrite(test_query)
                    pipeline_status["query_rewriter"] = "ok"
                except Exception as e:
                    pipeline_status["query_rewriter"] = f"error: {str(e)}"
            else:
                pipeline_status["query_rewriter"] = "disabled"

            # 测试查询分解器
            if self.query_decomposer:
                try:
                    should_decompose = self.query_decomposer.should_decompose(test_query)
                    pipeline_status["query_decomposer"] = "ok"
                    pipeline_status["should_decompose"] = should_decompose
                except Exception as e:
                    pipeline_status["query_decomposer"] = f"error: {str(e)}"
            else:
                pipeline_status["query_decomposer"] = "disabled"

            # 测试重排序器
            if self.reranker:
                try:
                    # 这里需要至少有一些文档才能测试重排序
                    if pipeline_status.get("retriever") == "ok" and pipeline_status.get("retriever_result_count", 0) > 0:
                        docs = self.retriever.retrieve(test_query, top_k=2)
                        if docs:
                            ranked = self.reranker.rerank(test_query, docs, top_k=1)
                            pipeline_status["reranker"] = "ok"
                        else:
                            pipeline_status["reranker"] = "no_documents_to_rerank"
                    else:
                        pipeline_status["reranker"] = "cannot_test_no_documents"
                except Exception as e:
                    pipeline_status["reranker"] = f"error: {str(e)}"
            else:
                pipeline_status["reranker"] = "disabled"

            # 测试响应合成器
            try:
                empty_response = self.response_synthesizer.synthesize(test_query, [])
                pipeline_status["response_synthesizer"] = "ok"
            except Exception as e:
                pipeline_status["response_synthesizer"] = f"error: {str(e)}"

            return {
                "status": "completed",
                "test_query": test_query,
                "pipeline_status": pipeline_status,
                "overall_health": "healthy" if all(
                    status in ["ok", "disabled", "no_documents_to_rerank", "cannot_test_no_documents"]
                    for status in pipeline_status.values()
                    if isinstance(status, str)
                ) else "unhealthy"
            }

        except Exception as e:
            error_msg = f"管道测试失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "error": str(e)
            }