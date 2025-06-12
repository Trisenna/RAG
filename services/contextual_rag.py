"""
上下文感知RAG服务实现
结合对话历史的完整RAG服务
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .rag import RAGService
from components.conversation.memory import ConversationMemory
from components.conversation.manager import ConversationManager
from components.query.analyzers import ContextAnalyzer
from components.query.transformers import ContextualQueryRewriter
from components.response.synthesizers import ContextualResponseSynthesizer
from core.exceptions.errors import ConversationException, LLMException

logger = logging.getLogger(__name__)


class ContextualRAGService:
    """上下文感知RAG服务 - 结合对话历史的完整RAG服务"""

    def __init__(self,
                 temperature: float = 0.7,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3,
                 use_query_rewriting: bool = True,
                 use_query_decomposition: bool = True,
                 use_reranking: bool = True,
                 use_citation: bool = True,
                 max_history_turns: int = 5,
                 reranking_strategy: str = "hybrid",
                 session_timeout_hours: int = 24):
        """
        初始化上下文感知RAG服务

        Args:
            temperature: LLM温度参数
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
            use_query_rewriting: 是否使用查询重写
            use_query_decomposition: 是否使用查询分解
            use_reranking: 是否使用重排序
            use_citation: 是否使用引用
            max_history_turns: 最大历史轮次
            reranking_strategy: 重排序策略
            session_timeout_hours: 会话超时时间（小时）
        """
        try:
            # 初始化基础RAG服务
            self.base_rag = RAGService(
                temperature=temperature,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                use_query_rewriting=False,  # 由上下文感知重写器处理
                use_query_decomposition=use_query_decomposition,
                use_reranking=use_reranking,
                use_citation=use_citation,
                reranking_strategy=reranking_strategy
            )

            # 初始化对话管理组件
            self.memory = ConversationMemory(
                max_conversation_turns=max_history_turns,
                session_timeout_hours=session_timeout_hours
            )
            self.context_analyzer = ContextAnalyzer(self.base_rag.llm)
            self.conversation_manager = ConversationManager(
                self.memory,
                self.context_analyzer,
                max_history_turns=max_history_turns
            )

            # 初始化上下文感知组件
            self.contextual_rewriter = ContextualQueryRewriter(self.base_rag.llm) if use_query_rewriting else None
            self.contextual_synthesizer = ContextualResponseSynthesizer(self.base_rag.llm)

            # 保存配置
            self.max_history_turns = max_history_turns
            self.use_query_rewriting = use_query_rewriting
            self.use_query_decomposition = use_query_decomposition
            self.use_reranking = use_reranking
            self.use_citation = use_citation

            # 线程池执行器
            self.executor = ThreadPoolExecutor(max_workers=5)

            logger.info(f"上下文感知RAG服务初始化完成，最大历史轮次: {max_history_turns}")

        except Exception as e:
            error_msg = f"上下文感知RAG服务初始化失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg, error_code="CONTEXTUAL_RAG_INIT_ERROR")

    def query(self,
             query: str,
             session_id: Optional[str] = None,
             top_k: int = 5) -> Dict[str, Any]:
        """
        执行上下文感知RAG查询

        Args:
            query: 用户查询
            session_id: 会话ID（如果为None将创建新会话）
            top_k: 返回的最大文档数量

        Returns:
            查询结果
        """
        start_time = time.time()

        try:
            if not query.strip():
                raise LLMException("查询文本不能为空", error_code="EMPTY_QUERY")

            # 创建或获取会话ID
            if not session_id:
                session_id = self.conversation_manager.create_session()

            logger.debug(f"开始上下文感知RAG查询: {query[:50]}..., 会话: {session_id}")

            # 处理查询，获取上下文信息
            context_data = self.conversation_manager.process_user_query(session_id, query)
            has_context = context_data.get("has_context", False)

            # 根据查询复杂度选择处理策略
            if (self.use_query_decomposition and
                self.base_rag.query_decomposer and
                self.base_rag.query_decomposer.should_decompose(query)):

                logger.debug("使用复杂查询处理流程")
                result = self._process_complex_query(query, context_data, top_k)
            else:
                logger.debug("使用简单查询处理流程")
                result = self._process_simple_query(query, context_data, top_k)

            # 将回答添加到对话历史
            answer = result.get("answer", "")
            self.conversation_manager.add_assistant_response(
                session_id,
                answer,
                {"sources": result.get("sources", [])}
            )

            # 添加会话信息到结果
            result.update({
                "session_id": session_id,
                "processing_time": time.time() - start_time,
                "used_context": has_context,
                "turn_count": context_data.get("turn_count", 0)
            })

            logger.info(f"上下文感知RAG查询完成，耗时: {result['processing_time']:.2f}秒")
            return result

        except ConversationException as e:
            logger.error(f"对话处理失败: {str(e)}")
            return {
                "answer": f"对话处理时出错: {str(e)}",
                "query": query,
                "session_id": session_id,
                "processing_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"上下文感知RAG查询失败: {str(e)}")

            # 尝试将错误响应添加到会话历史
            if session_id:
                try:
                    self.conversation_manager.add_assistant_response(
                        session_id,
                        f"处理查询时出错: {str(e)}",
                        {"error": str(e)}
                    )
                except:
                    pass  # 忽略添加错误响应的失败

            return {
                "answer": f"处理查询时出错: {str(e)}",
                "query": query,
                "session_id": session_id,
                "processing_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }

    async def query_async(self,
                         query: str,
                         session_id: Optional[str] = None,
                         top_k: int = 5) -> Dict[str, Any]:
        """异步执行上下文感知RAG查询"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.query(query, session_id, top_k)
        )

    def _process_simple_query(self,
                            query: str,
                            context_data: Dict[str, Any],
                            top_k: int) -> Dict[str, Any]:
        """处理简单查询（考虑对话上下文）"""
        processing_info = {}
        session_id = context_data.get("session_id")
        history = context_data.get("history", [])
        has_context = context_data.get("has_context", False)

        try:
            # 1. 上下文感知查询重写
            if self.use_query_rewriting and self.contextual_rewriter and has_context:
                conv_context = context_data.get("context_data", {}).get("conversation_context", {})
                rewritten_query = self.contextual_rewriter.rewrite_with_history(
                    query,
                    history,
                    conv_context
                )
                processing_info["rewritten_query"] = rewritten_query
                processing_info["original_query"] = query
                processing_info["context_enhanced"] = True
            else:
                rewritten_query = query
                processing_info["rewritten_query"] = query
                processing_info["context_enhanced"] = False

            # 2. 检索文档
            retrieved_docs = self.base_rag.retriever.retrieve(rewritten_query, top_k=top_k * 2)
            processing_info["retrieved_count"] = len(retrieved_docs)

            # 3. 重排序（如果启用）
            if self.use_reranking and self.base_rag.reranker and retrieved_docs:
                ranked_docs = self.base_rag.reranker.rerank(query, retrieved_docs, top_k=top_k)
                processing_info["reranked"] = True
            else:
                ranked_docs = retrieved_docs[:top_k]
                processing_info["reranked"] = False

            # 4. 上下文感知回答生成
            result = self.contextual_synthesizer.synthesize_with_history(
                query,
                ranked_docs,
                history,
                use_citation=self.use_citation
            )

            # 添加处理信息
            result["processing_info"] = processing_info
            result["query"] = query
            result["status"] = "success"

            return result

        except Exception as e:
            logger.error(f"简单查询上下文处理失败: {str(e)}")
            return {
                "answer": f"处理查询时出错: {str(e)}",
                "sources": [],
                "processing_info": processing_info,
                "query": query,
                "status": "error",
                "error": str(e)
            }

    def _process_complex_query(self,
                             query: str,
                             context_data: Dict[str, Any],
                             top_k: int) -> Dict[str, Any]:
        """处理复杂查询（查询分解 + 对话上下文）"""
        processing_info = {}
        session_id = context_data.get("session_id")
        history = context_data.get("history", [])
        has_context = context_data.get("has_context", False)

        try:
            # 1. 先对原始查询进行上下文重写
            if self.use_query_rewriting and self.contextual_rewriter and has_context:
                conv_context = context_data.get("context_data", {}).get("conversation_context", {})
                contextual_query = self.contextual_rewriter.rewrite_with_history(
                    query,
                    history,
                    conv_context
                )
                processing_info["contextual_query"] = contextual_query
                processing_info["context_enhanced"] = True
            else:
                contextual_query = query
                processing_info["contextual_query"] = query
                processing_info["context_enhanced"] = False

            # 2. 分解上下文增强的查询
            sub_queries = self.base_rag.query_decomposer.decompose(contextual_query)
            processing_info["sub_queries"] = sub_queries
            processing_info["sub_query_count"] = len(sub_queries)

            # 3. 处理子查询（不需要再考虑上下文，因为上下文已经被合并到子查询中）
            sub_results = []
            for sub_query in sub_queries:
                sub_result = self.base_rag._process_simple_query(sub_query, top_k=3)
                sub_results.append(sub_result)

            # 4. 使用上下文感知合成器生成最终回答
            result = self.contextual_synthesizer.synthesize_with_history(
                query,
                self._extract_documents_from_sub_results(sub_results, top_k),
                history,
                use_citation=self.use_citation
            )

            # 添加处理信息
            result["processing_info"] = processing_info
            result["sub_results"] = sub_results
            result["query"] = query
            result["status"] = "success"

            return result

        except Exception as e:
            logger.error(f"复杂查询上下文处理失败: {str(e)}")
            return {
                "answer": f"处理复杂查询时出错: {str(e)}",
                "sources": [],
                "processing_info": processing_info,
                "query": query,
                "status": "error",
                "error": str(e)
            }

    def _extract_documents_from_sub_results(self, sub_results: list, top_k: int) -> list:
        """从子查询结果中提取文档"""
        from langchain.schema import Document

        final_docs = []
        for result in sub_results:
            if "sources" in result:
                for source in result.get("sources", []):
                    if "content" in source:
                        doc = Document(
                            page_content=source["content"],
                            metadata=source
                        )
                        final_docs.append(doc)

        return final_docs[:top_k]

    def create_session(self) -> str:
        """创建新会话"""
        session_id = self.conversation_manager.create_session()
        logger.info(f"创建新会话: {session_id}")
        return session_id

    def clear_session(self, session_id: str) -> bool:
        """清除会话历史"""
        try:
            success = self.conversation_manager.clear_session_history(session_id)
            if success:
                logger.info(f"清除会话历史: {session_id}")
            return success
        except Exception as e:
            logger.error(f"清除会话历史失败: {session_id}, 错误: {str(e)}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        try:
            success = self.conversation_manager.delete_session(session_id)
            if success:
                logger.info(f"删除会话: {session_id}")
            return success
        except Exception as e:
            logger.error(f"删除会话失败: {session_id}, 错误: {str(e)}")
            return False

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        try:
            return self.conversation_manager.get_session_info(session_id)
        except Exception as e:
            logger.error(f"获取会话信息失败: {session_id}, 错误: {str(e)}")
            return {
                "error": str(e),
                "session_id": session_id
            }

    def get_conversation_history(self, session_id: str) -> list:
        """获取对话历史"""
        try:
            return self.memory.get_conversation_history(session_id, self.max_history_turns)
        except Exception as e:
            logger.error(f"获取对话历史失败: {session_id}, 错误: {str(e)}")
            return []

    def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        try:
            cleaned_count = self.conversation_manager.cleanup_expired_sessions()
            if cleaned_count > 0:
                logger.info(f"清理过期会话: {cleaned_count} 个")
            return cleaned_count
        except Exception as e:
            logger.error(f"清理过期会话失败: {str(e)}")
            return 0

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        try:
            base_info = self.base_rag.get_service_info()
            base_info.update({
                "service_type": "contextual_rag",
                "conversation_features": {
                    "max_history_turns": self.max_history_turns,
                    "context_analysis": True,
                    "contextual_rewriting": self.use_query_rewriting,
                    "session_management": True
                },
                "conversation_statistics": self.conversation_manager.get_conversation_statistics()
            })
            return base_info
        except Exception as e:
            logger.error(f"获取服务信息失败: {str(e)}")
            return {
                "service_type": "contextual_rag",
                "status": "error",
                "error": str(e)
            }

    def update_configuration(self, **kwargs) -> Dict[str, Any]:
        """更新服务配置"""
        try:
            updated_params = []

            # 更新基础RAG配置
            base_result = self.base_rag.update_configuration(**kwargs)
            if base_result.get("status") == "success":
                updated_params.extend(base_result.get("updated_parameters", []))

            # 更新对话相关配置
            if "max_history_turns" in kwargs:
                new_max_turns = kwargs["max_history_turns"]
                if new_max_turns > 0:
                    self.max_history_turns = new_max_turns
                    self.conversation_manager.set_max_history_turns(new_max_turns)
                    updated_params.append("max_history_turns")

            # 更新上下文分析开关
            if "enable_context_analysis" in kwargs:
                enabled = kwargs["enable_context_analysis"]
                self.conversation_manager.set_context_analysis(enabled)
                updated_params.append("enable_context_analysis")

            logger.info(f"更新上下文感知RAG服务配置: {updated_params}")
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

    def test_contextual_pipeline(self, test_query: str = "这是一个测试查询") -> Dict[str, Any]:
        """测试上下文感知查询流水线"""
        try:
            # 创建测试会话
            test_session_id = self.create_session()

            # 执行测试查询
            result = self.query(test_query, test_session_id, top_k=2)

            # 获取会话信息
            session_info = self.get_session_info(test_session_id)

            # 清理测试会话
            self.delete_session(test_session_id)

            return {
                "status": "completed",
                "test_query": test_query,
                "test_session_id": test_session_id,
                "query_result": result,
                "session_info": session_info,
                "overall_health": "healthy" if result.get("status") == "success" else "unhealthy"
            }

        except Exception as e:
            error_msg = f"上下文流水线测试失败: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "error": str(e)
            }