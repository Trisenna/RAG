"""
检索器实现
包含基础检索器、混合检索器、事件检索器等
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
        # 检查是否为事件相关过滤
        if self._is_event_filter(filters):
            return self._retrieve_events_with_filters(query, filters, top_k)

        # TODO: 实现其他类型的过滤
        logger.warning("通用过滤检索功能待实现，使用基础检索")
        return self.retrieve(query, top_k)

    def _is_event_filter(self, filters: Dict[str, Any]) -> bool:
        """判断是否为事件相关过滤"""
        event_filter_keys = ["event_type", "event_participants", "event_time", "event_location"]
        return any(key in filters for key in event_filter_keys)

    def _retrieve_events_with_filters(self,
                                    query: str,
                                    filters: Dict[str, Any],
                                    top_k: int) -> List[Document]:
        """事件过滤检索"""
        try:
            return self.vector_store.search_events(
                query=query,
                event_type=filters.get("event_type"),
                participants=filters.get("event_participants"),
                time_range=filters.get("event_time"),
                k=top_k
            )
        except Exception as e:
            logger.warning(f"事件过滤检索失败，回退到基础检索: {e}")
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


class EventRetriever(BaseRetriever):
    """专门的事件检索器"""

    def __init__(self, vector_store: ElasticsearchVectorStore):
        """
        初始化事件检索器

        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        logger.info("事件检索器初始化完成")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索相关事件

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            检索到的事件列表
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            # 使用向量存储的事件搜索功能
            events = self.vector_store.search_events(query=query, k=top_k)

            logger.debug(f"事件检索完成，查询: {query[:50]}..., 结果数量: {len(events)}")
            return events

        except Exception as e:
            error_msg = f"事件检索失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="EVENT_RETRIEVAL_ERROR")

    def retrieve_by_type(self, event_type: str, top_k: int = 10) -> List[Document]:
        """
        按事件类型检索

        Args:
            event_type: 事件类型
            top_k: 返回的最大文档数量

        Returns:
            检索到的事件列表
        """
        try:
            events = self.vector_store.search_events(
                event_type=event_type,
                k=top_k
            )

            logger.debug(f"按类型检索事件完成，类型: {event_type}, 结果数量: {len(events)}")
            return events

        except Exception as e:
            error_msg = f"按类型检索事件失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="EVENT_TYPE_RETRIEVAL_ERROR")

    def retrieve_by_participants(self, participants: List[str], top_k: int = 10) -> List[Document]:
        """
        按参与者检索事件

        Args:
            participants: 参与者列表
            top_k: 返回的最大文档数量

        Returns:
            检索到的事件列表
        """
        try:
            events = self.vector_store.search_events(
                participants=participants,
                k=top_k
            )

            logger.debug(f"按参与者检索事件完成，参与者: {participants}, 结果数量: {len(events)}")
            return events

        except Exception as e:
            error_msg = f"按参与者检索事件失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="EVENT_PARTICIPANTS_RETRIEVAL_ERROR")


class IntelligentRetriever(BaseRetriever):
    """智能检索器 - 根据查询内容自动选择最优检索策略"""

    def __init__(self,
                 vector_store: ElasticsearchVectorStore,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        初始化智能检索器

        Args:
            vector_store: 向量存储实例
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
        """
        self.vector_store = vector_store

        # 初始化子检索器
        self.basic_retriever = BasicRetriever(vector_store)
        self.hybrid_retriever = HybridRetriever(vector_store, semantic_weight, keyword_weight)
        self.event_retriever = EventRetriever(vector_store)

        logger.info("智能检索器初始化完成")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        智能检索 - 根据查询内容自动选择策略

        Args:
            query: 查询文本
            top_k: 返回的最大文档数量

        Returns:
            检索到的文档列表
        """
        try:
            if not query.strip():
                raise RetrievalException("查询文本不能为空", error_code="EMPTY_QUERY")

            # 分析查询类型和意图
            query_analysis = self._analyze_query(query)

            # 根据分析结果选择检索策略
            strategy = self._select_strategy(query_analysis)

            # 执行检索
            results = self._execute_strategy(strategy, query, query_analysis, top_k)

            logger.debug(f"智能检索完成，查询: {query[:50]}..., 策略: {strategy}, 结果数量: {len(results)}")
            return results

        except Exception as e:
            error_msg = f"智能检索失败: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg, error_code="INTELLIGENT_RETRIEVAL_ERROR")

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询内容和意图"""
        analysis = {
            "is_event_query": False,
            "is_chat_query": False,
            "is_temporal_query": False,
            "is_person_query": False,
            "event_type": None,
            "participants": [],
            "confidence": 0.0
        }

        query_lower = query.lower()

        # 事件相关关键词
        event_keywords = {
            "meeting_plan": ["会议", "开会", "讨论", "商量", "决定"],
            "meal_plan": ["吃饭", "用餐", "午餐", "晚餐", "早餐", "聚餐", "饭店", "餐厅"],
            "travel_plan": ["去", "回", "出差", "旅行", "交通", "地铁", "公交", "打车"],
            "shopping_plan": ["买", "购物", "商场", "逛街", "键盘", "电脑"],
            "work_task": ["工作", "项目", "任务", "完成", "截止", "加班"],
            "social_event": ["约会", "聚会", "见面", "陪", "一起"]
        }

        # 时间相关关键词
        time_keywords = ["时间", "几点", "什么时候", "今天", "明天", "昨天", "周", "月", "年", "点半"]

        # 人物相关关键词
        person_keywords = ["谁", "人", "和", "陪", "一起", "参与"]

        # 聊天相关关键词
        chat_keywords = ["聊天", "对话", "讨论", "说", "谈", "回复", "消息"]

        # 检查事件类型
        max_event_matches = 0
        best_event_type = None

        for event_type, keywords in event_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_event_matches:
                max_event_matches = matches
                best_event_type = event_type

        if max_event_matches > 0:
            analysis["is_event_query"] = True
            analysis["event_type"] = best_event_type
            analysis["confidence"] = min(max_event_matches * 0.3, 1.0)

        # 检查时间查询
        if any(keyword in query_lower for keyword in time_keywords):
            analysis["is_temporal_query"] = True
            analysis["confidence"] += 0.2

        # 检查人物查询
        if any(keyword in query_lower for keyword in person_keywords):
            analysis["is_person_query"] = True
            analysis["confidence"] += 0.2

        # 检查聊天查询
        if any(keyword in query_lower for keyword in chat_keywords):
            analysis["is_chat_query"] = True
            analysis["confidence"] += 0.3

        return analysis

    def _select_strategy(self, analysis: Dict[str, Any]) -> str:
        """根据分析结果选择检索策略"""

        # 如果是高置信度的事件查询
        if analysis["is_event_query"] and analysis["confidence"] > 0.5:
            return "event_focused"

        # 如果是聊天相关查询
        if analysis["is_chat_query"]:
            return "chat_focused"

        # 如果包含时间或人物信息
        if analysis["is_temporal_query"] or analysis["is_person_query"]:
            return "mixed_event_document"

        # 默认使用混合检索
        return "hybrid"

    def _execute_strategy(self,
                         strategy: str,
                         query: str,
                         analysis: Dict[str, Any],
                         top_k: int) -> List[Document]:
        """执行选定的检索策略"""

        if strategy == "event_focused":
            # 事件优先策略
            return self._event_focused_retrieval(query, analysis, top_k)

        elif strategy == "chat_focused":
            # 聊天优先策略
            return self._chat_focused_retrieval(query, top_k)

        elif strategy == "mixed_event_document":
            # 混合事件和文档策略
            return self._mixed_retrieval(query, analysis, top_k)

        else:
            # 默认混合检索
            return self.hybrid_retriever.retrieve(query, top_k)

    def _event_focused_retrieval(self,
                                query: str,
                                analysis: Dict[str, Any],
                                top_k: int) -> List[Document]:
        """事件优先检索策略"""
        try:
            # 主要检索事件
            events = self.event_retriever.retrieve(query, top_k)

            # 如果事件数量不足，补充一些相关文档
            if len(events) < top_k:
                supplement_k = top_k - len(events)
                docs = self.basic_retriever.retrieve(query, supplement_k)

                # 过滤掉已经包含的文档
                event_ids = {doc.metadata.get("chunk_id", "") for doc in events}
                supplementary_docs = [
                    doc for doc in docs
                    if doc.metadata.get("chunk_id", "") not in event_ids
                ]

                events.extend(supplementary_docs[:supplement_k])

            return events[:top_k]

        except Exception as e:
            logger.warning(f"事件优先检索失败，回退到混合检索: {e}")
            return self.hybrid_retriever.retrieve(query, top_k)

    def _chat_focused_retrieval(self, query: str, top_k: int) -> List[Document]:
        """聊天优先检索策略"""
        try:
            # 使用向量存储的聊天搜索
            return self.vector_store._chat_search(query, top_k)

        except Exception as e:
            logger.warning(f"聊天优先检索失败，回退到混合检索: {e}")
            return self.hybrid_retriever.retrieve(query, top_k)

    def _mixed_retrieval(self,
                        query: str,
                        analysis: Dict[str, Any],
                        top_k: int) -> List[Document]:
        """混合事件和文档检索策略"""
        try:
            # 分配检索数量
            event_k = max(1, top_k // 2)
            doc_k = top_k - event_k

            # 并行检索
            events = self.event_retriever.retrieve(query, event_k)
            docs = self.basic_retriever.retrieve(query, doc_k)

            # 合并结果
            all_results = events + docs

            # 按相关性排序
            all_results.sort(
                key=lambda x: x.metadata.get("search_score", 0),
                reverse=True
            )

            return all_results[:top_k]

        except Exception as e:
            logger.warning(f"混合检索失败，回退到基础检索: {e}")
            return self.basic_retriever.retrieve(query, top_k)

    def retrieve_with_filters(self,
                             query: str,
                             filters: Dict[str, Any],
                             top_k: int = 5) -> List[Document]:
        """
        带过滤条件的智能检索

        Args:
            query: 查询文本
            filters: 过滤条件
            top_k: 返回的最大文档数量

        Returns:
            检索到的文档列表
        """
        try:
            # 检查过滤条件类型
            if self.basic_retriever._is_event_filter(filters):
                return self.basic_retriever._retrieve_events_with_filters(query, filters, top_k)
            else:
                # 对于其他过滤条件，使用标准检索
                return self.retrieve(query, top_k)

        except Exception as e:
            logger.warning(f"过滤检索失败，使用基础检索: {e}")
            return self.basic_retriever.retrieve(query, top_k)

    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """更新混合检索权重"""
        self.hybrid_retriever.set_weights(semantic_weight, keyword_weight)