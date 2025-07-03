"""
ElasticSearch向量存储实现
支持语义搜索、混合检索、命题递归检索和事件检索
"""

import logging
import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema import Document
from elasticsearch import Elasticsearch

from core.config.settings import settings
from core.exceptions.errors import VectorStoreException
from components.embeddings.bge_embeddings import BGEEmbeddings

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore:
    """ElasticSearch向量存储实现"""

    def __init__(self,
                 index_name: Optional[str] = None,
                 es_url: Optional[str] = None):
        """
        初始化ElasticSearch向量存储

        Args:
            index_name: 索引名称
            es_url: ElasticSearch URL
        """
        self.es_url = es_url or settings.ELASTICSEARCH_URL
        self.index_name = index_name or settings.ELASTICSEARCH_INDEX_NAME

        try:
            # 初始化ES客户端
            self.es_client = Elasticsearch(
                self.es_url,
                retry_on_timeout=settings.ELASTICSEARCH_RETRY_ON_TIMEOUT,
                request_timeout=settings.ELASTICSEARCH_TIMEOUT
            )

            # 初始化嵌入模型
            self.embeddings = BGEEmbeddings()

            # 初始化LangChain ElasticsearchStore
            self.vector_store = ElasticsearchStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                es_connection=self.es_client
            )

            logger.info(f"ElasticSearch向量存储初始化成功: {self.es_url}/{self.index_name}")

        except Exception as e:
            error_msg = f"ElasticSearch初始化失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="ES_INIT_ERROR")

    def create_index_if_not_exists(self) -> None:
        """创建增强的索引结构，支持事件数据"""
        if self.es_client.indices.exists(index=self.index_name):
            logger.debug(f"索引已存在: {self.index_name}")
            return

        try:
            # 创建增强的索引映射
            mappings = {
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",  # 使用标准分析器
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "vector": {
                            "type": "dense_vector",
                            "dims": self.embeddings.dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        # 分块相关字段
                        "chunk_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "document_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "prev_chunk_id": {"type": "keyword"},
                        "next_chunk_id": {"type": "keyword"},

                        # 命题相关字段
                        "is_proposition": {"type": "boolean"},
                        "semantic_level": {"type": "integer"},
                        "parent_node_id": {"type": "keyword"},

                        # 事件相关字段
                        "is_event": {"type": "boolean"},
                        "event_type": {"type": "keyword"},
                        "event_title": {"type": "text"},
                        "event_time": {"type": "text"},
                        "event_location": {"type": "text"},
                        "event_participants": {"type": "keyword"},  # 支持多值
                        "event_status": {"type": "keyword"},

                        # 文件类型字段
                        "file_type": {"type": "keyword"},  # "chat" 或 "document"
                        "is_chat_file": {"type": "boolean"},
                        "chat_participants": {"type": "keyword"},  # 支持多值
                        "is_backup_chunk": {"type": "boolean"},
                        "has_events": {"type": "boolean"},

                        # 通用元数据
                        "metadata": {"type": "object"}
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "refresh_interval": "1s"
                    }
                }
            }

            self.es_client.indices.create(index=self.index_name, body=mappings)
            logger.info(f"成功创建增强索引: {self.index_name}")

        except Exception as e:
            error_msg = f"创建索引失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="CREATE_INDEX_ERROR")

    import uuid
    import os

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量存储 - 完全修复版本
        绕过LangChain直接使用ElasticSearch API

        Args:
            documents: 文档列表

        Returns:
            文档ID列表

        Raises:
            VectorStoreException: 当添加失败时
        """
        if not documents:
            return []

        try:
            # 直接使用ES客户端批量插入，完全绕过LangChain
            doc_ids = []
            bulk_body = []

            # 统计调试信息
            debug_counts = {
                "total": 0,
                "events": 0,
                "propositions": 0,
                "chat_files": 0,
                "backup_chunks": 0,
                "regular": 0,
                "missing_chunk_id": 0
            }

            for doc in documents:
                if not doc.metadata:
                    logger.warning("文档缺少元数据，跳过")
                    continue

                metadata = doc.metadata.copy()
                chunk_id = metadata.get("chunk_id", "")

                # 如果没有chunk_id，生成一个可靠的ID
                if not chunk_id:
                    document_id = metadata.get("document_id", "")
                    base_name = metadata.get("base_name", "")
                    chunk_index = metadata.get("chunk_index", 0)

                    if base_name:
                        chunk_id = f"{base_name}_chunk_{chunk_index}"
                    elif document_id:
                        base_name = os.path.splitext(os.path.basename(document_id))[0]
                        chunk_id = f"{base_name}_chunk_{chunk_index}"
                    else:
                        chunk_id = f"doc_{uuid.uuid4().hex[:8]}_chunk_{chunk_index}"

                    metadata["chunk_id"] = chunk_id
                    debug_counts["missing_chunk_id"] += 1
                    logger.warning(f"生成缺失的chunk_id: {chunk_id}")

                # 生成向量嵌入
                try:
                    text_vector = self.embeddings.embed_query(doc.page_content)
                except Exception as e:
                    logger.error(f"生成向量嵌入失败 {chunk_id}: {e}")
                    continue

                # 提取布尔字段并确保类型正确
                is_event = bool(metadata.get("is_event", False))
                is_proposition = bool(metadata.get("is_proposition", False))
                is_chat_file = bool(metadata.get("is_chat_file", False))
                is_backup_chunk = bool(metadata.get("is_backup_chunk", False))
                has_events = bool(metadata.get("has_events", False))

                # 统计调试信息
                debug_counts["total"] += 1
                if is_event:
                    debug_counts["events"] += 1
                if is_proposition:
                    debug_counts["propositions"] += 1
                if is_chat_file:
                    debug_counts["chat_files"] += 1
                if is_backup_chunk:
                    debug_counts["backup_chunks"] += 1
                if not is_event and not is_proposition:
                    debug_counts["regular"] += 1

                # 构建完整的ES文档 - 确保所有字段类型正确
                es_doc = {
                    "text": str(doc.page_content),
                    "vector": text_vector,

                    # 关键字段在顶层 - 确保数据类型正确
                    "chunk_id": str(chunk_id),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "total_chunks": int(metadata.get("total_chunks", 1)),
                    "document_id": str(metadata.get("document_id", "")),
                    "filename": str(metadata.get("filename", "")),
                    "prev_chunk_id": str(metadata.get("prev_chunk_id", "")),
                    "next_chunk_id": str(metadata.get("next_chunk_id", "")),
                    "parent_node_id": str(metadata.get("parent_node_id", "")),

                    # 布尔字段 - 明确使用Python布尔类型
                    "is_proposition": is_proposition,
                    "is_event": is_event,
                    "is_chat_file": is_chat_file,
                    "is_backup_chunk": is_backup_chunk,
                    "has_events": has_events,

                    # 数值字段
                    "semantic_level": int(metadata.get("semantic_level", 0)),

                    # 事件相关字段
                    "event_type": str(metadata.get("event_type", "")),
                    "event_title": str(metadata.get("event_title", "")),
                    "event_time": str(metadata.get("event_time", "")),
                    "event_location": str(metadata.get("event_location", "")),
                    "event_participants": list(metadata.get("event_participants", [])),
                    "event_status": str(metadata.get("event_status", "")),

                    # 文件类型字段
                    "file_type": str(metadata.get("file_type", "document")),
                    "chat_participants": list(metadata.get("chat_participants", [])),

                    # 完整的元数据对象（用于备份和兼容性）
                    "metadata": metadata
                }

                # 调试日志（仅前3个文档）
                if debug_counts["total"] <= 3:
                    logger.debug(f"准备存储文档 {chunk_id}:")
                    logger.debug(f"  is_event: {es_doc['is_event']} (type: {type(es_doc['is_event'])})")
                    logger.debug(
                        f"  is_proposition: {es_doc['is_proposition']} (type: {type(es_doc['is_proposition'])})")
                    logger.debug(f"  is_chat_file: {es_doc['is_chat_file']} (type: {type(es_doc['is_chat_file'])})")
                    logger.debug(f"  document_id: {es_doc['document_id']}")

                # 添加到批量操作
                bulk_body.extend([
                    {"index": {"_index": self.index_name, "_id": chunk_id}},
                    es_doc
                ])
                doc_ids.append(chunk_id)

            logger.info(f"准备批量插入 {len(doc_ids)} 个文档")
            logger.info(f"文档类型统计: {debug_counts}")

            # 执行批量插入
            if bulk_body:
                response = self.es_client.bulk(body=bulk_body, refresh=True)

                # 检查批量操作结果
                success_count = 0
                error_count = 0

                if response.get("errors"):
                    for item in response["items"]:
                        if "error" in item.get("index", {}):
                            error_count += 1
                            if error_count <= 3:  # 只记录前3个错误
                                logger.error(f"插入错误: {item['index']['error']}")
                        else:
                            success_count += 1

                    logger.warning(f"批量插入部分失败，成功: {success_count}, 失败: {error_count}")
                else:
                    success_count = len(doc_ids)
                    logger.info(f"批量插入全部成功: {success_count}")

            # 刷新索引确保立即可搜索
            self.es_client.indices.refresh(index=self.index_name)

            # 验证存储结果
            if doc_ids:
                verification_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": min(3, len(doc_ids)),
                        "query": {"terms": {"chunk_id": doc_ids[:3]}},
                        "_source": ["chunk_id", "is_event", "is_proposition", "is_chat_file", "is_backup_chunk",
                                    "document_id"]
                    }
                )

                logger.info(f"存储验证结果 (查询了 {min(3, len(doc_ids))} 个文档):")
                found_docs = verification_response["hits"]["hits"]

                if found_docs:
                    for hit in found_docs:
                        source = hit["_source"]
                        logger.info(f"  ✓ 文档 {source.get('chunk_id', 'unknown')}: "
                                    f"is_event={source.get('is_event')}, "
                                    f"is_proposition={source.get('is_proposition')}, "
                                    f"is_chat_file={source.get('is_chat_file')}, "
                                    f"document_id={source.get('document_id', 'N/A')}")
                else:
                    logger.warning("❌ 验证查询未找到任何文档！")

            # 最终统计
            final_event_count = debug_counts["events"]
            final_prop_count = debug_counts["propositions"]
            final_chat_count = debug_counts["chat_files"]

            logger.info(f"成功添加 {len(documents)} 个文档到向量存储, "
                        f"事件: {final_event_count}, 命题: {final_prop_count}, 聊天: {final_chat_count}")

            return doc_ids

        except Exception as e:
            error_msg = f"添加文档到向量存储失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreException(error_msg, error_code="ADD_DOCUMENTS_ERROR")
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        智能搜索 - 根据查询内容自动选择最优搜索策略

        Args:
            query: 查询文本
            k: 返回的最大文档数量

        Returns:
            检索到的文档列表

        Raises:
            VectorStoreException: 当搜索失败时
        """
        try:
            # 分析查询类型
            query_type = self._analyze_query_type(query)

            if query_type == "event":
                # 事件相关查询，优先搜索事件
                return self._event_search(query, k)
            elif query_type == "chat":
                # 聊天相关查询，搜索聊天和事件
                return self._chat_search(query, k)
            else:
                # 普通查询，智能混合搜索
                return self._intelligent_search(query, k)

        except Exception as e:
            error_msg = f"智能搜索失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="SEARCH_ERROR")

    def _analyze_query_type(self, query: str) -> str:
        """分析查询类型"""
        # 事件关键词
        event_keywords = ["会议", "开会", "吃饭", "用餐", "午餐", "晚餐", "购物", "买", "去", "约", "计划", "安排", "时间", "地点"]
        # 聊天关键词
        chat_keywords = ["聊天", "对话", "讨论", "说", "谈", "回复", "消息"]

        query_lower = query.lower()

        # 检查是否包含事件关键词
        if any(keyword in query_lower for keyword in event_keywords):
            return "event"

        # 检查是否包含聊天关键词
        if any(keyword in query_lower for keyword in chat_keywords):
            return "chat"

        return "general"

    def _event_search(self, query: str, k: int = 4) -> List[Document]:
        """事件专用搜索"""
        try:
            query_vector = self.embeddings.embed_query(query)

            # 构建事件搜索查询
            search_body = {
                "size": k * 2,
                "query": {
                    "bool": {
                        "should": [
                            # 语义搜索
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                        "params": {"query_vector": query_vector}
                                    },
                                    "boost": 2.0
                                }
                            },
                            # 事件标题匹配
                            {"match": {"event_title": {"query": query, "boost": 3.0}}},
                            # 事件内容匹配
                            {"match": {"text": {"query": query, "boost": 1.5}}},
                        ],
                        "filter": [
                            {"term": {"is_event": True}}
                        ]
                    }
                }
            }

            response = self.es_client.search(index=self.index_name, body=search_body)

            docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["search_score"] = hit["_score"]
                metadata["search_type"] = "event"

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                docs.append(doc)

            logger.debug(f"事件搜索完成，结果数量: {len(docs)}")
            return docs[:k]

        except Exception as e:
            logger.warning(f"事件搜索失败，回退到基本搜索: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def _chat_search(self, query: str, k: int = 4) -> List[Document]:
        """聊天专用搜索"""
        try:
            query_vector = self.embeddings.embed_query(query)

            # 构建聊天搜索查询
            search_body = {
                "size": k * 2,
                "query": {
                    "bool": {
                        "should": [
                            # 语义搜索
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                        "params": {"query_vector": query_vector}
                                    },
                                    "boost": 1.0
                                }
                            },
                            # 文本内容匹配
                            {"match": {"text": {"query": query, "boost": 2.0}}},
                        ],
                        "filter": [
                            {
                                "bool": {
                                    "should": [
                                        {"term": {"is_chat_file": True}},
                                        {"term": {"is_event": True}}
                                    ]
                                }
                            }
                        ]
                    }
                }
            }

            response = self.es_client.search(index=self.index_name, body=search_body)

            docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["search_score"] = hit["_score"]
                metadata["search_type"] = "chat"

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                docs.append(doc)

            logger.debug(f"聊天搜索完成，结果数量: {len(docs)}")
            return docs[:k]

        except Exception as e:
            logger.warning(f"聊天搜索失败，回退到基本搜索: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def _intelligent_search(self, query: str, k: int = 4) -> List[Document]:
        """智能混合搜索"""
        try:
            # 检查是否有事件数据
            has_events = self._has_events()
            # 检查是否有命题数据
            has_propositions = self._has_propositions()

            if has_events:
                # 如果有事件数据，使用多策略搜索
                return self._multi_strategy_search(query, k)
            elif has_propositions:
                # 如果有命题数据，使用命题搜索
                return self._proposition_search(query, k)
            else:
                # 回退到基本语义搜索
                docs = self.vector_store.similarity_search(query, k=k)
                docs = self._enhance_consecutive_chunks(docs)
                return docs[:k]

        except Exception as e:
            logger.warning(f"智能搜索失败，回退到基本搜索: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def _multi_strategy_search(self, query: str, k: int = 4) -> List[Document]:
        """多策略搜索 - 结合事件、命题和普通文档"""
        try:
            query_vector = self.embeddings.embed_query(query)

            # 分别搜索不同类型的内容
            event_results = self._event_search(query, k//2 + 1)
            doc_results = self.vector_store.similarity_search(query, k=k//2 + 1)

            # 合并结果并去重
            all_results = []
            seen_ids = set()

            # 添加事件结果
            for doc in event_results:
                doc_id = doc.metadata.get("chunk_id", "")
                if doc_id not in seen_ids:
                    all_results.append(doc)
                    seen_ids.add(doc_id)

            # 添加文档结果
            for doc in doc_results:
                doc_id = doc.metadata.get("chunk_id", "")
                if doc_id not in seen_ids:
                    all_results.append(doc)
                    seen_ids.add(doc_id)

            # 按相关性分数排序
            all_results.sort(key=lambda x: x.metadata.get("search_score", 0), reverse=True)

            logger.debug(f"多策略搜索完成，事件: {len(event_results)}, 文档: {len(doc_results)}, 合并后: {len(all_results)}")
            return all_results[:k]

        except Exception as e:
            logger.warning(f"多策略搜索失败: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def search_events(self,
                      query: str = None,
                      event_type: str = None,
                      participants: List[str] = None,
                      time_range: str = None,
                      k: int = 4) -> List[Document]:
        """
        专门的事件搜索接口 - 修复参与者筛选

        Args:
            query: 查询文本
            event_type: 事件类型
            participants: 参与者列表
            time_range: 时间范围
            k: 返回的最大文档数量

        Returns:
            检索到的事件文档列表
        """
        try:
            logger.info(f"执行事件搜索 - 查询: {query}, 参与者: {participants}, 类型: {event_type}")

            # 首先检查是否有事件数据
            if not self._has_events():
                logger.warning("索引中没有事件数据")
                return []

            search_body = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"is_event": True}}
                        ],
                        "should": [],
                        "filter": []
                    }
                }
            }

            # 添加文本查询
            if query:
                search_body["query"]["bool"]["should"].extend([
                    {"match": {"event_title": {"query": query, "boost": 3.0}}},
                    {"match": {"text": {"query": query, "boost": 1.5}}},
                ])

            # 添加事件类型过滤
            if event_type:
                search_body["query"]["bool"]["filter"].append(
                    {"term": {"event_type": event_type}}
                )

            # 修复参与者过滤 - 关键修复点
            if participants:
                logger.info(f"添加参与者筛选: {participants}")

                # 清理参与者列表
                clean_participants = []
                for p in participants:
                    if isinstance(p, str):
                        cleaned = p.strip()
                        if cleaned:
                            clean_participants.append(cleaned)
                    elif p:
                        clean_participants.append(str(p).strip())

                if clean_participants:
                    logger.info(f"清理后的参与者列表: {clean_participants}")

                    # 使用改进的查询策略
                    participant_filter = {
                        "bool": {
                            "should": [
                                # 策略1: 精确匹配任一参与者
                                {"terms": {"event_participants": clean_participants}},
                                # 策略2: 使用keyword字段匹配（如果存在）
                                {"terms": {"event_participants.keyword": clean_participants}},
                                # 策略3: 文本匹配（模糊匹配）
                                {
                                    "bool": {
                                        "should": [
                                            {"match": {
                                                "event_participants": {
                                                    "query": " ".join(clean_participants),
                                                    "operator": "or"
                                                }
                                            }}
                                        ]
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    }

                    search_body["query"]["bool"]["filter"].append(participant_filter)

            # 确保有查询条件
            if not search_body["query"]["bool"]["should"] and not query:
                search_body["query"]["bool"]["should"].append({"match_all": {}})

            logger.debug(f"事件搜索查询体: {search_body}")

            # 执行搜索
            response = self.es_client.search(index=self.index_name, body=search_body)

            docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["search_score"] = hit["_score"]

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                docs.append(doc)

            logger.info(f"事件搜索完成，返回 {len(docs)} 个结果")
            return docs

        except Exception as e:
            error_msg = f"事件搜索失败: {str(e)}"
            logger.error(error_msg)

            # 如果复杂查询失败，尝试简单查询
            try:
                logger.info("尝试简化事件搜索")
                return self._simple_participants_search(query, event_type, participants, k)
            except Exception as fallback_error:
                logger.error(f"简化事件搜索也失败: {fallback_error}")
                raise VectorStoreException(error_msg, error_code="EVENT_SEARCH_ERROR")

    def _simple_participants_search(self, query: str = None, event_type: str = None,
                                    participants: List[str] = None, k: int = 4) -> List[Document]:
        """简化的参与者搜索，只使用基本匹配"""
        try:
            search_body = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [{"term": {"is_event": True}}],
                        "should": [],
                        "filter": []
                    }
                }
            }

            # 文本查询
            if query:
                search_body["query"]["bool"]["should"].extend([
                    {"match": {"event_title": {"query": query, "boost": 2.0}}},
                    {"match": {"text": {"query": query, "boost": 1.0}}},
                ])

            # 事件类型
            if event_type:
                search_body["query"]["bool"]["filter"].append(
                    {"term": {"event_type": event_type}}
                )

            # 参与者 - 使用最简单的匹配方式
            if participants:
                for participant in participants:
                    if participant.strip():
                        search_body["query"]["bool"]["should"].append(
                            {"match": {"event_participants": participant.strip()}}
                        )

            # 如果没有搜索条件，使用match_all
            if not search_body["query"]["bool"]["should"]:
                search_body["query"]["bool"]["should"].append({"match_all": {}})

            response = self.es_client.search(index=self.index_name, body=search_body)

            docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["search_score"] = hit["_score"]
                metadata["search_type"] = "simple_participants"

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                docs.append(doc)

            logger.info(f"简化参与者搜索完成，返回 {len(docs)} 个结果")
            return docs

        except Exception as e:
            logger.error(f"简化参与者搜索失败: {e}")
            return []


    def keyword_search(self, query: str, k: int = 4) -> List[Document]:
        """
        关键词搜索（BM25）

        Args:
            query: 查询文本
            k: 返回的最大文档数量

        Returns:
            检索到的文档列表
        """
        try:
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"text": {"query": query, "boost": 2.0}}},
                            {"match_phrase": {"text": {"query": query, "boost": 3.0, "slop": 2}}},
                            # 为事件添加特殊的匹配
                            {"match": {"event_title": {"query": query, "boost": 4.0}}},
                        ]
                    }
                },
                "size": k
            }

            response = self.es_client.search(
                index=self.index_name,
                body=search_body
            )

            docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["score"] = hit["_score"]

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                docs.append(doc)

            logger.debug(f"关键词搜索完成，结果数量: {len(docs)}")
            return docs

        except Exception as e:
            error_msg = f"关键词搜索失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="KEYWORD_SEARCH_ERROR")

    def _proposition_search(self, query: str, k: int = 4, include_context: bool = True) -> List[Document]:
        """基于命题的递归检索"""
        try:
            query_vector = self.embeddings.embed_query(query)

            # 第一阶段：检索匹配的命题
            proposition_query = {
                "size": k * 2,
                "query": {
                    "bool": {
                        "must": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                    "params": {"query_vector": query_vector}
                                }
                            }
                        },
                        "filter": [
                            {"term": {"is_proposition": True}}
                        ]
                    }
                }
            }

            prop_response = self.es_client.search(
                index=self.index_name,
                body=proposition_query
            )

            # 处理命题结果
            prop_docs = []
            parent_ids = set()

            for hit in prop_response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["search_score"] = hit["_score"]

                parent_id = source.get("parent_node_id")
                if parent_id:
                    parent_ids.add(parent_id)

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                prop_docs.append(doc)

            if not include_context or not parent_ids:
                return prop_docs[:k]

            # 第二阶段：检索父节点内容
            parent_query = {
                "size": len(parent_ids),
                "query": {
                    "terms": {
                        "chunk_id": list(parent_ids)
                    }
                }
            }

            parent_response = self.es_client.search(
                index=self.index_name,
                body=parent_query
            )

            # 处理父节点结果
            parent_docs = []
            for hit in parent_response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                metadata["is_context"] = True

                doc = Document(
                    page_content=source.get("text", ""),
                    metadata=metadata
                )
                parent_docs.append(doc)

            # 合并并排序结果
            results = self._merge_and_rank_results(prop_docs, parent_docs, k)
            return results

        except Exception as e:
            logger.warning(f"命题搜索失败，回退到基本搜索: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def _has_events(self) -> bool:
        """检查索引中是否有事件数据"""
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "query": {"term": {"is_event": True}}
                }
            )
            return response["hits"]["total"]["value"] > 0
        except Exception:
            return False

    def _has_propositions(self) -> bool:
        """检查索引中是否有命题数据"""
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "query": {"term": {"is_proposition": True}}
                }
            )
            return response["hits"]["total"]["value"] > 0
        except Exception:
            return False

    def _enhance_consecutive_chunks(self, docs: List[Document]) -> List[Document]:
        """增强连续块的处理"""
        if len(docs) <= 1:
            return docs

        # 按文档ID分组
        doc_groups = {}
        for doc in docs:
            doc_id = doc.metadata.get("document_id", "")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(doc)

        # 处理连续块
        for doc_id, group_docs in doc_groups.items():
            if len(group_docs) > 1:
                # 按chunk索引排序
                group_docs.sort(key=lambda x: x.metadata.get("chunk_index", 0))

                # 检查连续性
                for i in range(1, len(group_docs)):
                    curr_idx = group_docs[i].metadata.get("chunk_index", -1)
                    prev_idx = group_docs[i-1].metadata.get("chunk_index", -2)
                    if curr_idx - prev_idx == 1:
                        # 标记为连续上下文
                        group_docs[i].metadata["consecutive_context"] = True
                        group_docs[i-1].metadata["consecutive_context"] = True

        # 连续上下文的文档优先
        docs.sort(key=lambda x: x.metadata.get("consecutive_context", False), reverse=True)
        return docs

    def _merge_and_rank_results(self, prop_docs: List[Document], parent_docs: List[Document], k: int) -> List[Document]:
        """合并并排序命题和父节点文档"""
        parent_map = {doc.metadata.get("chunk_id", ""): doc for doc in parent_docs}

        enhanced_results = []
        for doc in prop_docs:
            parent_id = doc.metadata.get("parent_node_id", "")

            if parent_id in parent_map:
                parent_doc = parent_map[parent_id]
                doc.metadata["parent_content"] = parent_doc.page_content
                doc.metadata["parent_id"] = parent_id

            enhanced_results.append(doc)

        # 按搜索分数排序
        enhanced_results.sort(key=lambda x: x.metadata.get("search_score", 0), reverse=True)
        return enhanced_results[:k]

    def delete_index(self) -> None:
        """删除索引"""
        try:
            if self.es_client.indices.exists(index=self.index_name):
                self.es_client.indices.delete(index=self.index_name)
                logger.info(f"成功删除索引: {self.index_name}")
        except Exception as e:
            error_msg = f"删除索引失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="DELETE_INDEX_ERROR")

    def get_document_count(self) -> int:
        """获取索引中的文档数量"""
        try:
            result = self.es_client.count(index=self.index_name)
            return result['count']
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    # 在 elasticsearch_store.py 中完全替换 get_statistics 方法

    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        try:
            # 基础统计
            total_docs = self.get_document_count()
            stats = {
                "total_documents": total_docs,
                "events": 0,
                "propositions": 0,
                "chat_files": 0,
                "regular_documents": 0,
                "event_types": {},
                "file_types": {},
                "detailed_breakdown": {}
            }

            # 1. 统计事件数量和类型分布
            try:
                event_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {"term": {"is_event": True}},
                        "aggs": {
                            "event_types": {
                                "terms": {"field": "event_type", "size": 20}
                            }
                        }
                    }
                )
                stats["events"] = event_response["hits"]["total"]["value"]
                if "aggregations" in event_response:
                    for bucket in event_response["aggregations"]["event_types"]["buckets"]:
                        stats["event_types"][bucket["key"]] = bucket["doc_count"]
            except Exception as e:
                logger.warning(f"获取事件统计失败: {e}")

            # 2. 统计命题数量
            try:
                prop_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {"term": {"is_proposition": True}}
                    }
                )
                stats["propositions"] = prop_response["hits"]["total"]["value"]
            except Exception as e:
                logger.warning(f"获取命题统计失败: {e}")

            # 3. 统计聊天文件数量（标记为聊天的文档块）
            try:
                chat_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {"term": {"is_chat_file": True}}
                    }
                )
                stats["chat_files"] = chat_response["hits"]["total"]["value"]
            except Exception as e:
                logger.warning(f"获取聊天文件统计失败: {e}")

            # 4. 统计文件类型分布
            try:
                file_type_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "aggs": {
                            "file_types": {
                                "terms": {"field": "file_type", "size": 10}
                            }
                        }
                    }
                )
                if "aggregations" in file_type_response:
                    for bucket in file_type_response["aggregations"]["file_types"]["buckets"]:
                        stats["file_types"][bucket["key"]] = bucket["doc_count"]
            except Exception as e:
                logger.warning(f"获取文件类型统计失败: {e}")

            # 5. 精确统计普通文档数量（既不是事件也不是命题的文档）
            try:
                regular_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {
                            "bool": {
                                "must_not": [
                                    {"term": {"is_event": True}},
                                    {"term": {"is_proposition": True}}
                                ]
                            }
                        }
                    }
                )
                stats["regular_documents"] = regular_response["hits"]["total"]["value"]
            except Exception as e:
                logger.warning(f"获取普通文档统计失败: {e}")
                # 如果查询失败，使用计算方式作为后备
                stats["regular_documents"] = max(0, total_docs - stats["events"] - stats["propositions"])

            # 6. 详细分类统计
            detailed_breakdown = {
                "only_events": 0,
                "only_propositions": 0,
                "only_regular": 0,
                "event_and_proposition": 0,
                "chat_with_events": 0,
                "document_with_events": 0,
                "backup_chunks": 0
            }

            try:
                # 统计只有事件标记的文档
                only_events_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {
                            "bool": {
                                "must": [{"term": {"is_event": True}}],
                                "must_not": [{"term": {"is_proposition": True}}]
                            }
                        }
                    }
                )
                detailed_breakdown["only_events"] = only_events_response["hits"]["total"]["value"]

                # 统计只有命题标记的文档
                only_props_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {
                            "bool": {
                                "must": [{"term": {"is_proposition": True}}],
                                "must_not": [{"term": {"is_event": True}}]
                            }
                        }
                    }
                )
                detailed_breakdown["only_propositions"] = only_props_response["hits"]["total"]["value"]

                # 统计既是事件又是命题的文档
                both_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"is_event": True}},
                                    {"term": {"is_proposition": True}}
                                ]
                            }
                        }
                    }
                )
                detailed_breakdown["event_and_proposition"] = both_response["hits"]["total"]["value"]

                # 统计聊天文件中的事件
                chat_events_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"is_event": True}},
                                    {"term": {"is_chat_file": True}}
                                ]
                            }
                        }
                    }
                )
                detailed_breakdown["chat_with_events"] = chat_events_response["hits"]["total"]["value"]

                # 统计普通文档中的事件
                doc_events_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {
                            "bool": {
                                "must": [{"term": {"is_event": True}}],
                                "must_not": [{"term": {"is_chat_file": True}}]
                            }
                        }
                    }
                )
                detailed_breakdown["document_with_events"] = doc_events_response["hits"]["total"]["value"]

                # 统计备份块
                backup_response = self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "query": {"term": {"is_backup_chunk": True}}
                    }
                )
                detailed_breakdown["backup_chunks"] = backup_response["hits"]["total"]["value"]

            except Exception as e:
                logger.warning(f"获取详细分类统计失败: {e}")

            detailed_breakdown["only_regular"] = stats["regular_documents"]
            stats["detailed_breakdown"] = detailed_breakdown

            # 7. 验证统计数据的一致性
            calculated_total = (detailed_breakdown["only_events"] +
                                detailed_breakdown["only_propositions"] +
                                detailed_breakdown["only_regular"] +
                                detailed_breakdown["event_and_proposition"])

            stats["consistency_check"] = {
                "calculated_total": calculated_total,
                "actual_total": stats["total_documents"],
                "is_consistent": calculated_total == stats["total_documents"],
                "difference": stats["total_documents"] - calculated_total
            }

            # 8. 添加索引健康状态
            stats["index_health"] = {
                "has_events": stats["events"] > 0,
                "has_propositions": stats["propositions"] > 0,
                "has_chat_files": stats["chat_files"] > 0,
                "has_regular_docs": stats["regular_documents"] > 0,
                "total_indexed": stats["total_documents"] > 0
            }

            # 9. 添加数据分布百分比
            if stats["total_documents"] > 0:
                stats["distribution_percentages"] = {
                    "events_pct": round((stats["events"] / stats["total_documents"]) * 100, 2),
                    "propositions_pct": round((stats["propositions"] / stats["total_documents"]) * 100, 2),
                    "regular_pct": round((stats["regular_documents"] / stats["total_documents"]) * 100, 2),
                    "chat_files_pct": round((stats["chat_files"] / stats["total_documents"]) * 100, 2)
                }
            else:
                stats["distribution_percentages"] = {
                    "events_pct": 0,
                    "propositions_pct": 0,
                    "regular_pct": 0,
                    "chat_files_pct": 0
                }

            logger.info(f"统计信息获取完成: 总计{total_docs}, 事件{stats['events']}, "
                        f"命题{stats['propositions']}, 普通{stats['regular_documents']}, "
                        f"聊天{stats['chat_files']}")

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}", exc_info=True)
            return {
                "total_documents": 0,
                "error": str(e),
                "events": 0,
                "propositions": 0,
                "chat_files": 0,
                "regular_documents": 0,
                "event_types": {},
                "file_types": {},
                "detailed_breakdown": {},
                "consistency_check": {"is_consistent": False},
                "index_health": {"total_indexed": False}
            }