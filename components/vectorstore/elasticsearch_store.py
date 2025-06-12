"""
ElasticSearch向量存储实现
支持语义搜索、混合检索和命题递归检索
"""

import logging
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
        """创建增强的索引结构"""
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
            logger.info(f"成功创建索引: {self.index_name}")

        except Exception as e:
            error_msg = f"创建索引失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="CREATE_INDEX_ERROR")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量存储

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
            # 处理元数据并转换为ES友好格式
            processed_docs = []
            doc_ids = []

            for doc in documents:
                metadata = doc.metadata.copy() if doc.metadata else {}
                chunk_id = metadata.get("chunk_id", "")

                # 如果没有chunk_id，创建一个
                if not chunk_id:
                    doc_id = metadata.get("source", "unknown")
                    chunk_index = metadata.get("chunk_index", 0)
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    metadata["chunk_id"] = chunk_id

                # 构建ES文档
                processed_doc = {
                    "text": doc.page_content,
                    "metadata": metadata,
                    # 提取关键字段到顶层
                    "chunk_id": chunk_id,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 1),
                    "document_id": metadata.get("document_id", metadata.get("source", "")),
                    "filename": metadata.get("filename", ""),
                    "prev_chunk_id": metadata.get("prev_chunk_id", ""),
                    "next_chunk_id": metadata.get("next_chunk_id", ""),
                    "is_proposition": metadata.get("is_proposition", False),
                    "semantic_level": metadata.get("semantic_level", 0),
                    "parent_node_id": metadata.get("parent_node_id", "")
                }

                processed_docs.append(processed_doc)
                doc_ids.append(chunk_id)

            # 使用vector_store添加文档
            texts = [doc["text"] for doc in processed_docs]
            metadatas = [doc for doc in processed_docs]

            result_ids = self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=doc_ids
            )

            # 刷新索引确保立即可搜索
            self.es_client.indices.refresh(index=self.index_name)

            logger.info(f"成功添加 {len(documents)} 个文档到向量存储")
            return result_ids

        except Exception as e:
            error_msg = f"添加文档到向量存储失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="ADD_DOCUMENTS_ERROR")

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        语义搜索

        Args:
            query: 查询文本
            k: 返回的最大文档数量

        Returns:
            检索到的文档列表

        Raises:
            VectorStoreException: 当搜索失败时
        """
        try:
            # 优先尝试命题搜索
            if self._has_propositions():
                return self._proposition_search(query, k)

            # 回退到基本语义搜索
            docs = self.vector_store.similarity_search(query, k=k)

            # 增强处理：处理连续块
            docs = self._enhance_consecutive_chunks(docs)

            logger.debug(f"语义搜索完成，查询: {query[:50]}..., 结果数量: {len(docs)}")
            return docs[:k]

        except Exception as e:
            error_msg = f"语义搜索失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg, error_code="SEARCH_ERROR")

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