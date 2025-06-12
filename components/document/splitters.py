"""
文档分割器实现
支持传统分割和LLM增强的命题提取分割
"""

import re
import time
import uuid
import asyncio
import logging
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatTongyi

from core.config.settings import settings
from core.exceptions.errors import DocumentProcessingException

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """改进版文档分割器，支持传统分割和LLM增强的命题提取"""

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 use_proposition_extraction: bool = True,
                 max_llm_chunk_size: int = 2000,
                 max_retries: int = 2):
        """
        初始化文档分割器

        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            use_proposition_extraction: 是否使用命题提取
            max_llm_chunk_size: LLM处理的最大文本长度
            max_retries: LLM调用最大重试次数
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.use_proposition_extraction = use_proposition_extraction
        self.max_llm_chunk_size = max_llm_chunk_size
        self.max_retries = max_retries

        # 初始化传统文本分割器
        try:
            # 优先使用SpaCy分割器（更好的语义分割）
            self.text_splitter = SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                pipeline="zh_core_web_sm"
            )
        except Exception:
            # 回退到递归字符分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
            )
            logger.warning("SpaCy分割器不可用，使用递归字符分割器")

        # 初始化LLM（用于命题提取）
        self.llm = None
        if use_proposition_extraction and settings.TONGYI_API_KEY:
            try:
                self.llm = ChatTongyi(
                    model_name="qwen-turbo",
                    temperature=0.1,
                    dashscope_api_key=settings.TONGYI_API_KEY
                )
                logger.info("已初始化通义千问LLM用于命题提取")
            except Exception as e:
                logger.warning(f"LLM初始化失败，将回退到传统分割: {e}")
                self.use_proposition_extraction = False

    async def split_documents_async(self, documents: List[Document]) -> List[Document]:
        """
        异步分割文档并保留上下文信息

        Args:
            documents: 待分割的文档列表

        Returns:
            分割后的文档列表

        Raises:
            DocumentProcessingException: 当分割失败时
        """
        if not documents:
            return []

        try:
            result_docs = []
            tasks = []

            for doc in documents:
                tasks.append(self._process_document_async(doc))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"处理文档 {i} 失败: {result}")
                    # 回退到传统分割
                    fallback_result = self._process_document_sync(documents[i])
                    result_docs.extend(fallback_result)
                else:
                    result_docs.extend(result)

            logger.info(f"文档分割完成，原始: {len(documents)}, 分割后: {len(result_docs)}")
            return result_docs

        except Exception as e:
            error_msg = f"异步文档分割失败: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg, error_code="SPLIT_ASYNC_ERROR")

    def split_documents_sync(self, documents: List[Document]) -> List[Document]:
        """
        同步分割文档

        Args:
            documents: 待分割的文档列表

        Returns:
            分割后的文档列表
        """
        if not documents:
            return []

        result_docs = []
        for doc in documents:
            doc_result = self._process_document_sync(doc)
            result_docs.extend(doc_result)

        logger.info(f"同步文档分割完成，原始: {len(documents)}, 分割后: {len(result_docs)}")
        return result_docs

    async def _process_document_async(self, doc: Document) -> List[Document]:
        """异步处理单个文档"""
        doc_result = []
        source_id = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_id = source_id or filename or f"doc_{uuid.uuid4().hex[:8]}"

        try:
            # 先使用传统方法分割获取原始chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            original_nodes = []

            # 创建原始节点
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_node_{i}"

                # 提取前后文本作为上下文预览
                prev_context = chunks[i - 1][-150:] if i > 0 else ""
                next_context = chunks[i + 1][:150] if i < len(chunks) - 1 else ""

                # 构建增强的元数据
                enhanced_metadata = {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "prev_chunk_id": f"{doc_id}_node_{i - 1}" if i > 0 else None,
                    "next_chunk_id": f"{doc_id}_node_{i + 1}" if i < len(chunks) - 1 else None,
                    "prev_context_preview": prev_context,
                    "next_context_preview": next_context,
                    "document_id": doc_id,
                    "is_proposition": False,
                    "semantic_level": 0,
                }

                chunk_doc = Document(
                    page_content=chunk,
                    metadata=enhanced_metadata
                )
                original_nodes.append(chunk_doc)
                doc_result.append(chunk_doc)

            # 如果启用命题提取且LLM可用
            if self.use_proposition_extraction and self.llm:
                for i, node in enumerate(original_nodes):
                    propositions = await self._extract_propositions_async(node.page_content)

                    # 为每个命题创建文档对象
                    for j, prop in enumerate(propositions):
                        if not prop.strip():
                            continue

                        prop_id = f"{doc_id}_node_{i}_prop_{j}"

                        prop_metadata = {
                            **doc.metadata,
                            "chunk_id": prop_id,
                            "chunk_index": j,
                            "total_chunks": len(propositions),
                            "parent_node_id": node.metadata["chunk_id"],
                            "document_id": doc_id,
                            "is_proposition": True,
                            "semantic_level": 1,
                        }

                        prop_doc = Document(
                            page_content=prop,
                            metadata=prop_metadata
                        )
                        doc_result.append(prop_doc)

            return doc_result

        except Exception as e:
            logger.error(f"异步处理文档失败: {e}")
            # 回退到同步处理
            return self._process_document_sync(doc)

    def _process_document_sync(self, doc: Document) -> List[Document]:
        """同步处理单个文档（回退方法）"""
        doc_result = []
        source_id = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_id = source_id or filename or f"doc_{uuid.uuid4().hex[:8]}"

        chunks = self.text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"

            prev_context = chunks[i - 1][-150:] if i > 0 else ""
            next_context = chunks[i + 1][:150] if i < len(chunks) - 1 else ""

            enhanced_metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "prev_chunk_id": f"{doc_id}_chunk_{i - 1}" if i > 0 else None,
                "next_chunk_id": f"{doc_id}_chunk_{i + 1}" if i < len(chunks) - 1 else None,
                "prev_context_preview": prev_context,
                "next_context_preview": next_context,
                "document_id": doc_id,
                "is_proposition": False,
                "semantic_level": 0,
            }

            chunk_doc = Document(
                page_content=chunk,
                metadata=enhanced_metadata
            )
            doc_result.append(chunk_doc)

        return doc_result

    async def _extract_propositions_async(self, text: str) -> List[str]:
        """异步从文本中提取命题"""
        if not text or not text.strip():
            return []

        # 如果文本太长，先分割成小块
        if len(text) > self.max_llm_chunk_size:
            sub_chunks = self.text_splitter.split_text(text)
            all_propositions = []

            for sub_chunk in sub_chunks:
                props = await self._extract_propositions_async(sub_chunk)
                all_propositions.extend(props)

            return all_propositions

        # 对适合LLM处理的文本长度，直接提取命题
        for attempt in range(self.max_retries):
            try:
                proposition_prompt = self._build_proposition_prompt(text)
                response = await self.llm.apredict(proposition_prompt)
                propositions = self._parse_proposition_response(response)

                if propositions:
                    return propositions

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"命题提取失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

        # 如果所有重试都失败，回退到传统分割
        return self.text_splitter.split_text(text)

    def _build_proposition_prompt(self, text: str) -> str:
        """构建命题提取提示"""
        return f"""
        从以下文本中提取独立的命题(propositions)。

        命题是文本中的原子表达，代表一个独特的事实或概念，具备以下特点：
        1. 完整且独立，不依赖上下文也能理解
        2. 简明扼要，表达单一事实或概念
        3. 使用自然语言完整地呈现
        4. 不包含冗余信息

        请执行以下操作：
        1. 分解复合句为简单句
        2. 保留原始表达方式
        3. 去上下文化：将代词替换为完整实体名称
        4. 将结果作为JSON数组返回

        文本：
        {text}

        仅返回JSON数组格式: ["命题1", "命题2", ...]
        """

    def _parse_proposition_response(self, response: str) -> List[str]:
        """解析命题提取响应"""
        try:
            import json

            # 查找JSON数组
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                propositions = json.loads(json_str)
                if isinstance(propositions, list):
                    return [p for p in propositions if p and isinstance(p, str)]
        except json.JSONDecodeError:
            pass

        # 如果JSON解析失败，尝试按行分割
        propositions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # 移除编号和前缀符号
            line = re.sub(r'^\d+[\.\)、]?\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            if line and not line.startswith(('JSON', 'json', '命题', '以下是')):
                propositions.append(line)

        return propositions