"""
索引服务实现
负责文档的加载、处理和索引
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
import  time
from core.config.settings import settings
from core.exceptions.errors import IndexingException, DocumentProcessingException
from components.document.loaders import DocumentLoader
from components.document.processors import DocumentPreprocessor
from components.document.splitters import DocumentSplitter
from components.vectorstore.elasticsearch_store import ElasticsearchVectorStore

logger = logging.getLogger(__name__)


class IndexingService:
    """索引服务 - 统一管理文档索引流程"""

    def __init__(self,
                 use_proposition_extraction: bool = True,
                 max_concurrent_files: int = 3):
        """
        初始化索引服务

        Args:
            use_proposition_extraction: 是否使用命题提取
            max_concurrent_files: 最大并发文件数
        """
        try:
            # 初始化组件
            self.document_loader = DocumentLoader()
            self.preprocessor = DocumentPreprocessor()
            self.splitter = DocumentSplitter(use_proposition_extraction=use_proposition_extraction)
            self.vector_store = ElasticsearchVectorStore()

            self.max_concurrent_files = max_concurrent_files
            self.use_proposition_extraction = use_proposition_extraction

            # 确保索引存在
            self.vector_store.create_index_if_not_exists()

            logger.info(f"索引服务初始化完成，命题提取: {use_proposition_extraction}, 最大并发: {max_concurrent_files}")

        except Exception as e:
            error_msg = f"索引服务初始化失败: {str(e)}"
            logger.error(error_msg)
            raise IndexingException(error_msg, error_code="INDEXING_SERVICE_INIT_ERROR")

    async def index_file(self, file_path: str) -> Dict[str, Any]:
        """
        索引单个文件

        Args:
            file_path: 文件路径

        Returns:
            索引结果

        Raises:
            IndexingException: 当索引失败时
        """
        try:
            # 验证文件存在性
            if not os.path.exists(file_path):
                raise IndexingException(
                    f"文件不存在: {file_path}",
                    error_code="FILE_NOT_FOUND"
                )

            # 检查文件类型支持
            if not self.document_loader.factory.is_supported(file_path):
                raise IndexingException(
                    f"不支持的文件类型: {file_path}",
                    error_code="UNSUPPORTED_FILE_TYPE"
                )

            logger.info(f"开始索引文件: {file_path}")

            # 1. 加载文档
            documents = self.document_loader.load_file(file_path)

            # 2. 预处理文档
            documents = self.preprocessor.preprocess(documents)

            # 3. 添加文件级别的元数据
            documents = self._add_file_metadata(documents, file_path)

            # 4. 文档分割（异步）
            split_docs = await self.splitter.split_documents_async(documents)

            # 5. 添加到向量存储
            doc_ids = self.vector_store.add_documents(split_docs)

            # 6. 生成统计信息
            stats = self._generate_file_stats(file_path, documents, split_docs)

            result = {
                "status": "success",
                "message": f"成功索引文件: {os.path.basename(file_path)}",
                "file_path": file_path,
                "document_count": len(split_docs),
                "document_ids": doc_ids,
                "statistics": stats
            }

            logger.info(f"文件索引完成: {file_path}, 生成块数: {len(split_docs)}")
            return result

        except IndexingException:
            raise
        except DocumentProcessingException as e:
            error_msg = f"文档处理失败: {str(e)}"
            logger.error(error_msg)
            raise IndexingException(error_msg, error_code="DOCUMENT_PROCESSING_ERROR")
        except Exception as e:
            error_msg = f"索引文件失败: {file_path}, 错误: {str(e)}"
            logger.error(error_msg)
            raise IndexingException(error_msg, error_code="INDEX_FILE_ERROR")

    async def index_directory(self, directory_path: Optional[str] = None) -> Dict[str, Any]:
        """
        索引目录中的所有文件

        Args:
            directory_path: 目录路径，None表示使用默认文档目录

        Returns:
            索引结果

        Raises:
            IndexingException: 当索引失败时
        """
        directory_path = directory_path or settings.DOCUMENTS_DIR

        try:
            # 确保目录存在
            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
                return {
                    "status": "success",
                    "message": f"创建了目录: {directory_path}, 但没有文件可索引",
                    "directory_path": directory_path,
                    "processed_files": 0,
                    "total_chunks": 0
                }

            logger.info(f"开始索引目录: {directory_path}")

            # 收集所有支持的文件
            file_paths = self._collect_supported_files(directory_path)

            if not file_paths:
                return {
                    "status": "success",
                    "message": f"目录中没有支持的文件: {directory_path}",
                    "directory_path": directory_path,
                    "processed_files": 0,
                    "total_chunks": 0
                }

            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(self.max_concurrent_files)

            async def index_with_semaphore(file_path: str):
                async with semaphore:
                    return await self.index_file(file_path)

            # 并发处理所有文件
            tasks = [index_with_semaphore(file_path) for file_path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计结果
            success_count = 0
            error_count = 0
            total_chunks = 0
            processed_results = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "file_path": file_paths[i],
                        "status": "error",
                        "message": f"索引失败: {str(result)}"
                    })
                    error_count += 1
                else:
                    processed_results.append(result)
                    if result["status"] == "success":
                        success_count += 1
                        total_chunks += result.get("document_count", 0)
                    else:
                        error_count += 1

            overall_status = "success" if error_count == 0 else "partial_success"

            result = {
                "status": overall_status,
                "message": f"目录索引完成，成功: {success_count}, 失败: {error_count}, 总块数: {total_chunks}",
                "directory_path": directory_path,
                "processed_files": len(file_paths),
                "successful_files": success_count,
                "failed_files": error_count,
                "total_chunks": total_chunks,
                "file_results": processed_results
            }

            logger.info(f"目录索引完成: {directory_path}, 成功: {success_count}/{len(file_paths)}")
            return result

        except Exception as e:
            error_msg = f"索引目录失败: {directory_path}, 错误: {str(e)}"
            logger.error(error_msg)
            raise IndexingException(error_msg, error_code="INDEX_DIRECTORY_ERROR")

    def delete_all_indexes(self) -> Dict[str, Any]:
        """
        删除所有索引

        Returns:
            删除结果

        Raises:
            IndexingException: 当删除失败时
        """
        try:
            self.vector_store.delete_index()

            logger.info("所有索引已删除")
            return {
                "status": "success",
                "message": "所有索引已删除"
            }

        except Exception as e:
            error_msg = f"删除索引失败: {str(e)}"
            logger.error(error_msg)
            raise IndexingException(error_msg, error_code="DELETE_INDEX_ERROR")

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            统计信息
        """
        try:
            document_count = self.vector_store.get_document_count()

            return {
                "total_documents": document_count,
                "index_name": self.vector_store.index_name,
                "proposition_extraction_enabled": self.use_proposition_extraction,
                "supported_file_types": self.document_loader.get_supported_extensions()
            }

        except Exception as e:
            logger.error(f"获取索引统计信息失败: {str(e)}")
            return {
                "total_documents": 0,
                "error": str(e)
            }

    def _collect_supported_files(self, directory_path: str) -> list:
        """收集目录中所有支持的文件"""
        file_paths = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.document_loader.factory.is_supported(file_path):
                    file_paths.append(file_path)

        return file_paths

    def _add_file_metadata(self, documents: list, file_path: str) -> list:
        """为文档添加文件级别的元数据"""
        file_metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1].lower(),
            "indexed_at": int(time.time())
        }

        # 为每个文档添加文件元数据
        for doc in documents:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(file_metadata)

        return documents

    def _generate_file_stats(self, file_path: str, original_docs: list, split_docs: list) -> Dict[str, Any]:
        """生成文件索引统计信息"""
        original_stats = self.preprocessor.extract_text_statistics(original_docs)
        split_stats = self.preprocessor.extract_text_statistics(split_docs)

        # 统计命题数量
        proposition_count = sum(1 for doc in split_docs
                              if doc.metadata and doc.metadata.get("is_proposition", False))

        return {
            "file_info": {
                "path": file_path,
                "size_bytes": os.path.getsize(file_path),
                "type": os.path.splitext(file_path)[1].lower()
            },
            "original_documents": {
                "count": original_stats["total_documents"],
                "total_characters": original_stats["total_characters"],
                "total_words": original_stats["total_words"]
            },
            "processed_documents": {
                "total_chunks": split_stats["total_documents"],
                "proposition_chunks": proposition_count,
                "regular_chunks": split_stats["total_documents"] - proposition_count,
                "total_characters": split_stats["total_characters"],
                "average_chunk_length": split_stats["average_length"]
            }
        }

    def recreate_index(self) -> Dict[str, Any]:
        """
        重新创建索引

        Returns:
            操作结果
        """
        try:
            # 删除现有索引
            self.vector_store.delete_index()

            # 创建新索引
            self.vector_store.create_index_if_not_exists()

            logger.info("索引重新创建完成")
            return {
                "status": "success",
                "message": "索引重新创建完成"
            }

        except Exception as e:
            error_msg = f"重新创建索引失败: {str(e)}"
            logger.error(error_msg)
            raise IndexingException(error_msg, error_code="RECREATE_INDEX_ERROR")


