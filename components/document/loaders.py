"""
文档加载器实现
根据文件类型自动选择合适的加载器
"""

import os
import logging
from typing import Dict, Type, Optional, List
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import (
    UnstructuredFileLoader, UnstructuredMarkdownLoader,
    PyPDFLoader, TextLoader, Docx2txtLoader
)

from core.exceptions.errors import DocumentProcessingException

logger = logging.getLogger(__name__)


class DocumentLoaderFactory:
    """文档加载器工厂类"""

    # 支持的文件类型和对应的加载器
    _loaders: Dict[str, Type[UnstructuredFileLoader]] = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str) -> Optional[UnstructuredFileLoader]:
        """
        根据文件扩展名获取相应的加载器

        Args:
            file_path: 文件路径

        Returns:
            文档加载器实例，如果不支持该文件类型则返回None

        Raises:
            DocumentProcessingException: 当文件不存在时
        """
        if not os.path.exists(file_path):
            raise DocumentProcessingException(
                f"文件不存在: {file_path}",
                error_code="FILE_NOT_FOUND"
            )

        ext = os.path.splitext(file_path)[1].lower()
        loader_cls = cls._loaders.get(ext)

        if loader_cls:
            logger.debug(f"为文件 {file_path} 选择加载器: {loader_cls.__name__}")
            return loader_cls(file_path)

        logger.warning(f"不支持的文件类型: {ext}")
        return None

    @classmethod
    def register_loader(cls, extension: str, loader_cls: Type[UnstructuredFileLoader]) -> None:
        """
        注册新的文档加载器

        Args:
            extension: 文件扩展名（包含点号，如 ".xlsx"）
            loader_cls: 加载器类
        """
        cls._loaders[extension.lower()] = loader_cls
        logger.info(f"已注册新的文档加载器: {extension} -> {loader_cls.__name__}")

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        获取支持的文件扩展名列表

        Returns:
            支持的文件扩展名列表
        """
        return list(cls._loaders.keys())

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        检查文件类型是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持该文件类型
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in cls._loaders


class DocumentLoader:
    """文档加载器统一接口"""

    def __init__(self):
        self.factory = DocumentLoaderFactory()

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            加载的文档列表

        Raises:
            DocumentProcessingException: 当加载失败时
        """
        try:
            loader = self.factory.get_loader(file_path)
            if not loader:
                raise DocumentProcessingException(
                    f"不支持的文件类型: {file_path}",
                    error_code="UNSUPPORTED_FILE_TYPE"
                )

            documents = loader.load()
            logger.info(f"成功加载文件: {file_path}，文档数量: {len(documents)}")
            return documents

        except DocumentProcessingException:
            raise
        except Exception as e:
            error_msg = f"加载文件失败: {file_path}, 错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg, error_code="LOAD_ERROR")

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        加载目录中的所有支持的文件

        Args:
            directory_path: 目录路径

        Returns:
            加载的所有文档列表

        Raises:
            DocumentProcessingException: 当目录不存在时
        """
        if not os.path.exists(directory_path):
            raise DocumentProcessingException(
                f"目录不存在: {directory_path}",
                error_code="DIRECTORY_NOT_FOUND"
            )

        all_documents = []
        loaded_files = 0
        failed_files = 0

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)

                if not self.factory.is_supported(file_path):
                    logger.debug(f"跳过不支持的文件: {file_path}")
                    continue

                try:
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)
                    loaded_files += 1
                except DocumentProcessingException as e:
                    logger.error(f"加载文件失败: {file_path}, 错误: {e.message}")
                    failed_files += 1

        logger.info(f"目录加载完成: {directory_path}, 成功: {loaded_files}, 失败: {failed_files}")
        return all_documents

    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名"""
        return self.factory.get_supported_extensions()