"""
文档预处理器实现
清理和标准化文档文本
"""

import re
import logging
from typing import List
from langchain.schema import Document

from core.exceptions.errors import DocumentProcessingException

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """文档预处理器，清理和标准化文本"""

    def __init__(self,
                 remove_extra_whitespace: bool = True,
                 remove_special_chars: bool = True,
                 min_content_length: int = 10):
        """
        初始化文档预处理器

        Args:
            remove_extra_whitespace: 是否移除多余的空格和换行符
            remove_special_chars: 是否移除特殊控制字符
            min_content_length: 最小内容长度，短于此长度的文档将被过滤
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_content_length = min_content_length

    def preprocess(self, documents: List[Document]) -> List[Document]:
        """
        预处理文档列表

        Args:
            documents: 原始文档列表

        Returns:
            预处理后的文档列表

        Raises:
            DocumentProcessingException: 当预处理失败时
        """
        if not documents:
            return []

        try:
            processed_docs = []
            filtered_count = 0

            for doc in documents:
                processed_content = self._clean_text(doc.page_content)

                # 过滤内容过短的文档
                if len(processed_content.strip()) < self.min_content_length:
                    filtered_count += 1
                    logger.debug(f"过滤内容过短的文档，长度: {len(processed_content)}")
                    continue

                # 创建新的文档对象
                processed_doc = Document(
                    page_content=processed_content,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )

                # 添加预处理信息到元数据
                processed_doc.metadata["preprocessed"] = True
                processed_doc.metadata["original_length"] = len(doc.page_content)
                processed_doc.metadata["processed_length"] = len(processed_content)

                processed_docs.append(processed_doc)

            logger.info(f"文档预处理完成，原始: {len(documents)}, 处理后: {len(processed_docs)}, 过滤: {filtered_count}")
            return processed_docs

        except Exception as e:
            error_msg = f"文档预处理失败: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingException(error_msg, error_code="PREPROCESSING_ERROR")

    def _clean_text(self, text: str) -> str:
        """
        清理文本内容

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        cleaned_text = text

        # 移除特殊控制字符
        if self.remove_special_chars:
            cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', cleaned_text)

        # 移除多余的空格和换行符
        if self.remove_extra_whitespace:
            # 将多个连续的空白字符替换为单个空格
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            # 移除行首行尾的空格
            cleaned_text = cleaned_text.strip()

        return cleaned_text

    def clean_metadata(self, metadata: dict) -> dict:
        """
        清理元数据

        Args:
            metadata: 原始元数据

        Returns:
            清理后的元数据
        """
        if not metadata:
            return {}

        cleaned_metadata = {}

        for key, value in metadata.items():
            # 清理键名
            clean_key = str(key).strip()

            # 清理值
            if isinstance(value, str):
                clean_value = self._clean_text(value)
            else:
                clean_value = value

            if clean_key and clean_value is not None:
                cleaned_metadata[clean_key] = clean_value

        return cleaned_metadata

    def add_metadata_fields(self, documents: List[Document], additional_fields: dict) -> List[Document]:
        """
        为文档添加额外的元数据字段

        Args:
            documents: 文档列表
            additional_fields: 要添加的额外字段

        Returns:
            添加元数据后的文档列表
        """
        for doc in documents:
            if doc.metadata is None:
                doc.metadata = {}

            doc.metadata.update(additional_fields)

        return documents

    def extract_text_statistics(self, documents: List[Document]) -> dict:
        """
        提取文档文本统计信息

        Args:
            documents: 文档列表

        Returns:
            文本统计信息
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "total_words": 0,
                "average_length": 0
            }

        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_length": total_chars / len(documents) if documents else 0,
            "average_words": total_words / len(documents) if documents else 0
        }