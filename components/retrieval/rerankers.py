"""
重排序器实现
使用LLM对检索结果进行智能重排序
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document

from core.base.llm import BaseLLM
from core.base.retriever import BaseReranker
from core.exceptions.errors import RetrievalException

logger = logging.getLogger(__name__)


class LLMReranker(BaseReranker):
    """基于LLM的重排序器"""

    def __init__(self,
                 llm: BaseLLM,
                 batch_size: int = 10,
                 max_retries: int = 2):
        """
        初始化LLM重排序器

        Args:
            llm: 语言模型实例
            batch_size: 批处理大小
            max_retries: 最大重试次数
        """
        self.llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        logger.info(f"LLM重排序器初始化完成，批大小: {batch_size}")

    def rerank(self,
               query: str,
               documents: List[Document],
               top_k: int = 5) -> List[Document]:
        """
        使用LLM对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的最大文档数量

        Returns:
            重排序后的文档列表

        Raises:
            RetrievalException: 当重排序失败时
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        try:
            # 按批次处理文档以避免超出上下文限制
            all_ranked_docs = []

            # 将文档分批处理
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                ranked_batch = self._rerank_batch(query, batch)
                all_ranked_docs.extend(ranked_batch)

            # 返回排名最高的top_k个文档
            result = all_ranked_docs[:top_k]
            logger.debug(f"重排序完成，原始: {len(documents)}, 批次数: {len(range(0, len(documents), self.batch_size))}, 返回: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            # 出错时返回原始排序的前top_k个文档
            return documents[:top_k]

    def _rerank_batch(self, query: str, batch: List[Document]) -> List[Document]:
        """重排序单个批次的文档"""
        if len(batch) <= 1:
            return batch

        for attempt in range(self.max_retries):
            try:
                # 生成重排序提示
                prompt = self._create_reranking_prompt(query, batch)

                # 获取LLM重排序结果
                response = self.llm.generate(prompt)

                # 解析重排序结果
                ranked_indices = self._parse_reranking_response(response, len(batch))

                # 根据排序重新组织文档
                ranked_batch = [batch[i] for i in ranked_indices if i < len(batch)]

                # 添加重排序信息到元数据
                for i, doc in enumerate(ranked_batch):
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["rerank_position"] = i + 1
                    doc.metadata["reranked"] = True

                return ranked_batch

            except Exception as e:
                logger.warning(f"重排序批次失败 (尝试 {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    continue
                else:
                    # 最后一次尝试失败，返回原始顺序
                    logger.error(f"重排序批次最终失败，返回原始顺序")
                    return batch

        return batch

    def _create_reranking_prompt(self, query: str, documents: List[Document]) -> str:
        """创建重排序提示"""
        prompt = f"""
        作为一个专家重排序系统，你的任务是对检索到的文档片段进行重排序，使最相关的文档排在前面。
        
        用户查询: {query}
        
        以下是检索到的文档片段:
        
        """

        for i, doc in enumerate(documents):
            # 限制文档内容长度以避免提示过长
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            prompt += f"文档 {i+1}:\n{content}\n\n"

        prompt += f"""
        请根据与用户查询的相关性对这些文档进行重新排序。你的回答应该是一个数字列表，表示文档的新顺序。
        
        评判标准：
        1. 内容相关性：文档内容与查询的匹配程度
        2. 信息完整性：文档是否包含查询所需的完整信息
        3. 信息质量：文档内容的准确性和权威性
        
        例如：
        如果你认为文档2最相关，然后是文档1，最后是文档3，你应该回答:
        2, 1, 3
        
        只需返回逗号分隔的数字列表，不要添加额外的文字。
        """

        return prompt

    def _parse_reranking_response(self, response: str, num_docs: int) -> List[int]:
        """
        解析LLM的重排序响应

        Args:
            response: LLM响应
            num_docs: 文档数量

        Returns:
            重排序后的文档索引列表
        """
        try:
            # 清理响应，移除多余的空格和换行
            response = response.strip()

            # 尝试解析逗号分隔的数字列表
            doc_nums = []
            for num_str in response.split(','):
                try:
                    num = int(num_str.strip())
                    doc_nums.append(num)
                except ValueError:
                    continue

            # 将文档编号（从1开始）转换为索引（从0开始）
            doc_indices = [num - 1 for num in doc_nums if 1 <= num <= num_docs]

            # 验证并补充缺失的索引
            valid_indices = []
            used_indices = set()

            # 添加有效的重排序索引
            for idx in doc_indices:
                if 0 <= idx < num_docs and idx not in used_indices:
                    valid_indices.append(idx)
                    used_indices.add(idx)

            # 补充缺失的索引（按原始顺序）
            for i in range(num_docs):
                if i not in used_indices:
                    valid_indices.append(i)

            logger.debug(f"重排序解析成功: {doc_nums} -> {valid_indices}")
            return valid_indices[:num_docs]

        except Exception as e:
            logger.warning(f"解析重排序响应失败: {str(e)}, 使用原始顺序")
            return list(range(num_docs))


class ScoreBasedReranker(BaseReranker):
    """基于分数的简单重排序器"""

    def __init__(self, score_field: str = "rrf_score"):
        """
        初始化基于分数的重排序器

        Args:
            score_field: 用于排序的分数字段名
        """
        self.score_field = score_field
        logger.info(f"基于分数的重排序器初始化完成，分数字段: {score_field}")

    def rerank(self,
               query: str,
               documents: List[Document],
               top_k: int = 5) -> List[Document]:
        """
        基于分数重排序

        Args:
            query: 查询文本（未使用）
            documents: 待排序的文档列表
            top_k: 返回的最大文档数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        try:
            # 按指定分数字段排序
            scored_docs = []
            unscored_docs = []

            for doc in documents:
                if (hasattr(doc, 'metadata') and
                    doc.metadata and
                    self.score_field in doc.metadata):
                    scored_docs.append(doc)
                else:
                    unscored_docs.append(doc)

            # 按分数降序排序
            scored_docs.sort(
                key=lambda x: x.metadata.get(self.score_field, 0),
                reverse=True
            )

            # 合并结果：有分数的在前，无分数的在后
            result = scored_docs + unscored_docs

            logger.debug(f"基于分数重排序完成，有分数: {len(scored_docs)}, 无分数: {len(unscored_docs)}")
            return result[:top_k]

        except Exception as e:
            error_msg = f"基于分数的重排序失败: {str(e)}"
            logger.error(error_msg)
            return documents[:top_k]


class HybridReranker(BaseReranker):
    """混合重排序器，结合多种重排序策略"""

    def __init__(self,
                 llm_reranker: LLMReranker,
                 score_reranker: ScoreBasedReranker,
                 use_llm_threshold: int = 10):
        """
        初始化混合重排序器

        Args:
            llm_reranker: LLM重排序器
            score_reranker: 基于分数的重排序器
            use_llm_threshold: 使用LLM重排序的文档数量阈值
        """
        self.llm_reranker = llm_reranker
        self.score_reranker = score_reranker
        self.use_llm_threshold = use_llm_threshold
        logger.info(f"混合重排序器初始化完成，LLM阈值: {use_llm_threshold}")

    def rerank(self,
               query: str,
               documents: List[Document],
               top_k: int = 5) -> List[Document]:
        """
        混合重排序策略

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的最大文档数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        try:
            # 根据文档数量选择重排序策略
            if len(documents) >= self.use_llm_threshold:
                # 文档数量较多时，使用LLM重排序
                logger.debug(f"使用LLM重排序，文档数量: {len(documents)}")
                return self.llm_reranker.rerank(query, documents, top_k)
            else:
                # 文档数量较少时，使用分数重排序
                logger.debug(f"使用分数重排序，文档数量: {len(documents)}")
                return self.score_reranker.rerank(query, documents, top_k)

        except Exception as e:
            error_msg = f"混合重排序失败: {str(e)}"
            logger.error(error_msg)
            return documents[:top_k]