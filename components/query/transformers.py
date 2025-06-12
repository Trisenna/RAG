"""
查询转换器实现
包含查询重写、查询分解等功能
"""

import json
import logging
from typing import List, Dict, Any

from core.base.llm import BaseLLM
from core.exceptions.errors import QueryException

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询重写器 - 优化用户查询以提高检索质量"""

    def __init__(self, llm: BaseLLM):
        """
        初始化查询重写器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("查询重写器初始化完成")

    def rewrite(self, original_query: str) -> str:
        """
        重写查询以优化检索效果

        Args:
            original_query: 原始查询

        Returns:
            重写后的查询

        Raises:
            QueryException: 当重写失败时
        """
        try:
            if not original_query.strip():
                raise QueryException("查询文本不能为空", error_code="EMPTY_QUERY")

            prompt = self._build_rewrite_prompt(original_query)
            rewritten_query = self.llm.generate(prompt).strip()

            logger.debug(f"查询重写: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query

        except QueryException:
            raise
        except Exception as e:
            logger.error(f"查询重写失败: {str(e)}")
            return original_query  # 失败时返回原始查询

    def rewrite_with_context(self, original_query: str, context_hint: str) -> str:
        """
        基于上下文提示重写查询

        Args:
            original_query: 原始查询
            context_hint: 上下文提示

        Returns:
            重写后的查询
        """
        try:
            prompt = self._build_contextual_rewrite_prompt(original_query, context_hint)
            rewritten_query = self.llm.generate(prompt).strip()

            logger.debug(f"上下文查询重写: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query

        except Exception as e:
            logger.error(f"上下文查询重写失败: {str(e)}")
            return original_query

    def _build_rewrite_prompt(self, original_query: str) -> str:
        """构建查询重写提示"""
        return f"""
        你是一个查询重写专家，你的任务是将用户原始查询重写为更适合向量检索的形式，使其能够更好地匹配相关文档。
        
        请遵循以下准则：
        1. 保留原始查询的核心意图和所有关键信息
        2. 扩展查询中的缩写和专业术语
        3. 添加可能的同义词和相关术语
        4. 移除不必要的修饰词
        5. 保持重写后的查询简洁明了
        6. 不要添加原始查询中不存在的问题或假设

        原始查询：{original_query}
        
        重写后的查询：
        """

    def _build_contextual_rewrite_prompt(self, original_query: str, context_hint: str) -> str:
        """构建上下文查询重写提示"""
        return f"""
        你是一个查询重写专家，你的任务是将用户原始查询重写为更适合向量检索的形式，使其能够更好地匹配相关文档。
        
        请根据以下的上下文线索来重写查询:
        上下文线索: {context_hint}
        
        请遵循以下准则：
        1. 保留原始查询的核心意图和所有关键信息
        2. 利用上下文线索来扩展和明确查询
        3. 添加可能的同义词和相关术语
        4. 确保重写后的查询更具体、更有针对性
        
        原始查询：{original_query}
        
        重写后的查询：
        """


class ContextualQueryRewriter(QueryRewriter):
    """上下文感知查询重写器"""

    def rewrite_with_history(self,
                            original_query: str,
                            conversation_history: List[Dict[str, str]],
                            context_info: Dict[str, Any]) -> str:
        """
        结合对话历史重写查询

        Args:
            original_query: 原始查询
            conversation_history: 对话历史
            context_info: 上下文分析信息

        Returns:
            重写后的查询
        """
        if not conversation_history or not context_info.get("has_context", False):
            return self.rewrite(original_query)

        try:
            # 提取上下文信息
            entities = context_info.get("entities", [])
            topics = context_info.get("topics", [])

            # 格式化对话历史
            history_text = self._format_conversation_history(conversation_history)

            # 构建上下文感知提示
            prompt = self._build_contextual_history_prompt(
                original_query, history_text, entities, topics
            )

            rewritten_query = self.llm.generate(prompt).strip()
            logger.debug(f"上下文历史重写: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query

        except Exception as e:
            logger.error(f"上下文历史查询重写失败: {str(e)}")
            return original_query

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """格式化对话历史"""
        # 只使用最近的3轮对话
        recent_history = history[-6:] if len(history) > 6 else history

        history_text = ""
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]
            # 限制内容长度
            if len(content) > 100:
                content = content[:100] + "..."
            history_text += f"{role}: {content}\n"

        return history_text

    def _build_contextual_history_prompt(self,
                                       query: str,
                                       history_text: str,
                                       entities: List[str],
                                       topics: List[str]) -> str:
        """构建上下文历史提示"""
        return f"""
        你是一个查询重写专家，任务是解析含糊不清的查询，将其扩展为明确、具体的查询，特别是处理代词和隐含引用。
        
        请基于以下对话历史和上下文信息，将用户的原始查询重写为更明确的查询，使其能够有效检索相关文档。
        
        对话历史:
        {history_text}
        
        提取的上下文信息:
        - 实体: {", ".join(entities) if entities else "无"}
        - 主题: {", ".join(topics) if topics else "无"}
        
        原始查询: {query}
        
        请遵循以下准则:
        1. 解析所有代词(它、他们、这个等)，使用确切的名称或描述替换
        2. 扩展隐含的上下文引用
        3. 添加必要的上下文信息使查询自包含
        4. 保持查询语义不变，不要添加原始查询中不存在的新问题
        5. 返回的重写查询应该是一个完整、清晰的问题，任何人看了都能理解，而不需要之前的对话历史
        
        重写后的查询:
        """


class QueryDecomposer:
    """查询分解器 - 将复杂查询分解为多个子查询"""

    def __init__(self, llm: BaseLLM):
        """
        初始化查询分解器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("查询分解器初始化完成")

    def should_decompose(self, query: str) -> bool:
        """
        判断查询是否需要分解

        Args:
            query: 查询文本

        Returns:
            是否需要分解
        """
        try:
            prompt = self._build_complexity_prompt(query)
            response = self.llm.generate(prompt).strip().lower()

            should_decompose = "是" in response or "yes" in response
            logger.debug(f"查询复杂度分析: '{query}' -> 需要分解: {should_decompose}")
            return should_decompose

        except Exception as e:
            logger.error(f"查询复杂度分析失败: {str(e)}")
            return False  # 默认不分解

    def decompose(self, complex_query: str, max_subqueries: int = 3) -> List[str]:
        """
        将复杂查询分解为多个子查询

        Args:
            complex_query: 复杂查询
            max_subqueries: 最大子查询数量

        Returns:
            子查询列表

        Raises:
            QueryException: 当分解失败时
        """
        try:
            if not complex_query.strip():
                raise QueryException("查询文本不能为空", error_code="EMPTY_QUERY")

            prompt = self._build_decomposition_prompt(complex_query, max_subqueries)
            response = self.llm.generate(prompt)

            subqueries = self._parse_decomposition_response(response, max_subqueries)

            logger.debug(f"查询分解: '{complex_query}' -> {len(subqueries)} 个子查询")
            return subqueries

        except QueryException:
            raise
        except Exception as e:
            logger.error(f"查询分解失败: {str(e)}")
            return [complex_query]  # 失败时返回原始查询

    def _build_complexity_prompt(self, query: str) -> str:
        """构建复杂度分析提示"""
        return f"""
        分析以下查询，并判断它是否是一个复杂查询需要被分解为多个子查询。
        
        复杂查询的特征：
        1. 包含多个不同但相关的问题
        2. 需要来自多个知识领域或文档的信息
        3. 询问比较或对比多个事物
        4. 包含条件或假设性情况
        
        查询: {query}
        
        请只回答 "是" 或 "否"。
        """

    def _build_decomposition_prompt(self, complex_query: str, max_subqueries: int) -> str:
        """构建查询分解提示"""
        return f"""
        你是一个专家查询分析师。请将以下复杂查询分解为最多{max_subqueries}个独立的子查询，以便更好地检索相关信息。
        
        分解规则：
        1. 每个子查询应该是完整的、独立的问题
        2. 子查询应该覆盖原始复杂查询的不同方面
        3. 子查询应该简单直接，针对单一信息点
        4. 子查询之间应尽量减少重叠
        5. 如果原始查询已经足够简单，可以返回少于{max_subqueries}个子查询
        
        复杂查询: {complex_query}
        
        请以JSON数组格式返回子查询列表，格式为: ["子查询1", "子查询2", ...]
        """

    def _parse_decomposition_response(self, response: str, max_subqueries: int) -> List[str]:
        """解析查询分解响应"""
        try:
            import re

            # 尝试提取JSON数组
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                subqueries = json.loads(json_str)
                if isinstance(subqueries, list):
                    # 过滤空查询并限制数量
                    valid_queries = [q for q in subqueries if q and isinstance(q, str)]
                    return valid_queries[:max_subqueries]

            # 回退处理：按行分割并清理
            subqueries = []
            for line in response.split('\n'):
                line = line.strip(' "\'')
                if line and not line.startswith(('[', ']', '{', '}')):
                    subqueries.append(line)

            return subqueries[:max_subqueries] if subqueries else []

        except Exception as e:
            logger.error(f"解析查询分解响应失败: {str(e)}")
            return []