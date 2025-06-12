"""
查询分析器实现
分析对话历史，提取关键信息和上下文
"""

import json
import logging
import re
from typing import List, Dict, Any

from core.base.llm import BaseLLM
from core.exceptions.errors import ConversationException

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """上下文分析器 - 分析对话历史，提取关键信息"""

    def __init__(self, llm: BaseLLM):
        """
        初始化上下文分析器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("上下文分析器初始化完成")

    def extract_conversation_context(self,
                                    messages: List[Dict[str, str]],
                                    current_query: str) -> Dict[str, Any]:
        """
        从对话历史中提取上下文信息

        Args:
            messages: 对话历史消息
            current_query: 当前查询

        Returns:
            上下文信息字典

        Raises:
            ConversationException: 当上下文提取失败时
        """
        if not messages:
            return self._empty_context()

        try:
            prompt = self._build_context_extraction_prompt(messages, current_query)
            response = self.llm.generate(prompt)

            context_info = self._parse_context_response(response)
            context_info["has_context"] = self._has_meaningful_context(context_info)

            logger.debug(f"上下文提取完成，实体: {len(context_info.get('entities', []))}, 主题: {len(context_info.get('topics', []))}")
            return context_info

        except Exception as e:
            logger.error(f"提取对话上下文失败: {str(e)}")
            return self._empty_context()

    def detect_references(self, query: str) -> Dict[str, Any]:
        """
        检测查询中的指代和引用

        Args:
            query: 查询文本

        Returns:
            引用检测结果
        """
        try:
            prompt = self._build_reference_detection_prompt(query)
            response = self.llm.generate(prompt)

            reference_info = self._parse_reference_response(response)

            logger.debug(f"引用检测完成，查询: {query[:50]}..., 有引用: {reference_info.get('has_references', False)}")
            return reference_info

        except Exception as e:
            logger.error(f"检测查询引用失败: {str(e)}")
            return self._fallback_reference_detection(query)

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图

        Args:
            query: 查询文本

        Returns:
            意图分析结果
        """
        try:
            # 简单的意图分析
            intent_info = {
                "is_question": self._is_question(query),
                "is_comparison": self._is_comparison(query),
                "is_definition": self._is_definition(query),
                "is_procedural": self._is_procedural(query),
                "complexity_level": self._assess_complexity(query)
            }

            logger.debug(f"意图分析完成: {intent_info}")
            return intent_info

        except Exception as e:
            logger.error(f"查询意图分析失败: {str(e)}")
            return {"is_question": True, "complexity_level": "medium"}

    def _empty_context(self) -> Dict[str, Any]:
        """返回空的上下文信息"""
        return {
            "entities": [],
            "topics": [],
            "references": [],
            "has_context": False
        }

    def _has_meaningful_context(self, context_info: Dict[str, Any]) -> bool:
        """判断是否有有意义的上下文"""
        return bool(
            context_info.get("entities") or
            context_info.get("topics") or
            context_info.get("references")
        )

    def _build_context_extraction_prompt(self,
                                       messages: List[Dict[str, str]],
                                       current_query: str) -> str:
        """构建上下文提取提示"""
        # 格式化对话历史
        conversation = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            # 限制内容长度
            if len(content) > 200:
                content = content[:200] + "..."
            conversation += f"{role}: {content}\n\n"

        return f"""
        请分析以下对话历史和当前问题，提取对理解当前问题可能有帮助的上下文信息。
        
        对话历史:
        {conversation}
        
        当前问题: {current_query}
        
        请提取以下信息并以JSON格式返回:
        1. entities: 对话中提及的实体(如人物、技术、产品、概念等)列表
        2. topics: 对话中讨论的主题列表
        3. references: 当前问题中可能引用对话历史的内容(如代词"它"可能指代的实体)
        
        返回格式:
        {{"entities": ["实体1", "实体2"], "topics": ["主题1", "主题2"], "references": [{{"term": "引用词", "possible_referent": "可能指代的实体"}}]}}
        
        只返回JSON，不要有其他说明。
        """

    def _build_reference_detection_prompt(self, query: str) -> str:
        """构建引用检测提示"""
        return f"""
        请分析以下用户查询中是否包含对之前对话内容的指代或引用。例如代词（如"它"、"他们"、"这个"）
        或短语（如"你刚才提到的"、"之前说的方法"）。
        
        用户查询: "{query}"
        
        请以JSON格式返回分析结果，包括以下字段:
        1. has_references: 布尔值，表示是否包含指代或引用
        2. reference_type: 如果有引用，指明引用类型（代词/短语引用/隐含引用）
        3. reference_terms: 引用的具体词语或短语列表
        
        返回格式:
        {{"has_references": true|false, "reference_type": "类型", "reference_terms": ["词语1", "词语2"]}}
        """

    def _parse_context_response(self, response: str) -> Dict[str, Any]:
        """解析上下文响应"""
        try:
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                context_info = json.loads(json_match.group(0))

                # 验证并修正格式
                context_info["entities"] = self._ensure_list(context_info.get("entities", []))
                context_info["topics"] = self._ensure_list(context_info.get("topics", []))
                context_info["references"] = self._ensure_list(context_info.get("references", []))

                return context_info

        except Exception as e:
            logger.warning(f"解析上下文响应失败: {str(e)}")

        return self._empty_context()

    def _parse_reference_response(self, response: str) -> Dict[str, Any]:
        """解析引用检测响应"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                # 验证并修正格式
                return {
                    "has_references": bool(result.get("has_references", False)),
                    "reference_type": result.get("reference_type", "unknown"),
                    "reference_terms": self._ensure_list(result.get("reference_terms", []))
                }

        except Exception as e:
            logger.warning(f"解析引用响应失败: {str(e)}")

        return {"has_references": False, "reference_type": "none", "reference_terms": []}

    def _fallback_reference_detection(self, query: str) -> Dict[str, Any]:
        """回退的引用检测方法"""
        reference_keywords = ["它", "这", "那", "前面", "之前", "刚才", "上面", "他们", "她们"]

        has_references = any(keyword in query for keyword in reference_keywords)
        found_terms = [keyword for keyword in reference_keywords if keyword in query]

        return {
            "has_references": has_references,
            "reference_type": "pronoun" if has_references else "none",
            "reference_terms": found_terms
        }

    def _ensure_list(self, value) -> List:
        """确保值是列表类型"""
        if isinstance(value, list):
            return value
        elif value is None:
            return []
        else:
            return [value]

    def _is_question(self, query: str) -> bool:
        """判断是否为问句"""
        question_markers = ["什么", "如何", "怎么", "为什么", "哪里", "哪个", "谁", "?", "？"]
        return any(marker in query for marker in question_markers)

    def _is_comparison(self, query: str) -> bool:
        """判断是否为比较查询"""
        comparison_markers = ["比较", "对比", "区别", "不同", "相同", "优缺点", "vs", "与"]
        return any(marker in query for marker in comparison_markers)

    def _is_definition(self, query: str) -> bool:
        """判断是否为定义查询"""
        definition_markers = ["是什么", "定义", "含义", "意思", "概念"]
        return any(marker in query for marker in definition_markers)

    def _is_procedural(self, query: str) -> bool:
        """判断是否为过程查询"""
        procedural_markers = ["如何", "怎么", "步骤", "方法", "流程", "过程"]
        return any(marker in query for marker in procedural_markers)

    def _assess_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        # 简单的复杂度评估
        word_count = len(query.split())
        question_count = query.count("？") + query.count("?")
        conjunction_count = sum(query.count(conj) for conj in ["和", "或", "以及", "同时"])

        if word_count > 20 or question_count > 1 or conjunction_count > 2:
            return "high"
        elif word_count > 10 or question_count == 1 or conjunction_count > 0:
            return "medium"
        else:
            return "low"


class QueryClassifier:
    """查询分类器 - 对查询进行分类"""

    def __init__(self):
        """初始化查询分类器"""
        self.query_patterns = {
            "factual": ["什么是", "定义", "含义"],
            "procedural": ["如何", "怎么", "步骤"],
            "comparison": ["比较", "对比", "区别"],
            "causal": ["为什么", "原因", "导致"],
            "location": ["哪里", "位置", "地点"],
            "temporal": ["什么时候", "时间", "日期"]
        }
        logger.info("查询分类器初始化完成")

    def classify(self, query: str) -> str:
        """
        对查询进行分类

        Args:
            query: 查询文本

        Returns:
            查询类型
        """
        query_lower = query.lower()

        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                logger.debug(f"查询分类: '{query[:30]}...' -> {query_type}")
                return query_type

        return "general"

    def get_query_features(self, query: str) -> Dict[str, Any]:
        """
        提取查询特征

        Args:
            query: 查询文本

        Returns:
            查询特征字典
        """
        features = {
            "length": len(query),
            "word_count": len(query.split()),
            "has_question_mark": "?" in query or "？" in query,
            "question_words": sum(1 for word in ["什么", "如何", "为什么", "哪里", "谁"] if word in query),
            "classification": self.classify(query),
            "complexity": self._get_complexity_score(query)
        }

        return features

    def _get_complexity_score(self, query: str) -> float:
        """计算查询复杂度分数"""
        score = 0.0

        # 基于长度
        score += min(len(query.split()) / 20.0, 1.0) * 0.3

        # 基于问句数量
        question_count = query.count("？") + query.count("?")
        score += min(question_count / 3.0, 1.0) * 0.3

        # 基于连接词
        conjunctions = ["和", "或", "以及", "同时", "但是", "然而"]
        conjunction_count = sum(query.count(conj) for conj in conjunctions)
        score += min(conjunction_count / 3.0, 1.0) * 0.4

        return min(score, 1.0)