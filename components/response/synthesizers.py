"""
响应合成器实现
根据检索结果和上下文生成最终回答
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from core.base.llm import BaseLLM
from core.exceptions.errors import LLMException

logger = logging.getLogger(__name__)


class BaseResponseSynthesizer:
    """基础响应合成器"""

    def __init__(self, llm: BaseLLM):
        """
        初始化响应合成器

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        logger.info("基础响应合成器初始化完成")

    def synthesize(self,
                  query: str,
                  documents: List[Document],
                  use_citation: bool = True) -> Dict[str, Any]:
        """
        根据检索文档合成响应

        Args:
            query: 用户查询
            documents: 检索到的文档
            use_citation: 是否使用引用

        Returns:
            合成的响应结果

        Raises:
            LLMException: 当合成失败时
        """
        if not documents:
            return self._empty_response()

        try:
            # 提取文档内容和来源
            contexts, sources = self._extract_contexts_and_sources(documents)

            # 构建合成提示
            prompt = self._build_synthesis_prompt(query, contexts, use_citation)

            # 生成回答
            answer = self.llm.generate(prompt)

            result = {
                "answer": answer,
                "sources": sources,
                "context_count": len(contexts),
                "has_context": True
            }

            logger.debug(f"响应合成完成，上下文数量: {len(contexts)}")
            return result

        except Exception as e:
            error_msg = f"响应合成失败: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": f"生成回答时出现错误: {str(e)}",
                "sources": self._extract_sources(documents),
                "context_count": len(documents),
                "has_context": True,
                "error": str(e)
            }

    def synthesize_from_multiple_queries(self,
                                        main_query: str,
                                        sub_queries: List[str],
                                        sub_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从多个子查询结果合成最终回答

        Args:
            main_query: 主查询
            sub_queries: 子查询列表
            sub_results: 子查询结果列表

        Returns:
            合成的响应结果
        """
        try:
            # 收集有效的子查询结果
            valid_results = []
            all_sources = []

            for i, (sub_query, result) in enumerate(zip(sub_queries, sub_results)):
                if result.get("has_context", False) and result.get("answer"):
                    valid_results.append({
                        "query": sub_query,
                        "answer": result.get("answer", "")
                    })
                    all_sources.extend(result.get("sources", []))

            if not valid_results:
                return self._empty_response()

            # 构建多查询合成提示
            prompt = self._build_multi_query_synthesis_prompt(main_query, valid_results)

            # 生成最终回答
            answer = self.llm.generate(prompt)

            result = {
                "answer": answer,
                "sources": all_sources,
                "sub_query_count": len(valid_results),
                "context_count": len(all_sources),
                "has_context": True
            }

            logger.debug(f"多查询响应合成完成，子查询数量: {len(valid_results)}")
            return result

        except Exception as e:
            error_msg = f"多查询响应合成失败: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": "合成最终回答时出现错误。",
                "sources": all_sources if 'all_sources' in locals() else [],
                "has_context": True,
                "error": str(e)
            }

    def _extract_contexts_and_sources(self, documents: List[Document]) -> tuple:
        """提取文档内容和来源信息"""
        contexts = []
        sources = []

        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue

            contexts.append(f"[{i+1}] {content}")

            # 提取源文件信息
            source_info = self._extract_source_info(doc, i+1)
            sources.append(source_info)

        return contexts, sources

    def _extract_source_info(self, doc: Document, index: int) -> Dict[str, Any]:
        """提取单个文档的源信息"""
        source_info = {"index": index}

        if hasattr(doc, "metadata") and doc.metadata:
            source_info.update({
                "filename": doc.metadata.get("filename", "未知文件"),
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": doc.metadata.get("rrf_score", doc.metadata.get("score", 0))
            })

        return source_info

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """仅提取源信息"""
        _, sources = self._extract_contexts_and_sources(documents)
        return sources

    def _build_synthesis_prompt(self, query: str, contexts: List[str], use_citation: bool) -> str:
        """构建合成提示"""
        context_text = "\n\n".join(contexts)

        if use_citation:
            prompt = f"""
            你是一个知识丰富的助手。请回答用户的问题，仅基于以下提供的上下文信息。如果上下文中没有足够的信息来回答问题，请坦诚地说"抱歉，我没有足够的信息来回答这个问题"。
            
            在回答中，请使用方括号引用相关的上下文编号（例如[1]、[2]等），以说明信息来源。引用应该放在相关句子的末尾。
            
            上下文信息:
            {context_text}
            
            用户问题: {query}
            
            请提供一个全面、准确且有条理的回答：
            """
        else:
            prompt = f"""
            你是一个知识丰富的助手。请回答用户的问题，仅基于以下提供的上下文信息。如果上下文中没有足够的信息来回答问题，请坦诚地说"抱歉，我没有足够的信息来回答这个问题"。
            
            上下文信息:
            {context_text}
            
            用户问题: {query}
            
            请提供一个全面、准确且有条理的回答：
            """

        return prompt

    def _build_multi_query_synthesis_prompt(self, main_query: str, valid_results: List[Dict[str, str]]) -> str:
        """构建多查询合成提示"""
        prompt = f"""
        你是一个知识丰富的助手。你收到了一个主要问题和几个相关子问题的答案。请综合这些信息，为主要问题提供一个全面、连贯的回答。

        主要问题: {main_query}

        子问题和答案:
        """

        for i, result in enumerate(valid_results):
            prompt += f"\n子问题 {i+1}: {result['query']}\n答案: {result['answer']}\n"

        prompt += """
        请综合上述所有信息，为主要问题提供一个全面、准确的回答。确保你的回答:
        1. 完全回应主要问题
        2. 融合所有子问题答案的相关信息
        3. 保持连贯性和逻辑性
        4. 避免重复信息
        5. 如有必要，指出子问题答案之间的任何矛盾

        最终回答:
        """

        return prompt

    def _empty_response(self) -> Dict[str, Any]:
        """返回空响应"""
        return {
            "answer": "抱歉，没有找到相关的信息来回答您的问题。",
            "sources": [],
            "context_count": 0,
            "has_context": False
        }


class ContextualResponseSynthesizer(BaseResponseSynthesizer):
    """上下文感知响应合成器"""

    def synthesize_with_history(self,
                               query: str,
                               documents: List[Document],
                               conversation_history: List[Dict[str, str]],
                               use_citation: bool = True) -> Dict[str, Any]:
        """
        结合对话历史和检索文档合成响应

        Args:
            query: 用户查询
            documents: 检索到的文档
            conversation_history: 对话历史
            use_citation: 是否使用引用

        Returns:
            合成的响应结果
        """
        if not documents:
            return self._empty_response()

        try:
            # 提取文档内容和来源
            contexts, sources = self._extract_contexts_and_sources(documents)

            # 格式化对话历史
            history_text = self._format_conversation_history(conversation_history)

            # 构建上下文感知提示
            prompt = self._build_contextual_synthesis_prompt(
                query, contexts, history_text, use_citation
            )

            # 生成回答
            answer = self.llm.generate(prompt)

            result = {
                "answer": answer,
                "sources": sources,
                "context_count": len(contexts),
                "history_length": len(conversation_history),
                "has_context": True
            }

            logger.debug(f"上下文感知响应合成完成，上下文: {len(contexts)}, 历史: {len(conversation_history)}")
            return result

        except Exception as e:
            error_msg = f"上下文感知响应合成失败: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": f"生成回答时出现错误: {str(e)}",
                "sources": self._extract_sources(documents),
                "has_context": True,
                "error": str(e)
            }

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        # 只使用最近的几轮对话
        recent_history = history[-6:] if len(history) > 6 else history

        history_text = ""
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]
            # 限制内容长度
            if len(content) > 80:
                content = content[:80] + "..."
            history_text += f"{role}: {content}\n"

        return history_text

    def _build_contextual_synthesis_prompt(self,
                                         query: str,
                                         contexts: List[str],
                                         history_text: str,
                                         use_citation: bool) -> str:
        """构建上下文感知合成提示"""
        context_text = "\n\n".join(contexts)

        conversation_context = ""
        if history_text:
            conversation_context = f"""
            此外，请考虑以下对话历史背景，它可能对理解用户当前问题的上下文很重要：
            
            {history_text}
            """

        if use_citation:
            prompt = f"""
            你是一个知识丰富的助手。请回答用户的问题，主要基于以下提供的文档内容，同时参考对话历史上下文。
            
            如果文档内容不足以回答问题，可以使用你自己的知识，但请明确指出哪些信息来自文档、哪些是你的补充。
            
            在回答中，请使用方括号引用相关的文档编号（例如[1]、[2]等），以说明信息来源。文档引用应该放在相关句子的末尾。如果使用自己的知识，可以不加引用。
            
            文档内容:
            {context_text}
            {conversation_context}
            
            用户问题: {query}
            
            请提供一个全面、准确且有条理的回答：
            """
        else:
            prompt = f"""
            你是一个知识丰富的助手。请回答用户的问题，主要基于以下提供的文档内容，同时参考对话历史上下文。
            
            如果文档内容不足以回答问题，可以使用你自己的知识，但请明确指出哪些信息来自文档、哪些是你的补充。
            
            文档内容:
            {context_text}
            {conversation_context}
            
            用户问题: {query}
            
            请提供一个全面、准确且有条理的回答：
            """

        return prompt


class AdaptiveResponseSynthesizer(ContextualResponseSynthesizer):
    """自适应响应合成器 - 根据查询类型选择合成策略"""

    def __init__(self, llm: BaseLLM):
        """
        初始化自适应响应合成器

        Args:
            llm: 语言模型实例
        """
        super().__init__(llm)
        logger.info("自适应响应合成器初始化完成")

    def synthesize_adaptive(self,
                           query: str,
                           documents: List[Document],
                           query_type: str = "general",
                           conversation_history: Optional[List[Dict[str, str]]] = None,
                           use_citation: bool = True) -> Dict[str, Any]:
        """
        根据查询类型自适应合成响应

        Args:
            query: 用户查询
            documents: 检索到的文档
            query_type: 查询类型
            conversation_history: 对话历史
            use_citation: 是否使用引用

        Returns:
            合成的响应结果
        """
        try:
            # 根据查询类型选择合成策略
            if query_type == "comparison":
                return self._synthesize_comparison(query, documents, use_citation)
            elif query_type == "procedural":
                return self._synthesize_procedural(query, documents, use_citation)
            elif query_type == "factual":
                return self._synthesize_factual(query, documents, use_citation)
            else:
                # 默认使用上下文感知合成
                if conversation_history:
                    return self.synthesize_with_history(
                        query, documents, conversation_history, use_citation
                    )
                else:
                    return self.synthesize(query, documents, use_citation)

        except Exception as e:
            error_msg = f"自适应响应合成失败: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": f"生成回答时出现错误: {str(e)}",
                "sources": self._extract_sources(documents),
                "has_context": bool(documents),
                "error": str(e)
            }

    def _synthesize_comparison(self, query: str, documents: List[Document], use_citation: bool) -> Dict[str, Any]:
        """合成比较类型的响应"""
        contexts, sources = self._extract_contexts_and_sources(documents)
        context_text = "\n\n".join(contexts)

        prompt = f"""
        你是一个专业的分析师。用户询问了一个比较问题，请基于提供的信息进行客观、全面的比较分析。
        
        请按照以下结构组织你的回答：
        1. 概述：简要说明比较的对象
        2. 相同点：列出主要的相似之处
        3. 不同点：详细分析关键差异
        4. 总结：提供简洁的结论
        
        {"请在回答中使用方括号引用相关信息来源（例如[1]、[2]等）。" if use_citation else ""}
        
        信息来源:
        {context_text}
        
        用户问题: {query}
        
        比较分析:
        """

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
            "synthesis_type": "comparison",
            "context_count": len(contexts),
            "has_context": True
        }

    def _synthesize_procedural(self, query: str, documents: List[Document], use_citation: bool) -> Dict[str, Any]:
        """合成过程类型的响应"""
        contexts, sources = self._extract_contexts_and_sources(documents)
        context_text = "\n\n".join(contexts)

        prompt = f"""
        你是一个实用的指导助手。用户询问了一个操作或过程问题，请基于提供的信息给出清晰、可操作的指导。
        
        请按照以下结构组织你的回答：
        1. 准备工作：列出需要的工具、材料或前置条件
        2. 详细步骤：按顺序列出具体的操作步骤
        3. 注意事项：提醒重要的注意点或可能的问题
        4. 预期结果：说明完成后的预期效果
        
        {"请在回答中使用方括号引用相关信息来源（例如[1]、[2]等）。" if use_citation else ""}
        
        信息来源:
        {context_text}
        
        用户问题: {query}
        
        操作指南:
        """

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
            "synthesis_type": "procedural",
            "context_count": len(contexts),
            "has_context": True
        }

    def _synthesize_factual(self, query: str, documents: List[Document], use_citation: bool) -> Dict[str, Any]:
        """合成事实类型的响应"""
        contexts, sources = self._extract_contexts_and_sources(documents)
        context_text = "\n\n".join(contexts)

        prompt = f"""
        你是一个准确的信息提供者。用户询问了一个事实性问题，请基于提供的信息给出准确、简洁的回答。
        
        请确保回答：
        1. 直接回答用户的问题
        2. 提供准确的事实信息
        3. 避免不必要的修饰或推测
        4. 如果信息不完整，明确指出
        
        {"请在回答中使用方括号引用相关信息来源（例如[1]、[2]等）。" if use_citation else ""}
        
        信息来源:
        {context_text}
        
        用户问题: {query}
        
        事实回答:
        """

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
            "synthesis_type": "factual",
            "context_count": len(contexts),
            "has_context": True
        }