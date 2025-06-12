"""
OpenAI LLM实现
基于OpenAI API的语言模型实现
"""

import openai
import logging
from typing import List, Dict, Any, Optional

from core.base.llm import BaseLLM
from core.config.llm_config import OpenAIConfig, llm_config_manager
from core.exceptions.errors import LLMException

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI语言模型实现"""

    def __init__(self,
                 config: Optional[OpenAIConfig] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        """
        初始化OpenAI LLM

        Args:
            config: OpenAI配置，如果为None则使用默认配置
            temperature: 温度参数，覆盖配置中的默认值
            max_tokens: 最大token数，覆盖配置中的默认值
        """
        self.config = config or llm_config_manager.get_openai_config()
        self.temperature = temperature or self.config.temperature
        self.max_tokens = max_tokens or self.config.max_tokens

        # 验证配置
        self.config.validate()

        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )

        logger.info(f"已初始化OpenAI LLM，模型: {self.config.model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本响应

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数（temperature, max_tokens等）

        Returns:
            生成的文本响应

        Raises:
            LLMException: 当生成失败时
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            result = response.choices[0].message.content
            logger.debug(f"LLM生成成功，输入长度: {len(prompt)}, 输出长度: {len(result)}")
            return result

        except Exception as e:
            error_msg = f"OpenAI API调用失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg, error_code="OPENAI_API_ERROR")

    def generate_with_context(self,
                             query: str,
                             context: List[str],
                             **kwargs) -> str:
        """
        基于上下文生成响应

        Args:
            query: 用户查询
            context: 上下文信息列表
            **kwargs: 其他参数

        Returns:
            基于上下文的响应

        Raises:
            LLMException: 当生成失败时
        """
        system_prompt = kwargs.get(
            "system_prompt",
            "You are a helpful assistant. Answer based only on the provided context."
        )

        # 构建包含上下文的提示
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        full_prompt = (
            f"Context information is below.\n\n{context_text}\n\n"
            f"Given the context information and not prior knowledge, "
            f"answer the following query:\nQuery: {query}\nAnswer:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )

            result = response.choices[0].message.content
            logger.debug(f"基于上下文的LLM生成成功")
            return result

        except Exception as e:
            error_msg = f"OpenAI API上下文生成失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg, error_code="OPENAI_CONTEXT_ERROR")

    def chat(self,
             messages: List[Dict[str, str]],
             **kwargs) -> Dict[str, Any]:
        """
        聊天模式对话

        Args:
            messages: 对话消息列表
            **kwargs: 其他参数

        Returns:
            聊天响应结果

        Raises:
            LLMException: 当聊天失败时
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )

            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

            logger.debug(f"聊天模式LLM调用成功")
            return result

        except Exception as e:
            error_msg = f"OpenAI API聊天调用失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg, error_code="OPENAI_CHAT_ERROR")