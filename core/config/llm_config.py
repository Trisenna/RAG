"""
LLM相关配置管理
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenAIConfig:
    """OpenAI LLM配置"""
    api_key: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    api_base: str = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model: str = os.getenv("DEFAULT_CHAT_MODEL", "qwen-plus")
    max_tokens: int = 8192
    temperature: float = 0.7

    def validate(self) -> bool:
        """验证配置有效性"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not self.api_base:
            raise ValueError("OpenAI API base URL is required")
        return True


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    # BGE配置
    bge_model_name: str = "BAAI/bge-small-zh"

    # OpenAI嵌入配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    openai_api_base: str = os.getenv("OPENAI_E_BASE_URL", "https://llmapi.blsc.cn/v1")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "Embedding-3")

    # 千问嵌入配置
    qwen_model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"


class LLMConfigManager:
    """LLM配置管理器"""

    def __init__(self):
        self.openai_config = OpenAIConfig()
        self.embedding_config = EmbeddingConfig()

    def get_openai_config(self) -> OpenAIConfig:
        """获取OpenAI配置"""
        return self.openai_config

    def get_embedding_config(self) -> EmbeddingConfig:
        """获取嵌入模型配置"""
        return self.embedding_config

    def validate_all_configs(self) -> bool:
        """验证所有配置"""
        try:
            self.openai_config.validate()
            return True
        except Exception as e:
            print(f"LLM配置验证失败: {e}")
            return False


# 全局LLM配置管理器实例
llm_config_manager = LLMConfigManager()