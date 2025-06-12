"""
项目主配置文件
集中管理所有配置项
"""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """项目配置类"""

    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent.parent

    # 数据目录配置
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")

    # API服务配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # ElasticSearch配置
    ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "rag-documents")
    ELASTICSEARCH_TIMEOUT = 30
    ELASTICSEARCH_RETRY_ON_TIMEOUT = True

    # 嵌入模型配置
    EMBEDDING_MODEL_DEVICE = os.getenv(
        "EMBEDDING_MODEL_DEVICE",
        "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    )
    EMBEDDING_BATCH_SIZE = 8
    EMBEDDING_MAX_LENGTH = 4096

    # 文档处理配置
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200

    # API密钥配置
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY", "your-tongyi-api-key")

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.DOCUMENTS_DIR, exist_ok=True)

    @classmethod
    def validate_config(cls) -> bool:
        """验证配置有效性"""
        try:
            # 检查必要的API密钥
            if not cls.TONGYI_API_KEY:
                print("警告: TONGYI_API_KEY 未设置")

            # 确保目录存在
            cls.ensure_directories()

            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False


# 全局配置实例
settings = Settings()