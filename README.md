# RAG系统

一个功能完整、结构清晰的检索增强生成(RAG)系统，支持文档索引、智能搜索和上下文感知问答。

## 🌟 主要特性

### 核心功能
- **文档处理**: 支持多种文件格式（PDF、Word、文本、Markdown）
- **智能索引**: 基于ElasticSearch的高效向量存储
- **混合检索**: 结合语义搜索和关键词搜索
- **基础RAG**: 无状态的智能问答服务
- **上下文感知RAG**: 支持对话历史的智能问答

### 高级特性
- **查询重写**: 自动优化用户查询以提高检索质量
- **查询分解**: 将复杂查询分解为多个子查询
- **重排序**: 使用LLM对检索结果进行智能重排序
- **命题提取**: 使用LLM提取文档中的原子命题
- **引用支持**: 生成带有来源引用的回答
- **对话记忆**: 管理多轮对话的上下文信息

## 🏗️ 系统架构

```
RAG系统/
├── core/                          # 核心层
│   ├── base/                      # 基础接口和抽象类
│   ├── config/                    # 配置管理
│   └── exceptions/                # 异常定义
├── components/                    # 组件层
│   ├── llm/                       # 语言模型实现
│   ├── embeddings/                # 嵌入模型实现
│   ├── vectorstore/               # 向量存储实现
│   ├── document/                  # 文档处理组件
│   ├── retrieval/                 # 检索组件
│   ├── query/                     # 查询处理组件
│   ├── conversation/              # 对话管理组件
│   └── response/                  # 响应生成组件
├── services/                      # 服务层
│   ├── indexing.py                # 索引服务
│   ├── search.py                  # 搜索服务
│   ├── rag.py                     # 基础RAG服务
│   └── contextual_rag.py          # 上下文感知RAG服务
├── api/                           # API接口层
│   ├── app.py                     # FastAPI应用主文件
│   └── routers/                   # 路由模块
└── main.py                        # 项目主入口
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- ElasticSearch 7.0+
- 充足的内存用于加载嵌入模型

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
# 设置API密钥
export OPENAI_API_KEY="your-openai-api-key"
export TONGYI_API_KEY="your-tongyi-api-key"

# 设置ElasticSearch地址
export ELASTICSEARCH_URL="http://localhost:9200"

# 可选：设置模型缓存目录
export TRANSFORMERS_CACHE="/path/to/cache"
```

### 启动ElasticSearch

```bash
# 使用Docker启动
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.17.0
```

### 运行系统

#### 1. 命令行方式

```bash
# 查看帮助
python main.py --help

# 索引文档
python main.py index --path ./data/documents

# 搜索文档
python main.py search --text "你的搜索词"

# RAG问答
python main.py rag --query "你的问题"

# 上下文感知RAG问答
python main.py rag --query "你的问题" --contextual

# 启动API服务
python main.py serve --host 0.0.0.0 --port 8000
```

#### 2. API方式

```bash
# 启动API服务
python main.py serve

# 或者直接运行
python api/app.py
```

访问 `http://localhost:8000/docs` 查看API文档。

使用 `ui.html` 索引`test.md`进行测试。运行截图如下：
![1749712842134_6etw152nlb.png](https://71e6ab2.webp.li/1749712842134_6etw152nlb.png)
## 📚 模块说明

### 核心层 (core/)

提供系统的基础设施，包括：

- **基础接口**: 定义LLM、嵌入模型、检索器等组件的统一接口
- **配置管理**: 集中管理所有配置项和环境变量
- **异常处理**: 定义系统的异常类型和错误处理机制

### 组件层 (components/)

实现具体的功能组件：

- **语言模型**: OpenAI兼容的LLM实现
- **嵌入模型**: BGE中文嵌入模型等
- **向量存储**: ElasticSearch向量存储实现
- **文档处理**: 文档加载、预处理、分割
- **检索组件**: 混合检索、重排序等
- **查询处理**: 查询重写、分解、分析
- **对话管理**: 会话管理、上下文分析
- **响应生成**: 回答合成、引用生成

### 服务层 (services/)

提供高级业务服务：

- **索引服务**: 统一的文档索引接口
- **搜索服务**: 多种搜索策略的统一接口
- **RAG服务**: 基础的问答服务
- **上下文感知RAG**: 支持对话历史的问答服务

### API层 (api/)

提供REST API接口：

- **文档API**: 文档上传、索引、搜索
- **RAG API**: 基础和上下文感知的问答接口
- **会话管理**: 对话会话的创建、管理、清理

## 🛠️ 配置选项

### 基础配置

```python
# core/config/settings.py
API_HOST = "0.0.0.0"
API_PORT = 8000
DOCUMENTS_DIR = "./data/documents"
ELASTICSEARCH_URL = "http://localhost:9200"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
```

### LLM配置(按需配置即可)

```python
#core/config/setting.py
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY", "your-tongyi-api-key")
# core/config/llm_config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 8192
TEMPERATURE = 0.7
```

## 🔧 高级功能

### 查询重写

自动优化用户查询，提高检索准确性：

```python
# 启用查询重写
rag_service = RAGService(use_query_rewriting=True)
```

### 查询分解

将复杂查询分解为多个子查询：

```python
# 启用查询分解
rag_service = RAGService(use_query_decomposition=True)
```

### 重排序策略

提供多种重排序策略：

```python
# LLM重排序
rag_service = RAGService(reranking_strategy="llm")

# 分数重排序
rag_service = RAGService(reranking_strategy="score")

# 混合重排序
rag_service = RAGService(reranking_strategy="hybrid")
```

### 命题提取

使用LLM提取文档中的原子命题：

```python
# 启用命题提取
indexing_service = IndexingService(use_proposition_extraction=True)
```

## 📊 性能优化

### 检索优化

- **混合检索**: 结合语义和关键词搜索
- **权重调节**: 可调整语义和关键词权重
- **重排序**: 使用LLM提高结果相关性

### 索引优化

- **并发处理**: 支持并发文档处理
- **增量索引**: 支持增量文档添加
- **命题提取**: 提高检索精度

### 对话优化

- **会话管理**: 高效的对话状态管理
- **上下文分析**: 智能的上下文理解
- **记忆机制**: 自动清理过期会话

## 🔍 监控和调试

### 系统状态

```bash
# 查看系统状态
python main.py status

# 查看详细状态
python main.py status --detail
```

### 健康检查

```bash
# API健康检查
curl http://localhost:8000/health

# 文档服务健康检查
curl http://localhost:8000/api/v1/documents/health

# RAG服务健康检查
curl http://localhost:8000/api/v1/rag/health
```

### 日志配置

```python
# 设置日志级别
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```


