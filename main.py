"""
RAG系统主入口文件
支持命令行操作和API服务启动
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config.settings import settings
from core.config.llm_config import llm_config_manager
from services.indexing import IndexingService
from services.search import SearchService
from services.rag import RAGService
from services.contextual_rag import ContextualRAGService

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def setup_environment():
    """设置环境变量和配置"""
    # 设置 Hugging Face 镜像和缓存路径
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/transformers_cache')

    # 验证配置
    if not settings.validate_config():
        logger.error("配置验证失败")
        sys.exit(1)

    # 验证LLM配置
    if not llm_config_manager.validate_all_configs():
        logger.error("LLM配置验证失败")
        sys.exit(1)

    logger.info("环境配置完成")


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="高级RAG系统命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s index --path /path/to/documents    # 索引指定目录
  %(prog)s query --text "你的问题"            # 执行搜索查询
  %(prog)s rag --query "智能问答"             # 执行RAG问答
  %(prog)s serve --host 0.0.0.0 --port 8000  # 启动API服务
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 索引命令
    index_parser = subparsers.add_parser("index", help="索引文档")
    index_parser.add_argument("--path", "-p", help="要索引的文件或目录路径")
    index_parser.add_argument("--delete", "-d", action="store_true", help="删除现有索引后重新索引")
    index_parser.add_argument("--recreate", action="store_true", help="重新创建索引结构")
    index_parser.add_argument("--concurrent", "-c", type=int, default=3, help="并发文件数量")

    # 搜索命令
    search_parser = subparsers.add_parser("search", help="搜索文档")
    search_parser.add_argument("--text", "-t", required=True, help="搜索文本")
    search_parser.add_argument("--top-k", "-k", type=int, default=4, help="返回的最大文档数量")
    search_parser.add_argument("--type", choices=["hybrid", "semantic", "keyword"],
                              default="hybrid", help="搜索类型")

    # RAG查询命令
    rag_parser = subparsers.add_parser("rag", help="RAG问答")
    rag_parser.add_argument("--query", "-q", required=True, help="问答查询")
    rag_parser.add_argument("--top-k", "-k", type=int, default=5, help="检索的最大文档数量")
    rag_parser.add_argument("--contextual", action="store_true", help="使用上下文感知RAG")
    rag_parser.add_argument("--session-id", help="会话ID（用于上下文感知RAG）")
    rag_parser.add_argument("--no-rewrite", action="store_true", help="禁用查询重写")
    rag_parser.add_argument("--no-decompose", action="store_true", help="禁用查询分解")
    rag_parser.add_argument("--no-rerank", action="store_true", help="禁用重排序")
    rag_parser.add_argument("--no-citation", action="store_true", help="禁用引用")

    # 服务启动命令
    serve_parser = subparsers.add_parser("serve", help="启动API服务")
    serve_parser.add_argument("--host", default=settings.API_HOST, help="主机地址")
    serve_parser.add_argument("--port", "-p", type=int, default=settings.API_PORT, help="端口号")
    serve_parser.add_argument("--reload", action="store_true", help="启用自动重载")
    serve_parser.add_argument("--workers", type=int, default=1, help="工作进程数量")

    # 状态命令
    status_parser = subparsers.add_parser("status", help="查看系统状态")
    status_parser.add_argument("--detail", action="store_true", help="显示详细信息")

    # 配置命令
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_parser.add_argument("--show", action="store_true", help="显示当前配置")
    config_parser.add_argument("--test", action="store_true", help="测试配置")

    return parser


async def handle_index_command(args):
    """处理索引命令"""
    try:
        indexing_service = IndexingService(max_concurrent_files=args.concurrent)

        if args.recreate:
            logger.info("重新创建索引结构...")
            result = indexing_service.recreate_index()
            print(json.dumps(result, ensure_ascii=False, indent=2))

        if args.delete:
            logger.info("删除现有索引...")
            indexing_service.delete_all_indexes()

        path = args.path or settings.DOCUMENTS_DIR

        if os.path.isfile(path):
            logger.info(f"索引文件: {path}")
            result = await indexing_service.index_file(path)
        else:
            logger.info(f"索引目录: {path}")
            result = await indexing_service.index_directory(path)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"索引操作失败: {str(e)}")
        sys.exit(1)


async def handle_search_command(args):
    """处理搜索命令"""
    try:
        search_service = SearchService()

        if args.type == "semantic":
            result = search_service.semantic_search(args.text, top_k=args.top_k)
        elif args.type == "keyword":
            result = search_service.keyword_search(args.text, top_k=args.top_k)
        else:  # hybrid
            result = search_service.search_documents(args.text, top_k=args.top_k)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"搜索操作失败: {str(e)}")
        sys.exit(1)


async def handle_rag_command(args):
    """处理RAG问答命令"""
    try:
        # 构建RAG配置
        rag_config = {
            "use_query_rewriting": not args.no_rewrite,
            "use_query_decomposition": not args.no_decompose,
            "use_reranking": not args.no_rerank,
            "use_citation": not args.no_citation
        }

        if args.contextual:
            # 使用上下文感知RAG
            rag_service = ContextualRAGService(**rag_config)
            result = await rag_service.query_async(
                args.query,
                session_id=args.session_id,
                top_k=args.top_k
            )
        else:
            # 使用基础RAG
            rag_service = RAGService(**rag_config)
            result = await rag_service.query_async(args.query, top_k=args.top_k)

        # 格式化输出
        print("=" * 60)
        print(f"查询: {args.query}")
        print("=" * 60)
        print(f"回答: {result.get('answer', 'N/A')}")

        if result.get("sources"):
            print("\n来源:")
            for i, source in enumerate(result["sources"], 1):
                filename = source.get("filename", "未知文件")
                score = source.get("score", 0)
                print(f"  [{i}] {filename} (相关性: {score:.3f})")

        if args.contextual and "session_id" in result:
            print(f"\n会话ID: {result['session_id']}")

        print(f"\n处理时间: {result.get('processing_time', 0):.2f}秒")

        # 保存完整结果到文件（可选）
        if os.environ.get("RAG_SAVE_RESULTS"):
            with open("rag_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("完整结果已保存到 rag_result.json")

    except Exception as e:
        logger.error(f"RAG问答失败: {str(e)}")
        sys.exit(1)


def handle_serve_command(args):
    """处理服务启动命令"""
    try:
        import uvicorn
        from api.app import app

        logger.info(f"启动RAG API服务器: {args.host}:{args.port}")

        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=settings.LOG_LEVEL.lower()
        )

    except Exception as e:
        logger.error(f"启动API服务失败: {str(e)}")
        sys.exit(1)


async def handle_status_command(args):
    """处理状态查看命令"""
    try:
        status_info = {}

        # 基本信息
        status_info["system"] = {
            "python_version": sys.version,
            "project_root": str(project_root),
            "documents_directory": settings.DOCUMENTS_DIR,
            "log_level": settings.LOG_LEVEL
        }

        # 索引状态
        try:
            indexing_service = IndexingService()
            status_info["indexing"] = indexing_service.get_index_statistics()
        except Exception as e:
            status_info["indexing"] = {"error": str(e)}

        # 搜索状态
        try:
            search_service = SearchService()
            status_info["search"] = search_service.test_connection()
        except Exception as e:
            status_info["search"] = {"error": str(e)}

        if args.detail:
            # RAG服务状态
            try:
                rag_service = RAGService()
                status_info["rag"] = rag_service.get_service_info()
            except Exception as e:
                status_info["rag"] = {"error": str(e)}

            # 上下文感知RAG服务状态
            try:
                contextual_rag = ContextualRAGService()
                status_info["contextual_rag"] = contextual_rag.get_service_info()
            except Exception as e:
                status_info["contextual_rag"] = {"error": str(e)}

        print(json.dumps(status_info, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"获取状态信息失败: {str(e)}")
        sys.exit(1)


def handle_config_command(args):
    """处理配置管理命令"""
    try:
        if args.show:
            config_info = {
                "settings": {
                    "API_HOST": settings.API_HOST,
                    "API_PORT": settings.API_PORT,
                    "DOCUMENTS_DIR": settings.DOCUMENTS_DIR,
                    "ELASTICSEARCH_URL": settings.ELASTICSEARCH_URL,
                    "ELASTICSEARCH_INDEX_NAME": settings.ELASTICSEARCH_INDEX_NAME,
                    "CHUNK_SIZE": settings.CHUNK_SIZE,
                    "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
                    "LOG_LEVEL": settings.LOG_LEVEL
                },
                "llm_config": {
                    "openai_model": llm_config_manager.openai_config.model,
                    "openai_api_base": llm_config_manager.openai_config.api_base,
                    "temperature": llm_config_manager.openai_config.temperature,
                    "max_tokens": llm_config_manager.openai_config.max_tokens
                }
            }
            print(json.dumps(config_info, ensure_ascii=False, indent=2))

        if args.test:
            print("测试配置...")

            # 测试基础配置
            config_valid = settings.validate_config()
            print(f"基础配置: {'✓' if config_valid else '✗'}")

            # 测试LLM配置
            llm_valid = llm_config_manager.validate_all_configs()
            print(f"LLM配置: {'✓' if llm_valid else '✗'}")

            # 测试目录
            docs_dir_exists = os.path.exists(settings.DOCUMENTS_DIR)
            print(f"文档目录: {'✓' if docs_dir_exists else '✗'} ({settings.DOCUMENTS_DIR})")

            overall_status = config_valid and llm_valid and docs_dir_exists
            print(f"\n总体状态: {'✓ 配置正常' if overall_status else '✗ 配置异常'}")

            if not overall_status:
                sys.exit(1)

    except Exception as e:
        logger.error(f"配置操作失败: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    # 设置环境
    setup_environment()

    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 根据命令执行相应操作
    try:
        if args.command == "index":
            asyncio.run(handle_index_command(args))
        elif args.command == "search":
            asyncio.run(handle_search_command(args))
        elif args.command == "rag":
            asyncio.run(handle_rag_command(args))
        elif args.command == "serve":
            handle_serve_command(args)  # 🚫 不用 asyncio.run
        elif args.command == "status":
            asyncio.run(handle_status_command(args))
        elif args.command == "config":
            handle_config_command(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"命令执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
