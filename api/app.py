"""
FastAPI应用主文件
整合所有API路由和中间件
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config.settings import settings
from core.exceptions.errors import RAGException
from api.routers import documents, rag

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时的初始化
    logger.info("RAG API服务启动中...")

    # 验证配置
    if not settings.validate_config():
        logger.error("配置验证失败")
        raise RuntimeError("配置验证失败")

    # 确保必要的目录存在
    settings.ensure_directories()

    logger.info(f"RAG API服务启动完成，文档目录: {settings.DOCUMENTS_DIR}")

    yield

    # 关闭时的清理
    logger.info("RAG API服务关闭中...")
    # 这里可以添加清理逻辑，如关闭数据库连接等
    logger.info("RAG API服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="RAG系统",
    description="""
   
    """,
    version="2.0.0",
    contact={
        "name": "王明辉",
        "email": "22301022@bjtu.edu.cn",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = time.time()

    # 记录请求信息
    logger.info(f"收到请求: {request.method} {request.url}")

    # 处理请求
    response = await call_next(request)

    # 记录响应信息
    process_time = time.time() - start_time
    logger.info(f"请求完成: {request.method} {request.url} - "
               f"状态码: {response.status_code} - "
               f"耗时: {process_time:.3f}秒")

    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)

    return response


# 全局异常处理器
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP错误",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "请求验证失败",
            "message": "请求参数格式错误",
            "details": exc.errors(),
            "path": str(request.url)
        }
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """RAG系统异常处理"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # 根据错误代码确定HTTP状态码
    if exc.error_code in ["EMPTY_QUERY", "INVALID_PARAMETER"]:
        status_code = status.HTTP_400_BAD_REQUEST
    elif exc.error_code in ["SESSION_NOT_FOUND", "FILE_NOT_FOUND"]:
        status_code = status.HTTP_404_NOT_FOUND

    return JSONResponse(
        status_code=status_code,
        content={
            "error": "RAG系统错误",
            "message": exc.message,
            "error_code": exc.error_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {type(exc).__name__}: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "内部服务器错误",
            "message": "服务器遇到了意外错误，请稍后重试",
            "path": str(request.url)
        }
    )


# 注册路由
app.include_router(documents.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")


# 根路径
@app.get("/",
         summary="API根路径",
         description="返回API基本信息")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用RAG系统",
        "version": "2.0.0",
        "description": "提供文档索引、搜索和智能问答服务",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_prefix": "/api/v1"
    }


@app.get("/health",
         summary="系统健康检查",
         description="检查整个系统的健康状态")
async def system_health_check():
    """系统健康检查"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "api": "running",
                "documents": "检查中...",
                "rag": "检查中..."
            },
            "configuration": {
                "documents_directory": settings.DOCUMENTS_DIR,
                "elasticsearch_url": settings.ELASTICSEARCH_URL,
                "api_host": settings.API_HOST,
                "api_port": settings.API_PORT
            }
        }



        return health_status

    except Exception as e:
        logger.error(f"系统健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.get("/api/v1/info",
         summary="API信息",
         description="获取API的详细信息和配置")
async def api_info():
    """API信息"""
    return {
        "api": {
            "name": "RAG系统",
            "version": "2.0.0",
            "description": "提供文档索引、搜索和智能问答服务"
        },
        "endpoints": {
            "documents": "/api/v1/documents",
            "rag": "/api/v1/rag",
            "health": "/health",
            "docs": "/docs"
        },
        "features": {
            "document_upload": True,
            "document_indexing": True,
            "document_search": True,
            "basic_rag": True,
            "contextual_rag": True,
            "conversation_memory": True,
            "query_rewriting": True,
            "query_decomposition": True,
            "reranking": True,
            "citation": True
        },
        "supported_file_types": [".pdf", ".docx", ".doc", ".txt", ".md"],
        "technology_stack": {
            "web_framework": "FastAPI",
            "vector_store": "ElasticSearch",
            "embedding_model": "BGE (BAAI/bge-small-zh)",
            "llm": "通义千问 (Qwen)",
            "retrieval": "Hybrid (Semantic + Keyword)"
        },
        "configuration": {
            "max_file_size": "unlimited",
            "max_query_length": "unlimited",
            "session_timeout": "24 hours",
            "max_history_turns": 5,
            "default_top_k": 5
        }
    }


# 如果直接运行此文件，启动服务器
if __name__ == "__main__":
    import uvicorn

    logger.info(f"启动RAG API服务器: {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )