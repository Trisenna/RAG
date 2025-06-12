"""
文档相关API路由
处理文档上传、索引、搜索等操作
"""

import os
import shutil
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException, status
from fastapi.responses import JSONResponse

from core.config.settings import settings
from core.exceptions.errors import IndexingException, RetrievalException
from services.indexing import IndexingService
from services.search import SearchService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/documents", tags=["documents"])

# 初始化服务
indexing_service = IndexingService()
search_service = SearchService()


@router.post("/upload",
             summary="上传并索引文档",
             description="上传单个文档文件并进行索引")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """上传并索引单个文档"""
    try:
        # 验证文件
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件名不能为空"
            )

        # 确保文档目录存在
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)

        # 检查文件类型
        if not indexing_service.document_loader.factory.is_supported(file.filename):
            supported_types = indexing_service.document_loader.get_supported_extensions()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型。支持的类型: {', '.join(supported_types)}"
            )

        # 保存文件
        file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)

        # 检查文件是否已存在
        if os.path.exists(file_path):
            logger.warning(f"文件已存在，将覆盖: {file_path}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"文件保存成功: {file_path}")

        # 索引文件
        result = await indexing_service.index_file(file_path)

        if result["status"] == "error":
            # 如果索引失败，删除已保存的文件
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "文档上传并索引成功",
                "filename": file.filename,
                "file_path": file_path,
                "indexing_result": result
            }
        )

    except HTTPException:
        raise
    except IndexingException as e:
        logger.error(f"索引文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"索引文档失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传文档失败: {str(e)}"
        )


@router.post("/index",
             summary="索引所有文档",
             description="索引文档目录中的所有文档")
async def index_all_documents() -> JSONResponse:
    """索引文档目录中的所有文档"""
    try:
        result = await indexing_service.index_directory()

        status_code = status.HTTP_200_OK
        if result.get("failed_files", 0) > 0:
            status_code = status.HTTP_207_MULTI_STATUS

        return JSONResponse(
            status_code=status_code,
            content={
                "message": "文档索引操作完成",
                "indexing_result": result
            }
        )

    except IndexingException as e:
        logger.error(f"批量索引失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量索引失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"批量索引失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量索引失败: {str(e)}"
        )


@router.delete("/indexes",
               summary="删除所有索引",
               description="删除ElasticSearch中的所有文档索引")
async def delete_all_indexes() -> JSONResponse:
    """删除所有索引"""
    try:
        result = indexing_service.delete_all_indexes()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "索引删除成功",
                "delete_result": result
            }
        )

    except IndexingException as e:
        logger.error(f"删除索引失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除索引失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"删除索引失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除索引失败: {str(e)}"
        )


@router.post("/indexes/recreate",
             summary="重新创建索引",
             description="删除现有索引并重新创建")
async def recreate_indexes() -> JSONResponse:
    """重新创建索引"""
    try:
        result = indexing_service.recreate_index()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "索引重建成功",
                "recreate_result": result
            }
        )

    except IndexingException as e:
        logger.error(f"重建索引失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重建索引失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"重建索引失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重建索引失败: {str(e)}"
        )


@router.get("/search",
            summary="搜索文档",
            description="在索引中搜索相关文档")
async def search_documents(
    query: str = Query(..., description="搜索查询"),
    top_k: int = Query(4, ge=1, le=20, description="返回的最大文档数量"),
    search_type: str = Query("hybrid", regex="^(hybrid|semantic|keyword)$", description="搜索类型")
) -> JSONResponse:
    """搜索相关文档"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本不能为空"
            )

        # 根据搜索类型选择搜索方法
        if search_type == "semantic":
            result = search_service.semantic_search(query, top_k=top_k)
        elif search_type == "keyword":
            result = search_service.keyword_search(query, top_k=top_k)
        else:  # hybrid
            result = search_service.search_documents(query, top_k=top_k)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "搜索完成",
                "search_result": result
            }
        )

    except HTTPException:
        raise
    except RetrievalException as e:
        logger.error(f"文档搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档搜索失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"文档搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档搜索失败: {str(e)}"
        )


@router.post("/search/filtered",
             summary="过滤搜索",
             description="使用过滤条件搜索文档")
async def filtered_search(
    query: str = Form(..., description="搜索查询"),
    filters: Dict[str, Any] = Form(..., description="过滤条件"),
    top_k: int = Form(4, ge=1, le=20, description="返回的最大文档数量")
) -> JSONResponse:
    """带过滤条件的文档搜索"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本不能为空"
            )

        result = search_service.search_with_filters(query, filters, top_k=top_k)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "过滤搜索完成",
                "search_result": result
            }
        )

    except HTTPException:
        raise
    except RetrievalException as e:
        logger.error(f"过滤搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"过滤搜索失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"过滤搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"过滤搜索失败: {str(e)}"
        )


@router.get("/statistics",
            summary="获取文档统计信息",
            description="获取索引和搜索相关的统计信息")
async def get_document_statistics() -> JSONResponse:
    """获取文档统计信息"""
    try:
        indexing_stats = indexing_service.get_index_statistics()
        search_stats = search_service.get_search_statistics()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "indexing_statistics": indexing_stats,
                "search_statistics": search_stats,
                "documents_directory": settings.DOCUMENTS_DIR
            }
        )

    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.put("/search/weights",
            summary="更新搜索权重",
            description="更新混合搜索的语义和关键词权重")
async def update_search_weights(
    semantic_weight: float = Form(..., ge=0.0, le=1.0, description="语义搜索权重"),
    keyword_weight: float = Form(..., ge=0.0, le=1.0, description="关键词搜索权重")
) -> JSONResponse:
    """更新搜索权重"""
    try:
        result = search_service.update_retrieval_weights(semantic_weight, keyword_weight)

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "搜索权重更新成功",
                "update_result": result
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新搜索权重失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新搜索权重失败: {str(e)}"
        )


@router.get("/health",
            summary="文档服务健康检查",
            description="检查文档相关服务的健康状态")
async def document_health_check() -> JSONResponse:
    """文档服务健康检查"""
    try:
        # 测试搜索服务连接
        connection_status = search_service.test_connection()

        # 获取基本统计信息
        stats = indexing_service.get_index_statistics()

        health_status = {
            "status": "healthy" if connection_status["status"] == "connected" else "unhealthy",
            "connection": connection_status,
            "document_count": stats.get("total_documents", 0),
            "supported_file_types": stats.get("supported_file_types", []),
            "documents_directory": settings.DOCUMENTS_DIR,
            "directory_exists": os.path.exists(settings.DOCUMENTS_DIR)
        }

        status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(
            status_code=status_code,
            content=health_status
        )

    except Exception as e:
        logger.error(f"文档服务健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )