"""
RAG相关API路由
处理基础RAG和上下文感知RAG查询
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Form, Query, HTTPException, status, Depends, Cookie
from fastapi.responses import JSONResponse

from core.exceptions.errors import LLMException, ConversationException
from services.rag import RAGService
from services.contextual_rag import ContextualRAGService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/rag", tags=["rag"])

# 初始化服务（使用默认配置）
basic_rag_service = RAGService()
contextual_rag_service = ContextualRAGService()


def get_session_id(session_id: Optional[str] = Cookie(None)):
    """从Cookie获取会话ID，如果不存在则返回None"""
    return session_id


@router.post("/query",
             summary="基础RAG查询",
             description="执行不考虑对话历史的RAG查询")
async def basic_rag_query(
    query: str = Form(..., description="用户查询"),
    top_k: int = Form(5, ge=1, le=20, description="返回的最大文档数量"),
    use_decomposition: bool = Form(True, description="是否使用查询分解"),
    use_reranking: bool = Form(True, description="是否使用重排序"),
    use_citation: bool = Form(True, description="是否使用引用"),
    reranking_strategy: str = Form("hybrid", regex="^(llm|score|hybrid)$", description="重排序策略")
) -> JSONResponse:
    """基础RAG查询"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本不能为空"
            )

        # 创建临时配置的RAG服务实例
        custom_rag = RAGService(
            use_query_decomposition=use_decomposition,
            use_reranking=use_reranking,
            use_citation=use_citation,
            reranking_strategy=reranking_strategy
        )

        # 执行查询
        result = await custom_rag.query_async(query, top_k=top_k)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "RAG查询失败")
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "RAG查询完成",
                "result": result
            }
        )

    except HTTPException:
        raise
    except LLMException as e:
        logger.error(f"RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG查询失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG查询失败: {str(e)}"
        )


@router.post("/contextual/query",
             summary="上下文感知RAG查询",
             description="执行考虑对话历史的RAG查询")
async def contextual_rag_query(
    query: str = Form(..., description="用户查询"),
    session_id: Optional[str] = Depends(get_session_id),
    top_k: int = Form(5, ge=1, le=20, description="返回的最大文档数量"),
    use_decomposition: bool = Form(True, description="是否使用查询分解"),
    use_reranking: bool = Form(True, description="是否使用重排序"),
    use_citation: bool = Form(True, description="是否使用引用"),
    reranking_strategy: str = Form("hybrid", regex="^(llm|score|hybrid)$", description="重排序策略")
) -> JSONResponse:
    """上下文感知RAG查询"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本不能为空"
            )

        # 创建临时配置的上下文感知RAG服务实例
        custom_rag = ContextualRAGService(
            use_query_decomposition=use_decomposition,
            use_reranking=use_reranking,
            use_citation=use_citation,
            reranking_strategy=reranking_strategy
        )

        # 执行查询
        result = await custom_rag.query_async(query, session_id, top_k=top_k)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "上下文感知RAG查询失败")
            )

        # 创建响应，如果需要设置会话Cookie
        response_content = {
            "message": "上下文感知RAG查询完成",
            "result": result
        }

        response = JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_content
        )

        # 如果是新会话或会话ID变更，设置Cookie
        if not session_id and "session_id" in result:
            response.set_cookie(
                key="session_id",
                value=result["session_id"],
                httponly=True,
                max_age=3600 * 24 * 7,  # 7天过期
                samesite="lax"
            )

        return response

    except HTTPException:
        raise
    except (LLMException, ConversationException) as e:
        logger.error(f"上下文感知RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上下文感知RAG查询失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"上下文感知RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上下文感知RAG查询失败: {str(e)}"
        )


@router.post("/contextual/session/new",
             summary="创建新会话",
             description="创建新的对话会话")
async def create_new_session() -> JSONResponse:
    """创建新会话"""
    try:
        session_id = contextual_rag_service.create_session()

        response = JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "新会话创建成功",
                "session_id": session_id
            }
        )

        # 设置会话Cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=3600 * 24 * 7,  # 7天过期
            samesite="lax"
        )

        return response

    except Exception as e:
        logger.error(f"创建新会话失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建新会话失败: {str(e)}"
        )


@router.post("/contextual/session/clear",
             summary="清除会话历史",
             description="清除指定会话的对话历史")
async def clear_session(
    session_id: str = Form(..., description="会话ID")
) -> JSONResponse:
    """清除会话历史"""
    try:
        success = contextual_rag_service.clear_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在或清除失败"
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "会话历史已清除",
                "session_id": session_id
            }
        )

    except HTTPException:
        raise
    except ConversationException as e:
        logger.error(f"清除会话历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "不存在" in str(e) else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清除会话历史失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"清除会话历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清除会话历史失败: {str(e)}"
        )


@router.delete("/contextual/session/{session_id}",
               summary="删除会话",
               description="删除指定的对话会话")
async def delete_session(session_id: str) -> JSONResponse:
    """删除会话"""
    try:
        success = contextual_rag_service.delete_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在或删除失败"
            )

        response = JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "会话已删除",
                "session_id": session_id
            }
        )

        # 清除会话Cookie
        response.delete_cookie(key="session_id")

        return response

    except HTTPException:
        raise
    except ConversationException as e:
        logger.error(f"删除会话失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "不存在" in str(e) else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除会话失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除会话失败: {str(e)}"
        )


@router.get("/contextual/session/{session_id}",
            summary="获取会话信息",
            description="获取指定会话的详细信息")
async def get_session_info(session_id: str) -> JSONResponse:
    """获取会话信息"""
    try:
        session_info = contextual_rag_service.get_session_info(session_id)

        if "error" in session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在或获取信息失败"
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "会话信息获取成功",
                "session_info": session_info
            }
        )

    except HTTPException:
        raise
    except ConversationException as e:
        logger.error(f"获取会话信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "不存在" in str(e) else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话信息失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"获取会话信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话信息失败: {str(e)}"
        )


@router.get("/contextual/session/{session_id}/history",
            summary="获取对话历史",
            description="获取指定会话的对话历史")
async def get_conversation_history(session_id: str) -> JSONResponse:
    """获取对话历史"""
    try:
        history = contextual_rag_service.get_conversation_history(session_id)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "对话历史获取成功",
                "session_id": session_id,
                "history": history,
                "turn_count": len(history)
            }
        )

    except ConversationException as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "不存在" in str(e) else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话历史失败: {e.message}"
        )
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话历史失败: {str(e)}"
        )


@router.post("/contextual/sessions/cleanup",
             summary="清理过期会话",
             description="清理所有过期的对话会话")
async def cleanup_expired_sessions() -> JSONResponse:
    """清理过期会话"""
    try:
        cleaned_count = contextual_rag_service.cleanup_expired_sessions()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"过期会话清理完成，清理了 {cleaned_count} 个会话",
                "cleaned_count": cleaned_count
            }
        )

    except Exception as e:
        logger.error(f"清理过期会话失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清理过期会话失败: {str(e)}"
        )


@router.get("/info",
            summary="获取RAG服务信息",
            description="获取RAG服务的配置和状态信息")
async def get_rag_service_info() -> JSONResponse:
    """获取RAG服务信息"""
    try:
        basic_info = basic_rag_service.get_service_info()
        contextual_info = contextual_rag_service.get_service_info()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "basic_rag": basic_info,
                "contextual_rag": contextual_info
            }
        )

    except Exception as e:
        logger.error(f"获取RAG服务信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取RAG服务信息失败: {str(e)}"
        )


@router.put("/configuration",
            summary="更新RAG配置",
            description="更新RAG服务的配置参数")
async def update_rag_configuration(
    service_type: str = Form("basic", regex="^(basic|contextual)$", description="服务类型"),
    semantic_weight: Optional[float] = Form(None, ge=0.0, le=1.0, description="语义搜索权重"),
    keyword_weight: Optional[float] = Form(None, ge=0.0, le=1.0, description="关键词搜索权重"),
    use_query_rewriting: Optional[bool] = Form(None, description="是否使用查询重写"),
    use_query_decomposition: Optional[bool] = Form(None, description="是否使用查询分解"),
    use_reranking: Optional[bool] = Form(None, description="是否使用重排序"),
    use_citation: Optional[bool] = Form(None, description="是否使用引用"),
    reranking_strategy: Optional[str] = Form(None, regex="^(llm|score|hybrid)$", description="重排序策略"),
    max_history_turns: Optional[int] = Form(None, ge=1, le=20, description="最大历史轮次（仅上下文服务）")
) -> JSONResponse:
    """更新RAG配置"""
    try:
        # 构建更新参数
        update_params = {}

        if semantic_weight is not None:
            update_params["semantic_weight"] = semantic_weight
        if keyword_weight is not None:
            update_params["keyword_weight"] = keyword_weight
        if use_query_rewriting is not None:
            update_params["use_query_rewriting"] = use_query_rewriting
        if use_query_decomposition is not None:
            update_params["use_query_decomposition"] = use_query_decomposition
        if use_reranking is not None:
            update_params["use_reranking"] = use_reranking
        if use_citation is not None:
            update_params["use_citation"] = use_citation
        if reranking_strategy is not None:
            update_params["reranking_strategy"] = reranking_strategy
        if max_history_turns is not None and service_type == "contextual":
            update_params["max_history_turns"] = max_history_turns

        if not update_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="没有提供有效的更新参数"
            )

        # 选择服务并更新配置
        if service_type == "basic":
            result = basic_rag_service.update_configuration(**update_params)
        else:  # contextual
            result = contextual_rag_service.update_configuration(**update_params)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "配置更新失败")
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "RAG配置更新成功",
                "service_type": service_type,
                "update_result": result
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新RAG配置失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新RAG配置失败: {str(e)}"
        )


@router.get("/health",
            summary="RAG服务健康检查",
            description="检查RAG服务的健康状态")
async def rag_health_check() -> JSONResponse:
    """RAG服务健康检查"""
    try:
        # 测试基础RAG服务
        basic_test = basic_rag_service.test_query_pipeline()

        # 测试上下文感知RAG服务
        contextual_test = contextual_rag_service.test_contextual_pipeline()

        overall_health = "healthy" if (
            basic_test.get("overall_health") == "healthy" and
            contextual_test.get("overall_health") == "healthy"
        ) else "unhealthy"

        health_status = {
            "status": overall_health,
            "basic_rag": basic_test,
            "contextual_rag": contextual_test,
            "timestamp": time.time()
        }

        status_code = status.HTTP_200_OK if overall_health == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(
            status_code=status_code,
            content=health_status
        )

    except Exception as e:
        logger.error(f"RAG服务健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# 导入time模块
import time