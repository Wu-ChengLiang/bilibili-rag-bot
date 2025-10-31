"""FastAPI 服务 - RAG 多轮对话 API"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.models import ChatRequest, ChatResponse
from src.services.rag_chat_service import RAGChatService

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG 多轮对话 API",
    description="支持多轮对话记忆的 RAG 系统 API",
    version="1.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
try:
    llm_api_key = os.getenv("MOONSHOT_API_KEY")
    if not llm_api_key:
        raise ValueError("MOONSHOT_API_KEY not set in environment")

    service = RAGChatService(
        llm_provider="kimi",
        llm_api_key=llm_api_key,
        data_directory=os.getenv("DATA_DIRECTORY", "./data"),
        history_dir=os.getenv("HISTORY_DIR", "./history"),
    )
    logger.info("✅ RAG 服务初始化成功")
except Exception as e:
    logger.error(f"❌ 服务初始化失败: {e}")
    service = None


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "RAG Chat API",
        "initialized": service is not None,
    }


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return service.get_stats()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    处理聊天请求

    请求格式：
    ```json
    {
        "platform": "bilibili",
        "user_id": "123456",
        "user_name": "阿良",
        "message": "你好",
        "history": [
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"}
        ]
    }
    ```

    响应格式：
    ```json
    {
        "success": true,
        "reply": "回答内容"
    }
    ```

    Args:
        request: 聊天请求

    Returns:
        聊天响应
    """
    if not service:
        return ChatResponse(
            success=False,
            error="服务未初始化，请检查 MOONSHOT_API_KEY 环境变量",
        )

    try:
        logger.info(f"[{request.platform}/{request.user_id}] 收到请求")

        # 调用服务处理
        reply = service.chat(
            platform=request.platform,
            user_id=request.user_id,
            user_name=request.user_name,
            message=request.message,
        )

        return ChatResponse(
            success=True,
            reply=reply,
        )

    except Exception as e:
        logger.error(f"处理请求时出错: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            error=str(e),
        )


@app.post("/clear-history")
async def clear_history(platform: str, user_id: str):
    """清空用户的对话历史"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        service.clear_user_history(platform, user_id)
        return {
            "success": True,
            "message": f"已清空 {platform}/{user_id} 的对话历史",
        }
    except Exception as e:
        logger.error(f"清空历史失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
