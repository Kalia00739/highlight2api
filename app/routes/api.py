import json
import time

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from identifier import get_identifier
from ..auth import get_user_info_from_token, get_access_token, get_highlight_headers
from ..chat_service import stream_generator, non_stream_response
from ..file_service import messages_image_upload
from ..model_service import get_models
from ..models import ChatCompletionRequest, ModelsResponse, Model
from ..utils import format_messages_to_prompt, format_openai_tools

router = APIRouter()
security = HTTPBearer()


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """返回可用模型列表"""
    user_info = await get_user_info_from_token(credentials)

    rt = user_info["rt"]
    access_token = await get_access_token(rt)
    models = await get_models(access_token)

    # 构造返回数据
    model_list = []
    for model_name, model_info in models.items():
        model_list.append(
            Model(
                id=model_name,  # 使用model name作为对外的id
                object="model",
                created=int(time.time()),
                owned_by=model_info["provider"],
            )
        )

    return ModelsResponse(object="list", data=model_list)


@router.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """处理聊天完成请求"""
    user_info = await get_user_info_from_token(credentials)

    required_fields = ["rt", "user_id", "client_uuid"]
    if not all(field in user_info for field in required_fields):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization token - missing required fields",
        )

    rt = user_info["rt"]
    user_id = user_info["user_id"]
    client_uuid = user_info["client_uuid"]

    # 获取access token
    access_token = await get_access_token(rt)

    # 获取模型信息
    models = await get_models(access_token)
    model_info = models.get(request.model)
    if not model_info:
        raise HTTPException(
            status_code=400, detail=f"Model '{request.model}' not found"
        )

    model_id = model_info["id"]
    # 将 OpenAI 格式的消息转换为单个提示
    prompt = format_messages_to_prompt(request.messages)

    # 处理tool
    tools = format_openai_tools(request.tools)

    # 处理图片
    images = await messages_image_upload(request.messages, access_token)
    attached_context = [
        {
            'type': 'image',
            'fileId': image['fileId'],
            'fileName': image['fileName']
        } for image in images
    ]

    # 获取identifier
    identifier = get_identifier(user_id, client_uuid)

    # 准备 Highlight 请求
    highlight_data = {
        "prompt": prompt,
        "attachedContext": attached_context,
        "modelId": model_id,
        "additionalTools": tools,
        "backendPlugins": [],
        "useMemory": False,
        "useKnowledge": False,
        "ephemeral": True,
        "timezone": "Asia/Hong_Kong",
    }
    # logger.debug(json.dumps(highlight_data,ensure_ascii=False))

    if request.stream:
        return EventSourceResponse(
            stream_generator(highlight_data, access_token, identifier, request.model, rt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await non_stream_response(highlight_data, access_token, identifier, request.model, rt)


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": int(time.time())}
