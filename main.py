import asyncio
import base64
import hashlib
import json
import time
import uuid
from typing import List, Dict, Any, Literal, Optional, AsyncGenerator, Tuple, Union

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from filetype import filetype
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# 导入identifier模块
from identifier import get_identifier

TLS_VERIFY = True
app = FastAPI(title="Highlight AI API Proxy", version="1.0.0")

# 认证令牌
HIGHLIGHT_BASE_URL = "https://chat-backend.highlightai.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Highlight/1.3.61 Chrome/132.0.6834.210 Electron/34.5.8 Safari/537.36"

# 存储格式：{rt: {"access_token": str, "expires_at": int}}
access_tokens: Dict[str, Dict[str, Any]] = {}
# 模型缓存，格式：{model_name: {"id": str, "name": str, "provider": str, "isFree": bool}}
model_cache: Dict[str, Dict[str, Any]] = {}
# 缓存文件上传信息，结构: { sha256: {"fileName": str, "fileId": str} }
file_upload_cache: Dict[str, Dict[str, str]] = {}
# 设置Bearer token认证
security = HTTPBearer()


class OpenAIToolCallFunction(BaseModel):
    """工具调用函数"""

    name: str | None = Field(None, description="函数名称")
    arguments: str | None = Field(None, description="JSON格式的函数参数")


class OpenAIDeltaToolCall(BaseModel):
    index: int | None = Field(None, description="工具调用索引")
    id: str | None = Field(None, description="工具调用ID")
    type: Literal["function"] | None = Field(None, description="调用类型")
    function: OpenAIToolCallFunction | None = Field(None, description="函数详情增量")


class OpenAIMessageContent(BaseModel):
    """OpenAI消息内容项"""

    type: Literal["text", "image_url"] = Field(description="内容类型")
    text: str | None = Field(None, description="文本内容")
    image_url: dict[str, str] | None = Field(None, description="图像URL配置")


# Pydantic 模型定义
class Message(BaseModel):
    role: str
    content: str | list[OpenAIMessageContent] | None = Field(
        None, description="消息内容"
    )
    tool_call_id: str | None = Field(None)
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="工具调用信息（当role为assistant时）"
    )


class OpenAIToolFunction(BaseModel):
    """OpenAI工具函数定义"""

    name: str = Field(description="函数名称")
    description: str | None = Field(None, description="函数描述")
    parameters: dict[str, Any] | None = Field(
        None, description="JSON Schema格式的函数参数"
    )


class OpenAITool(BaseModel):
    """OpenAI工具定义"""

    type: Literal["function"] = Field("function", description="工具类型")
    function: OpenAIToolFunction = Field(description="函数定义")


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    model: Optional[str] = "gpt-4o"
    tools: list[OpenAITool] | None = Field(None, description="可用工具定义")


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str
    data: List[Model]


class Choice(BaseModel):
    index: int
    message: Optional[Dict[str, str]] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


def parse_api_key(api_key_base64: str) -> Optional[Dict[str, Any]]:
    """解析base64编码的JSON API Key"""
    try:
        decoded_bytes = base64.b64decode(api_key_base64)
        data = json.loads(decoded_bytes)
        return data
    except Exception:
        return None


def parse_jwt_payload(jwt_token: str) -> Optional[Dict[str, Any]]:
    """解析JWT token的payload部分"""
    try:
        # JWT格式：header.payload.signature
        parts = jwt_token.split(".")
        if len(parts) != 3:
            return None

        # 解析payload部分（第二部分）
        payload = parts[1]
        # 补齐base64编码所需的padding
        padding = len(payload) % 4
        if padding:
            payload += "=" * (4 - padding)

        decoded_bytes = base64.urlsafe_b64decode(payload)
        payload_data = json.loads(decoded_bytes)
        return payload_data
    except Exception:
        return None


async def refresh_access_token(rt: str) -> str:
    """刷新access token"""
    async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=30.0) as client:
        try:
            response = await client.post(
                f"{HIGHLIGHT_BASE_URL}/api/v1/auth/refresh",
                json={"refreshToken": rt},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="无法刷新access token")

            resp_json = response.json()
            if not resp_json.get("success"):
                raise HTTPException(status_code=401, detail="刷新access token失败")

            new_access_token = resp_json["data"]["accessToken"]

            # 解析JWT获取过期时间
            payload = parse_jwt_payload(new_access_token)
            expires_at = (
                payload.get("exp", int(time.time()) + 3600)
                if payload
                else int(time.time()) + 3600
            )

            # 更新缓存
            access_tokens[rt] = {
                "access_token": new_access_token,
                "expires_at": expires_at,
            }

            return new_access_token

        except httpx.HTTPError as e:
            raise HTTPException(status_code=401, detail=f"刷新token失败: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"刷新token失败: {str(e)}")


async def get_access_token(rt: str) -> str:
    """获取有效的access token"""
    token_info = access_tokens.get(rt)
    current_time = int(time.time())

    # 检查token是否存在且未过期（提前60秒刷新）
    if token_info and token_info["expires_at"] > current_time + 60:
        return token_info["access_token"]

    # token不存在或即将过期，需要刷新
    return await refresh_access_token(rt)


async def download_image(url: str) -> bytes:
    """下载图片数据（bytes）"""
    async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def is_base64_image(data: str) -> Tuple[bool, Union[bytes, None]]:
    """判断是否为base64图片字符串，是则返回(真，解码后的bytes)"""
    if data.startswith("data:image/"):
        try:
            header, base64_data = data.split(",", 1)
            decoded = base64.b64decode(base64_data)
            return True, decoded
        except Exception:
            return False, None
    return False, None


def detect_image_type_and_extension(image_bytes: bytes) -> tuple[str, str]:
    kind = filetype.guess(image_bytes)
    if kind is None or not kind.mime.startswith("image/"):
        raise ValueError("无法识别或不是图片格式")
    return kind.mime, kind.extension


async def prepare_file_upload(
        access_token: str, file_name: str, mime_type: str, file_size: int
) -> Dict[str, Any]:
    """调用文件准备接口，申请上传链接"""
    url = f"{HIGHLIGHT_BASE_URL}/api/v1/files/prepare"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    json_data = {"name": file_name, "type": mime_type, "size": file_size}
    async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=json_data)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success") or "data" not in data:
            raise ValueError("文件准备接口返回失败")

        logger.debug(f'{file_size}{data}')
        return data["data"]


async def upload_file_to_url(upload_url: str, file_bytes: bytes, access_token: str) -> None:
    """PUT 请求上传文件二进制数据"""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream",
        "User-Agent": USER_AGENT,
    }
    async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=60.0) as client:
        resp = await client.put(upload_url, content=file_bytes, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise ValueError(f"上传文件失败 {resp.text}")


async def upload_single_image(
        access_token: str, image_data: str
) -> Dict[str, str]:
    """
    上传单张图片，支持base64和URL。
    返回: {"fileName": "...", "fileId": "..."}
    """
    # 先判断是否base64图片
    is_base64, image_bytes = is_base64_image(image_data)
    if not is_base64:
        # 否则按URL下载
        try:
            image_bytes = await download_image(image_data)
        except Exception as e:
            logger.error(f"下载图片失败：{e}")
            raise
    # 计算文件哈希用作缓存key
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    if sha256 in file_upload_cache:
        # 缓存命中，直接返回
        return file_upload_cache[sha256]
    # 探测图片类型及扩展名
    try:
        mime_type, ext = detect_image_type_and_extension(image_bytes)
    except Exception as e:
        logger.error(f"解析图片类型失败：{e}")
        raise HTTPException(status_code=400, detail=f"图片格式不支持：{str(e)}")
    file_name = f"image.{ext}"
    file_size = len(image_bytes)
    # 准备上传
    upload_info = await prepare_file_upload(access_token, file_name, mime_type, file_size)
    # 上传文件内容
    await upload_file_to_url(upload_info["uploadUrl"], image_bytes, access_token)
    result = {"fileName": file_name, "fileId": upload_info["id"]}
    # 缓存结果
    file_upload_cache[sha256] = result
    return result


async def messages_image_upload(
        messages: List[Message], access_token: str
) -> List[Dict[str, str]]:
    """
    遍历消息，上传所有图片，返回文件名和文件ID列表
    每个元素示例：{"fileName": "...", "fileId": "..."}
    """
    results = []
    # 收集所有图片url/base64
    images: List[str] = []
    for message in messages:
        if message.content and isinstance(message.content, list):
            for content_item in message.content:
                if content_item.type == "image_url" and content_item.image_url:
                    url = content_item.image_url.get("url")
                    if url:
                        images.append(url)
    # 并行上传所有图片
    # 为节省资源，可以用信号量限制并发数，比如最多5个并发
    semaphore = asyncio.Semaphore(5)

    async def upload_wrapper(_url: str):
        async with semaphore:
            try:
                return await upload_single_image(access_token, _url)
            except Exception as e:
                logger.exception(f"上传图片失败: {_url}", e)
                return None

    upload_tasks = [upload_wrapper(url) for url in images]
    upload_results = await asyncio.gather(*upload_tasks)
    for r in upload_results:
        if r and r["fileId"]:
            results.append(r)
    results.reverse()
    return results


async def fetch_models_from_upstream(access_token: str) -> Dict[str, Dict[str, Any]]:
    """从上游获取模型列表"""
    async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=30.0) as client:
        try:
            response = await client.get(
                f"{HIGHLIGHT_BASE_URL}/api/v1/models",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": USER_AGENT,
                },
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="获取模型列表失败")

            resp_json = response.json()
            if not resp_json.get("success"):
                raise HTTPException(status_code=500, detail="获取模型数据失败")

            # 清空并重新填充缓存
            model_cache.clear()
            for model in resp_json["data"]:
                model_name = model["name"]
                model_cache[model_name] = {
                    "id": model["id"],
                    "name": model["name"],
                    "provider": model["provider"],
                    "isFree": model.get("pricing", {}).get("isFree", False),
                }

            return model_cache

        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


async def get_models(access_token: str) -> Dict[str, Dict[str, Any]]:
    """获取模型列表（带缓存）"""
    if not model_cache:
        # 缓存为空，从上游获取
        return await fetch_models_from_upstream(access_token)
    return model_cache


# API 请求头
def get_highlight_headers(
        access_token: str, identifier: Optional[str] = None
) -> Dict[str, str]:
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN",
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "user-agent": USER_AGENT,
        "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }

    if identifier:
        headers["identifier"] = identifier

    return headers


async def get_user_info_from_token(
        credentials: HTTPAuthorizationCredentials,
) -> Dict[str, Any]:
    """从Bearer token中解析用户信息"""
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    # 解析API Key
    user_info = parse_api_key(credentials.credentials)
    if not user_info or "rt" not in user_info:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    return user_info


@app.get("/v1/models", response_model=ModelsResponse)
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


def format_messages_to_prompt(messages: List[Message]) -> str:
    """将 OpenAI 格式的消息列表转换为单个提示字符串"""
    formatted_messages = []
    for message in messages:
        if message.role:
            if message.content:
                if isinstance(message.content, list):
                    for item in message.content:
                        formatted_messages.append(f"{message.role}: {item.text}")
                else:
                    formatted_messages.append(f"{message.role}: {message.content}")
            if message.tool_calls:
                formatted_messages.append(
                    f"{message.role}: {json.dumps(message.tool_calls)}"
                )
            if message.tool_call_id:
                formatted_messages.append(
                    f"{message.role}: tool_call_id: {message.tool_call_id} {message.content}"
                )
    return "\n\n".join(formatted_messages)


def format_openai_tools(openai_tools: Optional[list[OpenAITool]]) -> List:
    tools = []
    if openai_tools is None:
        return tools

    for tool in openai_tools:
        tool = tool.function
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        )
    return tools


async def parse_sse_line(line: str) -> Optional[str]:
    """解析SSE数据行"""
    line = line.strip()
    if line.startswith("data: "):
        return line[6:]  # 去掉 'data: ' 前缀
    return None


async def stream_generator(
        highlight_data: Dict[str, Any], headers: Dict[str, str], model: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """生成SSE流数据的异步生成器"""
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created = int(time.time())

    try:
        # 使用httpx的流式请求
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=timeout) as client:
            async with client.stream(
                    "POST",
                    HIGHLIGHT_BASE_URL + "/api/v1/chat",
                    headers=headers,
                    json=highlight_data,
            ) as response:
                if response.status_code != 200:
                    error_content = await response.aread()
                    logger.error(error_content)
                    error_data = {
                        "error": {
                            "message": f"Highlight API returned status code {response.status_code}",
                            "type": "api_error",
                        }
                    }
                    yield {"event": "error", "data": json.dumps(error_data)}
                    return

                # 发送初始消息
                initial_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield {"data": json.dumps(initial_chunk)}

                # 处理流式响应
                buffer = ""
                async for chunk in response.aiter_bytes():
                    if chunk:
                        # 解码字节数据
                        try:
                            chunk_text = chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            chunk_text = chunk.decode("utf-8", errors="ignore")

                        buffer += chunk_text

                        # 按行处理数据
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)

                            # 解析SSE行
                            data = await parse_sse_line(line)
                            if data and data.strip():
                                try:
                                    event_data = json.loads(data)
                                    if event_data.get("type") == "text":
                                        content = event_data.get("content", "")
                                        if content:
                                            chunk_data = {
                                                "id": response_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {"content": content},
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                            yield {"data": json.dumps(chunk_data)}
                                    elif event_data.get("type") == "toolUse":
                                        tool_name = event_data.get("name", "")
                                        tool_id = event_data.get("toolId", "")
                                        tool_input = event_data.get("input", "")
                                        if tool_name:
                                            chunk_data = {
                                                "id": response_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {
                                                            "tool_calls": [
                                                                {
                                                                    "index": 0,
                                                                    "id": tool_id,
                                                                    "type": "function",
                                                                    "function": {
                                                                        "name": tool_name,
                                                                        "arguments": tool_input,
                                                                    },
                                                                }
                                                            ]
                                                        },
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                            yield {"data": json.dumps(chunk_data)}

                                        pass
                                except json.JSONDecodeError:
                                    # 忽略无效的JSON数据
                                    continue

                # 发送完成消息
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield {"data": json.dumps(final_chunk)}
                yield {"data": "[DONE]"}

    except httpx.HTTPError as e:
        logger.exception(f"HTTP error during streaming: {e}")
        error_data = {
            "error": {"message": f"HTTP error: {str(e)}", "type": "http_error"}
        }
        yield {"event": "error", "data": json.dumps(error_data)}
    except Exception as e:
        logger.exception(f"Unexpected error during streaming: {e}")
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield {"event": "error", "data": json.dumps(error_data)}


async def non_stream_response(
        highlight_data: Dict[str, Any], headers: Dict[str, str], model: str
) -> JSONResponse:
    """处理非流式响应"""
    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=timeout) as client:
            async with client.stream(
                    "POST",
                    HIGHLIGHT_BASE_URL + "/api/v1/chat",
                    headers=headers,
                    json=highlight_data,
            ) as response:

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail={
                            "error": {
                                "message": f"Highlight API returned status code {response.status_code}",
                                "type": "api_error",
                            }
                        },
                    )

                # 收集完整响应
                full_response = ""
                buffer = ""

                async for chunk in response.aiter_bytes():
                    if chunk:
                        # 解码字节数据
                        try:
                            chunk_text = chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            chunk_text = chunk.decode("utf-8", errors="ignore")

                        buffer += chunk_text

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            data = await parse_sse_line(line)
                            if data and data.strip():
                                try:
                                    event_data = json.loads(data)
                                    if event_data.get("type") == "text":
                                        full_response += event_data.get("content", "")
                                except json.JSONDecodeError:
                                    continue

                # 创建 OpenAI 格式的响应
                response_id = f"chatcmpl-{str(uuid.uuid4())}"
                created = int(time.time())
                response_data = ChatCompletionResponse(
                    id=response_id,
                    object="chat.completion",
                    created=created,
                    model=model,
                    choices=[
                        Choice(
                            index=0,
                            message={"role": "assistant", "content": full_response},
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )
                return JSONResponse(content=response_data.dict())

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {"message": f"HTTP error: {str(e)}", "type": "http_error"}
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "server_error"}},
        )


@app.post("/v1/chat/completions")
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
    logger.debug(attached_context)

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
        "ephemeral": False,
        "timezone": "Asia/Hong_Kong",
    }
    headers = get_highlight_headers(access_token, identifier)

    if request.stream:
        return EventSourceResponse(
            stream_generator(highlight_data, headers, request.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await non_stream_response(highlight_data, headers, request.model)


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",  # 假设文件名为 main.py
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
