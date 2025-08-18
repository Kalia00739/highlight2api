import json
import time
import uuid
from typing import Dict, Any, AsyncGenerator, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from .auth import get_access_token, get_highlight_headers
from .config import HIGHLIGHT_BASE_URL, TLS_VERIFY
from .models import ChatCompletionResponse, Choice, Usage


async def parse_sse_line(line: str) -> Optional[str]:
    """解析SSE数据行"""
    line = line.strip()
    if line.startswith("data: "):
        return line[6:]  # 去掉 'data: ' 前缀
    return None


async def stream_generator(
        highlight_data: Dict[str, Any], access_token: str, identifier: str, model: str, rt: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """生成流式响应"""
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created = int(time.time())

    try:
        for i in range(2):
            # 使用httpx的流式请求
            headers = get_highlight_headers(access_token, identifier)
            timeout = httpx.Timeout(60.0, connect=10.0)
            tool_call_idx = 0
            async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=timeout) as client:
                async with client.stream(
                        "POST",
                        HIGHLIGHT_BASE_URL + "/api/v1/chat",
                        headers=headers,
                        json=highlight_data,
                ) as response:
                    if response.status_code == 401 and i == 0:
                        access_token = await get_access_token(rt, True)
                        continue
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
                    yield {"event": "data", "data": json.dumps(initial_chunk)}

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
                                # logger.debug(line)

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
                                                                        "index": tool_call_idx,
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
                                                tool_call_idx += 1
                                                # logger.debug(
                                                #     json.dumps({"data": json.dumps(chunk_data)}, ensure_ascii=False))
                                                yield {"data": json.dumps(chunk_data)}
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
                    return

    except httpx.HTTPError as e:
        error_data = {
            "error": {"message": f"HTTP error: {str(e)}", "type": "http_error"}
        }
        yield {"event": "error", "data": json.dumps(error_data)}
    except Exception as e:
        error_data = {
            "error": {"message": str(e), "type": "server_error"}
        }
        yield {"event": "error", "data": json.dumps(error_data)}


async def non_stream_response(
        highlight_data: Dict[str, Any], access_token: str, identifier: str, model: str, rt: str
) -> JSONResponse:  # type: ignore
    """处理非流式响应"""
    try:
        for i in range(2):
            headers = get_highlight_headers(access_token, identifier)
            timeout = httpx.Timeout(60.0, connect=10.0)
            async with httpx.AsyncClient(verify=TLS_VERIFY, timeout=timeout) as client:
                async with client.stream(
                        "POST",
                        HIGHLIGHT_BASE_URL + "/api/v1/chat",
                        headers=headers,
                        json=highlight_data,
                ) as response:
                    if response.status_code == 401 and i == 0:
                        access_token = await get_access_token(rt, True)
                        continue

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
                    tool_calls = []
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
                                        elif event_data.get("type") == "toolUse":
                                            tool_name = event_data.get("name", "")
                                            tool_id = event_data.get("toolId", "")
                                            tool_input = event_data.get("input", "")
                                            if tool_name:
                                                tool_calls.append({
                                                    "id": tool_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_name,
                                                        "arguments": tool_input,
                                                    }
                                                })
                                    except json.JSONDecodeError:
                                        continue

                    # 创建 OpenAI 格式的响应
                    response_id = f"chatcmpl-{str(uuid.uuid4())}"
                    created = int(time.time())

                    # 构建消息内容
                    message_content: Dict[str, any] = {"role": "assistant"}
                    if full_response:
                        message_content["content"] = full_response
                    if tool_calls:
                        message_content["tool_calls"] = tool_calls

                    response_data = ChatCompletionResponse(
                        id=response_id,
                        object="chat.completion",
                        created=created,
                        model=model,
                        choices=[
                            Choice(
                                index=0,
                                message=message_content,
                                finish_reason="stop",
                            )
                        ],
                        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    )
                    return JSONResponse(content=response_data.model_dump())


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
