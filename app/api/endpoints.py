import json
import time
import uuid

import aiohttp
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import verify_api_key
from app.api.models import ChatCompletionRequest
from app.rag.graph import rag_graph
from app.utils.helpers import (
    convert_to_langgraph_messages,
    create_openai_response,
    estimate_tokens,
    format_sse_chunk,
)
from app.utils.logging import logger

router = APIRouter()


@router.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models by forwarding request to DeepInfra API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.deepinfra.com/v1/openai/models"
            ) as response:
                return await response.json()
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Process chat completions in OpenAI format."""
    logger.info(f"Received chat request for model: {request.model}")
    # Handle streaming responses
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request), media_type="text/event-stream"
        )

    # For non-streaming responses
    try:
        # Convert messages to LangGraph format
        input_messages = convert_to_langgraph_messages(request.messages)

        # Generate conversation ID to track history
        conversation_id = str(uuid.uuid4())

        # Invoke the graph
        result = await rag_graph.ainvoke({"messages": input_messages})

        # Extract the final assistant message
        final_message = result["messages"][-1]
        content = final_message.content if hasattr(final_message, "content") else ""

        # Estimate token usage (rough estimation)
        prompt_tokens = sum(estimate_tokens(msg.content) for msg in request.messages)
        completion_tokens = estimate_tokens(content)

        # Format response like OpenAI
        response = create_openai_response(
            content=content,
            model=request.model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_response(request: ChatCompletionRequest):
    """Stream chat response in OpenAI SSE format."""
    try:
        # Convert messages to LangGraph format
        input_messages = convert_to_langgraph_messages(request.messages)

        # Send the first chunk with role
        first_chunk = format_sse_chunk(model=request.model, role="assistant")
        yield f"data: {json.dumps(first_chunk)}\n\n"

        # Stream the content
        async for message, metadata in rag_graph.astream(
            {"messages": input_messages},
            stream_mode="messages",
        ):
            if (
                hasattr(message, "content")
                and message.content
                and message.type != "tool"
            ):
                chunk = format_sse_chunk(model=request.model, content=message.content)
                yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk
        final_chunk = format_sse_chunk(model=request.model, finish_reason="stop")
        yield f"data: {json.dumps(final_chunk)}\n\n"

        # End the stream
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}", exc_info=True)
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}
