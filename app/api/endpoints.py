import json
import time
import uuid
from typing import List

import aiohttp
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.dependencies import verify_api_key
from app.api.models import (
    BulkUploadResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthCheckResponse,
)
from app.core.vector_store import (
    add_documents_to_vector_store,
    get_vector_from_csv,
    get_vector_from_docx,
    get_vector_from_pdf,
    get_vector_from_txt,
)
from app.rag.graph import rag_graph
from app.utils.helpers import (
    convert_to_langgraph_messages,
    create_openai_response,
    estimate_tokens,
    format_sse_chunk,
    get_hash_from_bytes,
    validate_file_type,
)
from app.utils.logging import logger

router = APIRouter(tags=["AI Chat"])


@router.get(
    "/models",
    summary="List Available Models",
    description="List available language models by forwarding request to DeepInfra API.",
)
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


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create Chat Completion",
    description="Process chat completions in OpenAI format with optional streaming.",
)
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


@router.post(
    "/documents",
    response_model=BulkUploadResponse,
    summary="Upload Documents",
    description="Upload multiple documents for vectorization and storage in the vector store",
)
async def upload_document(
    files: List[UploadFile] = File(
        ...,
        description="List of files to upload. Supported formats: PDF, DOCX, TXT, CSV",
    ),
):
    """Upload multiple documents, vectorize them and store them in the vector store"""
    try:
        results = []
        for file in files:
            result = await process_single_file(file)
            results.append(result)

        return {"status": "completed", "total_files": len(files), "results": results}
    except Exception as e:
        logger.error(f"Error in bulk upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_single_file(file: UploadFile) -> dict:
    """Process a single file and return its processing result."""
    try:
        validate_file_type(file)
        logger.info(f"Processing file: {file.filename}")
        logger.info(file.content_type)
        content = await file.read()
        hash = get_hash_from_bytes(content)
        docs = await get_vector_by_content_type(content, file.content_type)
        logger.info(f"Extracted {len(docs)} documents from {file.filename}")
        await add_documents_to_vector_store(docs, hash)

        return {
            "filename": file.filename,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return {"filename": file.filename, "status": "error", "error": str(e)}


async def get_vector_by_content_type(content: bytes, content_type: str):
    """Get vector based on file content type."""
    content_type_handlers = {
        "application/pdf": get_vector_from_pdf,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": get_vector_from_docx,
        "text/plain": get_vector_from_txt,
        "text/csv": get_vector_from_csv,
    }

    handler = content_type_handlers.get(content_type)
    if not handler:
        raise ValueError(f"Unsupported content type: {content_type}")

    return await handler(content)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the API",
)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}
