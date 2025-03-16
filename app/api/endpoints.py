import time
from typing import List

import aiohttp
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.api.dependencies import verify_api_key
from app.api.models import (
    BulkUploadResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthCheckResponse,
)
from app.config import settings
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
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
            async with session.get(settings.DEEPINFRA_ENDPOINT_MODELS) as response:
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
    chat_service = ChatService()
    return await chat_service.chat(request)


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
    document_service = DocumentService()
    return await document_service.process_documents(files)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the API",
)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}
