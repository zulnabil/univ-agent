from typing import Any, Dict, List, Literal, Optional

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    messages: List[Message] = Field(
        ..., description="List of messages in the conversation"
    )
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


class UploadDocumentRequest(BaseModel):
    file: UploadFile = File(...)


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        ..., description="Number of tokens in the completion"
    )
    total_tokens: int = Field(..., description="Total number of tokens used")


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = Field(
        None, description="Reason for finishing the completion"
    )


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(
        ..., description="Unix timestamp of when the completion was created"
    )
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")


class DocumentUploadResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    status: Literal["success", "error"] = Field(..., description="Processing status")
    error: Optional[str] = Field(None, description="Error message if processing failed")


class BulkUploadResponse(BaseModel):
    status: str = Field(..., description="Overall status of the upload")
    total_files: int = Field(..., description="Total number of files processed")
    results: List[DocumentUploadResponse] = Field(
        ..., description="Results for each file"
    )


class HealthCheckResponse(BaseModel):
    status: Literal["ok"] = Field(..., description="API health status")
    timestamp: float = Field(..., description="Current timestamp")
