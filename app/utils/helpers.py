import hashlib
import time
import uuid

from fastapi import HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def convert_to_langgraph_messages(messages):
    """Convert OpenAI format messages to LangGraph format."""
    result = []
    for msg in messages:
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            result.append(SystemMessage(content=msg.content))
    return result


def estimate_tokens(text):
    """Estimate token count from text (rough approximation)."""
    if not text:
        return 0
    # Rough estimation: average English word is ~1.3 tokens
    return int(len(text.split()) * 1.3)


def create_openai_response(content, model, prompt_tokens=0, completion_tokens=0):
    """Create a response in OpenAI API format."""
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def format_sse_chunk(model, content=None, role=None, finish_reason=None):
    """Format a chunk for SSE streaming in OpenAI format."""
    delta = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content

    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def validate_file_type(file: UploadFile):
    """Validate the file type, only accept pdf, txt, csv, excel, pptx, docx, doc, image"""

    if file.content_type not in [
        "application/pdf",
        "text/plain",
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file extension, only accept pdf, txt, csv, excel, pptx, docx, doc, image",
        )
    return True


def get_hash_from_bytes(content: bytes):
    """Get the hash of the bytes."""
    return hashlib.sha256(content).hexdigest()
