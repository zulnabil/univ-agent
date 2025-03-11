from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import json
import time
import uuid

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
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
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
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }