import json

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.api.models import ChatCompletionRequest
from app.rag.graph import rag_graph
from app.utils.helpers import (
    convert_to_langgraph_messages,
    create_openai_response,
    estimate_tokens,
    format_sse_chunk,
)
from app.utils.logging import logger


class ChatService:
    def __init__(self):
        self.rag_graph = rag_graph

    async def chat(self, request: ChatCompletionRequest):
        """Process chat completions in OpenAI format."""
        if request.stream:
            return StreamingResponse(
                self._stream_chat_response(request), media_type="text/event-stream"
            )
        return await self._direct_chat_response(request)

    async def _stream_chat_response(self, request: ChatCompletionRequest):
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
                    chunk = format_sse_chunk(
                        model=request.model, content=message.content
                    )
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

    async def _direct_chat_response(self, request: ChatCompletionRequest):
        """Direct chat response."""
        try:
            # Convert messages to LangGraph format
            input_messages = convert_to_langgraph_messages(request.messages)

            # Invoke the graph
            result = await rag_graph.ainvoke({"messages": input_messages})

            # Extract the final assistant message
            final_message = result["messages"][-1]
            content = final_message.content if hasattr(final_message, "content") else ""

            # Estimate token usage (rough estimation)
            prompt_tokens = sum(
                estimate_tokens(msg.content) for msg in request.messages
            )
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
