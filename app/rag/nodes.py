import json
import re

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages.tool import ToolCall
from langgraph.graph import MessagesState

from app.core.llm import get_llm
from app.rag.tools import get_all_tools
from app.utils.logging import logger

llm = get_llm()


async def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools(get_all_tools())

    system_prompt = (
        "Anda adalah asisten AI untuk Universitas, hanya berikan jawaban yang sesuai dengan konteks Universitas. "
        "Jawablah semua pertanyaan dalam Bahasa Indonesia. "
        "PENTING: Gunakan alat pencarian HANYA jika pertanyaan tentang informasi faktual universitas "
        "seperti skripsi, jadwal kuliah, fakultas, atau data akademik yang memerlukan retrieval. "
        "JANGAN gunakan alat pencarian untuk: "
        "- Pertanyaan tentang identitas Anda (siapa kamu, apa kamu, dll) "
        "- Sapaan umum (halo, hai, terima kasih, dll) "
        "- Pertanyaan tentang kemampuan Anda "
        "Jika Anda tidak tahu jawabannya, katakan bahwa Anda tidak tahu. "
        "Gunakan tiga kalimat maksimum dan biarkan jawabannya singkat. "
        "Jangan mention tentang nama fungsi atau apapun tentang sistem ini, kamu harus berbahasa manusia. "
        "Jika sumber tertulis dalam context, selalu tulis sumber di akhir."
    )

    # Add system prompt to the beginning of the messages
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    logger.info(f"Generating response or tool for prompt: {messages[-1].content}")
    response = await llm_with_tools.ainvoke(messages)

    log_message = "Send direct response without tool call"

    # Handle the case of string-based function calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        log_message = f"Generated tool call: {response.tool_calls[0].get('name')}('{response.tool_calls[0].get('args').get('query')}, {response.tool_calls[0].get('args').get('tags')}')"
    elif hasattr(response, "content") and isinstance(response.content, str):
        content = response.content
        function_pattern = r"<function=(\w+)({.*?})(?:</function>|></function>)"
        match = re.search(function_pattern, content)

        if match:
            func_name = match.group(1)
            func_args = match.group(2)

            try:
                args_dict = json.loads(func_args)
                tool_calls = [
                    ToolCall(
                        name=func_name,
                        args=args_dict,
                        id=f"call_{func_name}_{hash(str(args_dict)) % 10000}",
                    )
                ]
                log_message = f"Generated tool call: {func_name}('{args_dict}')"
                return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
            except json.JSONDecodeError:
                logger.warning("Failed to parse function call arguments")
                pass

    logger.info(log_message)

    return {"messages": [response]}


async def generate(state: MessagesState):
    """Generate final answer using retrieved information."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    instruction_message_content = (
        f"Berikut hasil pencarianmu, Agen Universitas:\n{docs_content}"
    )

    prompt = conversation_messages + [
        SystemMessage(content=instruction_message_content)
    ]

    logger.info("Generating final response with retrieved information")
    response = await llm.ainvoke(prompt)
    logger.info(f"Final response: {response.content}")

    return {"messages": [response]}
