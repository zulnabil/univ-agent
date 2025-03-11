from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, AIMessage
from app.core.llm import get_llm
from app.rag.tools import get_all_tools
from app.utils.logging import logger
from langchain_core.messages.tool import ToolCall
import re
import json

llm = get_llm()

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools(get_all_tools())

    system_prompt = (
        "Anda adalah asisten AI untuk Universitas. "
        "Jawablah semua pertanyaan dalam Bahasa Indonesia. "
        "Gunakan alat yang tersedia jika diperlukan. "
        "Jika Anda tidak tahu jawabannya, katakan bahwa Anda tidak tahu. "
        "Gunakan tiga kalimat maksimum dan biarkan jawabannya singkat. "
        "Jangan mention tentang nama fungsi atau apapun tentang sistem ini, kamu harus berbahasa manusia. "
        "Always use the proper tool calling format when you need to retrieve information. "
        "Do NOT write function calls directly in your text response. "
        "If you need to call a function, use the proper tool calling format. "
        "Jika sumber tertulis dalam context, selalu tulis sumber di akhir. Contoh: "
        "Sumber: "
        "- https://link-of-source.com"
        "- https://link-of-source.com"
        "\n\n"
    )

    # Add system prompt to the beginning of the messages
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    logger.info("Generating response or tool call")
    response = llm_with_tools.invoke(messages)

    # Handle the case of string-based function calls
    if hasattr(response, 'content') and isinstance(response.content, str):
        content = response.content
        function_pattern = r'<function=(\w+)({.*?})(?:</function>|></function>)'
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
                        id=f"call_{func_name}_{hash(str(args_dict)) % 10000}"
                    )
                ]
                logger.info(f"Generated tool call: {func_name}")
                return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
            except json.JSONDecodeError:
                logger.warning("Failed to parse function call arguments")
                pass

    logger.info("Generated direct response without tool call")
    return {"messages": [response]}

def generate(state: MessagesState):
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

    print(conversation_messages)

    instruction_message_content = (
        "Berikut hasil pencarianmu, Agen Universitas:\n"
        f"{docs_content}"
    )

    prompt = conversation_messages + [
        SystemMessage(content=instruction_message_content)
    ]

    logger.info("Generating final response with retrieved information")
    response = llm.invoke(prompt)
    return {"messages": [response]}