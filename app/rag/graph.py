from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from app.rag.nodes import query_or_respond, generate
from app.rag.tools import get_all_tools
from app.utils.logging import logger

def build_rag_graph():
    """Build and return the RAG graph."""
    logger.info("Building RAG graph")
    
    # Create the graph builder
    graph_builder = StateGraph(MessagesState)
    
    # Add nodes
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", ToolNode(get_all_tools()))
    graph_builder.add_node("generate", generate)
    
    # Set entry point
    graph_builder.set_entry_point("query_or_respond")
    
    # Add edges
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    
    # Compile the graph
    graph = graph_builder.compile()
    logger.info("RAG graph built successfully")
    
    return graph

# Create the graph singleton
rag_graph = build_rag_graph()