from langchain_core.tools import tool
from app.core.vector_store import get_vector_store
from app.utils.logging import logger

vector_store = get_vector_store()

@tool(response_format="content_and_artifact")
def get_student_thesis(query: str):
    """
    Retrieve the thesis title and details for a given student.

    Use this tool when the user asks about a student's thesis.

    Example queries:
    - "What is John Doe's thesis about?"
    - "Give me the thesis details for Alice."
    - "Is there a thesis with the title 'Blockchain'?"
    """
    logger.info(f"Retrieving student thesis information for query: {query}")
    
    retrieved_docs = vector_store.as_retriever(
        search_kwargs={"expr": "document_type == 'student_thesis'"}
    ).invoke(
        query,
        k=5,
        ranker_type="weighted",
        ranker_params={"weights": [0.5, 0.5]}
    )
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source')}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def get_schedules(query: str):
    """
    Retrieve class schedules for a given date.

    Use this tool when the user asks about class schedules.

    Example queries:
    - "What classes are on March 5?"
    - "Show me the schedule for Monday."
    - "When is the math class?"
    """
    logger.info(f"Retrieving schedule information for query: {query}")
    
    retrieved_docs = vector_store.as_retriever(
        search_kwargs={"expr": "document_type == 'schedules'"}
    ).invoke(
        query,
        k=3,
        ranker_type="weighted",
        ranker_params={"weights": [0.5, 0.5]}
    )
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source')}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def get_other_info(query: str):
    """
    Use this tool only when you need additional info and user asks about other university information.
    Retrieve university information from given query
    """
    logger.info(f"Retrieving general university information for query: {query}")
    
    retrieved_docs = vector_store.similarity_search(
        query,
        k=5,
        ranker_type="weighted",
        ranker_params={"weights": [0.5, 0.5]}
    )
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source')}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs

def get_all_tools():
    """Return all available tools."""
    return [get_student_thesis, get_schedules, get_other_info]