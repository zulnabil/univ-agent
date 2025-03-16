from app.core.vector_store import get_vector_store
from app.utils.logging import logger


def retrieve_university_data(query: str, tags: list[str]):
    """
    Retrieve university data from given query and tags.
    Tags can be one or more of the following, if not fit any of them, it will be ignored:
    - student_thesis
    - schedules

    Example queries:
    - "classes on march 5" -> "What classes are on March 5?"
    - "schedule for monday" -> "Show me the schedule for Monday."
    - "math class" -> "When is the math class?"
    - "john doe thesis" -> "What is John Doe's thesis about?"
    - "alice thesis" -> "Give me the thesis details for Alice."
    - "thesis blockchain" -> "Is there a thesis with the title 'Blockchain'?"
    """
    logger.info(f"Retrieving university data for query: {query} and tags: {tags}")

    vector_store = get_vector_store()

    # Set up search parameters
    search_kwargs = {
        "k": 5,
        "ranker_type": "weighted",
        "ranker_params": {"weights": [0.5, 0.5]},
    }

    # Only add filter expression if tags are provided
    if tags and len(tags) > 0:
        search_kwargs["expr"] = f"tag in {tags}"

    retrieved_docs = vector_store.similarity_search(query, **search_kwargs)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def get_all_tools():
    """Return all available tools."""
    return [retrieve_university_data]
