from langchain_milvus import BM25BuiltInFunction, Milvus
from app.config import settings
from app.utils.logging import logger
from app.core.embeddings import get_embeddings

def get_vector_store():
    """Initialize and return the vector store."""
    logger.info(f"Connecting to Milvus at {settings.MILVUS_URI}")
    
    embeddings = get_embeddings()
    
    vector_store = Milvus(
        embeddings,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        consistency_level="Strong",
        connection_args={
            'uri': settings.MILVUS_URI,
            'token': settings.MILVUS_TOKEN
        },
        collection_name=settings.MILVUS_COLLECTION,
        auto_id=True
    )
    
    return vector_store