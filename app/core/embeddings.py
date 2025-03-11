from langchain_deepinfra import DeepInfraEmbeddings
from app.config import settings
from app.utils.logging import logger

def get_embeddings():
    """Initialize and return the embedding model."""
    logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL}")
    embeddings = DeepInfraEmbeddings(
        model=settings.EMBEDDING_MODEL,
    )
    return embeddings