from langchain_deepinfra import ChatDeepInfra

from app.config import settings
from app.utils.logging import logger


def get_llm():
    """Initialize and return the LLM."""
    logger.info(f"Initializing LLM: {settings.LLM_MODEL}")

    llm = ChatDeepInfra(model=settings.LLM_MODEL)
    llm.model_kwargs = {
        "temperature": settings.LLM_TEMPERATURE,
        "top_p": settings.LLM_TOP_P,
        "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
        "presence_penalty": settings.LLM_PRESENCE_PENALTY,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }

    return llm
