from langchain_deepinfra import ChatDeepInfra
from app.config import settings
from app.utils.logging import logger

def get_llm():
    """Initialize and return the LLM."""
    logger.info(f"Initializing LLM: {settings.LLM_MODEL}")
    
    llm = ChatDeepInfra(model=settings.LLM_MODEL)
    llm.model_kwargs = {
        "temperature": 0,
        "top_p": 0.1,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "max_tokens": 512,
    }
    
    return llm