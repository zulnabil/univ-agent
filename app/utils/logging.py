import logging
from app.config import settings

def setup_logging():
    logging_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Configure root logger
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create logger for our app
    logger = logging.getLogger("university-rag")
    logger.setLevel(logging_level)
    
    return logger

logger = setup_logging()