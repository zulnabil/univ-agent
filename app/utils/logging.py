import logging

import colorlog

from app.config import settings


def setup_logging():
    logging_level = getattr(logging, settings.LOG_LEVEL.upper())

    # Create color formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    # Create console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create logger for our app
    logger = colorlog.getLogger("university-rag")
    logger.setLevel(logging_level)

    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []

    # Add the console handler to logger
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
