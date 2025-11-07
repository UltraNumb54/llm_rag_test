# rag_vllm_app/app/utils/logger.py:
import logging
import os
import sys

from loguru import logger


def setup_logger():
    os.makedirs("logs", exist_ok=True)

    logger.remove()

    _ = logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    _ = logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        compression="zip",
    )
