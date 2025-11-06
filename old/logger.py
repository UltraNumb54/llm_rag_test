# rag_vllm_app/app/utils/logger.py:
import logging
import sys
from loguru import logger
import os


def setup_logger():
    # Создаем директорию для логов
    os.makedirs("logs", exist_ok=True)

    # Удаляем стандартный обработчик
    logger.remove()

    # Добавляем обработчик для stdout
    _ = logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Добавляем обработчик для файла
    _ = logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        compression="zip",
    )
