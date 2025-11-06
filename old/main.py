# rag_vllm_app/app/main.py:
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from loguru import logger

from rag_vllm_app.app.config import settings
from rag_vllm_app.app.core.database import get_chroma_client
from rag_vllm_app.app.routers import files, chat, health
from app.utils.logger import setup_logger

# Инициализация логгера
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Запуск RAG системы с LMStudio...")

    # Проверка ChromaDB
    try:
        client = get_chroma_client()
        client.heartbeat()
        logger.success("ChromaDB подключена успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к ChromaDB: {e}")

    # Проверка LMStudio
    from app.services.llm_service import LMStudioService

    try:
        llm_service = LMStudioService()
        logger.success("LMStudio сервис инициализирован")
    except Exception as e:
        logger.error(f"Ошибка подключения к LMStudio: {e}")

    yield

    # Shutdown
    logger.info("Приложение выключается")


app = FastAPI(
    title="RAG LMStudio API",
    description="API для RAG системы с LMStudio бэкендом",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(files.router, prefix="/api/v1", tags=["files"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# Статические файлы
if os.path.exists("app/static"):
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")


@app.get("/")
async def root():
    return {
        "message": "RAG LMStudio API работает",
        "status": "OK",
        "lmstudio_url": settings.LMSTUDIO_BASE_URL,
    }
