import os
from contextlib import asynccontextmanager

import uvicorn
from app.config import settings
from app.core.database import get_chroma_client
from app.routers import chat, files, health
from app.utils.logger import setup_logger
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Инициализация логгера
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск RAG системы с LMStudio...")

    # Проверка ChromaDB
    try:
        client = get_chroma_client()
        client.heartbeat()
        logger.success("ChromaDB подключена успешно")
    except Exception as e:
        logger.error(f"Ошибка подключения к ChromaDB: {e}")

    # Проверка LMStudio
    from app.services.llm_service import LMStudioService

    try:
        llm_service = LMStudioService()
        logger.success("LMStudio сервис инициализирован")
    except Exception as e:
        logger.error(f"Ошибка подключения к LMStudio: {e}")

    yield

    logger.info("Приложение выключается")


app = FastAPI(
    title="RAG LMStudio API",
    description="API для RAG системы с LMStudio бэкендом",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router, prefix="/api/v1", tags=["files"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Обслуживает главную страницу"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG API"}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=True)
