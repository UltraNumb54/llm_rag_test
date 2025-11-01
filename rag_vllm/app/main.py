from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from loguru import logger
import psutil
import torch

from app.config import settings
from app.api.endpoints import router as api_router
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore

embedding_service = None
llm_service = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, llm_service, vector_store
    
    logger.info("Инициализация сервисов...")
    
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    vector_store = VectorStore()
    
    # Проверка доступности GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        logger.info(f"Память GPU: {gpu_memory:.1f} GB")
    else:
        logger.warning("GPU не доступен, используется CPU")
    
    yield
    
    logger.info("Остановка сервисов...")
    if embedding_service:
        embedding_service.cleanup()

app = FastAPI(
    title="RAG System",
    description="RAG test",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if settings.API_KEY and api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API ключ")
    return api_key

app.include_router(api_router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])

@app.get("/")
async def root():
    return {
        "message": "Production RAG System", 
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    system_info = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        system_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
        system_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
    
    return {
        "status": "healthy",
        "system": system_info,
        "services": {
            "embedding": embedding_service is not None,
            "llm": llm_service is not None,
            "vector_store": vector_store is not None
        }
    }

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import logging
from app.config import settings
from app.api.endpoints import router as api_router
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore
from app.utils.file_handlers import ensure_directory_exists

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные сервисы
embedding_service = None
llm_service = None
vector_store = None

app = FastAPI(
    title="RAG Production System",
    description="Продакшен-готовая RAG система",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global embedding_service, vector_store
    
    logger.info("Запуск RAG системы...")
    
    # Создание необходимых директорий
    ensure_directory_exists(settings.CHROMA_PATH)
    ensure_directory_exists("data/raw")
    ensure_directory_exists("data/processed")
    
    try:
        # Инициализация сервисов
        embedding_service = EmbeddingService()
        vector_store = VectorStore()
        
        # LLM сервис будет инициализирован через API /configure
        
        logger.info("RAG система успешно запущена")
        logger.info(f"Документов в базе: {vector_store.count()}")
        
    except Exception as e:
        logger.error(f"Ошибка запуска системы: {e}")
        raise

# Подключение API роутов
app.include_router(api_router, prefix="/api/v1")

# Обслуживание статических файлов (веб-интерфейс)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_interface():
    """Главная страница с веб-интерфейсом"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "RAG System API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Добавьте ЭТО в main.py после создания app
app = FastAPI(...)

# ↓↓↓ ВСЕГО 3 СТРОКИ ДОБАВЛЕНИЯ ↓↓↓
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def serve_interface():
    return FileResponse("app/static/index.html")
# ↑↑↑ ВСЕГО 3 СТРОКИ ДОБАВЛЕНИЯ ↑↑↑

# Все остальное остается как было
@app.on_event("startup")
async def startup_event():
    # ваша существующая логика
    pass

# ваши существующие эндпоинты
@app.post("/api/v1/ask")
async def ask_question(...):
    # существующий код
    pass
