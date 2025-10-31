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
