# rag_vllm_app/app/routers/health.py
import os

import psutil
from app.core.database import get_chroma_client
from app.services.llm_service import LMStudioService
from fastapi import APIRouter, Depends
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    """Проверка здоровья всех компонентов системы"""
    checks = {}

    # Проверка ChromaDB
    try:
        client = get_chroma_client()
        client.heartbeat()
        checks["chromadb"] = {"status": "healthy", "message": "Connected"}
    except Exception as e:
        checks["chromadb"] = {"status": "unhealthy", "message": str(e)}

    # Проверка LMStudio
    try:
        llm_service = LMStudioService()
        test_response = llm_service.generate("Ответь 'OK' одним словом", max_tokens=5)
        checks["lmstudio"] = {
            "status": "healthy",
            "message": "Connected",
            "test_response": test_response[:50],
        }
    except Exception as e:
        checks["lmstudio"] = {"status": "unhealthy", "message": str(e)}

    # Проверка памяти
    memory = psutil.virtual_memory()
    checks["memory"] = {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_percent": memory.percent,
    }

    # Проверка диска
    disk = psutil.disk_usage("/")
    checks["disk"] = {
        "total_gb": round(disk.total / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "used_percent": disk.percent,
    }

    # Общий статус
    critical_services_healthy = all(
        checks[service]["status"] == "healthy" for service in ["chromadb", "lmstudio"]
    )

    overall_status = "healthy" if critical_services_healthy else "degraded"

    return {"status": overall_status, "version": "1.0.0", "checks": checks}
