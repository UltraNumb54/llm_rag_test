import uvicorn
import os
import sys
from app.config import settings

def setup_environment():
    """Настройка окружения"""
    # Создаем необходимые директории
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

if __name__ == "__main__":
    setup_environment()
    
    print("Запуск RAG системы с LMStudio...")
    print("Конфигурация:")
    print(f"Веб-интерфейс: http://{settings.HOST}:{settings.PORT}")
    print(f"API: http://{settings.HOST}:{settings.PORT}/api/v1")
    print(f"LMStudio: {settings.LMSTUDIO_BASE_URL}")
    print(f"Документация: http://{settings.HOST}:{settings.PORT}/docs")
    print()
    print("Доступные endpoints:")
    print("GET  / - Веб-интерфейс")
    print("POST /api/v1/upload - Загрузка файлов")
    print("POST /api/v1/chat - Чат с RAG")
    print("GET  /api/v1/health - Статус системы")
    
    # Запуск основного сервера
    print("Запуск сервера...")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
        access_log=True
    )