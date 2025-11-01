import uvicorn
import threading
import subprocess
import time
from app.config import settings

def start_openwebui():
    """Запуск Open WebUI в отдельном потоке"""
    try:
        print("Запуск Open WebUI...")
        # Используем быстрый скрипт
        subprocess.run(["./start_openwebui.sh"], check=True)
    except Exception as e:
        print(f"Open WebUI не запущен: {e}")

if __name__ == "__main__":
    print("Запуск RAG системы...")
    print(f"API: http://{settings.HOST}:{settings.PORT}")
    print(f"Документация: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"Веб-интерфейс: http://{settings.HOST}:{settings.PORT}")
    print(f"Open WebUI: http://localhost:3000")
    
    # Запуск Open WebUI в фоне
    webui_thread = threading.Thread(target=start_openwebui, daemon=True)
    webui_thread.start()
    
    # Даем время Open WebUI запуститься
    time.sleep(5)
    
    # Запуск основного сервера
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
