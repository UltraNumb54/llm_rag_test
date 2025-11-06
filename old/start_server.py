import uvicorn
import threading
import subprocess
import time
import os
import sys
from rag_vllm_app.app.config import settings


def setup_environment():
    """Настройка окружения"""
    # Создаем необходимые директории
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Добавляем текущую директорию в путь Python
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)


def start_openwebui():
    """Запуск Open WebUI в отдельном потоке"""
    try:
        print("Запуск Open WebUI...")
        # Проверяем установлен ли Open WebUI
        result = subprocess.run(["which", "open-webui"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Open WebUI не установлен. Установите: pip install open-webui")
            return

        # Запускаем Open WebUI
        env = os.environ.copy()
        subprocess.Popen(
            ["./start_openwebui.sh"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Open WebUI запущен на http://localhost:3000")
    except Exception as e:
        print(f"❌ Open WebUI не запущен: {e}")


if __name__ == "__main__":
    setup_environment()

    print("Запуск RAG системы с LMStudio...")
    print(f"Конфигурация:")
    print(f"- API сервер: http://{settings.HOST}:{settings.PORT}")
    print(f"- LMStudio: {settings.LMSTUDIO_BASE_URL}")
    print(f"- Модель: {settings.LMSTUDIO_MODEL_NAME}")
    print(f"- Embedding модель: {settings.EMBEDDING_MODEL}")
    print(f"- Документация: http://{settings.HOST}:{settings.PORT}/docs")

    # Запуск Open WebUI в фоне (опционально)
    try:
        webui_thread = threading.Thread(target=start_openwebui, daemon=True)
        webui_thread.start()
        print("Ожидание запуска Open WebUI...")
        time.sleep(5)
    except Exception as e:
        print(f"Не удалось запустить Open WebUI: {e}")

    # Запуск основного сервера
    print("Запуск основного API сервера...")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
        access_log=True,
    )
