import sys
import os
from rag_vllm_app.app.services.embedding import embedding

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from rag_vllm_app.app.config import settings

    print("app.config загружен")

    from app.core.database import get_chroma_client

    print("database загружен")

    print("embedding загружен")

    print(f"Настройки:")
    print(f"- LMStudio: {settings.LMSTUDIO_BASE_URL}")
    print(f"- Модель: {settings.LMSTUDIO_MODEL_NAME}")

    # Простой тест ChromaDB
    client = get_chroma_client()
    print("ChromaDB подключена")

    print("Все базовые компоненты работают!")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback

    traceback.print_exc()
