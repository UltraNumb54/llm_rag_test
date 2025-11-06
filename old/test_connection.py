#!/usr/bin/env python3
"""
Тестовый скрипт для проверки подключения всех компонентов с LMStudio
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import get_chroma_client
from app.services.embedding import embedding_service
from app.services.llm_service import LMStudioService
from rag_vllm_app.app.config import settings
from loguru import logger


def test_chromadb():
    print("🔍 Тестирование ChromaDB...")
    try:
        client = get_chroma_client()
        heartbeat = client.heartbeat()
        print(f"✅ ChromaDB: heartbeat = {heartbeat}")

        # Проверяем коллекцию
        collection = client.get_collection(settings.COLLECTION_NAME)
        count = collection.count()
        print(f"✅ ChromaDB коллекция: {count} документов")
        return True
    except Exception as e:
        print(f"❌ ChromaDB ошибка: {e}")
        return False


def test_embeddings():
    print("🔍 Тестирование эмбеддингов...")
    try:
        test_text = "Тестовый текст для проверки эмбеддингов"
        embedding = embedding_service.encode(test_text)
        print(
            f"✅ Эмбеддинги: размер вектора = {len(embedding[0])}, количество = {len(embedding)}"
        )
        return True
    except Exception as e:
        print(f"❌ Эмбеддинги ошибка: {e}")
        return False


def test_lmstudio():
    print("🔍 Тестирование LMStudio...")
    try:
        llm_service = LMStudioService()
        test_prompt = "Ответь одним предложением: как дела?"
        response = llm_service.generate(test_prompt, max_tokens=50)
        print(f"✅ LMStudio: ответ получен - '{response}'")
        return True
    except Exception as e:
        print(f"❌ LMStudio ошибка: {e}")
        print("Убедитесь, что:")
        print("1. LMStudio запущен")
        print("2. Модель загружена в LMStudio")
        print("3. Сервер API включен на порту 1234")
        print(f"4. URL доступен: {settings.LMSTUDIO_BASE_URL}")
        return False


def test_rag_pipeline():
    print("🔍 Тестирование RAG пайплайна...")
    try:
        from app.services.vector_store import vector_store
        from app.services.llm_service import llm_service

        # Добавляем тестовый документ
        test_docs = [
            "Техническая поддержка работает с 9:00 до 18:00 по московскому времени."
        ]
        vector_store.add_documents(
            test_docs, [{"source": "test", "type": "working_hours"}]
        )

        # Ищем документы
        results = vector_store.search("Когда работает поддержка?")

        if results:
            context = [result["document"] for result in results]
            response = llm_service.generate_with_context(
                "Когда работает поддержка?", context
            )
            print(f"✅ RAG пайплайн: ответ - '{response}'")
            return True
        else:
            print("❌ RAG пайплайн: не найдены документы")
            return False

    except Exception as e:
        print(f"❌ RAG пайплайн ошибка: {e}")
        return False


def main():
    print("🚀 Запуск тестов подключения к LMStudio...")
    print(f"📊 Конфигурация:")
    print(f"   - LMStudio URL: {settings.LMSTUDIO_BASE_URL}")
    print(f"   - Модель: {settings.LMSTUDIO_MODEL_NAME}")
    print(f"   - Embedding модель: {settings.EMBEDDING_MODEL}")
    print()

    tests = [test_chromadb(), test_embeddings(), test_lmstudio(), test_rag_pipeline()]

    print()
    passed_tests = sum(tests)
    total_tests = len(tests)

    if passed_tests == total_tests:
        print("🎉 Все тесты пройдены! Система готова к работе.")
        print(f"📚 API документация: http://{settings.HOST}:{settings.PORT}/docs")
        print(f"💬 Веб-интерфейс: http://localhost:3000")
    else:
        print(f"⚠️  Пройдено {passed_tests}/{total_tests} тестов.")
        print("❌ Некоторые тесты не пройдены. Проверьте конфигурацию.")
        sys.exit(1)


if __name__ == "__main__":
    main()
