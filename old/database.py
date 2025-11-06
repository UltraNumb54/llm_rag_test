# rag_vllm_app/app/core/database.py:
import chromadb
from rag_vllm_app.app.config import settings
from loguru import logger


class ChromaDBManager:
    _client = None
    _collection = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            try:
                cls._client = chromadb.PersistentClient(
                    path=settings.CHROMA_PATH,
                    settings=chromadb.Settings(allow_reset=True),
                )
                logger.info(f"ChromaDB клиент создан: {settings.CHROMA_PATH}")

                # Тестируем подключение
                heartbeat = cls._client.heartbeat()
                logger.debug(f"ChromaDB heartbeat: {heartbeat}")

            except Exception as e:
                logger.error(f"Ошибка инициализации ChromaDB: {e}")
                raise
        return cls._client

    @classmethod
    def get_collection(cls):
        if cls._collection is None:
            client = cls.get_client()
            try:
                # Пытаемся получить существующую коллекцию
                cls._collection = client.get_collection(settings.COLLECTION_NAME)
                logger.info(f"Коллекция '{settings.COLLECTION_NAME}' загружена")
                logger.info(
                    f"Количество документов в коллекции: {cls._collection.count()}"
                )
            except Exception as e:
                # Создаем новую коллекцию если не существует
                logger.info(f"Создание новой коллекции '{settings.COLLECTION_NAME}'")
                cls._collection = client.create_collection(
                    name=settings.COLLECTION_NAME,
                    metadata={
                        "hnsw:space": "cosine",
                        "description": "Документы технической поддержки",
                    },
                )
                logger.success(f"Создана новая коллекция '{settings.COLLECTION_NAME}'")
        return cls._collection


def get_chroma_client():
    return ChromaDBManager.get_client()


def get_collection():
    return ChromaDBManager.get_collection()
