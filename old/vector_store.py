# rag_vllm_app/app/services/vector_store.py:
from app.core.database import get_collection
from app.services.embedding import embedding_service
from rag_vllm_app.app.config import settings
from loguru import logger
import uuid
from typing import List, Dict, Any


class VectorStoreService:
    def __init__(self):
        self.collection = get_collection()

    def add_documents(
        self, documents: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Добавление документов в векторное хранилище"""
        if not documents:
            logger.warning("Попытка добавить пустой список документов")
            return []

        if metadatas is None:
            metadatas = [{}] * len(documents)
        elif len(metadatas) != len(documents):
            logger.warning(
                "Количество метаданных не совпадает с количеством документов"
            )
            metadatas = [{}] * len(documents)

        # Генерируем ID для документов
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        try:
            # Создаем эмбеддинги
            logger.info(f"Создание эмбеддингов для {len(documents)} документов...")
            embeddings = embedding_service.encode(documents)

            # Добавляем в коллекцию
            self.collection.add(
                embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
            )

            logger.success(
                f"Добавлено {len(documents)} документов в векторное хранилище"
            )
            return ids

        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Поиск похожих документов"""
        if top_k is None:
            top_k = settings.TOP_K

        try:
            # Создаем эмбеддинг для запроса
            query_embedding = embedding_service.encode(query)

            # Ищем похожие документы
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, 20),  # Ограничиваем максимальное количество
                include=["documents", "metadatas", "distances"],
            )

            # Форматируем результаты
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append(
                        {
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i]
                            if results["metadatas"]
                            else {},
                            "distance": results["distances"][0][i]
                            if results["distances"]
                            else 0,
                            "score": 1
                            - (
                                results["distances"][0][i]
                                if results["distances"]
                                else 0
                            ),  # Конвертируем расстояние в схожесть
                        }
                    )

            logger.debug(f"Найдено {len(formatted_results)} релевантных документов")
            return formatted_results

        except Exception as e:
            logger.error(f"Ошибка поиска в векторном хранилище: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        try:
            count = self.collection.count()
            return {
                "collection_name": settings.COLLECTION_NAME,
                "document_count": count,
                "status": "healthy",
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о коллекции: {e}")
            return {
                "collection_name": settings.COLLECTION_NAME,
                "document_count": 0,
                "status": "error",
                "error": str(e),
            }


# Глобальный экземпляр сервиса
vector_store = VectorStoreService()

