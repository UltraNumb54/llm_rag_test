# rag_vllm_app/app/services/embedding.py:
from sentence_transformers import SentenceTransformer
import numpy as np
from rag_vllm_app.app.config import settings
from loguru import logger


class EmbeddingService:
    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.success(f"Модель эмбеддингов загружена: {self.model_name}")
            logger.info(
                f"Размерность эмбеддингов: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели эмбеддингов: {e}")
            raise

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Кодирование текста в эмбеддинги"""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()

            logger.debug(
                f"Закодировано {len(embeddings)} текстов, размерность: {len(embeddings[0])}"
            )
            return embeddings

        except Exception as e:
            logger.error(f"Ошибка кодирования текста: {e}")
            raise


# Глобальный экземпляр сервиса
embedding_service = EmbeddingService()
