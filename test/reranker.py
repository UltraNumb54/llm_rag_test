from typing import List, Tuple

import torch
from app.config import settings
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RerankerService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = settings.RERANKER_MODEL
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Загрузка reranker модели: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.eval()
            logger.success(f"Reranker модель загружена: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки reranker модели: {e}")
            # Если не удалось загрузить, работаем без реранкера
            self.model = None
            self.tokenizer = None

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Реранкинг документов на основе запроса"""
        if not documents or self.model is None:
            # Если нет модели или документов, возвращаем исходные документы
            return [(doc, 1.0) for doc in documents[:top_k]]

        try:
            # Токенизация пар запрос-документ
            features = self.tokenizer(
                [query] * len(documents),
                documents,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            with torch.no_grad():
                scores = self.model(**features).logits

            # Сортируем документы по релевантности
            scored_docs = list(zip(documents, scores.squeeze().tolist()))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Реранкинг завершен: {len(scored_docs)} документов")
            return scored_docs[:top_k]

        except Exception as e:
            logger.error(f"Ошибка реранкинга: {e}")
            return [(doc, 1.0) for doc in documents[:top_k]]


reranker_service = RerankerService()
