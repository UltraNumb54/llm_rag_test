# rag_vllm_app/app/services/embedding.py
import numpy as np
import torch
import torch.nn.functional as F
from app.config import settings
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class EmbeddingService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.success(f"Модель эмбеддингов загружена: {self.model_name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            raise

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """Кодирование текста в эмбеддинги для E5"""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        try:
            texts = [f"passage: {text}" for text in texts]

            # Токенизация
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Вычисление эмбеддингов
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.numpy().tolist()

            logger.debug(
                f"Закодировано {len(embeddings)} текстов, размерность: {len(embeddings[0])}"
            )
            return embeddings

        except Exception as e:
            logger.error(f"Ошибка кодирования текста: {e}")
            raise


embedding_service = EmbeddingService()
