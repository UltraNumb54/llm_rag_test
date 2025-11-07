# rag_vllm_app/app/utils/text_splitter.py
import re
from typing import List

from app.config import settings


class TextSplitter:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def split_text(self, text: str) -> List[str]:
        """Разбивает текст на перекрывающиеся чанки"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Если не первая итерация, то  добавляем перекрытие
            if start > 0:
                start = max(0, start - self.chunk_overlap)

            # Не вышли за пределы текста
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Находим ближайший конец предложения
            chunk = text[start:end]
            last_period = max(
                chunk.rfind(". "),
                chunk.rfind("! "),
                chunk.rfind("? "),
                chunk.rfind("\n\n"),
            )

            if last_period != -1 and last_period > self.chunk_size // 2:
                end = start + last_period + 1

            chunks.append(text[start:end].strip())
            start = end

        return chunks
