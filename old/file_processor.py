# rag_vllm_app/app/services/file_processor.py:
import os
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import vector_store
from rag_vllm_app.app.config import settings
from loguru import logger
import html
from typing import List


class FileProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    async def process_file(self, file_path: str, filename: str):
        """Обработка файла и добавление в векторную БД"""
        try:
            logger.info(f"Начало обработки файла: {filename}")

            # Извлекаем текст в зависимости от формата
            text = await self.extract_text(file_path, filename)

            if not text:
                logger.warning(f"Не удалось извлечь текст из файла: {filename}")
                return

            # Разбиваем на чанки
            chunks = self.text_splitter.split_text(text)

            # Создаем метаданные для каждого чанка
            metadatas = [
                {"source": filename, "chunk_id": i, "total_chunks": len(chunks)}
                for i in range(len(chunks))
            ]

            # Добавляем в векторное хранилище
            vector_store.add_documents(chunks, metadatas)

            logger.success(f"Файл обработан: {filename} -> {len(chunks)} чанков")

        except Exception as e:
            logger.error(f"Ошибка обработки файла {filename}: {e}")

    async def extract_text(self, file_path: str, filename: str) -> str:
        """Извлечение текста из файла в зависимости от формата"""
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                return self._extract_from_pdf(file_path)
            elif ext == ".docx":
                return self._extract_from_docx(file_path)
            elif ext in [".txt", ".md"]:
                return self._extract_from_text(file_path)
            else:
                logger.warning(f"Неподдерживаемый формат файла: {ext}")
                return ""
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {filename}: {e}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        """Извлечение текста из PDF"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Извлечение текста из DOCX"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _extract_from_text(self, file_path: str) -> str:
        """Извлечение текста из текстового файла"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
