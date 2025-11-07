# rag_vllm_app/app/services/file_processor.py
import os
from typing import List

from app.services.vector_store import vector_store
from app.utils.text_splitter import TextSplitter
from docx import Document
from loguru import logger
from pypdf import PdfReader


class FileProcessor:
    def __init__(self):
        self.text_splitter = TextSplitter()

    def process_file(self, file_path: str, filename: str):
        """Обработка файла и добавление в векторную базу"""
        try:
            logger.info(f"Обработка файла: {filename}")

            # Определяем тип файла и извлекаем текст
            if filename.lower().endswith(".pdf"):
                text = self._extract_pdf_text(file_path)
            elif filename.lower().endswith((".docx", ".doc")):
                text = self._extract_docx_text(file_path)
            else:
                text = self._extract_txt_text(file_path)

            if not text.strip():
                logger.warning(f"Файл {filename} не содержит текста")
                return

            # Разбиваем текст на чанки
            chunks = self.text_splitter.split_text(text)

            # Добавляем в векторное хранилище
            metadatas = [
                {
                    "source": filename,
                    "chunk_id": i,
                    "file_type": os.path.splitext(filename)[1],
                }
                for i in range(len(chunks))
            ]

            vector_store.add_documents(chunks, metadatas)
            logger.success(
                f"Файл {filename} успешно обработан, создано {len(chunks)} чанков"
            )

        except Exception as e:
            logger.error(f"Ошибка обработки файла {filename}: {e}")
            raise

    def _extract_pdf_text(self, file_path: str) -> str:
        """Извлечение текста из PDF"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF: {e}")
            return ""

    def _extract_docx_text(self, file_path: str) -> str:
        """Извлечение текста из DOCX"""
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из DOCX: {e}")
            return ""

    def _extract_txt_text(self, file_path: str) -> str:
        """Извлечение текста из TXT"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Ошибка чтения текстового файла: {e}")
            return ""
