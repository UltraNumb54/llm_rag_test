from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar


# rag_vllm_app/app/config.py
class Settings(BaseSettings):
    # Серверные настройки
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Модели
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LMStudio настройки
    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"  # Стандартный порт LMStudio
    LMSTUDIO_API_KEY: str = "none"  # LMStudio обычно не требует ключ
    LMSTUDIO_MODEL_NAME: str = "local-model"  # Или конкретное имя модели

    # Векторная БД
    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "test_docs"

    # Обработка документов
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    # RAG параметры
    TOP_K: int = 15
    RERANK_TOP_K: int = 7
    MAX_CONCURRENT_REQUESTS: int = 10

    # Безопасность
    API_KEY: str | None = None
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env")


settings = Settings()
