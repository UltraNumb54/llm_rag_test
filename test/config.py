# rag_vllm_app/app/config.py
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    EMBEDDING_MODEL: str = (
        "/home/user/.cache/modelscope/hub/models/intfloat/multilingual-e5-small"
    )
    RERANKER_MODEL: str = (
        "/home/user/.cache/modelscope/hub/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    LMSTUDIO_BASE_URL: str = "http://26.246.206.164:1234/v1"
    LMSTUDIO_API_KEY: str = "none"
    LMSTUDIO_MODEL_NAME: str = "local-model"

    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "tech_support_docs"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE: int = 100 * 1024 * 1024

    TOP_K: int = 3
    RERANK_TOP_K: int = 2
    MAX_CONCURRENT_REQUESTS: int = 10

    API_KEY: Optional[str] = None
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])


settings = Settings()
