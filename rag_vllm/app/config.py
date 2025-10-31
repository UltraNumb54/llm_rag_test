import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "none"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    
    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "production_documents"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    TOP_K: int = 15
    RERANK_TOP_K: int = 7
    MAX_CONCURRENT_REQUESTS: int = 10
    BATCH_SIZE: int = 32
    

    API_KEY: Optional[str] = None
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"

settings = Settings()
