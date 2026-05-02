from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Embedding Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"  # 384 dims, good for semantic search
    # Alternative: "sentence-transformers/all-MiniLM-L6-v2" (faster, lighter)
    EMBEDDING_DIMENSION: int = 1024
    BATCH_SIZE: int = 100  # Batch size for Qdrant upsert
    
    # Qdrant Cloud
    QDRANT_URL: str = ""  # Qdrant Cloud URL
    QDRANT_API_KEY: str = ""  # Qdrant Cloud API Key
    QDRANT_USE_CLOUD: bool = True  # True for cloud, False for local
    
    # Qdrant Local (fallback)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # Collections
    CV_COLLECTION_NAME: str = "cv_embeddings"
    JD_COLLECTION_NAME: str = "jd_embeddings"

    # RabbitMQ
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 1000

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"          # Conversational / reasoning LLM
    SCORING_GEMINI_MODEL: str = "gemini-2.5-pro"    # Deep CV-JD scoring (tiered routing)
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_MAX_TOKENS: int = 2048
    
    # Internal Service API (Recruitment)
    RECRUITMENT_SERVICE_URL: str = "http://localhost:8082"
    INTERNAL_SERVICE_SECRET: str = "chatbot-service"
    MAX_HISTORY_TURNS: int = 20

    # Reranker — multilingual model to support both Vietnamese and English queries
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cache settings instance"""
    return Settings()