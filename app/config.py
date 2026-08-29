"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — all values come from env vars or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Groq API (free tier) ───
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # ─── Whisper ───
    WHISPER_MODEL_SIZE: str = "base"

    # ─── PostgreSQL ───
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/sales_analytics"
    )

    # ─── Redis ───
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── ChromaDB ───
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # ─── Embedding model ───
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
