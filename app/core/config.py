from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # LLM Provider
    OPENROUTER_API_KEY: str
    LLM_MODEL_NAME: str = "google/gemini-1.5-pro-latest"

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "http://langfuse:3000"  # Using Docker Compose service name

    # Security
    API_KEY: str  # For the chat endpoint
    ADMIN_API_KEY: str  # For the admin endpoint

    # Optional settings with defaults
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # PostgreSQL Settings
    POSTGRES_SERVER: str = "postgres"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "app"

    # Document Settings
    PDF_DOCUMENTS_DIR: str = "pdf_documents"

    # Redis for Rate Limiting
    REDIS_URL: str = "redis://:myredissecret@redis:6379/0"
    RATE_LIMIT_COUNT: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 86400  # 24 hours

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


settings = Settings()
