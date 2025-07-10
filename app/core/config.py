from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # LLM Provider Configuration
    OPENROUTER_API_KEY: str
    GROQ_API_KEY: str = ""  # Optional
    GROQ_MODEL_NAME: str = "llama3-70b-8192"  # Default Groq model
    GROQ_TIMEOUT_SECONDS: int = 30  # Groq-specific timeout
    OPENAI_API_KEY: str = ""  # Optional
    GOOGLE_AI_STUDIO_API_KEY: str = ""  # Optional
    LOCAL_LLM_PATH: str = ""  # Optional
    LLM_MODEL_NAME: str = "google/gemini-1.5-pro-latest"

    # LLM Router Configuration
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_FALLBACK_CACHE_SECONDS: int = 300  # 5 minutes

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

    # Vector Embeddings Settings
    EMBEDDING_DIM: int = 1536  # OpenAI text-embedding-3-small dimension

    # Metrics & Monitoring Settings
    METRICS_BACKEND: str = "auto"  # auto, prometheus, datadog, opentelemetry, noop
    DATADOG_API_KEY: str = ""  # Optional for DataDog metrics
    PROMETHEUS_ENABLED: bool = True  # Enable Prometheus metrics endpoint

    # Redis for Rate Limiting and Caching
    REDIS_URL: str = "redis://:myredissecret@redis:6379/0"
    RATE_LIMIT_COUNT: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 86400  # 24 hours

    # Cache Configuration
    CACHE_TTL_SECONDS: int = 3600  # 1 hour default TTL for chat responses
    CACHE_MAX_SIZE: int = 1000  # Max entries for in-memory cache fallback
    CACHE_ENABLED: bool = True  # Enable/disable caching globally

    # Structured Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_LEVEL_SQL: str = "WARNING"
    LOG_JSON: bool = True
    LOG_TO_FILE: bool = False
    LOG_FILE: str = "logs/app_%(process)d_%Y%m%d.log"

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


settings = Settings()
