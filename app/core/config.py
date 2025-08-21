from pydantic import ConfigDict, computed_field
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
    # Default chat model selection
    LLM_MODEL_NAME: str = "gemini-2.0-flash"  # Preferred default via Google AI Studio
    GOOGLE_MODEL_NAME: str = "gemini-2.0-flash"

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
    UPLOADED_DOCS_DIR: str = "uploaded_docs"
    STORAGE_BACKEND: str = "local"  # local|minio
    # MinIO (optional)
    MINIO_ENDPOINT: str = ""
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "uploads"
    MINIO_SECURE: bool = False

    # Vector Embeddings Settings
    # Options: qwen3 | qwen | hf | openai | google
    EMBEDDING_PROVIDER: str = "qwen3"
    # Default Qwen3 embedding model
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    # Qwen3 embedding dimension (1024 for 0.6B model)
    EMBEDDING_DIM: int = 1024

    # HuggingFace Embedding Service Settings (Local vLLM Server)
    EMBEDDING_SERVICE_ENDPOINT: str = ""  # e.g., http://embedding-service:8080
    EMBEDDING_SERVICE_MODEL_NAME: str = "qwen-qwen3-embedding-0.6b"  # Served model name
    EMBEDDING_SERVICE_DIMENSIONS: int = 1024  # Default dimension
    EMBEDDING_SERVICE_TIMEOUT: int = 120
    EMBEDDING_SERVICE_BATCH_SIZE: int = 32

    # Backward compatibility
    QWEN_ENDPOINT: str = ""  # Deprecated: use EMBEDDING_SERVICE_ENDPOINT
    QWEN_MODEL_NAME: str = "Qwen3-Embedding-0.6B"  # Deprecated
    QWEN_DIMENSIONS: int = 1024  # Deprecated
    QWEN_TIMEOUT: int = 120  # Deprecated
    QWEN_BATCH_SIZE: int = 32  # Deprecated

    # Metrics & Monitoring Settings
    METRICS_BACKEND: str = "auto"  # auto, prometheus, datadog, opentelemetry, noop
    DATADOG_API_KEY: str = ""  # Optional for DataDog metrics
    PROMETHEUS_ENABLED: bool = True  # Enable Prometheus metrics endpoint

    # Redis for Rate Limiting and Caching
    REDIS_URL: str = "redis://:myredissecret@redis:6379/0"
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "myredissecret"
    RATE_LIMIT_COUNT: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 86400  # 24 hours

    # Cache Configuration
    CACHE_TTL_SECONDS: int = 3600  # 1 hour default TTL for chat responses
    CACHE_MAX_SIZE: int = 1000  # Max entries for in-memory cache fallback
    CACHE_ENABLED: bool = True  # Enable/disable caching globally

    # CORS Configuration
    CORS_ALLOW_ORIGINS: str = ""  # Comma-separated list of allowed origins
    CORS_ALLOW_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"  # HTTP methods
    CORS_ALLOW_HEADERS: str = "Authorization,Content-Type,X-API-Key,X-Request-ID"  # Headers
    CORS_ALLOW_CREDENTIALS: bool = False  # Allow credentials (cookies, auth headers)
    CORS_MAX_AGE: int = 600  # Pre-flight cache duration in seconds (10 minutes)

    # Ingestion / Chunking
    CHUNKER_LANGUAGE: str = "turkish"  # Language for sentence tokenization (NLTK)
    # Migration flags
    INGEST_PARALLEL_DEPLOYMENT: bool = False  # Dual-run: legacy scripts + new API

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS_ALLOW_ORIGINS and return appropriate origins list."""
        # In DEBUG mode with development environment, default to wildcard if no origins specified
        if self.DEBUG and self.ENVIRONMENT == "development" and not self.CORS_ALLOW_ORIGINS.strip():
            return ["*"]

        # Parse comma-separated origins
        if self.CORS_ALLOW_ORIGINS.strip():
            origins = [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",")]
            # Filter out empty strings
            return [origin for origin in origins if origin]

        # No origins configured - restrictive by default
        return []

    @computed_field
    @property
    def cors_allow_credentials_safe(self) -> bool:
        """Return safe credentials setting - False when using wildcard origins."""
        # Cannot use credentials with wildcard origins per CORS spec
        if "*" in self.cors_origins:
            return False
        return self.CORS_ALLOW_CREDENTIALS

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
