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

    # Vector Embeddings Settings (default: Google Gemini Embedding)
    # Options: hf | openai | google
    EMBEDDING_PROVIDER: str = "google"
    # Default to GA Gemini embedding model id
    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"
    # Matryoshka Representation Learning dims supported: 3072 | 1536 | 768
    EMBEDDING_DIM: int = 1536

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

    # CORS Configuration
    CORS_ALLOW_ORIGINS: str = ""  # Comma-separated list of allowed origins
    CORS_ALLOW_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"  # HTTP methods
    CORS_ALLOW_HEADERS: str = "Authorization,Content-Type,X-API-Key,X-Request-ID"  # Headers
    CORS_ALLOW_CREDENTIALS: bool = False  # Allow credentials (cookies, auth headers)
    CORS_MAX_AGE: int = 600  # Pre-flight cache duration in seconds (10 minutes)

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
