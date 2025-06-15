from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    API_KEY: str

    # LlamaIndex Settings
    OPENAI_API_KEY: str

    # Langfuse Settings
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: Optional[str] = "https://cloud.langfuse.com"

    # Document Settings
    PDF_DOCUMENTS_DIR: str = "pdf_documents"

    class Config:
        env_file = ".env"


settings = Settings()
