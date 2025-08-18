from __future__ import annotations

import os
from typing import Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings


def _get_openai_embedding(model_name: str) -> Any:
    """Initialize OpenAI embedding with validation."""
    # Validate OpenAI API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        error_msg = (
            "OpenAI embedding provider selected but no API key found. "
            "Please set OPENAI_API_KEY in your environment."
        )
        raise ValueError(error_msg)

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model_name)
    except ImportError as e:
        error_msg = (
            f"OpenAI embedding provider selected but required dependencies not available: {e}. "
            "Please install openai or llama-index[openai] package."
        )
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to initialize OpenAI embedding provider: {e}. "
            "Please check your OPENAI_API_KEY and network connectivity."
        )
        raise RuntimeError(error_msg) from e


def _get_google_embedding(provider: str, model_name: str) -> Any:
    """Initialize Google embedding with validation."""
    # Validate Google API key is available before attempting to use Google provider
    google_api_key = settings.GOOGLE_AI_STUDIO_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        error_msg = (
            f"Google embedding provider selected ('{provider}') but no API key found. "
            "Please set GOOGLE_AI_STUDIO_API_KEY in your environment or configuration."
        )
        raise ValueError(error_msg)

    try:
        from llama_index.embeddings.google import GeminiEmbedding

        # Set the API key in environment if not already set
        if settings.GOOGLE_AI_STUDIO_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_AI_STUDIO_API_KEY

        model_id = model_name
        if not (model_id.startswith("models/") or model_id.startswith("tunedModels/")):
            model_id = f"models/{model_id}"

        # Initialize GeminiEmbedding with correct parameters
        return GeminiEmbedding(
            model_name=model_id, api_key=google_api_key, task_type="retrieval_document"
        )
    except ImportError as e:
        error_msg = (
            f"Google embedding provider selected but required dependencies not available: {e}. "
            "Please install google-generativeai or llama-index[google] package."
        )
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to initialize Google embedding provider: {e}. "
            "Please check your GOOGLE_AI_STUDIO_API_KEY and network connectivity."
        )
        raise RuntimeError(error_msg) from e


def get_embedding_model() -> Any:
    """Get embedding model based on configured provider with strict validation."""
    provider = (settings.EMBEDDING_PROVIDER or "hf").lower()
    model_name = settings.EMBEDDING_MODEL_NAME

    if provider == "openai":
        return _get_openai_embedding(model_name)
    elif provider in ("google", "gemini", "google_genai"):
        return _get_google_embedding(provider, model_name)

    # Default HF fallback
    return HuggingFaceEmbedding(model_name=model_name)
