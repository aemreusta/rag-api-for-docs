from __future__ import annotations

from typing import Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings


def get_embedding_model() -> Any:
    provider = (settings.EMBEDDING_PROVIDER or "hf").lower()
    model_name = settings.EMBEDDING_MODEL_NAME

    if provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(model=model_name)
        except Exception:  # pragma: no cover
            pass
    elif provider in ("google", "gemini", "google_genai"):
        try:
            # Prefer Google GenAI embedding class if available
            from llama_index.embeddings.google import GoogleTextEmbedding

            return GoogleTextEmbedding(model_name=model_name)
        except Exception:  # pragma: no cover
            # Fall through to HF if Google embeddings are unavailable
            pass

    # Default HF fallback
    return HuggingFaceEmbedding(model_name=model_name)
