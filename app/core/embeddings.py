from __future__ import annotations

import os
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
            # Use new GeminiEmbedding from LlamaIndex
            from llama_index.embeddings.google import GeminiEmbedding

            if settings.GOOGLE_AI_STUDIO_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_AI_STUDIO_API_KEY

            # Ensure fully-qualified model id for GenAI SDK
            model_id = model_name
            if not (model_id.startswith("models/") or model_id.startswith("tunedModels/")):
                model_id = f"models/{model_id}"

            try:
                return GeminiEmbedding(model_name=model_id, dimensions=settings.EMBEDDING_DIM)
            except TypeError:
                return GeminiEmbedding(
                    model_name=model_id, output_dimensionality=settings.EMBEDDING_DIM
                )
        except Exception:  # pragma: no cover
            # Fall back safely to a well-known HF model to keep tests/envs running
            return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Default HF fallback
    return HuggingFaceEmbedding(model_name=model_name)
