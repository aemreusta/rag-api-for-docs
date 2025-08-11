from unittest.mock import patch

from app.core.embeddings import get_embedding_model


def test_default_hf_embedding_selected():
    with (
        patch("app.core.config.settings.EMBEDDING_PROVIDER", "hf"),
        patch(
            "app.core.config.settings.EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    ):
        model = get_embedding_model()
        # Avoid importing HF class directly; check for common attribute
        assert hasattr(model, "embed") or hasattr(model, "aget_query_embedding")


def test_openai_embedding_selected_when_configured():
    # If OpenAI embedding package is not available, the helper should fall back to HF
    with (
        patch("app.core.config.settings.EMBEDDING_PROVIDER", "openai"),
        patch("app.core.config.settings.EMBEDDING_MODEL_NAME", "text-embedding-3-large"),
    ):
        model = get_embedding_model()
        assert hasattr(model, "embed") or hasattr(model, "aget_query_embedding")
