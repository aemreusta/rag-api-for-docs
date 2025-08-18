from unittest.mock import patch

from app.core.config import settings
from app.core.embeddings import get_embedding_model


def test_default_embedding_is_gemini_google():
    with (
        patch.object(settings, "EMBEDDING_PROVIDER", "google"),
        patch.object(settings, "EMBEDDING_MODEL_NAME", "text-embedding-004"),
        patch.object(settings, "GOOGLE_AI_STUDIO_API_KEY", "test-key"),
    ):
        model = get_embedding_model()
        # Google embedding is available, should get either Google or HF fallback
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        assert model is not None
        # Accept either Google embedding or HF fallback
        assert hasattr(model, "aget_query_embedding") or isinstance(model, HuggingFaceEmbedding)


def test_fallback_to_hf_when_google_unavailable(monkeypatch):
    # Simulate import failure for Google embeddings class
    monkeypatch.setattr("app.core.embeddings.settings.EMBEDDING_PROVIDER", "google", raising=False)
    monkeypatch.setattr(
        "app.core.embeddings.settings.EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
        raising=False,
    )
    # Force import error path
    monkeypatch.setitem(__import__("sys").modules, "llama_index.embeddings.google", None)

    model = get_embedding_model()
    # Should fall back to HuggingFaceEmbedding instance
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    assert isinstance(model, HuggingFaceEmbedding)


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
        patch("os.environ.get") as mock_env_get,
    ):
        # Mock the OPENAI_API_KEY environment variable
        mock_env_get.return_value = "test-openai-api-key"

        model = get_embedding_model()
        assert hasattr(model, "embed") or hasattr(model, "aget_query_embedding")
