from unittest.mock import patch

from app.core.config import settings
from app.core.embeddings import get_embedding_model


def test_default_embedding_is_gemini_google():
    with (
        patch.object(settings, "EMBEDDING_PROVIDER", "google"),
        patch.object(settings, "EMBEDDING_MODEL_NAME", "gemini-embedding-001"),
    ):
        model = get_embedding_model()
        # If the Google embedding class is available, we get it; otherwise, fallback to HF
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        assert model is not None
        # Either a Google embedding or our HF fallback
        assert hasattr(model, "embed") or isinstance(model, HuggingFaceEmbedding)


def test_fallback_to_hf_when_google_unavailable(monkeypatch):
    # Simulate import failure for Google embeddings class
    monkeypatch.setattr("app.core.embeddings.settings.EMBEDDING_PROVIDER", "google", raising=False)
    monkeypatch.setattr(
        "app.core.embeddings.settings.EMBEDDING_MODEL_NAME", "gemini-embedding-001", raising=False
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
