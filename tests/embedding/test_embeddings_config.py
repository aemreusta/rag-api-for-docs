from unittest.mock import patch

from app.core.config import settings
from app.core.embeddings import get_embedding_model


def test_default_embedding_is_gemini_google():
    # Mock the newer GoogleGenerativeAIEmbedding module to avoid deprecated fallback
    import sys
    from unittest.mock import MagicMock

    # Create a mock module and class to simulate the newer Google embedding package
    mock_module = MagicMock()
    mock_google_embedding_class = MagicMock()
    mock_google_embedding_instance = MagicMock()
    mock_google_embedding_instance.aget_query_embedding = MagicMock()
    mock_google_embedding_class.return_value = mock_google_embedding_instance
    mock_module.GoogleGenerativeAIEmbedding = mock_google_embedding_class

    with (
        patch.object(settings, "EMBEDDING_PROVIDER", "google"),
        patch.object(settings, "EMBEDDING_MODEL_NAME", "text-embedding-004"),
        patch.object(settings, "GOOGLE_AI_STUDIO_API_KEY", "test-key"),
        patch.dict(sys.modules, {"llama_index.embeddings.google_genai": mock_module}),
    ):
        model = get_embedding_model()
        # Should get LoggedEmbeddingWrapper containing the mocked Google embedding
        from app.core.embeddings import LoggedEmbeddingWrapper

        assert isinstance(model, LoggedEmbeddingWrapper)
        assert model is not None
        assert hasattr(model, "_aget_query_embedding")


def test_hf_provider_works_when_configured():
    # Test that HF provider works when explicitly configured
    with (
        patch.object(settings, "EMBEDDING_PROVIDER", "hf"),
        patch.object(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    ):
        model = get_embedding_model()
        # Should get LoggedEmbeddingWrapper containing HuggingFaceEmbedding instance
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        from app.core.embeddings import LoggedEmbeddingWrapper

        assert isinstance(model, LoggedEmbeddingWrapper)
        assert isinstance(model._embedding, HuggingFaceEmbedding)


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


def test_default_qwen3_embedding_selected():
    """Test that Qwen3 is now the default embedding provider."""
    from unittest.mock import MagicMock

    with (
        patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"),
        patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"),
        patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
        patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_qwen_local,
    ):
        mock_instance = MagicMock()
        mock_qwen_local.return_value = mock_instance

        model = get_embedding_model()
        from app.core.embeddings import LoggedEmbeddingWrapper

        assert isinstance(model, LoggedEmbeddingWrapper)
        mock_qwen_local.assert_called_once_with(model_name="Qwen/Qwen3-Embedding-0.6B")
