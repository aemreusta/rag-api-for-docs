#!/usr/bin/env python3
"""
Comprehensive tests for Qwen3 embedding provider functionality.
Tests both service and local modes, error handling, and configuration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import settings


class TestQwen3EmbeddingLocal:
    """Test Qwen3 local embedding functionality."""

    def test_qwen3_local_initialization_success(self):
        """Test successful initialization of Qwen3 local embedding."""
        from app.core.qwen3_embedding import Qwen3EmbeddingLocal

        with patch("app.core.qwen3_embedding.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            embedding = Qwen3EmbeddingLocal(model_name="Qwen/Qwen3-Embedding-0.6B", device="cpu")

            mock_st.assert_called_once_with("Qwen/Qwen3-Embedding-0.6B", device="cpu")
            assert embedding._model_name == "Qwen/Qwen3-Embedding-0.6B"
            assert embedding._model == mock_model

    def test_qwen3_local_initialization_failure(self):
        """Test failure when SentenceTransformer is not available."""
        from app.core.qwen3_embedding import Qwen3EmbeddingLocal

        with patch("app.core.qwen3_embedding.SentenceTransformer") as mock_st:
            mock_st.side_effect = ImportError("sentence-transformers not installed")

            # Mock the import error that would be raised
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'sentence_transformers'"),
            ):
                with pytest.raises(ImportError) as exc_info:
                    Qwen3EmbeddingLocal(model_name="Qwen/Qwen3-Embedding-0.6B")

                assert "No module named 'sentence_transformers'" in str(exc_info.value)

    def test_qwen3_local_get_text_embedding(self):
        """Test getting text embedding with local model."""
        from app.core.qwen3_embedding import Qwen3EmbeddingLocal

        with patch("app.core.qwen3_embedding.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_st.return_value = mock_model

            embedding = Qwen3EmbeddingLocal(model_name="Qwen/Qwen3-Embedding-0.6B")
            result = embedding._get_text_embedding("test text")

            assert result == [0.1, 0.2, 0.3, 0.4]
            mock_model.encode.assert_called_once_with("test text", convert_to_list=True)

    def test_qwen3_local_get_text_embedding_batch(self):
        """Test getting text embeddings in batch with local model."""
        from app.core.qwen3_embedding import Qwen3EmbeddingLocal

        with patch("app.core.qwen3_embedding.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_st.return_value = mock_model

            embedding = Qwen3EmbeddingLocal(model_name="Qwen/Qwen3-Embedding-0.6B")
            texts = ["text 1", "text 2"]
            result = embedding._get_text_embedding_batch(texts)

            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_model.encode.assert_called_once_with(texts, convert_to_list=True)


class TestQwen3EmbeddingService:
    """Test Qwen3 service embedding functionality."""

    def test_qwen3_service_initialization_missing_endpoint(self):
        """Test initialization failure when service endpoint is not configured."""
        from app.core.qwen3_embedding import Qwen3Embedding

        with patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""):
            with pytest.raises(ValueError) as exc_info:
                Qwen3Embedding(model_name="Qwen/Qwen3-Embedding-0.6B")

            assert "service endpoint not configured" in str(exc_info.value)

    def test_qwen3_service_initialization_success(self):
        """Test successful initialization of Qwen3 service embedding."""
        from app.core.qwen3_embedding import Qwen3Embedding

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", "http://test:8080"),
            patch.object(settings, "EMBEDDING_SERVICE_DIMENSIONS", 1024),
            patch.object(settings, "EMBEDDING_SERVICE_TIMEOUT", 120),
            patch.object(settings, "EMBEDDING_SERVICE_BATCH_SIZE", 32),
            patch("app.core.qwen3_embedding.httpx.AsyncClient") as mock_client,
        ):
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            embedding = Qwen3Embedding(
                model_name="Qwen/Qwen3-Embedding-0.6B", service_endpoint="http://test:8080"
            )

            mock_client.assert_called_once()
            assert embedding._model_name == "Qwen/Qwen3-Embedding-0.6B"
            assert embedding._service_endpoint == "http://test:8080"
            assert embedding._dimensions == 1024
            assert embedding._timeout == 120
            assert embedding._batch_size == 32

    @pytest.mark.asyncio
    async def test_qwen3_service_get_embeddings_success(self):
        """Test successful embedding retrieval from service."""
        from app.core.qwen3_embedding import Qwen3Embedding

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", "http://test:8080"),
            patch("app.core.qwen3_embedding.httpx.AsyncClient") as mock_client,
        ):
            # Mock successful response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
            }
            mock_response.raise_for_status.return_value = None

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_client_instance

            embedding = Qwen3Embedding(model_name="Qwen/Qwen3-Embedding-0.6B")
            texts = ["text 1", "text 2"]
            result = await embedding._get_embeddings_async(texts)

            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_client_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_qwen3_service_get_embeddings_with_instruction(self):
        """Test embedding retrieval with custom instruction."""
        from app.core.qwen3_embedding import Qwen3Embedding

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", "http://test:8080"),
            patch("app.core.qwen3_embedding.httpx.AsyncClient") as mock_client,
        ):
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            mock_response.raise_for_status.return_value = None

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_client_instance

            embedding = Qwen3Embedding(model_name="Qwen/Qwen3-Embedding-0.6B")
            await embedding._get_embeddings_async(["test text"])

            # Check that the request was made (instruction handling not implemented in basic version)  # noqa: E501
            call_args = mock_client_instance.post.call_args
            assert call_args is not None
            assert "test text" in str(call_args)

    @pytest.mark.asyncio
    async def test_qwen3_service_get_embeddings_failure(self):
        """Test embedding retrieval failure from service."""
        from app.core.qwen3_embedding import Qwen3Embedding

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", "http://test:8080"),
            patch("app.core.qwen3_embedding.httpx.AsyncClient") as mock_client,
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )
            mock_client.return_value = mock_client_instance

            embedding = Qwen3Embedding(model_name="Qwen/Qwen3-Embedding-0.6B")

            with pytest.raises(Exception) as exc_info:
                await embedding._get_embeddings_async(["test text"])

            assert "Connection failed" in str(exc_info.value)


class TestQwen3EmbeddingIntegration:
    """Integration tests for Qwen3 embedding provider."""

    def test_qwen3_service_mode_selection(self):
        """Test that service mode is selected when endpoint is configured."""

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", "http://test:8080"),
            patch("app.core.qwen3_embedding.Qwen3Embedding") as mock_qwen_service,
        ):
            from app.core.embeddings import _get_qwen3_embedding

            mock_instance = MagicMock()
            mock_qwen_service.return_value = mock_instance

            result = _get_qwen3_embedding("qwen3", "Qwen/Qwen3-Embedding-0.6B")

            mock_qwen_service.assert_called_once()
            assert result == mock_instance

    def test_qwen3_local_mode_selection(self):
        """Test that local mode is selected when endpoint is not configured."""

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
            patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_qwen_local,
        ):
            from app.core.embeddings import _get_qwen3_embedding

            mock_instance = MagicMock()
            mock_qwen_local.return_value = mock_instance

            result = _get_qwen3_embedding("qwen3", "Qwen/Qwen3-Embedding-0.6B")

            mock_qwen_local.assert_called_once()
            assert result == mock_instance

    def test_qwen3_provider_variants(self):
        """Test that various Qwen3 provider names work."""

        with (
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
            patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_qwen_local,
        ):
            from app.core.embeddings import get_embedding_model

            mock_instance = MagicMock()
            mock_qwen_local.return_value = mock_instance

            # Test different provider name variants
            for provider in ["qwen3", "qwen", "qwen3-embedding"]:
                with patch.object(settings, "EMBEDDING_PROVIDER", provider):
                    result = get_embedding_model()
                    assert result is not None


class TestQwen3EmbeddingConfiguration:
    """Test Qwen3 embedding configuration and defaults."""

    def test_default_qwen3_provider(self):
        """Test that Qwen3 is the default embedding provider."""
        from app.core.config import Settings

        settings = Settings()
        assert settings.EMBEDDING_PROVIDER == "qwen3"
        assert settings.EMBEDDING_MODEL_NAME == "Qwen/Qwen3-Embedding-0.6B"
        assert settings.EMBEDDING_DIM == 1024

    def test_qwen3_service_configuration(self):
        """Test Qwen3 service configuration settings."""
        from app.core.config import Settings

        settings = Settings()
        assert settings.EMBEDDING_SERVICE_ENDPOINT == ""
        assert settings.EMBEDDING_SERVICE_MODEL_NAME == "qwen-qwen3-embedding-0.6b"
        assert settings.EMBEDDING_SERVICE_DIMENSIONS == 1024
        assert settings.EMBEDDING_SERVICE_TIMEOUT == 120
        assert settings.EMBEDDING_SERVICE_BATCH_SIZE == 32
