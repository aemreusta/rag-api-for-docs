#!/usr/bin/env python3
"""
Tests for embedding provider validation and fail-fast behavior.
Ensures that configuration errors are caught early and clearly reported.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.embeddings import get_embedding_model


class TestEmbeddingProviderValidation:
    """Test embedding provider validation and error handling."""

    def test_google_provider_missing_api_key(self):
        """Test that Google provider fails fast when API key is missing."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "google"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "text-embedding-004"),
            patch.object(settings, "GOOGLE_AI_STUDIO_API_KEY", ""),
        ):
            # Ensure environment variables are cleared
            import os

            env_vars = ["GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_API_KEY"]
            original_values = {var: os.environ.get(var) for var in env_vars}
            for var in env_vars:
                os.environ.pop(var, None)

            try:
                with pytest.raises(RuntimeError) as exc_info:
                    get_embedding_model()

                error_msg = str(exc_info.value)
                assert "Google embedding provider" in error_msg
                assert "API key" in error_msg
                assert "GOOGLE_AI_STUDIO_API_KEY" in error_msg
                assert "GOOGLE_API_KEY" in error_msg
            finally:
                # Restore environment variables
                for var, value in original_values.items():
                    if value is not None:
                        os.environ[var] = value
                    elif var in os.environ:
                        os.environ.pop(var, None)

    def test_openai_provider_missing_api_key(self):
        """Test that OpenAI provider fails fast when API key is missing."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "openai"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        ):
            # Ensure OPENAI_API_KEY is not set
            import os

            original_key = os.environ.get("OPENAI_API_KEY")
            os.environ.pop("OPENAI_API_KEY", None)

            try:
                with pytest.raises(RuntimeError) as exc_info:
                    get_embedding_model()

                error_msg = str(exc_info.value)
                assert "OpenAI embedding provider" in error_msg
                assert "API key" in error_msg
                assert "OPENAI_API_KEY" in error_msg
            finally:
                # Restore environment variable
                if original_key is not None:
                    os.environ["OPENAI_API_KEY"] = original_key
                elif "OPENAI_API_KEY" in os.environ:
                    os.environ.pop("OPENAI_API_KEY", None)

    def test_qwen3_provider_missing_dependencies(self):
        """Test that Qwen3 provider fails fast when dependencies are missing."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"),
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
        ):
            # Mock import error for sentence-transformers by patching the import in the specific module  # noqa: E501
            with patch("app.core.qwen3_embedding.SentenceTransformer") as mock_st:
                mock_st.side_effect = ImportError("sentence-transformers not installed")
                with pytest.raises(RuntimeError) as exc_info:
                    get_embedding_model()

                error_msg = str(exc_info.value)
                assert "dependencies not available" in error_msg
                assert "sentence-transformers" in error_msg

    def test_qwen3_service_missing_endpoint(self):
        """Test that Qwen3 service mode fails when endpoint is not configured."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"),
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
            patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_local,
        ):
            # Mock the local embedding to fail with missing dependencies
            mock_local.side_effect = ValueError("Service endpoint not configured")

            with pytest.raises(RuntimeError) as exc_info:
                get_embedding_model()

            error_msg = str(exc_info.value)
            assert "Qwen3 embedding provider" in error_msg
            assert "failed to initialize" in error_msg

    def test_huggingface_provider_still_works(self):
        """Test that HuggingFace provider still works when explicitly configured."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "hf"),
            patch.object(
                settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            patch("app.core.embeddings.HuggingFaceEmbedding") as mock_hf,
        ):
            mock_instance = MagicMock()
            mock_hf.return_value = mock_instance

            result = get_embedding_model()

            # Should return a LoggedEmbeddingWrapper containing the HF embedding
            assert result is not None
            mock_hf.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_unknown_provider_defaults_to_qwen3(self):
        """Test that unknown provider defaults to Qwen3."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "unknown_provider"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"),
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
            patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_qwen_local,
        ):
            mock_instance = MagicMock()
            mock_qwen_local.return_value = mock_instance

            result = get_embedding_model()

            assert result is not None
            mock_qwen_local.assert_called_once()

    def test_qwen3_provider_success(self):
        """Test that Qwen3 provider works when properly configured."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"),
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
            patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_qwen_local,
        ):
            mock_instance = MagicMock()
            mock_qwen_local.return_value = mock_instance

            result = get_embedding_model()

            assert result is not None
            mock_qwen_local.assert_called_once_with(model_name="Qwen/Qwen3-Embedding-0.6B")

    def test_qwen3_service_mode_success(self):
        """Test that Qwen3 service mode works when endpoint is configured."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"),
            patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"),
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", "http://embedding-service:8080"),
            patch("app.core.qwen3_embedding.Qwen3Embedding") as mock_qwen_service,
        ):
            mock_instance = MagicMock()
            mock_qwen_service.return_value = mock_instance

            result = get_embedding_model()

            assert result is not None
            mock_qwen_service.assert_called_once()


class TestErrorMessageQuality:
    """Test that error messages are helpful and actionable."""

    def test_google_error_message_includes_resolution_steps(self):
        """Test that Google provider error includes resolution steps."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "google"),
            patch.object(settings, "GOOGLE_AI_STUDIO_API_KEY", None),
        ):
            import os

            original_values = {
                "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
                "GOOGLE_AI_STUDIO_API_KEY": os.environ.get("GOOGLE_AI_STUDIO_API_KEY"),
            }
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GOOGLE_AI_STUDIO_API_KEY", None)

            try:
                with pytest.raises(RuntimeError) as exc_info:
                    get_embedding_model()

                error_msg = str(exc_info.value)
                assert "1. Set GOOGLE_AI_STUDIO_API_KEY" in error_msg
                assert "2. Or set GOOGLE_API_KEY" in error_msg
                assert "3. Check that the API key is valid" in error_msg
                assert "4. Install required dependencies" in error_msg
            finally:
                for var, value in original_values.items():
                    if value is not None:
                        os.environ[var] = value

    def test_openai_error_message_includes_resolution_steps(self):
        """Test that OpenAI provider error includes resolution steps."""
        with patch.object(settings, "EMBEDDING_PROVIDER", "openai"):
            import os

            original_key = os.environ.get("OPENAI_API_KEY")
            os.environ.pop("OPENAI_API_KEY", None)

            try:
                with pytest.raises(RuntimeError) as exc_info:
                    get_embedding_model()

                error_msg = str(exc_info.value)
                assert "1. Set OPENAI_API_KEY" in error_msg
                assert "2. Check that the API key is valid" in error_msg
                assert "3. Verify the model name is correct" in error_msg
                assert "4. Install required dependencies" in error_msg
            finally:
                if original_key is not None:
                    os.environ["OPENAI_API_KEY"] = original_key

    def test_qwen3_error_message_includes_resolution_steps(self):
        """Test that Qwen3 provider error includes resolution steps."""
        with (
            patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"),
            patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""),
            patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_local,
        ):
            mock_local.side_effect = ValueError("Service endpoint not configured")

            with pytest.raises(RuntimeError) as exc_info:
                get_embedding_model()

            error_msg = str(exc_info.value)
            assert "1. Set EMBEDDING_SERVICE_ENDPOINT" in error_msg
            assert "2. Install required dependencies" in error_msg
            assert "3. Verify the model name is correct" in error_msg
