"""
Comprehensive test suite for request tracking functionality.

This module tests all the enhanced tracking components:
- Embedding request tracking
- Vector store performance metrics
- Document processing pipeline metrics
- LLM provider response times and success rates
"""

import asyncio
import time

import pytest

from app.core.request_tracking import get_request_tracker


class TestRequestTracking:
    """Test request tracking functionality."""

    @pytest.mark.asyncio
    async def test_embedding_request_tracking_success(self):
        """Test successful embedding request tracking."""
        tracker = get_request_tracker()

        with tracker.track_embedding_request(
            provider="test_provider", model="test-model", input_type="text", batch_size=1
        ) as context:
            # Simulate embedding generation
            await asyncio.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "input_size_chars": 100,
                    "input_size_tokens": 25,
                    "output_dimensions": 384,
                }
            )

        # Test completed without exceptions
        assert context["provider"] == "test_provider"
        assert context["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_embedding_request_tracking_failure(self):
        """Test failed embedding request tracking."""
        tracker = get_request_tracker()

        with pytest.raises(ValueError, match="Test embedding error"):
            with tracker.track_embedding_request(
                provider="test_provider", model="test-model", input_type="text", batch_size=1
            ):
                # Simulate embedding failure
                raise ValueError("Test embedding error")

    @pytest.mark.asyncio
    async def test_vector_store_operation_tracking_success(self):
        """Test successful vector store operation tracking."""
        tracker = get_request_tracker()

        with tracker.track_vector_operation(
            operation="query", collection="test_collection", query_type="similarity"
        ) as context:
            # Simulate vector store query
            await asyncio.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "similarity_top_k": 5,
                    "query_length": 50,
                    "result_count": 3,
                }
            )

        assert context["operation"] == "query"
        assert context["collection"] == "test_collection"

    @pytest.mark.asyncio
    async def test_vector_store_operation_tracking_add(self):
        """Test vector store insertion tracking."""
        tracker = get_request_tracker()

        with tracker.track_vector_operation(
            operation="add", collection="test_collection"
        ) as context:
            # Simulate vector store insertion
            await asyncio.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "node_count": 10,
                    "table_name": "content_embeddings",
                }
            )

        assert context["operation"] == "add"
        assert context["collection"] == "test_collection"

    @pytest.mark.asyncio
    async def test_document_processing_tracking_success(self):
        """Test successful document processing pipeline tracking."""
        tracker = get_request_tracker()

        with tracker.track_document_processing(
            document_id="test-doc-123", stage="text_extraction", filename="test_document.pdf"
        ) as context:
            # Simulate document processing
            await asyncio.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "page_count": 5,
                    "word_count": 1500,
                    "text_length": 8000,
                    "detected_language": "en",
                }
            )

        assert context["document_id"] == "test-doc-123"
        assert context["stage"] == "text_extraction"
        assert context["filename"] == "test_document.pdf"

    @pytest.mark.asyncio
    async def test_document_processing_tracking_failure(self):
        """Test failed document processing tracking."""
        tracker = get_request_tracker()

        with pytest.raises(RuntimeError, match="Test document processing error"):
            with tracker.track_document_processing(
                document_id="test-doc-456", stage="chunking", filename="test_document2.pdf"
            ):
                # Simulate processing failure
                raise RuntimeError("Test document processing error")

    @pytest.mark.asyncio
    async def test_llm_request_tracking_success(self):
        """Test successful LLM provider request tracking."""
        tracker = get_request_tracker()

        async with tracker.track_llm_request(
            provider="test_provider", model="test-model", request_type="completion"
        ) as context:
            # Simulate LLM completion
            await asyncio.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "message_count": 3,
                    "completion_length": 200,
                    "tokens_used": 150,
                }
            )

        assert context["provider"] == "test_provider"
        assert context["model"] == "test-model"
        assert context["request_type"] == "completion"

    @pytest.mark.asyncio
    async def test_llm_request_tracking_failure(self):
        """Test failed LLM request tracking."""
        tracker = get_request_tracker()

        with pytest.raises(ConnectionError, match="Test LLM connection error"):
            async with tracker.track_llm_request(
                provider="test_provider", model="test-model", request_type="completion"
            ):
                # Simulate LLM failure
                raise ConnectionError("Test LLM connection error")

    def test_sync_embedding_tracking(self):
        """Test synchronous embedding tracking functionality."""
        tracker = get_request_tracker()

        with tracker.track_embedding_request(
            provider="sync_provider", model="sync-model", input_type="text", batch_size=1
        ) as context:
            # Simulate sync embedding generation
            time.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "input_size_chars": 75,
                    "output_dimensions": 768,
                }
            )

        assert context["provider"] == "sync_provider"
        assert context["model"] == "sync-model"

    def test_sync_vector_operation_tracking(self):
        """Test synchronous vector operation tracking."""
        tracker = get_request_tracker()

        with tracker.track_vector_operation(
            operation="delete", collection="test_collection"
        ) as context:
            # Simulate sync vector deletion
            time.sleep(0.01)

            # Add metrics
            context.update(
                {
                    "deleted_count": 5,
                }
            )

        assert context["operation"] == "delete"
        assert context["collection"] == "test_collection"

    def test_request_tracker_singleton(self):
        """Test that get_request_tracker returns the same instance."""
        tracker1 = get_request_tracker()
        tracker2 = get_request_tracker()

        assert tracker1 is tracker2

    def test_context_manager_properties(self):
        """Test that context managers provide expected request context."""
        tracker = get_request_tracker()

        with tracker.track_embedding_request(
            provider="test_provider", model="test-model"
        ) as context:
            # Test that all expected keys are present
            expected_keys = {
                "provider",
                "model",
                "input_type",
                "batch_size",
                "start_time",
                "request_id",
                "trace_id",
            }
            assert expected_keys.issubset(context.keys())

            # Test that values match what we passed
            assert context["provider"] == "test_provider"
            assert context["model"] == "test-model"
            assert context["input_type"] == "text"  # default value
            assert context["batch_size"] == 1  # default value
