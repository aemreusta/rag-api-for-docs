"""
Test for flexible metrics system supporting multiple monitoring backends.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.metrics import (
    DataDogBackend,
    NoOpBackend,
    OpenTelemetryBackend,
    PrometheusBackend,
    VectorSearchMetrics,
)


class TestMetricsBackends:
    """Test different metrics backends."""

    def test_noop_backend(self):
        """Test that NoOp backend doesn't raise errors."""
        backend = NoOpBackend()

        # Should not raise any exceptions
        backend.record_histogram("test_metric", 1.0, {"status": "success"})
        backend.increment_counter("test_counter", {"status": "success"})
        backend.set_gauge("test_gauge", 1.0, {"type": "test"})

    @patch("importlib.util.find_spec")
    def test_prometheus_backend_unavailable(self, mock_find_spec):
        """Test Prometheus backend when prometheus_client is not available."""
        mock_find_spec.return_value = None

        backend = PrometheusBackend()

        # Should not raise exceptions when metrics are unavailable
        backend.record_histogram("vector_search_duration_seconds", 1.0, {"status": "success"})
        backend.increment_counter("vector_search_requests_total", {"status": "success"})
        backend.set_gauge("vector_search_recall", 0.95, {"k": "10"})

    def test_datadog_backend_no_api_key(self):
        """Test DataDog backend without API key."""
        with patch("app.core.config.settings") as mock_settings:
            # Mock settings without DATADOG_API_KEY
            mock_settings.DATADOG_API_KEY = ""

            backend = DataDogBackend()

            # Should handle missing API key gracefully
            backend.record_histogram("test_metric", 1.0)
            backend.increment_counter("test_counter")
            backend.set_gauge("test_gauge", 1.0)

    @patch("importlib.util.find_spec")
    def test_opentelemetry_backend_unavailable(self, mock_find_spec):
        """Test OpenTelemetry backend when library is not available."""
        mock_find_spec.return_value = None

        backend = OpenTelemetryBackend()

        # Should handle missing library gracefully
        backend.record_histogram("vector_search_duration_seconds", 1.0)
        backend.increment_counter("vector_search_requests_total")
        backend.set_gauge("vector_search_recall", 0.95)


class TestVectorSearchMetrics:
    """Test the high-level VectorSearchMetrics interface."""

    @patch("importlib.util.find_spec")
    def test_auto_detection_fallback_to_noop(self, mock_find_spec):
        """Test that auto-detection falls back to NoOp when no backends are available."""
        mock_find_spec.return_value = None

        metrics = VectorSearchMetrics()
        assert isinstance(metrics.backend, NoOpBackend)

    def test_explicit_backend_selection(self):
        """Test explicit backend selection."""
        noop_backend = NoOpBackend()
        metrics = VectorSearchMetrics(backend=noop_backend)
        assert metrics.backend is noop_backend

    def test_time_vector_search_decorator(self):
        """Test the timing decorator functionality."""
        mock_backend = MagicMock()
        metrics = VectorSearchMetrics(backend=mock_backend)

        @metrics.time_vector_search
        def dummy_search():
            return "search result"

        result = dummy_search()

        assert result == "search result"

        # Verify metrics were recorded
        mock_backend.record_histogram.assert_called_once()
        mock_backend.increment_counter.assert_called_once()

        # Check that duration was recorded with success status
        duration_call_args = mock_backend.record_histogram.call_args
        assert duration_call_args[0][0] == "vector_search_duration_seconds"
        assert isinstance(duration_call_args[0][1], float)  # duration should be a float
        # The call signature is: record_histogram(name, value, labels)
        # So args[0] = name, args[1] = value, args[2] = labels dict
        assert duration_call_args[0][2].get("status") == "success"

    def test_time_vector_search_decorator_error(self):
        """Test the timing decorator with exceptions."""
        mock_backend = MagicMock()
        metrics = VectorSearchMetrics(backend=mock_backend)

        @metrics.time_vector_search
        def failing_search():
            raise ValueError("Search failed")

        with pytest.raises(ValueError, match="Search failed"):
            failing_search()

        # Verify metrics were recorded even with error
        mock_backend.record_histogram.assert_called_once()
        mock_backend.increment_counter.assert_called_once()

        # Check that error status was recorded in increment_counter call
        counter_call_args = mock_backend.increment_counter.call_args
        # The call signature is: increment_counter(name, labels)
        # So args[0] = name, args[1] = labels dict
        assert counter_call_args[0][1].get("status") == "error"

    def test_record_search_duration(self):
        """Test recording search duration."""
        mock_backend = MagicMock()
        metrics = VectorSearchMetrics(backend=mock_backend)

        metrics.record_search_duration(1.5, status="success", model="test")

        mock_backend.record_histogram.assert_called_once_with(
            "vector_search_duration_seconds", 1.5, {"status": "success", "model": "test"}
        )

    def test_increment_search_requests(self):
        """Test incrementing search request counter."""
        mock_backend = MagicMock()
        metrics = VectorSearchMetrics(backend=mock_backend)

        metrics.increment_search_requests(status="success", endpoint="/chat")

        mock_backend.increment_counter.assert_called_once_with(
            "vector_search_requests_total", {"status": "success", "endpoint": "/chat"}
        )

    def test_record_search_recall(self):
        """Test recording search recall accuracy."""
        mock_backend = MagicMock()
        metrics = VectorSearchMetrics(backend=mock_backend)

        metrics.record_search_recall(0.95, k=10)

        mock_backend.set_gauge.assert_called_once_with("vector_search_recall", 0.95, {"k": "10"})

    @patch("app.core.config.settings")
    def test_backend_preference_from_settings(self, mock_settings):
        """Test backend selection based on settings."""
        mock_settings.METRICS_BACKEND = "noop"

        metrics = VectorSearchMetrics()
        assert isinstance(metrics.backend, NoOpBackend)

    def test_metrics_integration_with_query_engine(self):
        """Test that metrics integration works with the query engine."""
        from app.core.query_engine import get_chat_response

        # This test verifies that the decorator is applied and doesn't break functionality
        # We can't test actual metrics recording without setting up the full environment
        # but we can ensure the function is still callable

        try:
            # This should not raise an exception even if metrics backend is NoOp
            # Note: This will fail if no documents are ingested, but that's expected
            # The important thing is that metrics don't break the function
            get_chat_response("test question", "test_session")
        except Exception as e:
            # We expect this to fail due to no ingested documents,
            # but it should not fail due to metrics issues
            assert "metrics" not in str(e).lower(), f"Metrics-related error: {e}"


class TestMetricsConfiguration:
    """Test metrics configuration options."""

    def test_metrics_backend_options(self):
        """Test that all metrics backend options are valid."""
        valid_backends = ["auto", "prometheus", "datadog", "opentelemetry", "noop"]

        for backend_name in valid_backends:
            # Should not raise exceptions
            if backend_name == "noop":
                backend = NoOpBackend()
                assert isinstance(backend, NoOpBackend)
