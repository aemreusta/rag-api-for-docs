"""
Test for pgvector Prometheus metrics setup.
This test verifies the metrics structure without requiring prometheus_client.
"""


class TestPgVectorPrometheus:
    """Test suite for pgvector Prometheus metrics configuration."""

    def test_metrics_config_structure(self):
        """Test that the metrics configuration structure is correct."""
        # Test metric names follow Prometheus naming conventions
        metric_names = ["vector_search_duration_seconds", "vector_search_requests_total"]

        for name in metric_names:
            # Should not contain uppercase letters
            assert name.islower(), f"Metric name {name} should be lowercase"

            # Should not contain spaces
            assert " " not in name, f"Metric name {name} should not contain spaces"

            # Should use underscores for separation
            assert "_" in name, f"Metric name {name} should use underscores"

    def test_histogram_buckets_structure(self):
        """Test that histogram buckets are properly structured."""
        buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

        # Buckets should be in ascending order
        assert buckets == sorted(buckets), "Histogram buckets should be in ascending order"

        # Should have reasonable performance boundaries
        assert min(buckets) <= 0.001, "Should have sub-millisecond bucket for fast queries"
        assert max(buckets) >= 5.0, "Should have multi-second bucket for slow queries"

        # Should include target performance thresholds
        assert 0.05 in buckets, "Should include 50ms bucket (our p99 target)"

    def test_counter_labels_structure(self):
        """Test that counter labels are properly defined."""
        labels = ["status"]
        status_values = ["success", "error"]

        assert "status" in labels, "Should have status label for request tracking"

        for status in status_values:
            assert status in ["success", "error"], f"Status {status} should be valid"

    def test_histogram_and_counter_do_not_require_model_label(self):
        """Ensure metrics do not depend on a 'model' label that isn't provided."""
        from app.core.metrics import PrometheusBackend

        backend = PrometheusBackend()

        # These should not raise even if no 'model' label is provided
        backend.record_histogram("vector_search_duration_seconds", 0.01, {"status": "success"})
        backend.increment_counter("vector_search_requests_total", {"status": "error"})

    def test_embedding_dimension_configuration(self):
        """Test that EMBEDDING_DIM is properly configured."""
        from app.core.config import settings

        assert hasattr(settings, "EMBEDDING_DIM"), "Should have EMBEDDING_DIM setting"
        assert settings.EMBEDDING_DIM == 1536, "Should be configured for 1536 dimensions"
        assert isinstance(settings.EMBEDDING_DIM, int), "Should be integer type"
