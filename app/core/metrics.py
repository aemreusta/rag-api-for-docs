"""
Flexible metrics abstraction for monitoring vector search performance.
Supports multiple monitoring backends: Prometheus, DataDog, New Relic, OpenTelemetry, etc.
"""

import importlib.util
import time
from abc import ABC, abstractmethod

from app.core.config import settings


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram/timing metric."""
        pass

    @abstractmethod
    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric value."""
        pass


class PrometheusBackend(MetricsBackend):
    """Prometheus metrics backend using prometheus_client."""

    def __init__(self):
        if importlib.util.find_spec("prometheus_client") is not None:
            try:
                from prometheus_client import Counter, Gauge, Histogram

                self.vector_search_duration = Histogram(
                    "vector_search_duration_seconds",
                    "Time spent on vector similarity search operations",
                    labelnames=["status", "model"],
                    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                )

                self.vector_search_requests = Counter(
                    "vector_search_requests_total",
                    "Total number of vector search requests",
                    labelnames=["status", "model"],
                )

                self.vector_search_recall = Gauge(
                    "vector_search_recall", "Vector search recall accuracy", labelnames=["k"]
                )

                self._metrics_available = True
            except ImportError:
                self._metrics_available = False
        else:
            self._metrics_available = False

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        if not self._metrics_available:
            return

        if name == "vector_search_duration_seconds":
            label_values = labels or {}
            self.vector_search_duration.labels(**label_values).observe(value)

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        if name == "vector_search_requests_total":
            label_values = labels or {}
            self.vector_search_requests.labels(**label_values).inc()

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        if name == "vector_search_recall":
            label_values = labels or {}
            self.vector_search_recall.labels(**label_values).set(value)


class DataDogBackend(MetricsBackend):
    """DataDog metrics backend using datadog library."""

    def __init__(self):
        if importlib.util.find_spec("datadog") is not None:
            try:
                import datadog

                # Initialize DataDog if API key is available
                if hasattr(settings, "DATADOG_API_KEY") and settings.DATADOG_API_KEY:
                    datadog.initialize(api_key=settings.DATADOG_API_KEY)
                    self._metrics_available = True
                else:
                    self._metrics_available = False
            except ImportError:
                self._metrics_available = False
        else:
            self._metrics_available = False

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        if not self._metrics_available:
            return

        if importlib.util.find_spec("datadog") is not None:
            try:
                import datadog

                tags = [f"{k}:{v}" for k, v in (labels or {}).items()]
                datadog.statsd.histogram(name, value, tags=tags)
            except ImportError:
                pass

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        if importlib.util.find_spec("datadog") is not None:
            try:
                import datadog

                tags = [f"{k}:{v}" for k, v in (labels or {}).items()]
                datadog.statsd.increment(name, tags=tags)
            except ImportError:
                pass

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        if importlib.util.find_spec("datadog") is not None:
            try:
                import datadog

                tags = [f"{k}:{v}" for k, v in (labels or {}).items()]
                datadog.statsd.gauge(name, value, tags=tags)
            except ImportError:
                pass


class OpenTelemetryBackend(MetricsBackend):
    """OpenTelemetry metrics backend."""

    def __init__(self):
        if importlib.util.find_spec("opentelemetry") is not None:
            try:
                from opentelemetry import metrics
                from opentelemetry.sdk.metrics import MeterProvider
                from opentelemetry.sdk.metrics.export import (
                    ConsoleMetricExporter,
                    PeriodicExportingMetricReader,
                )

                # Set up basic OpenTelemetry metrics
                reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
                provider = MeterProvider(metric_readers=[reader])
                metrics.set_meter_provider(provider)

                self.meter = metrics.get_meter(__name__)
                self.vector_search_histogram = self.meter.create_histogram(
                    "vector_search_duration_seconds",
                    description="Time spent on vector similarity search operations",
                    unit="s",
                )
                self.vector_search_counter = self.meter.create_counter(
                    "vector_search_requests_total",
                    description="Total number of vector search requests",
                )
                self._metrics_available = True

            except ImportError:
                self._metrics_available = False
        else:
            self._metrics_available = False

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        if not self._metrics_available:
            return

        if name == "vector_search_duration_seconds":
            self.vector_search_histogram.record(value, attributes=labels or {})

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        if name == "vector_search_requests_total":
            self.vector_search_counter.add(1, attributes=labels or {})

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        # OpenTelemetry doesn't have explicit gauges in the same way
        pass


class NoOpBackend(MetricsBackend):
    """No-operation metrics backend that does nothing."""

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        pass

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        pass

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        pass


class VectorSearchMetrics:
    """High-level interface for vector search metrics."""

    def __init__(self, backend: MetricsBackend | None = None):
        if backend is None:
            # Auto-detect available backend
            backend = self._auto_detect_backend()
        self.backend = backend

    def _auto_detect_backend(self) -> MetricsBackend:
        """Auto-detect the best available metrics backend."""
        # Check for environment variable preference
        backend_preference = getattr(settings, "METRICS_BACKEND", "auto").lower()

        if backend_preference == "prometheus":
            return PrometheusBackend()
        elif backend_preference == "datadog":
            return DataDogBackend()
        elif backend_preference == "opentelemetry":
            return OpenTelemetryBackend()
        elif backend_preference == "noop":
            return NoOpBackend()

        # Auto-detection order: Prometheus -> DataDog -> OpenTelemetry -> NoOp
        if importlib.util.find_spec("prometheus_client") is not None:
            return PrometheusBackend()

        if importlib.util.find_spec("datadog") is not None:
            if hasattr(settings, "DATADOG_API_KEY") and settings.DATADOG_API_KEY:
                return DataDogBackend()

        if importlib.util.find_spec("opentelemetry") is not None:
            return OpenTelemetryBackend()

        # Fallback to no-op
        return NoOpBackend()

    def time_vector_search(self, func):
        """Decorator to time vector search operations."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start_time
                self.record_search_duration(duration, status=status)
                self.increment_search_requests(status=status)

        return wrapper

    def record_search_duration(self, duration: float, **labels) -> None:
        """Record vector search duration."""
        self.backend.record_histogram("vector_search_duration_seconds", duration, labels)

    def increment_search_requests(self, **labels) -> None:
        """Increment vector search request counter."""
        self.backend.increment_counter("vector_search_requests_total", labels)

    def record_search_recall(self, recall: float, k: int) -> None:
        """Record vector search recall accuracy."""
        self.backend.set_gauge("vector_search_recall", recall, {"k": str(k)})


# Global metrics instance
vector_metrics = VectorSearchMetrics()
