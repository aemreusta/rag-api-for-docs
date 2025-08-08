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

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        """Increment a counter metric with optional value."""
        # Default implementation for backward compatibility
        for _ in range(int(value)):
            self.increment_counter(name, labels)


class PrometheusBackend(MetricsBackend):
    """Prometheus metrics backend using prometheus_client."""

    _instance = None
    _metrics_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization of metrics
        if self._metrics_initialized:
            return

        if importlib.util.find_spec("prometheus_client") is not None:
            try:
                from prometheus_client import REGISTRY, Counter, Gauge, Histogram

                self._create_vector_metrics(Histogram, Counter, Gauge, REGISTRY)
                self._create_cache_metrics(Counter, REGISTRY)

                self._metrics_available = True
                self._metrics_initialized = True
            except ImportError:
                self._metrics_available = False
        else:
            self._metrics_available = False

    def _create_vector_metrics(self, Histogram, Counter, Gauge, registry):
        """Create vector search metrics."""
        self.vector_search_duration = self._create_metric_safe(
            lambda: Histogram(
                "vector_search_duration_seconds",
                "Time spent on vector similarity search operations",
                labelnames=["status"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            ),
            "vector_search_duration_seconds",
            registry,
        )

        self.vector_search_requests = self._create_metric_safe(
            lambda: Counter(
                "vector_search_requests_total",
                "Total number of vector search requests",
                labelnames=["status"],
            ),
            "vector_search_requests_total",
            registry,
        )

        self.vector_search_recall = self._create_metric_safe(
            lambda: Gauge(
                "vector_search_recall", "Vector search recall accuracy", labelnames=["k"]
            ),
            "vector_search_recall",
            registry,
        )

    def _create_cache_metrics(self, Counter, registry):
        """Create cache metrics."""
        self.cache_hits = self._create_metric_safe(
            lambda: Counter(
                "cache_hits_total",
                "Total number of cache hits",
                labelnames=["backend"],
            ),
            "cache_hits_total",
            registry,
        )

        self.cache_misses = self._create_metric_safe(
            lambda: Counter(
                "cache_misses_total",
                "Total number of cache misses",
                labelnames=["backend"],
            ),
            "cache_misses_total",
            registry,
        )

        self.cache_evictions = self._create_metric_safe(
            lambda: Counter(
                "cache_evictions_total",
                "Total number of cache evictions",
                labelnames=["backend"],
            ),
            "cache_evictions_total",
            registry,
        )

        self.cache_errors = self._create_metric_safe(
            lambda: Counter(
                "cache_errors_total",
                "Total number of cache errors",
                labelnames=["backend", "operation"],
            ),
            "cache_errors_total",
            registry,
        )

    def _create_metric_safe(self, create_func, metric_name, registry):
        """Safely create a metric, handling duplicates by finding existing ones."""
        try:
            return create_func()
        except ValueError:
            # Metric already registered, find and reuse it
            for collector in registry._collector_to_names:
                if hasattr(collector, "_name") and getattr(collector, "_name", "") == metric_name:
                    return collector
            # If we can't find it, return None to avoid errors
            return None

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        if not self._metrics_available:
            return

        if (
            name == "vector_search_duration_seconds"
            and hasattr(self, "vector_search_duration")
            and self.vector_search_duration is not None
        ):
            label_values = labels or {}
            self.vector_search_duration.labels(**label_values).observe(value)

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        label_values = labels or {}

        if (
            name == "vector_search_requests_total"
            and hasattr(self, "vector_search_requests")
            and self.vector_search_requests is not None
        ):
            self.vector_search_requests.labels(**label_values).inc()
        elif (
            name == "cache_hits_total"
            and hasattr(self, "cache_hits")
            and self.cache_hits is not None
        ):
            self.cache_hits.labels(**label_values).inc()
        elif (
            name == "cache_misses_total"
            and hasattr(self, "cache_misses")
            and self.cache_misses is not None
        ):
            self.cache_misses.labels(**label_values).inc()
        elif (
            name == "cache_evictions_total"
            and hasattr(self, "cache_evictions")
            and self.cache_evictions is not None
        ):
            self.cache_evictions.labels(**label_values).inc()
        elif (
            name == "cache_errors_total"
            and hasattr(self, "cache_errors")
            and self.cache_errors is not None
        ):
            self.cache_errors.labels(**label_values).inc()

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        """Increment a counter with a specific value."""
        if not self._metrics_available:
            return

        label_values = labels or {}

        if (
            name == "cache_evictions_total"
            and hasattr(self, "cache_evictions")
            and self.cache_evictions is not None
        ):
            self.cache_evictions.labels(**label_values).inc(value)
        elif (
            name == "cache_hits_total"
            and hasattr(self, "cache_hits")
            and self.cache_hits is not None
        ):
            self.cache_hits.labels(**label_values).inc(value)
        elif (
            name == "cache_misses_total"
            and hasattr(self, "cache_misses")
            and self.cache_misses is not None
        ):
            self.cache_misses.labels(**label_values).inc(value)
        elif (
            name == "cache_errors_total"
            and hasattr(self, "cache_errors")
            and self.cache_errors is not None
        ):
            self.cache_errors.labels(**label_values).inc(value)
        else:
            # Fallback to regular counter increment
            for _ in range(int(value)):
                self.increment_counter(name, labels)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        if not self._metrics_available:
            return

        if (
            name == "vector_search_recall"
            and hasattr(self, "vector_search_recall")
            and self.vector_search_recall is not None
        ):
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

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        """Increment a counter with a specific value."""
        if not self._metrics_available:
            return

        if importlib.util.find_spec("datadog") is not None:
            try:
                import datadog

                tags = [f"{k}:{v}" for k, v in (labels or {}).items()]
                datadog.statsd.increment(name, value=value, tags=tags)
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

                # Vector search metrics
                self.vector_search_histogram = self.meter.create_histogram(
                    "vector_search_duration_seconds",
                    description="Time spent on vector similarity search operations",
                    unit="s",
                )
                self.vector_search_counter = self.meter.create_counter(
                    "vector_search_requests_total",
                    description="Total number of vector search requests",
                )

                # Cache metrics
                self.cache_hits_counter = self.meter.create_counter(
                    "cache_hits_total",
                    description="Total number of cache hits",
                )
                self.cache_misses_counter = self.meter.create_counter(
                    "cache_misses_total",
                    description="Total number of cache misses",
                )
                self.cache_evictions_counter = self.meter.create_counter(
                    "cache_evictions_total",
                    description="Total number of cache evictions",
                )
                self.cache_errors_counter = self.meter.create_counter(
                    "cache_errors_total",
                    description="Total number of cache errors",
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

        attributes = labels or {}
        if name == "vector_search_requests_total":
            self.vector_search_counter.add(1, attributes=attributes)
        elif name == "cache_hits_total":
            self.cache_hits_counter.add(1, attributes=attributes)
        elif name == "cache_misses_total":
            self.cache_misses_counter.add(1, attributes=attributes)
        elif name == "cache_evictions_total":
            self.cache_evictions_counter.add(1, attributes=attributes)
        elif name == "cache_errors_total":
            self.cache_errors_counter.add(1, attributes=attributes)

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        """Increment a counter with a specific value."""
        if not self._metrics_available:
            return

        attributes = labels or {}
        if name == "cache_hits_total":
            self.cache_hits_counter.add(value, attributes=attributes)
        elif name == "cache_misses_total":
            self.cache_misses_counter.add(value, attributes=attributes)
        elif name == "cache_evictions_total":
            self.cache_evictions_counter.add(value, attributes=attributes)
        elif name == "cache_errors_total":
            self.cache_errors_counter.add(value, attributes=attributes)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        # OpenTelemetry doesn't have explicit gauges in the same way
        # You would typically use an async gauge instrument for this
        pass


class NoOpBackend(MetricsBackend):
    """No-op metrics backend for testing/development."""

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        pass

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        pass

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        pass

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        pass


class VectorSearchMetrics:
    """Wrapper class for vector search and cache metrics."""

    def __init__(self, backend: MetricsBackend | None = None):
        self.backend = backend or self._auto_detect_backend()

    def _auto_detect_backend(self) -> MetricsBackend:
        """Auto-detect the best available metrics backend."""
        backend_name = getattr(settings, "METRICS_BACKEND", "auto").lower()

        if backend_name == "prometheus":
            return PrometheusBackend()
        elif backend_name == "datadog":
            return DataDogBackend()
        elif backend_name == "opentelemetry":
            return OpenTelemetryBackend()
        elif backend_name == "noop":
            return NoOpBackend()
        elif backend_name == "auto":
            # Auto-detect available backend
            if importlib.util.find_spec("prometheus_client") is not None:
                return PrometheusBackend()
            elif (
                importlib.util.find_spec("datadog") is not None
                and hasattr(settings, "DATADOG_API_KEY")
                and settings.DATADOG_API_KEY
            ):
                return DataDogBackend()
            elif importlib.util.find_spec("opentelemetry") is not None:
                return OpenTelemetryBackend()
            else:
                return NoOpBackend()
        else:
            # Default to no-op backend for unknown configurations
            return NoOpBackend()

    def time_vector_search(self, func):
        """Decorator to time vector search operations."""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.record_search_duration(duration, status="success")
                self.increment_search_requests(status="success")
                return result
            except Exception:
                duration = time.time() - start_time
                self.record_search_duration(duration, status="error")
                self.increment_search_requests(status="error")
                raise

        return wrapper

    def record_search_duration(self, duration: float, **labels) -> None:
        """Record vector search duration."""
        self.backend.record_histogram("vector_search_duration_seconds", duration, labels)

    def increment_search_requests(self, **labels) -> None:
        """Increment vector search request counter."""
        self.backend.increment_counter("vector_search_requests_total", labels)

    def record_search_recall(self, recall: float, k: int) -> None:
        """Record search recall accuracy."""
        self.backend.set_gauge("vector_search_recall", recall, {"k": str(k)})

    def increment_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        """Generic counter increment."""
        self.backend.increment_counter(name, labels)

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        """Generic counter increment with value."""
        self.backend.increment(name, labels, value)


# Create a default instance for backward compatibility
vector_metrics = VectorSearchMetrics()


def get_metrics_backend() -> VectorSearchMetrics:
    """Get the default metrics backend instance."""
    return vector_metrics
