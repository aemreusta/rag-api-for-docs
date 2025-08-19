"""
Comprehensive request tracking system with detailed metrics.

This module provides detailed tracking for:
- Embedding request tracking with provider performance
- Vector store performance metrics
- Document processing pipeline metrics
- LLM provider response times and success rates
- Request correlation and tracing
"""

import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any

from app.core.logging_config import get_logger, get_request_id, get_trace_id
from app.core.metrics import get_metrics_backend

logger = get_logger(__name__)
metrics = get_metrics_backend()


class RequestTracker:
    """Comprehensive request tracking with metrics and logging."""

    def __init__(self):
        self.metrics = get_metrics_backend()
        self.logger = get_logger(f"{__name__}.tracker")

    # Embedding Request Tracking
    @contextmanager
    def track_embedding_request(
        self, provider: str, model: str, input_type: str = "text", batch_size: int = 1
    ) -> Generator[dict[str, Any], None, None]:
        """Track embedding generation request with detailed metrics."""
        start_time = time.time()
        request_context = {
            "provider": provider,
            "model": model,
            "input_type": input_type,
            "batch_size": batch_size,
            "start_time": start_time,
            "request_id": get_request_id(),
            "trace_id": get_trace_id(),
        }

        self.logger.info(
            "Embedding request started",
            provider=provider,
            model=model,
            input_type=input_type,
            batch_size=batch_size,
            request_id=request_context["request_id"],
            trace_id=request_context["trace_id"],
        )

        try:
            yield request_context

            # Success metrics
            duration = time.time() - start_time
            self._record_embedding_success(request_context, duration)

        except Exception as e:
            # Error metrics
            duration = time.time() - start_time
            self._record_embedding_error(request_context, duration, e)
            raise

    def _record_embedding_success(self, context: dict[str, Any], duration: float) -> None:
        """Record successful embedding request metrics."""
        labels = {
            "provider": context["provider"],
            "model": context["model"],
            "input_type": context["input_type"],
            "status": "success",
        }

        # Record timing
        self.metrics.record_histogram("embedding_request_duration_seconds", duration, labels)

        # Record success count
        self.metrics.increment_counter("embedding_requests_total", labels)

        # Record batch size
        self.metrics.record_histogram(
            "embedding_batch_size",
            float(context["batch_size"]),
            {"provider": context["provider"], "model": context["model"]},
        )

        self.logger.info(
            "Embedding request completed successfully",
            provider=context["provider"],
            model=context["model"],
            duration_seconds=round(duration, 3),
            batch_size=context["batch_size"],
            request_id=context["request_id"],
            trace_id=context["trace_id"],
        )

    def _record_embedding_error(
        self, context: dict[str, Any], duration: float, error: Exception
    ) -> None:
        """Record failed embedding request metrics."""
        labels = {
            "provider": context["provider"],
            "model": context["model"],
            "input_type": context["input_type"],
            "status": "error",
            "error_type": type(error).__name__,
        }

        # Record timing
        self.metrics.record_histogram("embedding_request_duration_seconds", duration, labels)

        # Record error count
        self.metrics.increment_counter("embedding_requests_total", labels)

        self.logger.error(
            "Embedding request failed",
            provider=context["provider"],
            model=context["model"],
            duration_seconds=round(duration, 3),
            error=str(error),
            error_type=type(error).__name__,
            request_id=context["request_id"],
            trace_id=context["trace_id"],
            exc_info=True,
        )

    # Vector Store Performance Tracking
    @contextmanager
    def track_vector_operation(
        self, operation: str, collection: str = "default", query_type: str = "similarity"
    ) -> Generator[dict[str, Any], None, None]:
        """Track vector store operations with performance metrics."""
        start_time = time.time()
        request_context = {
            "operation": operation,
            "collection": collection,
            "query_type": query_type,
            "start_time": start_time,
            "request_id": get_request_id(),
            "trace_id": get_trace_id(),
        }

        self.logger.info(
            "Vector store operation started",
            operation=operation,
            collection=collection,
            query_type=query_type,
            request_id=request_context["request_id"],
            trace_id=request_context["trace_id"],
        )

        try:
            yield request_context

            # Success metrics
            duration = time.time() - start_time
            self._record_vector_success(request_context, duration)

        except Exception as e:
            # Error metrics
            duration = time.time() - start_time
            self._record_vector_error(request_context, duration, e)
            raise

    def _record_vector_success(self, context: dict[str, Any], duration: float) -> None:
        """Record successful vector store operation metrics."""
        labels = {
            "operation": context["operation"],
            "collection": context["collection"],
            "query_type": context["query_type"],
            "status": "success",
        }

        # Record timing
        self.metrics.record_histogram("vector_operation_duration_seconds", duration, labels)

        # Record success count
        self.metrics.increment_counter("vector_operations_total", labels)

        self.logger.info(
            "Vector store operation completed",
            operation=context["operation"],
            collection=context["collection"],
            duration_seconds=round(duration, 3),
            request_id=context["request_id"],
            trace_id=context["trace_id"],
        )

    def _record_vector_error(
        self, context: dict[str, Any], duration: float, error: Exception
    ) -> None:
        """Record failed vector store operation metrics."""
        labels = {
            "operation": context["operation"],
            "collection": context["collection"],
            "query_type": context["query_type"],
            "status": "error",
            "error_type": type(error).__name__,
        }

        # Record timing
        self.metrics.record_histogram("vector_operation_duration_seconds", duration, labels)

        # Record error count
        self.metrics.increment_counter("vector_operations_total", labels)

        self.logger.error(
            "Vector store operation failed",
            operation=context["operation"],
            collection=context["collection"],
            duration_seconds=round(duration, 3),
            error=str(error),
            error_type=type(error).__name__,
            request_id=context["request_id"],
            trace_id=context["trace_id"],
            exc_info=True,
        )

    # Document Processing Pipeline Tracking
    @contextmanager
    def track_document_processing(
        self, document_id: str, stage: str, filename: str | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """Track document processing pipeline stages."""
        start_time = time.time()
        request_context = {
            "document_id": document_id,
            "stage": stage,
            "filename": filename,
            "start_time": start_time,
            "request_id": get_request_id(),
            "trace_id": get_trace_id(),
        }

        self.logger.info(
            "Document processing stage started",
            document_id=document_id,
            stage=stage,
            filename=filename,
            request_id=request_context["request_id"],
            trace_id=request_context["trace_id"],
        )

        try:
            yield request_context

            # Success metrics
            duration = time.time() - start_time
            self._record_document_success(request_context, duration)

        except Exception as e:
            # Error metrics
            duration = time.time() - start_time
            self._record_document_error(request_context, duration, e)
            raise

    def _record_document_success(self, context: dict[str, Any], duration: float) -> None:
        """Record successful document processing stage metrics."""
        labels = {"stage": context["stage"], "status": "success"}

        # Record timing
        self.metrics.record_histogram("document_processing_duration_seconds", duration, labels)

        # Record success count
        self.metrics.increment_counter("document_processing_stages_total", labels)

        self.logger.info(
            "Document processing stage completed",
            document_id=context["document_id"],
            stage=context["stage"],
            filename=context["filename"],
            duration_seconds=round(duration, 3),
            request_id=context["request_id"],
            trace_id=context["trace_id"],
        )

    def _record_document_error(
        self, context: dict[str, Any], duration: float, error: Exception
    ) -> None:
        """Record failed document processing stage metrics."""
        labels = {"stage": context["stage"], "status": "error", "error_type": type(error).__name__}

        # Record timing
        self.metrics.record_histogram("document_processing_duration_seconds", duration, labels)

        # Record error count
        self.metrics.increment_counter("document_processing_stages_total", labels)

        self.logger.error(
            "Document processing stage failed",
            document_id=context["document_id"],
            stage=context["stage"],
            filename=context["filename"],
            duration_seconds=round(duration, 3),
            error=str(error),
            error_type=type(error).__name__,
            request_id=context["request_id"],
            trace_id=context["trace_id"],
            exc_info=True,
        )

    # LLM Provider Response Tracking
    @asynccontextmanager
    async def track_llm_request(
        self, provider: str, model: str, request_type: str = "completion"
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Track LLM provider requests with response times and success rates."""
        start_time = time.time()
        request_context = {
            "provider": provider,
            "model": model,
            "request_type": request_type,
            "start_time": start_time,
            "request_id": get_request_id(),
            "trace_id": get_trace_id(),
        }

        self.logger.info(
            "LLM request started",
            provider=provider,
            model=model,
            request_type=request_type,
            request_id=request_context["request_id"],
            trace_id=request_context["trace_id"],
        )

        try:
            yield request_context

            # Success metrics
            duration = time.time() - start_time
            self._record_llm_success(request_context, duration)

        except Exception as e:
            # Error metrics
            duration = time.time() - start_time
            self._record_llm_error(request_context, duration, e)
            raise

    def _record_llm_success(self, context: dict[str, Any], duration: float) -> None:
        """Record successful LLM request metrics."""
        labels = {
            "provider": context["provider"],
            "model": context["model"],
            "request_type": context["request_type"],
            "status": "success",
        }

        # Record timing
        self.metrics.record_histogram("llm_request_duration_seconds", duration, labels)

        # Record success count
        self.metrics.increment_counter("llm_requests_total", labels)

        self.logger.info(
            "LLM request completed successfully",
            provider=context["provider"],
            model=context["model"],
            request_type=context["request_type"],
            duration_seconds=round(duration, 3),
            request_id=context["request_id"],
            trace_id=context["trace_id"],
        )

    def _record_llm_error(self, context: dict[str, Any], duration: float, error: Exception) -> None:
        """Record failed LLM request metrics."""
        labels = {
            "provider": context["provider"],
            "model": context["model"],
            "request_type": context["request_type"],
            "status": "error",
            "error_type": type(error).__name__,
        }

        # Record timing
        self.metrics.record_histogram("llm_request_duration_seconds", duration, labels)

        # Record error count
        self.metrics.increment_counter("llm_requests_total", labels)

        self.logger.error(
            "LLM request failed",
            provider=context["provider"],
            model=context["model"],
            request_type=context["request_type"],
            duration_seconds=round(duration, 3),
            error=str(error),
            error_type=type(error).__name__,
            request_id=context["request_id"],
            trace_id=context["trace_id"],
            exc_info=True,
        )


# Global request tracker instance
request_tracker = RequestTracker()


def get_request_tracker() -> RequestTracker:
    """Get the global request tracker instance."""
    return request_tracker


# Decorator functions for easy usage


def track_embedding_request(provider: str, model: str, input_type: str = "text"):
    """Decorator to track embedding requests."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            batch_size = kwargs.get("batch_size", 1)
            with request_tracker.track_embedding_request(provider, model, input_type, batch_size):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_vector_operation(operation: str, collection: str = "default"):
    """Decorator to track vector store operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with request_tracker.track_vector_operation(operation, collection):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_document_processing(stage: str):
    """Decorator to track document processing stages."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract document_id from args/kwargs
            document_id = kwargs.get("document_id") or (args[0] if args else "unknown")
            filename = kwargs.get("filename")

            with request_tracker.track_document_processing(document_id, stage, filename):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_llm_request(provider: str, model: str, request_type: str = "completion"):
    """Decorator to track LLM requests."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with request_tracker.track_llm_request(provider, model, request_type):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle this differently
            # Since we can't use async context manager, we'll track manually
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                labels = {
                    "provider": provider,
                    "model": model,
                    "request_type": request_type,
                    "status": "success",
                }
                request_tracker.metrics.record_histogram(
                    "llm_request_duration_seconds", duration, labels
                )
                request_tracker.metrics.increment_counter("llm_requests_total", labels)

                return result
            except Exception as e:
                duration = time.time() - start_time

                labels = {
                    "provider": provider,
                    "model": model,
                    "request_type": request_type,
                    "status": "error",
                    "error_type": type(e).__name__,
                }
                request_tracker.metrics.record_histogram(
                    "llm_request_duration_seconds", duration, labels
                )
                request_tracker.metrics.increment_counter("llm_requests_total", labels)
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
