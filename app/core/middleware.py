"""
FastAPI middleware for structured logging and request correlation.
"""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import get_logger, set_request_id, set_trace_id


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds structured logging with correlation IDs and request timing.
    """

    def __init__(self, app, mask_body_paths: list[str] = None):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application instance
            mask_body_paths: List of paths where request/response bodies should be masked
        """
        super().__init__(app)
        self.logger = get_logger(__name__)
        self.mask_body_paths = mask_body_paths or ["/api/v1/chat"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and add structured logging.
        """
        # Generate and set correlation IDs
        request_id = str(uuid.uuid4())
        set_request_id(request_id)

        # Set trace_id from headers if provided (for correlation with external systems)
        trace_id = request.headers.get("X-Trace-ID") or request.headers.get("traceparent")
        if trace_id:
            # Extract trace_id from W3C trace context if needed
            if trace_id.startswith("00-"):
                # W3C traceparent format: version-trace_id-parent_id-flags
                parts = trace_id.split("-")
                if len(parts) >= 2:
                    trace_id = parts[1]
            set_trace_id(trace_id)
        else:
            # Generate a new trace_id if not provided
            trace_id = str(uuid.uuid4())
            set_trace_id(trace_id)

        # Record request start time
        start_time = time.time()

        # Extract request information
        method = request.method
        path = str(request.url.path)
        query_params = str(request.url.query) if request.url.query else None
        user_agent = request.headers.get("user-agent")
        client_ip = self._get_client_ip(request)

        # Log request start
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            query_params=query_params,
            user_agent=user_agent,
            client_ip=client_ip,
            request_id=request_id,
            trace_id=trace_id,
        )

        # Process the request
        try:
            response = await call_next(request)
        except Exception as exc:
            # Log the exception
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds

            self.logger.error(
                "Request failed with exception",
                method=method,
                path=path,
                duration_ms=round(duration, 2),
                error=str(exc),
                error_type=type(exc).__name__,
                request_id=request_id,
                trace_id=trace_id,
                exc_info=True,
            )
            raise

        # Calculate request duration
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log response
        self.logger.info(
            "Request completed",
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration, 2),
            response_size=response.headers.get("content-length"),
            request_id=request_id,
            trace_id=trace_id,
        )

        # Add correlation headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract the client IP address from the request.

        Args:
            request: FastAPI request object

        Returns:
            Client IP address
        """
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, take the first one
            return forwarded_for.split(",")[0].strip()

        forwarded = request.headers.get("X-Forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"


class RateLimitLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware specifically for logging rate limiting events.
    """

    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log rate limiting events.
        """
        response = await call_next(request)

        # Log rate limiting events
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            client_ip = self._get_client_ip(request)

            self.logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=str(request.url.path),
                method=request.method,
                retry_after_seconds=retry_after,
                status_code=429,
            )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
