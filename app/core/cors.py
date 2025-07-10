"""CORS middleware configuration and helpers."""

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_cors_config() -> dict:
    """Get CORS configuration dict for adding middleware to FastAPI app.

    Returns:
        dict: CORS configuration parameters
    """
    origins = settings.cors_origins
    credentials = settings.cors_allow_credentials_safe

    # Parse methods and headers into lists
    methods = [method.strip() for method in settings.CORS_ALLOW_METHODS.split(",")]
    headers = [header.strip() for header in settings.CORS_ALLOW_HEADERS.split(",")]

    # Log CORS configuration for debugging
    if settings.DEBUG:
        logger.info(
            "Configuring CORS middleware",
            extra={
                "origins": origins,
                "allow_credentials": credentials,
                "allow_methods": methods,
                "allow_headers": headers,
                "max_age": settings.CORS_MAX_AGE,
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG,
            },
        )

        # Warn about wildcard in production-like environments
        if "*" in origins and settings.ENVIRONMENT != "development":
            logger.warning(
                "CORS wildcard (*) detected in non-development environment",
                extra={"environment": settings.ENVIRONMENT, "origins": origins},
            )

    return {
        "allow_origins": origins,
        "allow_credentials": credentials,
        "allow_methods": methods,
        "allow_headers": headers,
        "max_age": settings.CORS_MAX_AGE,
    }


def log_cors_rejection(origin: str, method: str, path: str) -> None:
    """Log CORS rejection for debugging purposes.

    Args:
        origin: The origin that was rejected
        method: HTTP method of the request
        path: Request path
    """
    if settings.DEBUG:
        logger.warning(
            "CORS request rejected",
            extra={
                "origin": origin,
                "method": method,
                "path": path,
                "allowed_origins": settings.cors_origins,
                "environment": settings.ENVIRONMENT,
            },
        )
