import contextlib
import importlib.util

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.api.v1 import chat, docs, status
from app.api.v1 import models as models_api
from app.core.config import settings
from app.core.cors import get_cors_config
from app.core.logging_config import get_logger, setup_logging
from app.core.middleware import RateLimitLoggingMiddleware, StructuredLoggingMiddleware
from app.core.redis import redis_client

# Setup logging before anything else
setup_logging()
logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    """
    logger.info("Application startup", environment=settings.ENVIRONMENT, debug=settings.DEBUG)
    try:
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning("Redis connection failed", error=str(e))
    yield
    logger.info("Application shutdown")
    await redis_client.close()


app = FastAPI(
    title="Charity Policy AI Chatbot API",
    description="API for answering questions about charity policies.",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Add structured logging middleware (order matters - this should be first)
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(RateLimitLoggingMiddleware)

# Configure CORS using centralized configuration
cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)


@app.get("/health", status_code=200, tags=["Health"])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    logger.info("Health check requested")
    return {"status": "ok", "environment": settings.ENVIRONMENT, "version": app.version}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus metrics endpoint.
    Only enabled if PROMETHEUS_ENABLED is true and prometheus_client is available.
    """
    if not settings.PROMETHEUS_ENABLED:
        logger.warning("Metrics endpoint requested but PROMETHEUS_ENABLED is false")
        raise HTTPException(status_code=404, detail="Metrics endpoint not enabled")

    if importlib.util.find_spec("prometheus_client") is None:
        logger.warning("Metrics endpoint requested but prometheus_client not available")
        raise HTTPException(status_code=404, detail="Prometheus client not available")

    try:
        from prometheus_client import REGISTRY, generate_latest

        logger.info("Metrics endpoint requested")
        metrics_data = generate_latest(REGISTRY)
        return Response(content=metrics_data, media_type="text/plain; version=0.0.4; charset=utf-8")
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate metrics") from e


# Include API routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(status.router, prefix="/api/v1")
app.include_router(models_api.router, prefix="/api/v1")
app.include_router(docs.router, prefix="/api/v1")
