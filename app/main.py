import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import chat, status
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


# Include API routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(status.router, prefix="/api/v1")
