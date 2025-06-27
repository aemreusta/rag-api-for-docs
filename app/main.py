import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import chat
from app.core.config import settings
from app.core.redis import redis_client


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    """
    await redis_client.ping()
    yield
    await redis_client.close()


app = FastAPI(
    title="Charity Policy AI Chatbot API",
    description="API for answering questions about charity policies.",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=200, tags=["Health"])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "environment": settings.ENVIRONMENT, "version": app.version}


# Include API routers
app.include_router(chat.router, prefix="/api/v1")
