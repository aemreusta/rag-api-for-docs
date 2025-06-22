from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

app = FastAPI(
    title="Charity Policy AI Chatbot API",
    description="API for answering questions about charity policies.",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
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


# Placeholder for future routers
# from app.api.v1 import chat, admin
# app.include_router(chat.router, prefix="/api/v1")
# app.include_router(admin.router, prefix="/api/v1/admin")
