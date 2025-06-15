from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import chat

app = FastAPI(
    title="Chatbot API Service",
    description="A FastAPI service for document-based chatbot using LlamaIndex",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API Service"}
