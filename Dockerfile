# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --upgrade pip uv

# Copy *both* lock-files before install to keep the cache
COPY requirements-dev.txt ./

# Use uv to install dependencies from the requirements-dev.txt file into system interpreter,
# caching wheels and artifacts to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache \
    uv pip sync --system requirements-dev.txt

# Create necessary directories
RUN mkdir -p pdf_documents

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command for production using uv package manager
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development command (used in docker-compose)
# CMD ["uv", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
