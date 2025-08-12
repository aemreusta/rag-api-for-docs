# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

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

FROM base AS deps-core
COPY requirements/requirements-core.in requirements-core.in
RUN --mount=type=cache,target=/root/.cache \
    uv pip compile requirements-core.in -o requirements-core.txt && \
    uv pip sync --system requirements-core.txt

FROM base AS deps-dev
COPY requirements/requirements-dev.txt ./
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
