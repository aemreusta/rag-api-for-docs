FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip-tools for dependency management
RUN pip install --no-cache-dir pip-tools

# Copy requirements first to leverage Docker cache
COPY requirements.in requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install development tools
RUN pip install --no-cache-dir \
    black \
    ruff \
    pytest \
    pytest-cov \
    pytest-asyncio \
    ipython

# Create necessary directories
RUN mkdir -p pdf_documents

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development command (used in docker-compose)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 