# Development Environment Setup

This document outlines our Docker-based development environment setup, which ensures consistency across all development machines and matches our production environment.

## Prerequisites

- Docker Desktop (latest version)
- Git
- A text editor (VS Code recommended)
- Make (optional, for using Makefile shortcuts)

## Environment Setup Options

We provide three setup options, with the Docker-based setup being our recommended approach:

### Option 1: Full Docker Setup (Recommended)

This is our standard development setup. It provides:

- Maximum consistency with production
- One-command setup process
- Clean, isolated environment
- Simplified CI/CD pipeline integration

#### Setup Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/chatbot-api-service.git
   cd chatbot-api-service
   ```

2. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

3. Configure your `.env` file with your settings:

   ```env
   # API Settings
   API_V1_STR=/api/v1
   PROJECT_NAME=Charity Policy AI Chatbot
   BACKEND_CORS_ORIGINS=["http://localhost:8000", "http://localhost:3000"]

   # OpenRouter API
   OPENROUTER_API_KEY=your-api-key
   
   # PostgreSQL
   POSTGRES_SERVER=postgres
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=app
   
   # Redis
   REDIS_HOST=redis
   REDIS_PORT=6379
   ```

4. Start the development environment:

   ```bash
   docker-compose up --build
   ```

The API will be available at `http://localhost:8000`.

#### Development Workflow

1. **Code Changes**: Edit code in your local IDE. Changes are automatically synced to the container and trigger a reload.

2. **Dependency Updates**: If you add new dependencies:

   ```bash
   # Update requirements.in
   docker-compose exec app pip-compile requirements.in
   docker-compose up --build  # Rebuild with new dependencies
   ```

3. **Database Migrations**: Run through Docker:

   ```bash
   docker-compose exec app alembic revision --autogenerate -m "description"
   docker-compose exec app alembic upgrade head
   ```

4. **Running Tests**: Execute in the container:

   ```bash
   docker-compose exec app pytest
   ```

### Option 2: Local Python with Docker Services

For developers who prefer running the Python application locally:

1. Create a conda environment:

   ```bash
   conda create -n chatbot python=3.11
   conda activate chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start supporting services:

   ```bash
   docker-compose up postgres redis
   ```

4. Run the application:

   ```bash
   uvicorn app.main:app --reload
   ```

### Option 3: Fully Local Setup (Not Recommended)

Only for situations where Docker cannot be used. Requires manual setup of PostgreSQL and Redis.

## Docker Configuration Details

### Key Components

Our `docker-compose.yml` sets up:

- FastAPI application with live reload
- PostgreSQL with pgvector
- Redis for rate limiting
- Volumes for code and data persistence

### Volume Mappings

- `./app:/app/app`: Live code reloading
- `./pdf_documents:/app/pdf_documents`: PDF document storage
- `postgres_data:/var/lib/postgresql/data`: Database persistence
- `redis_data:/data`: Redis persistence

### Network Configuration

- FastAPI: Port 8000
- PostgreSQL: Port 5432
- Redis: Port 6379

## Best Practices

1. **Always Use Docker Compose**
   - Ensures consistent environment
   - Manages service dependencies
   - Simplifies commands

2. **Environment Variables**
   - Never commit `.env` files
   - Use `.env.example` as a template
   - Document all environment variables

3. **Database Migrations**
   - Always run migrations through Docker
   - Keep migration messages descriptive
   - Test migrations locally before committing

4. **Dependency Management**
   - Use `requirements.in` for direct dependencies
   - Generate `requirements.txt` with `pip-compile`
   - Rebuild containers after dependency changes

5. **Troubleshooting**
   - Check logs: `docker-compose logs -f [service]`
   - Rebuild: `docker-compose up --build`
   - Clean start: `docker-compose down -v && docker-compose up --build`

## Common Issues and Solutions

1. **Port Conflicts**

   ```bash
   # Error: port already in use
   sudo lsof -i :8000  # Find process using port
   kill -9 <PID>       # Kill process if needed
   ```

2. **Permission Issues**

   ```bash
   # Fix ownership of generated files
   sudo chown -R $USER:$USER .
   ```

3. **Container Won't Start**

   ```bash
   # Check logs
   docker-compose logs app
   # Rebuild
   docker-compose up --build
   ```

## IDE Setup

### VS Code Configuration

Recommended extensions:

- Python
- Docker
- Remote Containers
- GitLens

Settings for optimal development:

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## Continuous Integration

Our Docker setup integrates seamlessly with CI/CD:

```yaml
# Example GitHub Actions workflow
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and test
        run: |
          docker-compose -f docker-compose.yml up -d
          docker-compose exec -T app pytest
```
