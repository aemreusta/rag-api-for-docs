# Development Environment Setup

This document outlines our Docker-based development environment setup, which ensures consistency across all development machines and matches our production environment.

## Prerequisites

- Docker Desktop (latest version)
- Git
- A text editor (VS Code recommended)
- Make (for using Makefile shortcuts)

## 🚀 Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/chatbot-api-service.git
cd chatbot-api-service

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start everything
make up

# 4. Run initial data ingestion
make ingest

# 5. Verify system
curl http://localhost:8000/health
```

**Your development environment is now running!**

- **API**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs>
- **Langfuse**: <http://localhost:3000>

## Available Make Commands

### System Management

```bash
make up          # Start all services (recommended)
make down        # Stop all services
make logs        # View logs from all containers
make help        # Show all available commands
```

### Development

```bash
make shell       # Open bash shell in app container
make test        # Run all tests (7 tests should pass)
make test-cov    # Run tests with coverage report
make lint        # Check code quality with ruff
make format      # Format code with ruff
```

### Database Operations

```bash
make db-shell    # Open PostgreSQL shell
make ingest      # Run data ingestion (index PDF documents)
```

### Maintenance

```bash
make clean       # Remove all containers and volumes
make clean-pyc   # Remove Python cache files
```

## Environment Configuration

### Required Environment Variables

Edit `.env` with your actual API keys:

```env
# Database (automatically configured for Docker)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/app
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=app

# OpenRouter API (REQUIRED - Get from openrouter.ai)
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL_NAME=google/gemini-1.5-pro-latest

# Langfuse Observability (REQUIRED - Get from langfuse.com)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=http://langfuse:3000

# Application Security (REQUIRED - Generate strong random keys)
API_KEY=your_very_strong_api_key_here
ADMIN_API_KEY=your_admin_api_key_here

# Langfuse Service Configuration
NEXTAUTH_SECRET=your_strong_nextauth_secret_here
SALT=your_strong_salt_here
```

### Optional Settings

```env
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO
PDF_DOCUMENTS_DIR=pdf_documents
```

## Docker Services

### Current Stack

- **FastAPI Application**: Main API server with live reload
- **PostgreSQL + pgvector**: Database with vector similarity search
- **Redis**: In-memory cache for session management
- **Langfuse**: LLM observability and tracing

### Service Health Checks

All services include health checks and dependency management:

- PostgreSQL: `pg_isready` check
- Redis: `redis-cli ping` check
- App: Depends on healthy database and cache

### Volume Mappings

- `./app:/app/app`: Live code reloading
- `./pdf_documents:/app/pdf_documents`: Document storage
- `./tests:/app/tests`: Test directory
- Database and Redis data persisted in named volumes

## Development Workflow

### 1. Daily Development

```bash
# Start your dev session
make up

# View logs if needed
make logs

# Run tests after changes
make test

# Format and lint code
make format
make lint
```

### 2. Adding New Features

```bash
# Create feature branch
git checkout -b feat/your-feature-name

# Make changes, then test
make test
make lint

# Commit with conventional commits
git commit -m "feat(scope): add new feature"
```

### 3. Working with Dependencies

```bash
# Add new dependency to requirements.in
echo "new-package" >> requirements.in

# Compile new requirements.txt
make shell
pip-compile requirements.in
exit

# Rebuild container with new dependencies
make down
make up
```

### 4. Database Operations

```bash
# Access database shell
make db-shell

# Re-run data ingestion
make ingest

# Check database tables
make db-shell
\dt  # List tables
\d charity_policies  # Describe vector table
```

### 5. Debugging

```bash
# View all service logs
make logs

# View specific service logs
docker-compose logs app
docker-compose logs postgres

# Access app container for debugging
make shell
python  # Interactive Python
```

## Testing & Quality Assurance

### Running Tests

```bash
# Run all tests (should show 7/7 passing)
make test

# Run with coverage report
make test-cov

# Expected output: 35% coverage, all tests passing
```

### Code Quality

```bash
# Format code
make format

# Check linting (should pass cleanly)
make lint
```

### Pre-commit Hooks

The repository includes comprehensive pre-commit hooks:

- **Code formatting** with Ruff
- **Security scanning** with Gitleaks
- **Markdown linting**
- **YAML/TOML validation**
- **Conventional commit enforcement**

Install hooks: `pre-commit install`

## Troubleshooting

### Common Issues

1. **Services Won't Start**

   ```bash
   # Check what's using ports
   lsof -i :8000
   lsof -i :5432
   
   # Clean restart
   make down
   docker system prune -f
   make up
   ```

2. **Database Connection Issues**

   ```bash
   # Verify PostgreSQL is healthy
   docker-compose ps
   
   # Check database logs
   docker-compose logs postgres
   
   # Recreate database
   make down
   docker volume rm chatbot-api-service_postgres_data
   make up
   make ingest
   ```

3. **Ingestion Fails**

   ```bash
   # Check if pgvector extension is enabled
   make db-shell
   SELECT * FROM pg_extension WHERE extname = 'vector';
   
   # Re-enable if needed
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Permission Issues**

   ```bash
   # Fix file ownership
   sudo chown -R $USER:$USER .
   
   # Fix Docker permissions
   docker-compose down
   docker-compose up
   ```

### Health Check Commands

```bash
# API health
curl http://localhost:8000/health

# Database connection
make db-shell -c "SELECT 1;"

# Redis connection  
docker-compose exec redis redis-cli ping

# All services status
docker-compose ps
```

## Performance Tips

1. **Use Make Commands**: Always prefer `make up` over `docker-compose up -d`
2. **Volume Optimization**: Use named volumes for better performance
3. **Resource Limits**: Adjust Docker Desktop memory allocation if needed
4. **Cleanup Regularly**: Run `make clean` occasionally to free space

## IDE Integration

### VS Code Setup

Install recommended extensions:

- Python
- Docker
- GitLens
- Ruff (Python linting/formatting)

### Configuration

The repository includes:

- `.vscode/settings.json` for consistent formatting
- `pyproject.toml` for tool configuration
- `.pre-commit-config.yaml` for automated quality checks

## Deployment Preparation

The development environment matches production deployment:

- Same Docker images and configurations
- Environment-based configuration
- Health checks for orchestration
- Comprehensive logging

This ensures smooth deployment and debugging of production issues locally.
