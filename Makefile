# Chatbot API Service - Professional Development Makefile
# Comprehensive build, test, and deployment automation

.DEFAULT_GOAL := help
.PHONY: help system-check env-check build up down clean rebuild logs shell test migrate deps deps-update lint format

# Colors for output
GREEN := \033[32m
YELLOW := \033[33m  
RED := \033[31m
BLUE := \033[34m
NC := \033[0m # No Color

# Configuration
DOCKER_BUILDKIT := 1
COMPOSE_PROJECT_NAME := chatbot-api-service
COMPOSE_FILE := docker-compose.yml
REQUIREMENTS_DIR := requirements
DATABASE_URL := postgresql+psycopg2://postgres:postgres@localhost:15432/app
TEST_DATABASE_URL := postgresql+psycopg2://postgres:postgres@localhost:15432/test_app

# Export environment variables for sub-processes
export DOCKER_BUILDKIT
export COMPOSE_PROJECT_NAME

##@ Help

help: ## Display this help message
	@echo "$(BLUE)Chatbot API Service Development Environment$(NC)"
	@echo "$(BLUE)=============================================$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@awk '/^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ System Management

system-check: ## Comprehensive system health check
	@echo "$(BLUE)🔍 Running System Health Check...$(NC)"
	@./scripts/system_check.sh

env-check: ## Validate environment configuration
	@echo "$(BLUE)🔧 Checking Environment Configuration...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  .env file not found, copying from .env.example$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✅ Created .env file from template$(NC)"; \
	fi
	@echo "$(GREEN)✅ Environment file exists$(NC)"

##@ Build & Deploy

build: env-check ## Build all services with BuildKit optimization
	@echo "$(BLUE)🏗️  Building services with Docker BuildKit...$(NC)"
	DOCKER_BUILDKIT=1 docker compose build --parallel

rebuild: clean build ## Clean rebuild all services

build-embedding: ## Build only the embedding service
	@echo "$(BLUE)🧠 Building HuggingFace Embedding Service...$(NC)"
	DOCKER_BUILDKIT=1 docker compose build embedding-service

##@ Container Management

up: build ## Start all services with health checks
	@echo "$(BLUE)🚀 Starting all services...$(NC)"
	docker compose up -d
	@echo "$(BLUE)⏳ Waiting for services to be healthy...$(NC)"
	@$(MAKE) wait-for-services

down: ## Stop all services gracefully
	@echo "$(BLUE)⏹️  Stopping all services...$(NC)"
	docker compose down

clean: ## Remove all containers, volumes, and networks
	@echo "$(BLUE)🧹 Cleaning up Docker resources...$(NC)"
	docker compose down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

logs: ## View logs from core services
	docker compose logs -f app postgres redis embedding-service

logs-all: ## View logs from all services  
	docker compose logs -f

shell: ## Open shell in app container
	docker compose exec app /bin/bash

##@ Database & Migrations

migrate: ## Run database migrations with automatic retry
	@echo "$(BLUE)🗃️  Running database migrations...$(NC)"
	@./scripts/run_migrations.sh

migrate-create: ## Create new migration (usage: make migrate-create name="your_migration_name")
	@if [ -z "$(name)" ]; then \
		echo "$(RED)❌ Please provide migration name: make migrate-create name=\"your_migration_name\"$(NC)"; \
		exit 1; \
	fi
	docker compose exec app alembic revision --autogenerate -m "$(name)"

db-shell: ## Open database shell
	docker compose exec postgres psql -U postgres -d app

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)⚠️  This will destroy all database data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker compose down postgres; \
		docker volume rm $(COMPOSE_PROJECT_NAME)_postgres_data 2>/dev/null || true; \
		docker compose up -d postgres; \
		sleep 5; \
		$(MAKE) migrate; \
	fi

##@ Testing

test: ## Run all tests with proper database setup
	@echo "$(BLUE)🧪 Running test suite...$(NC)"
	@$(MAKE) test-db-setup
	PYTHONPATH=. DATABASE_URL=$(TEST_DATABASE_URL) pytest -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)🔬 Running unit tests...$(NC)"
	PYTHONPATH=. pytest -v -m "not integration"

test-integration: ## Run integration tests
	@echo "$(BLUE)🔗 Running integration tests...$(NC)"
	@$(MAKE) test-db-setup
	PYTHONPATH=. DATABASE_URL=$(TEST_DATABASE_URL) pytest -v -m "integration"

test-embeddings: ## Test embedding system specifically
	@echo "$(BLUE)🧠 Testing embedding system...$(NC)"
	@$(MAKE) test-db-setup
	PYTHONPATH=. DATABASE_URL=$(TEST_DATABASE_URL) pytest tests/test_embeddings_config.py tests/test_incremental_processor.py -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)📊 Running tests with coverage...$(NC)"
	@$(MAKE) test-db-setup
	PYTHONPATH=. DATABASE_URL=$(TEST_DATABASE_URL) pytest --cov=app --cov-report=html --cov-report=term-missing

test-db-setup: ## Setup test database
	@echo "$(BLUE)🗃️  Setting up test database...$(NC)"
	@docker compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS test_app;" 2>/dev/null || true
	@docker compose exec postgres psql -U postgres -c "CREATE DATABASE test_app;" 2>/dev/null || true
	@DATABASE_URL=$(TEST_DATABASE_URL) alembic upgrade head

##@ Code Quality

lint: ## Run code linting
	@echo "$(BLUE)🔍 Running linting...$(NC)"
	ruff check .

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)🔧 Fixing linting issues...$(NC)"
	ruff check . --fix

format: ## Format code
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	ruff format .

type-check: ## Run type checking
	@echo "$(BLUE)🏷️  Running type checks...$(NC)"
	mypy app/ || echo "$(YELLOW)⚠️  mypy not available or errors found$(NC)"

quality: lint format type-check ## Run all code quality checks

##@ Dependencies

deps: ## 📦 Compile all locked requirements files from the .in files.
	@echo "--- Compiling backend requirements ---"
	uv pip compile --upgrade $(REQUIREMENTS_DIR)/requirements-core.in -o $(REQUIREMENTS_DIR)/requirements-core.txt
	@echo "--- Compiling development requirements ---"
	uv pip compile --upgrade $(REQUIREMENTS_DIR)/requirements-dev.in -o $(REQUIREMENTS_DIR)/requirements-dev.txt
	@echo "--- Compiling AI/ML requirements ---"
	uv pip compile --upgrade $(REQUIREMENTS_DIR)/requirements-ml.in -o $(REQUIREMENTS_DIR)/requirements-ml.txt
	@echo "--- Compiling common requirements ---"
	uv pip compile --upgrade $(REQUIREMENTS_DIR)/requirements.in -o $(REQUIREMENTS_DIR)/requirements.txt
	@echo "✅ All requirements compiled."

deps-update: ## Update and sync dependencies
	@echo "$(BLUE)📦 Updating dependencies...$(NC)"
	uv pip compile requirements/requirements.in -o requirements/requirements.txt
	uv pip compile requirements/requirements-dev.in -o requirements/requirements-dev.txt
	docker compose exec app uv pip sync requirements/requirements-dev.txt

deps-lock: ## Lock current dependencies
	@echo "$(BLUE)🔒 Locking dependencies...$(NC)"
	uv pip freeze > requirements/requirements-lock.txt

##@ Embedding Service

embedding-test: ## Test embedding service specifically
	@echo "$(BLUE)🧠 Testing embedding service...$(NC)"
	python scripts/manage_embedding_providers.py health

embedding-models: ## List available embedding models
	@echo "$(BLUE)📋 Available HuggingFace Embedding Models:$(NC)"
	@echo "Default: Qwen/Qwen3-Embedding-0.6B (1024 dims)"
	@echo "Alternatives:"
	@echo "  - sentence-transformers/all-MiniLM-L6-v2 (384 dims)"
	@echo "  - BAAI/bge-small-en-v1.5 (384 dims)"
	@echo "  - intfloat/e5-small-v2 (384 dims)"
	@echo "To change: Set EMBEDDING_MODEL_NAME in .env"

embedding-change-model: ## Change embedding model (usage: make embedding-change-model model="model/name")
	@if [ -z "$(model)" ]; then \
		echo "$(RED)❌ Please provide model name: make embedding-change-model model=\"Qwen/Qwen3-Embedding-0.6B\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)🔄 Changing embedding model to $(model)...$(NC)"
	@sed -i '' 's/EMBEDDING_MODEL_NAME=.*/EMBEDDING_MODEL_NAME=$(model)/' .env || \
	 sed -i 's/EMBEDDING_MODEL_NAME=.*/EMBEDDING_MODEL_NAME=$(model)/' .env
	docker compose restart embedding-service

##@ Monitoring

health-check: ## Comprehensive health check
	@echo "$(BLUE)🏥 Running comprehensive health check...$(NC)"
	@./scripts/health_check.sh

logs-errors: ## Show only error logs
	docker compose logs --since=1h | grep -i error || echo "$(GREEN)✅ No errors found in recent logs$(NC)"

stats: ## Show system statistics
	@echo "$(BLUE)📊 System Statistics:$(NC)"
	@echo "Containers: $(shell docker compose ps --format table | wc -l)"
	@echo "Images: $(shell docker images | wc -l)"
	@echo "Volumes: $(shell docker volume ls | wc -l)"
	@echo "Networks: $(shell docker network ls | wc -l)"

##@ Utilities

wait-for-services: ## Wait for services to be healthy
	@echo "$(BLUE)⏳ Waiting for services...$(NC)"
	@timeout=60; \
	while [ $$timeout -gt 0 ]; do \
		if docker compose exec postgres pg_isready -U postgres >/dev/null 2>&1; then \
			echo "$(GREEN)✅ PostgreSQL ready$(NC)"; \
			break; \
		fi; \
		echo "Waiting for PostgreSQL... ($$timeout)"; \
		sleep 1; \
		timeout=$$((timeout-1)); \
	done
	@timeout=60; \
	while [ $$timeout -gt 0 ]; do \
		if docker compose exec redis redis-cli ping >/dev/null 2>&1; then \
			echo "$(GREEN)✅ Redis ready$(NC)"; \
			break; \
		fi; \
		echo "Waiting for Redis... ($$timeout)"; \
		sleep 1; \
		timeout=$$((timeout-1)); \
	done

backup: ## Backup database and volumes
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	@mkdir -p backups
	docker compose exec postgres pg_dump -U postgres app > backups/db_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Database backup created$(NC)"

full-reset: ## Complete system reset (WARNING: destroys all data)
	@echo "$(RED)⚠️  This will destroy ALL data and rebuild everything!$(NC)"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(MAKE) clean; \
		$(MAKE) build; \
		$(MAKE) up; \
		$(MAKE) migrate; \
		$(MAKE) test; \
	fi