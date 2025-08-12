.PHONY: help build up down logs shell test test-all test-pgvector test-performance test-cov lint format clean rebuild env-generate-secrets env-rotate-secrets env-lockdown env-set-retention logs-all

help: ## Show this help message
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  %-20s %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

## Development
build: ## Build or rebuild services
	docker compose build

up: ## Start all services (FastAPI + Demo UI + PostgreSQL + Redis + etc.)
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## View output from all containers (core app only)
	docker compose logs -f app postgres redis

logs-all: ## View output from all containers (app + langfuse + clickhouse + ui)
	docker compose logs -f app demo-ui langfuse langfuse-worker clickhouse postgres redis minio

shell: ## Open a shell in the app container
	docker compose exec app /bin/bash

## Testing
test: ## Run all tests (default test suite)
	docker compose exec app pytest -v

test-all: ## Run all tests with verbose output and performance metrics
	docker compose exec app pytest -v -s

test-pgvector: ## Run pgvector performance and configuration tests
	docker-compose exec app pytest tests/test_pgvector_performance.py tests/test_pgvector_prometheus.py -v -s

test-metrics: ## Run flexible metrics system tests
	docker-compose exec app pytest tests/test_flexible_metrics.py -v

test-performance: ## Run performance tests with detailed output
	docker-compose exec app pytest tests/test_pgvector_performance.py::TestPgVectorPerformance::test_vector_search_latency -v -s

test-unit: ## Run unit tests only (fast)
	docker-compose exec app pytest -m "not integration" -v

test-integration: ## Run integration tests only
	docker-compose exec app pytest tests/test_chat.py tests/test_pgvector_performance.py -v

test-cov: ## Run tests with coverage report
	docker-compose exec app pytest --cov=app --cov-report=term-missing --cov-report=html

test-cov-pgvector: ## Run pgvector tests with coverage
	docker-compose exec app pytest tests/test_pgvector_performance.py tests/test_pgvector_prometheus.py --cov=app.core.query_engine --cov=app.db.models --cov-report=term-missing

## Code Quality
lint: ## Run linting
	docker-compose exec app ruff check .

lint-fix: ## Run linting with auto-fix
	docker-compose exec app ruff check . --fix

format: ## Format code
	docker-compose exec app ruff format .

type-check: ## Run type checking (if mypy available)
	docker-compose exec app python -m mypy app/ || echo "mypy not available, skipping type check"

quality-check: ## Run all quality checks (lint + format + type)
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) type-check

## Database
db-shell: ## Open a database shell
	docker compose exec postgres psql -U postgres -d app

db-status: ## Show database status and pgvector info
	@echo "=== Database Status ==="
	docker compose exec postgres psql -U postgres -d app -c "SELECT version();"
	@echo "=== pgvector Extension ==="
	docker compose exec postgres psql -U postgres -d app -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
	@echo "=== content_embeddings Table ==="
	docker compose exec postgres psql -U postgres -d app -c "\d content_embeddings"

migrate: ## Run database migrations
	docker compose exec app alembic upgrade head

migrate-create: ## Create a new migration
	docker compose exec app alembic revision --autogenerate -m "$(name)"

ingest: ## Run data ingestion (one-time setup)
	docker compose exec app python scripts/ingest_simple.py

## Dependencies
deps-compile: ## Compile development dependencies (single source of truth)
	docker compose exec app uv pip compile requirements-dev.in -o requirements-dev.txt -c requirements.in

deps-sync: ## Sync development dependencies
	docker compose exec app uv pip sync --system requirements-dev.txt

## Cleanup
clean: ## Remove all containers, volumes, and images
	docker compose down -v --rmi all

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## ClickHouse & Langfuse
clickhouse-reset: ## Reset ClickHouse volume (fixes auth issues)
	docker compose down
	docker volume rm chatbot-api-service_clickhouse_data || true
	docker compose up -d

clickhouse-test: ## Run ClickHouse smoke tests
	@echo "Testing ClickHouse authentication..."
	docker compose exec clickhouse clickhouse-client -u langfuse --password $(CLICKHOUSE_PASSWORD) -q 'SELECT 1'
	@echo "Testing Langfuse health..."
	curl -f http://localhost:13000/api/public/health

langfuse-logs: ## View Langfuse service logs
	docker compose logs -f langfuse langfuse-worker clickhouse 

## Build & Deploy
rebuild: ## Rebuild all Docker images without cache
	docker compose build --no-cache

rebuild-app: ## Rebuild only the app service
	docker compose build --no-cache app

## Health Checks
health-check: ## Run comprehensive health checks
	@echo "=== System Health Check ==="
	@echo "1. Testing container connectivity..."
	docker compose exec app python -c "print('✅ App container: OK')"
	@echo "2. Testing database connectivity..."
	docker compose exec postgres psql -U postgres -d app -c "SELECT '✅ Database: OK';"
	@echo "3. Testing Redis connectivity..."
	docker compose exec redis redis-cli ping || echo "⚠️  Redis auth required (normal for secured setup)"
	@echo "4. Testing pgvector functionality..."
	$(MAKE) test-pgvector
	@echo "5. Testing vector search performance..."
	@echo "=== Vector Search Performance Benchmark ==="
	docker compose exec app pytest tests/test_pgvector_performance.py::TestPgVectorPerformance::test_vector_search_latency -v -s
	@echo "=== Health Check Complete ==="

## Cache Management
cache-clear: ## Clear Redis cache
	docker compose exec redis redis-cli -a $(shell grep REDIS_AUTH .env | cut -d= -f2) FLUSHDB || echo "Redis connection failed, cache may be using in-memory fallback"

cache-stats: ## Show cache statistics
	@echo "=== Cache Statistics ==="
	docker compose exec app python -c "import asyncio; from app.core.cache import get_cache_stats; print('Cache Stats:', asyncio.run(get_cache_stats()))"

cache-test: ## Test cache functionality
	docker compose exec app python -c "import asyncio; from app.core.cache import get_cache_backend; backend = asyncio.run(get_cache_backend()); print('Cache backend:', type(backend).__name__)"

## CI/CD Simulation
ci-test: ## Run tests as they would run in CI
	$(MAKE) lint
	$(MAKE) test-all
	$(MAKE) test-cov

## Env Management
env-generate-secrets: ## Generate required secrets into .env if missing
	./scripts/manage_env.sh generate-secrets

env-rotate-secrets: ## Force rotate NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY, CLICKHOUSE_PASSWORD
	./scripts/manage_env.sh generate-secrets --rotate

env-lockdown: ## Disable public signup and set default retention to 90d
	./scripts/manage_env.sh lockdown-signup
	./scripts/manage_env.sh set-retention 90