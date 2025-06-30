.PHONY: help build up down logs shell test lint format clean rebuild

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
	docker-compose build

up: ## Start all services in the background
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View output from containers
	docker-compose logs -f

shell: ## Open a shell in the app container
	docker-compose exec app /bin/bash

## Testing
test: ## Run tests
	docker-compose exec app pytest

test-cov: ## Run tests with coverage report
	docker-compose exec app pytest --cov=app --cov-report=term-missing

## Code Quality
lint: ## Run linting
	docker-compose exec app ruff check .

format: ## Format code
	docker-compose exec app ruff format .

## Database
db-shell: ## Open a database shell
	docker-compose exec postgres psql -U postgres -d app

migrate: ## Run database migrations
	docker-compose exec app alembic upgrade head

migrate-create: ## Create a new migration
	docker-compose exec app alembic revision --autogenerate -m "$(name)"

ingest: ## Run data ingestion (one-time setup)
	docker-compose exec app python scripts/ingest_simple.py

## Dependencies
deps-compile: ## Compile dependencies
	docker-compose exec app pip-compile requirements.in

deps-sync: ## Sync dependencies
	docker-compose exec app pip-sync requirements.txt

## Cleanup
clean: ## Remove all containers, volumes, and images
	docker-compose down -v --rmi all

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## ClickHouse & Langfuse
clickhouse-reset: ## Reset ClickHouse volume (fixes auth issues)
	docker-compose down
	docker volume rm chatbot-api-service_clickhouse_data || true
	docker-compose up -d

clickhouse-test: ## Run ClickHouse smoke tests
	@echo "Testing ClickHouse authentication..."
	docker-compose exec clickhouse clickhouse-client -u langfuse --password $(CLICKHOUSE_PASSWORD) -q 'SELECT 1'
	@echo "Testing Langfuse health..."
	curl -f http://localhost:3000/api/public/health

langfuse-logs: ## View Langfuse service logs
	docker-compose logs -f langfuse langfuse-worker clickhouse 

# Rebuild all Docker images without cache
rebuild:
	docker compose build --no-cache 