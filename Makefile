.PHONY: help build up down logs shell test lint format clean

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