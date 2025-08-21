#!/bin/bash

# Application Startup Script with Migration Support
# Handles database migrations before starting the FastAPI application

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] APP STARTUP:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

wait_for_postgres() {
    log "⏳ Waiting for PostgreSQL to be ready..."
    
    max_tries=60
    try_count=0
    
    while [ $try_count -lt $max_tries ]; do
        if pg_isready -h postgres -p 5432 -U postgres >/dev/null 2>&1; then
            success "✅ PostgreSQL is ready"
            return 0
        fi
        
        try_count=$((try_count + 1))
        log "Waiting for PostgreSQL... (attempt $try_count/$max_tries)"
        sleep 2
    done
    
    error "❌ PostgreSQL did not become ready within $(($max_tries * 2)) seconds"
    exit 1
}

run_migrations() {
    log "🗃️  Running database migrations..."
    
    if alembic upgrade head; then
        success "✅ Database migrations completed successfully"
    else
        error "❌ Database migrations failed"
        exit 1
    fi
}

validate_environment() {
    log "🔧 Validating environment configuration..."
    
    # Check critical environment variables
    required_vars=(
        "DATABASE_URL"
        "REDIS_HOST"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "❌ Required environment variable $var is not set"
            exit 1
        fi
    done
    
    success "✅ Environment configuration is valid"
}

start_application() {
    log "🚀 Starting FastAPI application..."
    
    # Default to development mode if not specified
    MODE=${MODE:-development}
    
    if [ "$MODE" = "development" ]; then
        success "🔥 Starting in development mode with auto-reload"
        exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    else
        success "🏭 Starting in production mode"
        exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
    fi
}

main() {
    log "🌟 Initializing Chatbot API Service..."
    
    validate_environment
    wait_for_postgres
    run_migrations
    start_application
}

# Handle script interruption
trap 'error "❌ Application startup interrupted"; exit 130' INT TERM

main "$@"