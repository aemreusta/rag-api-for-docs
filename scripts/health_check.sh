#!/bin/bash

# Comprehensive System Health Check
# Tests all components of the chatbot API service

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ❌ ERROR:${NC} $1"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ SUCCESS:${NC} $1"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
}

warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  WARNING:${NC} $1"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
}

check_docker_compose() {
    log "🐳 Checking Docker Compose status..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker compose ps --format table >/dev/null 2>&1; then
        success "Docker Compose is running"
        
        # Show service status
        echo -e "${BLUE}Service Status:${NC}"
        docker compose ps --format table
    else
        error "Docker Compose is not running or not configured"
        return 1
    fi
}

check_database() {
    log "🗄️  Checking PostgreSQL database..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker compose exec postgres pg_isready -U postgres >/dev/null 2>&1; then
        success "PostgreSQL is responsive"
        
        # Check database exists
        if docker compose exec postgres psql -U postgres -lqt | cut -d \| -f 1 | grep -qw app; then
            success "Application database 'app' exists"
        else
            error "Application database 'app' does not exist"
            return 1
        fi
        
        # Check pgvector extension
        vector_installed=$(docker compose exec postgres psql -U postgres -d app -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');" 2>/dev/null | tr -d ' \n\r')
        if [[ "$vector_installed" == "t" ]]; then
            success "pgvector extension is installed"
        else
            error "pgvector extension is not installed"
        fi
        
        # Check embedding tables
        embedding_tables=$(docker compose exec postgres psql -U postgres -d app -t -c "
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name LIKE 'embeddings_%'
        " 2>/dev/null | tr -d ' \n\r')
        
        if [ "$embedding_tables" -ge 4 ]; then
            success "Dimension-specific embedding tables found ($embedding_tables tables)"
        else
            warn "Expected 4 embedding tables, found $embedding_tables"
        fi
        
    else
        error "PostgreSQL is not responsive"
        return 1
    fi
}

check_redis() {
    log "💾 Checking Redis cache..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker compose exec redis redis-cli ping >/dev/null 2>&1; then
        success "Redis is responsive"
    else
        # Try with auth (common setup)
        if docker compose exec redis redis-cli -a myredissecret ping >/dev/null 2>&1; then
            success "Redis is responsive (authenticated)"
        else
            error "Redis is not responsive"
            return 1
        fi
    fi
}

check_app_container() {
    log "🚀 Checking application container..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker compose exec app python -c "print('App container is responsive')" >/dev/null 2>&1; then
        success "Application container is responsive"
        
        # Check Python imports
        if docker compose exec app python -c "from app.core.config import settings; print('Config loaded successfully')" >/dev/null 2>&1; then
            success "Application configuration loads successfully"
        else
            error "Application configuration failed to load"
        fi
        
    else
        error "Application container is not responsive"
        return 1
    fi
}

check_embedding_service() {
    log "🧠 Checking embedding service..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    # Check if container is running
    if docker compose ps embedding-service --format json | grep -q '"State":"running"'; then
        success "Embedding service container is running"
        
        # Check health endpoint
        if timeout 10 docker compose exec embedding-service curl -f http://localhost:8080/health >/dev/null 2>&1; then
            success "Embedding service health endpoint is responsive"
        else
            warn "Embedding service health endpoint not responding (may still be starting)"
        fi
    else
        warn "Embedding service container is not running"
    fi
}

check_migrations() {
    log "🗃️  Checking database migrations..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker compose exec app alembic current >/dev/null 2>&1; then
        current_revision=$(docker compose exec app alembic current 2>/dev/null | head -n1)
        if [ -n "$current_revision" ]; then
            success "Database migrations are current: $current_revision"
        else
            error "Could not determine current migration status"
        fi
    else
        error "Alembic migration check failed"
    fi
}

check_environment_config() {
    log "⚙️  Checking environment configuration..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ -f .env ]; then
        success ".env file exists"
        
        # Check critical settings
        if grep -q "DATABASE_URL" .env; then
            success "DATABASE_URL is configured"
        else
            warn "DATABASE_URL not found in .env"
        fi
        
        if grep -q "GOOGLE_AI_STUDIO_API_KEY" .env && [ "$(grep GOOGLE_AI_STUDIO_API_KEY .env | cut -d= -f2)" != "" ]; then
            success "Google AI API key is configured"
        else
            warn "Google AI API key not configured - embedding fallback will be used"
        fi
        
    else
        error ".env file does not exist"
    fi
}

check_ports() {
    log "🔌 Checking port availability..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    ports_to_check="18000 15432 16379 18080"
    
    for port in $ports_to_check; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            success "Port $port is in use (service likely running)"
        elif lsof -i :$port >/dev/null 2>&1; then
            success "Port $port is in use (service likely running)"
        else
            warn "Port $port is not in use"
        fi
    done
}

test_embedding_functionality() {
    log "🧪 Testing embedding functionality..."
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ -f scripts/manage_embedding_providers.py ]; then
        if PYTHONPATH=. python scripts/manage_embedding_providers.py config >/dev/null 2>&1; then
            success "Embedding configuration test passed"
        else
            warn "Embedding configuration test failed - check logs for details"
        fi
    else
        warn "Embedding test script not found"
    fi
}

show_summary() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}         HEALTH CHECK SUMMARY${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Total Checks: $TOTAL_CHECKS"
    echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
    echo -e "${YELLOW}Warnings: $WARNING_CHECKS${NC}"
    echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        if [ $WARNING_CHECKS -eq 0 ]; then
            echo -e "${GREEN}🎉 All checks passed! System is healthy.${NC}"
            exit 0
        else
            echo -e "${YELLOW}⚠️  System is operational but has warnings.${NC}"
            exit 0
        fi
    else
        echo -e "${RED}❌ System has critical issues that need attention.${NC}"
        exit 1
    fi
}

main() {
    echo -e "${BLUE}🏥 Chatbot API Service - Health Check${NC}"
    echo -e "${BLUE}====================================${NC}"
    echo ""
    
    check_docker_compose
    check_database
    check_redis
    check_app_container
    check_embedding_service
    check_migrations
    check_environment_config
    check_ports
    test_embedding_functionality
    
    show_summary
}

main "$@"