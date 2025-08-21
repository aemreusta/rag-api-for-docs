#!/bin/bash

# Professional Database Migration Runner
# Handles automatic retries, health checks, and error reporting

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_RETRIES=5
RETRY_DELAY=10
DATABASE_URL="${DATABASE_URL:-postgresql+psycopg2://postgres:postgres@localhost:15432/app}"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

check_database_connectivity() {
    log "🔍 Checking database connectivity..."
    
    # Check if running in Docker environment
    if docker compose ps postgres >/dev/null 2>&1; then
        if ! docker compose exec postgres pg_isready -U postgres >/dev/null 2>&1; then
            error "PostgreSQL is not ready"
            return 1
        fi
    else
        # Direct connection test
        if ! timeout 10 pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; then
            error "Cannot connect to database"
            return 1
        fi
    fi
    
    success "✅ Database connectivity confirmed"
    return 0
}

check_pgvector_extension() {
    log "🧩 Checking pgvector extension..."
    
    if docker compose ps postgres >/dev/null 2>&1; then
        result=$(docker compose exec postgres psql -U postgres -d app -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');" 2>/dev/null || echo "f")
    else
        result=$(psql "$DATABASE_URL" -t -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');" 2>/dev/null || echo "f")
    fi
    
    if [[ "$result" =~ "t" ]]; then
        success "✅ pgvector extension is available"
    else
        warn "⚠️  pgvector extension not found - will be created during migration"
    fi
}

run_migration() {
    log "🗃️  Running Alembic migrations..."
    
    # Check if running in Docker environment
    if docker compose ps app >/dev/null 2>&1; then
        docker compose exec app alembic upgrade head
    else
        # Direct alembic execution
        DATABASE_URL="$DATABASE_URL" alembic upgrade head
    fi
}

verify_migration_success() {
    log "✅ Verifying migration success..."
    
    if docker compose ps postgres >/dev/null 2>&1; then
        # Check if dimension-specific tables exist
        tables=$(docker compose exec postgres psql -U postgres -d app -t -c "
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'embeddings_%'
            ORDER BY table_name;
        " 2>/dev/null)
    else
        tables=$(psql "$DATABASE_URL" -t -c "
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'embeddings_%'
            ORDER BY table_name;
        " 2>/dev/null)
    fi
    
    if echo "$tables" | grep -q "embeddings_"; then
        success "✅ Dimension-specific embedding tables found"
        echo "$tables" | sed 's/^/    - /'
    else
        warn "⚠️  No embedding tables found - this might be expected for a fresh installation"
    fi
    
    # Check alembic version table
    if docker compose ps postgres >/dev/null 2>&1; then
        current_revision=$(docker compose exec postgres psql -U postgres -d app -t -c "
            SELECT version_num FROM alembic_version LIMIT 1;
        " 2>/dev/null | tr -d ' \n\r')
    else
        current_revision=$(psql "$DATABASE_URL" -t -c "
            SELECT version_num FROM alembic_version LIMIT 1;
        " 2>/dev/null | tr -d ' \n\r')
    fi
    
    if [ -n "$current_revision" ]; then
        success "✅ Current database revision: $current_revision"
    else
        error "❌ Could not determine database revision"
        return 1
    fi
}

main() {
    log "🚀 Starting database migration process..."
    
    for attempt in $(seq 1 $MAX_RETRIES); do
        log "📝 Migration attempt $attempt/$MAX_RETRIES"
        
        if check_database_connectivity; then
            check_pgvector_extension
            
            if run_migration; then
                verify_migration_success
                success "🎉 Migration completed successfully!"
                exit 0
            else
                error "❌ Migration failed on attempt $attempt"
            fi
        else
            error "❌ Database connectivity check failed on attempt $attempt"
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            warn "⏳ Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    done
    
    error "❌ Migration failed after $MAX_RETRIES attempts"
    exit 1
}

# Handle script interruption
trap 'error "❌ Migration interrupted"; exit 130' INT TERM

main "$@"