# AI Gateway Project – Hürriyet Partisi

Multilingual, policy-aware AI chat support for the **hurriyetpartisi.org** WordPress site. Visitors can ask questions about the party's programme, constitution and activities in natural language. The chatbot is backed by a **high-performance Retrieval-Augmented Generation (RAG) pipeline** with **HNSW vector search** and monitored through **Langfuse v3**.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/gencturkler/ai-gateway.git
cd ai-gateway

# Configure environment (add your API keys)
cp .env.example .env
# Edit .env with your actual API keys and ClickHouse configuration

# Start the complete system
make up

# Run initial data ingestion (one-time setup)
make ingest

# Verify system health
curl http://localhost:8000/health
```

**🎉 Your system is now running at:**

- **API**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs>  
- **AI Gateway**: <http://localhost:8080>
- **Langfuse UI**: <http://localhost:3000>
- **ClickHouse**: <http://localhost:8123>
- **MinIO Console**: <http://localhost:9091>
- **Metrics**: <http://localhost:8000/metrics> (if Prometheus enabled)

## 🎯 Purpose

Provide multilingual, policy-aware AI chat support on the public WordPress site **hurriyetpartisi.org** so visitors can ask questions about the party's programme, constitution and activities in natural language. The chatbot is backed by a Retrieval-Augmented Generation (RAG) pipeline and monitored through **Langfuse v3**.

## 👥 Stakeholders & Identities

| Role | Entity / Contact |
|------|------------------|
| Sponsoring party | **Hürriyet Partisi** |
| Technical partner | **Genç Türkler** |
| Project repo | _ai-gateway_ |

## 🏗 High-Level Architecture

1. **FastAPI + PGVector RAG Service** – ingests party PDFs, posts and policy pages with **HNSW vector search**
2. **OPENROUTER** – single entry-point that proxies/load-balances calls to multiple LLM providers (Gemini, GPT-4o, Claude-Sonnet, local llama.cpp)
3. **Langfuse v3** – traces + eval; ClickHouse for OLAP, Postgres for metadata
4. **Redis** – both chat memory and per-IP rate-limit
5. **WordPress Chat Widget** – embeds a JS snippet
6. **Flexible Monitoring** – supports Prometheus, DataDog, OpenTelemetry, or NoOp backends

```
Browser → WP Script ↔ ai-gateway ↔ RAG API ↔ LLMs
                           ↘︎ Langfuse (ClickHouse + Postgres)
                           ↘︎ Metrics (Prometheus/DataDog/OTEL)
```

## Core Features

- **High-Performance Vector Search:** ✅ **HNSW index with pgvector** achieving **P99: 1.88ms latency** (27× faster than target <50ms)
- **Multilingual Support:** Turkish and English prompt templates for bilingual audience
- **High-Accuracy RAG:** Leverages **LlamaIndex** for state-of-the-art data ingestion, indexing, and retrieval from PDF documents
- **AI Gateway:** Single entry-point with load balancing across multiple LLM providers
- **LLM Router:** ✅ **Production-ready** priority-based routing (OpenRouter→Groq→OpenAI→Local) with automatic fallback, Redis caching, and comprehensive error handling
- **Conversational Memory:** The chatbot remembers conversation context using Redis
- **Production-Grade API:** Built with **FastAPI** with 95 passing tests, structured logging, and comprehensive error handling
- **Rate Limiting:** Per-IP rate limiting using **Redis** protects the API from abuse and controls operational costs
- **LLM Observability:** Integrated with **Langfuse v3** for detailed tracing, debugging, and cost/performance monitoring
- **Flexible Monitoring:** ✅ **Multi-backend metrics** supporting Prometheus, DataDog, OpenTelemetry, and NoOp
- **Model Flexibility:** Uses multiple LLM providers (Gemini, Llama 3, GPT) to optimize for cost and performance
- **WordPress Integration:** Easy embedding via script or InsertChat plugin

## Tech Stack

| Component | Technology | Rationale / Purpose |
|-----------|------------|-------------------|
| Backend Framework | FastAPI | High-performance Python framework with automatic API documentation and data validation |
| LLM App Framework | LlamaIndex | Specialized data-centric framework providing SOTA components for ingestion, indexing, and advanced retrieval |
| Vector Database | **PostgreSQL w/ pgvector + HNSW** | **High-performance vector similarity search with P99 <2ms latency** |
| AI Gateway | Go | High-performance proxy for load balancing across multiple LLM providers |
| LLM Observability | Langfuse v3 | Purpose-built platform for tracing, debugging, evaluating, and monitoring LLM applications |
| **Metrics & Monitoring** | **Prometheus/DataDog/OpenTelemetry** | **Flexible multi-backend observability (auto-detection)** |
| OLAP Database | ClickHouse 24.3 | High-performance analytical database for Langfuse metrics and analytics |
| Language | Python 3.10+ | Modern Python with full type hints and async support |
| In-Memory Datastore | Redis 7 | High-speed key-value store for session management, caching, and rate limiting |
| Frontend Integration | WordPress | Chat widget integration for public website |
| Containerization | Docker | Consistent deployment with Docker Compose for local development |

## 🛠 Deployment Stack

Docker Compose file spins up:

- `app` (FastAPI)
- `postgres:15` + `pgvector extension` + **HNSW index**
- `redis:7`
- `clickhouse`
- `minio`
- `langfuse` & `langfuse-worker`

## 🛠 Available Commands

### System Management

```bash
make up                     # Start all services (FastAPI + AI Gateway + PostgreSQL + Redis + ClickHouse + Langfuse)
make down                  # Stop all services
make logs                  # View output from all containers
make help                  # Show all available commands
make health-check          # Comprehensive system health verification
```

### Development & Testing

```bash
make test                  # Run all tests (pytest) - 95 tests passing
make test-cov             # Run tests with coverage report
make test-pgvector        # pgvector performance & configuration tests
make test-metrics         # flexible metrics system tests
make benchmark            # Vector search performance benchmarks
make lint                 # Check code quality with ruff
make format               # Format code with ruff
make quality-check        # All quality checks (lint + format + type)
make shell                # Open shell in app container
make chat                 # Test local chat endpoint with curl
```

### Database Operations  

```bash
make db-shell             # Open PostgreSQL shell
make db-status            # Database & pgvector status information
make ingest               # Run data ingestion (index PDF documents)
make clickhouse-reset     # Drop OLAP volume (dev only)
make clickhouse-test      # One-shot smoke tests
```

### Maintenance & Dependencies

```bash
make clean             # Remove all containers, volumes, and images
make clean-pyc         # Remove Python cache files
```

## Performance Achievements ✅

### Vector Search Performance (HNSW Index)

- **P99 latency: 1.88ms** (target was <50ms) — **27× better than target**
- **Average latency: 1.3ms** with P50: 1.3ms, P95: 1.9ms
- **100% recall accuracy** at top-1, top-5, and top-10 results
- **HNSW parameters**: m=32, ef_construction=64, ef_search=100
- **Index type**: HNSW (Hierarchical Navigable Small World) for optimal performance

### Test Coverage

- **95 tests passing** with comprehensive coverage
- Performance benchmarks and monitoring integration tests
- Security and fuzz testing hooks
- Zero warnings or linter errors

## Project Structure

```
ai-gateway/
│
├── app/                         # FastAPI application
│   ├── api/v1/                  # API endpoints
│   │   ├── chat.py              # Chat endpoint logic
│   │   └── admin.py             # Admin endpoints
│   ├── core/                    # Core configuration
│   │   ├── config.py            # Environment & settings
│   │   ├── llm_router.py        # LLM provider routing & fallback
│   │   ├── query_engine.py      # LlamaIndex integration
│   │   ├── logging_config.py    # Structured logging setup
│   │   ├── middleware.py        # Request/response logging
│   │   ├── ratelimit.py         # Redis-based rate limiting
│   │   └── redis.py             # Redis connection management
│   ├── db/                      # Database models
│   │   └── models.py            # SQLAlchemy models with VECTOR(1536)
│   ├── schemas/                 # Pydantic models
│   └── main.py                  # Application entry point
│
├── scripts/                     # Utility scripts
│   ├── ingest.py                # Data ingestion with Langfuse
│   ├── ingest_simple.py         # Simplified ingestion  
│   ├── create_sample_pdf.py     # Generate test documents
│   └── run_baseline_evaluation.py  # Performance evaluation
│
├── tests/                       # Test suite (95 tests)
│   ├── test_main.py             # API tests
│   ├── test_ingestion.py        # Ingestion tests
│   ├── test_llm_router.py       # LLM router tests
│   ├── test_ratelimit.py        # Rate limiting tests
│   ├── test_structured_logging.py  # Logging tests
│   └── test_pgvector_performance.py  # Vector search performance tests
│
├── docs/                        # Documentation
│   ├── diagrams/                # Mermaid diagrams
│   ├── 10.0-llm-router.md       # LLM router architecture
│   ├── 9.0-structured-logging.md   # Logging documentation
│   └── 5.0-project_phases.md    # Project roadmap
├── pdf_documents/               # PDF files to index
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # App container definition
├── Makefile                     # Development commands (17 total)
├── .env.example                 # Environment template
└── requirements.txt             # Python dependencies
```

## 📋 Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Git

### 1. Clone and Configure

```bash
git clone https://github.com/gencturkler/ai-gateway.git
cd ai-gateway

# Copy environment template and add your API keys
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` and replace placeholder values:

#### Critical Environment Variables

**⚠️ BREAKING CHANGE**: Langfuse v3 requires S3 storage configuration. Without it, you'll get crashes.

| Key | Sample value | Notes |
|-----|--------------|--------|
| `NEXTAUTH_SECRET` | `<32-hex>` | **REQUIRED**: `openssl rand -hex 32` |
| `SALT` | `<32-hex>` | **REQUIRED**: `openssl rand -hex 32` |
| `ENCRYPTION_KEY` | `<32-hex>` | **REQUIRED**: `openssl rand -hex 32` |
| `CLICKHOUSE_PASSWORD` | `<32-hex>` | **REQUIRED**: `openssl rand -hex 32` |
| `LANGFUSE_S3_EVENT_UPLOAD_BUCKET` | `langfuse` | **REQUIRED**: MinIO bucket name |
| `MINIO_ROOT_PASSWORD` | `miniosecret` | **REQUIRED**: Strong password |
| `REDIS_AUTH` | `myredissecret` | **REQUIRED**: Strong password |
| `OPENROUTER_API_KEY` | `your_key` | For OpenRouter/Gemini access |
| `GROQ_API_KEY` | `your_key` | For Groq/Llama3 access |
| `OPENAI_API_KEY` | `your_key` | For OpenAI/ChatGPT access |
| `LLM_MODEL_NAME` | `google/gemini-1.5-pro-latest` | Default model name |

#### **New: Flexible Monitoring Configuration**

| Key | Sample value | Notes |
|-----|--------------|--------|
| `METRICS_BACKEND` | `auto` | **auto/prometheus/datadog/opentelemetry/noop** |
| `DATADOG_API_KEY` | `your-api-key` | Required if using DataDog backend |
| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus metrics endpoint |

#### **CORS Configuration for Frontend Integration**

| Key | Sample value | Notes |
|-----|--------------|--------|
| `CORS_ALLOW_ORIGINS` | `` (empty for dev) | Comma-separated origins; wildcard in DEBUG mode |
| `CORS_ALLOW_METHODS` | `GET,POST,PUT,DELETE,OPTIONS` | HTTP methods for CORS |
| `CORS_ALLOW_HEADERS` | `Authorization,Content-Type,X-API-Key` | Headers for CORS |
| `CORS_ALLOW_CREDENTIALS` | `false` | Disable with wildcard origins |
| `CORS_MAX_AGE` | `600` | Pre-flight cache duration (seconds) |

**CORS Behavior:**

- **Development** (`DEBUG=true` + `ENVIRONMENT=development`): Wildcard (`*`) when origins empty
- **Production**: Requires explicit origins; restrictive by default

#### Step-by-step .env Configuration

```bash
# 1. Copy the template
cp .env.example .env

# 2. Generate required secrets (CRITICAL - run these 4 commands)
echo "NEXTAUTH_SECRET=$(openssl rand -hex 32)" >> .env
echo "SALT=$(openssl rand -hex 32)" >> .env  
echo "ENCRYPTION_KEY=$(openssl rand -hex 32)" >> .env
echo "CLICKHOUSE_PASSWORD=$(openssl rand -hex 32)" >> .env

# 3. Add your LLM provider API key
echo "OPENROUTER_API_KEY=your_actual_openrouter_key" >> .env

# 4. Set strong passwords for MinIO and Redis
sed -i 's/miniosecret/your_strong_minio_password/' .env
sed -i 's/myredissecret/your_strong_redis_password/' .env

# 5. Configure monitoring (optional)
echo "METRICS_BACKEND=auto" >> .env  # auto-detect available backend
# echo "DATADOG_API_KEY=your_key" >> .env  # if using DataDog
```

All other variables have sensible defaults for local development.

### 3. Add Your Documents

Place your PDF files in the `pdf_documents/` directory:

```bash
cp your-policy-documents.pdf pdf_documents/
cp party-constitution.pdf pdf_documents/
cp programme-documents.pdf pdf_documents/
```

### 4. Start the System

```bash
# Build and start all services
make up

# Wait for services to be healthy, then run ingestion
make ingest
```

### 5. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Expected response: {"status":"ok","environment":"development","version":"0.1.0"}
```

## 🌐 WordPress Integration

### Option A – Embed Script (Recommended for MVP)

1. In WP **Appearance → Theme Editor** add before `</body>`:

```html
<script src="https://gateway.hurriyetpartisi.org/static/chat.js"
        data-endpoint="https://gateway.hurriyetpartisi.org/chat"
        data-project="hurriyet" data-lang="tr"></script>
```

1. The script initializes the floating chat icon and streams responses via SSE.

### Option B – InsertChat Plugin

1. Install **InsertChat** plugin
2. Go to _Settings → InsertChat_, paste the same endpoint and project id
3. Use `[insertchat]` shortcode on any page

_For MVP choose Option A – zero plugin overhead._

## 🔗 API Endpoints

### Health Check

- **GET** `/health` - System health status

### Documentation  

- **GET** `/docs` - Interactive Swagger UI
- **GET** `/redoc` - ReDoc documentation
- See `docs/9.0-structured-logging.md` - Comprehensive structured logging guide

### Chat API

- **POST** `/api/v1/chat` - Ask questions about policy documents
- **POST** `/api/v1/admin/re-ingest-data` - Trigger data re-ingestion

### AI Gateway Endpoints

- **POST** `/chat` - Proxied chat endpoint with load balancing
- **GET** `/static/chat.js` - WordPress widget script

## 📊 Monitoring & Observability

### Flexible Monitoring Architecture ✅

The system supports **multiple monitoring backends** through a pluggable architecture:

| Backend | Configuration | Use Case |
|---------|---------------|----------|
| **Prometheus** ✅ | `METRICS_BACKEND=prometheus` | Default for Kubernetes/Docker deployments |
| **DataDog** ✅ | `METRICS_BACKEND=datadog` + `DATADOG_API_KEY` | SaaS APM, enterprise environments |
| **OpenTelemetry** ✅ | `METRICS_BACKEND=opentelemetry` | Cloud-native, vendor-neutral observability |
| **NoOp** ✅ | `METRICS_BACKEND=noop` | Development/testing (no metrics overhead) |
| **Auto-detection** ✅ | `METRICS_BACKEND=auto` (default) | Automatically detects available libraries |

#### Available Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `vector_search_duration_seconds` | Histogram | Vector similarity search latency | `status`, `model` |
| `vector_search_requests_total` | Counter | Total vector search requests | `status`, `model` |
| `vector_search_recall` | Gauge | Search recall accuracy | `k` (top-k results) |

### Structured Logging

The system provides comprehensive JSON-formatted structured logging with:

- **Correlation IDs**: trace_id and request_id for cross-service tracing
- **Sensitive Data Masking**: Automatic redaction of API keys, passwords, emails
- **Request/Response Timing**: Performance metrics and business context  
- **Log Aggregation Ready**: Compatible with ELK, Loki, and OTEL collectors
- **Langfuse Integration**: Shared correlation IDs for 1-click drill-down from logs to traces

### Langfuse v3 Integration

Access the Langfuse UI at `http://localhost:3000` to monitor:

- **Request Tracing**: Detailed logs of every API call
- **Performance Metrics**: Response times and token usage  
- **Cost Analysis**: Track LLM provider API costs
- **Quality Evaluation**: Assess response quality over time
- **ClickHouse Analytics**: OLAP queries for advanced analytics

### Health Monitoring

The `/health` endpoints provide system status information for monitoring tools.

## 🔒 Security & Compliance

- All traffic terminates at Cloudflare → Nginx → ai-gateway
- Redis & Postgres only on docker-internal network
- GPT provider keys stored in HashiCorp Vault; retrieved at container start
- Fallback flow returns generic apology + trace-id
- Per-IP rate limiting to prevent abuse
- SMTP via Amazon SES for secure email delivery

## 🧪 Testing & Quality

### Running Tests

```bash
# Run all tests (95 tests)
make test

# Run with coverage report  
make test-cov

# Test chat functionality
make chat

# Test ClickHouse integration
make clickhouse-test

# Expected: 95/95 tests passing with 0 warnings
```

### Performance Benchmarks

```bash
# Run comprehensive vector search benchmarks
make benchmark

# Expected results:
# - P99 latency: ~1.88ms (target <50ms)
# - 100% recall accuracy at top-1, top-5, top-10
# - HNSW index performance validation
```

### Code Quality

```bash
# Check code formatting
make format

# Run linting  
make lint

# Both should pass with no issues
```

## 🚀 Recent Achievements (v0.5)

| Completed Feature | Performance Impact |
|------------------|-------------------|
| **pgvector HNSW Migration** ✅ | **P99: 1.88ms** (27× faster than target) |
| **Flexible Monitoring Architecture** ✅ | **4 backend support** (Prometheus/DataDog/OpenTelemetry/NoOp) |
| **95 Test Suite** ✅ | **Zero warnings**, comprehensive coverage |
| **Enhanced Makefile** ✅ | **17 commands** for development efficiency |
| **Structured Logging** ✅ | **JSON format**, correlation IDs, sensitive data masking |
| **LLM Router Production** ✅ | **Automatic fallback**, Redis caching, error handling |

## 🚀 Roadmap Delta (v0.5 → v0.6)

| New scope | Why |
|-----------|-----|
| WordPress widget delivery | Unblock public launch |
| SMTP via Amazon SES | Send verification mails from **<ai@gencturkler.co>** |
| Turkish & English prompt templates | Bilingual audience |
| Basic analytics dashboard | Daily active users, token spend |
| Redis LRU Cache Migration | Eliminate cache staleness across workers |
| OAuth 2.1 PKCE Authentication | Security hardening and user management |

## 🔧 Development

### Local Development

```bash
# Start services for development
make up

# Open shell in app container
make shell

# View logs from all services
make logs

# Test chat locally
make chat
```

### Adding New Features

1. Create a feature branch following conventional naming
2. Implement changes with tests
3. Run quality checks: `make test`, `make lint`, `make format`
4. Commit with conventional commit messages
5. Submit pull request

## 🐛 Troubleshooting Cheat-Sheet

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ZodError: LANGFUSE_S3_EVENT_UPLOAD_BUCKET expected string` | Missing S3 env vars | Follow step 2 above to configure .env properly |
| `TypeError: Cannot set property message of ZodError` | Same S3 issue | Same fix + `docker compose restart langfuse langfuse-worker` |
| "Authentication failed" in Langfuse logs | Password mismatch | Ensure `CLICKHOUSE_PASSWORD` identical in both services |
| `getaddrinfo ENOTFOUND minio` | MinIO not ready | Wait for MinIO healthcheck: `docker compose logs minio` |
| Worker & web race on migrations | Missing env var | Ensure `LANGFUSE_AUTO_CLICKHOUSE_MIGRATION_DISABLED=true` in worker |
| Chat widget 404 | Cloudflare cache or wrong subdomain | Purge cache; check DNS `gateway.hurriyetpartisi.org` |
| UI e-mail rejects address | Must include `@` | Use `admin@example.com` not `admin` |
| ClickHouse connection failed | Service not ready | Wait for ClickHouse to fully start before running migrations |
| Redis connection timeout | Network issues or service down | Check Redis container logs and restart if needed |

## 🚀 Deployment

The system is designed for easy deployment with:

- **Docker containers** for consistent environments
- **Environment-based configuration** for different stages
- **Health checks** for container orchestration
- **Comprehensive logging** for debugging
- **Multi-service architecture** with proper service discovery

## Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This helps maintain a clear and standardized commit history.

Format: `<type>(<scope>): <description>`

Types:

- `feat`: A new feature
- `fix`: A bug fix  
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `build`: Changes to build system or external dependencies
- `ci`: Changes to CI configuration files and scripts

Examples:

```bash
feat(chat): add Turkish language support
fix(gateway): resolve load balancing issue
docs(readme): update WordPress integration guide
test(api): add integration tests for chat endpoint
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to:

1. Follow the conventional commits specification for commit messages
2. Update tests as appropriate
3. Update documentation to reflect any changes  
4. Ensure all CI checks pass (`make test`, `make lint`, `make format`)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Repository Topics

This repository is tagged with the following topics to make it more discoverable:

### Core Technologies

- Python
- FastAPI
- LlamaIndex
- Langfuse
- PostgreSQL
- pgvector
- ClickHouse
- Redis
- Docker
- API

### AI/ML Concepts

- RAG (Retrieval-Augmented Generation)
- Chatbot
- LLM (Large Language Models)
- Generative AI
- NLP (Natural Language Processing)
- LLMOps

### Integration & Deployment

- WordPress
- Observability
- Multilingual
- Political

### Problem Domain

- Political Party
- Policy Analysis
- Turkish Politics

These topics help make the repository more discoverable for developers looking for examples of:

- AI Gateway implementation with multiple LLM providers
- Langfuse v3 integration with ClickHouse
- WordPress chatbot integration
- Multilingual RAG applications
- Production-ready AI systems for political organizations

---

### Last Updated

2025-06-27
