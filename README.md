# Charity Policy AI Chatbot API

![Build Status](https://img.shields.io/badge/build-passing-green)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Tests](https://img.shields.io/badge/tests-7/7_passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-35%25-yellow)

An intelligent, production-ready API service designed to answer user questions about a charity's policies based on a provided set of PDF documents. This project uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers while integrating best-in-class tools for observability and maintainability.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/chatbot-api-service.git
cd chatbot-api-service

# Configure environment (add your API keys)
cp .env.example .env
# Edit .env with your actual API keys

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
- **Langfuse UI**: <http://localhost:3000>

## Core Features

- **High-Accuracy RAG:** Leverages **LlamaIndex** for state-of-the-art data ingestion, indexing, and retrieval from PDF documents.
- **Conversational Memory:** The chatbot remembers the last few turns of the conversation to answer follow-up questions effectively.
- **Production-Grade API:** Built with **FastAPI** for high performance, automatic data validation, and interactive documentation.
- **Usage Limiting:** A session-based rate limiter using **Redis** protects the API from abuse and controls operational costs.
- **LLM Observability:** Integrated with **Langfuse** for detailed tracing, debugging, and cost/performance monitoring of every request.
- **Model Flexibility:** Uses **OpenRouter** to easily switch between different LLMs (e.g., Gemini, Claude) to optimize for cost and performance.
- **Quality Assurance:** Comprehensive testing, linting, and pre-commit hooks ensure code quality.

## Tech Stack

| Component | Technology | Rationale / Purpose |
|-----------|------------|-------------------|
| Backend Framework | FastAPI | High-performance Python framework with automatic API documentation and data validation. |
| LLM App Framework | LlamaIndex | Specialized data-centric framework providing SOTA components for ingestion, indexing, and advanced retrieval. |
| LLM Observability | Langfuse | Purpose-built platform for tracing, debugging, evaluating, and monitoring LLM applications. |
| Language | Python 3.11+ | Modern Python with full type hints and async support. |
| Primary Database | PostgreSQL w/ pgvector | Robust SQL database with vector similarity search capabilities. |
| In-Memory Datastore | Redis | High-speed key-value store for session management and caching. |
| AI Model Access | OpenRouter | Unified API access to multiple LLM providers with cost optimization. |
| Containerization | Docker | Consistent deployment with Docker Compose for local development. |

## 🛠 Available Commands

### System Management

```bash
make up          # Start all services (FastAPI + PostgreSQL + Redis + Langfuse)
make down        # Stop all services
make logs        # View output from all containers
make help        # Show all available commands
```

### Development & Testing

```bash
make test        # Run all tests (pytest)
make test-cov    # Run tests with coverage report
make lint        # Check code quality with ruff
make format      # Format code with ruff
make shell       # Open shell in app container
```

### Database Operations  

```bash
make db-shell    # Open PostgreSQL shell
make ingest      # Run data ingestion (index PDF documents)
```

### Maintenance

```bash
make clean       # Remove all containers, volumes, and images
make clean-pyc   # Remove Python cache files
```

## Project Structure

```
chatbot-api-service/
│
├── app/                         # FastAPI application
│   ├── api/v1/                  # API endpoints
│   │   ├── chat.py              # Chat endpoint logic
│   │   └── admin.py             # Admin endpoints
│   ├── core/                    # Core configuration
│   │   ├── config.py            # Environment & settings
│   │   └── query_engine.py      # LlamaIndex integration
│   ├── schemas/                 # Pydantic models
│   └── main.py                  # Application entry point
│
├── scripts/                     # Utility scripts
│   ├── ingest.py                # Data ingestion with Langfuse
│   ├── ingest_simple.py         # Simplified ingestion  
│   └── create_sample_pdf.py     # Generate test documents
│
├── tests/                       # Test suite
│   ├── test_main.py             # API tests
│   └── test_ingestion.py        # Ingestion tests
│
├── docs/                        # Documentation
├── pdf_documents/               # PDF files to index
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # App container definition
├── Makefile                     # Development commands
├── .env.example                 # Environment template
└── requirements.txt             # Python dependencies
```

## 📋 Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Git

### 1. Clone and Configure

```bash
git clone https://github.com/your-username/chatbot-api-service.git
cd chatbot-api-service

# Copy environment template and add your API keys
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` and replace placeholder values:

```bash
# Required: Add your actual API keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here  
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here

# Required: Generate strong random keys
API_KEY=your_very_strong_api_key_here
ADMIN_API_KEY=your_admin_api_key_here
NEXTAUTH_SECRET=your_strong_nextauth_secret_here
SALT=your_strong_salt_here
```

### 3. Add Your Documents

Place your PDF files in the `pdf_documents/` directory:

```bash
cp your-policy-documents.pdf pdf_documents/
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

## 🔗 API Endpoints

### Health Check

- **GET** `/health` - System health status

### Documentation  

- **GET** `/docs` - Interactive Swagger UI
- **GET** `/redoc` - ReDoc documentation

### Chat API *(Coming Soon)*

- **POST** `/api/v1/chat` - Ask questions about policy documents
- **POST** `/api/v1/admin/re-ingest-data` - Trigger data re-ingestion

## 📊 Monitoring & Observability

### Langfuse Integration

Access the Langfuse UI at `http://localhost:3000` to monitor:

- **Request Tracing**: Detailed logs of every API call
- **Performance Metrics**: Response times and token usage
- **Cost Analysis**: Track OpenRouter API costs
- **Quality Evaluation**: Assess response quality over time

### Health Monitoring

The `/health` endpoint provides system status information for monitoring tools.

## 🧪 Testing & Quality

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report  
make test-cov

# Expected: 7/7 tests passing with 35% coverage
```

### Code Quality

```bash
# Check code formatting
make format

# Run linting  
make lint

# Both should pass with no issues
```

### Pre-commit Hooks

The repository includes comprehensive pre-commit hooks that automatically:

- Format code with Ruff
- Check for security issues with Gitleaks  
- Lint markdown files
- Validate YAML/TOML/JSON
- Enforce conventional commit messages

## 🔧 Development

### Local Development

```bash
# Start services for development
make up

# Open shell in app container
make shell

# View logs from all services
make logs
```

### Adding New Features

1. Create a feature branch following conventional naming
2. Implement changes with tests
3. Run quality checks: `make test`, `make lint`, `make format`
4. Commit with conventional commit messages
5. Submit pull request

## 🚀 Deployment

The system is designed for easy deployment with:

- **Docker containers** for consistent environments
- **Environment-based configuration** for different stages
- **Health checks** for container orchestration
- **Comprehensive logging** for debugging

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
feat(chat): add conversation memory support
fix(query): resolve token limit issue in context window
docs(readme): update installation instructions
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
