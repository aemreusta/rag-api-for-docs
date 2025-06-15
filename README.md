# Charity Policy AI Chatbot API

![Build Status](https://img.shields.io/badge/build-passing-green)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An intelligent, production-ready API service designed to answer user questions about a charity's policies based on a provided set of PDF documents. This project uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers while integrating best-in-class tools for observability and maintainability.

### Core Features

- **High-Accuracy RAG:** Leverages **LlamaIndex** for state-of-the-art data ingestion, indexing, and retrieval from PDF documents.
- **Conversational Memory:** The chatbot remembers the last few turns of the conversation to answer follow-up questions effectively.
- **Production-Grade API:** Built with **FastAPI** for high performance, automatic data validation, and interactive documentation.
- **Usage Limiting:** A session-based rate limiter using **Redis** protects the API from abuse and controls operational costs.
- **LLM Observability:** Integrated with **Langfuse** for detailed tracing, debugging, and cost/performance monitoring of every request.
- **Model Flexibility:** Uses **OpenRouter** to easily switch between different LLMs (e.g., Gemini, Claude) to optimize for cost and performance.
- **Automated Data Ingestion:** A secure admin endpoint allows for triggering data re-ingestion without manual intervention.

### Tech Stack

| Component | Technology | Rationale / Purpose |
|-----------|------------|-------------------|
| Backend Framework | FastAPI | High-performance Python framework. Remains the best choice for the API layer due to its speed and Pydantic integration. |
| LLM App Framework | LlamaIndex | (Upgrade) Replaces custom RAG logic. A specialized data-centric framework providing SOTA components for ingestion, indexing, and advanced retrieval to maximize answer quality. |
| LLM Observability | Langfuse | (Upgrade) Replaces manual tracking. A purpose-built platform for tracing, debugging, evaluating, and monitoring LLM applications. Solves the "black box" problem and provides key metrics out-of-the-box. |
| Language | Python 3.11+ | The required language for the entire modern data and AI stack. |
| Primary Database | PostgreSQL w/ pgvector | A robust SQL database. Now managed via LlamaIndex's PGVectorStore integration, abstracting away manual queries. |
| In-Memory Datastore | Redis | A high-speed key-value store. Still used for the session-based rate limiting, but conversational memory is now handled by LlamaIndex components. |
| AI Model Access | OpenRouter | Provides model flexibility and cost comparison. Now integrated as the LLM provider within LlamaIndex. |
| Containerization | Docker | Used to package the FastAPI app and its dependencies into a portable container for consistent deployment. |

### Project Structure

```
chatbot-api-service/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── admin.py         # Secure admin endpoint for re-ingestion
│   │       └── chat.py          # Main chat endpoint logic
│   │
│   ├── core/
│   │   ├── config.py            # Environment variable management
│   │   └── query_engine.py      # LlamaIndex query engine setup
│   │
│   └── main.py                  # FastAPI application entry point
│
├── pdf_documents/               # Place source PDF files here
│
├── scripts/
│   └── ingest.py                # Data ingestion logic using LlamaIndex
│
├── .env.example                 # Example environment variables
├── Dockerfile                   # Production Docker image definition
├── docker-compose.yml           # Local development environment setup
└── requirements.txt             # Python dependencies
```

### Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/chatbot-api-service.git
   cd chatbot-api-service
   ```

2. **Configure Environment Variables**
   Copy the example `.env` file and fill in your credentials. **This file should never be committed to Git.**

   ```bash
   cp .env.example .env
   ```

   Now, edit `.env` with your API keys for OpenRouter and Langfuse, and your desired database credentials.

3. **Place PDF Documents**
   Add your charity's policy PDF files into the `pdf_documents/` directory.

4. **Build and Run with Docker Compose**
   This is the recommended method as it sets up the FastAPI app, PostgreSQL database, and Redis container all at once.

   ```bash
   docker-compose up --build
   ```

   The API will be available at `http://localhost:8000`.

5. **Run Initial Data Ingestion**
   The first time you run the service, you must populate the database. Execute the ingestion script *inside the running Docker container*.

   ```bash
   docker-compose exec app python -m scripts.ingest
   ```

### API Endpoints

- **Interactive Docs:** Navigate to `http://localhost:8000/docs` to see the auto-generated Swagger UI.

- **Chat Endpoint:**
  - `POST /api/v1/chat`
  - **Authentication:** Requires a bearer token API key.
  - **Body:** `{ "question": "string", "session_id": "string" }`

- **Admin Endpoint:**
  - `POST /api/v1/admin/re-ingest-data`
  - **Authentication:** Requires a separate `X-Admin-API-Key` in the header.
  - **Purpose:** Triggers a background task to re-process all documents in the `pdf_documents` folder.

### Observability with Langfuse

After running the service, you can view detailed traces of every API call in Langfuse. If using the default `docker-compose.yml`, the Langfuse UI will be available at `http://localhost:3000`. This is invaluable for debugging, analyzing costs, and evaluating the quality of your RAG pipeline.

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This helps maintain a clear and standardized commit history.

Format: `<type>(<scope>): <description>`

Types:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `perf`: Performance improvements
- `test`: Adding or modifying tests
- `build`: Changes to build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

Examples:

```bash
feat(chat): add conversation memory support
fix(query): resolve token limit issue in context window
docs(readme): update installation instructions
refactor(engine): optimize document chunking logic
test(api): add integration tests for chat endpoint
```

Scope is optional and should be the name of the module affected (chat, query, api, etc).

### Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to:

1. Follow the conventional commits specification for commit messages
2. Update tests as appropriate
3. Update documentation to reflect any changes
4. Ensure all CI checks pass
