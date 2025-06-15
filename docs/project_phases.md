# Project Phases

This updated plan leverages the new frameworks to accelerate development and focus on quality.

## Phase 1: Setup & Data Ingestion

**Goal**: Establish the core infrastructure and create a high-quality, indexed knowledge base.

**Details**:

- **Setup**: Configure the environment with FastAPI, LlamaIndex, Langfuse, Docker, Postgres, and Redis.
- **Ingestion Script**: Refactor `scripts/ingest.py` to use LlamaIndex's PyMuPDFReader and SentenceSplitter (or a more advanced node parser).
- **Indexing**: Configure the PGVectorStore in LlamaIndex and run the ingestion script to populate the database with embeddings.
- **Observability**: Integrate the Langfuse SDK from day one to trace the ingestion process.

**Outcome**: A versioned, indexed knowledge base in PostgreSQL and a repeatable ingestion script.

## Phase 2: Query Engine Development & Evaluation

**Goal**: Build and validate the core RAG pipeline that can answer questions accurately.

**Details**:

- **Build Engine**: In `core/query_engine.py`, construct a LlamaIndex QueryEngine (e.g., RetrieverQueryEngine). Configure it to use your vector store retriever and the OpenRouter LLM.
- **API Endpoint**: Wire the query engine to the FastAPI endpoint in `chat.py`.
- **Create Dataset**: In Langfuse, create a "golden dataset" of 20-50 important questions with ideal answers.
- **Evaluate**: Run your first evaluations using Langfuse to score the baseline engine on metrics like Faithfulness, Answer Relevancy, and Context Precision.

**Outcome**: A functional API endpoint whose quality is tracked and measured.

## Phase 3: Advanced Features & Deployment

**Goal**: Implement conversational memory and deploy the secure, observable service.

**Details**:

- **Conversational Memory**: Upgrade the QueryEngine to a ChatEngine using LlamaIndex's built-in memory components to handle follow-up questions.
- **Security & Rate Limiting**: Implement the API key authentication and Redis-based rate limiting. These remain outside the LlamaIndex logic.
- **Containerize & Deploy**: Finalize the Dockerfile and docker-compose.yml (including the Langfuse container if self-hosting) and deploy the stack to a cloud provider.
- **Integrate**: Work with the frontend team, providing the API spec and explaining how to capture user feedback scores to send to the Langfuse API.

**Outcome**: A smart, secure, and observable chatbot is live and accessible to the public.

## Phase 4: Continuous Improvement & Tuning

**Goal**: Use the observability platform to systematically improve the chatbot's performance, cost, and quality.

**Details**:

- **Monitor**: Actively monitor the Langfuse dashboards for cost spikes, latency degradation, or high rates of low-scoring user feedback.
- **Debug**: Use the detailed traces in Langfuse to analyze and fix any user-reported issues or bad responses.
- **A/B Test**: Use the Langfuse evaluation tools to A/B test different prompts, LLMs (via OpenRouter), or LlamaIndex retriever settings to find the optimal configuration.
- **Tune**: Based on data, make informed decisions to tune the system for better performance and user satisfaction.

**Outcome**: A mature, data-driven workflow for maintaining and enhancing a high-quality AI service.
