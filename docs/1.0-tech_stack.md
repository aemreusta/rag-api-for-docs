# Tech Stack Document

This document outlines the "best-of-breed" technologies for the Charity Policy Chatbot API.

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
