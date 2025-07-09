# Data Ingestion Scripts

This directory contains scripts for ingesting PDF documents into the PostgreSQL vector database with full observability through Langfuse.

## Files

- `ingest.py` - Main ingestion script that processes PDFs and stores them in PostgreSQL with Langfuse tracing
- `test_ingestion_integration.py` - Integration test script to verify the ingestion pipeline
- `create_sample_pdf.py` - Utility script to create sample PDF files for testing

## Prerequisites

1. **PostgreSQL Database** with pgvector extension installed
2. **Langfuse Instance** running and accessible (for observability)
3. **Environment Variables** set in `.env` file (see `.env.example`)
4. **PDF Documents** placed in the `pdf_documents/` directory

## Usage

### 1. Basic Ingestion

```bash
python scripts/ingest.py
```

This will:

- Check for pgvector extension in the database
- Load all PDF files from `pdf_documents/` directory
- Create embeddings using the local BGE model (`BAAI/bge-small-en-v1.5`)
- Store embeddings in PostgreSQL using PGVectorStore
- **Send detailed traces to Langfuse for observability**

### 2. Integration Testing

```bash
python scripts/test_ingestion_integration.py
```

This will run comprehensive checks including:

- Database connectivity
- pgvector extension verification
- **Langfuse connectivity and tracing**
- PDF file availability
- Full ingestion pipeline test
- Result verification

### 3. Create Sample PDFs

```bash
python scripts/create_sample_pdf.py
```

## Observability with Langfuse

The ingestion script is fully instrumented with Langfuse for comprehensive observability:

### 🔍 **What Gets Traced**

- **Complete ingestion pipeline** from start to finish
- **Document loading** with file counts and metadata
- **Vector store operations** and database interactions
- **Embedding generation** and model usage
- **Error handling** with detailed error context
- **Performance metrics** and timing information

### 📊 **Viewing Traces**

1. Run the ingestion script: `python scripts/ingest.py`
2. Open your Langfuse dashboard (default: `http://localhost:3000`)
3. Navigate to **"Traces"** section
4. Look for traces named **"pdf-ingestion"**
5. Click on any trace to see detailed spans and metadata

### 🚨 **Error Monitoring**

- Failed ingestions are automatically traced with error details
- Database connection issues are captured and logged
- Missing dependencies or configuration errors are tracked

## Configuration

All configuration is handled through environment variables:

```bash
# Required for ingestion
DATABASE_URL=postgresql://user:pass@localhost:5432/db
OPENROUTER_API_KEY=your_api_key
LLM_MODEL_NAME=google/gemini-1.5-pro-latest

# Required for observability
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=http://localhost:3000
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `uv pip sync requirements.txt`
2. **Database Connection**: Verify PostgreSQL is running and pgvector extension is installed
3. **Langfuse Connection**: Check that Langfuse is accessible at the configured host
4. **Missing PDFs**: Ensure PDF files are placed in the `pdf_documents/` directory

### Debug Mode

Set `DEBUG=True` in your environment to enable verbose logging and additional trace information.

## Performance Notes

- **Local Embeddings**: Uses BGE model locally for cost-effective embedding generation
- **Batch Processing**: Processes documents in optimized batches
- **Progress Tracking**: Shows real-time progress during ingestion
- **Memory Efficient**: Streams large documents to avoid memory issues

## Technical Details

### Embedding Model

- Uses `BAAI/bge-small-en-v1.5` local embedding model (384 dimensions)
- This is cost-effective and runs locally without API calls

### Text Processing

- Documents are split into chunks of 512 characters with 20 character overlap
- Uses LlamaIndex's `SentenceSplitter` for intelligent chunking

### Database Schema

- Table name: `charity_policies`
- Embedding dimension: 384
- Uses pgvector for similarity search

## Testing

Run the unit tests:

```bash
pytest tests/test_ingestion.py -v
```

Run integration tests (requires database):

```bash
python scripts/test_ingestion_integration.py
```
