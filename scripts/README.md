# Data Ingestion Scripts

This directory contains scripts for ingesting PDF documents into the PostgreSQL vector database.

## Files

- `ingest.py` - Main ingestion script that processes PDFs and stores them in PostgreSQL
- `test_ingestion_integration.py` - Integration test script to verify the ingestion pipeline
- `create_sample_pdf.py` - Utility script to create sample PDF files for testing

## Prerequisites

1. **PostgreSQL Database** with pgvector extension installed
2. **Environment Variables** set in `.env` file (see `.env.example`)
3. **PDF Documents** placed in the `pdf_documents/` directory

## Usage

### 1. Basic Ingestion

```bash
python scripts/ingest.py
```

This will:

- Check for pgvector extension in the database
- Load all PDF files from `pdf_documents/` directory
- Create embeddings using the local BGE model
- Store the embeddings in PostgreSQL table `charity_policies`

### 2. Integration Testing

```bash
python scripts/test_ingestion_integration.py
```

This will:

- Verify all prerequisites are met
- Run the ingestion process
- Validate that data was successfully stored

### 3. Create Sample PDF

```bash
python scripts/create_sample_pdf.py
```

This creates a sample PDF file for testing purposes.

## Configuration

The ingestion script uses the following configuration from `app/core/config.py`:

- `DATABASE_URL` - PostgreSQL connection string
- `OPENROUTER_API_KEY` - API key for OpenRouter LLM service
- `LLM_MODEL_NAME` - Model name (default: "google/gemini-1.5-pro-latest")

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

## Troubleshooting

### Common Issues

1. **pgvector extension not found**

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Database connection failed**
   - Check `DATABASE_URL` in your `.env` file
   - Ensure PostgreSQL is running

3. **No PDF files found**
   - Place PDF files in the `pdf_documents/` directory
   - Ensure files have `.pdf` extension

4. **Memory issues with large PDFs**
   - Consider reducing chunk size in the script
   - Process files in smaller batches

## Testing

Run the unit tests:

```bash
pytest tests/test_ingestion.py -v
```

Run integration tests (requires database):

```bash
python scripts/test_ingestion_integration.py
```
