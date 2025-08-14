# Parallel Ingest Deployment Runbook

This runbook describes how to run the legacy ingestion scripts side-by-side with the new ingestion API, verify parity, and roll back safely if needed.

## Prerequisites

- Docker Compose stack running: `make up`
- Database prepared with pgvector schema and indexes
- `.env` configured

## Enable dual-run mode

1. In `.env`, set:

   ```env
   INGEST_PARALLEL_DEPLOYMENT=true
   ```

2. Restart app to pick up config (if running outside compose).

## Execution paths

- Legacy path removed. Use the new API endpoints exclusively (`/api/v1/docs/*`).
- New: use API endpoints under `/api/v1/docs/*` (e.g., upload via `/api/v1/docs/upload`).

## Verification checklist

- Upload the same document via API and via legacy script.
- Validate:
  - Rows inserted in `content_embeddings` are similar in count.
  - Embedding dims equal `settings.EMBEDDING_DIM`.
  - No null vectors.
  - HNSW index present on `content_embeddings.content_vector`.
  - Logs include `request_id` and `trace_id` for API path.

## Metrics & logging

- Confirm Prometheus metrics exposed on `/metrics` include ingest counters.
- Confirm structured logs for both paths (legacy writes minimal logs).

## Rollback

- Set `INGEST_PARALLEL_DEPLOYMENT=false` to disable dual-run.
- Continue with legacy scripts if issues are found.

## Cutover

- Freeze legacy scripts (see cleanup chore branch) once checklist passes in staging and prod.
