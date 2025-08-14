# Legacy Ingest Scripts (Deprecated)

These scripts are maintained temporarily during the migration to the new ingestion API under `/api/v1/docs/*`.

- Preferred path: use API endpoints (`POST /api/v1/docs/upload`, etc.)
- Temporary dual-run: set `INGEST_PARALLEL_DEPLOYMENT=true` in `.env`
- Runbook: see `docs/ops/parallel_ingest_runbook.md`

Files:

- `ingest.py` — legacy full ingest flow
- `ingest_simple.py` — simplified ingest with retrying embeddings

Both scripts will be removed after migration completes.
