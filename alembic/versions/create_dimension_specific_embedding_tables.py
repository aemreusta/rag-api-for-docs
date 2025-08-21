"""create_dimension_specific_embedding_tables

Revision ID: create_dimension_tables
Revises: b37cd2b68a75
Create Date: 2025-01-20 12:00:00.000000

This migration creates separate embedding tables for different dimensions to handle
PostgreSQL pgvector HNSW index limitation of 2000 dimensions per column.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "create_dimension_tables"
down_revision: str | Sequence[str] | None = "b5889e3a7cbb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create dimension-specific embedding tables."""

    # Ensure pgvector extension is available
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Define dimension configurations for different providers
    dimension_configs = [
        {"dims": 768, "name": "embeddings_768", "provider": "huggingface_small"},
        {"dims": 1024, "name": "embeddings_1024", "provider": "qwen"},
        {"dims": 1536, "name": "embeddings_1536", "provider": "google_mrl"},
        {"dims": 3072, "name": "embeddings_3072", "provider": "google_full"},
    ]

    for config in dimension_configs:
        dims = config["dims"]
        table_name = config["name"]
        provider = config["provider"]

        # Create table with appropriate vector dimension
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                source_document VARCHAR(255) NOT NULL,
                page_number INTEGER NOT NULL,
                content_text TEXT NOT NULL,
                content_vector VECTOR({dims}),
                provider VARCHAR(50) DEFAULT '{provider}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create indexes
        op.execute(f"""
            CREATE INDEX idx_{table_name}_source_document ON {table_name} (source_document);
        """)
        op.execute(f"""
            CREATE INDEX idx_{table_name}_page_number ON {table_name} (page_number);
        """)
        op.execute(f"""
            CREATE INDEX idx_{table_name}_provider ON {table_name} (provider);
        """)

        # Create appropriate vector index based on dimension limit
        if dims <= 2000:
            # HNSW index for dimensions ≤ 2000
            with op.get_context().autocommit_block():
                op.execute(f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS {table_name}_vector_hnsw_idx
                    ON {table_name}
                    USING hnsw (content_vector vector_l2_ops)
                    WITH (m = 32, ef_construction = 64);
                """)
        else:
            # Regular B-tree index for dimensions > 2000 (like 3072)
            # Note: No specialized vector index available for >2000 dimensions in pgvector
            # Will use sequential scan for similarity search on these tables
            op.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_vector_gin_idx
                ON {table_name}
                USING gin (to_tsvector('english', content_text));
            """)

            print(f"Warning: Table {table_name} with {dims} dimensions cannot use vector indexes.")
            print(
                "Similarity search will use sequential scans. "
                "Consider using smaller dimensions for better performance."
            )

    # Migrate existing data from content_embeddings to appropriate dimension table
    # First, check if there's data to migrate
    op.execute("""
        INSERT INTO embeddings_1536 (
            source_document,
            page_number,
            content_text,
            content_vector,
            provider
        )
        SELECT
            source_document,
            page_number,
            content_text,
            content_vector,
            'legacy'
        FROM content_embeddings
        WHERE content_vector IS NOT NULL;
    """)

    # Create view for backward compatibility
    op.execute("""
        CREATE VIEW content_embeddings_view AS
        SELECT id, source_document, page_number, content_text, content_vector,
               'embeddings_768' as table_name, provider, created_at, updated_at
        FROM embeddings_768
        UNION ALL
        SELECT id, source_document, page_number, content_text, content_vector,
               'embeddings_1024' as table_name, provider, created_at, updated_at
        FROM embeddings_1024
        UNION ALL
        SELECT id, source_document, page_number, content_text, content_vector,
               'embeddings_1536' as table_name, provider, created_at, updated_at
        FROM embeddings_1536
        UNION ALL
        SELECT id, source_document, page_number, content_text, content_vector,
               'embeddings_3072' as table_name, provider, created_at, updated_at
        FROM embeddings_3072;
    """)


def downgrade() -> None:
    """Drop dimension-specific tables and restore original structure."""

    # Drop the view
    op.execute("DROP VIEW IF EXISTS content_embeddings_view;")

    # Drop dimension-specific tables and their indexes
    dimension_tables = ["embeddings_768", "embeddings_1024", "embeddings_1536", "embeddings_3072"]

    for table_name in dimension_tables:
        # Drop indexes first
        op.execute(f"DROP INDEX IF EXISTS {table_name}_vector_hnsw_idx;")
        op.execute(f"DROP INDEX IF EXISTS {table_name}_vector_ivfflat_idx;")
        op.execute(f"DROP INDEX IF EXISTS idx_{table_name}_source_document;")
        op.execute(f"DROP INDEX IF EXISTS idx_{table_name}_page_number;")
        op.execute(f"DROP INDEX IF EXISTS idx_{table_name}_provider;")

        # Drop table
        op.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

    # Note: Original content_embeddings table is preserved
    print("Downgrade completed. Original content_embeddings table preserved.")
