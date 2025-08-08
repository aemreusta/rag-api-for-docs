"""fix_pgvector_type_and_add_hnsw_index

Revision ID: b37cd2b68a75
Revises: 27cfcd70f4db
Create Date: 2025-07-08 07:32:14.612859

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b37cd2b68a75"
down_revision: str | Sequence[str] | None = "27cfcd70f4db"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Ensure pgvector extension is available (required for VECTOR type and HNSW)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Convert content_vector from TEXT to VECTOR(384)
    # Since DB is empty (development), we can do direct ALTER TYPE
    op.execute("""
        ALTER TABLE content_embeddings
        ALTER COLUMN content_vector TYPE VECTOR(384)
        USING CASE
            WHEN content_vector IS NULL THEN NULL
            ELSE content_vector::vector
        END;
    """)

    # 3. Create HNSW index with optimized parameters for development
    # m=32, ef_construction=64 balance performance vs accuracy
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS content_embeddings_vector_hnsw_idx
        ON content_embeddings
        USING hnsw (content_vector vector_l2_ops)
        WITH (m = 32, ef_construction = 64);
    """)

    # 4. Set query-time parameter for optimal recall (can be overridden per session)
    op.execute("ALTER DATABASE app SET hnsw.ef_search = 100;")


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Drop HNSW index
    op.execute("DROP INDEX IF EXISTS content_embeddings_vector_hnsw_idx;")

    # 2. Reset database parameter
    op.execute("ALTER DATABASE app RESET hnsw.ef_search;")

    # 3. Convert back to TEXT type
    op.execute("""
        ALTER TABLE content_embeddings
        ALTER COLUMN content_vector TYPE TEXT
        USING content_vector::text;
    """)
