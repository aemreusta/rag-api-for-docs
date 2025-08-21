"""
Embedding storage service with dimension-aware table routing.

Handles storing and retrieving embeddings from appropriate dimension-specific tables
to work around PostgreSQL pgvector HNSW index limitations.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.embedding_models import (
    DIMENSION_TABLE_MAP,
    get_embedding_table_for_dimension,
)

logger = get_logger(__name__)


class EmbeddingStorage:
    """Service for storing and retrieving embeddings with dimension-aware routing."""

    def __init__(self, session: Session):
        self.session = session

    def store_embeddings(
        self,
        embeddings_data: list[tuple[str, int, str, list[float]]],
        provider: str,
    ) -> int:
        """
        Store embeddings in the appropriate dimension-specific table.

        Args:
            embeddings_data: List of (source_document, page_number, content_text, embedding_vector)
            provider: Embedding provider name

        Returns:
            Number of embeddings stored
        """
        if not embeddings_data:
            return 0

        # Determine dimension from first embedding
        first_embedding = embeddings_data[0][3]
        dimension = len(first_embedding)

        # Get appropriate table class
        table_class = get_embedding_table_for_dimension(dimension)
        table_name = table_class.__tablename__

        logger.info(
            "Storing embeddings in dimension-specific table",
            provider=provider,
            dimension=dimension,
            table_name=table_name,
            count=len(embeddings_data),
        )

        # Prepare records for bulk insert
        records = []
        for source_document, page_number, content_text, embedding_vector in embeddings_data:
            record = table_class(
                source_document=source_document,
                page_number=page_number,
                content_text=content_text,
                content_vector=embedding_vector,
                provider=provider,
            )
            records.append(record)

        # Bulk insert
        self.session.add_all(records)
        self.session.commit()

        logger.info(
            "Successfully stored embeddings",
            provider=provider,
            dimension=dimension,
            table_name=table_name,
            count=len(records),
        )

        return len(records)

    def search_similar_embeddings(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.7,
        source_document: str | None = None,
        provider_preference: list[str] | None = None,
    ) -> list[tuple[str, int, str, float]]:
        """
        Search for similar embeddings across dimension-appropriate tables.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity threshold
            source_document: Filter by source document (optional)
            provider_preference: Preferred provider order (optional)

        Returns:
            List of (source_document, page_number, content_text, similarity_score)
        """
        dimension = len(query_embedding)
        table_class = get_embedding_table_for_dimension(dimension)
        table_name = table_class.__tablename__

        logger.debug(
            "Searching similar embeddings",
            dimension=dimension,
            table_name=table_name,
            limit=limit,
            min_similarity=min_similarity,
        )

        # Build SQL query for vector similarity search
        where_clause = ""
        params = {
            "query_embedding": query_embedding,
            "limit": limit,
            "min_similarity": min_similarity,
        }

        if source_document:
            where_clause = "AND source_document = :source_document"
            params["source_document"] = source_document

        # Use cosine similarity with L2 distance
        # cosine_similarity = 1 - (embedding <-> query)
        query = f"""
        SELECT
            source_document,
            page_number,
            content_text,
            (1 - (content_vector <-> :query_embedding::vector)) AS similarity_score
        FROM {table_name}
        WHERE content_vector IS NOT NULL
        {where_clause}
        AND (1 - (content_vector <-> :query_embedding::vector)) >= :min_similarity
        ORDER BY content_vector <-> :query_embedding::vector
        LIMIT :limit
        """

        result = self.session.execute(text(query), params)
        results = result.fetchall()

        logger.info(
            "Found similar embeddings",
            dimension=dimension,
            table_name=table_name,
            results_count=len(results),
            min_similarity=min_similarity,
        )

        return [(row[0], row[1], row[2], float(row[3])) for row in results]

    def get_embeddings_count_by_provider(self) -> dict[str, dict[str, int]]:
        """Get count of embeddings by provider and dimension."""
        counts = {}

        for dimension, table_class in DIMENSION_TABLE_MAP.items():
            table_name = table_class.__tablename__

            query = f"""
            SELECT provider, COUNT(*) as count
            FROM {table_name}
            WHERE content_vector IS NOT NULL
            GROUP BY provider
            """

            result = self.session.execute(text(query))

            for row in result:
                provider = row[0]
                count = row[1]

                if provider not in counts:
                    counts[provider] = {}

                counts[provider][f"{dimension}d"] = count

        return counts

    def cleanup_orphaned_embeddings(self, dry_run: bool = True) -> dict[str, int]:
        """Remove embeddings without corresponding documents (maintenance operation)."""
        cleanup_results = {}

        for _dimension, table_class in DIMENSION_TABLE_MAP.items():
            table_name = table_class.__tablename__

            # Find orphaned embeddings
            query = f"""
            SELECT COUNT(*)
            FROM {table_name} e
            LEFT JOIN documents d ON d.filename = e.source_document
            WHERE d.id IS NULL
            """

            result = self.session.execute(text(query))
            orphaned_count = result.scalar()

            if dry_run:
                cleanup_results[f"{table_name}_orphaned"] = orphaned_count
            else:
                # Actually delete orphaned records
                delete_query = f"""
                DELETE FROM {table_name}
                WHERE id IN (
                    SELECT e.id
                    FROM {table_name} e
                    LEFT JOIN documents d ON d.filename = e.source_document
                    WHERE d.id IS NULL
                )
                """

                result = self.session.execute(text(delete_query))
                deleted_count = result.rowcount
                self.session.commit()

                cleanup_results[f"{table_name}_deleted"] = deleted_count

                logger.info(
                    "Cleaned up orphaned embeddings",
                    table_name=table_name,
                    deleted_count=deleted_count,
                )

        return cleanup_results

    def migrate_embeddings_between_tables(
        self,
        from_dimension: int,
        to_dimension: int,
        provider: str,
        limit: int | None = None,
    ) -> int:
        """
        Migrate embeddings between dimension-specific tables.

        This is useful when changing embedding dimensions for a provider.
        Note: This requires re-generating embeddings with the new dimension.
        """
        from_table_class = get_embedding_table_for_dimension(from_dimension)
        to_table_class = get_embedding_table_for_dimension(to_dimension)

        from_table = from_table_class.__tablename__
        to_table = to_table_class.__tablename__

        if from_table == to_table:
            logger.warning(
                "Source and target tables are the same, no migration needed",
                from_dimension=from_dimension,
                to_dimension=to_dimension,
            )
            return 0

        # This is a placeholder - actual migration requires re-embedding
        logger.warning(
            "Embedding migration requires re-generating embeddings with new dimensions",
            from_table=from_table,
            to_table=to_table,
            provider=provider,
        )

        return 0


def get_embedding_storage(session: Session) -> EmbeddingStorage:
    """Factory function to create embedding storage service."""
    return EmbeddingStorage(session)
