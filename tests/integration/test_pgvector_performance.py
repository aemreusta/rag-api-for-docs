"""
Integration tests for pgvector performance with HNSW index.
Tests vector search latency and accuracy targets.
"""

import time

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ContentEmbedding


class TestPgVectorPerformance:
    """Test suite for pgvector HNSW index performance."""

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """Generate sample embeddings matching settings.EMBEDDING_DIM for testing."""
        np.random.seed(42)  # Deterministic for reproducible tests
        # Generate 100 sample embeddings for performance testing
        embeddings = []
        for _i in range(100):
            # Create normalized random vectors (typical for embeddings)
            vec = np.random.normal(0, 1, settings.EMBEDDING_DIM)
            vec = vec / np.linalg.norm(vec)  # Normalize to unit length
            embeddings.append(vec.tolist())
        return embeddings

    @pytest.fixture
    def populate_test_data(self, db_session: Session, sample_embeddings: list[list[float]]) -> None:
        """Populate database with test embeddings."""
        # Clear existing data
        db_session.execute(text("DELETE FROM content_embeddings"))

        # Insert test embeddings
        for i, embedding in enumerate(sample_embeddings):
            content = ContentEmbedding(
                source_document=f"test_doc_{i % 10}.pdf",
                page_number=i % 5 + 1,
                content_text=f"Test content for embedding {i}",
                content_vector=str(embedding),  # pgvector accepts string representation
            )
            db_session.add(content)

        db_session.commit()

    def test_vector_search_latency(
        self, db_session: Session, populate_test_data: None, sample_embeddings: list[list[float]]
    ) -> None:
        """Test that vector similarity search meets p99 < 50ms latency target."""
        # Use first embedding as query vector
        query_vector = sample_embeddings[0]

        # Perform multiple searches to get reliable latency measurements
        latencies = []
        num_trials = 20

        for _ in range(num_trials):
            start_time = time.perf_counter()

            # Execute similarity search using HNSW index
            result = db_session.execute(
                text("""
                    SELECT id, source_document, page_number, content_text,
                           content_vector <-> :query_vector AS distance
                    FROM content_embeddings
                    ORDER BY content_vector <-> :query_vector
                    LIMIT 10
                """),
                {"query_vector": str(query_vector)},
            )

            results = result.fetchall()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Verify we got results
            assert len(results) > 0, "Should return similarity search results"
            assert len(results) <= 10, "Should respect LIMIT clause"

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg_latency = sum(latencies) / len(latencies)

        print("\nVector Search Performance Results:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  P50 latency: {p50:.2f}ms")
        print(f"  P95 latency: {p95:.2f}ms")
        print(f"  P99 latency: {p99:.2f}ms")

        # Assert performance targets
        assert p99 < 50.0, f"P99 latency {p99:.2f}ms exceeds 50ms target"
        assert avg_latency < 20.0, f"Average latency {avg_latency:.2f}ms exceeds 20ms target"

    def test_hnsw_index_exists(self, db_session: Session) -> None:
        """Verify that HNSW index exists and is being used."""
        # Check index exists
        result = db_session.execute(
            text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'content_embeddings'
                AND indexname = 'content_embeddings_vector_hnsw_idx'
            """)
        )
        index_info = result.fetchone()

        assert index_info is not None, "HNSW index should exist"
        assert "hnsw" in index_info.indexdef, "Index should use HNSW method"
        assert "m='32'" in index_info.indexdef, "Index should have m=32 parameter"
        assert "ef_construction='64'" in index_info.indexdef, "Index should have ef_construction=64"

    def test_vector_column_type(self, db_session: Session) -> None:
        """Verify content_vector column is vector type (pgvector)."""
        result = db_session.execute(
            text("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'content_embeddings'
                AND column_name = 'content_vector'
            """)
        )
        column_info = result.fetchone()

        assert column_info is not None, "content_vector column should exist"
        # PostgreSQL reports vector type as USER-DEFINED
        assert column_info.data_type == "USER-DEFINED", "Column should be vector type"

    def test_ef_search_setting(self, db_session: Session) -> None:
        """Verify hnsw.ef_search is configured for optimal recall."""
        result = db_session.execute(text("SHOW hnsw.ef_search"))
        ef_search_value = result.fetchone()

        if ef_search_value is not None:
            # Check if ef_search is set to our target value
            assert int(ef_search_value[0]) >= 100, "ef_search should be >= 100 for good recall"

    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_search_accuracy_recall(
        self,
        db_session: Session,
        populate_test_data: None,
        sample_embeddings: list[list[float]],
        k: int,
    ) -> None:
        """Test that HNSW index maintains high recall accuracy."""
        query_vector = sample_embeddings[0]

        # Get exact (brute force) results by temporarily disabling index
        db_session.execute(text("SET enable_indexscan = off"))
        exact_result = db_session.execute(
            text("""
                SELECT id, content_vector <-> :query_vector AS distance
                FROM content_embeddings
                ORDER BY content_vector <-> :query_vector
                LIMIT :k
            """),
            {"query_vector": str(query_vector), "k": k},
        )
        exact_ids = {row.id for row in exact_result.fetchall()}

        # Re-enable index and get HNSW results
        db_session.execute(text("SET enable_indexscan = on"))
        hnsw_result = db_session.execute(
            text("""
                SELECT id, content_vector <-> :query_vector AS distance
                FROM content_embeddings
                ORDER BY content_vector <-> :query_vector
                LIMIT :k
            """),
            {"query_vector": str(query_vector), "k": k},
        )
        hnsw_ids = {row.id for row in hnsw_result.fetchall()}

        # Calculate recall
        intersection = len(exact_ids.intersection(hnsw_ids))
        recall = intersection / len(exact_ids) if exact_ids else 0

        print(f"\nRecall@{k}: {recall:.3f}")

        # Assert recall target (≥95% for top-k results)
        assert recall >= 0.95, f"Recall@{k} {recall:.3f} below 95% target"
