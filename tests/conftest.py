import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

# Ensure repository root is on sys.path before importing app modules (E402 compliant)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_configure(config):
    """Register custom markers used across the test-suite."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests that may require external services",
    )
    config.addinivalue_line(
        "markers",
        "minio: marks tests that exercise the MinIO/S3 storage path",
    )


# Path already configured above


@pytest.fixture(scope="session")
def db_engine():
    """Create a database engine that is aware of the containerized environment."""
    # Construct the URL from the container-aware settings, ensuring tests
    # connect to the 'postgres' service, not 'localhost'.
    test_db_url = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@"
        f"{settings.POSTGRES_SERVER}:5432/{settings.POSTGRES_DB}"
    )
    engine = create_engine(test_db_url)
    return engine


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create database session for testing."""
    # Create a new session for each test
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()

    try:
        # Ensure a clean schema that matches current settings (drop before create)
        Base.metadata.drop_all(bind=db_engine)
        # Ensure pgvector extension is available for tests
        with db_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # Create tables if they don't exist (for testing)
        Base.metadata.create_all(bind=db_engine)

        # Ensure HNSW index exists and tune session parameter for recall
        with db_engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS content_embeddings_vector_hnsw_idx
                    ON content_embeddings
                    USING hnsw (content_vector vector_l2_ops)
                    WITH (m = 32, ef_construction = 64);
                    """
                )
            )
            # Set per-session ef_search for tests to meet recall target
            conn.execute(text("SET hnsw.ef_search = 100;"))
            conn.commit()
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session", autouse=True)
def _setup_db_schema(db_engine):
    """Ensure core tables and extensions exist for API integration tests.

    This runs once per test session so endpoints using the default
    `get_db_session` dependency can operate against real tables without
    per-test overrides.
    """
    # Ensure pgvector extension is available
    with db_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    # Create ORM tables (idempotent)
    Base.metadata.create_all(bind=db_engine)
    # Ensure HNSW index exists and tune ef_search for recall in tests
    with db_engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS content_embeddings_vector_hnsw_idx
                ON content_embeddings
                USING hnsw (content_vector vector_l2_ops)
                WITH (m = 32, ef_construction = 64);
                """
            )
        )
        conn.execute(text("SET hnsw.ef_search = 100;"))
        conn.commit()
