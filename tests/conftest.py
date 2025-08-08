import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)


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
        # Ensure pgvector extension is available for tests
        with db_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        # Create tables if they don't exist (for testing)
        Base.metadata.create_all(bind=db_engine)
        yield session
    finally:
        session.close()
