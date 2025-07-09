import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def db_engine():
    """Create database engine for testing."""
    engine = create_engine(settings.DATABASE_URL)
    return engine


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create database session for testing."""
    # Create a new session for each test
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()

    try:
        # Create tables if they don't exist (for testing)
        Base.metadata.create_all(bind=db_engine)
        yield session
    finally:
        session.close()
