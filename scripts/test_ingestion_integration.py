#!/usr/bin/env python3
"""
Integration test script for the ingestion pipeline.
This script requires a running PostgreSQL database with pgvector extension.
"""

import logging
import os
import sys

import psycopg2

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_database_connection():
    """Check if we can connect to the database."""
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        logger.info("✅ Database connection successful")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def check_pgvector_extension():
    """Check if pgvector extension is installed."""
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone()
        conn.close()

        if result:
            logger.info("✅ pgvector extension is installed")
            return True
        else:
            logger.error("❌ pgvector extension is not installed")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking pgvector extension: {e}")
        return False


def check_pdf_files():
    """Check if PDF files exist in the documents directory."""
    pdf_dir = "pdf_documents/"
    if not os.path.exists(pdf_dir):
        logger.error(f"❌ PDF directory does not exist: {pdf_dir}")
        return False

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logger.error(f"❌ No PDF files found in {pdf_dir}")
        return False

    logger.info(f"✅ Found {len(pdf_files)} PDF file(s): {pdf_files}")
    return True


def check_environment_variables():
    """Check if all required environment variables are set."""
    required_vars = [
        "DATABASE_URL",
        "OPENROUTER_API_KEY",
        "LLM_MODEL_NAME",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "API_KEY",
        "ADMIN_API_KEY",
    ]

    missing_vars = []
    for var in required_vars:
        if not hasattr(settings, var) or not getattr(settings, var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        return False

    logger.info("✅ All required environment variables are set")
    return True


def run_ingestion_test():
    """Run the actual ingestion script."""
    try:
        logger.info("🚀 Running ingestion script...")
        from scripts.ingest import main

        main()
        logger.info("✅ Ingestion completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return False


def check_ingestion_results():
    """Check if data was actually ingested into the database."""
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()

        # Check if the table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'charity_policies'
            );
        """)
        table_exists = cur.fetchone()[0]

        if not table_exists:
            logger.error("❌ Table 'charity_policies' does not exist")
            conn.close()
            return False

        # Check if there are any records
        cur.execute("SELECT COUNT(*) FROM charity_policies;")
        count = cur.fetchone()[0]
        conn.close()

        if count > 0:
            logger.info(f"✅ Found {count} records in charity_policies table")
            return True
        else:
            logger.error("❌ No records found in charity_policies table")
            return False

    except Exception as e:
        logger.error(f"❌ Error checking ingestion results: {e}")
        return False


def main():
    """Run the complete integration test."""
    logger.info("🧪 Starting ingestion integration test...")

    # Pre-flight checks
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Database Connection", check_database_connection),
        ("pgvector Extension", check_pgvector_extension),
        ("PDF Files", check_pdf_files),
    ]

    for check_name, check_func in checks:
        logger.info(f"Checking {check_name}...")
        if not check_func():
            logger.error(f"❌ Integration test failed at: {check_name}")
            sys.exit(1)

    # Run ingestion
    if not run_ingestion_test():
        logger.error("❌ Integration test failed during ingestion")
        sys.exit(1)

    # Verify results
    if not check_ingestion_results():
        logger.error("❌ Integration test failed during result verification")
        sys.exit(1)

    logger.info("🎉 Integration test completed successfully!")


if __name__ == "__main__":
    main()
