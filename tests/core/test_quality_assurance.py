from __future__ import annotations

import io
from unittest.mock import Mock, patch

import pytest
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.quality import QualityAssurance
from app.db.models import ContentEmbedding


def test_score_content_quality_basic():
    text = (
        "Bu bir örnek metindir. İçerik kalitesi ölçümü için birkaç cümle içerir. "
        "Ayrıca noktalama işaretleri de vardır: virgül, noktalı virgül; parantez ()."
    )
    score = QualityAssurance.score_content_quality(text)
    assert 0.0 <= score.score <= 1.0
    assert score.length_words > 5
    assert score.avg_sentence_length > 2


def test_validate_embeddings_shape():
    # Use current embedding dimension from settings
    dim = settings.EMBEDDING_DIM

    # Good single vector
    good_vec = [0.0] * dim
    ok_single = QualityAssurance.validate_embeddings(good_vec)
    # Good batch
    good_batch = [[0.1] * dim, [0.2] * dim]
    ok_batch = QualityAssurance.validate_embeddings(good_batch)
    # Bad dims
    bad_vec = [0.0] * 10
    bad = QualityAssurance.validate_embeddings(bad_vec)

    assert ok_single is True
    assert ok_batch is True
    assert bad is False


def test_validate_embeddings_non_finite():
    # Contains NaN and inf
    dim = settings.EMBEDDING_DIM
    vec_nan = [0.0] * (dim - 1) + [float("nan")]
    vec_inf = [0.0] * (dim - 1) + [float("inf")]
    assert QualityAssurance.validate_embeddings(vec_nan) is False
    assert QualityAssurance.validate_embeddings(vec_inf) is False


def test_check_vector_store_integrity(db_session: Session):
    # Initially empty
    empty_report = QualityAssurance.check_vector_store_integrity(db_session)
    assert empty_report.total_rows >= 0
    assert isinstance(empty_report.ok, bool)

    # Insert a row with NULL vector to trigger null_vectors > 0 and ok False
    db_session.add(ContentEmbedding(source_document="doc.pdf", page_number=1, content_text="hello"))
    db_session.commit()
    report_null = QualityAssurance.check_vector_store_integrity(db_session)
    assert report_null.total_rows == 1
    assert report_null.null_vectors == 1
    assert report_null.ok is False
    # HNSW index is created in conftest
    assert report_null.hnsw_index_present in (True, None)

    # Insert a correct-dimension vector row; vector type accepts Python list
    dim = settings.EMBEDDING_DIM
    db_session.add(
        ContentEmbedding(
            source_document="doc.pdf",
            page_number=2,
            content_text="world",
            content_vector=[0.0] * dim,
        )
    )
    db_session.commit()
    report_ok = QualityAssurance.check_vector_store_integrity(db_session)
    assert report_ok.total_rows == 2
    assert report_ok.null_vectors == 1
    assert report_ok.wrong_dims in (0, None)


# Enhanced Input Validation Tests
class TestEnhancedInputValidation:
    """Test enhanced input validation features."""

    def test_validate_upload_success_pdf(self):
        """Test successful PDF upload validation."""
        content = b"%PDF-1.4\nTest PDF content"
        file = UploadFile(filename="test.pdf", file=io.BytesIO(content))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=True,
                content_hash="test_hash",
                file_size_bytes=len(content),
                reason=[content],
            )

            with patch("app.core.quality._validate_mime_types") as mock_mime:
                mock_mime.return_value = Mock(
                    ok=True,
                    detected_mime_type="application/pdf",
                )

                import asyncio

                result = asyncio.get_event_loop().run_until_complete(
                    QualityAssurance.validate_upload(file)
                )
                assert result.ok is True
                assert result.content_hash == "test_hash"
                assert result.validation_details is not None

    def test_validate_upload_invalid_filename(self):
        """Test filename validation with dangerous patterns."""
        dangerous_filenames = [
            "test..pdf",
            "test<script>evil.pdf",
            "test/javascript:alert(1).pdf",
            "test/data:text/html,<script>alert(1)</script>.pdf",
            "test/file:///etc/passwd.pdf",
        ]

        for filename in dangerous_filenames:
            file = UploadFile(filename=filename, file=io.BytesIO(b"x"))
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                QualityAssurance.validate_upload(file)
            )
            assert result.ok is False
            assert result.reason == "dangerous_filename_pattern"

    def test_validate_upload_empty_file(self):
        """Test empty file rejection."""
        file = UploadFile(filename="empty.txt", file=io.BytesIO(b""))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=False,
                reason="empty_file",
                content_hash="empty_hash",
            )

            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                QualityAssurance.validate_upload(file)
            )
            assert result.ok is False
            assert result.reason == "empty_file"

    def test_validate_upload_file_too_large(self):
        """Test file size validation."""
        file = UploadFile(filename="large.pdf", file=io.BytesIO(b"x"))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=False,
                reason="too_large",
                file_size_bytes=100 * 1024 * 1024,  # 100MB
            )

            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                QualityAssurance.validate_upload(file)
            )
            assert result.ok is False
            assert result.reason == "too_large"

    def test_validate_upload_unsupported_mime_type(self):
        """Test unsupported MIME type rejection."""
        file = UploadFile(filename="test.exe", file=io.BytesIO(b"x"))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=True,
                content_hash="test_hash",
                file_size_bytes=1024,
                reason=[b"test content"],
            )

            with patch("app.core.quality._validate_mime_types") as mock_mime:
                mock_mime.return_value = Mock(
                    ok=False,
                    reason="unsupported_declared_type",
                )

                import asyncio

                result = asyncio.get_event_loop().run_until_complete(
                    QualityAssurance.validate_upload(file)
                )
                assert result.ok is False
                assert result.reason == "unsupported_declared_type"

    def test_validate_upload_mime_mismatch(self):
        """Test MIME type mismatch detection."""
        file = UploadFile(filename="test.txt", file=io.BytesIO(b"x"))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=True,
                content_hash="test_hash",
                file_size_bytes=1024,
                reason=[b"test content"],
            )

            with patch("app.core.quality._validate_mime_types") as mock_mime:
                mock_mime.return_value = Mock(
                    ok=False,
                    reason="mime_mismatch",
                )

                import asyncio

                result = asyncio.get_event_loop().run_until_complete(
                    QualityAssurance.validate_upload(file)
                )
                assert result.ok is False
                assert result.reason == "mime_mismatch"

    def test_validate_upload_suspicious_content(self):
        """Test content security scanning."""
        file = UploadFile(filename="test.txt", file=io.BytesIO(b"x"))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=False,
                reason="suspicious_content_detected",
                content_hash="test_hash",
            )

            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                QualityAssurance.validate_upload(file)
            )
            assert result.ok is False
            assert result.reason == "suspicious_content_detected"

    def test_validate_upload_pdf_javascript_detected(self):
        """Test PDF JavaScript detection."""
        file = UploadFile(filename="test.pdf", file=io.BytesIO(b"%PDF-1.4\n"))

        with patch("app.core.quality._read_and_validate_content") as mock_read:
            mock_read.return_value = Mock(
                ok=False,
                reason="pdf_javascript_detected",
                content_hash="test_hash",
            )

            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                QualityAssurance.validate_upload(file)
            )
            assert result.ok is False
            assert result.reason == "pdf_javascript_detected"

    @pytest.mark.parametrize(
        "filename,expected_mime",
        [
            ("test.pdf", "application/pdf"),
            ("test.txt", "text/plain"),
            ("test.md", "text/markdown"),
            (
                "test.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("test.csv", "text/csv"),
            ("test.json", "application/json"),
            ("test.unknown", None),
        ],
    )
    def test_mime_detection_by_extension(self, filename, expected_mime):
        """Test MIME type detection by file extension."""
        from app.core.quality import _detect_mime_by_extension

        detected = _detect_mime_by_extension(filename)
        assert detected == expected_mime

    def test_mime_type_compatibility(self):
        """Test MIME type compatibility checking."""
        from app.core.quality import _is_mime_compatible

        # Compatible pairs
        assert _is_mime_compatible("text/plain", "text/markdown") is True
        assert _is_mime_compatible("text/markdown", "text/plain") is True
        assert _is_mime_compatible("text/plain", "text/csv") is True

        # Incompatible pairs
        assert _is_mime_compatible("application/pdf", "text/plain") is False
        assert _is_mime_compatible("text/plain", "application/json") is False

        # Exact matches
        assert _is_mime_compatible("application/pdf", "application/pdf") is True

    def test_validation_result_with_details(self):
        """Test ValidationResult with enhanced validation details."""
        from app.core.quality import ValidationResult

        details = {
            "file_size_mb": 2.5,
            "content_hash_short": "abc123...",
            "mime_type_validated": True,
        }

        result = ValidationResult(
            ok=True,
            content_hash="abc123def456",
            file_size_bytes=2621440,
            validation_details=details,
        )

        assert result.ok is True
        assert result.validation_details == details
        assert result.validation_details["file_size_mb"] == 2.5


class TestConfigurationBasedValidation:
    """Test validation behavior based on configuration settings."""

    @patch("app.core.quality.settings")
    def test_allowed_mime_types_from_config(self, mock_settings):
        """Test that allowed MIME types are loaded from configuration."""
        from app.core.quality import _parse_allowed_mime_types

        mock_settings.ALLOWED_MIME_TYPES = "application/pdf,text/plain,image/png"
        allowed_types = _parse_allowed_mime_types()

        assert "application/pdf" in allowed_types
        assert "text/plain" in allowed_types
        assert "image/png" in allowed_types

    @patch("app.core.quality.settings")
    def test_max_file_size_from_config(self, mock_settings):
        """Test that maximum file size is loaded from configuration."""
        from app.core.quality import _get_max_file_size

        # Test MB-based configuration
        mock_settings.MAX_FILE_SIZE_MB = 25
        max_size = _get_max_file_size()
        assert max_size == 25 * 1024 * 1024

        # Test bytes-based configuration
        mock_settings.MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
        max_size = _get_max_file_size()
        assert max_size == 50 * 1024 * 1024

        # Test fallback
        del mock_settings.MAX_FILE_SIZE_MB
        del mock_settings.MAX_FILE_SIZE_BYTES
        max_size = _get_max_file_size()
        assert max_size == 10 * 1024 * 1024  # Default 10MB

    @patch("app.core.quality.settings")
    def test_content_scanning_configurable(self, mock_settings):
        """Test that content scanning can be disabled via configuration."""
        from app.core.quality import _scan_content_security

        # Enable scanning
        mock_settings.ENABLE_CONTENT_SCANNING = True
        result = _scan_content_security(b"<script>evil</script>", "test.txt")
        assert result.ok is False
        assert result.reason == "suspicious_content_detected"

        # Disable scanning
        mock_settings.ENABLE_CONTENT_SCANNING = False
        result = _scan_content_security(b"<script>evil</script>", "test.txt")
        assert result.ok is True

    @patch("app.core.quality.settings")
    def test_magic_detection_configurable(self, mock_settings):
        """Test that magic detection can be disabled via configuration."""
        from app.core.quality import _validate_mime_types

        file = Mock()
        file.content_type = "application/pdf"

        # Enable magic detection
        mock_settings.ENABLE_MAGIC_DETECTION = True
        with patch("app.core.quality.MAGIC_AVAILABLE", True):
            # Ensure magic has from_buffer attribute even if module is a dummy namespace
            from app.core import quality as quality_mod

            if not hasattr(quality_mod.magic, "from_buffer"):
                quality_mod.magic.from_buffer = lambda *_a, **_k: "application/pdf"
            with patch("app.core.quality.magic.from_buffer") as mock_magic:
                mock_magic.return_value = "application/pdf"
                result = _validate_mime_types(file, "test.pdf", [b"test"], "hash", 1024)
                assert result.ok is True

        # Disable magic detection
        mock_settings.ENABLE_MAGIC_DETECTION = False
        result = _validate_mime_types(file, "test.pdf", [b"test"], "hash", 1024)
        assert result.ok is True  # Falls back to extension detection
