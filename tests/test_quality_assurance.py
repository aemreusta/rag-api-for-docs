from __future__ import annotations

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
