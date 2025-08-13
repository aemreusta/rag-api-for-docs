from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.quality import QualityAssurance


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
    # Good single vector
    good_vec = [0.0] * 1536
    ok_single = QualityAssurance.validate_embeddings(good_vec)
    # Good batch
    good_batch = [[0.1] * 1536, [0.2] * 1536]
    ok_batch = QualityAssurance.validate_embeddings(good_batch)
    # Bad dims
    bad_vec = [0.0] * 10
    bad = QualityAssurance.validate_embeddings(bad_vec)

    assert ok_single is True
    assert ok_batch is True
    assert bad is False


def test_check_vector_store_integrity(db_session: Session):
    # Should not raise; returns an IntegrityReport
    report = QualityAssurance.check_vector_store_integrity(db_session)
    assert report.total_rows >= 0
    assert report.null_vectors >= 0
    assert isinstance(report.ok, bool)
