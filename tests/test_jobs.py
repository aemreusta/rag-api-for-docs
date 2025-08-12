import uuid

from celery import Celery

from app.core.jobs import (
    celery_app,
    cleanup_failed_jobs,
    process_document_async,
    scrape_url_async,
)


def test_celery_app_configured() -> None:
    assert isinstance(celery_app, Celery)
    # Basic config expectations
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert "json" in celery_app.conf.accept_content


def test_tasks_run_in_eager_mode(monkeypatch) -> None:
    # Force tasks to run synchronously without a broker
    monkeypatch.setattr(celery_app.conf, "task_always_eager", True, raising=False)
    monkeypatch.setattr(celery_app.conf, "task_eager_propagates", True, raising=False)

    job_id = str(uuid.uuid4())

    # process_document_async
    r1 = process_document_async.apply(args=[job_id, {"filename": "doc.pdf"}]).get()
    assert r1["job_id"] == job_id
    assert r1["status"] == "queued"

    # scrape_url_async
    r2 = scrape_url_async.apply(args=[job_id, "https://example.com", {"depth": 1}]).get()
    assert r2["job_id"] == job_id
    assert r2["status"] == "queued"

    # cleanup_failed_jobs
    r3 = cleanup_failed_jobs.apply().get()
    assert isinstance(r3, int)
    assert r3 == 0
