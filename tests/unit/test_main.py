from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "version" in data


def test_metrics_enabled():
    """Test metrics endpoint when PROMETHEUS_ENABLED is true and prometheus_client is available."""
    with patch("app.main.settings.PROMETHEUS_ENABLED", True):
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = True  # prometheus_client is available
            # Provide a shim module if prometheus_client is not installed
            import sys
            from types import SimpleNamespace

            sys.modules.setdefault(
                "prometheus_client",
                SimpleNamespace(generate_latest=lambda: b"# HELP ingest_requests_total\n"),
            )
            with patch("prometheus_client.generate_latest") as mock_generate_latest:
                mock_generate_latest.return_value = (
                    b"# HELP ingest_requests_total Total number of document ingest requests\n"
                )

                response = client.get("/metrics")
                assert response.status_code == 200
                assert (
                    response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
                )
                assert b"ingest_requests_total" in response.content


def test_metrics_disabled():
    """Test metrics endpoint when PROMETHEUS_ENABLED is false."""
    with patch("app.main.settings.PROMETHEUS_ENABLED", False):
        response = client.get("/metrics")
        assert response.status_code == 404
        assert "Metrics endpoint not enabled" in response.json()["detail"]


def test_metrics_prometheus_unavailable():
    """Test metrics endpoint when prometheus_client is not available."""
    with patch("app.main.settings.PROMETHEUS_ENABLED", True):
        with patch(
            "importlib.util.find_spec", return_value=None
        ):  # prometheus_client not available
            response = client.get("/metrics")
            assert response.status_code == 404
            assert "Prometheus client not available" in response.json()["detail"]
