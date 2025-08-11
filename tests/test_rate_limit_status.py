import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize("count, ttl, expected_remaining", [(0, 10, 100), (5, 9, 95), (100, 5, 0)])
@patch("app.api.v1.status.settings.RATE_LIMIT_COUNT", 100)
@patch("app.api.v1.status.settings.RATE_LIMIT_WINDOW_SECONDS", 60)
def test_rate_limit_status_endpoint(expected_remaining, count, ttl):
    # Dummy Redis client that provides an async pipeline context manager
    class DummyRedis:
        def __init__(self, c: int, t: int):
            self._c = c
            self._t = t

        def pipeline(self):
            c = self._c
            t = self._t

            class _Pipe:
                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

                def get(self_inner, key):
                    return self_inner

                def ttl(self_inner, key):
                    return self_inner

                async def execute(self_inner):
                    return [c if c != 0 else None, t]

            return _Pipe()

    with patch("app.api.v1.status.redis_client", new=DummyRedis(count, ttl)):
        client = TestClient(app)
        r = client.get("/api/v1/rate-limit/status")
    assert r.status_code == 200

    data = r.json()
    assert data["limit"] == 100
    assert data["remaining"] == expected_remaining
    assert isinstance(data["reset"], int)
    # Should be within ttl seconds window from now (allow small drift)
    assert int(time.time()) <= data["reset"] <= int(time.time()) + ttl + 1

    # Cache-Control should be set to avoid thrashing Redis
    assert r.headers.get("Cache-Control") == "public, max-age=5"
