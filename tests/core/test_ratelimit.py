# fakeredis >=2.0 renamed "fakeredis.aioredis" to "fakeredis.asyncio".
# Import using the new path but fall back to the old one for compatibility.
try:
    import fakeredis.asyncio as fakeredis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – back-compat with <2.x
    import fakeredis.aioredis as fakeredis  # type: ignore
import pytest
from fastapi import Request
from starlette.exceptions import HTTPException

from app.core.ratelimit import RateLimiter

# Set all async tests to be asyncio mode
pytestmark = pytest.mark.asyncio


@pytest.fixture
def fake_redis():
    """Fixture to provide a fake redis client for testing."""
    return fakeredis.FakeRedis()


@pytest.fixture
def mock_request():
    """Fixture to provide a mock request object."""
    return Request({"type": "http", "client": ("127.0.0.1", 12345)})


async def test_rate_limiter_allows_under_limit(fake_redis, mock_request):
    """Test that the rate limiter allows requests under the limit."""
    limiter = RateLimiter(redis_client=fake_redis, limit=5, window=10)
    for _ in range(5):
        await limiter.check(mock_request)
    # The 5th request should be allowed, no exception thrown.
    assert await fake_redis.get("rate-limit:127.0.0.1") == b"5"


async def test_rate_limiter_rejects_over_limit(fake_redis, mock_request):
    """Test that the rate limiter rejects requests over the limit."""
    limiter = RateLimiter(redis_client=fake_redis, limit=2, window=10)
    await limiter.check(mock_request)
    await limiter.check(mock_request)

    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(mock_request)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


async def test_rate_limiter_resets_after_window(fake_redis, mock_request):
    """Test that the rate limit resets after the window expires."""
    limiter = RateLimiter(redis_client=fake_redis, limit=1, window=1)  # 1s window
    await limiter.check(mock_request)

    with pytest.raises(HTTPException):
        await limiter.check(mock_request)

    # Manually expire the key in fakeredis
    await fake_redis.expire("rate-limit:127.0.0.1", 0)

    # Now the next request should be allowed.
    await limiter.check(mock_request)
    assert await fake_redis.get("rate-limit:127.0.0.1") == b"1"
