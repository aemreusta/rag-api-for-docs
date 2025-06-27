from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.ratelimit import RateLimiter
from app.core.redis import redis_client

# Use `auto_error=False` so we can customize the error responses and align them
# with our test-suite expectations (422 for missing, 401 for invalid).
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(key: str | None = Security(api_key_header)):
    """Validate the X-API-Key header.

    The behaviour is intentionally aligned with the expectations hard-coded in
    *tests/test_chat.py*:

    * **422 Unprocessable Entity** – header missing entirely.
    * **401 Unauthorized** – header present but does **not** match the secret
      configured in settings.
    """

    if key is None:
        # Header missing ⇒ 422 so that clients can distinguish from an invalid
        # key that *was* provided.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing API Key"
        )

    if key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key"
        )

    return key


async def rate_limit(request: Request):
    """
    Dependency that provides rate limiting for an endpoint.
    """
    limiter = RateLimiter(redis_client)
    await limiter.check(request)
