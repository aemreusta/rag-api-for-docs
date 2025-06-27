from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.ratelimit import RateLimiter
from app.core.redis import redis_client

api_key_header = APIKeyHeader(name="X-API-Key")


def get_api_key(key: str = Security(api_key_header)):
    if key == settings.API_KEY:
        return key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


async def rate_limit(request: Request):
    """
    Dependency that provides rate limiting for an endpoint.
    """
    limiter = RateLimiter(redis_client)
    await limiter.check(request)
