"""
Rate limit status API endpoints.

Provides information about current rate limiting status for authenticated clients.
"""

import time

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app.core.config import settings
from app.core.redis import redis_client

router = APIRouter()


class RateLimitStatus(BaseModel):
    """Rate limit status information."""

    limit: int
    remaining: int
    reset: int  # Unix timestamp when the limit resets


@router.get("/rate-limit/status", response_model=RateLimitStatus)
async def get_rate_limit_status(request: Request, response: Response):
    """
    Returns the current rate limit status for the client's IP.

    This endpoint provides information about:
    - Total requests allowed per window
    - Remaining requests in current window
    - When the current window resets (Unix timestamp)

    Includes Cache-Control headers to prevent excessive Redis load.
    """
    # Set cache-control headers to advise clients/proxies to cache for 5 seconds
    response.headers["Cache-Control"] = "public, max-age=5"

    ip = request.client.host
    key = f"rate-limit:{ip}"

    # Get current count and TTL from Redis
    async with redis_client.pipeline() as pipe:
        pipe.get(key)
        pipe.ttl(key)
        results = await pipe.execute()

    count = int(results[0]) if results[0] else 0
    ttl = int(results[1]) if results[1] != -1 else settings.RATE_LIMIT_WINDOW_SECONDS

    remaining = max(0, settings.RATE_LIMIT_COUNT - count)
    reset_time = int(time.time()) + ttl

    return RateLimitStatus(limit=settings.RATE_LIMIT_COUNT, remaining=remaining, reset=reset_time)
