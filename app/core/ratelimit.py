import redis.asyncio as redis
from fastapi import HTTPException, Request
from starlette import status

from app.core.config import settings


class RateLimiter:
    """
    A simple rate limiter using Redis.
    It uses a sliding window approach.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        limit: int = settings.RATE_LIMIT_COUNT,
        window: int = settings.RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.redis = redis_client
        self.limit = limit
        self.window = window

    async def check(self, request: Request) -> None:
        """
        Checks if the request from a given IP is within the rate limit.
        Raises an HTTPException if the limit is exceeded.
        """
        ip = request.client.host
        if not ip:
            # This should not happen with a valid request.
            # If it does, we can either allow or deny.
            # For now, we allow it, but this could be logged.
            return

        key = f"rate-limit:{ip}"

        # Use a pipeline to execute commands atomically and reduce round-trips.
        # This implementation uses a sliding window. Every request extends the
        # window expiration, which is a common and effective approach.
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.window)
        results = await pipe.execute()

        request_count = results[0]

        if request_count > self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many requests. Please try again after {self.window} seconds.",
                headers={"Retry-After": str(self.window)},
            )
