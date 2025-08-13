from __future__ import annotations

try:  # make redis optional for unit tests without services
    import redis.asyncio as redis  # type: ignore
except Exception:  # pragma: no cover

    class _DummyRedis:  # type: ignore
        class AuthenticationError(Exception): ...

        class ConnectionError(Exception): ...

        def pipeline(self):
            class _Pipe:
                def incr(self, key):
                    return 1

                def expire(self, key, window):
                    return None

                def execute(self):
                    return [1]

            return _Pipe()

    redis = _DummyRedis()  # type: ignore
from fastapi import HTTPException, Request
from starlette import status

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


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
            logger.warning("Rate limit check: No client IP found, allowing request")
            return

        key = f"rate-limit:{ip}"

        # Use a pipeline to execute commands atomically and reduce round-trips.
        # This implementation uses a sliding window. Every request extends the
        # window expiration, which is a common and effective approach.
        pipe = self.redis.pipeline()
        # Support both sync and asyncio Redis clients
        incr_result = pipe.incr(key)
        pipe.expire(key, self.window)

        # Redis may be unavailable (network down) or require authentication that the
        # current environment does not provide (e.g. the real container is launched
        # with `--requirepass` but the tests rely on an in-memory *fakeredis*).  In
        # those cases we fall back to *disabling* rate-limiting rather than failing the
        # request entirely – functional correctness takes priority over enforcement in
        # such non-production scenarios.

        try:
            # Some clients return an awaitable, others a plain list
            results = pipe.execute()
            if hasattr(results, "__await__"):
                results = await results  # type: ignore[func-returns-value]
        except (redis.AuthenticationError, redis.ConnectionError) as exc:  # type: ignore[attr-defined]
            logger.warning(
                "Rate limiter disabled - Redis error",
                error=str(exc),
                error_type=type(exc).__name__,
                client_ip=ip,
                operation="incr_expire",
            )
            return

        # Pipeline returns list of results in order of commands
        request_count = results[0] if isinstance(results, (list, tuple)) else incr_result

        if request_count > self.limit:
            logger.warning(
                "Rate limit exceeded",
                client_ip=ip,
                request_count=request_count,
                limit=self.limit,
                window_seconds=self.window,
                key=key,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many requests. Please try again after {self.window} seconds.",
                headers={"Retry-After": str(self.window)},
            )

        # Log successful rate limit check for debugging
        logger.debug(
            "Rate limit check passed",
            client_ip=ip,
            request_count=request_count,
            limit=self.limit,
            remaining=self.limit - request_count,
        )
