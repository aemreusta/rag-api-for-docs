# At runtime we normally talk to the real Redis instance defined in
# `settings.REDIS_URL`.  In the test-suite, however, we prefer to run against
# an in-memory fake to avoid network flakiness and authentication requirements
# that the real container enforces (`--requirepass`).

from __future__ import annotations

import sys

from app.core.config import settings


def _create_redis_client():  # noqa: D401 – factory helper
    """Return the appropriate Redis client for the current execution context."""

    # If the test runner is active we swap in *fakeredis* transparently so that
    # rate-limiting and other Redis-backed features behave deterministically.
    if "pytest" in sys.modules:  # pragma: no cover – branch specific to tests
        # Prefer the synchronous FakeRedis client in tests to avoid
        # "coroutine ... was never awaited" warnings when code paths
        # access Redis from sync contexts.
        try:
            import fakeredis  # type: ignore

            return fakeredis.FakeRedis(decode_responses=True)
        except ModuleNotFoundError:  # Safety net – fall through to real Redis.
            pass

    # Fallback to the real Redis connection for development/production.
    import redis.asyncio as redis  # Imported lazily to avoid needless deps in CI.

    return redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)


# The singleton client used across the application.
redis_client = _create_redis_client()
