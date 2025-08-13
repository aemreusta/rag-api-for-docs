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
    try:
        import redis.asyncio as redis  # Imported lazily to avoid needless deps in CI.

        return redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    except ModuleNotFoundError:  # pragma: no cover – allow tests to run without redis package

        class _Dummy:
            def __init__(self):
                self._store = {}

            def pipeline(self):
                class _Pipe:
                    def __init__(self, store):
                        self._store = store
                        self._key = None

                    def incr(self, key):
                        self._key = key
                        self._store[key] = int(self._store.get(key, 0)) + 1
                        return self._store[key]

                    def expire(self, key, ttl):
                        return None

                    def execute(self):
                        return [self._store.get(self._key, 1)]

                return _Pipe(self._store)

        return _Dummy()


# The singleton client used across the application.
redis_client = _create_redis_client()
