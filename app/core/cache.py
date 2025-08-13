"""
Cache layer implementation with Redis and in-memory TTL cache backends.

This module provides a pluggable caching architecture that supports:
- Redis-based caching for production (multi-worker coherence)
- In-memory TTL cache as fallback for development/testing
- Prometheus metrics for cache performance monitoring
- Automatic backend selection based on Redis availability
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any

try:
    import structlog  # type: ignore

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover
    structlog = None  # type: ignore
    _HAS_STRUCTLOG = False

from app.core.config import settings
from app.core.metrics import get_metrics_backend

if _HAS_STRUCTLOG:
    logger = structlog.get_logger(__name__)
else:
    import logging

    logger = logging.getLogger(__name__)
metrics = get_metrics_backend()


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set a value in the cache with TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass


class RedisCache(CacheBackend):
    """Redis-based cache backend for production use."""

    def __init__(self, redis_client, prefix: str = "cache:"):
        self.redis = redis_client
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get a value from Redis cache."""
        try:
            prefixed_key = self._make_key(key)
            value = await self.redis.get(prefixed_key)
            if value is None:
                metrics.increment("cache_misses_total", {"backend": "redis"})
                return None

            metrics.increment("cache_hits_total", {"backend": "redis"})

            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            logger.warning("Redis cache get failed", key=key, error=str(e))
            metrics.increment("cache_errors_total", {"backend": "redis", "operation": "get"})
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set a value in Redis cache with TTL."""
        try:
            prefixed_key = self._make_key(key)

            # Serialize value to JSON if it's not a string
            if isinstance(value, (dict, list, tuple, bool, int, float)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)

            await self.redis.setex(prefixed_key, ttl, serialized_value)

        except Exception as e:
            logger.warning("Redis cache set failed", key=key, ttl=ttl, error=str(e))
            metrics.increment("cache_errors_total", {"backend": "redis", "operation": "set"})

    async def delete(self, key: str) -> None:
        """Delete a key from Redis cache."""
        try:
            prefixed_key = self._make_key(key)
            deleted_count = await self.redis.delete(prefixed_key)
            if deleted_count > 0:
                metrics.increment("cache_evictions_total", {"backend": "redis"})

        except Exception as e:
            logger.warning("Redis cache delete failed", key=key, error=str(e))
            metrics.increment("cache_errors_total", {"backend": "redis", "operation": "delete"})

    async def clear(self) -> None:
        """Clear all cached values with the prefix."""
        try:
            pattern = f"{self.prefix}*"
            keys = await self.redis.keys(pattern)
            if keys:
                deleted_count = await self.redis.delete(*keys)
                metrics.increment(
                    "cache_evictions_total", {"backend": "redis"}, value=deleted_count
                )

        except Exception as e:
            logger.warning("Redis cache clear failed", error=str(e))
            metrics.increment("cache_errors_total", {"backend": "redis", "operation": "clear"})

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        try:
            prefixed_key = self._make_key(key)
            return bool(await self.redis.exists(prefixed_key))

        except Exception as e:
            logger.warning("Redis cache exists check failed", key=key, error=str(e))
            metrics.increment("cache_errors_total", {"backend": "redis", "operation": "exists"})
            return False


class InProcessTTLCache(CacheBackend):
    """In-memory TTL cache backend for development/fallback use."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._lock = asyncio.Lock()

    def _is_expired(self, expiry_time: float) -> bool:
        """Check if a cache entry has expired."""
        return time.time() > expiry_time

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry_time) in self._cache.items() if current_time > expiry_time
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            metrics.increment(
                "cache_evictions_total", {"backend": "memory"}, value=len(expired_keys)
            )

    async def _evict_lru(self) -> None:
        """Evict least recently used entries if cache is at capacity."""
        if len(self._cache) >= self.max_size:
            # Simple LRU: remove oldest entries
            # In a real implementation, you'd track access times
            keys_to_remove = list(self._cache.keys())[: len(self._cache) - self.max_size + 1]
            for key in keys_to_remove:
                del self._cache[key]

            metrics.increment(
                "cache_evictions_total", {"backend": "memory"}, value=len(keys_to_remove)
            )

    async def get(self, key: str) -> Any | None:
        """Get a value from in-memory cache."""
        async with self._lock:
            await self._cleanup_expired()

            if key not in self._cache:
                metrics.increment("cache_misses_total", {"backend": "memory"})
                return None

            value, expiry_time = self._cache[key]
            if self._is_expired(expiry_time):
                del self._cache[key]
                metrics.increment("cache_misses_total", {"backend": "memory"})
                return None

            metrics.increment("cache_hits_total", {"backend": "memory"})
            return value

    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a value in in-memory cache with TTL."""
        async with self._lock:
            if ttl is None:
                ttl = self.default_ttl

            expiry_time = time.time() + ttl

            await self._cleanup_expired()
            await self._evict_lru()

            self._cache[key] = (value, expiry_time)

    async def delete(self, key: str) -> None:
        """Delete a key from in-memory cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                metrics.increment("cache_evictions_total", {"backend": "memory"})

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            evicted_count = len(self._cache)
            self._cache.clear()
            if evicted_count > 0:
                metrics.increment(
                    "cache_evictions_total", {"backend": "memory"}, value=evicted_count
                )

    async def exists(self, key: str) -> bool:
        """Check if a key exists in in-memory cache."""
        async with self._lock:
            if key not in self._cache:
                return False

            value, expiry_time = self._cache[key]
            if self._is_expired(expiry_time):
                del self._cache[key]
                return False

            return True


# Global cache backend instance
_cache_backend: CacheBackend | None = None


async def get_cache_backend() -> CacheBackend:
    """
    Factory function to get the appropriate cache backend.

    Returns:
        CacheBackend: Redis cache if available, otherwise in-memory TTL cache
    """
    global _cache_backend

    if _cache_backend is not None:
        return _cache_backend

    # Try to use Redis first
    try:
        from app.core.redis import redis_client

        # Test Redis connection
        await redis_client.ping()
        _cache_backend = RedisCache(redis_client, prefix="chat_cache:")
        logger.info("Cache backend initialized", backend="redis")

    except Exception as e:
        # Fall back to in-memory cache
        logger.warning("Redis unavailable, falling back to in-memory cache", error=str(e))
        _cache_backend = InProcessTTLCache(
            max_size=getattr(settings, "CACHE_MAX_SIZE", 1000),
            default_ttl=getattr(settings, "CACHE_TTL_SECONDS", 3600),
        )
        logger.info("Cache backend initialized", backend="memory")

    return _cache_backend


def cache_key_for_chat(question: str, session_id: str = "", model: str = "") -> str:
    """
    Generate a cache key for chat responses.

    Args:
        question: The user's question
        session_id: Optional session identifier
        model: Optional model name

    Returns:
        str: A cache key that uniquely identifies this chat request
    """
    import hashlib

    # Create a deterministic key from the inputs
    key_parts = [question.strip().lower()]
    if session_id:
        key_parts.append(f"session:{session_id}")
    if model:
        key_parts.append(f"model:{model}")

    key_string = "|".join(key_parts)

    # Hash to create a fixed-length key (avoid Redis key length limits)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def cached(ttl: int = 3600, key_func: callable | None = None):
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds (default: 1 hour)
        key_func: Optional function to generate cache keys

    Returns:
        Decorated function that caches results
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_cache_backend()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation based on function name and args
                import hashlib

                key_parts = (
                    [func.__name__]
                    + [str(arg) for arg in args]
                    + [f"{k}={v}" for k, v in sorted(kwargs.items())]
                )
                key_string = "|".join(key_parts)
                cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:32]

            # Try to get from cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return cached_result

            # Execute function and cache result
            logger.debug("Cache miss, executing function", function=func.__name__, key=cache_key)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Cache the result
                await cache.set(cache_key, result, ttl)

                logger.debug(
                    "Function result cached",
                    function=func.__name__,
                    key=cache_key,
                    execution_time=execution_time,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    error=str(e),
                    execution_time=execution_time,
                )
                raise

        return wrapper

    return decorator


async def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidate all cache keys matching a pattern.

    Args:
        pattern: Pattern to match (e.g., "user:123:*")

    Returns:
        int: Number of keys invalidated
    """
    cache = await get_cache_backend()

    if isinstance(cache, RedisCache):
        try:
            # Redis pattern matching
            full_pattern = f"{cache.prefix}{pattern}"
            keys = await cache.redis.keys(full_pattern)
            if keys:
                deleted_count = await cache.redis.delete(*keys)
                metrics.increment(
                    "cache_evictions_total", {"backend": "redis"}, value=deleted_count
                )
                logger.info(
                    "Cache pattern invalidated", pattern=pattern, deleted_count=deleted_count
                )
                return deleted_count

        except Exception as e:
            logger.warning("Cache pattern invalidation failed", pattern=pattern, error=str(e))

    else:
        # For in-memory cache, we'd need to implement pattern matching
        # For now, just log that it's not supported
        logger.warning("Pattern invalidation not supported for in-memory cache", pattern=pattern)

    return 0


# Cache statistics for monitoring
async def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics for monitoring.

    Returns:
        dict: Cache statistics including hit rate, size, etc.
    """
    cache = await get_cache_backend()

    stats = {
        "backend": "redis" if isinstance(cache, RedisCache) else "memory",
        "timestamp": time.time(),
    }

    if isinstance(cache, RedisCache):
        try:
            # Get Redis info
            info = await cache.redis.info()
            stats.update(
                {
                    "redis_used_memory": info.get("used_memory", 0),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_keyspace_hits": info.get("keyspace_hits", 0),
                    "redis_keyspace_misses": info.get("keyspace_misses", 0),
                }
            )

            # Calculate hit rate
            hits = stats["redis_keyspace_hits"]
            misses = stats["redis_keyspace_misses"]
            if hits + misses > 0:
                stats["hit_rate"] = hits / (hits + misses)
            else:
                stats["hit_rate"] = 0.0

        except Exception as e:
            logger.warning("Failed to get Redis stats", error=str(e))

    elif isinstance(cache, InProcessTTLCache):
        stats.update(
            {
                "memory_cache_size": len(cache._cache),
                "memory_cache_max_size": cache.max_size,
                "memory_cache_default_ttl": cache.default_ttl,
            }
        )

    return stats
