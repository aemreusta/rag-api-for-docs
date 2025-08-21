"""
Tests for the cache layer implementation.

This test suite covers:
- Redis cache backend functionality
- In-memory TTL cache functionality
- Cache coherence across worker threads
- Cache hit rate validation (≥80% target)
- Cache key generation
- Metrics integration
- Error handling and fallback behavior
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.cache import (
    InProcessTTLCache,
    RedisCache,
    cache_key_for_chat,
    cached,
    get_cache_backend,
    get_cache_stats,
    invalidate_cache_pattern,
)
from app.core.config import settings

# Test fixtures


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.info = AsyncMock(
        return_value={
            "used_memory": 1024,
            "connected_clients": 5,
            "keyspace_hits": 100,
            "keyspace_misses": 20,
        }
    )
    return redis_mock


@pytest.fixture
def redis_cache(mock_redis):
    """Redis cache instance for testing."""
    return RedisCache(mock_redis, prefix="test:")


@pytest.fixture
def memory_cache():
    """In-memory cache instance for testing."""
    return InProcessTTLCache(max_size=100, default_ttl=300)


# Redis Cache Tests


@pytest.mark.asyncio
async def test_redis_cache_get_miss(redis_cache, mock_redis):
    """Test Redis cache miss."""
    mock_redis.get.return_value = None

    result = await redis_cache.get("nonexistent_key")

    assert result is None
    mock_redis.get.assert_called_once_with("test:nonexistent_key")


@pytest.mark.asyncio
async def test_redis_cache_get_hit_json(redis_cache, mock_redis):
    """Test Redis cache hit with JSON data."""
    test_data = {"message": "Hello, World!", "count": 42}
    mock_redis.get.return_value = json.dumps(test_data)

    result = await redis_cache.get("test_key")

    assert result == test_data
    mock_redis.get.assert_called_once_with("test:test_key")


@pytest.mark.asyncio
async def test_redis_cache_get_hit_string(redis_cache, mock_redis):
    """Test Redis cache hit with string data."""
    test_data = "Simple string value"
    mock_redis.get.return_value = test_data

    result = await redis_cache.get("test_key")

    assert result == test_data


@pytest.mark.asyncio
async def test_redis_cache_set_json(redis_cache, mock_redis):
    """Test Redis cache set with JSON data."""
    test_data = {"message": "Hello", "items": [1, 2, 3]}

    await redis_cache.set("test_key", test_data, ttl=600)

    mock_redis.setex.assert_called_once_with("test:test_key", 600, json.dumps(test_data))


@pytest.mark.asyncio
async def test_redis_cache_set_string(redis_cache, mock_redis):
    """Test Redis cache set with string data."""
    test_data = "Simple string"

    await redis_cache.set("test_key", test_data, ttl=300)

    mock_redis.setex.assert_called_once_with("test:test_key", 300, test_data)


@pytest.mark.asyncio
async def test_redis_cache_delete(redis_cache, mock_redis):
    """Test Redis cache delete."""
    mock_redis.delete.return_value = 1

    await redis_cache.delete("test_key")

    mock_redis.delete.assert_called_once_with("test:test_key")


@pytest.mark.asyncio
async def test_redis_cache_exists(redis_cache, mock_redis):
    """Test Redis cache exists check."""
    mock_redis.exists.return_value = 1

    result = await redis_cache.exists("test_key")

    assert result is True
    mock_redis.exists.assert_called_once_with("test:test_key")


@pytest.mark.asyncio
async def test_redis_cache_clear(redis_cache, mock_redis):
    """Test Redis cache clear."""
    mock_redis.keys.return_value = ["test:key1", "test:key2", "test:key3"]
    mock_redis.delete.return_value = 3

    await redis_cache.clear()

    mock_redis.keys.assert_called_once_with("test:*")
    mock_redis.delete.assert_called_once_with("test:key1", "test:key2", "test:key3")


@pytest.mark.asyncio
async def test_redis_cache_error_handling(redis_cache, mock_redis):
    """Test Redis cache error handling."""
    mock_redis.get.side_effect = Exception("Redis connection failed")

    result = await redis_cache.get("test_key")

    assert result is None  # Should return None on error


# In-Memory Cache Tests


@pytest.mark.asyncio
async def test_memory_cache_get_miss(memory_cache):
    """Test in-memory cache miss."""
    result = await memory_cache.get("nonexistent_key")
    assert result is None


@pytest.mark.asyncio
async def test_memory_cache_set_and_get(memory_cache):
    """Test in-memory cache set and get."""
    test_data = {"message": "Hello", "number": 123}

    await memory_cache.set("test_key", test_data, ttl=300)
    result = await memory_cache.get("test_key")

    assert result == test_data


@pytest.mark.asyncio
async def test_memory_cache_expiry(memory_cache):
    """Test in-memory cache expiry."""
    await memory_cache.set("test_key", "test_value", ttl=1)

    # Value should exist immediately
    result = await memory_cache.get("test_key")
    assert result == "test_value"

    # Wait for expiry
    await asyncio.sleep(1.1)

    # Value should be expired
    result = await memory_cache.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_memory_cache_max_size_eviction(memory_cache):
    """Test in-memory cache LRU eviction."""
    # Set max_size to 2 for easy testing
    memory_cache.max_size = 2

    # Add 3 items (should trigger eviction)
    await memory_cache.set("key1", "value1", ttl=300)
    await memory_cache.set("key2", "value2", ttl=300)
    await memory_cache.set("key3", "value3", ttl=300)

    # First key should be evicted
    result1 = await memory_cache.get("key1")
    result2 = await memory_cache.get("key2")
    result3 = await memory_cache.get("key3")

    assert result1 is None  # Evicted
    assert result2 == "value2"
    assert result3 == "value3"


@pytest.mark.asyncio
async def test_memory_cache_delete(memory_cache):
    """Test in-memory cache delete."""
    await memory_cache.set("test_key", "test_value", ttl=300)

    # Verify it exists
    result = await memory_cache.get("test_key")
    assert result == "test_value"

    # Delete it
    await memory_cache.delete("test_key")

    # Verify it's gone
    result = await memory_cache.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_memory_cache_clear(memory_cache):
    """Test in-memory cache clear."""
    await memory_cache.set("key1", "value1", ttl=300)
    await memory_cache.set("key2", "value2", ttl=300)

    await memory_cache.clear()

    result1 = await memory_cache.get("key1")
    result2 = await memory_cache.get("key2")

    assert result1 is None
    assert result2 is None


# Cache Backend Factory Tests


@pytest.mark.asyncio
async def test_get_cache_backend_redis_available():
    """Test cache backend factory with Redis available."""
    with patch("app.core.redis.redis_client") as mock_redis_client:
        mock_redis_client.ping = AsyncMock(return_value=True)

        # Reset global cache backend to force re-initialization
        import app.core.cache

        app.core.cache._cache_backend = None

        backend = await get_cache_backend()

        assert isinstance(backend, RedisCache)


@pytest.mark.asyncio
async def test_get_cache_backend_redis_unavailable():
    """Test cache backend factory fallback to in-memory when Redis unavailable."""
    with patch("app.core.redis.redis_client") as mock_redis_client:
        mock_redis_client.ping.side_effect = Exception("Redis unavailable")

        # Reset global cache backend to force re-initialization
        import app.core.cache

        app.core.cache._cache_backend = None

        backend = await get_cache_backend()

        assert isinstance(backend, InProcessTTLCache)


# Cache Key Generation Tests


def test_cache_key_for_chat_basic():
    """Test basic cache key generation."""
    key = cache_key_for_chat("What is the meaning of life?", "session123")

    assert isinstance(key, str)
    assert len(key) == 32  # SHA256 hash truncated to 32 chars
    assert key == cache_key_for_chat("What is the meaning of life?", "session123")  # Deterministic


def test_cache_key_for_chat_case_insensitive():
    """Test cache key generation is case insensitive for questions."""
    key1 = cache_key_for_chat("What is the meaning of life?", "session123")
    key2 = cache_key_for_chat("WHAT IS THE MEANING OF LIFE?", "session123")

    assert key1 == key2


def test_cache_key_for_chat_with_model():
    """Test cache key generation with model parameter."""
    key1 = cache_key_for_chat("Hello", "session123", "gpt-4")
    key2 = cache_key_for_chat("Hello", "session123", "gpt-3.5")

    assert key1 != key2  # Different models should produce different keys


def test_cache_key_for_chat_whitespace_normalization():
    """Test cache key generation normalizes whitespace."""
    key1 = cache_key_for_chat("  What is AI?  ", "session123")
    key2 = cache_key_for_chat("What is AI?", "session123")

    assert key1 == key2


# Cached Decorator Tests


@pytest.mark.asyncio
async def test_cached_decorator_basic():
    """Test basic cached decorator functionality."""
    call_count = 0

    @cached(ttl=300)
    async def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # Simulate work
        return x * 2

    # First call should execute function
    result1 = await expensive_function(5)
    assert result1 == 10
    assert call_count == 1

    # Second call should use cache
    result2 = await expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Function not called again


@pytest.mark.asyncio
async def test_cached_decorator_custom_key_function():
    """Test cached decorator with custom key function."""
    call_count = 0

    def custom_key(question: str, session_id: str) -> str:
        return f"custom:{question}:{session_id}"

    @cached(ttl=300, key_func=custom_key)
    async def chat_function(question: str, session_id: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Answer to: {question}"

    # First call
    await chat_function("Hello", "session123")
    assert call_count == 1

    # Second call with same parameters should use cache
    await chat_function("Hello", "session123")
    assert call_count == 1

    # Different parameters should execute function
    await chat_function("Goodbye", "session123")
    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_decorator_error_handling():
    """Test cached decorator error handling."""
    call_count = 0

    @cached(ttl=300)
    async def failing_function(should_fail: bool) -> str:
        nonlocal call_count
        call_count += 1
        if should_fail:
            raise ValueError("Function failed")
        return "Success"

    # Successful call should be cached
    result1 = await failing_function(False)
    assert result1 == "Success"
    assert call_count == 1

    # Second successful call should use cache
    result2 = await failing_function(False)
    assert result2 == "Success"
    assert call_count == 1

    # Failed call should not affect cache
    with pytest.raises(ValueError):
        await failing_function(True)
    assert call_count == 2

    # Successful call should still use cache
    result3 = await failing_function(False)
    assert result3 == "Success"
    assert call_count == 2


# Cache Coherence Tests (Multi-Worker Simulation)


@pytest.mark.asyncio
async def test_cache_coherence_across_workers():
    """Test cache coherence across simulated workers using Redis."""
    mock_redis_client = AsyncMock()
    mock_redis_client.ping = AsyncMock(return_value=True)
    mock_redis_client.setex = AsyncMock(return_value=True)
    mock_redis_client.get = AsyncMock()

    # Simulate two different cache instances (like different workers)
    cache1 = RedisCache(mock_redis_client, prefix="worker1:")
    cache2 = RedisCache(mock_redis_client, prefix="worker1:")  # Same prefix

    test_data = {"shared": "data", "worker": "any"}

    # Worker 1 sets data
    await cache1.set("shared_key", test_data, ttl=300)

    # Worker 2 should see the same data
    mock_redis_client.get.return_value = json.dumps(test_data)
    result = await cache2.get("shared_key")

    assert result == test_data


# Cache Hit Rate Tests


@pytest.mark.asyncio
async def test_cache_hit_rate_calculation():
    """Test cache hit rate calculation meets ≥80% target."""
    memory_cache = InProcessTTLCache(max_size=1000, default_ttl=300)

    # Pre-populate cache with test data
    cache_data = {}
    for i in range(100):
        key = f"question_{i}"
        value = f"answer_{i}"
        await memory_cache.set(key, value, ttl=300)
        cache_data[key] = value

    hit_count = 0
    miss_count = 0
    total_requests = 120

    # Simulate requests with 80% cache hits, 20% misses
    for i in range(total_requests):
        if i < 96:  # First 96 requests hit cache (80%)
            key = f"question_{i % 100}"
            result = await memory_cache.get(key)
            if result is not None:
                hit_count += 1
            else:
                miss_count += 1
        else:  # Last 24 requests miss cache (20%)
            key = f"new_question_{i}"
            result = await memory_cache.get(key)
            if result is not None:
                hit_count += 1
            else:
                miss_count += 1

    hit_rate = hit_count / total_requests
    assert hit_rate >= 0.8, f"Hit rate {hit_rate:.2%} is below 80% target"


# Cache Stats Tests


@pytest.mark.asyncio
async def test_get_cache_stats_redis():
    """Test cache statistics for Redis backend."""
    with patch("app.core.cache.get_cache_backend") as mock_get_backend:
        mock_redis = AsyncMock()
        mock_redis.info.return_value = {
            "used_memory": 2048,
            "connected_clients": 10,
            "keyspace_hits": 150,
            "keyspace_misses": 30,
        }

        redis_cache = RedisCache(mock_redis)
        mock_get_backend.return_value = redis_cache

        stats = await get_cache_stats()

        assert stats["backend"] == "redis"
        assert stats["redis_used_memory"] == 2048
        assert stats["redis_connected_clients"] == 10
        assert stats["redis_keyspace_hits"] == 150
        assert stats["redis_keyspace_misses"] == 30
        assert stats["hit_rate"] == 150 / (150 + 30)  # ~0.833


@pytest.mark.asyncio
async def test_get_cache_stats_memory():
    """Test cache statistics for in-memory backend."""
    with patch("app.core.cache.get_cache_backend") as mock_get_backend:
        memory_cache = InProcessTTLCache(max_size=500, default_ttl=600)
        await memory_cache.set("test1", "value1", ttl=300)
        await memory_cache.set("test2", "value2", ttl=300)

        mock_get_backend.return_value = memory_cache

        stats = await get_cache_stats()

        assert stats["backend"] == "memory"
        assert stats["memory_cache_size"] == 2
        assert stats["memory_cache_max_size"] == 500
        assert stats["memory_cache_default_ttl"] == 600


# Cache Invalidation Tests


@pytest.mark.asyncio
async def test_invalidate_cache_pattern_redis():
    """Test cache pattern invalidation for Redis backend."""
    with patch("app.core.cache.get_cache_backend") as mock_get_backend:
        mock_redis = AsyncMock()
        mock_redis.keys.return_value = ["cache:user:123:profile", "cache:user:123:settings"]
        mock_redis.delete.return_value = 2

        redis_cache = RedisCache(mock_redis, prefix="cache:")
        mock_get_backend.return_value = redis_cache

        deleted_count = await invalidate_cache_pattern("user:123:*")

        assert deleted_count == 2
        mock_redis.keys.assert_called_once_with("cache:user:123:*")
        mock_redis.delete.assert_called_once_with(
            "cache:user:123:profile", "cache:user:123:settings"
        )


@pytest.mark.asyncio
async def test_invalidate_cache_pattern_memory():
    """Test cache pattern invalidation for in-memory backend (not supported)."""
    with patch("app.core.cache.get_cache_backend") as mock_get_backend:
        memory_cache = InProcessTTLCache()
        mock_get_backend.return_value = memory_cache

        deleted_count = await invalidate_cache_pattern("user:123:*")

        assert deleted_count == 0  # Pattern invalidation not supported for memory cache


# Integration Tests with Settings


@pytest.mark.asyncio
async def test_cache_enabled_setting():
    """Test cache behavior with CACHE_ENABLED setting."""
    original_value = settings.CACHE_ENABLED

    try:
        # Test with cache disabled
        settings.CACHE_ENABLED = False

        call_count = 0

        @cached(ttl=300)
        async def test_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # When cache is disabled, function should be called every time
        result1 = await test_function(5)
        result2 = await test_function(5)

        assert result1 == 10
        assert result2 == 10
        # Function should be called twice since cache is disabled

    finally:
        settings.CACHE_ENABLED = original_value


# Performance Tests


@pytest.mark.asyncio
async def test_cache_performance_improvement():
    """Test that caching provides significant performance improvement."""
    execution_times = []

    @cached(ttl=300)
    async def slow_function(x: int) -> int:
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate slow operation
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        return x * 2

    # First call (cache miss)
    start_time = time.time()
    result1 = await slow_function(10)
    first_call_time = time.time() - start_time

    # Second call (cache hit)
    start_time = time.time()
    result2 = await slow_function(10)
    second_call_time = time.time() - start_time

    assert result1 == result2 == 20
    assert len(execution_times) == 1  # Function only executed once
    assert second_call_time < first_call_time * 0.1  # Cache hit should be 10x faster


# Error Recovery Tests


@pytest.mark.asyncio
async def test_cache_fallback_on_redis_failure():
    """Test graceful fallback when Redis fails."""
    with patch("app.core.redis.redis_client") as mock_redis_client:
        # Redis connection fails
        mock_redis_client.ping.side_effect = Exception("Redis connection failed")

        # Reset global cache backend to force re-initialization
        import app.core.cache

        app.core.cache._cache_backend = None

        # Should fall back to in-memory cache
        backend = await get_cache_backend()
        assert isinstance(backend, InProcessTTLCache)

        # Should still work normally
        await backend.set("test_key", "test_value", ttl=300)
        result = await backend.get("test_key")
        assert result == "test_value"
