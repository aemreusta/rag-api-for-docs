#!/usr/bin/env python3
"""
Embedding Provider Management Script

This script provides utilities for managing embedding providers,
monitoring quota status, and handling provider fallback.
"""

import asyncio
import sys
import time
from typing import Any

# Add the app directory to the Python path
sys.path.append("/app")

from app.core.embedding_manager import QuotaError, get_embedding_manager
from app.core.embedding_validation import validate_configuration, validate_embedding_system
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def print_status_table(status_data: dict[str, Any]):
    """Print provider status in a nice table format."""
    print("\n" + "=" * 80)
    print("🔋 EMBEDDING PROVIDER STATUS")
    print("=" * 80)

    for provider, info in status_data.get("providers", {}).items():
        status_icon = "✅" if info.get("available") else "❌"
        status = info.get("status", "unknown")

        print(f"{status_icon} {provider.upper():<12} | Status: {status:<15}", end="")

        if info.get("response_time_ms"):
            print(f" | Response: {info['response_time_ms']:>6.1f}ms", end="")

        if info.get("embedding_dimensions"):
            print(f" | Dims: {info['embedding_dimensions']:>4d}", end="")

        print()

        if info.get("error"):
            print(f"    └─ Error: {info['error']}")

        if "wait_seconds" in info and info["wait_seconds"] > 0:
            wait_mins = info["wait_seconds"] / 60
            print(f"    └─ Quota reset in: {wait_mins:.1f} minutes")

    print("\n📊 OVERALL STATUS:", status_data.get("overall_status", "unknown").upper())

    if status_data.get("fallback_chain"):
        print("🔄 FALLBACK CHAIN:", " → ".join(status_data["fallback_chain"]))

    if status_data.get("errors"):
        print("\n❌ ERRORS:")
        for error in status_data["errors"]:
            component = error.get("component", error.get("provider", "unknown"))
            error_msg = error.get("error")
            print(f"    • {component}: {error_msg}")


async def check_provider_status():
    """Check and display current provider status."""
    print("🔍 Checking embedding provider status...")

    try:
        validation_result = await validate_embedding_system()
        print_status_table(validation_result)

        return validation_result["overall_status"] not in ["all_failed", "system_error"]

    except Exception as e:
        print(f"❌ Failed to check provider status: {e}")
        return False


async def test_embedding_generation(text: str = None):
    """Test embedding generation with current providers."""
    if not text:
        text = "Merhaba, bu bir test metnidir."

    print(f"🧪 Testing embedding generation with text: '{text[:50]}...'")

    try:
        manager = get_embedding_manager()

        start_time = time.time()
        embeddings = manager.get_text_embeddings([text])
        duration = time.time() - start_time

        print("✅ Embedding generation successful!")
        print(f"    • Response time: {duration * 1000:.1f}ms")
        print(f"    • Embedding dimensions: {len(embeddings[0])}")
        print(f"    • First 5 values: {embeddings[0][:5]}")

        return True

    except QuotaError as e:
        print(f"⏳ Quota exhausted: {e}")
        return False
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False


def reset_quota_status(provider: str = None):
    """Reset quota status for a specific provider or all providers."""
    try:
        manager = get_embedding_manager()

        if provider:
            quota_key = f"embedding_quota:{provider}"
            result = manager.redis_client.delete(quota_key)
            if result:
                print(f"✅ Quota status reset for {provider}")
            else:
                print(f"ℹ️  No quota status found for {provider}")
        else:
            # Reset all provider quotas
            keys = manager.redis_client.keys("embedding_quota:*")
            if keys:
                result = manager.redis_client.delete(*keys)
                print(f"✅ Quota status reset for {result} providers")
            else:
                print("ℹ️  No quota status found for any provider")

    except Exception as e:
        print(f"❌ Failed to reset quota status: {e}")


def wait_for_quota_reset():
    """Wait for quota reset and monitor progress."""
    try:
        manager = get_embedding_manager()
        provider_status = manager.get_provider_status()

        # Find the minimum wait time
        wait_times = []
        for provider, status in provider_status.items():
            if status.get("wait_seconds", 0) > 0:
                wait_times.append((provider, status["wait_seconds"]))

        if not wait_times:
            print("✅ No providers are quota-limited!")
            return True

        # Sort by wait time
        wait_times.sort(key=lambda x: x[1])

        print("⏳ Waiting for quota reset...")
        for provider, wait_seconds in wait_times:
            wait_mins = wait_seconds / 60
            print(f"    • {provider}: {wait_mins:.1f} minutes")

        # Wait for the shortest time
        min_wait = wait_times[0][1]
        print(f"\n⏱️  Waiting {min_wait / 60:.1f} minutes for {wait_times[0][0]} quota reset...")

        # Progress updates every minute
        for elapsed in range(0, int(min_wait), 60):
            remaining = (min_wait - elapsed) / 60
            print(f"    ⏰ {remaining:.1f} minutes remaining...")
            time.sleep(60)

        print("✅ Quota should now be reset! Re-checking status...")
        return True

    except Exception as e:
        print(f"❌ Error while waiting for quota reset: {e}")
        return False


def validate_config():
    """Validate embedding configuration."""
    print("🔧 Validating embedding configuration...")

    config_result = validate_configuration()

    if config_result["valid"]:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors:")
        for error in config_result["errors"]:
            print(f"    • {error}")

    if config_result["warnings"]:
        print("⚠️  Configuration warnings:")
        for warning in config_result["warnings"]:
            print(f"    • {warning}")

    print("\n📋 Current configuration:")
    for key, value in config_result["configuration"].items():
        print(f"    • {key}: {value}")

    return config_result["valid"]


async def full_health_check():
    """Perform a comprehensive health check."""
    print("🏥 Performing full embedding system health check...\n")

    # Step 1: Configuration validation
    config_valid = validate_config()
    if not config_valid:
        print("\n❌ Health check failed: Invalid configuration")
        return False

    print("\n" + "-" * 50)

    # Step 2: Provider status check
    status_ok = await check_provider_status()
    if not status_ok:
        print("\n❌ Health check failed: No available providers")
        return False

    print("\n" + "-" * 50)

    # Step 3: Embedding generation test
    test_ok = await test_embedding_generation()
    if not test_ok:
        print("\n❌ Health check failed: Embedding generation failed")
        return False

    print("\n" + "=" * 50)
    print("✅ HEALTH CHECK PASSED - System is operational!")
    print("=" * 50)
    return True


def show_help():
    """Show help message."""
    print("""
🤖 Embedding Provider Management Tool

COMMANDS:
    status           - Check current provider status
    test [text]      - Test embedding generation (optional custom text)
    reset [provider] - Reset quota status (optional specific provider)
    wait             - Wait for quota reset
    config           - Validate configuration
    health           - Full health check
    help             - Show this help message

EXAMPLES:
    python manage_embedding_providers.py status
    python manage_embedding_providers.py test "Custom test text"
    python manage_embedding_providers.py reset google
    python manage_embedding_providers.py health
    """)


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    async_actions = {
        "status": lambda argv: check_provider_status(),
        "test": lambda argv: test_embedding_generation(" ".join(argv) if argv else None),
        "health": lambda argv: full_health_check(),
    }

    sync_actions = {
        "reset": lambda argv: reset_quota_status(argv[0] if argv else None),
        "wait": lambda argv: wait_for_quota_reset(),
        "config": lambda argv: validate_config(),
        "help": lambda argv: show_help(),
    }

    try:
        argv = sys.argv[2:]
        if command in async_actions:
            await async_actions[command](argv)
        elif command in sync_actions:
            sync_actions[command](argv)
        else:
            print(f"❌ Unknown command: {command}")
            show_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
