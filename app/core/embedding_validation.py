"""
Embedding system validation and health checks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.core.embedding_manager import get_embedding_manager
from app.core.logging_config import get_logger

logger = get_logger(__name__)


async def validate_embedding_system() -> dict[str, Any]:
    """
    Comprehensive validation of the embedding system with all providers.

    Returns:
        Dict with validation results for each provider and overall system status.
    """
    validation_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "unknown",
        "providers": {},
        "fallback_chain": [],
        "quota_status": {},
        "errors": [],
    }

    try:
        # Get embedding manager
        manager = get_embedding_manager()

        # Get provider status
        provider_status = manager.get_provider_status()
        validation_results["quota_status"] = provider_status

        # Test each provider individually
        test_texts = [
            "Merhaba dünya",  # Turkish
            "Hello world",  # English
            "This is a test embedding text to verify functionality.",
        ]

        for provider_name in ["google", "qwen", "huggingface"]:
            if provider_name not in manager.providers:
                validation_results["providers"][provider_name] = {
                    "status": "not_configured",
                    "available": False,
                    "error": "Provider not configured",
                }
                continue

            provider_result = {
                "status": "unknown",
                "available": False,
                "response_time_ms": None,
                "embedding_dimensions": None,
                "test_successful": False,
                "error": None,
            }

            try:
                start_time = asyncio.get_event_loop().time()

                # Test single embedding
                provider = manager.providers[provider_name]
                embedding = provider.get_text_embedding(test_texts[0])

                response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                provider_result.update(
                    {
                        "status": "active",
                        "available": True,
                        "response_time_ms": round(response_time, 2),
                        "embedding_dimensions": len(embedding) if embedding else 0,
                        "test_successful": True,
                    }
                )

                logger.info(
                    f"Provider {provider_name} validation successful",
                    provider=provider_name,
                    dimensions=len(embedding),
                    response_time_ms=round(response_time, 2),
                )

            except Exception as e:
                error_msg = str(e)
                provider_result.update(
                    {
                        "status": "error",
                        "available": False,
                        "error": error_msg,
                        "test_successful": False,
                    }
                )

                validation_results["errors"].append(
                    {"provider": provider_name, "error": error_msg, "error_type": type(e).__name__}
                )

                logger.warning(
                    f"Provider {provider_name} validation failed",
                    provider=provider_name,
                    error=error_msg,
                    error_type=type(e).__name__,
                )

            validation_results["providers"][provider_name] = provider_result

        # Build fallback chain
        available_providers = [
            name
            for name, result in validation_results["providers"].items()
            if result.get("available", False)
        ]
        validation_results["fallback_chain"] = available_providers

        # Determine overall status
        if not available_providers:
            validation_results["overall_status"] = "all_failed"
        elif "google" in available_providers:
            validation_results["overall_status"] = "optimal"  # Primary provider working
        elif len(available_providers) > 0:
            validation_results["overall_status"] = "degraded"  # Only fallbacks working
        else:
            validation_results["overall_status"] = "failed"

        # Test full integration with manager
        try:
            manager_embeddings = manager.get_text_embeddings(test_texts)
            validation_results["integration_test"] = {
                "successful": True,
                "embeddings_count": len(manager_embeddings),
                "embedding_dimensions": len(manager_embeddings[0]) if manager_embeddings else 0,
            }

            logger.info(
                "Embedding manager integration test successful",
                embeddings_count=len(manager_embeddings),
                dimensions=len(manager_embeddings[0]) if manager_embeddings else 0,
            )

        except Exception as e:
            validation_results["integration_test"] = {
                "successful": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            validation_results["errors"].append(
                {"component": "integration_test", "error": str(e), "error_type": type(e).__name__}
            )

            logger.error(
                "Embedding manager integration test failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    except Exception as e:
        validation_results["overall_status"] = "system_error"
        validation_results["errors"].append(
            {"component": "system", "error": str(e), "error_type": type(e).__name__}
        )

        logger.error(
            "Embedding system validation failed", error=str(e), error_type=type(e).__name__
        )

    return validation_results


def validate_configuration() -> dict[str, Any]:
    """
    Validate embedding configuration without making API calls.
    """
    config_validation = {
        "timestamp": datetime.utcnow().isoformat(),
        "valid": True,
        "warnings": [],
        "errors": [],
        "configuration": {},
    }

    # Check primary embedding provider
    config_validation["configuration"]["embedding_provider"] = settings.EMBEDDING_PROVIDER
    config_validation["configuration"]["embedding_model"] = settings.EMBEDDING_MODEL_NAME
    config_validation["configuration"]["embedding_dimensions"] = settings.EMBEDDING_DIM

    # Validate Google configuration
    if settings.EMBEDDING_PROVIDER == "google":
        if not settings.GOOGLE_AI_STUDIO_API_KEY:
            config_validation["errors"].append(
                "Google embedding provider selected but GOOGLE_AI_STUDIO_API_KEY not configured"
            )
            config_validation["valid"] = False
        else:
            config_validation["configuration"]["google_api_key"] = "configured"

    # Check Qwen configuration
    if settings.QWEN_ENDPOINT:
        config_validation["configuration"]["qwen_endpoint"] = settings.QWEN_ENDPOINT
        config_validation["configuration"]["qwen_dimensions"] = settings.QWEN_DIMENSIONS

        if not settings.QWEN_ENDPOINT.startswith(("http://", "https://")):
            config_validation["warnings"].append(
                f"Qwen endpoint should include protocol: {settings.QWEN_ENDPOINT}"
            )
    else:
        config_validation["warnings"].append("Qwen fallback provider not configured")

    # Check Redis configuration
    if not settings.REDIS_HOST:
        config_validation["errors"].append(
            "Redis host not configured - required for quota tracking"
        )
        config_validation["valid"] = False

    config_validation["configuration"]["redis"] = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "password_configured": bool(settings.REDIS_PASSWORD),
    }

    # Dimension compatibility check
    expected_dims = {
        "google": [768, 1536, 3072],  # Gemini MRL dimensions
        "qwen": [1024],  # Qwen3-Embedding-0.6B default
        "huggingface": [384, 512, 768],  # Common HF dimensions
    }

    if settings.EMBEDDING_PROVIDER in expected_dims:
        expected_for_provider = expected_dims[settings.EMBEDDING_PROVIDER]
        if settings.EMBEDDING_DIM not in expected_for_provider:
            warning_message = (
                f"Unusual dimension {settings.EMBEDDING_DIM} for provider "
                f"{settings.EMBEDDING_PROVIDER}. Expected: {expected_for_provider}"
            )
            config_validation["warnings"].append(warning_message)

    return config_validation


if __name__ == "__main__":
    """CLI validation tool."""
    import json

    # Configuration validation
    print("🔧 Validating embedding configuration...")
    config_result = validate_configuration()
    print(json.dumps(config_result, indent=2))

    if not config_result["valid"]:
        print("❌ Configuration validation failed!")
        exit(1)

    # System validation
    print("\n🧪 Testing embedding system...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        system_result = loop.run_until_complete(validate_embedding_system())
        print(json.dumps(system_result, indent=2))

        if system_result["overall_status"] in ["all_failed", "system_error", "failed"]:
            print("❌ System validation failed!")
            exit(1)
        elif system_result["overall_status"] == "degraded":
            print(
                "⚠️  System validation passed with degraded performance (using fallback providers)"
            )
            exit(0)
        else:
            print("✅ System validation successful!")
            exit(0)

    finally:
        loop.close()
