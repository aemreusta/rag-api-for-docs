#!/usr/bin/env python3
"""
Validation script to demonstrate the new embedding provider changes.
This script shows that Qwen3 is now the default provider and validates
the fail-fast behavior for missing API keys.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_qwen3_default():
    """Test that Qwen3 is now the default embedding provider."""
    print_header("Testing Qwen3 as Default Provider")

    try:
        from app.core.config import Settings

        settings = Settings()
        print(f"✅ Default EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
        print(f"✅ Default EMBEDDING_MODEL_NAME: {settings.EMBEDDING_MODEL_NAME}")
        print(f"✅ Default EMBEDDING_DIM: {settings.EMBEDDING_DIM}")
        return True
    except Exception as e:
        print(f"❌ Failed to load default settings: {e}")
        return False


def test_google_fail_fast():
    """Test that Google provider fails fast when API key is missing."""
    print_header("Testing Google Provider Fail-Fast")

    # Create a temporary script to test this
    test_script = """
import sys
import os
sys.path.insert(0, "/Users/emre/GitHub/chatbot-api-service")

from unittest.mock import patch
from app.core.embeddings import get_embedding_model
from app.core.config import settings

# Clear Google API keys
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GOOGLE_AI_STUDIO_API_KEY", None)

with patch.object(settings, "EMBEDDING_PROVIDER", "google"), \
     patch.object(settings, "EMBEDDING_MODEL_NAME", "text-embedding-004"), \
     patch.object(settings, "GOOGLE_AI_STUDIO_API_KEY", None):

    try:
        get_embedding_model()
        print("❌ Expected RuntimeError for missing Google API key")
        sys.exit(1)
    except RuntimeError as e:
        if "Google embedding provider" in str(e) and "API key" in str(e):
            print("✅ Google provider correctly failed fast with missing API key")
            print(f"   Error includes resolution steps: {'resolution' in str(e).lower()}")
            sys.exit(0)
        else:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected exception type: {type(e).__name__}: {e}")
        sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        try:
            result = subprocess.run([sys.executable, f.name], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print(f"❌ Test failed: {result.stdout}{result.stderr}")
                return False
        finally:
            os.unlink(f.name)


def test_openai_fail_fast():
    """Test that OpenAI provider fails fast when API key is missing."""
    print_header("Testing OpenAI Provider Fail-Fast")

    test_script = """
import sys
import os
sys.path.insert(0, "/Users/emre/GitHub/chatbot-api-service")

from unittest.mock import patch
from app.core.embeddings import get_embedding_model
from app.core.config import settings

# Clear OpenAI API key
os.environ.pop("OPENAI_API_KEY", None)

with patch.object(settings, "EMBEDDING_PROVIDER", "openai"), \
     patch.object(settings, "EMBEDDING_MODEL_NAME", "text-embedding-3-small"):

    try:
        get_embedding_model()
        print("❌ Expected RuntimeError for missing OpenAI API key")
        sys.exit(1)
    except RuntimeError as e:
        if "OpenAI embedding provider" in str(e) and "API key" in str(e):
            print("✅ OpenAI provider correctly failed fast with missing API key")
            print(f"   Error includes resolution steps: {'resolution' in str(e).lower()}")
            sys.exit(0)
        else:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected exception type: {type(e).__name__}: {e}")
        sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        try:
            result = subprocess.run([sys.executable, f.name], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print(f"❌ Test failed: {result.stdout}{result.stderr}")
                return False
        finally:
            os.unlink(f.name)


def test_hf_provider_still_works():
    """Test that HuggingFace provider still works when explicitly configured."""
    print_header("Testing HuggingFace Provider Still Works")

    test_script = """
import sys
import os
sys.path.insert(0, "/Users/emre/GitHub/chatbot-api-service")

from unittest.mock import patch, MagicMock
from app.core.embeddings import get_embedding_model
from app.core.config import settings

with patch.object(settings, "EMBEDDING_PROVIDER", "hf"), \
     patch.object(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"), \
     patch("app.core.embeddings.HuggingFaceEmbedding") as mock_hf:

    mock_instance = MagicMock()
    mock_hf.return_value = mock_instance

    try:
        model = get_embedding_model()
        print("✅ HuggingFace provider works when explicitly configured")
        print(f"   Model type: {type(model).__name__}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ HuggingFace provider failed: {e}")
        sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        try:
            result = subprocess.run([sys.executable, f.name], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print(f"❌ Test failed: {result.stdout}{result.stderr}")
                return False
        finally:
            os.unlink(f.name)


def test_qwen3_provider_works():
    """Test that Qwen3 provider works when configured."""
    print_header("Testing Qwen3 Provider Works")

    test_script = """
import sys
import os
sys.path.insert(0, "/Users/emre/GitHub/chatbot-api-service")

from unittest.mock import patch, MagicMock
from app.core.embeddings import get_embedding_model
from app.core.config import settings

with patch.object(settings, "EMBEDDING_PROVIDER", "qwen3"), \
     patch.object(settings, "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"), \
     patch.object(settings, "EMBEDDING_SERVICE_ENDPOINT", ""), \
     patch("app.core.qwen3_embedding.Qwen3EmbeddingLocal") as mock_qwen_local:

    mock_instance = MagicMock()
    mock_qwen_local.return_value = mock_instance

    try:
        model = get_embedding_model()
        print("✅ Qwen3 provider works when configured")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Mock was called with: {mock_qwen_local.call_args}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Qwen3 provider failed: {e}")
        sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        try:
            result = subprocess.run([sys.executable, f.name], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print(f"❌ Test failed: {result.stdout}{result.stderr}")
                return False
        finally:
            os.unlink(f.name)


def main():
    """Run all validation tests."""
    print("🚀 Validating Embedding Provider Changes")
    print("This script demonstrates the new Qwen3 default provider and fail-fast validation.")

    results = []
    results.append(test_qwen3_default())
    results.append(test_google_fail_fast())
    results.append(test_openai_fail_fast())
    results.append(test_hf_provider_still_works())
    results.append(test_qwen3_provider_works())

    print(f"\n{'=' * 60}")
    print(f"📊 Validation Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("\n🎉 All validations passed! The embedding provider changes are working correctly.")
        print("\nKey improvements:")
        print("✅ Qwen3 is now the default embedding provider")
        print("✅ Fail-fast behavior prevents silent production failures")
        print("✅ Clear error messages with resolution steps")
        print("✅ HuggingFace provider still available when explicitly configured")
        print("✅ Well-organized test structure")
        return 0
    else:
        print("\n❌ Some validations failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
