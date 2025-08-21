"""Tests for CORS configuration and middleware."""

import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.cors import get_cors_config, log_cors_rejection


class TestCORSConfiguration:
    """Test CORS configuration parsing and middleware creation."""

    def test_cors_origins_debug_development_default(self):
        """Test that DEBUG + development defaults to wildcard when no origins set."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "true",
                "ENVIRONMENT": "development",
                "CORS_ALLOW_ORIGINS": "",
            },
        ):
            settings = Settings()
            assert settings.cors_origins == ["*"]

    def test_cors_origins_debug_production_no_default(self):
        """Test that DEBUG + production does not default to wildcard."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "true",
                "ENVIRONMENT": "production",
                "CORS_ALLOW_ORIGINS": "",
            },
        ):
            settings = Settings()
            assert settings.cors_origins == []

    def test_cors_origins_parse_comma_separated(self):
        """Test parsing comma-separated origins."""
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_ORIGINS": "https://example.com,http://localhost:3000,https://app.example.com",
            },
        ):
            settings = Settings()
            expected = ["https://example.com", "http://localhost:3000", "https://app.example.com"]
            assert settings.cors_origins == expected

    def test_cors_origins_strips_whitespace(self):
        """Test that whitespace is stripped from origins."""
        origins = " https://example.com , http://localhost:3000 , https://app.example.com "
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_ORIGINS": origins,
            },
        ):
            settings = Settings()
            expected = ["https://example.com", "http://localhost:3000", "https://app.example.com"]
            assert settings.cors_origins == expected

    def test_cors_origins_filters_empty_strings(self):
        """Test that empty strings are filtered out."""
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_ORIGINS": "https://example.com,,http://localhost:3000,",
            },
        ):
            settings = Settings()
            expected = ["https://example.com", "http://localhost:3000"]
            assert settings.cors_origins == expected

    def test_cors_credentials_safe_with_wildcard(self):
        """Test that credentials are disabled with wildcard origins."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "true",
                "ENVIRONMENT": "development",
                "CORS_ALLOW_ORIGINS": "",
                "CORS_ALLOW_CREDENTIALS": "true",
            },
        ):
            settings = Settings()
            assert settings.cors_origins == ["*"]
            assert settings.cors_allow_credentials_safe is False

    def test_cors_credentials_safe_with_specific_origins(self):
        """Test that credentials can be enabled with specific origins."""
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_ORIGINS": "https://example.com",
                "CORS_ALLOW_CREDENTIALS": "true",
            },
        ):
            settings = Settings()
            assert settings.cors_origins == ["https://example.com"]
            assert settings.cors_allow_credentials_safe is True

    def test_cors_config_factory(self):
        """Test that get_cors_config returns properly configured dictionary."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "false",
                "ENVIRONMENT": "production",
                "CORS_ALLOW_ORIGINS": "https://example.com,http://localhost:3000",
                "CORS_ALLOW_METHODS": "GET,POST,PUT",
                "CORS_ALLOW_HEADERS": "Authorization,Content-Type",
                "CORS_MAX_AGE": "300",
            },
        ):
            # Create a fresh settings instance that uses the patched environment
            from app.core.config import Settings

            with patch("app.core.cors.settings", Settings()):
                config = get_cors_config()

                assert config["allow_origins"] == ["https://example.com", "http://localhost:3000"]
                assert config["allow_methods"] == ["GET", "POST", "PUT"]
                assert config["allow_headers"] == ["Authorization", "Content-Type"]
                assert config["max_age"] == 300

    def test_environment_override_staging_to_production(self):
        """Test environment override from staging to production."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "false",
                "ENVIRONMENT": "staging",
                "CORS_ALLOW_ORIGINS": "https://staging.example.com",
            },
        ):
            settings = Settings()
            assert settings.cors_origins == ["https://staging.example.com"]
            assert settings.ENVIRONMENT == "staging"

        # Override to production
        with patch.dict(
            os.environ,
            {
                "DEBUG": "false",
                "ENVIRONMENT": "production",
                "CORS_ALLOW_ORIGINS": "https://example.com",
            },
        ):
            settings = Settings()
            assert settings.cors_origins == ["https://example.com"]
            assert settings.ENVIRONMENT == "production"

    def test_environment_override_complex_scenarios(self):
        """Test complex environment override scenarios."""
        # Test multiple environment switches
        environments = [
            ("development", True, "", ["*"]),
            ("staging", False, "https://staging.example.com", ["https://staging.example.com"]),
            (
                "production",
                False,
                "https://example.com,https://app.example.com",
                ["https://example.com", "https://app.example.com"],
            ),
            ("testing", True, "", []),  # Testing environment without default wildcard
        ]

        for env, debug, origins, expected in environments:
            with patch.dict(
                os.environ,
                {
                    "DEBUG": str(debug).lower(),
                    "ENVIRONMENT": env,
                    "CORS_ALLOW_ORIGINS": origins,
                },
            ):
                settings = Settings()
                assert settings.cors_origins == expected, f"Failed for environment: {env}"

    def test_cors_headers_and_methods_customization(self):
        """Test that CORS methods and headers can be customized via environment."""
        custom_methods = "GET,POST,PATCH,DELETE,OPTIONS"
        custom_headers = "Authorization,Content-Type,X-Custom-Header,X-Request-ID"

        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_METHODS": custom_methods,
                "CORS_ALLOW_HEADERS": custom_headers,
                "CORS_MAX_AGE": "1200",
            },
        ):
            settings = Settings()
            assert settings.CORS_ALLOW_METHODS == custom_methods
            assert settings.CORS_ALLOW_HEADERS == custom_headers
            assert settings.CORS_MAX_AGE == 1200

    def test_cors_boolean_environment_override(self):
        """Test CORS boolean settings environment override."""
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_CREDENTIALS": "true",
                "CORS_ALLOW_ORIGINS": "https://example.com",
            },
        ):
            settings = Settings()
            assert settings.CORS_ALLOW_CREDENTIALS is True
            assert settings.cors_allow_credentials_safe is True

        # Test false override
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_CREDENTIALS": "false",
                "CORS_ALLOW_ORIGINS": "https://example.com",
            },
        ):
            settings = Settings()
            assert settings.CORS_ALLOW_CREDENTIALS is False
            assert settings.cors_allow_credentials_safe is False


class TestCORSLogging:
    """Test CORS logging functionality."""

    @patch("app.core.cors.logger")
    def test_cors_config_logging_in_debug(self, mock_logger):
        """Test that CORS configuration is logged in DEBUG mode."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "true",
                "ENVIRONMENT": "development",
                "CORS_ALLOW_ORIGINS": "https://example.com",
                "CORS_ALLOW_METHODS": "GET,POST",
                "CORS_ALLOW_HEADERS": "Authorization,Content-Type",
                "CORS_MAX_AGE": "600",
            },
        ):
            from app.core.config import Settings

            with patch("app.core.cors.settings", Settings()):
                get_cors_config()

                # Verify logging was called
                mock_logger.info.assert_called_once()
                call_args = mock_logger.info.call_args
                assert "Configuring CORS middleware" in call_args[0][0]

    @patch("app.core.cors.logger")
    def test_wildcard_warning_in_production(self, mock_logger):
        """Test that wildcard warning is logged in non-development environments."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "true",
                "ENVIRONMENT": "production",
                "CORS_ALLOW_ORIGINS": "*",  # Explicit wildcard in production
            },
        ):
            from app.core.config import Settings

            with patch("app.core.cors.settings", Settings()):
                get_cors_config()

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert (
                    "CORS wildcard (*) detected in non-development environment" in call_args[0][0]
                )

    @patch("app.core.cors.logger")
    def test_cors_rejection_logging(self, mock_logger):
        """Test CORS rejection logging functionality."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "true",
                "ENVIRONMENT": "development",
            },
        ):
            from app.core.config import Settings

            with patch("app.core.cors.settings", Settings()):
                log_cors_rejection("https://malicious-site.com", "GET", "/api/v1/chat")

                # Verify rejection was logged
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert "CORS request rejected" in call_args[0][0]

    @patch("app.core.cors.logger")
    def test_no_logging_when_debug_false(self, mock_logger):
        """Test that CORS logging is disabled when DEBUG is false."""
        with patch.dict(
            os.environ,
            {
                "DEBUG": "false",
                "ENVIRONMENT": "production",
                "CORS_ALLOW_ORIGINS": "https://example.com",
            },
        ):
            from app.core.config import Settings

            with patch("app.core.cors.settings", Settings()):
                get_cors_config()
                log_cors_rejection("https://malicious-site.com", "GET", "/api/v1/chat")

                # Verify no logging occurred
                mock_logger.info.assert_not_called()
                mock_logger.warning.assert_not_called()


class TestCORSIntegration:
    """Test CORS integration with FastAPI."""

    def create_app_with_cors(self, env_vars=None):
        """Create a test FastAPI app with CORS middleware for specific env vars."""
        from fastapi.middleware.cors import CORSMiddleware

        from app.core.config import Settings

        app = FastAPI()

        # Use provided env vars or default
        if env_vars:
            with patch.dict(os.environ, env_vars):
                with patch("app.core.cors.settings", Settings()):
                    cors_config = get_cors_config()
        else:
            cors_config = get_cors_config()

        app.add_middleware(CORSMiddleware, **cors_config)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        return app

    def test_debug_development_allows_wildcard(self):
        """Test that DEBUG + development allows any origin."""
        env_vars = {
            "DEBUG": "true",
            "ENVIRONMENT": "development",
            "CORS_ALLOW_ORIGINS": "",
        }
        app = self.create_app_with_cors(env_vars)
        client = TestClient(app)

        # Test with any origin
        response = client.get("/test", headers={"Origin": "https://random-domain.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"

    def test_production_specific_origins_only(self):
        """Test that production only allows specified origins."""
        env_vars = {
            "DEBUG": "false",
            "ENVIRONMENT": "production",
            "CORS_ALLOW_ORIGINS": "https://example.com,http://localhost:3000",
        }
        app = self.create_app_with_cors(env_vars)
        client = TestClient(app)

        # Test allowed origin
        response = client.get("/test", headers={"Origin": "https://example.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://example.com"

        # Test disallowed origin
        response = client.get("/test", headers={"Origin": "https://malicious-site.com"})
        assert response.status_code == 200
        # CORS rejection doesn't return 403, it just omits the header
        assert "access-control-allow-origin" not in response.headers

    def test_preflight_options_request(self):
        """Test that OPTIONS pre-flight requests return correct headers."""
        env_vars = {
            "DEBUG": "false",
            "ENVIRONMENT": "production",
            "CORS_ALLOW_ORIGINS": "https://example.com",
            "CORS_ALLOW_METHODS": "GET,POST,PUT,DELETE",
            "CORS_ALLOW_HEADERS": "Authorization,Content-Type,X-API-Key",
            "CORS_MAX_AGE": "600",
        }
        app = self.create_app_with_cors(env_vars)
        client = TestClient(app)

        # Send OPTIONS pre-flight request
        response = client.options(
            "/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization,Content-Type",
            },
        )

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://example.com"
        assert "POST" in response.headers.get("access-control-allow-methods", "")
        assert "Authorization" in response.headers.get("access-control-allow-headers", "")
        assert response.headers.get("access-control-max-age") == "600"

    def test_credentials_with_specific_origins(self):
        """Test credentials flow with specific origins."""
        env_vars = {
            "DEBUG": "false",
            "ENVIRONMENT": "production",
            "CORS_ALLOW_ORIGINS": "https://example.com",
            "CORS_ALLOW_CREDENTIALS": "true",
        }
        app = self.create_app_with_cors(env_vars)
        client = TestClient(app)

        response = client.get("/test", headers={"Origin": "https://example.com"})

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://example.com"
        assert response.headers.get("access-control-allow-credentials") == "true"

    def test_no_credentials_with_wildcard(self):
        """Test that credentials are disabled with wildcard origins."""
        env_vars = {
            "DEBUG": "true",
            "ENVIRONMENT": "development",
            "CORS_ALLOW_ORIGINS": "",
            "CORS_ALLOW_CREDENTIALS": "true",  # This should be ignored
        }
        app = self.create_app_with_cors(env_vars)
        client = TestClient(app)

        response = client.get("/test", headers={"Origin": "https://example.com"})

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        # Credentials should not be set with wildcard
        assert response.headers.get("access-control-allow-credentials") != "true"

    def test_localhost_development_support(self):
        """Test that common development origins work in DEBUG mode."""
        env_vars = {
            "DEBUG": "true",
            "ENVIRONMENT": "development",
            "CORS_ALLOW_ORIGINS": "",
        }
        app = self.create_app_with_cors(env_vars)
        client = TestClient(app)

        # Test common development origins
        dev_origins = [
            "http://localhost:3000",
            "http://localhost:8501",  # Streamlit default
            "http://127.0.0.1:3000",
            "http://localhost:8080",
        ]

        for origin in dev_origins:
            response = client.get("/test", headers={"Origin": origin})
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == "*"
