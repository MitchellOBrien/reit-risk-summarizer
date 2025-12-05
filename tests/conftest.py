"""Pytest configuration and shared fixtures."""

import pytest

from reit_risk_summarizer.config import Settings


@pytest.fixture
def mock_settings():
    """Create a mock Settings instance for testing."""
    return Settings(
        sec_api_user_agent="TestUser test@example.com",
        openai_api_key="test-key",
        cache_enabled=False,  # Disable caching in tests
    )
