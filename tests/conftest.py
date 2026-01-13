"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

from reit_risk_summarizer.config import Settings

# Add project root to Python path so evaluation module can be imported
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_settings():
    """Create a mock Settings instance for testing."""
    return Settings(
        sec_api_user_agent="TestUser test@example.com",
        openai_api_key="test-key",
        cache_enabled=False,  # Disable caching in tests
    )
