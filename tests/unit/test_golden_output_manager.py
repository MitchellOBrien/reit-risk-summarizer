"""Unit tests for GoldenOutputManager."""

import json
import sys
from pathlib import Path

import pytest

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evaluation"))

from golden_output_manager import GoldenOutputManager
from reit_risk_summarizer.services.llm.summarizer import RiskSummary


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def manager(temp_cache_dir):
    """Create a GoldenOutputManager with temporary cache directory."""
    return GoldenOutputManager(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_risk_summary():
    """Create a sample RiskSummary for testing."""
    return RiskSummary(
        ticker="TEST",
        company_name="Test REIT Company",
        risks=[
            "Interest rate risk from variable-rate debt exposure",
            "Market risk from economic downturn affecting occupancy rates",
            "Regulatory risk from changes in REIT tax requirements",
            "Concentration risk from geographic market exposure",
            "Liquidity risk from refinancing obligations"
        ],
        model="llama-3.3-70b-versatile",
        prompt_version="1.0"
    )


class TestGoldenOutputManagerInit:
    """Test GoldenOutputManager initialization."""

    def test_init_with_custom_cache_dir(self, temp_cache_dir):
        """Test initialization with custom cache directory."""
        manager = GoldenOutputManager(cache_dir=temp_cache_dir)
        assert manager.cache_dir == temp_cache_dir
        assert manager.cache_dir.exists()

    def test_init_creates_cache_dir_if_not_exists(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "nonexistent"
        assert not cache_dir.exists()
        
        manager = GoldenOutputManager(cache_dir=cache_dir)
        assert cache_dir.exists()


class TestGoldenOutputManagerCachePath:
    """Test cache path generation."""

    def test_get_cache_path_uppercase(self, manager, temp_cache_dir):
        """Test that ticker is uppercased in cache path."""
        path = manager.get_cache_path("amt")
        assert path == temp_cache_dir / "AMT.json"

    def test_get_cache_path_already_uppercase(self, manager, temp_cache_dir):
        """Test cache path with already uppercase ticker."""
        path = manager.get_cache_path("AMT")
        assert path == temp_cache_dir / "AMT.json"


class TestGoldenOutputManagerSaveLoad:
    """Test save and load operations."""

    def test_save_and_load_output(self, manager, sample_risk_summary):
        """Test saving and loading a RiskSummary."""
        # Save
        manager.save_output(
            summary=sample_risk_summary,
            input_text_length=5000,
            cache_hit=False
        )

        # Load
        loaded = manager.load_cached_output("TEST")
        
        assert loaded is not None
        assert loaded.ticker == sample_risk_summary.ticker
        assert loaded.company_name == sample_risk_summary.company_name
        assert loaded.risks == sample_risk_summary.risks
        assert loaded.model == sample_risk_summary.model
        assert loaded.prompt_version == sample_risk_summary.prompt_version

    def test_load_nonexistent_ticker_returns_none(self, manager):
        """Test loading a ticker that doesn't exist returns None."""
        result = manager.load_cached_output("NONEXISTENT")
        assert result is None

    def test_save_creates_json_file(self, manager, sample_risk_summary, temp_cache_dir):
        """Test that save creates a JSON file with correct content."""
        manager.save_output(
            summary=sample_risk_summary,
            input_text_length=5000,
            cache_hit=True
        )

        cache_path = temp_cache_dir / "TEST.json"
        assert cache_path.exists()

        # Verify JSON content
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        assert data["ticker"] == "TEST"
        assert data["company_name"] == "Test REIT Company"
        assert len(data["risks"]) == 5
        assert data["model"] == "llama-3.3-70b-versatile"
        assert data["prompt_version"] == "1.0"
        assert data["input_text_length"] == 5000
        assert data["cache_hit"] is True
        assert "generated_at" in data

    def test_load_corrupted_json_returns_none(self, manager, temp_cache_dir):
        """Test that loading corrupted JSON returns None gracefully."""
        # Create corrupted JSON file
        cache_path = temp_cache_dir / "CORRUPT.json"
        with open(cache_path, 'w') as f:
            f.write("{ invalid json }")

        result = manager.load_cached_output("CORRUPT")
        assert result is None


class TestGoldenOutputManagerHasCached:
    """Test has_cached_output checks."""

    def test_has_cached_output_true(self, manager, sample_risk_summary):
        """Test has_cached_output returns True for existing cache."""
        manager.save_output(sample_risk_summary)
        assert manager.has_cached_output("TEST") is True

    def test_has_cached_output_false(self, manager):
        """Test has_cached_output returns False for non-existing cache."""
        assert manager.has_cached_output("NONEXISTENT") is False

    def test_has_cached_output_case_insensitive(self, manager, sample_risk_summary):
        """Test has_cached_output is case-insensitive."""
        manager.save_output(sample_risk_summary)
        assert manager.has_cached_output("test") is True
        assert manager.has_cached_output("TEST") is True
        assert manager.has_cached_output("TeSt") is True


class TestGoldenOutputManagerDelete:
    """Test delete operations."""

    def test_delete_output_existing(self, manager, sample_risk_summary):
        """Test deleting an existing cached output."""
        manager.save_output(sample_risk_summary)
        assert manager.has_cached_output("TEST") is True

        result = manager.delete_output("TEST")
        assert result is True
        assert manager.has_cached_output("TEST") is False

    def test_delete_output_nonexistent(self, manager):
        """Test deleting a non-existent cached output."""
        result = manager.delete_output("NONEXISTENT")
        assert result is False


class TestGoldenOutputManagerList:
    """Test listing cached tickers."""

    def test_list_cached_tickers_empty(self, manager):
        """Test listing tickers when cache is empty."""
        tickers = manager.list_cached_tickers()
        assert tickers == []

    def test_list_cached_tickers_with_data(self, manager):
        """Test listing tickers with cached data."""
        # Create multiple cached outputs
        for ticker in ["AMT", "PLD", "EQIX"]:
            summary = RiskSummary(
                ticker=ticker,
                company_name=f"{ticker} Company",
                risks=[f"Risk {i}" for i in range(5)],
                model="llama-3.3-70b-versatile",
                prompt_version="1.0"
            )
            manager.save_output(summary)

        tickers = manager.list_cached_tickers()
        assert sorted(tickers) == ["AMT", "EQIX", "PLD"]


class TestGoldenOutputManagerClearAll:
    """Test clear all operation."""

    def test_clear_all_empty_cache(self, manager):
        """Test clearing an empty cache."""
        count = manager.clear_all()
        assert count == 0

    def test_clear_all_with_data(self, manager):
        """Test clearing cache with multiple files."""
        # Create multiple cached outputs
        for ticker in ["AMT", "PLD", "EQIX"]:
            summary = RiskSummary(
                ticker=ticker,
                company_name=f"{ticker} Company",
                risks=[f"Risk {i}" for i in range(5)],
                model="llama-3.3-70b-versatile",
                prompt_version="1.0"
            )
            manager.save_output(summary)

        assert len(manager.list_cached_tickers()) == 3

        count = manager.clear_all()
        assert count == 3
        assert len(manager.list_cached_tickers()) == 0
