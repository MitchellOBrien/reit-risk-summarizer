"""
Unit tests for evaluation pipeline components.

Focused on critical data transformation logic:
- Golden dataset parsing (grouping, sorting)
- Ticker processing with cache scenarios
- Aggregate metrics calculation
- Edge cases

Target: 6-7 essential tests covering core functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from evaluation.run_evaluation import load_golden_dataset, process_ticker
from evaluation.golden_output_manager import GoldenOutputManager
from reit_risk_summarizer.services.llm.summarizer import RiskSummary


class TestGoldenDatasetLoading:
    """Test golden dataset parsing and transformation."""
    
    def test_groups_and_sorts_risks_by_ticker(self, tmp_path):
        """Ensure risks are correctly grouped by ticker and sorted by rank."""
        # Create test CSV with out-of-order ranks
        csv_content = """ticker,company_name,sector,filing_year,risk_rank,risk_category,risk_title,risk_description,why_material,unique_to_sector
PLD,Prologis,Industrial/Logistics,2023,2,Customer,Amazon,Amazon is 6.4%,Key tenant loss,No
PLD,Prologis,Industrial/Logistics,2023,1,Geographic,California,30% in California,Downturn impacts,No
AMT,American Tower,Infrastructure/Towers,2023,1,Customer,Carriers,Revenue from carriers,Carrier loss,Yes
"""
        csv_path = tmp_path / "test_golden.csv"
        csv_path.write_text(csv_content)
        
        # Load dataset
        tickers = load_golden_dataset(csv_path)
        
        # Verify structure and metadata
        assert len(tickers) == 2
        assert tickers[0]["ticker"] == "PLD"
        assert tickers[0]["company_name"] == "Prologis"
        assert tickers[0]["sector"] == "Industrial/Logistics"
        assert len(tickers[0]["expert_risks"]) == 2
        
        # Verify risks are sorted by rank (not CSV order)
        assert "30% in California" in tickers[0]["expert_risks"][0]  # Rank 1
        assert "Amazon is 6.4%" in tickers[0]["expert_risks"][1]  # Rank 2


class TestTickerProcessing:
    """Test ticker processing logic with different cache scenarios."""
    
    def test_returns_cached_output_when_available(self):
        """Should use cached output if available and use_cached=True."""
        # Mock components
        mock_manager = Mock(spec=GoldenOutputManager)
        mock_orchestrator = Mock()
        
        cached_summary = RiskSummary(
            ticker="TEST",
            company_name="Test Corp",
            risks=["Risk 1", "Risk 2", "Risk 3", "Risk 4", "Risk 5"],
            model="test-model",
            prompt_version="v1.0"
        )
        mock_manager.load_cached_output.return_value = cached_summary
        
        # Process ticker
        result = process_ticker(
            "TEST",
            mock_orchestrator,
            mock_manager,
            use_cached=True,
            regenerate=False,
            cached_only=False
        )
        
        # Verify used cache
        mock_manager.load_cached_output.assert_called_once_with("TEST")
        mock_orchestrator.process_reit.assert_not_called()
        assert result["ticker"] == "TEST"
        assert result["source"] == "cached_golden_output"
        assert len(result["risks"]) == 5
    
    def test_returns_none_when_cached_only_and_no_cache(self):
        """Should return None in cached_only mode when cache missing."""
        mock_manager = Mock(spec=GoldenOutputManager)
        mock_orchestrator = Mock()
        
        # No cached output
        mock_manager.load_cached_output.return_value = None
        
        # Process ticker in cached_only mode
        result = process_ticker(
            "TEST",
            mock_orchestrator,
            mock_manager,
            use_cached=True,
            regenerate=False,
            cached_only=True
        )
        
        # Should return None without calling API
        assert result is None
        mock_orchestrator.process_reit.assert_not_called()
    
    def test_skips_cache_when_regenerate_true(self):
        """Should ignore cache and call API when regenerate=True."""
        mock_manager = Mock(spec=GoldenOutputManager)
        mock_orchestrator = Mock()
        
        # Setup mock API response
        api_summary = RiskSummary(
            ticker="TEST",
            company_name="Test Corp",
            risks=["New Risk 1", "New Risk 2", "New Risk 3", "New Risk 4", "New Risk 5"],
            model="test-model",
            prompt_version="v1.0"
        )
        mock_orchestrator.process_reit.return_value = api_summary
        
        # Process with regenerate=True
        result = process_ticker(
            "TEST",
            mock_orchestrator,
            mock_manager,
            use_cached=True,
            regenerate=True,
            cached_only=False
        )
        
        # Should not check cache
        mock_manager.load_cached_output.assert_not_called()
        mock_orchestrator.process_reit.assert_called_once()
        assert result["source"] == "groq_api"


class TestAggregateMetrics:
    """Test aggregate metrics calculation logic."""
    
    def test_calculates_mean_min_max_correctly(self):
        """Verify aggregate statistics are computed correctly."""
        ticker_results = [
            {"ticker": "T1", "metrics": {"semantic_similarity": 0.8, "ndcg_at_5": 0.9, "sector_specificity": 0.6}},
            {"ticker": "T2", "metrics": {"semantic_similarity": 0.6, "ndcg_at_5": 0.7, "sector_specificity": 0.4}},
            {"ticker": "T3", "metrics": {"semantic_similarity": 1.0, "ndcg_at_5": 0.8, "sector_specificity": 0.5}},
        ]
        
        # Calculate aggregates (simulate run_evaluation.py logic)
        valid_metrics = [t["metrics"] for t in ticker_results if t.get("metrics")]
        
        aggregates = {}
        for metric_name in ["semantic_similarity", "ndcg_at_5", "sector_specificity"]:
            values = [m[metric_name] for m in valid_metrics]
            aggregates[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        # Verify calculations
        assert aggregates["semantic_similarity"]["mean"] == pytest.approx(0.8)
        assert aggregates["semantic_similarity"]["min"] == 0.6
        assert aggregates["semantic_similarity"]["max"] == 1.0
        assert aggregates["ndcg_at_5"]["mean"] == pytest.approx(0.8)
        assert aggregates["sector_specificity"]["mean"] == pytest.approx(0.5)
    
    def test_handles_missing_metrics_gracefully(self):
        """Should skip tickers without metrics and handle empty results."""
        # Mix of valid and missing metrics
        ticker_results = [
            {"ticker": "T1", "metrics": {"semantic_similarity": 0.8, "ndcg_at_5": 0.9, "sector_specificity": 0.6}},
            {"ticker": "T2"},  # No metrics key
            {"ticker": "T3", "metrics": None},  # Explicit None
        ]
        
        valid_metrics = [t["metrics"] for t in ticker_results if t.get("metrics")]
        assert len(valid_metrics) == 1
        
        # Empty results should not crash
        empty_results = []
        valid_empty = [t["metrics"] for t in empty_results if t.get("metrics")]
        assert len(valid_empty) == 0
