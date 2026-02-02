"""Unit tests for generate_evaluation_report.py script.

Tests the report generation functionality including:
- Happy path with complete data
- Error handling for missing files
- Ticker limiting (first 5 only)
- Number formatting
- Missing summary metrics
"""

import json
import pytest
from pathlib import Path
import sys

# Import the function we're testing
from scripts.generate_evaluation_report import generate_report


@pytest.fixture
def mock_complete_results(tmp_path):
    """Fixture that creates a complete evaluation_results.json with all fields."""
    results = {
        "metadata": {
            "run_date": "2026-01-28T09:00:00Z",
            "total_tickers": 10,
            "cached_only_mode": True
        },
        "tickers": [
            {
                "ticker": "AMT",
                "company_name": "American Tower",
                "sector": "Infrastructure",
                "metrics": {
                    "semantic_similarity": 0.82,
                    "ndcg_at_5": 0.78,
                    "sector_specificity": 0.65
                },
                "source": "cached_golden_output"
            },
            {
                "ticker": "PLD",
                "company_name": "Prologis",
                "sector": "Industrial",
                "metrics": {
                    "semantic_similarity": 0.75,
                    "ndcg_at_5": 1.0,
                    "sector_specificity": 0.54
                },
                "source": "cached_golden_output"
            },
            {
                "ticker": "EQIX",
                "company_name": "Equinix",
                "sector": "Data Centers",
                "metrics": {
                    "semantic_similarity": 0.68,
                    "ndcg_at_5": 0.92,
                    "sector_specificity": 0.73
                },
                "source": "cached_golden_output"
            }
        ],
        "summary_metrics": {
            "semantic_similarity": {"mean": 0.75, "min": 0.55, "max": 1.0},
            "ndcg_at_5": {"mean": 0.90, "min": 0.79, "max": 1.0},
            "sector_specificity": {"mean": 0.62, "min": 0.54, "max": 0.73}
        }
    }
    
    json_path = tmp_path / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return json_path


class TestHappyPath:
    """Tests for normal, expected usage."""
    
    def test_generate_report_with_complete_data(self, mock_complete_results, tmp_path):
        """Test report generation with full, valid evaluation results."""
        output_path = tmp_path / "report.txt"
        
        # Execute
        generate_report(mock_complete_results, output_path)
        
        # Assert output file was created
        assert output_path.exists(), "Report file should be created"
        
        # Read the report
        with open(output_path) as f:
            report = f.read()
        
        # Validate content
        assert "# Evaluation Results" in report
        assert "Run Date: 2026-01-28T09:00:00Z" in report
        assert "Tickers Evaluated: 3/10" in report
        
        # Validate aggregate metrics section
        assert "## Aggregate Metrics" in report
        assert "Semantic Similarity: 0.750" in report
        assert "NDCG@5: 0.900" in report
        assert "Sector-Specificity: 0.620" in report
        
        # Validate per-ticker section
        assert "## Per-Ticker Results" in report
        assert "AMT:" in report
        assert "PLD:" in report
        assert "EQIX:" in report


class TestErrorHandling:
    """Tests for error conditions."""
    
    def test_missing_json_file(self, tmp_path):
        """Test behavior when evaluation_results.json doesn't exist."""
        nonexistent_path = tmp_path / "does_not_exist.json"
        output_path = tmp_path / "report.txt"
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            generate_report(nonexistent_path, output_path)
        
        assert exc_info.value.code == 1
        
        # Output file should NOT be created
        assert not output_path.exists()


class TestTickerLimiting:
    """Tests for ticker limiting functionality."""
    
    def test_only_shows_first_five_tickers(self, tmp_path):
        """Test that report limits to first 5 tickers when more exist."""
        # Create results with 10 tickers
        results = {
            "metadata": {
                "run_date": "2026-01-28T09:00:00Z",
                "total_tickers": 10,
                "cached_only_mode": True
            },
            "tickers": [
                {
                    "ticker": f"TICK{i}",
                    "company_name": f"Company {i}",
                    "sector": "Test",
                    "metrics": {
                        "semantic_similarity": 0.8,
                        "ndcg_at_5": 0.9,
                        "sector_specificity": 0.6
                    }
                }
                for i in range(10)
            ],
            "summary_metrics": {
                "semantic_similarity": {"mean": 0.75, "min": 0.55, "max": 1.0},
                "ndcg_at_5": {"mean": 0.90, "min": 0.79, "max": 1.0},
                "sector_specificity": {"mean": 0.62, "min": 0.54, "max": 0.73}
            }
        }
        
        json_path = tmp_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f)
        
        output_path = tmp_path / "report.txt"
        generate_report(json_path, output_path)
        
        with open(output_path) as f:
            report = f.read()
        
        # Should show all 10 in summary
        assert "Tickers Evaluated: 10/10" in report
        
        # Should show first 5 in per-ticker results
        assert "TICK0:" in report
        assert "TICK1:" in report
        assert "TICK2:" in report
        assert "TICK3:" in report
        assert "TICK4:" in report
        
        # Should NOT show tickers 5-9
        assert "TICK5:" not in report
        assert "TICK6:" not in report
        assert "TICK9:" not in report


class TestNumberFormatting:
    """Tests for number formatting."""
    
    def test_metric_formatting_three_decimals(self, tmp_path):
        """Test that metrics are formatted to exactly 3 decimal places."""
        results = {
            "metadata": {
                "run_date": "2026-01-28T09:00:00Z",
                "total_tickers": 3,
                "cached_only_mode": True
            },
            "tickers": [
                {
                    "ticker": "TEST",
                    "company_name": "Test Company",
                    "sector": "Test",
                    "metrics": {
                        "semantic_similarity": 0.123456,  # Should round to 0.123
                        "ndcg_at_5": 0.7,                 # Should show 0.700
                        "sector_specificity": 1.0         # Should show 1.000
                    }
                }
            ],
            "summary_metrics": {
                "semantic_similarity": {"mean": 0.9876543, "min": 0.5, "max": 1.0},
                "ndcg_at_5": {"mean": 0.1, "min": 0.0, "max": 0.2},
                "sector_specificity": {"mean": 0.555555, "min": 0.5, "max": 0.6}
            }
        }
        
        json_path = tmp_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f)
        
        output_path = tmp_path / "report.txt"
        generate_report(json_path, output_path)
        
        with open(output_path) as f:
            report = f.read()
        
        # Check summary metrics formatting (3 decimals)
        assert "Semantic Similarity: 0.988" in report  # Rounded from 0.9876543
        assert "NDCG@5: 0.100" in report              # Shows trailing zeros
        assert "Sector-Specificity: 0.556" in report  # Rounded from 0.555555
        
        # Check per-ticker formatting (3 decimals)
        assert "sim=0.123" in report  # Rounded from 0.123456
        assert "ndcg=0.700" in report  # Shows trailing zeros
        assert "spec=1.000" in report  # Shows trailing zeros


class TestMissingData:
    """Tests for handling missing or incomplete data."""
    
    def test_missing_summary_metrics(self, tmp_path):
        """Test when summary_metrics field is not present."""
        results = {
            "metadata": {
                "run_date": "2026-01-28T09:00:00Z",
                "total_tickers": 2,
                "cached_only_mode": True
            },
            "tickers": [
                {
                    "ticker": "AMT",
                    "company_name": "American Tower",
                    "sector": "Infrastructure",
                    "metrics": {
                        "semantic_similarity": 0.82,
                        "ndcg_at_5": 0.78,
                        "sector_specificity": 0.65
                    }
                }
            ]
            # No summary_metrics field
        }
        
        json_path = tmp_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f)
        
        output_path = tmp_path / "report.txt"
        generate_report(json_path, output_path)
        
        with open(output_path) as f:
            report = f.read()
        
        # Should still generate report
        assert "# Evaluation Results" in report
        assert "Run Date: 2026-01-28T09:00:00Z" in report
        
        # Should NOT have aggregate metrics section
        assert "## Aggregate Metrics" not in report
        assert "Semantic Similarity: 0." not in report  # No summary metrics
        
        # Should still show per-ticker results
        assert "## Per-Ticker Results" in report
        assert "AMT:" in report
        assert "sim=0.820" in report
