"""Unit tests for risk orchestrator."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from reit_risk_summarizer.exceptions import (
    SECFetchError,
    RiskExtractionError,
    LLMSummarizationError
)
from reit_risk_summarizer.services.orchestrator import RiskOrchestrator
from reit_risk_summarizer.services.llm.summarizer import RiskSummary


class TestRiskOrchestrator:
    """Test suite for RiskOrchestrator."""
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_init_default(self, mock_fetcher, mock_extractor, mock_create_summarizer):
        """Test orchestrator initialization with defaults."""
        orchestrator = RiskOrchestrator()
        
        assert orchestrator.cache_enabled is True
        assert orchestrator.cache is not None
        mock_create_summarizer.assert_called_once_with(provider="groq", model=None)
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_init_custom_provider(self, mock_fetcher, mock_extractor, mock_create_summarizer):
        """Test initialization with custom provider."""
        orchestrator = RiskOrchestrator(
            summarizer_provider="huggingface",
            summarizer_model="meta-llama/Llama-3.2-3B-Instruct"
        )
        
        mock_create_summarizer.assert_called_once_with(
            provider="huggingface",
            model="meta-llama/Llama-3.2-3B-Instruct"
        )
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_init_cache_disabled(self, mock_fetcher, mock_extractor, mock_create_summarizer):
        """Test initialization with cache disabled."""
        orchestrator = RiskOrchestrator(cache_enabled=False)
        
        assert orchestrator.cache_enabled is False
        assert orchestrator.cache is not None  # Still created, just not used
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_process_reit_success(self, mock_fetcher_class, mock_extractor_class, mock_create_summarizer):
        """Test successful REIT processing."""
        # Setup mocks
        mock_fetcher = mock_fetcher_class.return_value
        mock_extractor = mock_extractor_class.return_value
        mock_summarizer = mock_create_summarizer.return_value
        
        mock_fetcher.fetch_latest_10k.return_value = "<html>10-K content</html>"
        mock_extractor.extract_risk_factors.return_value = "Risk factors text..."
        
        mock_summary = RiskSummary(
            risks=["Risk 1", "Risk 2", "Risk 3", "Risk 4", "Risk 5"],
            ticker="AMT",
            company_name="American Tower Corporation",
            model="llama-3.3-70b-versatile",
            prompt_version="v1.0",
            raw_response="1. Risk 1\n2. Risk 2..."
        )
        mock_summarizer.summarize.return_value = mock_summary
        
        # Execute
        orchestrator = RiskOrchestrator()
        result = orchestrator.process_reit("AMT")
        
        # Verify
        assert result == mock_summary
        assert result.ticker == "AMT"
        assert len(result.risks) == 5
        
        mock_fetcher.fetch_latest_10k.assert_called_once_with("AMT")
        mock_extractor.extract_risk_factors.assert_called_once_with("<html>10-K content</html>")
        mock_summarizer.summarize.assert_called_once()
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_process_reit_with_cache(self, mock_fetcher_class, mock_extractor_class, mock_create_summarizer):
        """Test that cache is used on second call."""
        # Setup mocks
        mock_fetcher = mock_fetcher_class.return_value
        mock_extractor = mock_extractor_class.return_value
        mock_summarizer = mock_create_summarizer.return_value
        
        mock_fetcher.fetch_latest_10k.return_value = "<html>10-K content</html>"
        mock_extractor.extract_risk_factors.return_value = "Risk factors text..."
        
        mock_summary = RiskSummary(
            risks=["Risk 1", "Risk 2", "Risk 3", "Risk 4", "Risk 5"],
            ticker="PLD",
            company_name="Prologis",
            model="llama-3.3-70b-versatile",
            prompt_version="v1.0",
            raw_response="1. Risk 1..."
        )
        mock_summarizer.summarize.return_value = mock_summary
        
        # Execute
        orchestrator = RiskOrchestrator(cache_enabled=True)
        
        # First call - should fetch and cache
        result1 = orchestrator.process_reit("PLD")
        assert mock_fetcher.fetch_latest_10k.call_count == 1
        assert mock_extractor.extract_risk_factors.call_count == 1
        
        # Second call - should use cache
        result2 = orchestrator.process_reit("PLD")
        assert mock_fetcher.fetch_latest_10k.call_count == 1  # Not called again
        assert mock_extractor.extract_risk_factors.call_count == 1  # Not called again
        assert mock_summarizer.summarize.call_count == 2  # Summarizer still called (not cached)
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_process_reit_force_refresh(self, mock_fetcher_class, mock_extractor_class, mock_create_summarizer):
        """Test that force_refresh bypasses cache."""
        # Setup mocks
        mock_fetcher = mock_fetcher_class.return_value
        mock_extractor = mock_extractor_class.return_value
        mock_summarizer = mock_create_summarizer.return_value
        
        mock_fetcher.fetch_latest_10k.return_value = "<html>10-K content</html>"
        mock_extractor.extract_risk_factors.return_value = "Risk factors text..."
        
        mock_summary = RiskSummary(
            risks=["Risk 1", "Risk 2", "Risk 3", "Risk 4", "Risk 5"],
            ticker="EQIX",
            company_name="Equinix",
            model="llama-3.3-70b-versatile",
            prompt_version="v1.0",
            raw_response="1. Risk 1..."
        )
        mock_summarizer.summarize.return_value = mock_summary
        
        # Execute
        orchestrator = RiskOrchestrator(cache_enabled=True)
        
        # First call
        result1 = orchestrator.process_reit("EQIX")
        assert mock_fetcher.fetch_latest_10k.call_count == 1
        
        # Second call with force_refresh - should bypass cache
        result2 = orchestrator.process_reit("EQIX", force_refresh=True)
        assert mock_fetcher.fetch_latest_10k.call_count == 2  # Called again
        assert mock_extractor.extract_risk_factors.call_count == 2  # Called again
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_process_reit_fetch_error(self, mock_fetcher_class, mock_extractor_class, mock_create_summarizer):
        """Test handling of SEC fetch errors."""
        # Setup mocks
        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.fetch_latest_10k.side_effect = Exception("Network error")
        
        # Execute and verify
        orchestrator = RiskOrchestrator()
        
        with pytest.raises(SECFetchError, match="Failed to fetch 10-K for AMT"):
            orchestrator.process_reit("AMT", force_refresh=True)
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_process_reit_extraction_error(self, mock_fetcher_class, mock_extractor_class, mock_create_summarizer):
        """Test handling of extraction errors."""
        # Setup mocks
        mock_fetcher = mock_fetcher_class.return_value
        mock_extractor = mock_extractor_class.return_value
        
        mock_fetcher.fetch_latest_10k.return_value = "<html>10-K content</html>"
        mock_extractor.extract_risk_factors.side_effect = Exception("Could not locate Item 1A")
        
        # Execute and verify
        orchestrator = RiskOrchestrator()
        
        with pytest.raises(RiskExtractionError, match="Failed to extract risks for AMT"):
            orchestrator.process_reit("AMT", force_refresh=True)
    
    @patch('reit_risk_summarizer.services.orchestrator.create_summarizer')
    @patch('reit_risk_summarizer.services.orchestrator.RiskFactorExtractor')
    @patch('reit_risk_summarizer.services.orchestrator.SECFetcher')
    def test_process_reit_summarization_error(self, mock_fetcher_class, mock_extractor_class, mock_create_summarizer):
        """Test handling of summarization errors."""
        # Setup mocks
        mock_fetcher = mock_fetcher_class.return_value
        mock_extractor = mock_extractor_class.return_value
        mock_summarizer = mock_create_summarizer.return_value
        
        mock_fetcher.fetch_latest_10k.return_value = "<html>10-K content</html>"
        mock_extractor.extract_risk_factors.return_value = "Risk factors text..."
        mock_summarizer.summarize.side_effect = Exception("API rate limit exceeded")
        
        # Execute and verify
        orchestrator = RiskOrchestrator()
        
        with pytest.raises(LLMSummarizationError, match="Failed to summarize risks for AMT"):
            orchestrator.process_reit("AMT")


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit
