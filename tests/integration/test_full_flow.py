"""Integration test for complete request flow with mocked orchestrator.

This test validates the entire API request/response cycle by mocking
the orchestrator. It tests cache behavior and response transformations
without duplicating the orchestrator unit tests.

Note: Full-flow integration tests that mock SEC and Groq directly would
be valuable but are complex. For now, we test the API layer thoroughly
by mocking the orchestrator (which is already unit tested).
"""

from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from reit_risk_summarizer.main import app
from reit_risk_summarizer.dependencies import get_orchestrator
from reit_risk_summarizer.services.llm.summarizer import RiskSummary

client = TestClient(app)


class TestFullFlowViaOrchestrator:
    """Test complete API flow by mocking the orchestrator."""
    
    def setup_method(self):
        """Reset dependency overrides."""
        app.dependency_overrides.clear()
    
    def teardown_method(self):
        """Clean up after test."""
        app.dependency_overrides.clear()
    
    def test_cache_behavior_across_requests(self):
        """Test cache hit detection across multiple requests."""
        mock_orch = MagicMock()
        
        # First request: no cache
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.return_value = RiskSummary(
            ticker="PLD",
            company_name="Prologis",
            risks=["R1", "R2", "R3", "R4", "R5"],
            model="test",
            prompt_version="v1.0"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response1 = client.get("/api/v1/risks/PLD")
        assert response1.status_code == 200
        assert response1.json()["cached"] is False
        
        # Second request: cache hit
        mock_orch.cache.has.return_value = True
        response2 = client.get("/api/v1/risks/PLD")
        assert response2.status_code == 200
        assert response2.json()["cached"] is True
    
    def test_force_refresh_bypasses_cache(self):
        """Test force_refresh parameter."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = True  # Cache exists
        mock_orch.process_reit.return_value = RiskSummary(
            ticker="EQIX",
            company_name="Equinix",
            risks=["R1", "R2", "R3", "R4", "R5"],
            model="test",
            prompt_version="v1.0"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        # With force_refresh, cached should be False even though cache exists
        response = client.get("/api/v1/risks/EQIX?force_refresh=true")
        assert response.status_code == 200
        # force_refresh overrides cache detection in the endpoint
        assert response.json()["cached"] is False
        mock_orch.process_reit.assert_called_once_with("EQIX", force_refresh=True)
    
    def test_cache_clear_endpoints_workflow(self):
        """Test cache clearing workflow."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.return_value = RiskSummary(
            ticker="SPG",
            company_name="Simon Property Group",
            risks=["R1", "R2", "R3", "R4", "R5"],
            model="test",
            prompt_version="v1.0"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        # Get data
        response1 = client.get("/api/v1/risks/SPG")
        assert response1.status_code == 200
        
        # Clear specific ticker cache
        response2 = client.delete("/api/v1/risks/cache/SPG")
        assert response2.status_code == 204
        mock_orch.clear_cache.assert_called_with("SPG")
        
        # Clear all cache
        response3 = client.delete("/api/v1/risks/cache")
        assert response3.status_code == 204
        # Called twice: once for SPG, once for all
        assert mock_orch.clear_cache.call_count == 2
    
    def test_multiple_tickers_independent(self):
        """Test that different tickers are processed independently."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        
        # Different responses for different tickers
        def mock_process(ticker, force_refresh=False):
            return RiskSummary(
                ticker=ticker,
                company_name=f"Company {ticker}",
                risks=[f"{ticker}-R1", f"{ticker}-R2", f"{ticker}-R3", f"{ticker}-R4", f"{ticker}-R5"],
                model="test",
                prompt_version="v1.0"
            )
        
        mock_orch.process_reit.side_effect = mock_process
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        # Request AMT
        response1 = client.get("/api/v1/risks/AMT")
        assert response1.status_code == 200
        assert response1.json()["ticker"] == "AMT"
        assert response1.json()["risks"][0] == "AMT-R1"
        
        # Request PLD
        response2 = client.get("/api/v1/risks/PLD")
        assert response2.status_code == 200
        assert response2.json()["ticker"] == "PLD"
        assert response2.json()["risks"][0] == "PLD-R1"
        
        # Verify orchestrator called for both
        assert mock_orch.process_reit.call_count == 2
