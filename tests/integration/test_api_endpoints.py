"""Integration tests for API endpoints.

Tests the FastAPI layer including:
- Input validation (Pydantic)
- HTTP error responses  
- Endpoint request/response contracts
- Router exception handling

These tests complement unit tests by testing the API layer,
not the underlying business logic (which is tested in unit tests).
"""

from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from reit_risk_summarizer.main import app
from reit_risk_summarizer.dependencies import get_orchestrator
from reit_risk_summarizer.services.llm.summarizer import RiskSummary
from reit_risk_summarizer.exceptions import (
    SECFetchError,
    RiskExtractionError,
    LLMSummarizationError
)

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self):
        """Health endpoint returns 200 with service status."""
        response = client.get("/health")  # No /api/v1 prefix on health endpoint
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestInputValidation:
    """Test Pydantic input validation."""
    
    def setup_method(self):
        """Reset dependency overrides."""
        app.dependency_overrides.clear()
    
    def teardown_method(self):
        """Clean up after test."""
        app.dependency_overrides.clear()
    
    def test_ticker_too_long(self):
        """Ticker exceeding max length is rejected (422)."""
        response = client.get("/api/v1/risks/" + "A" * 11)  # 11 chars exceeds limit
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_ticker_empty_path(self):
        """Empty ticker path returns 404 (route not found)."""
        response = client.get("/api/v1/risks/")
        
        assert response.status_code == 404
    
    def test_valid_ticker_format(self):
        """Valid ticker format passes validation."""
        # Mock orchestrator
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.return_value = RiskSummary(
            ticker="AMT",
            company_name="Test",
            risks=["R1", "R2", "R3", "R4", "R5"],
            model="test",
            prompt_version="v1.0"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/AMT")
        
        # Should not get validation error
        assert response.status_code == 200


class TestErrorHandlingRoutes:
    """Test router exception handling (exceptions -> HTTP codes)."""
    
    def setup_method(self):
        """Reset dependency overrides."""
        app.dependency_overrides.clear()
    
    def teardown_method(self):
        """Clean up after test."""
        app.dependency_overrides.clear()
    
    def test_sec_fetch_error_returns_404(self):
        """SECFetchError is converted to 404 by router."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.side_effect = SECFetchError(
            "INVALID", "Ticker not found"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/INVALID")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "INVALID" in data["detail"]
    
    def test_risk_extraction_error_returns_500(self):
        """RiskExtractionError is converted to 500 by router."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.side_effect = RiskExtractionError(
            "Failed to extract Item 1A",
            {"ticker": "AMT"}
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/AMT")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_llm_error_returns_503(self):
        """LLMSummarizationError is converted to 503 by router."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.side_effect = LLMSummarizationError(
            "Groq API unavailable",
            {"model": "llama-3.3-70b-versatile"}
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/AMT")
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
    
    def test_unexpected_error_returns_500(self):
        """Unexpected exceptions are converted to 500 by router."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.side_effect = ValueError("Unexpected error")
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/AMT")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestRiskEndpoint:
    """Test GET /api/v1/risks/{ticker} endpoint."""
    
    def setup_method(self):
        """Reset dependency overrides."""
        app.dependency_overrides.clear()
    
    def teardown_method(self):
        """Clean up after test."""
        app.dependency_overrides.clear()
    
    def test_successful_risk_response(self):
        """Successful request returns proper RiskResponse format."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = False
        mock_orch.process_reit.return_value = RiskSummary(
            ticker="AMT",
            company_name="American Tower",
            risks=[
                "Market volatility risk",
                "Interest rate risk",
                "Regulatory risk",
                "Tenant concentration risk",
                "Geographic concentration risk"
            ],
            model="llama-3.3-70b-versatile",
            prompt_version="v1.0"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/AMT")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["ticker"] == "AMT"
        assert data["company_name"] == "American Tower"
        assert len(data["risks"]) == 5
        assert data["model"] == "llama-3.3-70b-versatile"
        assert data["prompt_version"] == "v1.0"
        assert "cached" in data
    
    def test_force_refresh_parameter(self):
        """force_refresh parameter bypasses cache."""
        mock_orch = MagicMock()
        mock_orch.cache.has.return_value = True  # Simulate cached data
        mock_orch.process_reit.return_value = RiskSummary(
            ticker="AMT",
            company_name="American Tower",
            risks=["R1", "R2", "R3", "R4", "R5"],
            model="test",
            prompt_version="v1.0"
        )
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.get("/api/v1/risks/AMT?force_refresh=true")
        
        assert response.status_code == 200
        # Verify process_reit was called with force_refresh=True
        mock_orch.process_reit.assert_called_once_with("AMT", force_refresh=True)


class TestCacheEndpoints:
    """Test cache management endpoints."""
    
    def setup_method(self):
        """Reset dependency overrides."""
        app.dependency_overrides.clear()
    
    def teardown_method(self):
        """Clean up after test."""
        app.dependency_overrides.clear()
    
    def test_clear_ticker_cache(self):
        """DELETE /cache/{ticker} clears cache for specific ticker."""
        mock_orch = MagicMock()
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.delete("/api/v1/risks/cache/AMT")
        
        assert response.status_code == 204  # No Content
        # Verify clear_cache was called with ticker
        mock_orch.clear_cache.assert_called_once_with("AMT")
    
    def test_clear_all_cache(self):
        """DELETE /cache clears all cached data."""
        mock_orch = MagicMock()
        app.dependency_overrides[get_orchestrator] = lambda: mock_orch
        
        response = client.delete("/api/v1/risks/cache")
        
        assert response.status_code == 204  # No Content
        # Verify clear_cache was called without parameters
        mock_orch.clear_cache.assert_called_once_with()
