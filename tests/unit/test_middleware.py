"""Unit tests for FastAPI middleware.

Tests the middleware layer including:
- Logging middleware (request/response logging, timing)
- Error handling middleware (exception to HTTP mapping)

These tests focus on middleware behavior, not the underlying
business logic (which is tested in other unit tests).
"""

import pytest
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from reit_risk_summarizer.middlewares import (
    logging_middleware,
    error_handling_middleware
)
from reit_risk_summarizer.exceptions import (
    SECFetchError,
    RiskExtractionError,
    LLMSummarizationError,
    REITRiskSummarizerError
)


class TestLoggingMiddleware:
    """Test logging middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_adds_process_time_header(self):
        """Logging middleware adds X-Process-Time header."""
        # Mock request
        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.client.host = "127.0.0.1"
        
        # Mock call_next to return response
        async def mock_call_next(req):
            response = Response(content="test", status_code=200)
            return response
        
        # Call middleware
        response = await logging_middleware(request, mock_call_next)
        
        # Check header was added
        assert "X-Process-Time" in response.headers
        # Should be a number (milliseconds)
        assert response.headers["X-Process-Time"].isdigit()
    
    @pytest.mark.asyncio
    async def test_logs_request_and_response(self, caplog):
        """Logging middleware logs request start and completion."""
        caplog.set_level(logging.INFO)
        
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/test"
        request.client.host = "192.168.1.1"
        
        async def mock_call_next(req):
            return Response(content="test", status_code=201)
        
        await logging_middleware(request, mock_call_next)
        
        # Check logs - logger uses structured logging (extra dict)
        assert "Request started" in caplog.text
        assert "Request completed" in caplog.text
        # Method and path are in the extra fields, not the message
        # Just verify the main log messages are present


class TestErrorHandlingMiddleware:
    """Test error handling middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_sec_fetch_error_mapping(self):
        """SECFetchError is mapped to 502 Bad Gateway."""
        request = MagicMock(spec=Request)
        
        async def mock_call_next(req):
            raise SECFetchError("SEC service down", {"ticker": "AMT"})
        
        response = await error_handling_middleware(request, mock_call_next)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 502
        # Parse response body
        import json
        body = json.loads(response.body.decode())
        assert body["error"] == "SECFetchError"
        assert body["message"] == "SEC service down"
        assert body["details"]["ticker"] == "AMT"
    
    @pytest.mark.asyncio
    async def test_risk_extraction_error_mapping(self):
        """RiskExtractionError is mapped to 500 Internal Server Error."""
        request = MagicMock(spec=Request)
        
        async def mock_call_next(req):
            raise RiskExtractionError("Parse failed", {"ticker": "PLD"})
        
        response = await error_handling_middleware(request, mock_call_next)
        
        assert response.status_code == 500
        import json
        body = json.loads(response.body.decode())
        assert body["error"] == "RiskExtractionError"
        assert body["message"] == "Parse failed"
    
    @pytest.mark.asyncio
    async def test_llm_error_mapping(self):
        """LLMSummarizationError is mapped to 500 Internal Server Error."""
        request = MagicMock(spec=Request)
        
        async def mock_call_next(req):
            raise LLMSummarizationError("Groq timeout", {"model": "llama"})
        
        response = await error_handling_middleware(request, mock_call_next)
        
        assert response.status_code == 500
        import json
        body = json.loads(response.body.decode())
        assert body["error"] == "LLMSummarizationError"
    
    @pytest.mark.asyncio
    async def test_generic_app_error_mapping(self):
        """REITRiskSummarizerError is mapped to 500."""
        request = MagicMock(spec=Request)
        
        async def mock_call_next(req):
            raise REITRiskSummarizerError("Generic error", {})
        
        response = await error_handling_middleware(request, mock_call_next)
        
        assert response.status_code == 500
        import json
        body = json.loads(response.body.decode())
        assert body["error"] == "REITRiskSummarizerError"
    
    @pytest.mark.asyncio
    async def test_unexpected_exception_mapping(self):
        """Unexpected exceptions are mapped to 500 with generic message."""
        request = MagicMock(spec=Request)
        
        async def mock_call_next(req):
            raise ValueError("Unexpected error")
        
        response = await error_handling_middleware(request, mock_call_next)
        
        assert response.status_code == 500
        import json
        body = json.loads(response.body.decode())
        assert body["error"] == "InternalServerError"
        assert "unexpected error occurred" in body["message"].lower()
        assert body["details"]["error_type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_successful_request_passthrough(self):
        """Successful requests pass through without modification."""
        request = MagicMock(spec=Request)
        
        expected_response = Response(content="success", status_code=200)
        
        async def mock_call_next(req):
            return expected_response
        
        response = await error_handling_middleware(request, mock_call_next)
        
        # Should return the same response
        assert response == expected_response
        assert response.status_code == 200
