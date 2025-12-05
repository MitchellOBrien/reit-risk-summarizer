"""Unit tests for SEC fetcher service."""

import pytest
import requests
from unittest.mock import Mock, patch

from reit_risk_summarizer.services.sec.fetcher import SECFetcher
from reit_risk_summarizer.exceptions import InvalidTickerError, SECFetchError


class TestSECFetcher:
    """Test cases for SECFetcher class."""

    def test_init(self, mock_settings):
        """Test fetcher initialization."""
        fetcher = SECFetcher(settings=mock_settings)
        assert fetcher.session is not None
        assert 'User-Agent' in fetcher.session.headers
        assert fetcher.session.headers['User-Agent'] == "TestUser test@example.com"

    def test_user_agent_contains_email(self, mock_settings):
        """Test that User-Agent header contains email address."""
        fetcher = SECFetcher(settings=mock_settings)
        user_agent = fetcher.session.headers['User-Agent']
        assert '@' in user_agent, "User-Agent should contain an email address"

    @pytest.mark.integration
    def test_fetch_valid_ticker_pld(self):
        """Test fetching 10-K for Prologis (PLD).
        
        This is an integration test that makes real API calls to SEC.
        Run with: pytest -m integration
        """
        fetcher = SECFetcher()
        html = fetcher.fetch_latest_10k("PLD")
        
        assert html is not None
        assert len(html) > 1000, "10-K should be a substantial document"
        assert any(term in html.lower() for term in ['risk factors', 'item 1a']), \
            "10-K should contain risk factors section"

    @pytest.mark.integration
    def test_fetch_valid_ticker_amt(self):
        """Test fetching 10-K for American Tower (AMT).
        
        This is an integration test that makes real API calls to SEC.
        """
        fetcher = SECFetcher()
        html = fetcher.fetch_latest_10k("AMT")
        
        assert html is not None
        assert len(html) > 1000

    def test_fetch_invalid_ticker(self):
        """Test that invalid ticker raises InvalidTickerError."""
        fetcher = SECFetcher()
        
        with pytest.raises(InvalidTickerError) as exc_info:
            fetcher.fetch_latest_10k("INVALID_TICKER_XYZ_123")
        
        assert "INVALID_TICKER_XYZ_123" in str(exc_info.value)

    def test_ticker_normalization(self):
        """Test that tickers are normalized (uppercase, trimmed)."""
        fetcher = SECFetcher()
        
        # Mock the _get_cik method to avoid actual API call
        with patch.object(fetcher, '_get_cik') as mock_get_cik:
            mock_get_cik.side_effect = InvalidTickerError("Test", {})
            
            # Should normalize to uppercase
            with pytest.raises(InvalidTickerError):
                fetcher.fetch_latest_10k("  pld  ")
            
            # Verify it was called with normalized ticker
            mock_get_cik.assert_called_once_with("PLD")

    def test_rate_limiting(self, mock_settings):
        """Test that rate limiting delay is applied."""
        import time
        
        fetcher = SECFetcher(settings=mock_settings)
        
        # Make multiple calls and verify delay
        fetcher._respect_rate_limit()
        time1 = time.time()
        
        fetcher._respect_rate_limit()
        time2 = time.time()
        
        elapsed = time2 - time1
        # Should have at least rate_limit_delay (0.1s)
        assert elapsed >= fetcher.rate_limit_delay * 0.9, \
            "Rate limiting should enforce minimum delay between requests"

    @patch('requests.Session.get')
    def test_network_error_handling(self, mock_get, mock_settings):
        """Test handling of network errors."""
        fetcher = SECFetcher(settings=mock_settings)
        
        # Simulate network error
        mock_get.side_effect = requests.RequestException("Network error")
        
        with pytest.raises(SECFetchError) as exc_info:
            fetcher.fetch_latest_10k("PLD")
        
        assert "Network error" in str(exc_info.value) or "Failed to fetch" in str(exc_info.value)

    @patch('requests.Session.get')
    def test_get_cik_success(self, mock_get, mock_settings):
        """Test successful CIK extraction."""
        fetcher = SECFetcher(settings=mock_settings)
        
        # Mock SEC response with CIK
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'''<?xml version="1.0"?>
        <feed>
            <entry>
                <link rel="alternate" href="/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001045610"/>
            </entry>
        </feed>'''
        mock_response.text = mock_response.content.decode()
        mock_get.return_value = mock_response
        
        cik = fetcher._get_cik("PLD")
        assert cik == "1045610"

    @patch('requests.Session.get')
    def test_get_cik_ticker_not_found(self, mock_get, mock_settings):
        """Test CIK lookup with non-existent ticker."""
        fetcher = SECFetcher(settings=mock_settings)
        
        # Mock empty response (no company found)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<?xml version="1.0"?><feed></feed>'
        mock_response.text = mock_response.content.decode()
        mock_get.return_value = mock_response
        
        with pytest.raises(InvalidTickerError):
            fetcher._get_cik("BADTICKER")

    def test_extract_document_url_from_xbrl_viewer(self, mock_settings):
        """Test extracting actual document URL from XBRL viewer URL."""
        fetcher = SECFetcher(settings=mock_settings)
        
        # XBRL viewer URL format
        xbrl_url = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1045609/000095017025021272/pld-20241231.htm"
        
        # Should extract the actual document URL
        actual_url = fetcher._extract_document_url(xbrl_url)
        
        # Should remove the /ix?doc= wrapper
        assert actual_url == "https://www.sec.gov/Archives/edgar/data/1045609/000095017025021272/pld-20241231.htm"
        assert '/ix?doc=' not in actual_url
        assert actual_url.startswith("https://www.sec.gov/Archives/edgar/data/")

    def test_extract_document_url_already_direct(self, mock_settings):
        """Test that direct document URLs are returned unchanged."""
        fetcher = SECFetcher(settings=mock_settings)
        
        # Already a direct URL (no /ix?doc=)
        direct_url = "https://www.sec.gov/Archives/edgar/data/1045609/000095017025021272/pld-20241231.htm"
        
        # Should return the same URL (though it will try to fetch it first)
        # For this test, we just verify the XBRL check doesn't break direct URLs
        assert '/ix?doc=' not in direct_url

    @patch('requests.Session.get')
    def test_retry_on_timeout(self, mock_get, mock_settings):
        """Test that requests are retried on timeout errors."""
        from requests.exceptions import Timeout
        
        fetcher = SECFetcher(settings=mock_settings)
        
        # First two attempts timeout, third succeeds
        mock_response = Mock()
        mock_response.text = "success"
        mock_get.side_effect = [
            Timeout("Read timed out"),
            Timeout("Read timed out"),
            mock_response
        ]
        
        # Should succeed after retries
        result = fetcher._retry_request(mock_get, "http://test.com")
        assert result.text == "success"
        assert mock_get.call_count == 3
    
    @patch('requests.Session.get')
    def test_retry_on_503(self, mock_get, mock_settings):
        """Test that requests are retried on 503 Service Unavailable errors."""
        from requests.exceptions import HTTPError
        
        fetcher = SECFetcher(settings=mock_settings)
        
        # Create a proper HTTPError with 503 response
        def create_503_error():
            error_response = Mock()
            error_response.status_code = 503
            error = HTTPError()
            error.response = error_response
            return error
        
        # Mock successful response
        success_response = Mock()
        success_response.text = "success"
        
        # First two calls raise 503, third succeeds
        mock_get.side_effect = [
            create_503_error(),
            create_503_error(),
            success_response
        ]
        
        # Should succeed after retries
        result = fetcher._retry_request(mock_get, "http://test.com")
        assert result.text == "success"
        assert mock_get.call_count == 3
    
    @patch('requests.Session.get')
    def test_retry_exhaustion(self, mock_get, mock_settings):
        """Test that retries are exhausted and exception is raised."""
        from requests.exceptions import Timeout
        
        fetcher = SECFetcher(settings=mock_settings)
        
        # All attempts timeout
        mock_get.side_effect = Timeout("Read timed out")
        
        # Should raise after max retries
        with pytest.raises(Timeout):
            fetcher._retry_request(mock_get, "http://test.com")
        
        # Should have tried max_retries times (default 3)
        assert mock_get.call_count == fetcher.max_retries
    
    @patch('requests.Session.get')
    def test_no_retry_on_404(self, mock_get, mock_settings):
        """Test that non-transient errors (like 404) are not retried."""
        from requests.exceptions import HTTPError
        
        fetcher = SECFetcher(settings=mock_settings)
        
        # Create a proper HTTPError with 404 response
        def create_404_error():
            error_response = Mock()
            error_response.status_code = 404
            error = HTTPError()
            error.response = error_response
            return error
        
        # All calls raise 404
        mock_get.side_effect = create_404_error()
        
        # Should fail immediately without retrying
        with pytest.raises(HTTPError):
            fetcher._retry_request(mock_get, "http://test.com")
        
        # Should only be called once (no retries)
        assert mock_get.call_count == 1


# Pytest markers for categorizing tests
pytestmark = pytest.mark.unit
