"""Orchestrate the REIT risk summarization pipeline.

This module coordinates the three-step process:
1. Fetch 10-K filing from SEC EDGAR
2. Extract Item 1A risk factors
3. Summarize to top 5 most material risks
"""

import logging
import time
from typing import Optional

from ..exceptions import SECFetchError, RiskExtractionError, LLMSummarizationError
from .cache import MemoryCache
from .sec.fetcher import SECFetcher
from .sec.extractor import RiskFactorExtractor
from .llm.summarizer import create_summarizer, RiskSummary

logger = logging.getLogger(__name__)


class RiskOrchestrator:
    """Orchestrates the end-to-end risk summarization pipeline.
    
    Coordinates fetching, extraction, and summarization with optional caching
    to improve performance and reduce redundant API calls.
    
    Examples:
        >>> orchestrator = RiskOrchestrator()
        >>> result = orchestrator.process_reit("AMT")
        >>> print(result.risks)
        ['Risk 1...', 'Risk 2...', ...]
        
        >>> # With custom provider
        >>> orchestrator = RiskOrchestrator(summarizer_provider="huggingface")
        >>> result = orchestrator.process_reit("PLD")
    """
    
    def __init__(
        self,
        summarizer_provider: str = "groq",
        summarizer_model: Optional[str] = None,
        cache_enabled: bool = True,
        cache: Optional[MemoryCache] = None
    ):
        """Initialize the orchestrator.
        
        Args:
            summarizer_provider: LLM provider ("groq" or "huggingface")
            summarizer_model: Specific model to use (uses provider default if None)
            cache_enabled: Whether to cache intermediate results
            cache: Custom cache instance (creates new MemoryCache if None)
        """
        self.fetcher = SECFetcher()
        self.extractor = RiskFactorExtractor()
        self.summarizer = create_summarizer(
            provider=summarizer_provider,
            model=summarizer_model
        )
        self.cache_enabled = cache_enabled
        self.cache = cache or MemoryCache()
        
        logger.info(
            f"Initialized RiskOrchestrator "
            f"(provider={summarizer_provider}, cache={cache_enabled})"
        )
    
    def process_reit(
        self,
        ticker: str,
        force_refresh: bool = False
    ) -> RiskSummary:
        """Process a REIT through the full pipeline.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AMT", "PLD")
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            RiskSummary with top 5 risks and metadata
            
        Raises:
            SECFetchError: If 10-K filing cannot be fetched
            RiskExtractionError: If risk factors cannot be extracted
            LLMSummarizationError: If summarization fails
            
        Examples:
            >>> orchestrator = RiskOrchestrator()
            >>> result = orchestrator.process_reit("AMT")
            >>> print(f"Found {len(result.risks)} risks for {result.company_name}")
        """
        ticker = ticker.upper()
        start_time = time.time()
        
        logger.info(f"Processing {ticker} (force_refresh={force_refresh})")
        
        try:
            # Step 1: Fetch 10-K HTML
            html = self._fetch_10k(ticker, force_refresh)
            
            # Step 2: Extract risk factors
            risk_text = self._extract_risks(ticker, html, force_refresh)
            
            # Step 3: Summarize to top 5 risks
            summary = self._summarize_risks(ticker, risk_text)
            
            # Add processing time metadata
            elapsed = time.time() - start_time
            logger.info(
                f"Successfully processed {ticker} in {elapsed:.2f}s "
                f"(model={summary.model})"
            )
            
            return summary
            
        except (SECFetchError, RiskExtractionError, LLMSummarizationError) as e:
            # Re-raise known errors with context
            logger.error(f"Failed to process {ticker}: {e}")
            raise
        except Exception as e:
            # Catch unexpected errors
            logger.error(f"Unexpected error processing {ticker}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error processing {ticker}: {e}") from e
    
    def _fetch_10k(self, ticker: str, force_refresh: bool) -> str:
        """Fetch 10-K HTML with caching.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Bypass cache if True
            
        Returns:
            10-K HTML content
        """
        cache_key = f"{ticker}_10k_html"
        
        # Check cache
        if self.cache_enabled and not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached 10-K HTML for {ticker}")
                return cached
        
        # Fetch from SEC
        logger.info(f"Fetching 10-K from SEC EDGAR for {ticker}")
        try:
            html = self.fetcher.fetch_latest_10k(ticker)
        except Exception as e:
            raise SECFetchError(f"Failed to fetch 10-K for {ticker}: {e}") from e
        
        # Cache result
        if self.cache_enabled:
            self.cache.set(cache_key, html)
            logger.debug(f"Cached 10-K HTML for {ticker} ({len(html):,} chars)")
        
        return html
    
    def _extract_risks(
        self,
        ticker: str,
        html: str,
        force_refresh: bool
    ) -> str:
        """Extract risk factors with caching.
        
        Args:
            ticker: Stock ticker symbol
            html: 10-K HTML content
            force_refresh: Bypass cache if True
            
        Returns:
            Extracted risk factors text
        """
        cache_key = f"{ticker}_risk_text"
        
        # Check cache
        if self.cache_enabled and not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached risk text for {ticker}")
                return cached
        
        # Extract from HTML
        logger.info(f"Extracting risk factors for {ticker}")
        try:
            risk_text = self.extractor.extract_risk_factors(html)
        except Exception as e:
            raise RiskExtractionError(f"Failed to extract risks for {ticker}: {e}") from e
        
        # Cache result
        if self.cache_enabled:
            self.cache.set(cache_key, risk_text)
            logger.debug(f"Cached risk text for {ticker} ({len(risk_text):,} chars)")
        
        return risk_text
    
    def _summarize_risks(self, ticker: str, risk_text: str) -> RiskSummary:
        """Summarize risks to top 5.
        
        Note: Summaries are NOT cached since they depend on the specific
        LLM model and parameters used.
        
        Args:
            ticker: Stock ticker symbol
            risk_text: Extracted risk factors text
            
        Returns:
            RiskSummary with top 5 risks
        """
        logger.info(f"Summarizing risks for {ticker}")
        
        # Get company name from ticker (simple mapping for common REITs)
        company_name = self._get_company_name(ticker)
        
        try:
            summary = self.summarizer.summarize(
                risk_text=risk_text,
                ticker=ticker,
                company_name=company_name
            )
        except Exception as e:
            raise LLMSummarizationError(f"Failed to summarize risks for {ticker}: {e}") from e
        
        return summary
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker.
        
        For now, uses a simple mapping of common REITs. In production,
        this could query SEC API for official company name.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company name or ticker if not found
        """
        # Common REIT ticker -> name mapping
        REIT_NAMES = {
            "AMT": "American Tower Corporation",
            "PLD": "Prologis",
            "EQIX": "Equinix",
            "CCI": "Crown Castle",
            "PSA": "Public Storage",
            "DLR": "Digital Realty",
            "WELL": "Welltower",
            "O": "Realty Income",
            "SBAC": "SBA Communications",
            "AVB": "AvalonBay Communities",
        }
        
        return REIT_NAMES.get(ticker, ticker)
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cache for specific ticker or entire cache.
        
        Args:
            ticker: Clear only this ticker's cache. If None, clear all.
            
        Examples:
            >>> orchestrator.clear_cache("AMT")  # Clear AMT only
            >>> orchestrator.clear_cache()       # Clear everything
        """
        if ticker:
            ticker = ticker.upper()
            self.cache.delete(f"{ticker}_10k_html")
            self.cache.delete(f"{ticker}_risk_text")
            logger.info(f"Cleared cache for {ticker}")
        else:
            self.cache.clear()
            logger.info("Cleared entire cache")
