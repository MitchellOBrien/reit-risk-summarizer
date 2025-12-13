"""Risk summarization API endpoints.

This module provides RESTful endpoints for processing REIT 10-K filings and 
extracting top material risks using LLM-powered summarization.

Endpoints:
    GET /api/v1/risks/{ticker} - Get top 5 risks for a REIT
    DELETE /api/v1/risks/cache/{ticker} - Clear cache for specific ticker
    DELETE /api/v1/risks/cache - Clear all cached data

The main endpoint orchestrates a 3-step pipeline:
1. Fetch latest 10-K filing from SEC EDGAR
2. Extract Item 1A Risk Factors section
3. Summarize to top 5 most material risks using LLM

Results are cached (HTML and extracted text) to improve performance on 
repeated requests. Summaries are not cached since they depend on model params.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

from ..dependencies import OrchestratorDep
from ..exceptions import SECFetchError, RiskExtractionError, LLMSummarizationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risks", tags=["risks"])


class RiskResponse(BaseModel):
    """Response model for risk summarization."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    risks: list[str] = Field(..., description="Top 5 material risks", min_length=5, max_length=5)
    model: str = Field(..., description="LLM model used for summarization")
    prompt_version: str = Field(..., description="Prompt version used")
    cached: bool = Field(..., description="Whether results were served from cache")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AMT",
                "company_name": "American Tower Corporation",
                "risks": [
                    "Significant exposure to technological disruption in telecommunications infrastructure...",
                    "Heavy debt burden with $40B+ in long-term obligations...",
                    "Geographic concentration risk in specific markets...",
                    "Regulatory uncertainty across multiple jurisdictions...",
                    "Tenant concentration with major wireless carriers..."
                ],
                "model": "llama-3.3-70b-versatile",
                "prompt_version": "v1.0",
                "cached": False
            }
        }


@router.get(
    "/{ticker}",
    response_model=RiskResponse,
    summary="Get top 5 risks for a REIT",
    description="Process a REIT's latest 10-K filing and return the top 5 most material risks",
    responses={
        200: {"description": "Successfully retrieved risk summary"},
        404: {"description": "Ticker not found or no 10-K filing available"},
        500: {"description": "Internal processing error"},
        503: {"description": "LLM service unavailable"}
    }
)
def get_risks(
    ticker: str = Path(..., description="Stock ticker symbol (e.g., AMT, PLD)", min_length=1, max_length=10),
    orchestrator: OrchestratorDep = None,
    force_refresh: bool = Query(False, description="Bypass cache and fetch fresh data"),
    model: Optional[str] = Query(None, description="Override default LLM model")
) -> RiskResponse:
    """Get top 5 material risks for a REIT.
    
    This endpoint:
    1. Fetches the latest 10-K filing from SEC EDGAR
    2. Extracts Item 1A Risk Factors section
    3. Uses LLM to summarize to top 5 most material risks
    
    Results are cached to improve performance on repeated requests.
    Use `force_refresh=true` to bypass the cache.
    
    Args:
        ticker: Stock ticker symbol
        orchestrator: Injected orchestrator dependency
        force_refresh: Whether to bypass cache
        model: Optional model override
        
    Returns:
        RiskResponse with top 5 risks and metadata
        
    Raises:
        HTTPException: 404 if ticker not found, 500 for processing errors
    """
    ticker = ticker.upper()
    
    # Check if cache was used (before processing)
    cache_key_html = f"{ticker}_10k_html"
    cache_key_risk = f"{ticker}_risk_text"
    was_cached = (
        orchestrator.cache.has(cache_key_html) and 
        orchestrator.cache.has(cache_key_risk) and 
        not force_refresh
    )
    
    logger.info(f"Processing request for {ticker} (force_refresh={force_refresh}, cached={was_cached})")
    
    try:
        # Process through orchestrator
        summary = orchestrator.process_reit(ticker, force_refresh=force_refresh)
        
        return RiskResponse(
            ticker=summary.ticker,
            company_name=summary.company_name,
            risks=summary.risks,
            model=summary.model,
            prompt_version=summary.prompt_version,
            cached=was_cached
        )
        
    except SECFetchError as e:
        logger.error(f"Failed to fetch 10-K for {ticker}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch 10-K filing for {ticker}. Ticker may be invalid or filing unavailable."
        )
        
    except RiskExtractionError as e:
        logger.error(f"Failed to extract risks for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract risk factors from 10-K filing for {ticker}."
        )
        
    except LLMSummarizationError as e:
        logger.error(f"Failed to summarize risks for {ticker}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable. Please try again later."
        )
        
    except Exception as e:
        logger.error(f"Unexpected error processing {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing {ticker}."
        )


@router.delete(
    "/cache/{ticker}",
    summary="Clear cache for a specific ticker",
    description="Remove cached 10-K and risk data for a specific ticker",
    status_code=204
)
def clear_ticker_cache(
    ticker: str = Path(..., description="Stock ticker symbol"),
    orchestrator: OrchestratorDep = None
):
    """Clear cached data for a specific ticker.
    
    This removes both the cached 10-K HTML and extracted risk text
    for the specified ticker. The next request will fetch fresh data.
    
    Args:
        ticker: Stock ticker symbol
        orchestrator: Injected orchestrator dependency
    """
    ticker = ticker.upper()
    logger.info(f"Clearing cache for {ticker}")
    orchestrator.clear_cache(ticker)
    return None  # 204 No Content


@router.delete(
    "/cache",
    summary="Clear all cache",
    description="Remove all cached data for all tickers",
    status_code=204
)
def clear_all_cache(orchestrator: OrchestratorDep = None):
    """Clear all cached data.
    
    This removes all cached 10-K HTML and risk text for all tickers.
    All subsequent requests will fetch fresh data.
    
    Args:
        orchestrator: Injected orchestrator dependency
    """
    logger.info("Clearing all cache")
    orchestrator.clear_cache()
    return None  # 204 No Content

    # TODO: Implement orchestrator call
    # result = await orchestrator.process(ticker, force_refresh=force_refresh)
    # return result

    # Temporary mock response for testing
    from datetime import datetime

    from ..schemas.responses import Metadata, Risk

    return SummarizeRiskResponse(
        ticker=ticker,
        company_name="Mock Company",
        risks=[
            Risk(
                rank=1,
                title="Mock Risk 1",
                description="This is a placeholder risk description.",
                category="Mock Category",
            ),
            Risk(
                rank=2,
                title="Mock Risk 2",
                description="This is another placeholder risk.",
                category="Mock Category",
            ),
            Risk(
                rank=3,
                title="Mock Risk 3",
                description="Yet another placeholder risk.",
                category="Mock Category",
            ),
            Risk(
                rank=4,
                title="Mock Risk 4",
                description="One more placeholder risk.",
                category="Mock Category",
            ),
            Risk(
                rank=5,
                title="Mock Risk 5",
                description="Final placeholder risk.",
                category="Mock Category",
            ),
        ],
        metadata=Metadata(
            filing_date="2023-12-31",
            processing_time_ms=1000,
            model="gpt-4",
            prompt_version="v1.0",
            cached=False,
            timestamp=datetime.utcnow(),
        ),
    )
