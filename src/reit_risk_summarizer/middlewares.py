"""Custom middleware for logging, error handling, and monitoring."""

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from .exceptions import (
    InvalidTickerError,
    LLMSummarizationError,
    REITRiskSummarizerError,
    RiskExtractionError,
    SECFetchError,
)

logger = logging.getLogger(__name__)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log all requests and responses with timing."""
    start_time = time.time()

    # Log request
    logger.info(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)

    # Log response
    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    # Add custom header with processing time
    response.headers["X-Process-Time"] = str(duration_ms)

    return response


async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
    """Handle exceptions and convert to proper API responses."""
    try:
        return await call_next(request)
    except InvalidTickerError as e:
        logger.warning(f"Invalid ticker: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "InvalidTickerError",
                "message": e.message,
                "details": e.details,
            },
        )
    except SECFetchError as e:
        logger.error(f"SEC fetch failed: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "SECFetchError",
                "message": "Failed to fetch SEC filing data",
                "details": e.details,
            },
        )
    except RiskExtractionError as e:
        logger.error(f"Risk extraction failed: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "RiskExtractionError",
                "message": "Failed to extract risks from filing",
                "details": e.details,
            },
        )
    except LLMSummarizationError as e:
        logger.error(f"LLM summarization failed: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "LLMSummarizationError",
                "message": "Failed to generate risk summary",
                "details": e.details,
            },
        )
    except REITRiskSummarizerError as e:
        logger.error(f"Application error: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": type(e).__name__,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.exception("Unexpected error occurred")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": {"error_type": type(e).__name__},
            },
        )
