"""FastAPI middleware for request/response processing and error handling.

This module provides middleware components that intercept all incoming requests
and outgoing responses to add cross-cutting concerns like logging, timing,
and centralized exception handling.

Middleware Components:
    - logging_middleware: Logs all requests/responses with timing metrics
    - error_handling_middleware: Catches exceptions and converts to proper HTTP responses

The logging middleware captures:
    - HTTP method, path, client IP
    - Response status code
    - Request duration in milliseconds

The error handling middleware maps application exceptions to HTTP status codes:
    - SECFetchError → 404 Not Found (invalid ticker or missing filing)
    - RiskExtractionError → 500 Internal Server Error (parsing failure)
    - LLMSummarizationError → 503 Service Unavailable (LLM API issues)
    - InvalidTickerError → 400 Bad Request (malformed ticker)
    - REITRiskSummarizerError → 500 Internal Server Error (general errors)

Both middleware functions are registered in main.py using app.middleware("http").
"""

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
    """Log all incoming requests and outgoing responses with performance timing.
    
    This middleware runs for EVERY request before it reaches your endpoint and
    after the endpoint returns. It provides visibility into API usage and performance.
    
    What it does:
        1. BEFORE endpoint: Logs request details (method, path, client IP)
        2. BEFORE endpoint: Starts a timer
        3. Calls next middleware/endpoint: await call_next(request)
        4. AFTER endpoint: Calculates request duration
        5. AFTER endpoint: Logs response details (status code, duration)
        6. AFTER endpoint: Adds X-Process-Time header to response
    
    Log Output Examples:
        Request:  "Request started" {method: "GET", path: "/api/v1/risks/AMT", client: "127.0.0.1"}
        Response: "Request completed" {method: "GET", path: "/api/v1/risks/AMT", status_code: 200, duration_ms: 3500}
    
    Response Headers Added:
        X-Process-Time: "3500" (milliseconds spent processing the request)
    
    Args:
        request: The incoming HTTP request object containing method, URL, headers, etc.
        call_next: Async function that passes request to next middleware/endpoint.
                   MUST be awaited to get the response. This is why the function
                   must be async - we need to await call_next().
    
    Returns:
        Response: The HTTP response from the endpoint, with added X-Process-Time header.
    
    Side Effects:
        - Writes two INFO log entries per request (start and completion)
        - Modifies response by adding X-Process-Time header
    
    Note:
        This middleware is registered in main.py with app.middleware("http").
        It runs for ALL endpoints automatically - no need to add decorators to
        individual endpoint functions.
    """
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
    """Catch exceptions from endpoints and convert to proper HTTP error responses.
    
    This middleware provides centralized exception handling for the entire API,
    ensuring consistent error responses and preventing endpoints from needing
    repetitive try/except blocks.
    
    What it does:
        1. Wraps the entire request in a try/except block
        2. Calls next middleware/endpoint: await call_next(request)
        3. If no exception: Returns response normally
        4. If exception raised: Catches it and converts to appropriate HTTP error
    
    Exception → HTTP Status Code Mapping:
        InvalidTickerError      → 400 Bad Request       (malformed/invalid ticker symbol)
        SECFetchError          → 502 Bad Gateway        (SEC EDGAR service unavailable/ticker not found)
        RiskExtractionError    → 500 Internal Error     (failed to parse Item 1A from 10-K)
        LLMSummarizationError  → 500 Internal Error     (Groq API call failed)
        REITRiskSummarizerError → 500 Internal Error    (general application errors)
        Exception (any other)   → 500 Internal Error    (unexpected errors, logs full traceback)
    
    Error Response Format:
        {
            "error": "SECFetchError",                    # Exception class name
            "message": "Could not find 10-K for INVALID", # Exception's message field
            "details": {"ticker": "INVALID"}              # Exception's details dict
        }
    
    Why This Matters:
        - Consistent error format across all endpoints
        - Correct HTTP status codes (clients can handle different errors appropriately)
        - Endpoints stay clean (no try/except needed in endpoint functions)
        - Security (hides internal tracebacks from generic exceptions)
        - All errors logged for debugging
    
    Args:
        request: The incoming HTTP request object.
        call_next: Async function that passes request to next middleware/endpoint.
                   This is where exceptions are raised from your endpoint code.
    
    Returns:
        Response: Either the successful response from the endpoint, or a JSONResponse
                  containing error details if an exception was caught.
    
    Side Effects:
        - Logs warnings for InvalidTickerError (user error, not system error)
        - Logs errors for all other custom exceptions (system/integration errors)
        - Logs exception with full traceback for unexpected exceptions
    
    Example Flow:
        Request → error_handling_middleware → logging_middleware → endpoint
                                               ↓
                                    endpoint raises SECFetchError
                                               ↓
                                    Caught by error_handling_middleware
                                               ↓
                                    Converts to JSONResponse(status=502)
                                               ↓
        Response (502 Bad Gateway with error details)
    
    Note:
        This middleware should be registered AFTER logging_middleware in main.py
        so that errors are still logged. Order matters - middleware runs in
        reverse order on the way back out.
    """
    try:
        return await call_next(request)
    except InvalidTickerError as e:
        logger.warning(f"Invalid ticker: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": type(e).__name__,
                "message": e.message,
                "details": e.details,
            },
        )
    except SECFetchError as e:
        logger.error(f"SEC fetch failed: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": type(e).__name__,
                "message": e.message,
                "details": e.details,
            },
        )
    except RiskExtractionError as e:
        logger.error(f"Risk extraction failed: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": type(e).__name__,
                "message": e.message,
                "details": e.details,
            },
        )
    except LLMSummarizationError as e:
        logger.error(f"LLM summarization failed: {e.message}", extra={"details": e.details})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": type(e).__name__,
                "message": e.message,
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
