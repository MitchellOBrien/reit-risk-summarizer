"""FastAPI dependency injection for shared services.

This module provides dependency injection factories for FastAPI endpoints,
ensuring that expensive resources (like the RiskOrchestrator) are created
once and reused across all API requests.

Key Components:
    - get_orchestrator(): Singleton factory for RiskOrchestrator
    - OrchestratorDep: Type alias for cleaner endpoint signatures

The orchestrator is configured with:
    - Groq as the LLM provider (llama-3.3-70b-versatile)
    - In-memory caching enabled (global singleton cache)
    - Default SEC fetcher and risk extractor

Using @lru_cache ensures the same orchestrator instance is injected into
all endpoint handlers, maintaining cache state and avoiding redundant
initialization overhead.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from .services.orchestrator import RiskOrchestrator


@lru_cache()
def get_orchestrator() -> RiskOrchestrator:
    """Get or create the shared orchestrator instance.
    
    Uses lru_cache to ensure a single orchestrator is shared across
    all requests, maintaining the global cache and connections.
    
    Returns:
        RiskOrchestrator instance with default configuration
    """
    return RiskOrchestrator(
        summarizer_provider="groq",
        summarizer_model="llama-3.3-70b-versatile",
        cache_enabled=True
    )


# Type alias for dependency injection
OrchestratorDep = Annotated[RiskOrchestrator, Depends(get_orchestrator)]
