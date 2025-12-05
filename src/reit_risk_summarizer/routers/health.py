"""Health check and status endpoints."""

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..schemas.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the application.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/ping")
async def ping() -> dict[str, str]:
    """Simple ping endpoint for quick health checks."""
    return {"status": "pong"}
