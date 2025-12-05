"""Request schemas for API endpoints."""

from pydantic import BaseModel, Field, field_validator


class SummarizeRiskRequest(BaseModel):
    """Request to summarize risks for a REIT ticker."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., PLD, AMT)")
    force_refresh: bool = Field(
        default=False,
        description="Force refresh even if cached result exists"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        ticker = v.strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty")
        if len(ticker) > 10:
            raise ValueError("Ticker too long")
        if not ticker.isalnum():
            raise ValueError("Ticker must be alphanumeric")
        return ticker
