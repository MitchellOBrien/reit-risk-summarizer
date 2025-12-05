"""Response schemas for API endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class Risk(BaseModel):
    """Individual risk summary."""

    rank: int = Field(..., ge=1, le=5, description="Risk ranking (1-5)")
    title: str = Field(..., min_length=1, description="Risk title")
    description: str = Field(..., min_length=1, description="Risk description")
    category: str = Field(..., description="Risk category")


class Metadata(BaseModel):
    """Metadata about the summarization process."""

    filing_date: str | None = Field(None, description="10-K filing date")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    model: str = Field(..., description="LLM model used")
    prompt_version: str = Field(..., description="Prompt version")
    cached: bool = Field(default=False, description="Whether result was from cache")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SummarizeRiskResponse(BaseModel):
    """Response containing risk summary for a REIT."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str | None = Field(None, description="Company name")
    risks: list[Risk] = Field(..., min_length=5, max_length=5, description="Top 5 risks")
    metadata: Metadata


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict = Field(default_factory=dict, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
