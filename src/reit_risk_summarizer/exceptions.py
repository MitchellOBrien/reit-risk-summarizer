"""Custom exceptions for the application."""


class REITRiskSummarizerError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class SECFetchError(REITRiskSummarizerError):
    """Raised when SEC data fetching fails."""

    pass


class RiskExtractionError(REITRiskSummarizerError):
    """Raised when risk extraction from 10-K fails."""

    pass


class LLMSummarizationError(REITRiskSummarizerError):
    """Raised when LLM summarization fails."""

    pass


class InvalidTickerError(REITRiskSummarizerError):
    """Raised when ticker symbol is invalid."""

    pass


class CacheError(REITRiskSummarizerError):
    """Raised when cache operations fail."""

    pass


class EvaluationError(REITRiskSummarizerError):
    """Raised when evaluation fails."""

    pass
