"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application
    app_name: str = "REIT Risk Summarizer"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # API Keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    # SEC EDGAR
    sec_api_email: str = "your.email@example.com"
    sec_api_user_agent: str = "YourName your.email@example.com"

    # LLM Configuration
    default_llm_model: str = "huggingface/meta-llama/Llama-3.2-1B-Instruct"  # LOCAL, requires HF_TOKEN
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours
    cache_type: Literal["memory", "redis"] = "memory"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_rate_limit_per_minute: int = 10

    # Evaluation
    golden_dataset_path: str = "evaluation/golden_dataset.csv"
    evaluation_model: str = "groq/llama-3.3-70b-versatile"  # FREE tier
    evaluation_runs_per_ticker: int = 3

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
