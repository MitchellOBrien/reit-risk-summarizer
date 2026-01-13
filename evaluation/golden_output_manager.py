"""Manager for golden LLM outputs cache.

Handles saving and loading LLM-generated risk summaries to avoid
repeated API calls during evaluation and testing.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from reit_risk_summarizer.services.llm.summarizer import RiskSummary

logger = logging.getLogger(__name__)


class GoldenOutputManager:
    """Manages persistent cache of LLM outputs for evaluation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize golden output manager.
        
        Args:
            cache_dir: Directory to store golden outputs. 
                      Defaults to evaluation/golden_outputs/
        """
        if cache_dir is None:
            # Default to golden_outputs in evaluation folder
            cache_dir = Path(__file__).parent / "golden_outputs"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Golden output cache directory: {self.cache_dir}")
    
    def get_cache_path(self, ticker: str) -> Path:
        """Get path to cache file for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Path to JSON cache file
        """
        return self.cache_dir / f"{ticker.upper()}.json"
    
    def has_cached_output(self, ticker: str) -> bool:
        """Check if cached output exists for ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if cached output exists
        """
        return self.get_cache_path(ticker).exists()
    
    def load_cached_output(self, ticker: str) -> Optional[RiskSummary]:
        """Load cached LLM output for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            RiskSummary if cached output exists, None otherwise
        """
        cache_path = self.get_cache_path(ticker)
        
        if not cache_path.exists():
            logger.debug(f"No cached output for {ticker}")
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded cached output for {ticker} (generated at {data.get('generated_at')})")
            
            # Convert back to RiskSummary object
            return RiskSummary(
                ticker=data["ticker"],
                company_name=data["company_name"],
                risks=data["risks"],
                model=data["model"],
                prompt_version=data["prompt_version"]
            )
        
        except Exception as e:
            logger.error(f"Failed to load cached output for {ticker}: {e}")
            return None
    
    def save_output(
        self, 
        summary: RiskSummary,
        input_text_length: Optional[int] = None,
        cache_hit: bool = False
    ) -> None:
        """Save LLM output to cache.
        
        Args:
            summary: RiskSummary to cache
            input_text_length: Length of input text (for reference)
            cache_hit: Whether this was from SEC cache
        """
        cache_path = self.get_cache_path(summary.ticker)
        
        data = {
            "ticker": summary.ticker,
            "company_name": summary.company_name,
            "risks": summary.risks,
            "model": summary.model,
            "prompt_version": summary.prompt_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_text_length": input_text_length,
            "cache_hit": cache_hit
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved golden output for {summary.ticker} to {cache_path}")
        
        except Exception as e:
            logger.error(f"Failed to save golden output for {summary.ticker}: {e}")
    
    def delete_output(self, ticker: str) -> bool:
        """Delete cached output for a ticker.
        
        Useful for regenerating outputs.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        cache_path = self.get_cache_path(ticker)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Deleted cached output for {ticker}")
            return True
        
        logger.debug(f"No cached output to delete for {ticker}")
        return False
    
    def list_cached_tickers(self) -> list[str]:
        """Get list of all tickers with cached outputs.
        
        Returns:
            List of ticker symbols
        """
        return [
            path.stem 
            for path in self.cache_dir.glob("*.json")
        ]
    
    def clear_all(self) -> int:
        """Delete all cached outputs.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cached outputs")
        return count
