"""Run evaluation against golden dataset.

This script processes REITs from the golden dataset and generates
risk summaries, using cached LLM outputs when available to avoid
burning API tokens.

Usage:
    # Process all tickers in golden dataset
    python -m evaluation.run_evaluation
    
    # Process specific tickers
    python -m evaluation.run_evaluation --tickers AMT PLD
    
    # Force regenerate (ignore cached outputs)
    python -m evaluation.run_evaluation --regenerate
    
    # Use cached outputs only (fail if not cached)
    python -m evaluation.run_evaluation --cached-only
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from reit_risk_summarizer.services.orchestrator import RiskOrchestrator
from reit_risk_summarizer.exceptions import (
    SECFetchError,
    RiskExtractionError,
    LLMSummarizationError
)
from evaluation.golden_output_manager import GoldenOutputManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_golden_dataset(dataset_path: Optional[Path] = None) -> list[dict]:
    """Load golden dataset CSV.
    
    Args:
        dataset_path: Path to golden dataset CSV. 
                     Defaults to reit-risk-golden-dataset.csv in project root.
    
    Returns:
        List of dicts with ticker, company_name, sector, and expert risks
    """
    if dataset_path is None:
        # Default to project root
        dataset_path = Path(__file__).parent.parent / "reit-risk-golden-dataset.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
    
    tickers = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract the 5 expert-labeled risks
            expert_risks = [
                row.get(f"risk_{i}", "").strip()
                for i in range(1, 6)
                if row.get(f"risk_{i}", "").strip()
            ]
            
            tickers.append({
                "ticker": row["ticker"].strip(),
                "company_name": row.get("company_name", "").strip(),
                "sector": row.get("sector", "").strip(),
                "expert_risks": expert_risks
            })
    
    logger.info(f"Loaded {len(tickers)} tickers from golden dataset")
    return tickers


def process_ticker(
    ticker: str,
    orchestrator: RiskOrchestrator,
    golden_manager: GoldenOutputManager,
    use_cached: bool = True,
    regenerate: bool = False
) -> Optional[dict]:
    """Process a single ticker and return results.
    
    Args:
        ticker: Stock ticker symbol
        orchestrator: RiskOrchestrator instance
        golden_manager: GoldenOutputManager instance
        use_cached: If True, use cached golden output if available
        regenerate: If True, ignore cache and regenerate
    
    Returns:
        Dict with ticker, generated risks, and metadata, or None if failed
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {ticker}")
    logger.info(f"{'='*60}")
    
    # Check for cached golden output
    if use_cached and not regenerate:
        cached_summary = golden_manager.load_cached_output(ticker)
        if cached_summary:
            logger.info(f"✓ Using cached golden output for {ticker}")
            return {
                "ticker": ticker,
                "company_name": cached_summary.company_name,
                "risks": cached_summary.risks,
                "model": cached_summary.model,
                "prompt_version": cached_summary.prompt_version,
                "source": "cached_golden_output",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Generate new output
    try:
        logger.info(f"Generating new output for {ticker} (calling Groq API)...")
        summary = orchestrator.process_reit(ticker, force_refresh=False)
        
        # Save to golden outputs cache
        golden_manager.save_output(summary)
        
        logger.info(f"✓ Successfully processed {ticker}")
        return {
            "ticker": ticker,
            "company_name": summary.company_name,
            "risks": summary.risks,
            "model": summary.model,
            "prompt_version": summary.prompt_version,
            "source": "groq_api",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except SECFetchError as e:
        logger.error(f"✗ Failed to fetch SEC data for {ticker}: {e}")
        return None
    
    except RiskExtractionError as e:
        logger.error(f"✗ Failed to extract risks for {ticker}: {e}")
        return None
    
    except LLMSummarizationError as e:
        logger.error(f"✗ Failed to generate summary for {ticker}: {e}")
        return None
    
    except Exception as e:
        logger.error(f"✗ Unexpected error processing {ticker}: {e}", exc_info=True)
        return None


def main():
    """Run evaluation on golden dataset."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM risk summarization against golden dataset"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specific tickers to process (default: all from golden dataset)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate all outputs (ignore cached golden outputs)"
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Use only cached outputs (fail if not cached)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to golden dataset CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "evaluation_results.json",
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    # Load golden dataset
    try:
        golden_data = load_golden_dataset(args.dataset)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    # Filter to specific tickers if requested
    if args.tickers:
        ticker_set = set(t.upper() for t in args.tickers)
        golden_data = [
            item for item in golden_data 
            if item["ticker"].upper() in ticker_set
        ]
        logger.info(f"Filtered to {len(golden_data)} requested tickers")
    
    # Initialize orchestrator and golden output manager
    orchestrator = RiskOrchestrator(cache_enabled=True)
    golden_manager = GoldenOutputManager()
    
    # Show cache status
    cached_tickers = golden_manager.list_cached_tickers()
    logger.info(f"Found {len(cached_tickers)} cached golden outputs: {', '.join(cached_tickers)}")
    
    # Process each ticker
    results = {
        "metadata": {
            "run_date": datetime.utcnow().isoformat(),
            "total_tickers": len(golden_data),
            "regenerate_mode": args.regenerate,
            "cached_only_mode": args.cached_only
        },
        "tickers": []
    }
    
    success_count = 0
    failure_count = 0
    cached_count = 0
    
    for item in golden_data:
        ticker = item["ticker"]
        
        # Process ticker
        result = process_ticker(
            ticker,
            orchestrator,
            golden_manager,
            use_cached=not args.cached_only,
            regenerate=args.regenerate
        )
        
        if result:
            # Add expert risks for comparison
            result["expert_risks"] = item["expert_risks"]
            result["sector"] = item["sector"]
            results["tickers"].append(result)
            
            success_count += 1
            if result["source"] == "cached_golden_output":
                cached_count += 1
        else:
            failure_count += 1
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tickers:    {len(golden_data)}")
    logger.info(f"✓ Successful:     {success_count}")
    logger.info(f"  - From cache:   {cached_count}")
    logger.info(f"  - From Groq:    {success_count - cached_count}")
    logger.info(f"✗ Failed:         {failure_count}")
    logger.info(f"\nResults saved to: {args.output}")
    logger.info(f"{'='*60}\n")
    
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
