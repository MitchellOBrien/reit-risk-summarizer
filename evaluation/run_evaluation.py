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

import pandas as pd

from reit_risk_summarizer.services.orchestrator import RiskOrchestrator
from evaluation.metrics import evaluate_summary
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
                     Defaults to golden_dataset.csv in evaluation folder.
    
    Returns:
        List of dicts with ticker, company_name, sector, and expert risks
    """
    if dataset_path is None:
        # Default to evaluation/golden_dataset.csv
        dataset_path = Path(__file__).parent / "golden_dataset.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
    
    # Read CSV and group by ticker
    df = pd.read_csv(dataset_path)
    
    tickers = []
    for ticker_name in df['ticker'].unique():
        # Get all rows for this ticker and sort by risk_rank
        ticker_data = df[df['ticker'] == ticker_name].sort_values('risk_rank')
        
        # Extract risk descriptions in rank order
        expert_risks = ticker_data['risk_description'].tolist()
        
        tickers.append({
            "ticker": ticker_name,
            "company_name": ticker_data['company_name'].iloc[0],
            "sector": ticker_data['sector'].iloc[0],
            "expert_risks": expert_risks
        })
    
    logger.info(f"Loaded {len(tickers)} tickers from golden dataset")
    return tickers


def process_ticker(
    ticker: str,
    orchestrator: RiskOrchestrator,
    golden_manager: GoldenOutputManager,
    use_cached: bool = True,
    regenerate: bool = False,
    cached_only: bool = False
) -> Optional[dict]:
    """Process a single ticker and return results.
    
    Args:
        ticker: Stock ticker symbol
        orchestrator: RiskOrchestrator instance
        golden_manager: GoldenOutputManager instance
        use_cached: If True, use cached golden output if available
        regenerate: If True, ignore cache and regenerate
        cached_only: If True, fail if no cached output exists (don't call API)
    
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
        elif cached_only:
            logger.error(f"✗ No cached output found for {ticker} (cached-only mode)")
            return None
    
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
    
    # Build sector-to-risks mapping for sector-specificity metric
    golden_dataset_path = Path(__file__).parent / "golden_dataset.csv"
    df = pd.read_csv(golden_dataset_path)
    all_sectors_risks = {}
    for sector in df['sector'].unique():
        sector_risks = df[df['sector'] == sector]['risk_description'].tolist()
        all_sectors_risks[sector] = sector_risks
    logger.info(f"Loaded {len(all_sectors_risks)} sectors for specificity scoring")
    
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
            use_cached=True,
            regenerate=args.regenerate,
            cached_only=args.cached_only
        )
        
        if result:
            # Add expert risks for comparison
            result["expert_risks"] = item["expert_risks"]
            result["sector"] = item["sector"]
            
            # Calculate Phase 2 metrics
            try:
                metrics = evaluate_summary(
                    generated_risks=result["risks"],
                    golden_risks=item["expert_risks"],
                    sector=item["sector"],
                    all_sectors_risks=all_sectors_risks
                )
                result["metrics"] = metrics
                logger.info(
                    f"Metrics - Similarity: {metrics['semantic_similarity']:.3f}, "
                    f"NDCG: {metrics['ndcg_at_5']:.3f}, "
                    f"Specificity: {metrics['sector_specificity']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {ticker}: {e}")
                result["metrics"] = None
            
            results["tickers"].append(result)
            
            success_count += 1
            if result["source"] == "cached_golden_output":
                cached_count += 1
        else:
            failure_count += 1
    
    # Calculate aggregate metrics
    if success_count > 0:
        ticker_results = results["tickers"]
        valid_metrics = [t["metrics"] for t in ticker_results if t.get("metrics")]
        
        if valid_metrics:
            results["summary_metrics"] = {
                "semantic_similarity": {
                    "mean": sum(m["semantic_similarity"] for m in valid_metrics) / len(valid_metrics),
                    "min": min(m["semantic_similarity"] for m in valid_metrics),
                    "max": max(m["semantic_similarity"] for m in valid_metrics),
                },
                "ndcg_at_5": {
                    "mean": sum(m["ndcg_at_5"] for m in valid_metrics) / len(valid_metrics),
                    "min": min(m["ndcg_at_5"] for m in valid_metrics),
                    "max": max(m["ndcg_at_5"] for m in valid_metrics),
                },
                "sector_specificity": {
                    "mean": sum(m["sector_specificity"] for m in valid_metrics) / len(valid_metrics),
                    "min": min(m["sector_specificity"] for m in valid_metrics),
                    "max": max(m["sector_specificity"] for m in valid_metrics),
                },
            }
    
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
    
    # Show aggregate metrics if available
    if "summary_metrics" in results:
        logger.info(f"\nAGGREGATE METRICS (n={len(valid_metrics)}):")
        sm = results["summary_metrics"]
        logger.info(f"  Semantic Similarity: {sm['semantic_similarity']['mean']:.3f} (min={sm['semantic_similarity']['min']:.3f}, max={sm['semantic_similarity']['max']:.3f})")
        logger.info(f"  NDCG@5:             {sm['ndcg_at_5']['mean']:.3f} (min={sm['ndcg_at_5']['min']:.3f}, max={sm['ndcg_at_5']['max']:.3f})")
        logger.info(f"  Sector-Specificity: {sm['sector_specificity']['mean']:.3f} (min={sm['sector_specificity']['min']:.3f}, max={sm['sector_specificity']['max']:.3f})")
    
    logger.info(f"\nResults saved to: {args.output}")
    logger.info(f"{'='*60}\n")
    
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
