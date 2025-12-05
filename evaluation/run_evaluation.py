#!/usr/bin/env python3
"""CLI script to run evaluation of risk summaries against golden dataset.

Usage:
    python -m evaluation.run_evaluation --ticker PLD
    python -m evaluation.run_evaluation --all
    python -m evaluation.run_evaluation --sector industrial
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import EvaluationResult, Evaluator


def print_results(results: list[EvaluationResult], show_details: bool = True) -> None:
    """Pretty print evaluation results.

    Args:
        results: List of evaluation results
        show_details: Whether to show per-REIT details
    """
    if not results:
        print("No results to display.")
        return

    # Print header
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80 + "\n")

    # Print individual results
    if show_details:
        for result in results:
            print(f"📊 {result.ticker} - {result.company_name}")
            print(f"   Sector: {result.sector}")
            print(f"   Semantic Similarity: {result.semantic_similarity:.3f}")
            print(f"   NDCG@5: {result.ndcg_at_5:.3f}")
            print(f"   Sector Specificity: {result.sector_specificity:.3f}")
            print()

    # Calculate and print aggregate metrics
    evaluator = Evaluator(Path(__file__).parent / "golden_dataset.csv")
    aggregate = evaluator.get_aggregate_metrics(results)

    print("-" * 80)
    print("AGGREGATE METRICS")
    print("-" * 80)
    print(f"Number of REITs evaluated: {aggregate['num_evaluated']}")
    print(f"Mean Semantic Similarity: {aggregate['mean_semantic_similarity']:.3f}")
    print(f"Mean NDCG@5: {aggregate['mean_ndcg_at_5']:.3f}")
    print(f"Mean Sector Specificity: {aggregate['mean_sector_specificity']:.3f}")
    print()

    # Sector breakdown
    sector_breakdown = evaluator.get_sector_breakdown(results)
    if len(sector_breakdown) > 1:
        print("-" * 80)
        print("SECTOR BREAKDOWN")
        print("-" * 80)
        for sector, metrics in sector_breakdown.items():
            print(f"\n{sector.upper()}")
            print(f"  REITs: {metrics['num_evaluated']}")
            print(f"  Semantic Similarity: {metrics['mean_semantic_similarity']:.3f}")
            print(f"  NDCG@5: {metrics['mean_ndcg_at_5']:.3f}")
            print(f"  Sector Specificity: {metrics['mean_sector_specificity']:.3f}")
        print()


def run_evaluation_for_ticker(ticker: str) -> EvaluationResult:
    """Run evaluation for a single ticker.

    This is a placeholder - in production, this would:
    1. Fetch the 10-K filing
    2. Extract risk factors
    3. Run LLM summarization
    4. Evaluate against golden dataset

    Args:
        ticker: REIT ticker symbol

    Returns:
        EvaluationResult
    """
    # TODO: Implement full pipeline
    # For now, this is a stub that would integrate with:
    # - src/reit_risk_summarizer/services/sec/fetcher.py
    # - src/reit_risk_summarizer/services/sec/extractor.py
    # - src/reit_risk_summarizer/services/llm/summarizer.py

    raise NotImplementedError(
        "Full pipeline integration not yet implemented. "
        "This script currently serves as a template for future integration."
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate REIT risk summaries against golden dataset"
    )

    # Mutually exclusive group for ticker selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Evaluate a single ticker (e.g., PLD)")
    group.add_argument("--all", action="store_true", help="Evaluate all tickers in golden dataset")
    group.add_argument(
        "--sector", type=str, help="Evaluate all tickers in a specific sector (e.g., industrial)"
    )

    # Output options
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    # Initialize evaluator
    golden_dataset_path = Path(__file__).parent / "golden_dataset.csv"
    evaluator = Evaluator(golden_dataset_path)

    # Determine which tickers to evaluate
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.all:
        tickers = evaluator.list_available_tickers()
    else:  # args.sector
        sector_filter = args.sector.lower()
        tickers = [
            ticker
            for ticker, record in evaluator.golden_records.items()
            if record.sector.lower() == sector_filter
        ]
        if not tickers:
            print(f"❌ No tickers found for sector: {args.sector}")
            print(f"Available sectors: {set(r.sector for r in evaluator.golden_records.values())}")
            sys.exit(1)

    print(f"\n🚀 Evaluating {len(tickers)} REIT(s)...\n")

    # Run evaluations
    # NOTE: This is a placeholder - actual implementation would run full pipeline
    try:
        results = []
        for ticker in tickers:
            print(f"⏳ Processing {ticker}...")
            # In production, this would call run_evaluation_for_ticker(ticker)
            # For now, we demonstrate the evaluator with mock data

            # Mock generated risks (in real implementation, these come from LLM)
            golden = evaluator.get_golden_record(ticker)
            mock_generated_risks = golden.risks  # Perfect match for demonstration

            result = evaluator.evaluate_single(ticker, mock_generated_risks)
            results.append(result)

        # Display results
        if not args.quiet:
            print_results(results, show_details=True)

        # Save to file if requested
        if args.output:
            output_data = {
                "results": [r.to_dict() for r in results],
                "aggregate": evaluator.get_aggregate_metrics(results),
                "sector_breakdown": evaluator.get_sector_breakdown(results),
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Results saved to {args.output}")

        print("\n✅ Evaluation complete!")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
