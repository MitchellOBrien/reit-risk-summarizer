#!/usr/bin/env python3
"""Generate human-readable evaluation report from JSON results.

This script reads evaluation_results.json and creates a formatted text report
for easy viewing in GitHub Actions artifacts and PR comments.

Usage:
    python scripts/generate_evaluation_report.py
"""

import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def generate_report(results_path: Path, output_path: Path) -> None:
    """Generate human-readable report from evaluation results JSON.
    
    Args:
        results_path: Path to evaluation_results.json
        output_path: Path to write the text report
    """
    if not results_path.exists():
        logging.error(f"No evaluation results found at {results_path}")
        sys.exit(1)
    
    with open(results_path) as f:
        data = json.load(f)
    
    lines = []
    lines.append("# Evaluation Results")
    lines.append(f"Run Date: {data['metadata']['run_date']}")
    lines.append(f"Tickers Evaluated: {len(data['tickers'])}/{data['metadata']['total_tickers']}")
    
    if 'summary_metrics' in data:
        sm = data['summary_metrics']
        lines.append("\n## Aggregate Metrics")
        lines.append(f"Semantic Similarity: {sm['semantic_similarity']['mean']:.3f}")
        lines.append(f"NDCG@5: {sm['ndcg_at_5']['mean']:.3f}")
        lines.append(f"Sector-Specificity: {sm['sector_specificity']['mean']:.3f}")
    
    lines.append("\n## Per-Ticker Results")
    for ticker in data['tickers'][:5]:  # Show first 5
        if 'metrics' in ticker:
            m = ticker['metrics']
            lines.append(
                f"{ticker['ticker']}: "
                f"sim={m['semantic_similarity']:.3f}, "
                f"ndcg={m['ndcg_at_5']:.3f}, "
                f"spec={m['sector_specificity']:.3f}"
            )
    
    report_text = "\n".join(lines)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logging.info(f"Report generated successfully at {output_path}")
    
    # Also print report content to stdout for logs
    print(report_text)


def main() -> None:
    """Main entry point."""
    # Default paths relative to repo root
    repo_root = Path(__file__).parent.parent
    results_path = repo_root / "evaluation" / "results" / "evaluation_results.json"
    output_path = repo_root / "evaluation" / "results" / "evaluation_report.txt"
    
    generate_report(results_path, output_path)


if __name__ == "__main__":
    main()
