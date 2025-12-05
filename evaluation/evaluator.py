"""Orchestrate evaluation of risk summaries against golden dataset.

This module handles:
- Loading golden dataset
- Running risk summarization pipeline
- Computing evaluation metrics
- Aggregating results across multiple REITs
"""

import csv
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer

from evaluation.metrics import evaluate_summary


@dataclass
class GoldenRecord:
    """Single record from the golden dataset."""
    ticker: str
    company_name: str
    sector: str
    filing_date: str
    risk_1: str
    risk_2: str
    risk_3: str
    risk_4: str
    risk_5: str
    
    @property
    def risks(self) -> List[str]:
        """Return risks as a list."""
        return [
            self.risk_1,
            self.risk_2,
            self.risk_3,
            self.risk_4,
            self.risk_5
        ]


@dataclass
class EvaluationResult:
    """Results from evaluating a single REIT summary."""
    ticker: str
    company_name: str
    sector: str
    semantic_similarity: float
    ndcg_at_5: float
    sector_specificity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class Evaluator:
    """Orchestrate evaluation of risk summaries."""
    
    def __init__(self, golden_dataset_path: str | Path):
        """Initialize evaluator with golden dataset.
        
        Args:
            golden_dataset_path: Path to golden_dataset.csv
        """
        self.golden_dataset_path = Path(golden_dataset_path)
        self.golden_records: Dict[str, GoldenRecord] = {}
        self._load_golden_dataset()
        
        # Pre-load embedding model for performance
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def _load_golden_dataset(self) -> None:
        """Load golden dataset from CSV."""
        if not self.golden_dataset_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {self.golden_dataset_path}"
            )
        
        with open(self.golden_dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = GoldenRecord(
                    ticker=row['ticker'],
                    company_name=row['company_name'],
                    sector=row['sector'],
                    filing_date=row['filing_date'],
                    risk_1=row['risk_1'],
                    risk_2=row['risk_2'],
                    risk_3=row['risk_3'],
                    risk_4=row['risk_4'],
                    risk_5=row['risk_5']
                )
                self.golden_records[record.ticker] = record
    
    def evaluate_single(
        self,
        ticker: str,
        generated_risks: List[str]
    ) -> EvaluationResult:
        """Evaluate a single REIT's risk summary.
        
        Args:
            ticker: REIT ticker symbol
            generated_risks: List of LLM-generated risks (in ranked order)
            
        Returns:
            EvaluationResult with all metrics
            
        Raises:
            ValueError: If ticker not found in golden dataset
        """
        if ticker not in self.golden_records:
            raise ValueError(
                f"Ticker '{ticker}' not found in golden dataset. "
                f"Available tickers: {list(self.golden_records.keys())}"
            )
        
        golden = self.golden_records[ticker]
        
        # Calculate all metrics
        metrics = evaluate_summary(
            generated_risks=generated_risks,
            golden_risks=golden.risks,
            sector=golden.sector,
            similarity_model=self.embedding_model
        )
        
        return EvaluationResult(
            ticker=golden.ticker,
            company_name=golden.company_name,
            sector=golden.sector,
            semantic_similarity=metrics["semantic_similarity"],
            ndcg_at_5=metrics["ndcg_at_5"],
            sector_specificity=metrics["sector_specificity"]
        )
    
    def evaluate_batch(
        self,
        results: Dict[str, List[str]]
    ) -> List[EvaluationResult]:
        """Evaluate multiple REIT summaries.
        
        Args:
            results: Dict mapping ticker -> list of generated risks
            
        Returns:
            List of EvaluationResults
        """
        return [
            self.evaluate_single(ticker, risks)
            for ticker, risks in results.items()
        ]
    
    def get_aggregate_metrics(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluations.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary with mean metrics across all REITs
        """
        if not results:
            return {
                "mean_semantic_similarity": 0.0,
                "mean_ndcg_at_5": 0.0,
                "mean_sector_specificity": 0.0,
                "num_evaluated": 0
            }
        
        return {
            "mean_semantic_similarity": sum(
                r.semantic_similarity for r in results
            ) / len(results),
            "mean_ndcg_at_5": sum(
                r.ndcg_at_5 for r in results
            ) / len(results),
            "mean_sector_specificity": sum(
                r.sector_specificity for r in results
            ) / len(results),
            "num_evaluated": len(results)
        }
    
    def get_sector_breakdown(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by sector.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dict mapping sector -> aggregate metrics for that sector
        """
        sector_results: Dict[str, List[EvaluationResult]] = {}
        
        # Group by sector
        for result in results:
            if result.sector not in sector_results:
                sector_results[result.sector] = []
            sector_results[result.sector].append(result)
        
        # Calculate aggregate for each sector
        return {
            sector: self.get_aggregate_metrics(sector_evals)
            for sector, sector_evals in sector_results.items()
        }
    
    def list_available_tickers(self) -> List[str]:
        """Get list of all tickers in golden dataset."""
        return sorted(self.golden_records.keys())
    
    def get_golden_record(self, ticker: str) -> GoldenRecord:
        """Get golden dataset record for a ticker.
        
        Args:
            ticker: REIT ticker symbol
            
        Returns:
            GoldenRecord for the ticker
            
        Raises:
            ValueError: If ticker not found
        """
        if ticker not in self.golden_records:
            raise ValueError(
                f"Ticker '{ticker}' not found in golden dataset"
            )
        return self.golden_records[ticker]
