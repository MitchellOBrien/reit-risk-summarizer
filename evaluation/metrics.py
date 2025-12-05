"""Evaluation metrics for REIT risk summarization quality.

This module provides metrics to assess the quality of LLM-generated risk summaries:
- Semantic similarity: How well does the summary capture the golden dataset content?
- NDCG@5: Are the top 5 risks ranked correctly?
- Sector-specificity: Does the summary capture sector-specific risks?
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityMetrics:
    """Calculate semantic similarity between generated and golden summaries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model.

        Args:
            model_name: HuggingFace model for sentence embeddings
        """
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, generated_risks: list[str], golden_risks: list[str]) -> float:
        """Calculate average cosine similarity between generated and golden risks.

        Args:
            generated_risks: List of risk descriptions from LLM
            golden_risks: List of risk descriptions from golden dataset

        Returns:
            Average cosine similarity score (0-1, higher is better)
        """
        if not generated_risks or not golden_risks:
            return 0.0

        # Get embeddings
        generated_embeddings = self.model.encode(generated_risks)
        golden_embeddings = self.model.encode(golden_risks)

        # Calculate pairwise similarities
        similarities = cosine_similarity(generated_embeddings, golden_embeddings)

        # Return average of max similarity for each generated risk
        max_similarities = similarities.max(axis=1)
        return float(np.mean(max_similarities))


def calculate_ndcg_at_k(ranked_risks: list[str], golden_risks: list[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG measures how well the ranking matches the ideal ranking.
    Perfect ranking = 1.0, random ranking ≈ 0.5, worst ranking = 0.0

    Args:
        ranked_risks: LLM-generated risks in ranked order
        golden_risks: Golden dataset risks in ideal order
        k: Number of top results to evaluate (default: 5)

    Returns:
        NDCG@K score (0-1, higher is better)
    """
    if not ranked_risks or not golden_risks:
        return 0.0

    # Truncate to top k
    ranked_risks = ranked_risks[:k]
    golden_risks = golden_risks[:k]

    # Calculate relevance scores (1 if in golden set, 0 otherwise)
    relevance_scores = [1.0 if risk in golden_risks else 0.0 for risk in ranked_risks]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = sum(
        rel / np.log2(idx + 2)  # +2 because idx starts at 0
        for idx, rel in enumerate(relevance_scores)
    )

    # Calculate IDCG (Ideal DCG) - perfect ranking
    ideal_relevance = [1.0] * min(len(ranked_risks), len(golden_risks))
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_sector_specificity(
    risks: list[str], sector: str, sector_keywords: dict[str, list[str]]
) -> float:
    """Calculate how sector-specific the identified risks are.

    Args:
        risks: List of risk descriptions
        sector: REIT sector (e.g., "industrial", "healthcare", "retail")
        sector_keywords: Mapping of sector -> list of sector-specific keywords

    Returns:
        Sector specificity score (0-1, higher means more sector-specific)
    """
    if not risks or sector not in sector_keywords:
        return 0.0

    keywords = sector_keywords.get(sector, [])
    if not keywords:
        return 0.0

    # Count how many risks mention sector-specific keywords
    sector_specific_count = 0
    for risk in risks:
        risk_lower = risk.lower()
        if any(keyword.lower() in risk_lower for keyword in keywords):
            sector_specific_count += 1

    return sector_specific_count / len(risks)


# Default sector keywords for REIT analysis
DEFAULT_SECTOR_KEYWORDS = {
    "industrial": [
        "logistics",
        "warehouse",
        "distribution",
        "e-commerce",
        "supply chain",
        "tenant demand",
        "lease renewal",
        "occupancy",
    ],
    "healthcare": [
        "medical",
        "hospital",
        "senior housing",
        "skilled nursing",
        "healthcare operators",
        "regulatory",
        "reimbursement",
        "Medicare",
    ],
    "retail": [
        "retail",
        "shopping center",
        "mall",
        "anchor tenant",
        "consumer spending",
        "online competition",
        "foot traffic",
    ],
    "residential": [
        "apartment",
        "multifamily",
        "rental",
        "tenant",
        "occupancy",
        "rent growth",
        "housing market",
        "residential property",
    ],
    "office": [
        "office",
        "tenant",
        "lease",
        "occupancy",
        "workplace",
        "remote work",
        "commercial real estate",
        "corporate demand",
    ],
    "data_center": [
        "data center",
        "cloud",
        "connectivity",
        "power",
        "cooling",
        "hyperscale",
        "colocation",
        "bandwidth",
        "uptime",
    ],
}


def evaluate_summary(
    generated_risks: list[str],
    golden_risks: list[str],
    sector: str,
    similarity_model: SentenceTransformer | None = None,
) -> dict[str, float]:
    """Comprehensive evaluation of a risk summary.

    Args:
        generated_risks: LLM-generated risks in ranked order
        golden_risks: Golden dataset risks in ideal order
        sector: REIT sector for specificity scoring
        similarity_model: Pre-loaded model (optional, for performance)

    Returns:
        Dictionary with all metric scores
    """
    # Initialize similarity calculator
    if similarity_model is None:
        sim_calc = SimilarityMetrics()
    else:
        sim_calc = SimilarityMetrics.__new__(SimilarityMetrics)
        sim_calc.model = similarity_model

    return {
        "semantic_similarity": sim_calc.calculate_similarity(generated_risks, golden_risks),
        "ndcg_at_5": calculate_ndcg_at_k(generated_risks, golden_risks, k=5),
        "sector_specificity": calculate_sector_specificity(
            generated_risks, sector, DEFAULT_SECTOR_KEYWORDS
        ),
    }
