"""Evaluation metrics for REIT risk summarization quality.

This module provides metrics to assess the quality of LLM-generated risk summaries:
- Semantic similarity: How well does the summary capture the golden dataset content?
- NDCG@5: Are the top 5 risks ranked correctly?
- Sector-specificity: Does the summary capture sector-specific risks?

Targets:
- Semantic Similarity: >0.75
- NDCG@5: >0.70
- Sector-Specificity: >0.40
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the sentence embedding model.
    
    Uses all-MiniLM-L6-v2 for fast, high-quality embeddings.
    Model is cached globally to avoid repeated loading.
    
    Returns:
        Loaded SentenceTransformer model
    """
    global _EMBEDDING_MODEL
    
    if _EMBEDDING_MODEL is None:
        logger.info("Loading sentence transformer model: all-MiniLM-L6-v2")
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    
    return _EMBEDDING_MODEL


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

    NDCG measures how well the ranking matches the ideal ranking using semantic similarity.
    Perfect ranking = 1.0, random ranking ≈ 0.5, worst ranking = 0.0

    Target: >0.70

    Algorithm:
    1. Use semantic similarity as "relevance" scores
    2. For each ranked risk, find similarity to each golden risk
    3. Assign relevance score based on best match
    4. Calculate NDCG using sklearn's implementation

    Args:
        ranked_risks: LLM-generated risks in ranked order
        golden_risks: Golden dataset risks in ideal order
        k: Number of top results to evaluate (default: 5)

    Returns:
        NDCG@K score (0-1, higher is better)
    """
    if not ranked_risks or not golden_risks:
        logger.warning("Empty risk lists provided to NDCG calculation")
        return 0.0

    # Truncate to top k
    ranked_risks = ranked_risks[:k]
    golden_risks = golden_risks[:k]

    model = get_embedding_model()

    # Generate embeddings
    ranked_embeddings = model.encode(ranked_risks, convert_to_numpy=True)
    golden_embeddings = model.encode(golden_risks, convert_to_numpy=True)

    # Calculate similarity matrix using cosine similarity
    similarities = cosine_similarity(ranked_embeddings, golden_embeddings)

    # For each ranked risk, get the max similarity as its relevance score
    relevance_scores = similarities.max(axis=1)

    # Create ideal relevance scores (best possible ranking of these items)
    # Sort the actual relevance scores in descending order to get ideal ranking
    ideal_scores = np.sort(relevance_scores)[::-1]

    # sklearn expects 2D arrays: [n_samples, n_items]
    # We have 1 sample (1 ticker evaluation)
    true_relevance = relevance_scores.reshape(1, -1)
    ideal_relevance = ideal_scores.reshape(1, -1)

    # Calculate NDCG
    ndcg = ndcg_score(ideal_relevance, true_relevance, k=k)

    logger.debug(
        f"NDCG@{k}: {ndcg:.3f} "
        f"(relevance scores: {relevance_scores.tolist()})"
    )

    return float(ndcg)


def calculate_sector_specificity(
    risk_text: str, sector: str, all_sectors_risks: dict[str, list[str]]
) -> float:
    """Calculate sector-specificity score for a risk using semantic embeddings.

    Measures whether a risk is specific to a REIT sector or generic boilerplate.

    Target: >0.40 (average across all risks)

    Algorithm:
    1. Embed the risk text
    2. Compare similarity to risks from same sector vs. other sectors
    3. Specificity = (avg same-sector similarity) - (avg other-sector similarity)
    4. Normalize to [0, 1] range

    Interpretation:
    - 0.65+ = Highly specific to sector (good)
    - 0.40-0.65 = Moderately specific (target range)
    - 0.20-0.40 = Somewhat generic
    - <0.20 = Generic boilerplate (bad)

    Args:
        risk_text: The risk description to evaluate
        sector: The REIT sector (e.g., "Industrial/Logistics", "Infrastructure/Towers")
        all_sectors_risks: Dict mapping sector names to lists of risk texts
                          Example: {"Industrial/Logistics": [...], "Healthcare": [...]}

    Returns:
        Specificity score between 0.0 (generic) and 1.0 (highly specific)
    """
    if not risk_text or sector not in all_sectors_risks:
        logger.warning(
            f"Invalid inputs to sector specificity: "
            f"risk_text={bool(risk_text)}, sector={sector}"
        )
        return 0.0

    model = get_embedding_model()

    # Embed the risk
    risk_embedding = model.encode([risk_text], convert_to_numpy=True)

    # Get same-sector risks and other-sector risks
    same_sector_risks = all_sectors_risks[sector]
    other_sectors_risks = [
        risk
        for other_sector, risks in all_sectors_risks.items()
        if other_sector != sector
        for risk in risks
    ]

    if not same_sector_risks or not other_sectors_risks:
        logger.warning(
            f"Insufficient data for sector specificity: "
            f"same={len(same_sector_risks)}, other={len(other_sectors_risks)}"
        )
        return 0.0

    # Embed sector risks
    same_sector_embeddings = model.encode(same_sector_risks, convert_to_numpy=True)
    other_sectors_embeddings = model.encode(other_sectors_risks, convert_to_numpy=True)

    # Calculate average similarity to same sector vs. other sectors
    same_sector_similarities = cosine_similarity(risk_embedding, same_sector_embeddings)
    other_sectors_similarities = cosine_similarity(
        risk_embedding, other_sectors_embeddings
    )

    avg_same = float(np.mean(same_sector_similarities))
    avg_other = float(np.mean(other_sectors_similarities))

    # Specificity = how much more similar to same sector vs. others
    # Raw difference is typically in [-1, 1] range
    # We normalize to [0, 1] by adding 1 and dividing by 2
    raw_specificity = avg_same - avg_other
    normalized_specificity = (raw_specificity + 1) / 2

    logger.debug(
        f"Sector specificity for {sector}: {normalized_specificity:.3f} "
        f"(same={avg_same:.3f}, other={avg_other:.3f})"
    )

    return normalized_specificity


def evaluate_summary(
    generated_risks: list[str],
    golden_risks: list[str],
    sector: str,
    all_sectors_risks: dict[str, list[str]],
) -> dict[str, float]:
    """Comprehensive evaluation of a risk summary.

    Calculates all three core metrics:
    - Semantic Similarity (target >0.75)
    - NDCG@5 (target >0.70)
    - Sector-Specificity (target >0.40)

    Args:
        generated_risks: LLM-generated risks in ranked order (5 risks)
        golden_risks: Golden dataset risks in ideal order (5 risks)
        sector: REIT sector for specificity scoring
        all_sectors_risks: Dict mapping all sectors to their risks for specificity

    Returns:
        Dictionary with all metric scores:
        {
            "semantic_similarity": 0.82,
            "ndcg_at_5": 0.78,
            "sector_specificity": 0.65
        }
    """
    # Initialize similarity calculator
    sim_calc = SimilarityMetrics()

    # Calculate per-risk sector specificity, then average
    sector_specificities = [
        calculate_sector_specificity(risk, sector, all_sectors_risks)
        for risk in generated_risks
    ]
    avg_sector_specificity = (
        float(np.mean(sector_specificities)) if sector_specificities else 0.0
    )

    return {
        "semantic_similarity": sim_calc.calculate_similarity(
            generated_risks, golden_risks
        ),
        "ndcg_at_5": calculate_ndcg_at_k(generated_risks, golden_risks, k=5),
        "sector_specificity": avg_sector_specificity,
    }
