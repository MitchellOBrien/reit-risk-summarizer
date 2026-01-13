"""Unit tests for evaluation metrics."""

import pytest
import numpy as np

from evaluation.metrics import (
    calculate_ndcg_at_k,
    calculate_sector_specificity,
    evaluate_summary,
    get_embedding_model,
    SimilarityMetrics,
)


class TestSimilarityMetrics:
    """Test semantic similarity calculations."""

    def test_identical_risks_perfect_score(self):
        """Identical risks should have similarity near 1.0."""
        sim = SimilarityMetrics()
        risks = ["Interest rate risk affecting debt refinancing costs"]

        score = sim.calculate_similarity(risks, risks)

        # Should be very close to 1.0 (perfect match)
        assert score > 0.99

    def test_similar_risks_high_score(self):
        """Semantically similar risks should score high."""
        sim = SimilarityMetrics()
        predicted = ["Rising interest rates increase refinancing costs"]
        golden = ["Higher rates impact debt servicing expenses"]

        score = sim.calculate_similarity(predicted, golden)

        # Different words but similar meaning
        assert score > 0.50

    def test_dissimilar_risks_low_score(self):
        """Unrelated risks should score low."""
        sim = SimilarityMetrics()
        predicted = ["Cybersecurity threats to data centers"]
        golden = ["Agricultural commodity price volatility"]

        score = sim.calculate_similarity(predicted, golden)

        # Completely different topics
        assert score < 0.30

    def test_empty_lists_return_zero(self):
        """Empty risk lists should return 0.0."""
        sim = SimilarityMetrics()

        assert sim.calculate_similarity([], ["risk"]) == 0.0
        assert sim.calculate_similarity(["risk"], []) == 0.0
        assert sim.calculate_similarity([], []) == 0.0


class TestNDCGCalculation:
    """Test NDCG@5 ranking quality metric."""

    def test_perfect_ranking(self):
        """Identical ranking should score 1.0."""
        risks = [
            "Risk 1: Interest rate exposure",
            "Risk 2: Tenant concentration",
            "Risk 3: Supply chain disruption",
            "Risk 4: Regulatory changes",
            "Risk 5: Market competition",
        ]

        ndcg = calculate_ndcg_at_k(risks, risks, k=5)

        # Perfect ranking
        assert ndcg > 0.99

    def test_similar_ranking_high_score(self):
        """Slightly different ranking should still score high."""
        predicted = [
            "Rising interest rates affect debt costs",
            "Major tenant represents 40% of revenue",
            "Supply chain issues delay projects",
            "New regulations increase compliance",
            "Competitive pressure on pricing",
        ]
        golden = [
            "Interest rate risk on refinancing",
            "Tenant concentration creates dependency",
            "Supply disruptions impact timelines",
            "Regulatory environment uncertainty",
            "Market competition affects margins",
        ]

        ndcg = calculate_ndcg_at_k(predicted, golden, k=5)

        # Similar content, good ranking
        assert ndcg > 0.70

    def test_empty_lists_return_zero(self):
        """Empty risk lists should return 0.0."""
        assert calculate_ndcg_at_k([], ["risk"], k=5) == 0.0
        assert calculate_ndcg_at_k(["risk"], [], k=5) == 0.0

    def test_truncates_to_k(self):
        """Should only evaluate top k risks."""
        predicted = [f"Risk {i}" for i in range(10)]
        golden = [f"Risk {i}" for i in range(10)]

        # Should only evaluate first 3
        ndcg = calculate_ndcg_at_k(predicted, golden, k=3)

        assert ndcg > 0.99


class TestSectorSpecificity:
    """Test sector-specificity score calculation."""

    def test_sector_specific_risk(self):
        """Risk specific to a sector should score high."""
        risk = "5G infrastructure deployment requires significant tower upgrades"
        sector = "Infrastructure/Towers"
        all_sectors = {
            "Infrastructure/Towers": [
                "Tower maintenance costs increasing",
                "Carrier concentration creates revenue risk",
                "5G equipment requires capital investment",
            ],
            "Healthcare": [
                "Medicare reimbursement changes",
                "Occupancy rates in senior housing",
                "Regulatory compliance costs",
            ],
        }

        score = calculate_sector_specificity(risk, sector, all_sectors)

        # Should be more similar to tower sector
        assert score > 0.50

    def test_generic_risk(self):
        """Generic risk should score lower."""
        risk = "Economic recession may impact business performance"
        sector = "Industrial/Logistics"
        all_sectors = {
            "Industrial/Logistics": [
                "E-commerce growth drives warehouse demand",
                "Supply chain efficiency critical",
                "Last-mile logistics competition",
            ],
            "Retail": [
                "Foot traffic declining in malls",
                "Anchor tenant bankruptcies",
                "Online shopping displacing stores",
            ],
        }

        score = calculate_sector_specificity(risk, sector, all_sectors)

        # Generic economic risk, not sector-specific
        assert score < 0.60

    def test_invalid_sector_returns_zero(self):
        """Unknown sector should return 0.0."""
        risk = "Some risk"
        all_sectors = {"Healthcare": ["Risk 1", "Risk 2"]}

        score = calculate_sector_specificity(risk, "Unknown", all_sectors)

        assert score == 0.0

    def test_empty_risk_returns_zero(self):
        """Empty risk text should return 0.0."""
        all_sectors = {"Healthcare": ["Risk 1", "Risk 2"]}

        score = calculate_sector_specificity("", "Healthcare", all_sectors)

        assert score == 0.0


class TestEvaluateSummary:
    """Test complete summary evaluation."""

    def test_returns_all_three_metrics(self):
        """Should return all three metric scores."""
        generated = [
            "Interest rate risk",
            "Tenant concentration",
            "Supply chain issues",
            "Regulatory changes",
            "Market competition",
        ]
        golden = [
            "Rising rates impact debt",
            "Major tenant dependency",
            "Supply disruptions",
            "New regulations",
            "Competitive pressure",
        ]
        all_sectors = {
            "Industrial/Logistics": golden,
            "Healthcare": ["Medical risk", "Hospital dependency"],
        }

        metrics = evaluate_summary(
            generated, golden, "Industrial/Logistics", all_sectors
        )

        # Should have all three keys
        assert "semantic_similarity" in metrics
        assert "ndcg_at_5" in metrics
        assert "sector_specificity" in metrics

        # All should be between 0 and 1
        for metric_name, score in metrics.items():
            assert 0.0 <= score <= 1.0, f"{metric_name} = {score} out of range"

    def test_perfect_summary_high_scores(self):
        """Identical generated and golden should score very high."""
        risks = [
            "Interest rate exposure on floating debt",
            "Major tenant represents 50% revenue",
            "Supply chain delays impact projects",
            "New zoning regulations restrict development",
            "Increased competition from new entrants",
        ]
        all_sectors = {
            "Industrial/Logistics": risks,
            "Healthcare": ["Different risk 1", "Different risk 2"],
        }

        metrics = evaluate_summary(
            risks, risks, "Industrial/Logistics", all_sectors
        )

        # Perfect match should score very high
        assert metrics["semantic_similarity"] > 0.99
        assert metrics["ndcg_at_5"] > 0.99
        # Sector specificity depends on comparison, but should be reasonable
        assert metrics["sector_specificity"] > 0.30


class TestEmbeddingModelCache:
    """Test global model caching."""

    def test_model_caching(self):
        """Model should be loaded once and cached."""
        # Clear cache if exists (for test isolation)
        import evaluation.metrics

        evaluation.metrics._EMBEDDING_MODEL = None

        # First call loads model
        model1 = get_embedding_model()
        assert model1 is not None

        # Second call returns same instance
        model2 = get_embedding_model()
        assert model2 is model1

    def test_model_is_sentence_transformer(self):
        """Cached model should be a SentenceTransformer."""
        from sentence_transformers import SentenceTransformer

        model = get_embedding_model()
        assert isinstance(model, SentenceTransformer)
