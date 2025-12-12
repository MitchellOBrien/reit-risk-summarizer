"""Unit tests for LLM risk summarizer."""

import pytest
from unittest.mock import patch, Mock, MagicMock

from reit_risk_summarizer.exceptions import LLMSummarizationError
from reit_risk_summarizer.services.llm.summarizer import (
    GroqRiskSummarizer,
    RiskSummary,
    create_summarizer
)


class TestRiskSummary:
    """Test suite for RiskSummary dataclass."""

    def test_valid_risk_summary(self):
        """Test creating a valid RiskSummary."""
        risks = [
            "Risk 1: Market volatility",
            "Risk 2: Regulatory changes",
            "Risk 3: Competition",
            "Risk 4: Economic downturn",
            "Risk 5: Technology disruption"
        ]

        summary = RiskSummary(
            risks=risks,
            ticker="PLD",
            company_name="Prologis",
            model="llama-3.3-70b-versatile",
            prompt_version="v1.0",
            raw_response="1. Risk 1\n2. Risk 2..."
        )

        assert len(summary.risks) == 5
        assert summary.ticker == "PLD"
        assert summary.company_name == "Prologis"
        assert summary.model == "llama-3.3-70b-versatile"
        assert summary.prompt_version == "v1.0"

    def test_wrong_number_of_risks_raises_error(self):
        """Test that wrong number of risks raises ValueError."""
        # Too few risks
        with pytest.raises(ValueError, match="Expected exactly 5 risks, got 3"):
            RiskSummary(
                risks=["Risk 1", "Risk 2", "Risk 3"],
                ticker="PLD",
                company_name="Prologis",
                model="llama-3.3-70b-versatile",
                prompt_version="v1.0"
            )

        # Too many risks
        with pytest.raises(ValueError, match="Expected exactly 5 risks, got 6"):
            RiskSummary(
                risks=["Risk 1", "Risk 2", "Risk 3", "Risk 4", "Risk 5", "Risk 6"],
                ticker="PLD",
                company_name="Prologis",
                model="llama-3.3-70b-versatile",
                prompt_version="v1.0"
            )

    def test_empty_risk_raises_error(self):
        """Test that empty risks raise ValueError."""
        with pytest.raises(ValueError, match="All risks must be non-empty strings"):
            RiskSummary(
                risks=["Risk 1", "", "Risk 3", "Risk 4", "Risk 5"],
                ticker="PLD",
                company_name="Prologis",
                model="llama-3.3-70b-versatile",
                prompt_version="v1.0"
            )

        with pytest.raises(ValueError, match="All risks must be non-empty strings"):
            RiskSummary(
                risks=["Risk 1", "   ", "Risk 3", "Risk 4", "Risk 5"],
                ticker="PLD",
                company_name="Prologis",
                model="llama-3.3-70b-versatile",
                prompt_version="v1.0"
            )


class TestGroqRiskSummarizer:
    """Test suite for GroqRiskSummarizer."""

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_init(self, mock_groq):
        """Test GroqRiskSummarizer initialization."""
        summarizer = GroqRiskSummarizer(api_key="test-key")

        assert summarizer.model == "llama-3.3-70b-versatile"
        assert summarizer.temperature == 0.0
        assert summarizer.max_tokens == 2000
        assert summarizer.prompt_version == "v1.0"
        mock_groq.assert_called_once_with(api_key="test-key")

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_init_custom_params(self, mock_groq):
        """Test initialization with custom parameters."""
        summarizer = GroqRiskSummarizer(
            model="qwen2.5-72b-versatile",
            temperature=0.3,
            max_tokens=1500,
            prompt_version="v1.0",
            api_key="test-key"
        )

        assert summarizer.model == "qwen2.5-72b-versatile"
        assert summarizer.temperature == 0.3
        assert summarizer.max_tokens == 1500

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_chunk_text_small_text(self, mock_groq):
        """Test that small text returns single chunk."""
        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        text = "Short text that doesn't need chunking."
        chunks = summarizer._chunk_text(text, chunk_size=1000)

        assert len(chunks) == 1
        assert chunks[0] == text

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_chunk_text_at_sentence_boundaries(self, mock_groq):
        """Test that chunking splits at sentence boundaries."""
        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Create text with clear sentence boundaries
        text = "First sentence. " * 1000 + "Second sentence. " * 1000
        chunks = summarizer._chunk_text(text, chunk_size=10000)

        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk (except possibly last) should end with sentence ending
        for chunk in chunks[:-1]:
            assert chunk.rstrip()[-1] in '.!?'

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_chunk_text_no_sentence_boundary(self, mock_groq):
        """Test chunking when no sentence boundary found in lookback window."""
        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Create text with no periods for 1000 chars
        text = "a" * 20000
        chunks = summarizer._chunk_text(text, chunk_size=10000)

        # Should still chunk (at target size since no sentence found)
        assert len(chunks) == 2

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_routes_to_single_pass(self, mock_groq):
        """Test that small documents use single-pass summarization."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """1. Market volatility poses significant risks to operations.
2. Regulatory changes could impact business model and compliance costs.
3. Competition from established players threatens market share.
4. Economic downturn would reduce demand for services.
5. Technology disruption requires ongoing investment and adaptation."""

        mock_groq.return_value.chat.completions.create.return_value = mock_response

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Small text (< 40,000 chars)
        small_text = "Risk factors here. " * 100
        result = summarizer.summarize(small_text, "PLD", "Prologis")

        # Should call API once (single-pass)
        assert mock_groq.return_value.chat.completions.create.call_count == 1
        assert len(result.risks) == 5
        assert result.ticker == "PLD"
        assert result.company_name == "Prologis"

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_routes_to_two_pass(self, mock_groq):
        """Test that large documents use two-pass summarization."""
        # Setup mock responses for Pass 1 (chunks) and Pass 2 (meta)
        chunk_response = Mock()
        chunk_response.choices = [Mock()]
        chunk_response.choices[0].message.content = """1. Chunk risk 1
2. Chunk risk 2
3. Chunk risk 3
4. Chunk risk 4
5. Chunk risk 5"""

        meta_response = Mock()
        meta_response.choices = [Mock()]
        meta_response.choices[0].message.content = """1. Final risk 1
2. Final risk 2
3. Final risk 3
4. Final risk 4
5. Final risk 5"""

        # Return chunk response for each chunk, then meta response
        # ~90k chars with 35k chunk size = 3 chunks, so 3 chunk calls + 1 meta call = 4 total
        mock_groq.return_value.chat.completions.create.side_effect = [
            chunk_response,
            chunk_response,
            chunk_response,
            meta_response
        ]

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Large text (> 40,000 chars) - should create 3 chunks (~90k / 35k = 3)
        large_text = "Risk factor text. " * 5000  # ~90,000 chars
        result = summarizer.summarize(large_text, "AMT", "American Tower")

        # Should call API 4 times (3 chunks + 1 meta)
        assert mock_groq.return_value.chat.completions.create.call_count == 4
        assert len(result.risks) == 5
        assert "(2-pass)" in result.model

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_parse_response_valid(self, mock_groq):
        """Test parsing valid numbered response."""
        summarizer = GroqRiskSummarizer(api_key="test-key")

        response = """1. Market volatility poses significant risks to operations.
2. Regulatory changes could impact business model.
3. Competition from established players threatens market share.
4. Economic downturn would reduce demand.
5. Technology disruption requires ongoing investment."""

        risks = summarizer._parse_response(response)

        assert len(risks) == 5
        assert "Market volatility" in risks[0]
        assert "Regulatory changes" in risks[1]
        assert "Competition" in risks[2]
        assert "Economic downturn" in risks[3]
        assert "Technology disruption" in risks[4]

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_parse_response_multiline_risks(self, mock_groq):
        """Test parsing response with multi-line risk descriptions."""
        summarizer = GroqRiskSummarizer(api_key="test-key")

        response = """1. Market volatility poses significant risks
to our operations and financial performance.
2. Regulatory changes could impact our business
model and increase compliance costs.
3. Competition from established players
threatens our market share.
4. Economic downturn would reduce
demand for our services.
5. Technology disruption requires
ongoing investment."""

        risks = summarizer._parse_response(response)

        assert len(risks) == 5
        # Multi-line risks should be joined
        assert "operations and financial performance" in risks[0]
        assert "compliance costs" in risks[1]

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_parse_response_wrong_count(self, mock_groq):
        """Test that wrong number of risks raises ValueError."""
        summarizer = GroqRiskSummarizer(api_key="test-key")

        # Only 3 risks
        response = """1. Risk one
2. Risk two
3. Risk three"""

        with pytest.raises(ValueError, match="Expected 5 risks, got 3"):
            summarizer._parse_response(response)

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_parse_response_flexible(self, mock_groq):
        """Test flexible parser accepts any number of risks."""
        summarizer = GroqRiskSummarizer(api_key="test-key")

        # 6 risks (would fail strict parser)
        response = """1. Risk one
2. Risk two
3. Risk three
4. Risk four
5. Risk five
6. Risk six"""

        risks = summarizer._parse_response_flexible(response)

        # Should parse all 6 without error
        assert len(risks) == 6

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_handles_six_risks_fallback(self, mock_groq):
        """Test that model returning 6 risks triggers fallback logic."""
        # Mock returns 6 risks instead of 5
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """1. Risk one
2. Risk two
3. Risk three
4. Risk four
5. Risk five
6. Risk six"""

        mock_groq.return_value.chat.completions.create.return_value = mock_response

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Should handle gracefully via fallback
        result = summarizer.summarize("Risk text", "AMT", "American Tower")

        # Should take first 5 risks
        assert len(result.risks) == 5
        assert "Risk one" in result.risks[0]
        assert "Risk five" in result.risks[4]

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_build_meta_prompt(self, mock_groq):
        """Test meta-prompt building for Pass 2."""
        summarizer = GroqRiskSummarizer(api_key="test-key")

        risks = [
            "Risk A from chunk 1",
            "Risk B from chunk 1",
            "Risk C from chunk 2",
            "Risk D from chunk 2",
            "Risk E from chunk 3"
        ]

        prompt = summarizer._build_meta_prompt(risks, "PLD", "Prologis")

        # Should include all risks
        assert "Risk A" in prompt
        assert "Risk E" in prompt
        # Should mention company
        assert "Prologis" in prompt or "PLD" in prompt
        # Should ask for exactly 5
        assert "5" in prompt or "five" in prompt.lower()

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_connection_error_retry(self, mock_groq):
        """Test that connection errors trigger retry logic."""
        # First call raises ConnectionError, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """1. Risk one
2. Risk two
3. Risk three
4. Risk four
5. Risk five"""

        mock_groq.return_value.chat.completions.create.side_effect = [
            ConnectionError("Network error"),
            mock_response
        ]

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Should retry and succeed
        result = summarizer.summarize("Risk text", "PLD", "Prologis")
        
        assert len(result.risks) == 5
        # Should have called twice (1 failure + 1 success)
        assert mock_groq.return_value.chat.completions.create.call_count == 2

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_connection_error_max_retries(self, mock_groq):
        """Test that max retries raises LLMError."""
        # All calls raise ConnectionError
        mock_groq.return_value.chat.completions.create.side_effect = ConnectionError("Network error")

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        # Should fail after 3 retries
        with pytest.raises(LLMSummarizationError, match="Groq connection failed"):
            summarizer.summarize("Risk text", "PLD", "Prologis")
        
        # Should have tried 3 times
        assert mock_groq.return_value.chat.completions.create.call_count == 3

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_api_error(self, mock_groq):
        """Test that API errors raise LLMError."""
        # Simulate API error (rate limit, auth, etc.)
        mock_groq.return_value.chat.completions.create.side_effect = Exception("API rate limit exceeded")

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        with pytest.raises(LLMSummarizationError, match="Groq summarization failed"):
            summarizer.summarize("Risk text", "PLD", "Prologis")

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_summarize_parse_error(self, mock_groq):
        """Test that unparseable responses raise LLMError."""
        # Return invalid format
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid response format without numbered list"

        mock_groq.return_value.chat.completions.create.return_value = mock_response

        summarizer = GroqRiskSummarizer(api_key="test-key")
        
        with pytest.raises(LLMSummarizationError, match="Response parsing failed"):
            summarizer.summarize("Risk text", "PLD", "Prologis")


class TestFactoryFunction:
    """Test suite for create_summarizer factory function."""

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_create_groq_default(self, mock_groq):
        """Test creating Groq summarizer with defaults."""
        summarizer = create_summarizer("groq")

        assert isinstance(summarizer, GroqRiskSummarizer)
        assert summarizer.model == "llama-3.3-70b-versatile"

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_create_groq_custom_model(self, mock_groq):
        """Test creating Groq summarizer with custom model."""
        summarizer = create_summarizer("groq", model="qwen2.5-72b-versatile")

        assert isinstance(summarizer, GroqRiskSummarizer)
        assert summarizer.model == "qwen2.5-72b-versatile"

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_create_groq_with_kwargs(self, mock_groq):
        """Test creating Groq summarizer with additional kwargs."""
        summarizer = create_summarizer(
            "groq",
            temperature=0.5,
            max_tokens=1500,
            api_key="test-key"
        )

        assert summarizer.temperature == 0.5
        assert summarizer.max_tokens == 1500

    def test_create_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_summarizer("openai")

        with pytest.raises(ValueError, match="Unsupported provider"):
            create_summarizer("anthropic")

    @patch('reit_risk_summarizer.services.llm.summarizer.Groq')
    def test_create_default_is_groq(self, mock_groq):
        """Test that default provider is Groq."""
        summarizer = create_summarizer()

        assert isinstance(summarizer, GroqRiskSummarizer)


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit
