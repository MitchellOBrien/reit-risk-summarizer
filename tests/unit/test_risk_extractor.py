"""Unit tests for risk factor extractor."""

import pytest

from reit_risk_summarizer.exceptions import RiskExtractionError
from reit_risk_summarizer.services.sec.extractor import RiskFactorExtractor


class TestRiskFactorExtractor:
    """Test suite for RiskFactorExtractor."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = RiskFactorExtractor()
        assert extractor is not None

    def test_extract_empty_html(self):
        """Test that empty HTML raises error."""
        extractor = RiskFactorExtractor()

        with pytest.raises(RiskExtractionError, match="Empty HTML content"):
            extractor.extract_risk_factors("")

        with pytest.raises(RiskExtractionError, match="Empty HTML content"):
            extractor.extract_risk_factors("   ")

    def test_extract_by_item_header_simple(self):
        """Test extraction with simple Item 1A header."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower minimum for testing

        html = """
        <html>
            <body>
                <h2>Item 1A. Risk Factors</h2>
                <p>Our business is subject to numerous risks and uncertainties.</p>
                <p>Market conditions may adversely affect our operations.</p>
                <p>Regulatory changes could impact our business model.</p>
                <p>Competition in our industry is intense and may increase.</p>
                <p>Cybersecurity threats pose ongoing risks to our operations.</p>
                <h2>Item 1B. Unresolved Staff Comments</h2>
                <p>None.</p>
            </body>
        </html>
        """

        result = extractor.extract_risk_factors(html)

        assert "risks and uncertainties" in result
        assert "Market conditions" in result
        assert "Regulatory changes" in result
        assert "Unresolved Staff Comments" not in result
        assert len(result) > 100  # Meets minimum length check

    def test_extract_by_item_header_variations(self):
        """Test extraction with various Item 1A header formats."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower minimum for testing

        # Test with "Item 1.A" format
        html1 = """
        <html>
            <body>
                <h1>Item 1.A - Risk Factors</h1>
                <p>Risk factor content goes here with sufficient length to pass validation.</p>
                <p>Additional risk information to ensure we meet minimum requirements.</p>
                <p>More detailed risk disclosures about market conditions and operations.</p>
                <p>Further elaboration on regulatory and competitive landscape risks.</p>
                <p>Cybersecurity and technology-related risk factors for our business.</p>
                <h1>Item 1.B</h1>
            </body>
        </html>
        """

        result1 = extractor.extract_risk_factors(html1)
        assert "Risk factor content" in result1
        assert "Item 1.B" not in result1

    def test_extract_stops_at_item_2(self):
        """Test that extraction stops at Item 2 if Item 1B is missing."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower minimum for testing

        html = """
        <html>
            <body>
                <h2>Item 1A. Risk Factors</h2>
                <p>These are the risk factors for our company and operations.</p>
                <p>We face various market and operational risks in our business.</p>
                <p>Regulatory compliance and legal risks affect our operations.</p>
                <p>Economic conditions may impact our financial performance.</p>
                <p>Technology and cybersecurity risks are increasingly important.</p>
                <h2>Item 2. Properties</h2>
                <p>Our principal offices are located in...</p>
            </body>
        </html>
        """

        result = extractor.extract_risk_factors(html)

        assert "risk factors" in result.lower()
        assert "Properties" not in result
        assert "principal offices" not in result

    def test_extract_with_nested_elements(self):
        """Test extraction with complex nested HTML structure."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower minimum for testing

        html = """
        <html>
            <body>
                <div>
                    <h3>Item 1A. Risk Factors</h3>
                    <div>
                        <p><b>Market Risks:</b> We operate in volatile markets.</p>
                        <p><b>Operational Risks:</b> Our operations face various challenges.</p>
                    </div>
                    <table>
                        <tr><td>Credit risk impacts our financial position significantly.</td></tr>
                        <tr><td>Liquidity risk affects our operational flexibility.</td></tr>
                    </table>
                    <div>
                        <p>Additional risk disclosures and management discussion points.</p>
                    </div>
                </div>
                <h3>Item 1B</h3>
            </body>
        </html>
        """

        result = extractor.extract_risk_factors(html)

        assert "Market Risks" in result
        assert "Operational Risks" in result
        assert "Credit risk" in result

    def test_clean_text_removes_headers(self):
        """Test that text cleaning removes Item 1A header."""
        extractor = RiskFactorExtractor(min_length=50)  # Lower minimum for testing

        text = "Item 1A. Risk Factors\nOur business faces various risks and uncertainties."
        cleaned = extractor._clean_text(text + " " * 50)  # Pad to meet minimum

        # Header should be removed
        assert "Item 1A" not in cleaned or cleaned.index("Risk") < cleaned.index("Item 1A")

    def test_clean_text_normalizes_whitespace(self):
        """Test that text cleaning normalizes whitespace."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower for testing

        text = "Line 1\n\n\n\n\nLine 2    with    spaces\n\n\nLine 3" + " content" * 100
        cleaned = extractor._clean_text(text)

        # Multiple blank lines should be reduced
        assert "\n\n\n" not in cleaned
        # Multiple spaces should be reduced
        assert "    " not in cleaned

    def test_clean_text_removes_page_numbers(self):
        """Test that text cleaning removes page numbers."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower for testing

        text = "Risk text here\n42\nMore risk text\n\n15\n\nEven more content" + " text" * 100
        cleaned = extractor._clean_text(text)

        # Standalone numbers should be removed
        assert "\n42\n" not in cleaned

    def test_clean_text_rejects_short_text(self):
        """Test that very short extracted text raises error."""
        extractor = RiskFactorExtractor()  # Uses default min_length=10,000

        short_text = "Too short"

        with pytest.raises(RiskExtractionError, match="too short"):
            extractor._clean_text(short_text)

    def test_no_risk_factors_found(self):
        """Test error when no risk factors section exists."""
        extractor = RiskFactorExtractor()

        html = """
        <html>
            <body>
                <h1>Annual Report</h1>
                <p>This document has no risk factors section.</p>
            </body>
        </html>
        """

        with pytest.raises(RiskExtractionError, match="Could not locate Item 1A"):
            extractor.extract_risk_factors(html)

    def test_extract_handles_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        extractor = RiskFactorExtractor(min_length=100)  # Lower minimum for testing

        html = """
        <html>
            <body>
                <h2>ITEM 1A. RISK FACTORS</h2>
                <p>Risk content in uppercase header format for testing purposes.</p>
                <p>Multiple paragraphs of risk factor disclosures and information.</p>
                <p>Detailed risk analysis covering various aspects of operations.</p>
                <p>Additional risk factors related to market and business conditions.</p>
                <p>Further elaboration on risks facing the organization and industry.</p>
                <h2>ITEM 1B</h2>
            </body>
        </html>
        """

        result = extractor.extract_risk_factors(html)
        assert "Risk content" in result

    def test_length_validation_with_defaults(self):
        """Test that default min_length (10,000 chars) catches suspiciously short extractions."""
        extractor = RiskFactorExtractor()  # Use defaults: min_length=10,000, raise_on_short=True

        # Short extraction (like table of contents - 500 chars)
        html = """
        <html>
            <body>
                <h2>Item 1A. Risk Factors</h2>
                <p>Business risks. Market risks. Regulatory risks. Financial risks. Operational risks.</p>
                <h2>Item 1B. Unresolved Staff Comments</h2>
            </body>
        </html>
        """

        with pytest.raises(RiskExtractionError, match="too short"):
            extractor.extract_risk_factors(html)

    def test_length_validation_warning_only(self):
        """Test that raise_on_short=False logs warning instead of raising error."""
        extractor = RiskFactorExtractor(min_length=10_000, raise_on_short=False)

        # Short extraction
        html = """
        <html>
            <body>
                <h2>Item 1A. Risk Factors</h2>
                <p>Business risks exist. Market volatility may impact results. Competition is intense.</p>
                <h2>Item 1B. Unresolved Staff Comments</h2>
            </body>
        </html>
        """

        # Should not raise, just log warning
        result = extractor.extract_risk_factors(html)
        assert len(result) > 0
        assert "risks exist" in result

    def test_length_validation_warn_threshold(self):
        """Test that warn_threshold triggers warning for borderline extractions."""
        # Set warn_threshold=5000, extraction will be ~200 chars (below threshold)
        extractor = RiskFactorExtractor(
            min_length=100,  # Won't trigger error
            warn_threshold=5000,  # Will trigger warning
            raise_on_short=True,
        )

        html = """
        <html>
            <body>
                <h2>Item 1A. Risk Factors</h2>
                <p>Our business faces various risks including market conditions and regulatory changes.</p>
                <p>Additional risks related to operations and competitive landscape are discussed below.</p>
                <h2>Item 1B. Unresolved Staff Comments</h2>
            </body>
        </html>
        """

        # Should extract without error but will log warning (we can't easily test logging in unit tests)
        result = extractor.extract_risk_factors(html)
        assert len(result) > 100
        assert len(result) < 5000


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit
