# Risk Factor Extraction Validation

The `RiskFactorExtractor` includes built-in length validation to catch extraction errors, such as accidentally capturing only the table of contents instead of the full risk factors section.

## Configuration Options

### `min_length` (default: 10,000)
Minimum acceptable character count for extracted risk factors. If extraction is below this threshold:
- **raise_on_short=True**: Raises `RiskExtractionError`
- **raise_on_short=False**: Logs warning and continues

**Typical values:**
- Most 10-K risk factors: 50,000 - 150,000 characters
- Minimum for production: 10,000 characters (default)
- Testing with mock data: 100-500 characters

### `warn_threshold` (default: 30,000)
Character count that triggers a warning log. Useful for catching borderline cases that pass validation but seem suspiciously short.

### `raise_on_short` (default: True)
Controls whether short extractions raise an error or just log a warning.

## Usage Examples

### Production Use (Strict Validation)
```python
from reit_risk_summarizer.services.sec import RiskFactorExtractor

# Default behavior - will error if extraction < 10,000 chars
extractor = RiskFactorExtractor()

try:
    risk_factors = extractor.extract_risk_factors(html)
except RiskExtractionError as e:
    print(f"Extraction failed: {e}")
    print(f"Details: {e.details}")
```

### Lenient Mode (Warning Only)
```python
# Log warnings instead of raising errors
extractor = RiskFactorExtractor(
    min_length=10_000,
    raise_on_short=False  # Only log warnings
)

risk_factors = extractor.extract_risk_factors(html)
# Will proceed even if short, but logs warning
```

### Custom Thresholds
```python
# Custom validation for edge cases
extractor = RiskFactorExtractor(
    min_length=5_000,      # Lower minimum (smaller companies)
    warn_threshold=20_000,  # Earlier warning
    raise_on_short=True     # Still enforce minimum
)
```

### Testing with Mock Data
```python
# Disable validation for unit tests
extractor = RiskFactorExtractor(
    min_length=100,          # Very low threshold
    warn_threshold=50_000,   # High warning (won't trigger)
    raise_on_short=True
)
```

## Error Details

When `raise_on_short=True` and extraction is too short, the error includes:
- **length**: Actual character count
- **min_length**: Required minimum
- **preview**: First 500 characters of extracted text

```python
try:
    risk_factors = extractor.extract_risk_factors(html)
except RiskExtractionError as e:
    print(f"Length: {e.details['length']:,} chars")
    print(f"Required: {e.details['min_length']:,} chars")
    print(f"Preview: {e.details['preview']}")
```

## Real-World Example

```python
from reit_risk_summarizer.services.sec import SECFetcher, RiskFactorExtractor
import logging

logging.basicConfig(level=logging.WARNING)

fetcher = SECFetcher()
extractor = RiskFactorExtractor(
    min_length=10_000,      # Catch table-of-contents extractions
    warn_threshold=30_000,  # Warn if < 30k (most are 50k-150k)
    raise_on_short=True
)

tickers = ["PLD", "AMT", "EQIX", "PSA", "O"]

for ticker in tickers:
    try:
        html = fetcher.fetch_latest_10k(ticker)
        risk_factors = extractor.extract_risk_factors(html)
        print(f"✓ {ticker}: {len(risk_factors):,} characters")
    except RiskExtractionError as e:
        print(f"✗ {ticker}: {e}")
        # Could retry with different strategy or flag for manual review
```

## Validation Scenarios

| Scenario | Character Count | Behavior (default) |
|----------|----------------|-------------------|
| Normal extraction | 50k - 150k | ✓ Success |
| Small company | 30k - 50k | ⚠️ Warning logged, proceeds |
| Table of contents only | 1k - 5k | ❌ RiskExtractionError |
| Failed extraction | < 500 | ❌ RiskExtractionError |
