# Development Handoff - REIT Risk Summarizer

**Date:** December 4, 2024  
**Phase:** Core Implementation  
**Status:** Project structure complete, ready for service implementation

---

## 📋 Project Overview

**What:** LLM-powered system that analyzes SEC 10-K filings to extract and summarize the top 5 material risks for REITs (Real Estate Investment Trusts).

**Why:** Help financial advisors quickly compare REITs across sectors without reading 150-page documents.

**How:** Fetch 10-K from SEC EDGAR → Extract Risk Factors section → Use LLM to summarize → Evaluate against golden dataset.

---

## 🎯 Product Specification

### User Persona
- **Who:** Financial advisor at a mid-sized RIA managing $200M in assets
- **Scenario:** Client asks for REIT exposure; advisor needs to evaluate 4-5 REITs across different sectors
- **Pain Point:** No time to read five 150-page 10-K documents
- **Solution:** Get focused top 5 risk summaries in under 10 seconds per REIT

### Output Format
```json
{
  "ticker": "PLD",
  "company_name": "Prologis",
  "risks": [
    {
      "rank": 1,
      "title": "California Market Exposure",
      "description": "Nearly one-third of properties and revenues come from California markets. Economic downturn, oversupply, or unfavorable tax changes in the state would significantly hurt performance.",
      "category": "Geographic Concentration"
    }
    // ... 4 more risks
  ]
}
```

**Quality Criteria:**
- 2-3 sentences per risk
- Plain English (not legalese)
- Explains business impact (not just what the risk is)
- Prioritizes material risks over generic/boilerplate

---

## 🏗️ Architecture

### System Flow
```
API Request (ticker: "PLD")
    ↓
Orchestrator
    ↓
Check Cache → [Cache Hit? Return cached result]
    ↓
SEC Fetcher (fetch 10-K HTML from EDGAR)
    ↓
Risk Extractor (parse Item 1A from HTML)
    ↓
LLM Summarizer (GPT-4 summarizes top 5 risks)
    ↓
Cache Result → Return to API
```

### Layered Architecture
```
routers/          # FastAPI endpoints
    ↓
services/         # Business logic
    ├── sec/      # SEC data handling
    ├── llm/      # LLM integration
    └── cache/    # Caching
    ↓
External APIs     # SEC EDGAR, OpenAI/Anthropic
```

---

## 📁 Current Project State

### ✅ Completed Files

**Configuration & Setup:**
- `pyproject.toml` - Dependencies (FastAPI, OpenAI, sentence-transformers, etc.)
- `Makefile` - Dev commands (make run, make test, make evaluate, etc.)
- `.pre-commit-config.yaml` - Code quality hooks (black, ruff, mypy)
- `.env.example` - Environment variables template

**Core Application:**
- `src/reit_risk_summarizer/config.py` - Settings with pydantic-settings
- `src/reit_risk_summarizer/exceptions.py` - Custom exceptions
- `src/reit_risk_summarizer/middlewares.py` - Logging & error handling
- `src/reit_risk_summarizer/main.py` - FastAPI app entry point

**API Layer:**
- `src/reit_risk_summarizer/routers/health.py` - Health check endpoints
- `src/reit_risk_summarizer/routers/risks.py` - Risk summarization endpoint (with mock data)
- `src/reit_risk_summarizer/schemas/requests.py` - Request validation
- `src/reit_risk_summarizer/schemas/responses.py` - Response formatting

**Data:**
- `evaluation/golden_dataset.csv` - Ground truth for 10 REITs (PLD, AMT, EQIX, PSA, O, AVB, WELL, VTR, DLR, SPG)

### ❌ Not Yet Implemented

**Services Layer (Priority Order):**
1. `services/sec/fetcher.py` - Fetch 10-K from SEC EDGAR ⬅️ **START HERE**
2. `services/sec/extractor.py` - Extract Item 1A Risk Factors
3. `services/llm/summarizer.py` - LLM risk summarization
4. `services/llm/prompts/v1_0.py` - Prompt templates
5. `services/orchestrator.py` - Pipeline coordination
6. `services/cache.py` - Caching layer

**Evaluation:**
7. `evaluation/metrics.py` - Semantic similarity, NDCG@5, sector-specificity
8. `evaluation/evaluator.py` - Evaluation runner
9. `evaluation/run_evaluation.py` - CLI script

**Testing:**
10. `tests/unit/` - Unit tests for each service
11. `tests/integration/` - End-to-end API tests

---

## 🎯 Next Task: Implement SEC Data Fetcher

### File Location
`src/reit_risk_summarizer/services/sec/fetcher.py`

### Requirements

**Input:** 
- `ticker: str` - Stock ticker symbol (e.g., "PLD", "AMT")

**Output:** 
- `str` - Raw HTML content of the latest 10-K filing

**Error Handling:**
- Raise `InvalidTickerError` if ticker is invalid
- Raise `SECFetchError` if fetch fails (network, not found, rate limit)

**Performance:**
- Respect SEC rate limit: 10 requests per second
- Cache raw HTML for 7 days (to avoid re-fetching)

### SEC EDGAR API Details

**Important SEC Requirements:**
1. **User-Agent MUST be set** with format: `"YourName your@email.com"`
2. Rate limit: 10 requests per second (enforced by SEC)
3. Company CIK (Central Index Key) must be looked up before fetching filings

**API Flow:**
```python
# Step 1: Get company CIK from ticker
# Endpoint: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&count=1
# Parse response to get CIK number

# Step 2: Find latest 10-K filing
# Endpoint: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=exclude&count=1
# Parse to get filing URL

# Step 3: Fetch filing HTML
# Download from filing URL
```

**Example:**
- Prologis ticker: "PLD"
- Prologis CIK: "1045610"
- Latest 10-K URL: `https://www.sec.gov/Archives/edgar/data/1045610/000095017024014539/pld-20231231.htm`

### Class Structure

```python
from typing import Optional
import requests
from bs4 import BeautifulSoup
from ..config import get_settings
from ..exceptions import InvalidTickerError, SECFetchError

class SECFetcher:
    """Fetches 10-K filings from SEC EDGAR."""
    
    def __init__(self):
        settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.sec_api_user_agent
        })
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    def fetch_latest_10k(self, ticker: str) -> str:
        """
        Fetch the latest 10-K filing HTML for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'PLD')
            
        Returns:
            Raw HTML content of the 10-K filing
            
        Raises:
            InvalidTickerError: If ticker is invalid or not found
            SECFetchError: If fetch fails for any reason
        """
        # TODO: Implement
        pass
    
    def _get_cik(self, ticker: str) -> str:
        """Get CIK (Central Index Key) for a ticker."""
        # TODO: Implement
        pass
    
    def _find_latest_filing_url(self, cik: str) -> str:
        """Find the URL of the latest 10-K filing."""
        # TODO: Implement
        pass
    
    def _fetch_filing_html(self, url: str) -> str:
        """Fetch the HTML content from a filing URL."""
        # TODO: Implement
        pass
```

### Test Cases to Consider

Create `tests/unit/test_sec_fetcher.py`:

```python
import pytest
from reit_risk_summarizer.services.sec.fetcher import SECFetcher
from reit_risk_summarizer.exceptions import InvalidTickerError, SECFetchError

def test_fetch_valid_ticker():
    """Test fetching 10-K for a valid ticker."""
    fetcher = SECFetcher()
    html = fetcher.fetch_latest_10k("PLD")
    assert html is not None
    assert len(html) > 1000  # 10-Ks are long
    assert "Risk Factors" in html or "risk factors" in html

def test_fetch_invalid_ticker():
    """Test that invalid ticker raises error."""
    fetcher = SECFetcher()
    with pytest.raises(InvalidTickerError):
        fetcher.fetch_latest_10k("INVALID_TICKER_XYZ")

def test_user_agent_is_set():
    """Test that User-Agent header is properly set."""
    fetcher = SECFetcher()
    assert 'User-Agent' in fetcher.session.headers
    assert '@' in fetcher.session.headers['User-Agent']  # Contains email

# TODO: Add test for rate limiting
# TODO: Add test for network failure handling
```

### Known Challenges

1. **HTML Parsing Complexity**
   - Different companies format 10-Ks differently
   - Some use tables, some use plain text
   - Need robust parsing that works across formats

2. **Rate Limiting**
   - SEC enforces 10 req/sec
   - Should implement exponential backoff for retries

3. **CIK Lookup**
   - Ticker → CIK mapping can fail for edge cases
   - Some tickers have multiple CIKs (different share classes)

4. **Error Cases**
   - Company hasn't filed 10-K yet this year
   - Network timeouts
   - SEC site temporarily down

---

## 📊 Evaluation Strategy (Phase 1)

Once services are implemented, we'll evaluate using 3 automated metrics:

### 1. Semantic Similarity (Primary)
- Measures how close LLM output is to expert ground truth
- Uses sentence transformers (all-MiniLM-L6-v2)
- **Target:** >0.75

### 2. NDCG@5 (Ranking Quality)
- Measures if top risks are correctly prioritized
- **Target:** >0.70

### 3. Sector-Specificity Score (Generic Detection)
- Detects generic vs. sector-specific risks
- Compares risk across different REIT sectors
- **Target:** >0.40 for all risks

### Golden Dataset
- 10 REITs across sectors: PLD, AMT, EQIX, PSA, O, AVB, WELL, VTR, DLR, SPG
- Each has manually curated top 5 risks
- CSV format with: ticker, risk_rank, risk_title, risk_description, risk_category

---

## 🔧 Development Commands

```bash
# Setup
uv venv
source .venv/bin/activate
make dev-install

# Run API
make run                    # Development with auto-reload
make run-prod              # Production mode

# Testing
make test                  # Run all tests
make test-verbose          # Verbose output
pytest tests/unit/test_sec_fetcher.py -v  # Specific test

# Code Quality
make format                # Auto-format with black & isort
make lint                  # Run ruff & mypy
make clean                 # Clean cache files

# Evaluation
make evaluate              # Run evaluation on golden dataset
```

---

## 🎨 Code Standards

- **Type hints:** Required for all function signatures
- **Docstrings:** Google style for public functions
- **Line length:** 100 characters (enforced by black)
- **Imports:** Organized by isort (stdlib, third-party, local)
- **Linting:** Ruff (replaces flake8, pylint, isort checks)
- **Error handling:** Use custom exceptions from `exceptions.py`

---

## 📚 Key Dependencies

```toml
# Core
fastapi = ">=0.109.0"
uvicorn = ">=0.27.0"
pydantic = ">=2.6.0"
pydantic-settings = ">=2.1.0"

# SEC Data
requests = ">=2.31.0"
beautifulsoup4 = ">=4.12.0"
lxml = ">=5.1.0"

# LLM
openai = ">=1.10.0"
anthropic = ">=0.18.0"

# Evaluation
sentence-transformers = ">=2.3.0"
scikit-learn = ">=1.4.0"
scipy = ">=1.12.0"
```

---

## 🚀 Deployment Target

**Platform:** Google Cloud Run  
**Why:** Serverless, auto-scaling, pay-per-use

Key considerations:
- Container must respond to HTTP on port 8000
- Environment variables via Cloud Run config or Secret Manager
- Timeout set to 60s (10-K processing can be slow)
- Memory: 2Gi, CPU: 2

---

## 💡 Implementation Tips

1. **Start Simple**
   - Get basic fetch working first
   - Add caching later
   - Add rate limiting after basic functionality works

2. **Test with Golden Dataset Tickers**
   - Use PLD, AMT, EQIX first (they're large, well-formatted)
   - Test edge cases with smaller REITs

3. **Look at Existing Code Patterns**
   - `config.py` shows how to use settings
   - `exceptions.py` shows custom exception patterns
   - `middlewares.py` shows error handling approach

4. **Use Dependencies Pattern**
   - FastAPI dependency injection for testability
   - See `routers/health.py` for example: `settings: Settings = Depends(get_settings)`

---

## 📞 Questions to Consider

As you implement, think about:
- How do we handle companies that haven't filed 10-K this year?
- Should we cache CIK lookups separately from filing HTML?
- What's the retry strategy for SEC rate limiting?
- How do we validate that fetched HTML is actually a 10-K (not 10-Q)?

---

## 🎯 Success Criteria

**SEC Fetcher is complete when:**
- ✅ Can fetch latest 10-K for all 10 tickers in golden dataset
- ✅ Handles invalid tickers gracefully
- ✅ Respects SEC rate limits
- ✅ Has proper error handling and logging
- ✅ Unit tests pass
- ✅ Type hints pass mypy check
- ✅ Code passes ruff linting

**Next Step After SEC Fetcher:**
Implement `services/sec/extractor.py` to parse Item 1A from the HTML.

---

## 📝 Notes from Previous Session

- We chose a hybrid structure combining FastAPI best practices with ML evaluation patterns
- Emphasis on evaluation-driven development (build → measure → iterate)
- Target audience is financial advisors, not technical users
- Focus on speed (<10s) and accuracy (>0.75 similarity)
- Use Cloud Run for deployment (not AWS or Railway)

---

**Ready to start? Begin with `services/sec/fetcher.py`!**