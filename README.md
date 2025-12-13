# REIT Risk Summarizer

> LLM-powered risk analysis from SEC 10-K filings for Real Estate Investment Trusts (REITs)

An end-to-end production ML system showcasing **Evaluation-Driven Development (EDD)** for LLM applications.

---

## 🎯 What This Project Does

This system helps financial advisors quickly understand REIT investment risks by:

1. **Fetching** the latest 10-K filings from SEC EDGAR
2. **Extracting** the Risk Factors section (Item 1A)  
3. **Analyzing** risks using LLMs to identify the top 5 most material threats
4. **Evaluating** output quality against expert-curated ground truth

**Use Case:** A financial advisor needs to compare 4-5 REITs across different sectors (industrial, healthcare, retail) to make a recommendation. Instead of reading five 150-page documents, they get focused risk summaries in under 10 seconds.

---

## ✨ Key Features

### For ML Engineers & Data Scientists
- 🎯 **Evaluation-Driven Development** - Metrics-first approach with golden dataset
- 📊 **Automated Quality Metrics** - Semantic similarity, NDCG@5, sector-specificity scoring
- 🔄 **Full ML Lifecycle** - Dev → Testing → Prod → Monitoring
- 📈 **LLM-as-Judge** - Sophisticated quality evaluation (Phase 2)
- 🧩 **Intelligent Chunking** - Automatic two-pass summarization for large documents (>40K chars)

### For Software Engineers  
- 🚀 **Production-Ready FastAPI** - Clean architecture, proper error handling
- 🔧 **Modern Python Tooling** - uv, pyproject.toml, pre-commit hooks
- 🐳 **Docker & Compose** - One-command infrastructure setup
- ✅ **CI/CD Ready** - GitHub Actions, automated testing & deployment
- 📝 **Type Safety** - Pydantic schemas, mypy type checking

### For Technical Showcasing
- 💼 **Portfolio-Ready** - Demonstrates full-stack ML engineering skills
- 📚 **Well-Documented** - Clear code, comprehensive README, inline comments
- 🧪 **Test Coverage** - Unit tests, integration tests, evaluation framework
- 🏗️ **Scalable Architecture** - Modular, layered design following best practices

---

## 🏗️ Project Structure

```
reit-risk-summarizer/
│
├── src/reit_risk_summarizer/           # Main application code
│   ├── routers/                        # API route definitions
│   │   ├── health.py                   # Health check endpoints
│   │   └── risks.py                    # Risk summarization endpoints
│   │
│   ├── services/                       # Business logic layer
│   │   ├── sec/                        # SEC data handling
│   │   │   ├── fetcher.py             # Fetch 10-K filings from EDGAR
│   │   │   └── extractor.py           # Extract Item 1A risk sections
│   │   ├── llm/                       # LLM integration (Groq + HuggingFace)
│   │   │   ├── summarizer.py          # Risk summarization with two-pass chunking
│   │   │   └── prompts/               # Versioned prompt templates
│   │   │       └── v1_0.py
│   │   ├── orchestrator.py            # Pipeline coordination
│   │   └── cache.py                   # Caching layer
│   │
│   ├── schemas/                       # Pydantic models
│   │   ├── requests.py               # API request validation
│   │   └── responses.py              # API response formatting
│   │
│   ├── config.py                     # Settings management
│   ├── exceptions.py                 # Custom exceptions
│   ├── middlewares.py                # Logging, error handling
│   └── main.py                       # FastAPI application entry
│
├── evaluation/                        # Evaluation framework
│   ├── metrics.py                    # Similarity, NDCG, specificity
│   ├── evaluator.py                  # Evaluation orchestration
│   ├── golden_dataset.csv            # Ground truth (10 REITs)
│   └── run_evaluation.py             # CLI evaluation script
│
├── tests/                            # Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   └── conftest.py                  # Pytest fixtures
│
├── notebooks/                        # Jupyter notebooks
│   └── exploration.ipynb            # Data exploration, analysis
│
├── scripts/                         # Utility scripts
│   └── fetch_10ks.py               # Pre-fetch 10-K data
│
├── docker/
│   ├── Dockerfile                   # Container definition
│   └── docker-compose.yml           # Multi-service orchestration
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Run tests on PRs
│       └── deploy.yml               # Deploy on merge
│
├── pyproject.toml                   # Project config & dependencies
├── uv.lock                          # Locked dependencies
├── Makefile                         # Common commands
├── .pre-commit-config.yaml          # Code quality automation
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (Python 3.13 recommended)
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **Groq API key** (free tier available - for cloud LLM)
  - OR **Hugging Face** (for local/offline LLM - no API key needed)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd reit-risk-summarizer

# 2. Install uv (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
pip install uv

# 3. Sync dependencies (creates .venv and installs everything)
uv sync --all-extras

# 4. Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key:
# - GROQ_API_KEY=gsk_...
# (Get free API key at https://console.groq.com/)
#
# OR use local HuggingFace models (no API key needed):
# - Set DEFAULT_LLM_PROVIDER=huggingface
```

### Running the Application

```bash
# Start the API server (with auto-reload)
make run

# API will be available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Health Check: http://localhost:8000/health
```

### Test the API

```bash
# Get risk summary for Prologis (PLD)
curl http://localhost:8000/risks/PLD

# Or use the interactive docs at http://localhost:8000/docs
```

---

## 📊 Running Evaluations

The evaluation framework measures system quality against a golden dataset of 10 REITs.

```bash
# Run full evaluation suite
make evaluate

# Expected output:
# ✓ Semantic Similarity: 0.82 (target: >0.75)
# ✓ NDCG@5: 0.78 (target: >0.70)
# ✓ Sector-Specificity: 0.65 (target: >0.40)
```

---

## 🧪 Development Workflow

### Managing Dependencies

```bash
# Sync dependencies from uv.lock (after git pull)
uv sync --all-extras

# Add a new production dependency
uv add requests
uv add "fastapi>=0.100.0"  # With version constraint

# Add a development dependency
uv add --dev pytest
uv add --dev "ruff>=0.1.0"

# Remove a dependency
uv remove requests

# Update a specific package
uv add --upgrade fastapi

# Update all dependencies (regenerates uv.lock)
uv sync --upgrade
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/unit/ --cov=src/reit_risk_summarizer --cov-report=html

# Run specific test file
pytest tests/unit/test_sec_fetcher.py -v

# Run with markers
pytest -m "not integration"  # Skip slow integration tests
pytest -m unit               # Only unit tests

# Run and open coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Code Quality

```bash
# Format code with ruff
ruff format .

# Run linting
ruff check .

# Run type checking
mypy src/

# Run all quality checks
ruff check . && ruff format --check . && mypy src/
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs (Try endpoints in browser)
- **ReDoc**: http://localhost:8000/redoc (Beautiful API reference)

---

### Endpoints

#### 1. `GET /api/v1/risks/{ticker}`

Get top 5 material risks for a REIT by processing its latest 10-K filing.

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `ticker` | string | Yes | Stock ticker symbol (1-10 chars) | `AMT`, `PLD`, `EQIX` |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `force_refresh` | boolean | No | `false` | Bypass cache and fetch fresh data |
| `model` | string | No | `llama-3.3-70b-versatile` | Override default LLM model |

**Success Response (200 OK):**
```json
{
  "ticker": "AMT",
  "company_name": "American Tower Corporation",
  "risks": [
    "Significant exposure to technological disruption in telecommunications infrastructure...",
    "Heavy debt burden with $40B+ in long-term obligations creating refinancing risk...",
    "Geographic concentration with 40% of revenue from U.S. market...",
    "Regulatory uncertainty across multiple international jurisdictions...",
    "Tenant concentration with major wireless carriers representing 75% of revenue..."
  ],
  "model": "llama-3.3-70b-versatile",
  "prompt_version": "v1.0",
  "cached": false
}
```

**Error Responses:**

| Status Code | Description | Example Response |
|------------|-------------|------------------|
| 404 | Ticker not found or no 10-K available | `{"detail": "Could not fetch 10-K filing for XYZ..."}` |
| 500 | Risk extraction failed | `{"detail": "Failed to extract risk factors..."}` |
| 503 | LLM service unavailable | `{"detail": "LLM service unavailable. Please try again later."}` |

**Examples:**

```bash
# Basic request
curl http://localhost:8000/api/v1/risks/AMT

# Force refresh (bypass cache)
curl http://localhost:8000/api/v1/risks/AMT?force_refresh=true

# Override model
curl http://localhost:8000/api/v1/risks/PLD?model=llama-3.1-70b-versatile
```

**Python Example:**
```python
import requests

response = requests.get("http://localhost:8000/api/v1/risks/AMT")
data = response.json()

print(f"{data['company_name']} - Top 5 Risks:")
for i, risk in enumerate(data['risks'], 1):
    print(f"{i}. {risk[:100]}...")  # First 100 chars
```

---

#### 2. `DELETE /api/v1/risks/cache/{ticker}`

Clear cached data (10-K HTML and extracted risks) for a specific ticker.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ticker` | string | Yes | Stock ticker to clear cache for |

**Success Response (204 No Content):**
```
(Empty response body)
```

**Example:**
```bash
# Clear cache for AMT
curl -X DELETE http://localhost:8000/api/v1/risks/cache/AMT

# Next request will fetch fresh data
curl http://localhost:8000/api/v1/risks/AMT
```

---

#### 3. `DELETE /api/v1/risks/cache`

Clear all cached data for all tickers.

**Success Response (204 No Content):**
```
(Empty response body)
```

**Example:**
```bash
# Clear entire cache
curl -X DELETE http://localhost:8000/api/v1/risks/cache
```

---

#### 4. `GET /health`

Health check endpoint for monitoring.

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Caching Behavior

The API caches two things to improve performance:

1. **10-K HTML** (key: `{ticker}_10k_html`)
   - Raw HTML from SEC EDGAR
   - Cached after first fetch
   - Bypass with `?force_refresh=true`

2. **Extracted Risk Text** (key: `{ticker}_risk_text`)
   - Item 1A section parsed from HTML
   - Cached after first extraction
   - Bypass with `?force_refresh=true`

**Not cached:** LLM summaries (depend on model parameters)

**Cache persistence:** In-memory (cleared on server restart)

**Typical performance:**
- First request: 8-15 seconds (fetch + extract + summarize)
- Cached request: 2-5 seconds (only summarize)
- Cache speedup: ~3-5x faster

---

### Rate Limits

**Groq Free Tier:**
- 30 requests/minute
- 14,400 tokens/minute

**Recommended:** Add caching and rate limiting middleware for production use.

---

## 🎯 Evaluation Metrics

The system uses three automated metrics to ensure quality:

### 1. **Semantic Similarity** (Primary Metric)
- Measures how close LLM output is to expert ground truth
- Uses sentence transformers (all-MiniLM-L6-v2)
- **Target:** >0.75
- **Interpretation:** 0.82 means output captures 82% of ground truth meaning

### 2. **NDCG@5** (Ranking Quality)
- Normalized Discounted Cumulative Gain
- Measures if top risks are correctly prioritized
- **Target:** >0.70
- **Interpretation:** Penalizes wrong items at top of ranking

### 3. **Sector-Specificity Score** (Generic Detection)
- Detects generic vs. sector-specific risks
- Compares risk text across different REIT sectors
- **Target:** >0.40 for all risks
- **Interpretation:** 0.65 = highly specific, 0.20 = generic boilerplate

### Phase 2: LLM-as-Judge (Coming Soon)
Multi-dimensional quality scoring:
- Accuracy (matches ground truth?)
- Business Impact Clarity (explains WHY it matters?)
- Specificity (concrete details vs. vague?)
- Actionability (can advisor explain to client?)

---

## 🏛️ Architecture & Design Decisions

### Layered Architecture
Following FastAPI + Domain-Driven Design best practices:

```
API Layer (FastAPI)
    ↓
Orchestration (Pipeline coordination)
    ↓
Services Layer (Business logic)
    ├── SEC Service (Data fetching & extraction)
    ├── LLM Service (Summarization)
    └── Cache Service (Performance)
    ↓
External APIs (SEC EDGAR, OpenAI/Anthropic)
```

### Key Design Patterns

**1. Two-Pass Chunking for Large Documents**
- Automatically handles documents >40K characters (exceeding API token limits)
- **Pass 1**: Split at sentence boundaries (35K chunks) → 5 risks per chunk
- **Pass 2**: Meta-summarize all chunk risks → select top 5 overall
- Benefits: 100% document coverage, no information loss, handles any size

**2. Repository Pattern** (Future)
- Clean abstraction over data access
- Easy to mock for testing
- Swap databases without changing business logic

**3. Dependency Injection**
- Services injected via FastAPI's `Depends()`
- Makes testing easier
- Loose coupling between components

**4. Pipeline/ETL Pattern**
- Clear data flow: Fetch → Extract → Summarize
- Each step can be tested independently
- Easy to add new steps (e.g., validation)

**5. Evaluation-Driven Development**
- Golden dataset defines "good"
- Metrics guide iteration
- Automated quality gates

### Why This Stack?

| Technology | Reason |
|-----------|--------|
| **FastAPI** | Modern, fast, auto-docs, async support |
| **uv** | 10-100x faster than pip, proper lock files |
| **Pydantic** | Type safety, validation, auto-docs |
| **Sentence Transformers** | Fast, deterministic embeddings for eval |
| **Pre-commit** | Enforce code quality automatically |
| **Docker** | Reproducible environments |

---

## 🧪 Testing Strategy

```bash
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_sec_fetcher.py
│   ├── test_extractor.py
│   └── test_llm_summarizer.py
├── integration/             # End-to-end tests
│   └── test_api.py
└── conftest.py             # Shared fixtures
```

**Test Coverage Goals:**
- Unit tests: >80%
- Integration tests: Core user flows
- Evaluation: Automated quality metrics

```bash
# Run specific test file
pytest tests/unit/test_sec_fetcher.py -v

# Run with coverage
pytest --cov=src/reit_risk_summarizer --cov-report=html

# Run only fast tests (skip integration)
pytest -m "not integration"
```

---

## 📦 Deployment

### Local Development with Docker

```bash
# Build and start services
make docker-up

# Services started:
# - API (port 8000)
# - Redis (port 6379)

# View logs
make docker-logs

# Stop services
make docker-down
```

### Google Cloud Run Deployment

This project is optimized for deployment on **Google Cloud Run**, offering automatic scaling, zero server management, and pay-per-use pricing.

#### Prerequisites
```bash
# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### Deploy to Cloud Run

```bash
# 1. Build and push container to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/reit-risk-summarizer

# 2. Deploy to Cloud Run
gcloud run deploy reit-risk-summarizer \
  --image gcr.io/YOUR_PROJECT_ID/reit-risk-summarizer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars GROQ_API_KEY=your-key \
  --set-env-vars DEFAULT_LLM_PROVIDER=groq \
  --set-env-vars CACHE_TYPE=memory \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60s \
  --max-instances 10

# 3. Get the service URL
gcloud run services describe reit-risk-summarizer \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

#### Cloud Run + Redis (Cloud Memorystore)

For production with Redis caching:

```bash
# 1. Create a VPC connector
gcloud compute networks vpc-access connectors create reit-connector \
  --region us-central1 \
  --range 10.8.0.0/28

# 2. Create Redis instance
gcloud redis instances create reit-cache \
  --size=1 \
  --region=us-central1 \
  --redis-version=redis_7_0

# 3. Get Redis host
REDIS_HOST=$(gcloud redis instances describe reit-cache \
  --region=us-central1 \
  --format='get(host)')

# 4. Deploy with Redis connection
gcloud run deploy reit-risk-summarizer \
  --image gcr.io/YOUR_PROJECT_ID/reit-risk-summarizer \
  --vpc-connector reit-connector \
  --set-env-vars REDIS_HOST=$REDIS_HOST \
  --set-env-vars CACHE_TYPE=redis \
  ... # other flags
```

#### Using Secret Manager for API Keys

```bash
# 1. Store API keys in Secret Manager
echo -n "your-groq-key" | gcloud secrets create groq-api-key \
  --data-file=-

# 2. Deploy with secrets
gcloud run deploy reit-risk-summarizer \
  --image gcr.io/YOUR_PROJECT_ID/reit-risk-summarizer \
  --update-secrets GROQ_API_KEY=groq-api-key:latest \
  ... # other flags
```

#### CI/CD with Cloud Build

Create `cloudbuild.yaml`:
```yaml
steps:
  # Run tests
  - name: 'python:3.11'
    entrypoint: pip
    args: ['install', 'pytest', 'pytest-cov']
  - name: 'python:3.11'
    entrypoint: pytest
    args: ['tests/']

  # Build container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/reit-risk-summarizer', '.']

  # Push to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/reit-risk-summarizer']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'reit-risk-summarizer'
      - '--image=gcr.io/$PROJECT_ID/reit-risk-summarizer'
      - '--region=us-central1'
      - '--platform=managed'
```

Trigger deployment:
```bash
gcloud builds submit --config cloudbuild.yaml
```

#### Cloud Run Cost Optimization

```bash
# Set minimum instances to 0 for cost savings
gcloud run services update reit-risk-summarizer \
  --min-instances 0 \
  --max-instances 10

# For production with SLA requirements:
gcloud run services update reit-risk-summarizer \
  --min-instances 1 \  # Always have 1 warm instance
  --max-instances 20
```

#### Monitoring & Logging

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=reit-risk-summarizer" \
  --limit 50 \
  --format json

# Set up alerts in Cloud Console:
# - High latency (P95 > 10s)
# - Error rate > 5%
# - Memory usage > 80%
```

---

## 🔧 Configuration

All settings via environment variables (`.env`):

```bash
# LLM Configuration
# Option 1: Groq (cloud, fast, FREE tier)
GROQ_API_KEY=gsk_...                    # Get at https://console.groq.com/
DEFAULT_LLM_PROVIDER=groq               # or 'huggingface'
DEFAULT_LLM_MODEL=llama-3.3-70b-versatile  # Groq models: llama-3.3-70b-versatile, qwen2.5-72b-versatile, mixtral-8x7b-32768

# Option 2: HuggingFace (local, offline, no API needed)
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct  # Or any HF model
HF_TOKEN=hf_...                         # Optional, only for gated models

# Common settings
LLM_TEMPERATURE=0.0          # Lower = more consistent (0.0 = deterministic)
LLM_MAX_TOKENS=2000          # Max response tokens (300 for HuggingFace)

# Caching
CACHE_ENABLED=true
CACHE_TTL_SECONDS=86400      # 24 hours
CACHE_TYPE=memory            # or 'redis' for production

# Application
ENVIRONMENT=development      # development, staging, production
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR

# SEC EDGAR
SEC_API_EMAIL=your@email.com
SEC_API_USER_AGENT="YourName your@email.com"
```



## 📚 Project Evolution

This project follows a phased approach:

### ✅ Phase 1: MVP (Weeks 1-2)
- [x] Project structure setup
- [x] FastAPI skeleton with health checks
- [x] Configuration management
- [ ] SEC data fetching
- [ ] Risk extraction
- [ ] Basic LLM summarization
- [ ] 3 core evaluation metrics

### 🔄 Phase 2: Quality & Evaluation (Weeks 3-4)
- [ ] LLM-as-judge evaluation
- [ ] Prompt optimization based on metrics
- [ ] Caching layer (Redis)
- [ ] Error handling & logging
- [ ] Integration tests

### 🚀 Phase 3: Production (Month 2)
- [ ] CI/CD pipeline
- [ ] Deployment (Railway/Render)
- [ ] Monitoring & observability
- [ ] Rate limiting
- [ ] Documentation site

### 🔮 Phase 4: Advanced Features (Future)
- [ ] Multiple LLM provider support
- [ ] Historical risk tracking
- [ ] Comparative REIT analysis
- [ ] Custom risk categories
- [ ] User authentication

---

## 🎓 Learning Resources

### For Understanding This Project
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [SEC EDGAR Guide](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm)
- [Sentence Transformers](https://www.sbert.net/)

### For LLM Evaluation
- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/)
- [Anthropic Evaluation Best Practices](https://docs.anthropic.com/claude/docs/evaluations)
- [Braintrust AI Evals](https://www.braintrustdata.com/)

### For Production ML Systems
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen
- [Building LLM Apps for Production](https://huyenchip.com/2023/04/11/llm-engineering.html)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Project structure** inspired by FastAPI production best practices
- **Evaluation framework** based on modern LLM evaluation patterns
- **Golden dataset** curated from actual SEC 10-K filings
- **Architecture design** influenced by Clean Architecture and Domain-Driven Design

---

## 📬 Contact & Support

- **Issues:** [GitHub Issues](your-repo/issues)
- **Discussions:** [GitHub Discussions](your-repo/discussions)
- **Email:** your.email@example.com

---

**⭐ If this project helped you, consider starring it on GitHub!**
