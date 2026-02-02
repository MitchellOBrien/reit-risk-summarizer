# CI/CD & Evaluation Workflows

This directory contains GitHub Actions workflows for automated testing, deployment, and quality monitoring.

## Overview

The project uses two main workflows:

1. **`ci-cd.yml`** - Continuous Integration and Deployment
2. **`scheduled-evaluation.yml`** - Weekly Quality Monitoring

```
┌─────────────────────────────────────────────────────────┐
│  Scheduled Workflow (Weekly)                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Run evaluation.run_evaluation                 │  │
│  │ 2. Generate evaluation_results.json              │  │
│  │ 3. Upload as artifact                            │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Creates/Updates
                        ▼
              evaluation_results.json
                        │
                        │ Read by
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Main CI/CD (Every Push)                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Run tests                                     │  │
│  │ 2. Build Docker image                            │  │
│  │ 3. Deploy FastAPI app (includes /evaluation)    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Deploys
                        ▼
       FastAPI App with /evaluation endpoints
                        │
                        │ Serves metrics via HTTP
                        ▼
              GET /evaluation/results
              GET /evaluation/metrics/{ticker}
```

---

## Workflow 1: CI/CD Pipeline (`ci-cd.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual trigger via `workflow_dispatch`

**Environment:**
- Python 3.13
- uv 0.5.14 (fast package manager)
- Ubuntu latest

### Jobs

#### 1. **Test** (Runs on all triggers)

**Purpose:** Validate code quality and functionality

**Steps:**
1. Checkout code
2. Set up Python 3.13
3. Install uv package manager
4. Cache dependencies (`.venv` and `~/.cache/uv`)
5. Install dependencies: `uv pip install -e ".[dev,test]"`
6. Run linting: `ruff check src/ tests/ evaluation/`
7. Run type checking: `mypy src/ evaluation/`
8. Run unit tests: `pytest tests/unit/ -v --cov`
9. Run evaluation pipeline tests: `pytest tests/unit/test_evaluation_pipeline.py`
10. Upload coverage to Codecov

**Quality Gates:**
- Unit tests must pass (hard requirement)
- Linting errors don't fail build (yet) - `continue-on-error: true`
- Type errors don't fail build (yet) - `continue-on-error: true`

**Why soft failures for linting/typing?**
- Allows iterative improvement
- Won't block urgent fixes
- Can be tightened later

#### 2. **Integration Test** (Runs after test job)

**Purpose:** Test end-to-end flows

**Steps:**
1. Checkout code
2. Set up Python and dependencies
3. Run integration tests: `pytest tests/integration/ -v`

**Note:** `continue-on-error: true` because integration tests may not exist yet

#### 3. **Docker Build** (Runs on push to main)

**Purpose:** Build and publish production Docker images

**Steps:**
1. Checkout code
2. Set up Docker Buildx (multi-platform builds)
3. Log in to Docker Hub (uses secrets)
4. Build and push image with tags:
   - `latest` (latest stable version)
   - `{git-sha}` (specific commit version)
5. Use GitHub Actions cache for faster builds

**Secrets Required:**
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token

**Conditional Execution:**
- Only runs on push to `main` branch
- Skips gracefully if Docker credentials not configured

#### 4. **Security Scan** (Runs after test job)

**Purpose:** Identify security vulnerabilities

**Steps:**
1. Run `safety check` - scans dependencies for known CVEs
2. Run `bandit` - static security analysis for Python
3. Upload reports as artifacts

**Note:** Doesn't fail build (yet) - provides visibility

---

## Workflow 2: Scheduled Evaluation (`scheduled-evaluation.yml`)

**Triggers:**
- **Scheduled:** Every Monday at 9:00 AM UTC (`cron: '0 9 * * 1'`)
- **Manual:** Via workflow_dispatch with option to regenerate

**Purpose:** Automated quality monitoring of LLM outputs

### Jobs

#### **Run Evaluation**

**Steps:**
1. Checkout code
2. Set up Python and dependencies
3. Set environment variables from secrets:
   - `GROQ_API_KEY` - For LLM API access
   - `SEC_API_USER_AGENT` - For SEC EDGAR API
4. **Run evaluation** (conditional based on mode):
   - **Cached-only** (default): `python -m evaluation.run_evaluation --cached-only`
   - **Regenerate** (manual): `python -m evaluation.run_evaluation --regenerate`
5. Generate evaluation report (Python script)
6. Upload artifacts:
   - `evaluation/results/evaluation_results.json`
   - `evaluation/results/evaluation_report.txt`

**Why two modes?**
- **Cached-only**: Fast, no API costs, validates pipeline
- **Regenerate**: Fresh LLM outputs, validates current model performance

**Secrets Required:**
- `GROQ_API_KEY` - Groq API key for LLM access
- `SEC_API_USER_AGENT` - SEC requires user agent with contact info

### Production vs. Current Implementation

> **Note:** The current implementation uses a **static golden dataset** (10 fixed tickers) for simplicity. In a real production system, the workflow would be different:

**Production Approach:**
1. **Continuous Labeling**: Experts continuously add labels for new tickers via labeling interface or database
2. **Dynamic Dataset**: Scheduled evaluation loads **latest** expert-labeled data (not static CSV)
3. **Growing Coverage**: Dataset expands over time (Week 1: 10 tickers → Week 20: 100 tickers)
4. **Version Tracking**: Each evaluation run documents which tickers were evaluated and when

**Production Flow:**
```
Expert Labels New Ticker → Store in Database/Git → Scheduled Evaluation Loads Latest → 
Metrics Computed on All Available Labels → Track Performance Over Time
```

**Benefits:**
- Continuous improvement with more diverse evaluation data
- Detect model degradation when new sectors/tickers added
- Enable A/B testing on same growing dataset
- Better representation across REIT sectors

**Implementation Options:**
- Database-backed labels (PostgreSQL, BigQuery) with versioning
- Git-based labels (commit new CSV rows, pull latest before evaluation)
- Hybrid (database for UI, export to CSV for evaluation)

### Evaluation Output

**Generated Files:**
```
evaluation/
├── results/
│   └── evaluation_results.json  # Full metrics + per-ticker results
└── golden_outputs/
    ├── AMT.json                 # Cached LLM output for American Tower
    ├── PLD.json                 # Cached LLM output for Prologis
    └── EQIX.json                # Cached LLM output for Equinix
```

**Results Structure:**
```json
{
  "metadata": {
    "run_date": "2026-01-28T09:00:00Z",
    "total_tickers": 10,
    "cached_only_mode": true
  },
  "tickers": [
    {
      "ticker": "AMT",
      "company_name": "American Tower",
      "sector": "Infrastructure/Towers",
      "metrics": {
        "semantic_similarity": 0.82,
        "ndcg_at_5": 0.78,
        "sector_specificity": 0.65
      },
      "source": "cached_golden_output"
    }
  ],
  "summary_metrics": {
    "semantic_similarity": {"mean": 0.75, "min": 0.55, "max": 1.0},
    "ndcg_at_5": {"mean": 0.90, "min": 0.79, "max": 1.0},
    "sector_specificity": {"mean": 0.62, "min": 0.54, "max": 0.73}
  }
}
```

---

## How the API Uses Evaluation Results

The FastAPI application exposes evaluation metrics via REST endpoints:

```python
# src/reit_risk_summarizer/routers/evaluation.py

@router.get("/evaluation/results")
async def get_evaluation_results():
    """Read evaluation_results.json and serve via API"""
    # Loads pre-computed metrics (no LLM calls)
    # Fast response time (~10ms)

@router.get("/evaluation/metrics/{ticker}")
async def get_ticker_metrics(ticker: str):
    """Get metrics for specific ticker (AMT, PLD, etc.)"""
```

**Key Point:** The API doesn't **run** evaluation, it just **serves** pre-computed results. This separation means:
- API stays fast (no expensive LLM calls)
- Evaluation runs independently on schedule
- Can manually trigger evaluation when needed

---

## Setting Up Secrets

### GitHub Repository Secrets

Navigate to: **Settings → Secrets and variables → Actions → New repository secret**

**Required Secrets:**

| Secret Name | Description | Example | Where to Get |
|------------|-------------|---------|--------------|
| `GROQ_API_KEY` | Groq API key for LLM | `gsk_abc123...` | https://console.groq.com/ |
| `SEC_API_USER_AGENT` | SEC EDGAR user agent | `YourName your@email.com` | Your name + email |
| `DOCKER_USERNAME` | Docker Hub username | `youruser` | https://hub.docker.com/ |
| `DOCKER_PASSWORD` | Docker Hub token | `dckr_pat_...` | Docker Hub → Account Settings → Security |

**Optional Secrets (for production):**
- `CODECOV_TOKEN` - For coverage uploads
- `SENTRY_DSN` - For error tracking
- `SLACK_WEBHOOK` - For notifications

### Local Development

For local testing, create `.env`:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Never commit `.env` to git!** (It's in `.gitignore`)

---

## Manual Workflow Triggers

### Trigger Scheduled Evaluation Manually

**Via GitHub UI:**
1. Go to **Actions** tab
2. Select **Scheduled Evaluation** workflow
3. Click **Run workflow**
4. Choose branch (usually `main`)
5. Check **Regenerate** if you want fresh LLM outputs
6. Click **Run workflow**

**Via GitHub CLI:**
```bash
# Run with cached outputs only
gh workflow run scheduled-evaluation.yml

# Run with regeneration
gh workflow run scheduled-evaluation.yml -f regenerate=true
```

### Trigger CI/CD Manually

```bash
gh workflow run ci-cd.yml
```

---

## Troubleshooting

### Common Issues

#### **1. Tests Fail in CI but Pass Locally**

**Possible causes:**
- Different Python version (CI uses 3.13)
- Missing environment variables
- Dependency version mismatch

**Solutions:**
```bash
# Use same Python version locally
pyenv install 3.13.0
pyenv local 3.13.0

# Sync exact dependencies
uv sync

# Run tests with same flags as CI
pytest tests/unit/ -v --cov
```

#### **2. Docker Build Fails**

**Possible causes:**
- Dockerfile syntax error
- Missing secrets
- Build context too large

**Solutions:**
```bash
# Test Docker build locally
docker build -t reit-risk-summarizer .

# Check .dockerignore excludes large files
cat .dockerignore
```

#### **3. Evaluation Workflow Fails**

**Possible causes:**
- Missing `GROQ_API_KEY` secret
- API rate limit exceeded
- No cached outputs (in cached-only mode)

**Solutions:**
- Verify secrets are set in GitHub
- Check Groq API usage limits
- Run with `regenerate=true` to generate fresh outputs

#### **4. Coverage Upload Fails**

**Non-critical** - marked with `continue-on-error: true`

**Causes:**
- Codecov token not configured
- Network timeout

**Solutions:**
- Add `CODECOV_TOKEN` secret
- Or ignore (doesn't block deployment)

---

## Monitoring Workflows

### View Workflow Runs

**GitHub UI:**
- Go to **Actions** tab
- Select workflow (CI/CD or Scheduled Evaluation)
- View run history, logs, artifacts

**GitHub CLI:**
```bash
# List recent runs
gh run list --workflow=ci-cd.yml

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

### Artifacts

Each evaluation run uploads:
- `evaluation-results` - Contains `evaluation_results.json` and report
- `security-reports` - Bandit security scan results

**Retention:** 90 days (GitHub default)

**Download:**
```bash
# Download latest evaluation results
gh run download --name evaluation-results
```

---

## Optimization Tips

### Faster CI Runs

**1. Cache Dependencies Effectively**
```yaml
- uses: actions/cache@v4
  with:
    path: |
      .venv
      ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('**/pyproject.toml') }}
```

**2. Run Jobs in Parallel**
- Test, Security Scan, Integration Test run simultaneously
- Docker Build only after tests pass

**3. Skip Unnecessary Steps**
```yaml
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

### Cost Optimization

**GitHub Actions free tier:**
- 2,000 minutes/month for private repos
- Unlimited for public repos

**Reduce usage:**
- Use `continue-on-error` for optional steps
- Limit scheduled workflow frequency
- Cache aggressively

---

## Adding New Workflows

### Workflow Template

Create `.github/workflows/my-workflow.yml`:

```yaml
name: My Workflow

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  my-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      
      - name: Run my task
        run: |
          echo "Running..."
```

### Best Practices

1. **Use `workflow_dispatch`** - Enable manual triggers for debugging
2. **Cache dependencies** - Faster runs, lower costs
3. **Use secrets** - Never hardcode credentials
4. **Add `continue-on-error`** - For non-critical steps
5. **Upload artifacts** - Preserve outputs for review
6. **Use concise names** - Clear job/step descriptions

---

## Future Enhancements

### Planned Improvements

- [ ] **Deploy to Cloud Run** - Automatic deployment on merge to `main`
- [ ] **Slack Notifications** - Alert on workflow failures
- [ ] **Performance Benchmarks** - Track API latency over time
- [ ] **Multi-environment** - Staging and production deployments
- [ ] **Rollback Support** - Automated rollback on failure

### Advanced Patterns

**1. Matrix Testing**
```yaml
strategy:
  matrix:
    python-version: [3.11, 3.12, 3.13]
```

**2. Conditional Jobs**
```yaml
if: github.event.pull_request.draft == false
```

**3. Reusable Workflows**
```yaml
uses: ./.github/workflows/shared-tests.yml
```

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [uv Documentation](https://github.com/astral-sh/uv)
- [Docker Build Optimization](https://docs.docker.com/build/cache/)

---

## Questions?

File an issue or start a discussion in the main repository.
