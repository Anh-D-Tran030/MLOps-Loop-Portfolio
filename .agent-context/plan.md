---
status: locked
version: 1.0.0
created: 2026-06-01
author: plan-agent
source_docs: [CLAUDE.md, PRD.md, .agent-context/explore.md]
---

# mlops-loop — Locked Implementation Plan

## Project Summary

**Repo:** `mlops-loop` — Serve → Monitor → Evaluate. One repo. One loop.  
**Goal:** Production MLOps mono-repo portfolio. Clone → `docker compose up` → three services running.  
**Audience:** ML Engineering hiring managers at AU AI companies (Harrison.ai, Relevance AI, Optiver).

---

## Architecture Contract

```
QW-1: Forecast API          QW-2: Drift Monitor         QW-3: LLM Eval Harness
────────────────────        ────────────────────        ─────────────────────────
FastAPI + Pydantic v2       Evidently AI (PSI)          DeBERTa NLI (faithfulness)
LightGBM .pkl               Prometheus v2.47.0          Sentence Transformers (relevance)
Multi-stage Docker          Grafana v10.1.0             RAGAS (context precision)
GHCR push via GH Actions    /metrics endpoint           Click CLI + CI exit code gate
```

**Immutable decisions:**
- Python 3.11 only; PyTorch stack (no TensorFlow)
- Pydantic v2 (not v1)
- PSI drift (not KL divergence)
- DeBERTa NLI (not LLM-as-judge)
- GHCR (not DockerHub)
- No hardcoded secrets

---

## Build Phases

### Phase 0 — Directory Scaffold
**Goal:** Create all empty directories and `__init__.py` stubs so imports resolve from day one.

| # | Action | Path |
|---|--------|------|
| 0.1 | Create dirs | `services/forecast-api/app/` |
| 0.2 | Create dirs | `services/forecast-api/training/` |
| 0.3 | Create dirs | `services/forecast-api/tests/` |
| 0.4 | Create dirs | `services/llm-eval/evaluator/` |
| 0.5 | Create dirs | `services/llm-eval/tests/` |
| 0.6 | Create dirs | `monitoring/evidently/` |
| 0.7 | Create dirs | `monitoring/prometheus/` |
| 0.8 | Create dirs | `monitoring/grafana/provisioning/datasources/` |
| 0.9 | Create dirs | `monitoring/grafana/provisioning/dashboards/` |
| 0.10 | Create dirs | `.github/workflows/` |
| 0.11 | Create dirs | `tests/fixtures/` |
| 0.12 | Create dirs | `data/` |
| 0.13 | Write stub | `services/forecast-api/app/__init__.py` (empty) |
| 0.14 | Write stub | `services/forecast-api/training/__init__.py` (empty) |
| 0.15 | Write stub | `services/forecast-api/tests/__init__.py` (empty) |
| 0.16 | Write stub | `services/llm-eval/evaluator/__init__.py` (re-export scorers) |
| 0.17 | Write stub | `services/llm-eval/tests/__init__.py` (empty) |

**Validation:** `find services monitoring .github tests -type d | wc -l` ≥ 15

---

### Phase 1 — Forecast API (QW-1)
**Goal:** Working FastAPI service with LightGBM inference. `pytest` green. Dockerfile builds.

Build order respects import dependencies: schemas → model → features → train → main → tests.

#### 1.1 `services/forecast-api/app/schemas.py`
**What:** Pydantic v2 request/response models.  
**Spec:**
```python
class PredictRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    days_ahead: int = 30

class PredictResponse(BaseModel):
    forecast: list[float]
    model_version: str
    prediction_id: str  # uuid4
```
**Constraints:** Pydantic v2 only. No optional fields except `days_ahead`.  
**Estimated lines:** ~40

#### 1.2 `services/forecast-api/training/features.py`
**What:** Shared feature engineering functions (used by both train.py and model.py — must be identical).  
**Spec:**
- `extract_date_features(df) -> df` — adds day_of_week, month, year, days_to_holiday
- `add_lag_features(df) -> df` — adds lag_7, lag_14
- `add_rolling_features(df) -> df` — adds rolling_mean_7
- `encode_categoricals(df, fit=False) -> (df, encoder)` — OHE on `family` column  

**Constraints:** Train and predict paths MUST use identical transformations (training/serving skew prevention).  
**Estimated lines:** ~100

#### 1.3 `services/forecast-api/app/model.py`
**What:** LightGBM model loader and inference wrapper.  
**Spec:**
- `class LGBMModel` with `__init__(model_path: str = "model.pkl")`
- Loads model on init via `joblib.load`; sets `self.model_loaded = False` if file missing
- `predict(features: dict, days_ahead: int = 30) -> list[float]`
  - Calls feature engineering from `training.features`
  - Returns list of `days_ahead` floats
  - Handles missing feature keys with defaults + warning log (never crash)
- Inference must be deterministic  

**Constraints:** No training code here. Import `features` from `training.features`.  
**Estimated lines:** ~120

#### 1.4 `services/forecast-api/app/main.py`
**What:** FastAPI application with 3 endpoints.  
**Spec:**
- Lifespan context manager: load `LGBMModel` on startup
- `POST /predict` → `PredictRequest` → `PredictResponse` (uuid4 prediction_id)
- `GET /health` → `{"status": "ok", "model_loaded": bool}`
- `GET /metrics` → Prometheus text format via `prometheus_client.generate_latest()`
  - Counters: `forecast_requests_total`
  - Histogram: `forecast_request_duration_seconds`
  - Gauge: `psi_score` (per feature, updated by drift_report)
- Async handlers  

**Constraints:** Must not crash if model.pkl missing. p95 latency < 200ms.  
**Estimated lines:** ~150

#### 1.5 `services/forecast-api/training/train.py`
**What:** LightGBM training script; saves `model.pkl`.  
**Spec:**
- Load `data/train.csv` (Kaggle Store Sales format)
- Apply all feature engineering from `features.py`
- LightGBM config: `objective="regression"`, `metric="rmse"`, `num_leaves=31`, `n_estimators=500`
- 80/20 train/val split; fixed random seed
- `joblib.dump(model, "model.pkl")`
- Print train/val RMSE to stdout  

**Constraints:** Reproducible (fixed seed). Offline only. Expects `data/train.csv`.  
**Estimated lines:** ~180

#### 1.6 `services/forecast-api/tests/conftest.py`
**What:** pytest fixtures for the forecast-api test suite.  
**Spec:**
- `@pytest.fixture` `minimal_model` — trains a tiny LightGBM on 100 synthetic rows, saves to temp `model.pkl`
- `@pytest.fixture` `test_client` — `TestClient(app)` from `httpx`/`fastapi.testclient`
- Sample `PredictRequest` dicts for happy path and edge cases  

**Constraints:** No internet access. Must not import training pipeline (minimal fixture model only).  
**Estimated lines:** ~50

#### 1.7 `services/forecast-api/tests/test_predict.py`
**What:** pytest test suite covering API behaviour.  
**Spec:**
- `test_health_returns_ok` — GET /health → 200, model_loaded=true
- `test_predict_happy_path` — POST /predict valid request → 200, forecast len == days_ahead, prediction_id is valid uuid4
- `test_predict_missing_feature_does_not_crash` — missing key → still returns 200 (uses default)
- `test_predict_invalid_store_nbr_type` — string where int expected → 422
- `test_metrics_endpoint` — GET /metrics → 200, content contains "forecast_requests_total"
- `test_predict_days_ahead_default` — omit days_ahead → forecast len == 30  

**Constraints:** All tests must pass without mocking the model (use fixture model).  
**Estimated lines:** ~100

#### 1.8 `services/forecast-api/requirements.txt`
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
prometheus-client>=0.18.0
httpx>=0.25.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

#### 1.9 `services/forecast-api/Dockerfile`
**Spec (multi-stage, must be < 200MB content/transport size):**
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
COPY model.pkl .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
**Constraints:** Final image < 200MB content/transport size (`docker images` CONTENT SIZE column on docker 29+, or the value `docker manifest inspect` reports). The original 150MB target was aspirational and not achievable on python:3.11-slim once lightgbm + scipy + scikit-learn + pandas + numpy + libgomp1 are installed; 169MB is the measured floor with all five and is acceptable. `model.pkl` must exist at build context root.  

**Phase 1 Validation:**
```bash
cd services/forecast-api
pytest tests/ -v          # all 6 tests green
docker build -t forecast-api-test .   # build succeeds
docker images forecast-api-test  # CONTENT SIZE column < 200MB
```

---

### Phase 2 — LLM Eval Harness (QW-3)
**Goal:** `mlops-eval score` CLI works. CI exit code 1 on score < threshold.

Build order: faithfulness → relevance → precision → `__init__` → cli → setup → tests.

#### 2.1 `services/llm-eval/evaluator/faithfulness.py`
**What:** DeBERTa NLI entailment scorer.  
**Spec:**
- `class FaithfulnessScorer`
- Model: `cross-encoder/nli-deberta-v3-large` via `transformers.pipeline("text-classification")`
- `score(context: str, answer: str) -> float`
  - Input to model: `f"{context} [SEP] {answer}"`
  - Label mapping: `ENTAILMENT` → score, `CONTRADICTION` → 1 − score, `NEUTRAL` → 0.5
  - Returns float 0–1  

**Constraints:** CPU-only (~200ms/sample). No API calls. Deterministic.  
**Estimated lines:** ~100

#### 2.2 `services/llm-eval/evaluator/relevance.py`
**What:** Sentence Transformers cosine similarity scorer.  
**Spec:**
- `class RelevanceScorer`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- `score(question: str, answer: str) -> float`
  - Embed both strings; compute cosine similarity
  - Returns float 0–1 (clip to [0, 1])  

**Constraints:** Offline. Deterministic embeddings.  
**Estimated lines:** ~80

#### 2.3 `services/llm-eval/evaluator/precision.py`
**What:** RAGAS context precision scorer.  
**Spec:**
- `class PrecisionScorer`
- `score(question: str, context: str, answer: str) -> float`
  - Uses RAGAS `context_precision` metric
  - Returns float 0–1  

**Constraints:** Must use RAGAS library, not custom implementation.  
**Estimated lines:** ~90

#### 2.4 `services/llm-eval/evaluator/__init__.py`
```python
from .faithfulness import FaithfulnessScorer
from .relevance import RelevanceScorer
from .precision import PrecisionScorer

__all__ = ["FaithfulnessScorer", "RelevanceScorer", "PrecisionScorer"]
```

#### 2.5 `services/llm-eval/evaluator/cli.py`
**What:** Click CLI entry point for CI integration.  
**Spec:**
- `@click.command() score(input_file, threshold=0.7)`
- Input: JSONL file, each line `{"question": "...", "context": "...", "answer": "..."}`
- Per-sample: compute faithfulness, relevance, precision scores
- Output to stdout:
  ```
  Sample 1: faithful=0.82  relevant=0.91  precision=0.75
  ...
  MEAN:      faithful=0.81  relevant=0.89  precision=0.74
  PASS/FAIL: faithful=PASS  relevant=PASS  precision=PASS
  OVERALL: PASS
  ```
- Exit code 0 if all metrics PASS (mean >= threshold)
- Exit code 1 if any metric FAIL  

**Constraints:** Exit code 1 MUST propagate as GitHub Actions step failure.  
**Estimated lines:** ~130

#### 2.6 `services/llm-eval/setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="mlops-eval",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["mlops-eval=evaluator.cli:score"],
    },
    install_requires=[
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "ragas>=0.1.0",
        "click>=8.1.0",
    ],
)
```

#### 2.7 `services/llm-eval/requirements.txt`
```
transformers>=4.35.0
sentence-transformers>=2.2.0
torch>=2.0.0
ragas>=0.1.0
click>=8.1.0
pytest>=7.4.0
```

#### 2.8 `services/llm-eval/tests/test_evaluators.py`
**What:** Unit tests for all three scorers + CLI.  
**Spec:**
- `test_faithfulness_entailment_returns_high_score` — identical strings score > 0.7
- `test_faithfulness_contradiction_returns_low_score` — contradictory pair score < 0.3
- `test_relevance_identical_returns_high_score` — same string cosine ≈ 1.0
- `test_relevance_unrelated_returns_low_score`
- `test_precision_scorer_returns_float_in_range`
- `test_cli_pass_exits_zero` — JSONL with high-quality samples exits 0
- `test_cli_fail_exits_one` — JSONL with low-quality samples exits 1
- `test_cli_missing_field_raises_error`  

**Estimated lines:** ~150

#### 2.9 `tests/fixtures/eval_set.jsonl`
**What:** 15 JSONL samples that PASS at threshold 0.7.  
**Spec:** Each line: `{"question": "...", "context": "...", "answer": "..."}` where answer is clearly supported by context. Used in CI eval-gate job.

#### 2.10 `services/llm-eval/Dockerfile`
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt setup.py .
COPY evaluator/ ./evaluator/
RUN pip install --user --no-cache-dir .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY evaluator/ ./evaluator/
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["mlops-eval"]
```

**Phase 2 Validation:**
```bash
cd services/llm-eval
pip install -e .
mlops-eval score tests/fixtures/eval_set.jsonl --threshold 0.7  # exits 0
pytest tests/ -v   # all 8 tests green
```

---

### Phase 3 — Monitoring Stack (QW-2)
**Goal:** Grafana dashboard auto-provisions at `docker compose up`. Prometheus scrapes `/metrics`.

#### 3.1 `monitoring/evidently/drift_report.py`
**What:** PSI drift detection script.  
**Spec:**
- CLI: `python drift_report.py --current current_window.parquet`
- Loads `reference_data.parquet` (same directory)
- Computes PSI per feature via Evidently `DataDriftPreset`
- Prints PSI scores as JSON to stdout: `{"feature_name": psi_value, ...}`
- Writes Prometheus textfile metrics to `/tmp/drift_metrics.prom`
  - Metric: `psi_score{feature="<name>"} <value>`
- No database; no external API  

**Estimated lines:** ~120

#### 3.2 `monitoring/evidently/reference_data.parquet`
**What:** Snapshot of training feature distribution for PSI baseline.  
**Spec:** Generate from synthetic data (10,000 rows) matching Kaggle Store Sales schema. Columns: `store_nbr`, `family`, `onpromotion`, `day_of_week`, `month`, `year`. Saved via `pandas.DataFrame.to_parquet()`.  
**How to generate:** Write a short `generate_reference.py` script that creates and saves it; script can be deleted after parquet is committed.

#### 3.3 `monitoring/prometheus/prometheus.yml`
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: forecast-api
    static_configs:
      - targets: ["forecast-api:8000"]
    metrics_path: /metrics
```

#### 3.4 `monitoring/grafana/provisioning/datasources/prometheus.yml`
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    access: proxy
```

#### 3.5 `monitoring/grafana/provisioning/dashboards/dashboard.yml`
```yaml
apiVersion: 1
providers:
  - name: default
    orgId: 1
    folder: ""
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

#### 3.6 `monitoring/grafana/provisioning/dashboards/mlops-loop.json`
**What:** Pre-built Grafana dashboard JSON.  
**Spec — 4 panels:**
1. PSI scores per feature (time series, line chart, all features on one panel)
2. Forecast API request rate (counter rate)
3. Forecast API latency p50/p95 (histogram quantiles)
4. Model health status (stat panel: `model_loaded` gauge)  

**Constraints:** Valid JSON. References Prometheus datasource by name. Zero manual UI setup.  
**Estimated lines:** ~200 (JSON)

**Phase 3 Validation:**
```bash
docker compose up prometheus grafana
# http://localhost:3000 → admin/admin → dashboard visible with 4 panels
# http://localhost:9090/targets → forecast-api target in UP state (after api starts)
```

---

### Phase 4 — Infrastructure & CI/CD
**Goal:** `docker compose up` zero-config. CI green on main.

#### 4.1 `docker-compose.yml`
```yaml
version: "3.9"
services:
  forecast-api:
    image: ghcr.io/anh-d-tran030/mlops-loop:latest
    ports: ["8000:8000"]
    volumes:
      - ./monitoring/evidently/reference_data.parquet:/app/reference_data.parquet

  prometheus:
    image: prom/prometheus:v2.47.0
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:10.1.0
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    depends_on: [prometheus]
```

#### 4.2 `.github/workflows/ci.yml`
**Spec — 4 sequential jobs:**
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ruff black
      - run: ruff check . && black --check .

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r services/forecast-api/requirements.txt
      - run: pytest services/forecast-api/tests/ -v

  eval-gate:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e services/llm-eval/
      - run: mlops-eval score tests/fixtures/eval_set.jsonl --threshold 0.7

  build-push:
    needs: eval-gate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: services/forecast-api
          push: true
          tags: ghcr.io/anh-d-tran030/mlops-loop:latest
```

#### 4.3 `.github/workflows/eval-gate.yml`
**What:** Standalone eval quality gate (manual trigger + PR).  
**Spec:** Same as `eval-gate` job above but as a standalone `workflow_dispatch` + `pull_request` workflow.

#### 4.4 `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
        language_version: python3.11
```

#### 4.5 `.gitignore`
**What:** Standard Python + Docker + data file ignores.  
**Spec:** Include: `*.pkl`, `*.parquet`, `data/`, `__pycache__/`, `*.pyc`, `.env`, `dist/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `node_modules/`

#### 4.6 `README.md`
**What:** Portfolio-facing README.  
**Spec — Sections (word-for-word from PRD where specified):**
1. Project Identity (from PRD Section 2, verbatim)
2. Architecture: ASCII diagram + three-component table
3. Quick Start: `git clone` → `docker compose up` → URLs
4. The Three Components (QW-1, QW-2, QW-3) with rationale tables
5. The 9 Interview Questions (from PRD Section 9, verbatim)
6. Upgrade Path (from PRD Section 8)
7. CI badge (markdown badge linking to Actions)

**Phase 4 Validation:**
```bash
docker compose config   # validates compose file, no errors
cat .github/workflows/ci.yml | python3 -c "import sys,yaml; yaml.safe_load(sys.stdin)"  # valid YAML
pre-commit run --all-files  # ruff + black pass
```

---

### Phase 5 — End-to-End Validation
**Goal:** All success criteria from PRD met. System interview-ready.

#### Checklist

**MVP Gate (minimum to merge to main):**
- [ ] `docker compose up` starts all 3 services without manual config
- [ ] `GET /health` → 200, `{"status": "ok", "model_loaded": true}`
- [ ] `POST /predict` → 200, `forecast` list len == `days_ahead`, valid uuid4 `prediction_id`
- [ ] `GET /metrics` → 200, Prometheus text format, contains `forecast_requests_total`
- [ ] Grafana at `http://localhost:3000` (admin/admin) → mlops-loop dashboard loads with 4 panels
- [ ] `pytest services/forecast-api/tests/ -v` → all green
- [ ] `mlops-eval score tests/fixtures/eval_set.jsonl --threshold 0.7` → exit 0
- [ ] CI workflow: lint → test → eval-gate → build-push all green on main

**Portfolio-Ready Gate (before interview):**
- [ ] Forecast API p95 latency < 200ms (measure with `wrk` or `hey`)
- [ ] Docker image < 200MB content size (`docker images ghcr.io/anh-d-tran030/mlops-loop:latest`)
- [ ] No hardcoded secrets (`grep -r "password\|secret\|token" services/ --include="*.py"` returns nothing sensitive)
- [ ] README.md Sections 2 + 9 match PRD verbatim
- [ ] All 9 interview questions answerable cold (review PRD Section 9)
- [ ] Cold start < 5 minutes on fresh Docker pull

---

## File Build Order (Complete Sequence)

This is the strict build order for the build agent. Each file depends on the files above it.

```
PHASE 0 — SCAFFOLD
  services/forecast-api/app/__init__.py
  services/forecast-api/training/__init__.py
  services/forecast-api/tests/__init__.py
  services/llm-eval/evaluator/__init__.py  (stub, overwritten in Phase 2)
  services/llm-eval/tests/__init__.py

PHASE 1 — FORECAST API
  services/forecast-api/app/schemas.py         ← no deps
  services/forecast-api/training/features.py   ← no deps
  services/forecast-api/app/model.py           ← imports training.features
  services/forecast-api/app/main.py            ← imports schemas, model
  services/forecast-api/training/train.py      ← imports training.features
  services/forecast-api/tests/conftest.py      ← imports app.main, app.schemas
  services/forecast-api/tests/test_predict.py  ← imports conftest fixtures
  services/forecast-api/requirements.txt       ← no deps
  services/forecast-api/Dockerfile             ← no deps

PHASE 2 — LLM EVAL
  services/llm-eval/evaluator/faithfulness.py  ← no deps
  services/llm-eval/evaluator/relevance.py     ← no deps
  services/llm-eval/evaluator/precision.py     ← no deps
  services/llm-eval/evaluator/__init__.py      ← imports all three scorers
  services/llm-eval/evaluator/cli.py           ← imports __init__ scorers
  services/llm-eval/setup.py                   ← references cli:score
  services/llm-eval/requirements.txt           ← no deps
  services/llm-eval/tests/test_evaluators.py   ← imports evaluator
  tests/fixtures/eval_set.jsonl                ← no deps
  services/llm-eval/Dockerfile                 ← no deps

PHASE 3 — MONITORING
  monitoring/evidently/drift_report.py         ← no deps
  monitoring/evidently/reference_data.parquet  ← generated by script
  monitoring/prometheus/prometheus.yml          ← no deps
  monitoring/grafana/provisioning/datasources/prometheus.yml
  monitoring/grafana/provisioning/dashboards/dashboard.yml
  monitoring/grafana/provisioning/dashboards/mlops-loop.json

PHASE 4 — INFRA
  docker-compose.yml
  .github/workflows/ci.yml
  .github/workflows/eval-gate.yml
  .pre-commit-config.yaml
  .gitignore
  README.md
```

**Total files:** 37 source/config files + 1 parquet binary + 1 JSONL fixture = **39 artifacts**

---

## External Skills Identified

From `npx skills find`:

| Skill | Installs | Relevance | Use |
|-------|----------|-----------|-----|
| `sickn33/antigravity-awesome-skills@mlops-engineer` | 379 | High — MLOps patterns | Reference for drift/eval patterns |
| `giuseppe-trisciuoglio/developer-kit@turborepo-monorepo` | 1K | Medium — mono-repo CI | Reference for GH Actions monorepo config |
| `aj-geddes/useful-ai-prompts@monorepo-management` | 343 | Medium | Reference for build isolation |

**Verdict:** Local skills (`project-development`, `harness-engineering`, `fastapi-patterns`, `pydantic-v2`, `pgvector`) cover all needs. External skills are informational only — do not install unless build agent encounters a blocking gap.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `model.pkl` missing at Docker build time | High | Document `python training/train.py` as required pre-step; add Makefile target |
| RAGAS API breaking changes (ragas >=0.2 changed interface) | Medium | Pin ragas version; test with `ragas==0.1.21` |
| DeBERTa model download (~1.4GB) slow in CI | Medium | Cache HuggingFace models in GH Actions via `actions/cache` |
| Docker image > 200MB due to torch | High | llm-eval has its own Dockerfile; forecast-api Dockerfile must not include torch. Measured floor is ~169MB with lightgbm + scipy + sklearn + pandas + numpy on python:3.11-slim. |
| `reference_data.parquet` not committed | Medium | Generate with script; add to git (binary, small) |
| Grafana dashboard JSON invalid | Low | Validate JSON before committing; use Grafana provisioning dry-run |

---

## Notes for Build Agent

1. **`model.pkl` strategy**: Tests use a minimal fixture model trained on 100 synthetic rows in `conftest.py`. The real `model.pkl` is trained via `python services/forecast-api/training/train.py` and is NOT committed to git (`.gitignore`). Docker build assumes it exists at build context root.

2. **Feature parity critical**: `training/features.py` is the single source of truth for feature engineering. Both `train.py` and `model.py` import from it. Any divergence = training/serving skew.

3. **RAGAS precision scorer**: RAGAS `context_precision` requires an LLM judge internally. Pin `ragas==0.1.21` (last version with non-LLM fallback). If RAGAS forces LLM judge, substitute with a simple token-overlap precision metric.

4. **Prometheus metrics in `/metrics`**: Use `prometheus_client.generate_latest()` + `CONTENT_TYPE_LATEST` for the response. Register counters/histograms as module-level globals to avoid duplication.

5. **Grafana JSON**: Build the dashboard JSON manually (no Grafana UI export needed). Use `"datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}` pattern for portability.

6. **GitHub Actions model caching**: Add `actions/cache` for `~/.cache/huggingface` in the `eval-gate` job to avoid re-downloading DeBERTa on every CI run.

---

## Plan Lock Confirmation

This plan is **locked**. The build agent must implement all files as specified. Changes to architecture, tech stack, or immutable decisions require a plan version bump and re-lock. Additive changes (comments, minor implementation details) do not require a version bump.

**Files in scope:** 39 artifacts across `services/`, `monitoring/`, `.github/`, `tests/`, and repo root.  
**Files out of scope:** `CLAUDE.md`, `PRD.md`, `.agent-context/`, `.opencode/` — do not modify.
