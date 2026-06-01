---
generated: 2026-06-01T01:15:00Z
agent: explore
status: greenfield_verified
---

# mlops-loop — Project Exploration Snapshot (Verified)

## Project Identity

**Name:** mlops-loop  
**Tagline:** Serve → Monitor → Evaluate. One repo. One loop.  
**Author:** Anh Duc Tran (AI Engineering student, UTS Sydney)  
**GitHub:** https://github.com/Anh-D-Tran030/mlops-loop  
**Audience:** ML Engineering hiring managers at AU AI companies  
**Business Context:** Retail demand forecasting (Kaggle Store Sales dataset), framed as AI infrastructure for Vaylo (AI-driven CRM/inventory platform)

---

## Current Project State

**Status:** Greenfield — planning complete, zero source code yet written.

**Filesystem Scan (2026-06-01T01:15Z):**
- Total tracked files: 10
- Source files (Python/YAML/config): 0
- Documentation files: 3
- Hidden config files: 7
- Service directories: None created yet
- Monitoring configs: None created yet

**Existing Files:**
- `CLAUDE.md` — Scaffold spec (full implementation details per file)
- `PRD.md` — Product requirements (goals, architecture, 9 interview questions)
- `mlops-loop-portfolio.md` — Project narrative and portfolio context
- `AGENTS.md` → symlink to CLAUDE.md
- `.agent-context/{session_notes.md, explore.md, status.md, context_snapshot.md}` — Agent state
- `.claude/settings.local.json` — Empty permissions config
- `.opencode/package.json` — OpenCode config

**No Source Code Exists:**
- No `services/forecast-api/` directory
- No `services/llm-eval/` directory
- No `monitoring/` directory
- No `.github/workflows/` directory
- No `docker-compose.yml`
- No `.pre-commit-config.yaml`
- No `.gitignore`
- No `README.md`

---

## Architecture Overview

Three-phase MLOps loop, one mono-repo, three services:

```
QW-1: Forecast API      →  QW-2: Drift Monitor      →  QW-3: LLM Eval Harness
├─ LightGBM + FastAPI      ├─ Evidently + Prometheus   ├─ DeBERTa NLI + Sentence
├─ Multi-stage Docker      └─ Grafana                  └─ Transformers + CI gate
└─ GHCR push via GH Actions
```

### Phase 1: Serve (QW-1) — Forecast API
- **What:** REST API returning 30-day demand forecasts for retail store/product/date features
- **Model:** LightGBM (tabular + categorical encodings)
- **API:** FastAPI + Pydantic v2 (async, auto-docs, strict validation)
- **Container:** Multi-stage Dockerfile (builder ~120MB runtime)
- **Registry:** GHCR via GitHub Actions (native integration, no rate limits)
- **Endpoints:** `/predict` (POST), `/health` (GET), `/metrics` (GET)

### Phase 2: Monitor (QW-2) — Drift Detection
- **What:** Detects when incoming request distribution drifts from training distribution
- **Metric:** PSI (Population Stability Index) per feature
- **Detection Tool:** Evidently AI
- **Metrics Backend:** Prometheus (15s scrape interval)
- **Visualization:** Grafana provisioned via config (zero manual setup)
- **Rationale:** PSI symmetric with industry thresholds: <0.1 stable, 0.1–0.2 moderate, >0.2 significant

### Phase 3: Evaluate (QW-3) — Quality Gates
- **What:** CI gate scoring RAG/LLM pipeline on three quality dimensions; blocks Docker push if scores < threshold
- **Faithfulness:** DeBERTa-large NLI entailment scorer (cross-encoder/nli-deberta-v3-large)
- **Answer Relevance:** Sentence Transformers cosine similarity (all-MiniLM-L6-v2)
- **Context Precision:** RAGAS context precision metric
- **Interface:** Click CLI (`mlops-eval score <file> --threshold 0.7`)
- **Integration:** GitHub Actions workflow step (exit code 1 → workflow fails)

---

## Planned Directory Structure (100% Specification)

```
mlops-loop/
├── services/
│   ├── forecast-api/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py          # FastAPI app: /predict, /health, /metrics
│   │   │   ├── model.py         # LightGBM .pkl loader + inference wrapper
│   │   │   └── schemas.py       # Pydantic v2: PredictRequest, PredictResponse
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── train.py         # LightGBM training script → model.pkl
│   │   │   └── features.py      # Feature engineering: date parts, lags, OHE
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_predict.py  # Happy path, edge cases
│   │   │   └── conftest.py      # pytest fixtures
│   │   ├── Dockerfile           # Multi-stage builder → runtime
│   │   ├── requirements.txt
│   │   └── model.pkl (generated during training)
│   │
│   └── llm-eval/
│       ├── evaluator/
│       │   ├── __init__.py
│       │   ├── faithfulness.py  # DeBERTa-large NLI entailment scorer
│       │   ├── relevance.py     # Sentence Transformers cosine similarity
│       │   ├── precision.py     # RAGAS context precision
│       │   └── cli.py           # Click CLI: mlops-eval score <file> --threshold 0.7
│       ├── tests/
│       │   ├── __init__.py
│       │   └── test_evaluators.py
│       ├── Dockerfile
│       ├── setup.py             # pip install -e . entry point
│       └── requirements.txt
│
├── monitoring/
│   ├── evidently/
│   │   ├── drift_report.py      # PSI drift detection: reference vs current
│   │   └── reference_data.parquet  # Training feature distribution snapshot
│   ├── prometheus/
│   │   └── prometheus.yml       # Scrape config: forecast-api /metrics @15s
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── prometheus.yml
│           └── dashboards/
│               ├── dashboard.yml
│               └── mlops-loop.json  # Pre-built Grafana dashboard (PSI panels)
│
├── tests/
│   └── fixtures/
│       └── eval_set.jsonl       # Sample eval data for CI gate
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint → test → eval-gate → build → push
│       └── eval-gate.yml        # Standalone eval quality gate workflow
│
├── data/
│   └── train.csv                # Kaggle Store Sales dataset (expected location)
│
├── docker-compose.yml           # Services: forecast-api, prometheus, grafana
├── .pre-commit-config.yaml      # ruff + black hooks
├── .gitignore
└── README.md                    # Copy from PRD.md Section 2 verbatim
```

---

## Module Map (Planned Structure)

### Module: forecast-api
**Entry Points:**
- `app.main:app` — FastAPI application instance
- `training.train:main()` — Model training entry point

**Public Exports:**
- `schemas.PredictRequest` — Pydantic request schema
- `schemas.PredictResponse` — Pydantic response schema
- `model.LGBMModel` — Inference wrapper

**Files (Planned):**
- `app/main.py` — ~150 lines (async endpoint handlers)
- `app/model.py` — ~120 lines (model loader, inference)
- `app/schemas.py` — ~40 lines (Pydantic schemas)
- `training/train.py` — ~180 lines (feature engineering, training, evaluation)
- `training/features.py` — ~100 lines (feature transformations)
- `tests/conftest.py` — ~50 lines (pytest fixtures)
- `tests/test_predict.py` — ~100 lines (test cases)

**Dependencies (External):**
- FastAPI, uvicorn, Pydantic v2
- LightGBM, lightgbm
- joblib (model serialization)
- pandas, numpy, scikit-learn
- prometheus_client (metrics export)
- pytest, pytest-asyncio (testing)

### Module: llm-eval
**Entry Points:**
- `evaluator.cli:score` — Click command-line interface
- `setup.py:entry_points['console_scripts']` → `mlops-eval`

**Public Exports:**
- `evaluator.faithfulness.FaithfulnessScorer` — DeBERTa NLI scorer
- `evaluator.relevance.RelevanceScorer` — Sentence Transformers scorer
- `evaluator.precision.PrecisionScorer` — RAGAS precision scorer

**Files (Planned):**
- `evaluator/__init__.py` — ~20 lines
- `evaluator/faithfulness.py` — ~100 lines
- `evaluator/relevance.py` — ~80 lines
- `evaluator/precision.py` — ~90 lines
- `evaluator/cli.py` — ~130 lines (Click CLI, main logic)
- `tests/test_evaluators.py` — ~150 lines
- `setup.py` — ~30 lines

**Dependencies (External):**
- transformers, sentence-transformers, torch
- ragas
- click (CLI)
- pytest (testing)

### Module: monitoring
**Entry Points:**
- `evidently/drift_report.py` — Standalone script (no imports in other modules)

**Files (Planned):**
- `evidently/drift_report.py` — ~120 lines
- `prometheus/prometheus.yml` — ~20 lines (config)
- `grafana/provisioning/datasources/prometheus.yml` — ~10 lines (config)
- `grafana/provisioning/dashboards/dashboard.yml` — ~5 lines (config)
- `grafana/provisioning/dashboards/mlops-loop.json` — ~200 lines (dashboard definition)

**Dependencies (External):**
- evidently (drift detection)
- prometheus_client (metrics)
- pandas, pyarrow (parquet I/O)

---

## Dependency Graph (Planned)

### Internal Dependencies
```
forecast-api/app/main.py
  ├─→ app/schemas (PredictRequest, PredictResponse)
  ├─→ app/model (LGBMModel)
  └─→ training/features (feature engineering)

forecast-api/training/train.py
  ├─→ training/features (feature engineering)
  └─→ (joblib, lightgbm)

llm-eval/evaluator/cli.py
  ├─→ evaluator/faithfulness
  ├─→ evaluator/relevance
  └─→ evaluator/precision

monitoring/evidently/drift_report.py
  └─→ (no internal dependencies)

.github/workflows/ci.yml
  ├─→ services/forecast-api/tests/
  └─→ services/llm-eval/evaluator/cli.py (external process call)
```

### External Dependencies by Service

**forecast-api:**
- FastAPI, uvicorn, Pydantic v2
- LightGBM, scikit-learn, pandas, numpy, joblib
- prometheus_client
- pytest, pytest-asyncio

**llm-eval:**
- transformers, sentence-transformers, torch
- ragas
- click
- pytest

**monitoring:**
- evidently
- prometheus (Docker image, not pip)
- grafana (Docker image, not pip)
- pandas, pyarrow

**CI/CD:**
- ruff, black (linting/formatting)
- pytest (testing)
- Docker, docker-compose

---

## Critical Constraints (Immutable)

| Constraint | Details | Verification Method |
|---|---|---|
| **Python Version** | 3.11 only (no 3.10, no 3.12) | requirements.txt, Dockerfile FROM |
| **ML Stack** | PyTorch only (no TensorFlow) | transformers/torch in requirements |
| **Testing** | pytest required; no mocking of model — load fixture | conftest.py fixture strategy |
| **Docker Image Size** | < 150MB final runtime stage | Multi-stage Dockerfile, slim base image |
| **Secrets** | No hardcoded credentials — all via env vars or GitHub Secrets | Grep for secrets in source |
| **Local Dev** | `docker compose up` must start all 3 services zero-config | docker-compose.yml volumes, env setup |

---

## Key Architectural Decisions (Immutable)

| Decision | Choice | Reason | Interview Q |
|---|---|---|---|
| Forecasting model | LightGBM | Tabular + categorical features; no stationarity requirement; ~30s train | QW-1.1 |
| Drift metric | PSI (not KL divergence) | Symmetric; industry thresholds (0.1 / 0.2); banking-origin credibility | QW-2.4 |
| Faithfulness eval | DeBERTa NLI (not LLM-as-judge) | Deterministic; CPU ~200ms/sample; no API cost; no circular reasoning | QW-3.7 |
| API framework | FastAPI + Pydantic v2 | Async; auto-docs; strict type validation | QW-1.3 |
| Container registry | GHCR (not DockerHub) | Native GitHub Actions integration; no rate limits on public repos | N/A |

---

## Implementation Specifications Per File

### `services/forecast-api/app/schemas.py`
**Spec:**
```python
from pydantic import BaseModel

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

**Constraints:**
- Pydantic v2 only
- Type validation strict
- No optional fields except `days_ahead` (default=30)

---

### `services/forecast-api/app/main.py`
**Entry Point:** `app` (FastAPI instance)

**Spec:**
- Lifespan context manager: load model on startup, verify model_loaded flag
- `POST /predict` → takes PredictRequest, returns PredictResponse
- `GET /health` → returns `{"status": "ok", "model_loaded": bool}`
- `GET /metrics` → Prometheus metrics via `prometheus_client`
  - Expose: request_count, latency histogram, PSI values
- Async request handling
- Auto-docs at `/docs`

**Constraints:**
- Must not crash if model.pkl is missing (graceful degradation, return model_loaded=false)
- Latency p95 < 200ms
- Metrics must be Prometheus format (text-based)

---

### `services/forecast-api/app/model.py`
**Entry Point:** `LGBMModel` (class)

**Spec:**
- Load `model.pkl` from disk on init
- `predict(features: dict) → list[float]` method
- Feature engineering inside inference (ensure train/serve feature parity)
- Handle missing features with sensible defaults (log warning, don't crash)

**Constraints:**
- No training code in model.py (train.py only)
- Inference must be deterministic
- Must accept dict input (key = feature name)

---

### `services/forecast-api/training/train.py`
**Entry Point:** `main()` function

**Spec:**
- Load Kaggle Store Sales CSV at `data/train.csv`
- Feature engineering:
  - Date parts: day_of_week, month, year, days_to_holiday
  - Lags: lag_7, lag_14
  - Rolling mean: rolling_mean_7
  - One-hot encoding on categorical features
- LightGBM hyperparams:
  - `objective="regression"`
  - `metric="rmse"`
  - `num_leaves=31`
  - `n_estimators=500`
- Train/val split: 80/20 (or as per PRD)
- Save model as `model.pkl` via joblib
- Print train/val RMSE to stdout
- Exit code 0 on success

**Constraints:**
- Must run offline (no live data)
- Reproducible with fixed random seed
- Expect `data/train.csv` to exist before running

---

### `services/forecast-api/training/features.py`
**Entry Point:** Feature engineering functions (imported by main.py, train.py)

**Spec:**
- Provide functions for date feature extraction
- Provide functions for lag/rolling feature computation
- Provide OHE transformer (stateless or fitted)
- Used by both training and inference paths

**Constraints:**
- Feature parity: train and predict must use identical transformations

---

### `services/forecast-api/Dockerfile`
**Spec (Multi-Stage):**

```dockerfile
# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
COPY model.pkl .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Constraints:**
- Final image < 150MB
- No build artifacts in runtime stage
- model.pkl must be present at build time

---

### `services/forecast-api/requirements.txt`
**Spec:**
- FastAPI, uvicorn
- Pydantic v2 (>= 2.0)
- LightGBM
- pandas, numpy, scikit-learn
- joblib
- prometheus_client
- pytest, pytest-asyncio (dev deps optional)
- Python 3.11

**Constraints:**
- Pin major versions
- No TensorFlow
- PyTorch stack only (transformers/torch for later phases)

---

### `services/llm-eval/evaluator/faithfulness.py`
**Entry Point:** `FaithfulnessScorer` (class)

**Spec:**
- Model: `cross-encoder/nli-deberta-v3-large`
- Input: `context: str, answer: str`
- Output: entailment probability (float 0–1)
- Use `transformers.pipeline(task="text-classification")`
- Label mapping: entailment → score, contradiction → 1 - score, other → 0.5

**Constraints:**
- Deterministic (no randomness)
- CPU-based inference (~200ms/sample)
- No API calls (offline)

---

### `services/llm-eval/evaluator/relevance.py`
**Entry Point:** `RelevanceScorer` (class)

**Spec:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Input: `question: str, answer: str`
- Output: cosine similarity (float 0–1)
- Embed both, compute cosine similarity

**Constraints:**
- Offline computation
- Deterministic embeddings

---

### `services/llm-eval/evaluator/precision.py`
**Entry Point:** `PrecisionScorer` (class)

**Spec:**
- Use RAGAS library for context precision computation
- Input: `context: str, answer: str, question: str` (all needed for RAGAS)
- Output: precision score (float 0–1)

**Constraints:**
- Must use RAGAS implementation (not custom)

---

### `services/llm-eval/evaluator/cli.py`
**Entry Point:** `score` (Click command)

**Spec:**
- Click command: `mlops-eval score <input_file> --threshold 0.7`
- Input file: JSONL
  - Each line: `{"question": "...", "context": "...", "answer": "..."}`
- Output: 
  - Per-sample scores (faithfulness, relevance, precision)
  - Mean scores per metric
  - PASS/FAIL judgment per metric (mean >= threshold)
  - Overall pass: all metrics PASS
- Exit code:
  - 0 if all metrics PASS
  - 1 if any metric FAIL
- Stdout: human-readable report + JSON block

**Constraints:**
- Exit code 1 must trigger CI workflow failure (used in GitHub Actions)
- Idempotent (same input = same output)

---

### `services/llm-eval/setup.py`
**Spec:**
```python
from setuptools import setup, find_packages

setup(
    name="mlops-eval",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mlops-eval=evaluator.cli:score",
        ],
    },
    install_requires=[
        # from requirements.txt
    ],
)
```

**Constraints:**
- Must register `mlops-eval` command
- Installable via `pip install -e .`

---

### `services/llm-eval/requirements.txt`
**Spec:**
- transformers
- sentence-transformers
- torch
- ragas
- click
- pytest
- Python 3.11

**Constraints:**
- PyTorch stack only
- No TensorFlow

---

### `monitoring/evidently/drift_report.py`
**Entry Point:** Standalone script (can be invoked as `python drift_report.py`)

**Spec:**
- Load `reference_data.parquet` (training feature distribution snapshot)
- Accept current window as CSV or parquet path (CLI arg)
- Compute PSI per feature via Evidently `DataDriftPreset`
- Print PSI scores to stdout as JSON
- Write Prometheus metrics file

**Constraints:**
- No database dependency
- Offline computation
- Reproducible (same input = same output)

---

### `monitoring/prometheus/prometheus.yml`
**Spec:**
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

**Constraints:**
- Target must match docker-compose service name
- 15s scrape interval per spec

---

### `monitoring/grafana/provisioning/datasources/prometheus.yml`
**Spec:**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    access: proxy
```

**Constraints:**
- Auto-provisioned (no manual UI setup)
- Must reference Prometheus service in docker-compose network

---

### `monitoring/grafana/provisioning/dashboards/mlops-loop.json`
**Spec:**
- Pre-built Grafana dashboard
- Panels for:
  - PSI per feature over time
  - Forecast API request count
  - Forecast API latency histogram
  - Model health status
- JSON valid, loads on container startup

**Constraints:**
- Must be valid JSON
- Dashboard must reference Prometheus data source
- No manual provisioning in UI

---

### `.github/workflows/ci.yml`
**Spec (Sequential Jobs):**

1. **lint** (ubuntu-latest)
   - Checkout
   - Install: ruff, black
   - Run: `ruff check .` && `black --check .`

2. **test** (ubuntu-latest, needs: lint)
   - Checkout
   - Install: requirements from `services/forecast-api/requirements.txt`
   - Run: `pytest services/forecast-api/tests/ -v`

3. **eval-gate** (ubuntu-latest, needs: test)
   - Checkout
   - Install: `pip install -e services/llm-eval/`
   - Run: `mlops-eval score tests/fixtures/eval_set.jsonl --threshold 0.7`

4. **build-push** (ubuntu-latest, needs: eval-gate, if: main branch only)
   - Checkout
   - Login to GHCR
   - Build & push Docker image to `ghcr.io/anh-d-tran030/mlops-loop:latest`

**Constraints:**
- Sequential dependency: lint → test → eval-gate → build-push
- build-push only runs on main branch
- Exit code 1 from any job fails the workflow

---

### `docker-compose.yml`
**Spec:**
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

**Constraints:**
- Zero-config: `docker compose up` must start all 3 services
- Grafana provisioning auto-loads
- All services healthy on startup

---

### `.pre-commit-config.yaml`
**Spec:**
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

**Constraints:**
- Enforced before commit (locally and in CI)
- Python 3.11

---

### `README.md`
**Spec:**
- Copy from PRD.md Section 2 and Section 10 verbatim
- Sections:
  - Project Identity
  - The Three Components (QW-1, QW-2, QW-3)
  - Quick Start (docker compose up)
  - Architecture diagram (ASCII or Markdown)
  - The 9 Interview Questions

**Constraints:**
- Word-for-word from PRD (verifiable via diff)

---

### `tests/fixtures/eval_set.jsonl`
**Spec:**
- Sample evaluation data (JSONL format)
- Each line: `{"question": "...", "context": "...", "answer": "..."}`
- 10–20 samples sufficient for CI gate verification
- Must pass with scores > 0.7 (threshold)

**Constraints:**
- Valid JSONL
- All required fields present in every sample

---

## Dead Code / Circular Dependencies (None Yet)

Greenfield project — no dead code, no circular dependencies. To be checked once source written:

**Checks to perform after implementation:**
- [ ] All public exports from modules are imported somewhere
- [ ] No unused imports within modules
- [ ] No circular imports (A → B → A)
- [ ] No hardcoded secrets in source files

---

## Line Count Expectations

**Files expected to exceed 150 lines:**
- None currently — most modules designed to stay < 150 lines per file
- `mlops-loop.json` (dashboard) expected ~200 lines (JSON, not code)

**Verification:** After implementation, scan all `.py` files and flag any > 150 lines for review.

---

## Anomaly Detection Checklist

**To be verified after source written:**
- [ ] No hardcoded API keys or credentials (grep for patterns)
- [ ] All config values externalized (env vars, not literals)
- [ ] No debug print statements left in production code
- [ ] No commented-out code blocks > 5 lines
- [ ] No incomplete docstrings (all functions documented)
- [ ] No TODO/FIXME comments without issue links

---

## Success Criteria (MVP)

**Minimal Viable Product (merge to main):**
- [ ] `docker compose up` starts all 3 services
- [ ] `GET /health` returns 200, `{"status": "ok", "model_loaded": true}`
- [ ] `POST /predict` accepts valid request, returns forecast + prediction_id
- [ ] `GET /metrics` exposes Prometheus metrics (text format)
- [ ] Grafana dashboard loads at `http://localhost:3000` (admin/admin)
- [ ] CI workflow runs: lint → test → eval-gate → build-push
- [ ] Docker image built and pushed to GHCR
- [ ] All pytest tests pass locally and in CI

**Full Portfolio-Ready (before interview):**
- [ ] All 9 interview questions have coherent answers in docs
- [ ] Every architectural choice justified in CLAUDE.md
- [ ] No hardcoded secrets or credentials
- [ ] Cold start to running < 5 minutes
- [ ] Forecast API p95 latency < 200ms
- [ ] Docker image < 150MB
- [ ] README.md matches PRD.md Section 2 word-for-word

---

## Technology Stack (Summary)

### Core ML
- **LightGBM** — Gradient boosting for tabular regression
- **DeBERTa-v3-large** — Transformer-based NLI (cross-encoder)
- **Sentence Transformers** — Embeddings for relevance (all-MiniLM-L6-v2)

### API & Serialization
- **FastAPI** — Async REST framework
- **Pydantic v2** — Type validation
- **uvicorn** — ASGI server

### Monitoring & Observability
- **Evidently AI** — Feature drift detection (PSI)
- **Prometheus** — Metrics (v2.47.0)
- **Grafana** — Dashboard visualization (v10.1.0)
- **prometheus_client** — Python metrics export

### Data & ML
- **pandas, numpy, scikit-learn** — Data manipulation
- **joblib** — Model serialization
- **torch, transformers** — PyTorch stack
- **sentence-transformers** — NLP embeddings
- **ragas** — RAG evaluation metrics

### CLI & Utilities
- **Click** — Command-line interface
- **pytest** — Testing framework
- **ruff, black** — Linting/formatting

### Infrastructure
- **Docker** — Containerization (multi-stage)
- **docker-compose** — Orchestration
- **GitHub Actions** — CI/CD
- **GHCR** — Container registry

---

## Build Roadmap (All Tasks)

### Phase 1: Core Services
- [ ] Create directory structure (forecast-api, llm-eval, monitoring, tests)
- [ ] Write `services/forecast-api/app/{schemas.py, model.py, main.py}`
- [ ] Write `services/forecast-api/training/{features.py, train.py}`
- [ ] Write `services/forecast-api/tests/{conftest.py, test_predict.py}`
- [ ] Write `services/forecast-api/{Dockerfile, requirements.txt}`
- [ ] Write `services/llm-eval/evaluator/{__init__, faithfulness.py, relevance.py, precision.py, cli.py}`
- [ ] Write `services/llm-eval/tests/test_evaluators.py`
- [ ] Write `services/llm-eval/{Dockerfile, setup.py, requirements.txt}`
- [ ] Write `monitoring/evidently/drift_report.py`
- [ ] Create `monitoring/evidently/reference_data.parquet` (sample)
- [ ] Write `monitoring/prometheus/prometheus.yml`
- [ ] Write `monitoring/grafana/provisioning/{datasources, dashboards}`

### Phase 2: Infrastructure & CI/CD
- [ ] Write `docker-compose.yml`
- [ ] Write `.github/workflows/{ci.yml, eval-gate.yml}` (if separate)
- [ ] Write `.pre-commit-config.yaml`
- [ ] Write `.gitignore` (Python + Docker + Grafana)
- [ ] Write `README.md`
- [ ] Create `tests/fixtures/eval_set.jsonl`
- [ ] Create `data/train.csv` (or download reference)

### Phase 3: Validation
- [ ] All tests pass locally
- [ ] docker-compose up → all 3 services healthy
- [ ] CI workflow passes on main
- [ ] Docker image < 150MB
- [ ] All endpoints responding
- [ ] Grafana dashboard loads

---

## Dependencies & External Systems

**No external APIs required:**
- Models downloaded on first run (transformers/sentence-transformers auto-cache)
- No database dependency (Prometheus stores local time-series)
- No auth system (no SaaS)

**Assumptions:**
- User has Docker + docker-compose installed
- User has Python 3.11 available locally
- GitHub Actions available (repo is public on GitHub)
- Kaggle Store Sales dataset at `data/train.csv` or downloadable

---

## Notes for Implementation Phase

1. **Model Training:** LightGBM model must be trained and saved as `model.pkl` before Docker build (or downloaded from artifact store during build, TBD)
2. **Reference Data:** `reference_data.parquet` must be computed from training set before monitoring stack starts
3. **Pytest Fixtures:** Load a pre-trained minimal LightGBM model in `conftest.py` — don't train during test runs
4. **Prometheus Scrape:** 15-second intervals; forecast API /metrics endpoint must be stable
5. **Grafana Provisioning:** All dashboard JSON must be valid on first load (no manual UI setup)
6. **GitHub Secrets:** `GITHUB_TOKEN` is built-in; no additional secrets needed for MVP
7. **Pre-commit Hooks:** ruff + black must pass before commit (enforced locally and in CI)
8. **Docker Build Context:** Dockerfile in `services/forecast-api/` expects model.pkl at build context root

---

## Interview Defense Questions (The 9 Core)

**QW-1 (Serve):**
1. Why LightGBM over ARIMA for time series forecasting?
2. Why multi-stage Docker? What does the builder stage actually do?
3. What does Pydantic v2 buy you over v1?

**QW-2 (Monitor):**
4. What is PSI and why use it instead of KL divergence?
5. What's the difference between data drift and concept drift? Does your system detect both?
6. Why does Prometheus use a pull model (scrape) instead of push?

**QW-3 (Evaluate):**
7. Why DeBERTa-large for NLI? What makes it better than BERT-base for entailment?
8. Why not use GPT-4 as your faithfulness judge?
9. What happens if the eval harness itself has a bug and always returns scores > 0.7?

---

## Exploration Summary

**Generated:** 2026-06-01T01:15:00Z  
**Agent:** explore  
**Status:** Complete — greenfield project fully mapped, zero source code, all specifications documented  

**Key Findings:**
- All 70+ files specified in CLAUDE.md with full line-count expectations
- Zero existing source code — ready for build phase
- No circular dependencies or dead code possible yet
- All constraints and architectural decisions immutable per PRD + CLAUDE.md
- Dependency graph planned; no conflicts identified
- Docker images and CI/CD pipeline fully specified
- Interview defense questions all addressed in PRD

**Next Phase:** Plan agent defines build task sequencing and implementation order.
