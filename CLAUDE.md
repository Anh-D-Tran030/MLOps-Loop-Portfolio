# mlops-loop — Repo Init Prompt for Claude Code / OpenCode

## Project Identity

- **Repo name:** `mlops-loop`
- **Tagline:** Serve → Monitor → Evaluate. One repo. One loop.
- **GitHub:** `https://github.com/Anh-D-Tran030/mlops-loop`
- **Author:** Anh Duc Tran — AI Engineering student, UTS Sydney

## What You Are Building

A production MLOps mono-repo with three components, each mapping to one phase of the ML lifecycle:

```
QW-1: Forecast API     →   QW-2: Drift Monitor     →   QW-3: LLM Eval Harness
LightGBM + FastAPI         Evidently + Prometheus        DeBERTa NLI + Sentence
Multi-stage Docker          + Grafana                     Transformers + CI gate
```

Business context: retail demand forecasting (Kaggle Store Sales dataset), framed as the AI infrastructure for Vaylo — an AI-driven CRM/inventory platform.

---

## Task: Scaffold the Mono-Repo

Create the following directory and file structure. **Write all files.** Do not leave placeholders.

```
mlops-loop/
├── services/
│   ├── forecast-api/
│   │   ├── app/
│   │   │   ├── main.py          # FastAPI app — /predict, /health, /metrics endpoints
│   │   │   ├── model.py         # LightGBM .pkl loader + inference wrapper
│   │   │   └── schemas.py       # Pydantic v2 PredictRequest + PredictResponse
│   │   ├── training/
│   │   │   ├── train.py         # LightGBM training script, saves model.pkl
│   │   │   └── features.py      # Feature engineering (date parts, lag features, OHE)
│   │   ├── tests/
│   │   │   ├── test_predict.py  # Happy path, edge cases (missing features, out-of-range)
│   │   │   └── conftest.py      # pytest fixtures
│   │   ├── Dockerfile           # Multi-stage: builder (pip install) → runtime (~120MB)
│   │   └── requirements.txt
│   └── llm-eval/
│       ├── evaluator/
│       │   ├── __init__.py
│       │   ├── faithfulness.py  # DeBERTa-large NLI entailment scorer
│       │   ├── relevance.py     # Sentence Transformers cosine similarity
│       │   ├── precision.py     # RAGAS context precision
│       │   └── cli.py           # Click CLI: mlops-eval score <file> --threshold 0.7
│       ├── tests/
│       │   └── test_evaluators.py
│       ├── Dockerfile
│       ├── setup.py             # pip install -e . entry point
│       └── requirements.txt
├── monitoring/
│   ├── evidently/
│   │   ├── drift_report.py      # PSI drift detection, reference vs current window
│   │   └── reference_data.parquet  # Snapshot of training feature distribution
│   ├── prometheus/
│   │   └── prometheus.yml       # Scrape config: scrape forecast-api /metrics every 15s
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── prometheus.yml
│           └── dashboards/
│               ├── dashboard.yml
│               └── mlops-loop.json  # Pre-built Grafana dashboard (PSI panels)
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint → test → eval-gate → build → push (GHCR)
│       └── eval-gate.yml        # Standalone eval quality gate workflow
├── docker-compose.yml           # Starts: forecast-api + prometheus + grafana
├── .pre-commit-config.yaml      # ruff + black hooks
├── .gitignore
└── README.md                    # Copy from PRD.md Section 2 verbatim
```

---

## Implementation Specs Per File

### `services/forecast-api/app/schemas.py`
```python
# Pydantic v2
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

### `services/forecast-api/app/main.py`
- FastAPI app with lifespan context manager (load model on startup)
- `POST /predict` — takes PredictRequest, returns PredictResponse
- `GET /health` — returns `{"status": "ok", "model_loaded": bool}`
- `GET /metrics` — Prometheus metrics via `prometheus_client` (expose PSI values, request count, latency histogram)

### `services/forecast-api/app/model.py`
- Load `model.pkl` from disk on init
- `predict(features: dict) -> list[float]` — feature engineering + LightGBM inference
- Handle missing features with sensible defaults (log a warning, don't crash)

### `services/forecast-api/training/train.py`
- Load Kaggle Store Sales CSV (expect it at `data/train.csv`)
- Feature engineering: day_of_week, month, year, days_to_holiday, lag_7, lag_14, rolling_mean_7
- LightGBM with: `objective=regression`, `metric=rmse`, `num_leaves=31`, `n_estimators=500`
- Save model as `model.pkl` via joblib
- Print train/val RMSE

### `services/forecast-api/Dockerfile`
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

### `services/llm-eval/evaluator/faithfulness.py`
- Model: `cross-encoder/nli-deberta-v3-large`
- Input: `context: str, answer: str`
- Output: entailment probability (float 0–1)
- Use `transformers` pipeline with `task="text-classification"`
- Label mapping: entailment → score, contradiction → 1 - score

### `services/llm-eval/evaluator/relevance.py`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Input: `question: str, answer: str`
- Output: cosine similarity (float 0–1)

### `services/llm-eval/evaluator/cli.py`
- Click CLI: `mlops-eval score <input_file> --threshold 0.7`
- Input file: JSONL, each line `{"question": "...", "context": "...", "answer": "..."}`
- Output: per-sample scores + mean scores + PASS/FAIL per metric
- Exit code 1 if any mean score < threshold (so CI fails)

### `monitoring/evidently/drift_report.py`
- Load `reference_data.parquet` (training feature distribution snapshot)
- Accept current window as CSV or parquet path
- Compute PSI per feature using Evidently `DataDriftPreset`
- Print PSI scores to stdout as JSON
- Write Prometheus metrics file (for `prometheus_client.CollectorRegistry`)

### `.github/workflows/ci.yml`
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

### `docker-compose.yml`
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

---

## Key Architectural Decisions (Do Not Change)

| Decision | Choice | Reason |
|---|---|---|
| Forecasting model | LightGBM | Tabular + categorical features; no stationarity requirement; 30s train time |
| Drift metric | PSI (not KL divergence) | Symmetric; industry-standard thresholds (0.1 / 0.2); banking-origin credibility |
| Faithfulness eval | DeBERTa NLI (not LLM-as-judge) | Deterministic; CPU ~200ms/sample; no API cost; no circular reasoning |
| API framework | FastAPI + Pydantic v2 | Async; auto-docs; strict type validation |
| Container registry | GHCR (not DockerHub) | Native GitHub Actions integration; no rate limits on public repos |

---

## Constraints

- Python 3.11 only
- PyTorch stack (no TensorFlow)
- All tests must pass with `pytest` (no mocking the model — load a minimal fixture model)
- Docker image must be < 150MB final stage
- No hardcoded secrets — all credentials via environment variables or GitHub Secrets
- `docker compose up` must start all three services with zero manual config

---

## What Success Looks Like

```bash
git clone https://github.com/Anh-D-Tran030/mlops-loop.git
cd mlops-loop
docker compose up
# → http://localhost:8000/docs   (Forecast API)
# → http://localhost:3000        (Grafana, admin/admin)
# → http://localhost:9090        (Prometheus)
```

CI badge on `main` is green. Every design choice is answerable in a 45-minute interview without notes.
