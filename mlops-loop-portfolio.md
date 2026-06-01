# MLOps Loop Portfolio — Master Document
**Author:** Anh Duc Tran | **Version:** 1.0 | **Date:** May 2026

---

## Table of Contents
1. [Product Requirements Document (PRD)](#prd)
2. [README (Mono-repo)](#readme)
3. [Sprint Plan (4 Weeks)](#sprint-plan)
4. [Technical Roadmap](#technical-roadmap)

---

# 1. PRODUCT REQUIREMENTS DOCUMENT (PRD)

## 1.1 Problem Statement

Most junior ML portfolios demonstrate model training in isolation. There is no evidence of production thinking: no serving layer, no monitoring, no quality gates. Hiring engineers at Harrison.ai, Relevance AI, and Optiver are not evaluating notebooks — they are evaluating whether a candidate can reason about a system that stays healthy in production.

This project solves that gap by building a single, coherent MLOps system that demonstrates all three phases of a production ML lifecycle as one deployable, observable, testable unit.

## 1.2 Project Identity

| Field | Value |
|---|---|
| **Project Name** | `mlops-loop` |
| **Tagline** | Serve → Monitor → Evaluate. One repo. One loop. |
| **Audience** | ML Engineering hiring managers at AU AI companies |
| **Narrative** | "I built the production ML infrastructure for Vaylo's inventory forecasting pipeline — from model serving through drift detection to LLM output quality gates." |

## 1.3 Goals

**Primary:** A single GitHub mono-repo that a recruiter can clone and run with one command, which demonstrates all three MLOps phases end-to-end.

**Secondary:**
- Every architectural choice is defensible in a 45-minute technical interview
- The system is framed around a real business context (Vaylo / retail demand forecasting)
- Portfolio site links to this repo as the centrepiece project

## 1.4 The Three Components

### Component 1 — Serve (QW-1 rebuild)
**What it does:** A REST API that takes retail store/product/date features and returns a 30-day demand forecast.

**Stack:**
- Model: LightGBM trained on Kaggle Store Sales dataset
- API: FastAPI with Pydantic v2 request/response schemas
- Container: Multi-stage Dockerfile (builder → runtime, ~120MB final image)
- Registry: GitHub Container Registry (GHCR) via GitHub Actions
- CI: On push to `main` — lint → test → build → push

**Why LightGBM, not LSTM/ARIMA:**
- Tabular features with categorical encodings → gradient boosting outperforms sequence models
- LSTM requires sequence padding and stationarity checks; LightGBM handles raw dates as features
- Training time: LightGBM ~30s vs LSTM ~15min for equivalent accuracy on this dataset
- This is the answer you give in the interview

### Component 2 — Monitor (QW-2 rebuild)
**What it does:** Detects when the incoming request distribution drifts from the training distribution, surfaces this as Prometheus metrics, visualises in Grafana.

**Stack:**
- Drift detection: Evidently AI (PSI for continuous features, chi-squared for categoricals)
- Metrics endpoint: `/metrics` on the FastAPI app, scraped by Prometheus
- Dashboard: Grafana pre-provisioned via `provisioning/` directory (no manual setup)
- Orchestration: Docker Compose (prometheus + grafana + api all start together)

**Why PSI, not KL divergence:**
- PSI is symmetric and has industry-standard thresholds (< 0.1 stable, 0.1–0.2 moderate, > 0.2 significant)
- KL divergence is asymmetric — P(train) vs Q(serve) ≠ Q(serve) vs P(train) — which makes alerting thresholds arbitrary
- PSI originated in credit risk (banking), which directly maps to the CBA/Westpac hiring context
- This is the answer you give in the interview

### Component 3 — Evaluate (QW-3 rebuild)
**What it does:** A CLI tool and CI gate that scores any RAG pipeline on three quality dimensions. If scores drop below threshold, the Docker image is not pushed.

**Stack:**
- Faithfulness: DeBERTa-large NLI (premise=context, hypothesis=answer → entailment score)
- Answer relevance: Sentence Transformers cosine similarity (question ↔ answer embedding)
- Context precision: RAGAS precision metric
- CI gate: GitHub Actions step that fails the workflow if any metric < threshold

**Why NLI-based faithfulness, not LLM-as-judge:**
- LLM-as-judge introduces a second LLM into your eval pipeline — latency, cost, and circular reasoning risk
- DeBERTa NLI is deterministic, fast (CPU inference ~200ms/sample), and free
- The trade-off: NLI is less nuanced for edge cases; LLM-as-judge is better for subjective quality
- You choose NLI for a CI gate specifically because it needs to be fast and deterministic
- This is the answer you give in the interview

## 1.5 Non-Goals

- This is not a SaaS product. No auth, no multi-tenancy, no database persistence.
- This is not a research project. No novel model architecture. LightGBM + DeBERTa are chosen for interview-defensibility, not SOTA performance.
- This is not a Streamlit dashboard. The Vaylo Dashboard (LT-1) is a separate project that consumes this system.

## 1.6 Success Metrics

| Metric | Target |
|---|---|
| Cold start (clone → running) | `docker compose up` in < 5 minutes |
| CI pipeline pass rate | 100% on `main` branch |
| Forecast API p95 latency | < 200ms |
| Drift detection false positive rate | < 5% on held-out reference data |
| LLM eval CI gate | Blocks bad images 100% of the time |
| Interview ability | Can explain every tool choice without notes |

## 1.7 Out-of-Scope (for this sprint)

- Kubernetes deployment (add post-internship)
- Feature store integration (Feast / Hopsworks)
- A/B model serving
- Real-time streaming data ingestion (Kafka/Flink)

---

# 2. README (MONO-REPO)

> Copy this verbatim into your GitHub repo's root `README.md`

---

```markdown
# mlops-loop

> Serve → Monitor → Evaluate. A production MLOps system in one repo.

[![CI](https://github.com/Anh-D-Tran030/mlops-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/Anh-D-Tran030/mlops-loop/actions)
[![Docker](https://ghcr.io/Anh-D-Tran030/mlops-loop)](https://github.com/Anh-D-Tran030/mlops-loop/pkgs/container/mlops-loop)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## What This Is

Most ML projects stop at training. This one doesn't.

`mlops-loop` is a complete production ML lifecycle built around a retail demand forecasting use case:

```
┌──────────────┐    predictions     ┌──────────────────┐    metrics      ┌──────────────────┐
│  Forecast    │ ─────────────────▶ │  Drift Monitor   │ ──────────────▶ │  LLM Eval        │
│  API (QW-1)  │                    │  Dashboard (QW-2) │                 │  Harness (QW-3)  │
│              │                    │                  │                 │                  │
│  LightGBM    │                    │  Evidently AI    │                 │  DeBERTa NLI     │
│  FastAPI     │                    │  Prometheus      │                 │  Sentence Trans. │
│  Docker      │                    │  Grafana         │                 │  CI Quality Gate │
└──────────────┘                    └──────────────────┘                 └──────────────────┘
```

**Business context:** Built as the AI infrastructure layer for [Vaylo](https://vaylo.com) — an AI-driven CRM and inventory platform for SMBs.

## Quick Start

```bash
git clone https://github.com/Anh-D-Tran030/mlops-loop.git
cd mlops-loop
docker compose up
```

| Service | URL |
|---|---|
| Forecast API | http://localhost:8000/docs |
| Grafana Dashboard | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

## Repo Structure

```
mlops-loop/
├── services/
│   ├── forecast-api/          # QW-1: LightGBM model + FastAPI
│   │   ├── app/
│   │   │   ├── main.py        # FastAPI app, /predict + /metrics endpoints
│   │   │   ├── model.py       # LightGBM inference wrapper
│   │   │   └── schemas.py     # Pydantic v2 request/response models
│   │   ├── training/
│   │   │   └── train.py       # Model training script (Kaggle Store Sales)
│   │   ├── tests/
│   │   │   └── test_predict.py
│   │   └── Dockerfile
│   └── llm-eval/              # QW-3: DeBERTa NLI + Sentence Transformers
│       ├── evaluator/
│       │   ├── faithfulness.py
│       │   ├── relevance.py
│       │   └── cli.py
│       ├── tests/
│       └── Dockerfile
├── monitoring/                # QW-2: Evidently + Prometheus + Grafana
│   ├── evidently/
│   │   └── drift_report.py    # PSI drift detection script
│   ├── prometheus/
│   │   └── prometheus.yml     # Scrape config
│   └── grafana/
│       └── provisioning/      # Pre-built dashboards (no manual setup)
├── .github/
│   └── workflows/
│       ├── ci.yml             # Lint → Test → Build → Push (GHCR)
│       └── eval-gate.yml      # LLM eval quality gate
├── docker-compose.yml         # Spin up everything
└── README.md
```

## The Three Systems

### QW-1 — Forecast API
LightGBM trained on [Kaggle Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting). Served via FastAPI. Multi-stage Docker build (builder stage installs deps, runtime stage is ~120MB).

**Key design choice:** LightGBM over LSTM — tabular features with categorical encodings (store_id, family) benefit from gradient boosting, not sequence modelling. LSTM would require stationarity testing and sequence padding for no accuracy gain.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_nbr": 1, "family": "GROCERY I", "onpromotion": 5, "days_ahead": 30}'
```

### QW-2 — Drift Monitor
Evidently AI computes **Population Stability Index (PSI)** on incoming request features vs the training reference dataset. Exposed as Prometheus metrics, visualised in Grafana.

**Key design choice:** PSI over KL divergence — PSI is symmetric and has industry-standard thresholds (< 0.1 stable, 0.1–0.2 monitor, > 0.2 alert). KL divergence requires manual threshold selection and is asymmetric.

### QW-3 — LLM Eval Harness
CLI tool that scores a RAG pipeline on three axes:
- **Faithfulness** — DeBERTa NLI entailment score (context → answer)
- **Answer Relevance** — cosine similarity between question and answer embeddings
- **Context Precision** — RAGAS precision metric

Integrated as a GitHub Actions step. If any score < threshold, the workflow fails and the Docker image is not pushed.

**Key design choice:** NLI over LLM-as-judge — deterministic, fast (CPU ~200ms/sample), zero API cost, no circular reasoning risk.

```bash
pip install mlops-loop-eval
mlops-eval score --pipeline my_rag_pipeline --threshold 0.7
```

## CI/CD

```
push to main
    │
    ├─▶ Lint (ruff + black)
    ├─▶ Unit tests (pytest)
    ├─▶ LLM Eval gate (fails if quality < threshold)
    ├─▶ Docker build (multi-stage)
    └─▶ Push to GHCR
```

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Model | LightGBM | Tabular data, fast training, interpretable |
| API | FastAPI + Pydantic v2 | Async, auto-docs, type-safe |
| Drift | Evidently AI | PSI out-of-box, no config needed |
| Metrics | Prometheus | Industry standard scrape model |
| Dashboard | Grafana | Pre-provisioned, zero manual setup |
| NLI Eval | DeBERTa-large | Deterministic, CPU-friendly, free |
| Embeddings | Sentence Transformers | State-of-art cosine similarity |
| Container | Docker + GHCR | Reproducible, registry-backed |
| CI/CD | GitHub Actions | Native, free for public repos |

## Local Development

```bash
# Forecast API only
cd services/forecast-api
pip install -r requirements.txt
uvicorn app.main:app --reload

# Run eval harness
cd services/llm-eval
pip install -e .
mlops-eval score --help

# Full stack
docker compose up --build
```

## Author

**Anh Duc Tran** — AI Engineering student at UTS Sydney, Co-Founder & CTO of [Vaylo Technologies](https://vaylo.com).

[Portfolio](https://anh-duc-portfolio.vercel.app) · [GitHub](https://github.com/Anh-D-Tran030) · [LinkedIn](#)
```

---

# 3. SPRINT PLAN (4 WEEKS)

## Principles
- Each week has one clear deliverable that could be demoed independently
- Studies are non-negotiable — deep work blocks are morning only
- No starting Week N+1 until Week N's deliverable is merged to `main` with green CI

## Week 1 — Repo Architecture + QW-1 Rebuild

**Goal:** Mono-repo scaffolded, Forecast API running locally with tests passing.

**Daily breakdown:**

| Day | Task | Output |
|---|---|---|
| Mon | Create `mlops-loop` GitHub repo, set up mono-repo structure, configure ruff + black + pre-commit | Repo live, linting passing |
| Tue | Copy QW-1 code into `services/forecast-api/`, refactor to Pydantic v2 schemas | `app/schemas.py` complete |
| Wed | Write `tests/test_predict.py` — test happy path, edge cases (missing features, out-of-range values) | `pytest` passing locally |
| Thu | Write multi-stage `Dockerfile` — builder stage (pip install), runtime stage (copy app only) | `docker build` succeeds, image < 150MB |
| Fri | Write `.github/workflows/ci.yml` — lint → test → build stages only (no push yet) | Green CI badge on PR |

**Definition of done:** `docker run` serves predictions, `pytest` passes in CI, README for QW-1 written.

---

## Week 2 — Monitoring Stack (QW-2 Rebuild)

**Goal:** Full observability stack spinning up with one command, drift visible in Grafana.

| Day | Task | Output |
|---|---|---|
| Mon | Write `monitoring/evidently/drift_report.py` — PSI on request features vs reference dataset | Script outputs drift scores to stdout |
| Tue | Expose drift scores as Prometheus metrics via `/metrics` endpoint on the FastAPI app | `curl localhost:8000/metrics` returns PSI values |
| Wed | Write `monitoring/prometheus/prometheus.yml` scrape config — scrape the API's `/metrics` every 15s | Prometheus scraping successfully |
| Thu | Set up Grafana provisioning — `datasources/` and `dashboards/` YAML, import pre-built dashboard JSON | `docker compose up` shows dashboard without manual config |
| Fri | Write `docker-compose.yml` integrating all three services (api + prometheus + grafana) | `docker compose up` → all three green |

**Definition of done:** `docker compose up`, open Grafana, see live PSI metrics updating as requests hit the API.

---

## Week 3 — LLM Eval Harness + CI Gate (QW-3 Rebuild)

**Goal:** `mlops-eval` CLI tool installable, CI gate blocking bad images.

| Day | Task | Output |
|---|---|---|
| Mon | Write `services/llm-eval/evaluator/faithfulness.py` — DeBERTa NLI entailment scorer | Unit tested, deterministic output |
| Tue | Write `evaluator/relevance.py` — Sentence Transformers cosine similarity | Unit tested |
| Wed | Write `evaluator/cli.py` — Click CLI that reads a JSONL file of (question, context, answer) triples and outputs scores | `mlops-eval score input.jsonl` works |
| Thu | Write `services/llm-eval/Dockerfile` and add to Docker Compose as a one-shot eval runner | `docker compose run llm-eval` outputs scores |
| Fri | Add `.github/workflows/eval-gate.yml` — runs eval on a fixed test set, fails if any score < 0.7 | CI blocks a deliberately bad input |

**Definition of done:** A commit with a low-quality RAG output causes CI to fail with a clear error message. A commit with a good output passes.

---

## Week 4 — Polish, GHCR Push, Portfolio Integration

**Goal:** Images in GHCR, README complete, portfolio site updated, system explainable cold.

| Day | Task | Output |
|---|---|---|
| Mon | Add GHCR push step to `ci.yml` — authenticate, tag, push on merge to `main` | Image visible at `ghcr.io/Anh-D-Tran030/mlops-loop` |
| Tue | Write the root `README.md` using the template from Section 2 | README live on GitHub |
| Wed | Update portfolio site (`anh-duc-portfolio.vercel.app`) — add mlops-loop as featured project with the system diagram and 3 architectural decision callouts | Portfolio updated |
| Thu | Interview prep — answer these 9 questions without notes (see Section 4.4) | Verbal fluency |
| Fri | Buffer — fix any CI flakiness, update portfolio copy, cold-start test (fresh machine, clone, `docker compose up`) | System works for a stranger |

**Definition of done:** Recruiter can clone the repo, run `docker compose up`, and see a working system. You can explain every design choice in under 60 seconds each.

---

# 4. TECHNICAL ROADMAP

## 4.1 Repository Structure (Final State)

```
mlops-loop/
├── services/
│   ├── forecast-api/
│   │   ├── app/
│   │   │   ├── main.py         # FastAPI, /predict /health /metrics
│   │   │   ├── model.py        # LightGBM .pkl loader + inference
│   │   │   └── schemas.py      # PredictRequest, PredictResponse (Pydantic v2)
│   │   ├── training/
│   │   │   ├── train.py        # LightGBM training, saves model.pkl
│   │   │   └── features.py     # Feature engineering (date parts, lag features)
│   │   ├── tests/
│   │   │   ├── test_predict.py
│   │   │   └── conftest.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── llm-eval/
│       ├── evaluator/
│       │   ├── __init__.py
│       │   ├── faithfulness.py  # DeBERTa NLI
│       │   ├── relevance.py     # Sentence Transformers
│       │   ├── precision.py     # RAGAS context precision
│       │   └── cli.py           # Click CLI entry point
│       ├── tests/
│       ├── Dockerfile
│       ├── setup.py             # pip install -e .
│       └── requirements.txt
├── monitoring/
│   ├── evidently/
│   │   ├── drift_report.py      # Runs PSI on reference vs current window
│   │   └── reference_data.parquet  # Training distribution snapshot
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── prometheus.yml
│           └── dashboards/
│               ├── dashboard.yml
│               └── mlops-loop.json
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── eval-gate.yml
├── docker-compose.yml
├── .pre-commit-config.yaml
└── README.md
```

## 4.2 CI/CD Pipeline Detail

```yaml
# .github/workflows/ci.yml (structure)
name: CI
on: push

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
      - run: pytest services/forecast-api/tests/

  eval-gate:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e services/llm-eval/
      - run: mlops-eval score tests/fixtures/eval_set.jsonl --threshold 0.7

  build-push:
    needs: eval-gate
    runs-on: ubuntu-latest
    steps:
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

## 4.3 Docker Compose (Full Stack)

```yaml
# docker-compose.yml
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
```

## 4.4 The 9 Interview Questions You Must Answer Cold

These are the exact questions a Harrison.ai or Optiver interviewer will ask. Rehearse until fluent — no notes.

**On QW-1 (Serve):**
1. Why LightGBM over ARIMA for time series forecasting?
2. Why multi-stage Docker? What does the builder stage actually do?
3. What does Pydantic v2 buy you over v1?

**On QW-2 (Monitor):**
4. What is PSI and why do you use it instead of KL divergence?
5. What's the difference between data drift and concept drift? Does your system detect both?
6. Why does Prometheus use a pull model (scrape) instead of push?

**On QW-3 (Evaluate):**
7. Why DeBERTa-large for NLI? What makes it better than BERT-base for entailment?
8. Why not use GPT-4 as your faithfulness judge?
9. What happens if the eval harness itself has a bug and always returns scores > 0.7?

## 4.5 The 60-Second Portfolio Pitch

Memorise this. Deliver it at the start of any "tell me about your projects" question:

> "I built an end-to-end MLOps system around a retail demand forecasting use case — the same problem I was solving at Vaylo. Three components, one loop. First: a LightGBM model served via FastAPI, containerised with multi-stage Docker, pushed to GHCR via GitHub Actions. Second: a monitoring layer — Evidently AI computes PSI drift on incoming request features against the training distribution, surfaces metrics to Prometheus, visualised in Grafana. Third: a CI quality gate — a CLI tool using DeBERTa NLI to score RAG pipeline faithfulness. If the score drops below 0.7, the image doesn't ship. The whole stack runs with `docker compose up`. The point isn't the tools — it's that the system is observable, testable, and self-healing. That's what production ML looks like."

That's 90 seconds when spoken. Fast delivery: 60 seconds.

## 4.6 Upgrade Path (Post-Internship)

These are intentionally out of scope now. Add them after you land the internship as continuous improvement signals.

| Addition | Timeline | Why It Matters |
|---|---|---|
| MLflow experiment tracking + model registry | Month 2 post-internship | Shows model versioning discipline |
| Kubernetes deployment (kind locally, EKS for resume) | Month 3 | Takes you from Docker Compose to container orchestration |
| Feature store (Feast) | Month 4 | Shows understanding of training/serving skew |
| Kafka ingestion for real-time drift | Month 5 | Streaming ML — Atlassian/Canva territory |
| A/B model serving | Month 6 | Senior MLE territory |

---

## Summary: What You're Building

```
Week 1  →  Forecast API (QW-1)          → green CI badge
Week 2  →  Monitoring Stack (QW-2)       → Grafana dashboard live
Week 3  →  LLM Eval Harness (QW-3)       → CI gate blocking bad images
Week 4  →  Polish + Portfolio integration → recruiter-ready
```

One repo. One loop. Every choice defensible. Every component runnable.
