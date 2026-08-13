# mlops-loop

> Serve → Monitor → Evaluate. One repo. One loop.

[![CI](https://github.com/Anh-D-Tran030/MLOps-Loop-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/Anh-D-Tran030/MLOps-Loop-Portfolio/actions/workflows/ci.yml)

Most ML portfolios stop at training. This one doesn't. `mlops-loop` is the production infrastructure layer for Vaylo's retail demand forecasting pipeline — model serving, drift detection, and LLM output quality gates as a single deployable system. One `docker compose up`. Three services. Zero manual config.

---

## The Loop

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  mlops-loop                         │
                        │                                                     │
   HTTP Request         │   ┌──────────────────┐                             │
  ────────────────────► │   │  QW-1: Forecast  │  /metrics (PSI, latency)   │
                        │   │  FastAPI +        │ ──────────────────────────►│──┐
   forecast: [float]    │   │  LightGBM +       │                            │  │
  ◄──────────────────── │   │  Pydantic v2      │                            │  │
                        │   └──────────────────┘                             │  │
                        │                                                     │  │
                        │   ┌──────────────────┐  ┌──────────────────┐      │  │
                        │   │  QW-2: Monitor   │  │    Grafana       │      │  │
                        │   │  Evidently PSI   │◄─│  Dashboard       │◄─────│──┘
                        │   │  Prometheus      │  │  (auto-provisioned)     │
                        │   └──────────────────┘  └──────────────────┘      │
                        │                                                     │
                        │   ┌──────────────────────────────────────────┐     │
                        │   │  QW-3: LLM Eval Harness (CI gate)        │     │
                        │   │  DeBERTa NLI · Sentence Transformers     │     │
                        │   │  score < 0.7 → blocks Docker push        │     │
                        │   └──────────────────────────────────────────┘     │
                        └─────────────────────────────────────────────────────┘
```

| Component | Stack | Role in the loop |
|---|---|---|
| **QW-1: Forecast API** | FastAPI + LightGBM + Pydantic v2 + multi-stage Docker | Serves predictions; exposes `/metrics` for drift surface |
| **QW-2: Drift Monitor** | Evidently (PSI) + Prometheus + Grafana | Watches feature distribution shift against training baseline |
| **QW-3: LLM Eval Harness** | DeBERTa NLI + Sentence Transformers + RAGAS + Click CLI | CI gate — blocks image push if RAG output quality drops |

---

## Quick Start

**Prerequisites:** Docker + Docker Compose. Train the model first (requires `data/train.csv` — [Kaggle Store Sales dataset](https://www.kaggle.com/c/store-sales-time-series-forecasting/data)).

```bash
git clone https://github.com/Anh-D-Tran030/MLOps-Loop-Portfolio.git
cd MLOps-Loop-Portfolio

# Step 1: train + save model.pkl
python services/forecast-api/training/train.py

# Step 2: spin up all three services
docker compose up
```

| Service | URL | Credentials |
|---|---|---|
| Forecast API | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

---

## Design Rationale

Every choice below is answerable in a 45-minute interview without notes.

### QW-1: Forecast API

| Decision | Choice | Why |
|---|---|---|
| Model | LightGBM | Tabular + categorical features. LSTM requires stationarity, ~15min train time, no accuracy gain here. LightGBM trains in ~30s. |
| API | FastAPI + Pydantic v2 | Async, auto-docs, strict runtime type validation. Pydantic v2 is 5–50× faster than v1 (Rust core). |
| Container | Multi-stage Dockerfile | Builder installs deps; runtime stage copies only `~/.local`. Keeps final image under 150MB. |
| Registry | GHCR via GitHub Actions | Native integration, no rate limits on public repos, zero secret config beyond `GITHUB_TOKEN`. |

### QW-2: Drift Monitor

| Decision | Choice | Why |
|---|---|---|
| Drift metric | PSI (Population Stability Index) | Symmetric — KL divergence isn't. PSI has industry-standard thresholds (`<0.1` stable, `0.1–0.2` moderate, `>0.2` significant) from credit risk. Thresholds are defensible; KL thresholds are arbitrary. |
| Metrics layer | Prometheus pull model | Pull = scrape on a schedule, not push on every event. Survives API restarts without losing metrics. |
| Dashboard | Grafana provisioned via `provisioning/` | Zero manual setup. `docker compose up` loads the dashboard automatically — no ClickOps. |

### QW-3: LLM Eval Harness

| Decision | Choice | Why |
|---|---|---|
| Faithfulness | DeBERTa-large NLI | Deterministic, CPU ~200ms/sample, free. LLM-as-judge adds a second LLM to the eval loop — latency, cost, and circular reasoning (grading its own output). |
| Relevance | Sentence Transformers cosine similarity | State-of-art dense embeddings, fully offline, no API dependency. |
| Context precision | RAGAS metric | Standard RAG eval benchmark — comparable across projects. |
| CI gate | Exit code 1 on failure | GitHub Actions step fails automatically. No custom logic needed to block the push. |

---

## CI Pipeline

```
push to main
    │
    ├── lint (ruff + black)
    │       │
    ├── test (pytest)
    │       │
    ├── eval-gate (mlops-eval score --threshold 0.7)
    │       │
    └── build + push to GHCR  ← only if all above pass
```

The eval gate is what makes this different from most portfolios. A model that scores below threshold *cannot ship*.

---

## Repo Structure

```
mlops-loop/
├── services/
│   ├── forecast-api/        # QW-1: LightGBM REST API
│   └── llm-eval/            # QW-3: evaluation CLI
├── monitoring/
│   ├── evidently/           # drift_report.py + reference_data.parquet
│   ├── prometheus/          # scrape config
│   └── grafana/             # auto-provisioned dashboard
├── .github/workflows/       # CI: lint → test → eval-gate → build → push
├── docker-compose.yml       # single command to run everything
└── tests/fixtures/          # eval_set.jsonl for CI gate
```

---

## The 9 Interview Questions

**Serve (QW-1)**
1. Why LightGBM over ARIMA for time series? *(tabular features + categorical encoding → gradient boosting wins)*
2. What does the multi-stage Dockerfile actually do? *(builder installs, runtime copies — shrinks final image)*
3. What does Pydantic v2 buy you over v1? *(Rust core, 5–50× faster validation, stricter by default)*

**Monitor (QW-2)**
4. What is PSI and why not KL divergence? *(symmetric metric, banking-origin industry thresholds)*
5. Data drift vs concept drift — does your system detect both? *(data drift yes; concept drift requires label feedback — out of scope, but I can explain the gap)*
6. Why does Prometheus pull instead of push? *(resilience — API restarts don't lose metrics)*

**Evaluate (QW-3)**
7. Why DeBERTa-large for NLI, not BERT-base? *(DeBERTa uses disentangled attention — better on entailment benchmarks by ~3–5% on MNLI)*
8. Why not GPT-4 as faithfulness judge? *(circular reasoning, API cost, non-deterministic — eval scores vary run-to-run)*
9. What if the eval harness has a bug and always returns > 0.7? *(this is the right question — answer: adversarial test fixtures with known-bad samples that must score below threshold)*

---

## Upgrade Path

| Addition | When | Why |
|---|---|---|
| MLflow model registry | Month 2 | Model versioning discipline — know which `model.pkl` is in prod |
| Kubernetes (kind → EKS) | Month 3 | Orchestration beyond Compose |
| Feature store (Feast) | Month 4 | Training/serving skew awareness |
| Kafka real-time drift | Month 5 | Streaming ML — Atlassian/Canva territory |
| A/B model serving | Month 6 | Senior MLE territory |

---

## The 60-Second Pitch

> "I built an end-to-end MLOps system around a retail demand forecasting use case. Three components, one loop. A LightGBM model served via FastAPI, containerised with multi-stage Docker, pushed to GHCR via GitHub Actions. A monitoring layer — Evidently computes PSI drift on incoming features against the training distribution, surfaces metrics to Prometheus, visualised in Grafana. A CI quality gate — a CLI using DeBERTa NLI to score RAG pipeline faithfulness. Score drops below 0.7, the image doesn't ship. One `docker compose up`. The point isn't the tools — it's that the system is observable, testable, and the bad model can't reach production."

---

**Author:** Anh Duc Tran — AI Engineering student, UTS Sydney · [GitHub](https://github.com/Anh-D-Tran030)

MIT License
