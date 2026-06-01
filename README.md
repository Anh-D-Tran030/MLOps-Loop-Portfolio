# mlops-loop

> Serve → Monitor → Evaluate. One repo. One loop.

[![CI](https://github.com/Anh-D-Tran030/mlops-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/Anh-D-Tran030/mlops-loop/actions/workflows/ci.yml)

---

## Project Identity

| Field | Value |
|---|---|
| **Project Name** | `mlops-loop` |
| **Tagline** | Serve → Monitor → Evaluate. One repo. One loop. |
| **Audience** | ML Engineering hiring managers at AU AI companies |
| **Narrative** | "I built the production ML infrastructure for Vaylo's inventory forecasting pipeline — from model serving through drift detection to LLM output quality gates." |

---

## Architecture

```
QW-1: Forecast API          QW-2: Drift Monitor         QW-3: LLM Eval Harness
────────────────────        ────────────────────        ─────────────────────────
FastAPI + Pydantic v2       Evidently AI (PSI)          DeBERTa NLI (faithfulness)
LightGBM .pkl               Prometheus v2.47.0          Sentence Transformers (relevance)
Multi-stage Docker          Grafana v10.1.0             RAGAS (context precision)
GHCR push via GH Actions    /metrics endpoint           Click CLI + CI exit code gate
```

| Component | Stack | Purpose |
|---|---|---|
| QW-1: Forecast API | FastAPI + LightGBM + Pydantic v2 + multi-stage Docker | Retail demand forecasting REST API; `/predict`, `/health`, `/metrics` |
| QW-2: Drift Monitor | Evidently AI (PSI) + Prometheus + Grafana | Feature drift detection; surface PSI per feature to Prometheus, visualise in Grafana |
| QW-3: LLM Eval Harness | DeBERTa NLI + Sentence Transformers + RAGAS + Click CLI | CI quality gate scoring RAG pipeline output; blocks Docker push if any metric < threshold |

---

## Quick Start

**Prerequisites:** Docker + docker-compose installed. `model.pkl` must exist at `services/forecast-api/model.pkl` before building the image.

```bash
# Step 1: Train the model (requires data/train.csv — Kaggle Store Sales dataset)
python services/forecast-api/training/train.py

# Step 2: Clone and run
git clone https://github.com/Anh-D-Tran030/mlops-loop.git
cd mlops-loop
docker compose up
```

Services available after startup:

- **Forecast API docs:** http://localhost:8000/docs
- **Grafana dashboard:** http://localhost:3000 (admin / admin)
- **Prometheus:** http://localhost:9090

---

## The Three Components

### QW-1: Forecast API

A REST API returning 30-day demand forecasts for retail store/product features. Built for Vaylo's inventory planning pipeline.

**Tech:** FastAPI + LightGBM + Pydantic v2 + multi-stage Docker

**Endpoints:**
- `POST /predict` — takes `{store_nbr, family, onpromotion, days_ahead}`, returns `{forecast: [float], model_version, prediction_id}`
- `GET /health` — returns `{"status": "ok", "model_loaded": bool}`
- `GET /metrics` — Prometheus text format: `forecast_requests_total`, `forecast_request_duration_seconds`, `psi_score`

**Key decisions:**

| Layer | Choice | Rationale |
|---|---|---|
| Model | LightGBM | Tabular features with categorical encodings → gradient boosting outperforms sequence models. LSTM requires stationarity + padding for no accuracy gain. Train time: ~30s vs ~15min for LSTM. |
| API | FastAPI + Pydantic v2 | Async, auto-docs, strict type validation |
| Container | Multi-stage Dockerfile | Builder stage installs deps; runtime stage is ~170MB content size (lightgbm + sklearn + pandas + numpy floor) |
| Registry | GHCR via GitHub Actions | Native integration, no rate limits on public repos |
| CI | On push to `main` | lint → test → build → push |

---

### QW-2: Drift Monitor

Detects when incoming request distribution drifts from the training distribution. Surfaces as Prometheus metrics, visualised in a pre-provisioned Grafana dashboard.

**Tech:** Evidently AI (PSI) + Prometheus v2.47.0 + Grafana v10.1.0

**How it works:**
1. `monitoring/evidently/drift_report.py` loads `reference_data.parquet` (training feature snapshot)
2. Computes PSI per feature via Evidently `DataDriftPreset`
3. Writes Prometheus textfile metrics (`psi_score{feature="..."}`)
4. Prometheus scrapes `/metrics` on the forecast-api every 15s
5. Grafana dashboard auto-provisions on `docker compose up` — no manual UI setup

**Why PSI, not KL divergence:**
PSI is symmetric with industry-standard thresholds: `<0.1` stable, `0.1–0.2` moderate, `>0.2` significant. KL divergence is asymmetric — thresholds become arbitrary. PSI originates in credit risk (banking), relevant for CBA/Westpac hiring context.

| Layer | Choice | Rationale |
|---|---|---|
| Drift detection | Evidently AI (PSI) | Symmetric with industry-standard thresholds. KL divergence is asymmetric — thresholds become arbitrary. |
| Metrics | Prometheus `/metrics` | Industry-standard pull model |
| Dashboard | Grafana provisioned via `provisioning/` | Zero manual setup — dashboard loads on `docker compose up` |

---

### QW-3: LLM Eval Harness

A CLI tool and CI gate that scores any RAG pipeline on three quality dimensions. If scores drop below the threshold, the Docker image is not pushed.

**Tech:** DeBERTa-large NLI + Sentence Transformers + RAGAS + Click CLI

**Usage:**
```bash
pip install -e services/llm-eval/
mlops-eval score tests/fixtures/eval_set.jsonl --threshold 0.7
```

Input: JSONL file, each line `{"question": "...", "context": "...", "answer": "..."}`

Output:
```
Sample 1: faithful=0.82  relevant=0.91  precision=0.75
...
MEAN:      faithful=0.81  relevant=0.89  precision=0.74
PASS/FAIL: faithful=PASS  relevant=PASS  precision=PASS
OVERALL: PASS
```

Exit code 1 if any mean score is below threshold — GitHub Actions step failure propagates automatically.

**Key decisions:**

| Layer | Choice | Rationale |
|---|---|---|
| Faithfulness | DeBERTa-large NLI | Deterministic, CPU ~200ms/sample, free, no circular reasoning risk. LLM-as-judge introduces a second LLM into the eval pipeline — latency, cost, circular reasoning. |
| Answer relevance | Sentence Transformers cosine similarity | State-of-art embeddings, offline |
| Context precision | RAGAS precision metric | Standard RAG eval benchmark |
| CI gate | GitHub Actions step | Fails workflow if any metric < threshold |

---

## The 9 Interview Questions to Answer Cold

**On QW-1 (Serve):**
1. Why LightGBM over ARIMA for time series forecasting?
2. Why multi-stage Docker? What does the builder stage actually do?
3. What does Pydantic v2 buy you over v1?

**On QW-2 (Monitor):**
4. What is PSI and why use it instead of KL divergence?
5. What's the difference between data drift and concept drift? Does your system detect both?
6. Why does Prometheus use a pull model (scrape) instead of push?

**On QW-3 (Evaluate):**
7. Why DeBERTa-large for NLI? What makes it better than BERT-base for entailment?
8. Why not use GPT-4 as your faithfulness judge?
9. What happens if the eval harness itself has a bug and always returns scores > 0.7?

---

## Upgrade Path (Post-Internship)

| Addition | Timeline | Why |
|---|---|---|
| MLflow experiment tracking + model registry | Month 2 | Model versioning discipline |
| Kubernetes (kind locally, EKS for resume) | Month 3 | Orchestration beyond Compose |
| Feature store (Feast) | Month 4 | Training/serving skew awareness |
| Kafka ingestion for real-time drift | Month 5 | Streaming ML — Atlassian/Canva territory |
| A/B model serving | Month 6 | Senior MLE territory |

---

## License

MIT
