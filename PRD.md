# Product Requirements Document — mlops-loop

**Author:** Anh Duc Tran | **Version:** 1.0 | **Date:** May 2026

---

## 1. Problem Statement

Most junior ML portfolios demonstrate model training in isolation. There is no evidence of production thinking: no serving layer, no monitoring, no quality gates. Hiring engineers at Harrison.ai, Relevance AI, and Optiver are not evaluating notebooks — they are evaluating whether a candidate can reason about a system that stays healthy in production.

This project solves that gap by building a single, coherent MLOps system that demonstrates all three phases of a production ML lifecycle as one deployable, observable, testable unit.

---

## 2. Project Identity

| Field | Value |
|---|---|
| **Project Name** | `mlops-loop` |
| **Tagline** | Serve → Monitor → Evaluate. One repo. One loop. |
| **Audience** | ML Engineering hiring managers at AU AI companies |
| **Narrative** | "I built the production ML infrastructure for Vaylo's inventory forecasting pipeline — from model serving through drift detection to LLM output quality gates." |

---

## 3. Goals

**Primary:** A single GitHub mono-repo that a recruiter can clone and run with one command, demonstrating all three MLOps phases end-to-end.

**Secondary:**
- Every architectural choice is defensible in a 45-minute technical interview
- The system is framed around a real business context (Vaylo / retail demand forecasting)
- Portfolio site links to this repo as the centrepiece project

---

## 4. The Three Components

### Component 1 — Serve (QW-1)

**What it does:** REST API that takes retail store/product/date features and returns a 30-day demand forecast.

| Layer | Choice | Rationale |
|---|---|---|
| Model | LightGBM | Tabular features with categorical encodings → gradient boosting outperforms sequence models. LSTM requires stationarity + padding for no accuracy gain. Train time: ~30s vs ~15min for LSTM. |
| API | FastAPI + Pydantic v2 | Async, auto-docs, strict type validation |
| Container | Multi-stage Dockerfile | Builder stage installs deps; runtime stage is ~120MB final image |
| Registry | GHCR via GitHub Actions | Native integration, no rate limits on public repos |
| CI | On push to `main` | lint → test → build → push |

### Component 2 — Monitor (QW-2)

**What it does:** Detects when incoming request distribution drifts from training distribution. Surfaces as Prometheus metrics, visualised in Grafana.

| Layer | Choice | Rationale |
|---|---|---|
| Drift detection | Evidently AI (PSI) | PSI is symmetric with industry-standard thresholds: <0.1 stable, 0.1–0.2 moderate, >0.2 significant. KL divergence is asymmetric — thresholds become arbitrary. PSI originates in credit risk (banking), relevant for CBA/Westpac hiring context. |
| Metrics | Prometheus `/metrics` | Industry-standard pull model |
| Dashboard | Grafana provisioned via `provisioning/` | Zero manual setup — dashboard loads on `docker compose up` |

### Component 3 — Evaluate (QW-3)

**What it does:** CLI tool and CI gate that scores any RAG pipeline on three quality dimensions. If scores drop below threshold, the Docker image is not pushed.

| Layer | Choice | Rationale |
|---|---|---|
| Faithfulness | DeBERTa-large NLI | Deterministic, CPU ~200ms/sample, free, no circular reasoning risk. LLM-as-judge introduces a second LLM into the eval pipeline — latency, cost, circular reasoning. |
| Answer relevance | Sentence Transformers cosine similarity | State-of-art embeddings, offline |
| Context precision | RAGAS precision metric | Standard RAG eval benchmark |
| CI gate | GitHub Actions step | Fails workflow if any metric < threshold |

---

## 5. Non-Goals

- Not a SaaS product — no auth, no multi-tenancy, no database persistence
- Not a research project — no novel model architecture; LightGBM + DeBERTa chosen for interview-defensibility, not SOTA
- Not a Streamlit dashboard — the Vaylo Dashboard is a separate project consuming this system

---

## 6. Success Metrics

| Metric | Target |
|---|---|
| Cold start (clone → running) | `docker compose up` in < 5 minutes |
| CI pipeline pass rate | 100% on `main` branch |
| Forecast API p95 latency | < 200ms |
| Drift detection false positive rate | < 5% on held-out reference data |
| LLM eval CI gate | Blocks bad images 100% of the time |
| Interview ability | Can explain every tool choice without notes |

---

## 7. Out of Scope (this sprint)

- Kubernetes deployment (post-internship addition)
- Feature store integration (Feast / Hopsworks)
- A/B model serving
- Real-time streaming data ingestion (Kafka / Flink)

---

## 8. Upgrade Path (Post-Internship)

| Addition | Timeline | Why |
|---|---|---|
| MLflow experiment tracking + model registry | Month 2 | Model versioning discipline |
| Kubernetes (kind locally, EKS for resume) | Month 3 | Orchestration beyond Compose |
| Feature store (Feast) | Month 4 | Training/serving skew awareness |
| Kafka ingestion for real-time drift | Month 5 | Streaming ML — Atlassian/Canva territory |
| A/B model serving | Month 6 | Senior MLE territory |

---

## 9. The 9 Interview Questions to Answer Cold

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

## 10. The 60-Second Portfolio Pitch

> "I built an end-to-end MLOps system around a retail demand forecasting use case — the same problem I was solving at Vaylo. Three components, one loop. First: a LightGBM model served via FastAPI, containerised with multi-stage Docker, pushed to GHCR via GitHub Actions. Second: a monitoring layer — Evidently AI computes PSI drift on incoming request features against the training distribution, surfaces metrics to Prometheus, visualised in Grafana. Third: a CI quality gate — a CLI tool using DeBERTa NLI to score RAG pipeline faithfulness. If the score drops below 0.7, the image doesn't ship. The whole stack runs with `docker compose up`. The point isn't the tools — it's that the system is observable, testable, and self-healing. That's what production ML looks like."
