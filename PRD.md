# Product Requirements Document — mlops-loop

**Author:** Anh Duc Tran | **Version:** 2.0 | **Date:** August 2026

**Changelog:** v2.0 expands the system from 3 components (Serve/Monitor/Evaluate) to the full 6-layer MLOps reference architecture (Data → Experiment → Registry → Serve → Monitor → Orchestration/Feedback). v1.0's three components ship unchanged; this version adds the layers that surround them.

---

## 1. Problem Statement

Most junior ML portfolios demonstrate model training in isolation. There is no evidence of production thinking: no serving layer, no monitoring, no quality gates, and — the gap this version closes — no evidence the candidate understands what happens *before* serving (data/experiment discipline) and *after* monitoring detects a problem (a real feedback loop back to retraining). Hiring engineers at Harrison.ai, Relevance AI, and Optiver are not evaluating notebooks — they are evaluating whether a candidate can reason about a system that stays healthy in production, end to end.

This project solves that gap by building a single, coherent MLOps system that demonstrates all six layers of a production ML lifecycle as one deployable, observable, testable unit — with an explicit, defensible answer for why each layer's automation stops (or doesn't stop) where it does.

---

## 2. Project Identity

| Field | Value |
|---|---|
| **Project Name** | `mlops-loop` |
| **Tagline** | Serve → Monitor → Evaluate. One repo. One loop. |
| **Audience** | ML Engineering hiring managers at AU AI companies |
| **Narrative** | "I built the production ML infrastructure for Vaylo's inventory forecasting pipeline — from data versioning and experiment tracking, through model serving and drift detection, to a closed feedback loop that proposes retrains automatically and promotes them only with human sign-off." |

---

## 3. Goals

**Primary:** A single GitHub mono-repo that a recruiter can clone and run with one command, demonstrating all six MLOps layers end-to-end, including a real (not diagram-only) drift → retrain → candidate-model feedback loop.

**Secondary:**
- Every architectural choice is defensible in a 45-minute technical interview — *including* why the loop keeps one human approval gate instead of auto-promoting to production
- The system is framed around a real business context (Vaylo / retail demand forecasting)
- No layer requires a paid cloud account to run or demo (data/model registry persistence piggybacks on GitHub itself — Releases + Actions — not S3/GCS/Postgres)
- Portfolio site links to this repo as the centrepiece project

---

## 4. The Six Layers

### L1 — Data & Feature Layer

**What it does:** Versions the raw Kaggle Store Sales dataset and formalizes feature definitions so training and serving read features from one governed source instead of an imported Python function.

| Layer | Choice | Rationale |
|---|---|---|
| Dataset versioning | DVC, remote = GitHub Release asset on this repo | No cloud account needed. `data/train.csv.dvc` (a small pointer file) is committed to git; the 122MB CSV itself lives as a GitHub Release asset. `dvc pull` in CI and locally resolves it. Swapping to S3/GCS later is a one-line `dvc remote` change — the abstraction is the point, not the specific store. |
| Feature store | Feast, file-based (local Parquet offline store + SQLite online store, no separate server) | Registers `training/features.py`'s feature definitions as `FeatureView`s so training and serving fetch identically-defined features via `feature_repo/`, closing the training/serving skew gap this PRD previously listed as future work. No standalone Feast server — keeps `docker compose up` at zero extra services for this layer. |

### L2 — Experiment Layer

**What it does:** Every training run is tracked, hyperparameters are searched (not hand-picked once), and runs are comparable.

| Layer | Choice | Rationale |
|---|---|---|
| Experiment tracking | MLflow Tracking, SQLite backend store + local artifact store | Full run/params/metrics/artifact logging without hosting a Postgres+S3-backed MLflow server — appropriate for portfolio scale. `mlflow.db` + `mlflow/artifacts/` are themselves DVC-tracked (same GitHub Release remote as L1), so state persists across local dev and CI without a long-lived server. |
| Hyperparameter search | Optuna, ~20 trials over `num_leaves`, `learning_rate`, `n_estimators` | Each trial is a nested MLflow run under one parent study run — gives real "Train → Track → Compare → Select," not a single fixed-hyperparameter run pretending to be an experiment layer. |
| UI | `mlflow server` exposed via `docker compose up` on port 5000 | Same "zero manual config, one command" bar as the rest of the repo. |

### L3 — Model Registry

**What it does:** A model earns a version and a stage (`Staging` → `Production` → `Archived`) before it's eligible to be baked into a deployable image.

| Layer | Choice | Rationale |
|---|---|---|
| Artifact/version registry | MLflow Model Registry | Every training run (including retrains triggered by L6) registers a new version under `mlops-loop-forecast`. Stage transitions are explicit and audited — this is what "model versioning discipline" (this PRD's own v1.0 Upgrade Path item) actually means, not just an image tag. |
| Deployable artifact registry | GHCR (unchanged, locked decision from v1.0) | GHCR still holds the container image — MLflow Registry and GHCR are not competing choices. MLflow tracks *which model*; GHCR ships *the container that runs it*. The image build step now pulls whichever model version is tagged `Production` in MLflow rather than a static local `model.pkl`. |

### L4 — Serve (QW-1) — *unchanged from v1.0*

**What it does:** REST API that takes retail store/product/date features and returns a 30-day demand forecast.

| Layer | Choice | Rationale |
|---|---|---|
| Model | LightGBM | Tabular features with categorical encodings → gradient boosting outperforms sequence models. |
| API | FastAPI + Pydantic v2 | Async, auto-docs, strict type validation |
| Container | Multi-stage Dockerfile | Builder stage installs deps; runtime stage ~170MB content size (measured) |
| Registry | GHCR via GitHub Actions | Native integration, no rate limits on public repos |
| Model source | Resolved from MLflow Registry (`models:/mlops-loop-forecast/Production`) at **image build time** (`promote.yml` / tag-triggered `ci.yml`), then baked into the image as `model.pkl` exactly as before | Keeps the serving container's runtime footprint, size budget, and startup path unchanged — no live dependency on MLflow being reachable in production. The registry only matters at build/promotion time, not at request time. |

### L5 — Monitor (QW-2) — *unchanged from v1.0*

**What it does:** Detects when incoming request distribution drifts from training distribution. Surfaces as Prometheus metrics, visualised in Grafana.

| Layer | Choice | Rationale |
|---|---|---|
| Drift detection | Evidently AI (PSI) | Symmetric, industry-standard thresholds (<0.1 stable, 0.1–0.2 moderate, >0.2 significant). |
| Metrics | Prometheus `/metrics` | Industry-standard pull model |
| Dashboard | Grafana, auto-provisioned | Zero manual setup |

### L6 — Orchestration & Feedback Loop

**What it does:** On a schedule (and on demand), checks drift, and — only if drift crosses the significant threshold — automatically retrains, tracks, and registers a candidate model. It does **not** automatically promote that candidate to production; it opens a comparison report and waits for a human to approve.

| Layer | Choice | Rationale |
|---|---|---|
| Orchestrator | GitHub Actions scheduled workflow (`schedule: cron` + `workflow_dispatch`), not Airflow | No persistent scheduler/infra to host — GitHub Actions is already the repo's CI engine and is one of the three components this PRD's own reference diagram names for this layer. Airflow would require standing up and operating a scheduler + metadata DB for a portfolio project with no production traffic to actually orchestrate around. |
| Drift trigger | `monitoring/evidently/drift_report.py` (existing script, reused) run against a held-out "simulated current window" (`scripts/simulate_current_window.py`) since there is no live production traffic to sample from | Reuses L5's existing detector rather than building a second one. The simulated window is clearly labeled as such — this is a portfolio demo of the mechanism, not a claim of live production drift. |
| Retrain trigger | If any feature's PSI > 0.2, run `train.py` (with Optuna search, MLflow logging — same code path as L2) inside the same workflow | One retraining code path, exercised by both manual (`python train.py`) and automated (drift-triggered) invocation — no separate "CI-only" training logic to drift out of sync. |
| Promotion gate | **Human-approved, by design.** The workflow registers the new model version as `Staging` and opens a GitHub Issue with a candidate-vs-production comparison (RMSE, PSI, MLflow run link). A separate `promote.yml` workflow (`workflow_dispatch` only — requires a human to click "Run workflow" with a version number) transitions the version to `Production` and triggers the tagged image rebuild/push. | See §11 — this is a deliberate, interview-defensible decision, not an unfinished automation. Fully automatic promotion-on-drift is a known failure mode: a bad data window (an outage, a schema change, an adversarial input pattern) can silently degrade a model with nobody watching. Every layer up to promotion is automatic; the last step keeps a person in the loop. |

### Evaluate (QW-3) — *unchanged, orthogonal to the 6-layer diagram*

**What it does:** CLI tool and CI gate that scores any RAG pipeline on three quality dimensions, independent of the forecasting model's lifecycle above. Kept as-is from v1.0 — it is a separate quality gate (for a hypothetical LLM/RAG output), not a stage of the forecasting model's L1–L6 loop.

| Layer | Choice | Rationale |
|---|---|---|
| Faithfulness | DeBERTa-large NLI | Deterministic, CPU ~200ms/sample, free, no circular reasoning risk. |
| Answer relevance | Sentence Transformers cosine similarity | State-of-art embeddings, offline |
| Context precision | RAGAS precision metric (token-overlap fallback active — see build notes) | Standard RAG eval benchmark |
| CI gate | GitHub Actions step | Fails workflow if any metric < threshold |

---

## 5. Non-Goals

- Not a SaaS product — no auth, no multi-tenancy, no user-facing database persistence
- Not a research project — no novel model architecture; LightGBM + DeBERTa chosen for interview-defensibility, not SOTA
- Not a Streamlit dashboard — the Vaylo Dashboard is a separate project consuming this system
- Not a claim of live production traffic — L6's drift trigger runs against a simulated current window, clearly labeled; this repo does not operate a public forecasting service with real users
- Not fully-automatic model promotion — see §4 L6 and §11. This is a scope decision, not a gap.

---

## 6. Success Metrics

| Metric | Target |
|---|---|
| Cold start (clone → running) | `docker compose up` in < 5 minutes |
| CI pipeline pass rate | 100% on `main` branch |
| Forecast API p95 latency | < 200ms |
| Drift detection false positive rate | < 5% on held-out reference data |
| LLM eval CI gate | Blocks bad images 100% of the time |
| `dvc pull` reproducibility | Fresh clone + `dvc pull` reproduces the exact dataset used for the committed `model.pkl` |
| Experiment reproducibility | Any MLflow run's params + code version fully reproduce its logged metrics |
| Retrain-check correctness | Workflow correctly fires (PSI > 0.2 → retrain) against an intentionally-shifted simulated window in a test run |
| Promotion safety | 0 automatic promotions to `Production` in MLflow Registry history — every `Production` version transition traces to a manually-triggered `promote.yml` run |
| Interview ability | Can explain every tool choice without notes, including why promotion isn't automatic |

---

## 7. Out of Scope (this sprint)

- Kubernetes deployment (post-internship addition)
- A/B model serving
- Real-time streaming data ingestion (Kafka / Flink)
- Hosted/managed MLflow (Databricks-managed, etc.) — self-hosted, file/SQLite-backed only
- Fully automatic production promotion (see §4 L6, §11 — deliberately out of scope, not deferred)

---

## 8. Upgrade Path (Post-Internship)

| Addition | Timeline | Why |
|---|---|---|
| Kubernetes (kind locally, EKS for resume) | Month 3 | Orchestration beyond Compose |
| Kafka ingestion for real-time drift | Month 5 | Streaming ML — Atlassian/Canva territory |
| A/B model serving | Month 6 | Senior MLE territory |
| Hosted MLflow (Postgres + S3 backend) | If/when there's real production traffic | Replaces the SQLite/DVC-backed registry once a persistent server is worth operating |

*(MLflow experiment tracking, model registry, and Feast feature store — previously listed here in v1.0 — are now delivered in this version; see §4 L1–L3.)*

---

## 9. The 15 Interview Questions to Answer Cold

**On L1 (Data & Features):**
1. Why DVC with a GitHub Release remote instead of just committing the CSV to git, or ignoring it entirely?
2. What problem does a feature store actually solve that a shared Python function doesn't?

**On L2 (Experiment):**
3. Why SQLite-backed MLflow instead of a hosted tracking server?
4. What does Optuna give you that a manual hyperparameter sweep doesn't?

**On L3 (Registry):**
5. Why do you need both MLflow Registry *and* GHCR — isn't that redundant?
6. What does a "Staging → Production" stage transition actually gate?

**On L4 (Serve):**
7. Why LightGBM over ARIMA for time series forecasting?
8. Why multi-stage Docker? What does the builder stage actually do?

**On L5 (Monitor):**
9. What is PSI and why use it instead of KL divergence?
10. What's the difference between data drift and concept drift? Does your system detect both?

**On L6 (Orchestration & Feedback Loop):**
11. Walk me through what happens, end to end, from drift detection to a new model serving traffic.
12. Why doesn't the loop auto-promote to production? Isn't that the whole point of a "closed loop"?
13. What happens if the retrain job itself is triggered by bad data — a schema change, an outage, an adversarial input pattern?
14. Why GitHub Actions `schedule:` instead of Airflow for the orchestrator?

**On Evaluate (QW-3):**
15. Why not use GPT-4 as your faithfulness judge?

---

## 10. The 60-Second Portfolio Pitch

> "I built an end-to-end MLOps system around a retail demand forecasting use case — the same problem I was solving at Vaylo. It covers all six layers of the standard MLOps reference architecture, not just serving. Data is versioned with DVC, features are governed through Feast, every training run is tracked and hyperparameter-searched with MLflow and Optuna, and models are promoted through a real registry with explicit stages. The forecast API serves whatever model is currently tagged Production, containerised and shipped through GHCR. Evidently watches for drift and surfaces it to Prometheus and Grafana. And the loop actually closes: a scheduled GitHub Actions workflow detects significant drift, automatically retrains and registers a candidate model, and opens a comparison report — but it deliberately stops short of auto-promoting that candidate. A human has to approve the last step. That gate isn't a gap, it's the answer to 'what happens when your automated retraining pipeline gets fed bad data' — which is exactly the question a real MLOps team asks before they'd trust this in production."

---

## 11. Design Note — Why the Loop Keeps a Human Gate

This is the single most-asked question this version of the system will get in an interview, so it gets its own section instead of hiding in a table cell.

**The naive version of "closed loop":** drift detected → retrain → auto-deploy. This is achievable in an afternoon and looks impressive in a diagram.

**Why it's the wrong default:** the retraining trigger (drift) and the retraining *input* (fresh data) are the same untrusted surface. If the thing causing drift is a broken upstream pipeline, a schema change, a promo-week distribution shift that isn't representative, or an adversarial input pattern, auto-promoting the resulting model ships a worse model *automatically and silently* — the failure mode this whole system exists to prevent.

**What this system does instead:** every step up to and including registering a new model version is fully automated (§4 L6). The only manual step is the transition from `Staging` to `Production` in MLflow Registry, performed by running `promote.yml` with an explicit version number after reading the auto-generated comparison report. This mirrors how mature MLOps teams actually operate CD-for-ML pipelines — automated up to a release candidate, human-gated at the production boundary — rather than either extreme (fully manual retraining, or fully automatic promotion with no review).

If a future version of this project wants true full automation, the gate is a single `if:` condition in `promote.yml` — the design isn't harder to make fully automatic, it's a considered choice to not.
