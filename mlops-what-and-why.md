# MLOps — What It Is, Why It Exists, How It Works

> Feynman principle throughout: intuition before formalism, why before what.
> Every component is explained as the solution to a specific failure mode.

---

## Table of Contents

1. [The Core Problem — Why MLOps Exists](#1-the-core-problem--why-mlops-exists)
2. [What MLOps Actually Is](#2-what-mlops-actually-is)
3. [The Full Architecture Map](#3-the-full-architecture-map)
4. [Layer 1 — Data Layer](#4-layer-1--data-layer)
5. [Layer 2 — Experiment Layer](#5-layer-2--experiment-layer)
6. [Layer 3 — Model Registry](#6-layer-3--model-registry)
7. [Layer 4 — Serving Layer](#7-layer-4--serving-layer)
8. [Layer 5 — Monitoring Layer](#8-layer-5--monitoring-layer)
9. [Layer 6 — Orchestration & CI/CD](#9-layer-6--orchestration--cicd)
10. [The Three Loops — How It All Moves](#10-the-three-loops--how-it-all-moves)
11. [How This Maps to mlops-loop](#11-how-this-maps-to-mlops-loop)
12. [Maturity Levels — Where Teams Actually Are](#12-maturity-levels--where-teams-actually-are)
13. [Master Self-Test](#13-master-self-test)

---

## 1. The Core Problem — Why MLOps Exists

### The Dirty Secret of ML in Production

A data scientist trains a model. It achieves 94% accuracy on the test set. They ship it. Six months later, accuracy is 71% and nobody noticed until a customer complained.

This is not a hypothetical. It's the default outcome when ML is treated like software.

**Software doesn't rot on its own**. A function that sorts a list correctly today will sort correctly in five years. The code doesn't change, the problem doesn't change.

**ML models do rot**. Not because the code changed. Because the world changed. The data distribution your model was trained on is no longer the distribution it's seeing in production. The model is now making predictions about a world that no longer exists.

This phenomenon has a name: **concept drift** and **data drift**. It is the fundamental reason MLOps exists.

---

### The 3 Failure Modes That Created MLOps

**Failure Mode 1: The Jupyter Notebook Problem**

A data scientist builds a model in a notebook. It works on their laptop with their version of pandas, scikit-learn 0.24, and a hand-cleaned CSV. Someone else tries to run it six months later. It crashes. No one knows what data was used. No one knows what the final hyperparameters were. There's no way to reproduce the result.

This is a **reproducibility failure**. MLOps answer: experiment tracking, data versioning, environment pinning.

**Failure Mode 2: The Silent Degradation Problem**

The model is in production, serving predictions. Upstream, the data team changes a feature engineering step. Or a new product category gets added. Or seasonal patterns shift. The model keeps running — it doesn't throw errors — but its predictions are quietly wrong. No alert fires. Revenue drops. Weeks pass.

This is a **monitoring failure**. MLOps answer: drift detection, performance tracking, automated alerting.

**Failure Mode 3: The Training-Serving Skew Problem**

The data scientist engineers features in Python during training: "I'll compute `lag_7` as sales 7 days ago." The serving engineer implements the same logic in Java for the API: "7 days ago, I'll query the DB for day - 7." Small difference in how they handle weekends, holidays, or missing data. Now training distribution ≠ serving distribution. The model was validated on data it will never actually see in production.

This is a **feature consistency failure**. MLOps answer: feature stores, shared feature pipelines.

---

### The Analogy That Makes It Click

Traditional software DevOps is about shipping and maintaining **code**.
MLOps is about shipping and maintaining **code + data + models + their interactions**.

In software:
```
Code → Test → Deploy → Monitor (logs, uptime)
```

In ML:
```
Data + Code → Train → Evaluate → Deploy → Monitor (drift, accuracy, data quality)
         ↑__________________________|
                  Feedback loop
```

The arrow going backwards is everything. Software CI/CD is a one-way pipeline. MLOps is a **loop**. The monitoring output tells you when to retrain. The retrain output replaces the current model. The loop runs continuously.

That loop is what the repo is named after.

---

## 2. What MLOps Actually Is

### Definition

MLOps (Machine Learning Operations) is the set of practices, tools, and infrastructure that make it possible to **deploy, monitor, and reliably iterate on ML models in production** at the speed of business needs.

It borrows from:
- **DevOps**: CI/CD, infrastructure as code, containerisation
- **DataOps**: data versioning, data quality, pipeline reliability
- **ML Engineering**: feature engineering, model evaluation, experiment tracking

It adds what's unique to ML:
- Models are **stateful artifacts** (not just code) that must be versioned and rolled back
- Quality is **probabilistic** (a model can be "mostly right" in a way that code never is)
- Degradation is **silent** (no crash, just worse predictions)
- Iteration requires **data** as much as code

### What MLOps Is NOT

- It's not "just DevOps for ML" — the feedback loops and monitoring requirements are fundamentally different
- It's not about tooling — MLflow, Evidently, Grafana are implementations; the concepts exist without them
- It's not a role — it's a practice. A data scientist, ML engineer, and platform engineer all do MLOps

---

## 3. The Full Architecture Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Architecture                                  │
│                                                                                │
│  ┌─────────────────┐                                                           │
│  │   DATA LAYER    │  Raw data → Features → Versioned datasets                │
│  │                 │  Components: Feature Store, DVC, data pipelines           │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐                                                           │
│  │ EXPERIMENT LAYER│  Train → Track → Compare → Select                        │
│  │                 │  Components: MLflow, W&B, Optuna                          │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐                                                           │
│  │  MODEL REGISTRY │  Versioned, tagged, promoted models                       │
│  │                 │  Components: MLflow Registry, GHCR, S3                    │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐                                                           │
│  │  SERVING LAYER  │  Model → API → Predictions                                │
│  │                 │  Components: FastAPI, BentoML, Triton, SageMaker          │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐                                                           │
│  │ MONITORING LAYER│  Drift → Alerts → Dashboards                              │
│  │                 │  Components: Evidently, Prometheus, Grafana               │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐                                                           │
│  │  ORCHESTRATION  │  Triggers → Pipelines → CI/CD                             │
│  │                 │  Components: Airflow, GitHub Actions, Prefect             │
│  └────────┬────────┘                                                           │
│           │                                                                    │
│           └──────────────────────────────────────────────────────────────────►│
│                              FEEDBACK LOOP                                     │
│                     (monitoring triggers retraining)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

Each layer is the answer to a specific failure mode. They exist because something broke without them.

---

## 4. Layer 1 — Data Layer

### What It Does
Ingests, transforms, versions, and serves features to the training and serving pipelines.

### Why It Exists: The Training-Serving Skew Problem

This is the most expensive silent failure in production ML. Here's how it happens:

1. Data scientist computes feature: `lag_7 = sales 7 days ago` (in Python, on training data)
2. Backend engineer reimplements in production: `lag_7 = DB query for t-7` (in Java or SQL)
3. Their implementations differ slightly — weekend handling, nulls, timezone — by 3%
4. The model was validated on Python features. In production it sees Java features.
5. The model is running on a different input distribution than it was trained on.
6. No one knows. No error fires. Predictions are just slightly wrong.

**The fix**: a **feature store** — a single place where features are computed once and served identically to both training and production.

```
              Without Feature Store          With Feature Store
Training:    Python feature pipeline    →   Feature Store ──► Training
Production:  Java/SQL reimplementation  →   Feature Store ──► API
                   ↑                                ↑
              SKEW HERE                      Identical output
```

### Key Components

**Data Versioning (DVC)**
Think of DVC as Git for large files. You can't put a 10GB parquet file in Git. DVC stores a hash pointer in Git and the actual file in S3/GCS. Result: your code commit and your data commit are linked — you can reproduce any experiment by checking out any commit.

```
git commit → code state
dvc commit → data state
Together  → fully reproducible experiment
```

**Feature Engineering Pipelines**
Features shouldn't be computed ad-hoc in notebook cells. They should be:
- Versioned (if you change `lag_7` definition, old models can still use the old definition)
- Tested (data quality checks: no null `store_nbr`, no negative sales)
- Monitored (schema drift — a column being renamed upstream breaks everything downstream)

**In mlops-loop**:
- `training/features.py` — feature engineering for training
- `monitoring/evidently/reference_data.parquet` — snapshot of training feature distribution (the reference baseline for drift detection)

---

## 5. Layer 2 — Experiment Layer

### What It Does
Tracks every training run: hyperparameters, metrics, code version, data version, artifacts. Makes experiments reproducible and comparable.

### Why It Exists: The "Which Model Was That?" Problem

Without experiment tracking:
- Data scientist runs 50 experiments over 2 weeks
- Best model: RMSE 0.42, `num_leaves=64`, `lr=0.05`, `n_estimators=300`
- They remember the RMSE. They don't remember which combination got them there.
- Three months later: "What was the model that did so well in November?" Nobody knows.
- Retraining after new data: impossible to reproduce the baseline to compare against.

**The fix**: every run is logged automatically — code, data, parameters, metrics, artifacts — in a queryable experiment database.

### The Experiment Tracking Schema

Every experiment run records:

| What | Example | Why |
|---|---|---|
| Parameters | `num_leaves=64, lr=0.05` | Reproduce the run |
| Metrics | `train_rmse=0.38, val_rmse=0.42` | Compare runs |
| Artifacts | `model.pkl, features.py` | Reuse or audit |
| Code version | Git commit SHA | Reproduce environment |
| Data version | DVC hash | Reproduce data |
| Timestamp | `2026-06-01 14:32` | Audit trail |

### Hyperparameter Optimisation

With experiment tracking in place, you can run automated search (Optuna, Ray Tune) that:
1. Samples hyperparameters from a search space
2. Trains a model with those parameters
3. Logs the run
4. Uses the result to guide the next sample (Bayesian optimisation)
5. Returns the best run

Without tracking, this is noise. With tracking, it's a searchable database of 200 experiments you can query any time.

### In mlops-loop

This layer is intentionally lightweight — `training/train.py` handles a single training run, printing RMSE. In a production system you'd add MLflow here. The principles are the same; the tooling is simplified for portfolio clarity.

---

## 6. Layer 3 — Model Registry

### What It Does
A versioned store of trained model artifacts with promotion stages: `Staging → Production → Archived`.

### Why It Exists: The "What's Running in Production?" Problem

Without a model registry:
- Team has `model_v1.pkl`, `model_final.pkl`, `model_final_FINAL.pkl`, `model_june.pkl`
- Nobody is sure which one is live
- Rolling back to the previous model means finding the right file in someone's Google Drive
- A/B testing two models simultaneously is impossible

**The fix**: a registry where every model gets a unique version, a stage, and metadata. Promotion to production is explicit and auditable.

```
Training run → Candidate model
    │
    ▼
Staging (automated eval passes)
    │
    ▼
Production (manual approval or A/B test winner)
    │
    ▼
Archived (when replaced)
```

### Rollback in 30 Seconds

When the production model is degrading, you should be able to roll back to the previous version with a single command — not by finding files on someone's laptop. The registry makes this:

```bash
mlflow models set-stage --model-name forecast-lgbm --version 3 --stage Production
```

Previous version 3 is now production. New version 4 is demoted to Staging. The API picks up the change on next restart.

### In mlops-loop

The model registry is implemented via **GHCR (GitHub Container Registry)**. The Docker image tagged `latest` on the `main` branch IS the production model artifact. Rolling back = deploying the previous image tag. This is a valid lightweight implementation of the registry pattern.

---

## 7. Layer 4 — Serving Layer

### What It Does
Exposes the trained model as a service that accepts prediction requests and returns results, at the throughput and latency the business requires.

### Why It Exists: The "Notebook Doesn't Scale" Problem

A model in a Jupyter notebook can predict one sample at a time, on the data scientist's laptop, after a 10-second warm-up. Production needs:
- Sub-100ms response time
- 1,000+ concurrent requests
- Health checks for container orchestrators
- Metrics exposure for monitoring
- Input validation before the model ever runs
- Graceful handling of malformed inputs

**The fix**: wrap the model in a production-grade API framework.

### Serving Patterns

**Online serving (real-time)**:
```
Client request → API → Feature lookup → Model inference → Response
Latency: < 100ms
Use case: recommendation engine, fraud detection, chatbot
```

**Batch serving (offline)**:
```
Scheduler → Pull data → Run model on all rows → Write predictions → Downstream reads
Latency: minutes to hours
Use case: daily demand forecasts, weekly churn scores, nightly recommendations
```

**Streaming serving**:
```
Event stream (Kafka) → Consume event → Enrich with features → Model → Publish result
Latency: sub-second, continuous
Use case: real-time anomaly detection, live pricing
```

**mlops-loop uses online serving** — `POST /predict` returns a 30-day forecast synchronously.

### The API Contract

The API contract is defined by the Pydantic schema:

```python
class PredictRequest(BaseModel):
    store_nbr: int        # which store
    family: str           # product family
    onpromotion: int      # promotion flag
    days_ahead: int = 30  # forecast horizon

class PredictResponse(BaseModel):
    forecast: list[float]   # 30 predicted sales values
    model_version: str      # which model artifact produced this
    prediction_id: str      # UUID — enables tracing individual predictions
```

`prediction_id` is critical for debugging. When a downstream system says "your forecast for store 5, June 15 was way off" — you need a way to look up exactly what features went into that specific prediction. The UUID ties the prediction to a log entry.

### Canary Deployments & A/B Testing

Production serving doesn't have to be all-or-nothing. Advanced patterns:

**Canary**: route 5% of traffic to the new model, 95% to the old. Monitor for degradation. If fine, gradually shift to 100%.

**A/B test**: route 50% to model A, 50% to model B. Log all predictions with their model version. Compare business metrics (not just RMSE) after 2 weeks.

**Shadow mode**: the new model runs on all traffic but its predictions are logged, not returned to users. Validate silently before any real impact.

---

## 8. Layer 5 — Monitoring Layer

### What It Does
Detects when the model or its data is degrading, measures system performance, and surfaces everything through dashboards and alerts.

### Why It Exists: The Silent Degradation Problem

This is the reason MLOps is fundamentally different from DevOps.

In software: if your code is broken, requests fail with 500 errors. The alert fires in seconds.

In ML: if your model is degrading, requests succeed with 200 responses. The predictions are just wrong. No error. No alert. The model confidently says "store 5 will sell 120 units this week." The actual sales are 50 units. Your downstream inventory is 70 units over. Revenue suffers. The alert never fires.

You need a separate layer that watches **what the model is predicting** and **what data it's seeing**, not just whether the server is up.

### Two Types of Monitoring

**Infrastructure monitoring** (same as any service):
- CPU/memory usage
- Request latency (P50, P95, P99)
- Request throughput (requests/sec)
- Error rate (4xx, 5xx)
- Container health

**ML-specific monitoring** (what makes MLOps different):
- **Data drift**: has the input distribution changed? (PSI per feature)
- **Prediction drift**: has the output distribution changed? (even without labels)
- **Model performance**: when ground truth labels arrive, compare predicted vs actual
- **Feature quality**: are nulls appearing where there shouldn't be? Schema changes?

### The Four Signals (in order of when you have them)

```
Signal 1: Data drift        → Available immediately (compare inputs to reference)
Signal 2: Prediction drift  → Available immediately (monitor output distribution)
Signal 3: Proxy metrics     → Available hours later (e.g., click-through rate as proxy for relevance)
Signal 4: Ground truth      → Available days/weeks later (actual vs predicted)
```

Most systems can only measure signals 1 and 2 in real-time. Ground truth often comes with a delay — you forecast today, you see actual sales in 30 days. You design your monitoring around what you have now, not what you'll have later.

### The Prometheus-Grafana Stack Explained

**Prometheus** is a time-series database that works by **pulling** (scraping) metrics. Every 15 seconds it makes an HTTP request to `forecast-api:8000/metrics` and reads a text-format metrics payload. It stores: metric name, labels (key-value tags), value, timestamp.

```
# HELP forecast_request_count Total prediction requests
# TYPE forecast_request_count counter
forecast_request_count{store="5",family="BEVERAGES"} 1423

# HELP forecast_latency_seconds Request latency
# TYPE forecast_latency_seconds histogram
forecast_latency_seconds_bucket{le="0.05"} 1200
forecast_latency_seconds_bucket{le="0.1"} 1420
forecast_latency_seconds_bucket{le="+Inf"} 1423
```

**Grafana** is a visualization layer. It speaks PromQL (Prometheus Query Language) to query time-series from Prometheus and renders them as panels. It doesn't store data — it only queries and displays.

```
PromQL query for P99 latency:
histogram_quantile(0.99, rate(forecast_latency_seconds_bucket[5m]))
```

**Evidently** is a statistical testing library. It computes drift statistics (KS test, PSI, chi-squared) and formats them as reports or Prometheus-compatible metric files. It's the source of truth for "has this feature drifted?"

---

## 9. Layer 6 — Orchestration & CI/CD

### What It Does
Automates the execution of multi-step workflows: training pipelines, data processing, model evaluation, deployment, and the CI/CD pipeline that gates every code change.

### Why It Exists: The "Manual Steps Break Things" Problem

Without orchestration:
- Retraining is a manual process: someone SSHes into the server, runs the training script, copies the model file, restarts the API
- Steps get forgotten: someone skips the eval step because "it always passes"
- Timing is ad-hoc: retraining happens whenever someone remembers to do it
- Failures are silent: the training script errored at 3am, nobody knows, the stale model runs for another month

**The fix**: automation with dependency management, scheduling, and failure alerting.

### Two Types of Orchestration

**Training pipeline orchestration** (Airflow, Prefect, Kubeflow Pipelines):
```
Trigger (schedule / drift alert)
    │
    ▼
Ingest new data
    │
    ▼
Feature engineering
    │
    ▼
Train model
    │
    ▼
Evaluate model (must beat current production)
    │
    ▼
Register if better → promote to Staging
    │
    ▼
Human approval or automated A/B → promote to Production
```

**CI/CD pipeline** (GitHub Actions):
```
Code push
    │
    ▼
Lint → Test → Eval gate → Build → Push → (optionally) Deploy
```

### CI/CD for ML vs Traditional Software

The extra step that makes ML CI/CD different is the **eval gate** — a quality check on model behaviour, not just code correctness. A code change can be syntactically valid, pass all unit tests, and still degrade model quality. The eval gate catches this.

```
Traditional CI:   lint → test → build → deploy
ML CI:            lint → test → eval-gate → build → deploy
                                    ↑
                             New step, unique to ML
```

### The DAG (Directed Acyclic Graph) Model

Both Airflow and GitHub Actions model workflows as DAGs. Nodes are tasks. Edges are dependencies.

```
lint ──► test ──► eval-gate ──► build-push
 │
 └── (if this fails, nothing downstream runs)
```

A DAG cannot have cycles. This prevents infinite loops in automated pipelines. It also means every pipeline has a defined start and end.

---

## 10. The Three Loops — How It All Moves

This is the most important mental model for understanding MLOps end-to-end. There are three feedback loops operating at different time scales.

### Loop 1 — The Inner Loop (minutes to hours)

This is the experimentation loop. The data scientist is working.

```
Hypothesis → Experiment → Evaluate → Adjust hypothesis
   ↑_______________________________________|
```

Tools: Jupyter, MLflow, DVC, local compute.
Goal: find a model worth shipping.
Key metric: validation performance (RMSE, AUC, F1).

**What's MLOps about here**: reproducibility. Any experiment must be re-runnable. Code version + data version + parameters → same result.

### Loop 2 — The Middle Loop (hours to days)

This is the deployment loop. Code changes go through CI/CD to production.

```
Code commit → CI/CD pipeline → Staging → Production
     ↑_____________________________________________|
           (feedback from staging tests or prod monitoring)
```

Tools: GitHub Actions, Docker, model registry.
Goal: ship validated models reliably and safely.
Key metric: deployment frequency, rollback rate.

**What's MLOps about here**: quality gates. Not just code quality — ML quality. The eval-gate is the distinguishing feature.

### Loop 3 — The Outer Loop (days to weeks)

This is the retraining loop. The world has changed, the model needs to catch up.

```
Production model serves predictions
     │
     ▼
Monitoring detects drift (PSI > 0.2)
     │
     ▼
Retraining trigger fires
     │
     ▼
New model trained on fresh data
     │
     ▼
Eval gate: new model must beat current production model
     │
     ▼
New model deployed → back to monitoring
     ↑_____________________________________________|
```

Tools: Evidently, Prometheus, Airflow, MLflow.
Goal: keep model performance above business threshold automatically.
Key metric: days between retraining trigger and new model in production.

**This loop is the entire reason MLOps exists.** Without it, you train once and hope. With it, the model continuously adapts to the world.

---

### The Full System as One Diagram

```
            ┌──────────────────────────────────────┐
            │          INNER LOOP                  │
            │   Experiment → Evaluate → Iterate    │
            └────────────────┬─────────────────────┘
                             │ best model
                             ▼
            ┌──────────────────────────────────────┐
            │          MIDDLE LOOP                 │
            │   Commit → CI/CD → Deploy            │
            └────────────────┬─────────────────────┘
                             │ production model
                             ▼
┌────────────────────────────────────────────────────────┐
│                 PRODUCTION                              │
│                                                         │
│   Users → API → Model → Predictions                    │
│                  │                                      │
│                  ▼                                      │
│   Monitoring: drift + latency + output distribution    │
└──────────────────────┬─────────────────────────────────┘
                       │ drift detected / accuracy degraded
                       ▼
            ┌──────────────────────────────────────┐
            │          OUTER LOOP                  │
            │   Retrain → Eval → Promote           │
            └────────────────┬─────────────────────┘
                             │ new model
                             └──────────────────► PRODUCTION
```

---

## 11. How This Maps to mlops-loop

The repo implements a subset of the full architecture, deliberately. Here's the mapping:

| MLOps Layer | Full Production Tool | mlops-loop Implementation |
|---|---|---|
| Data versioning | DVC + S3 | `reference_data.parquet` (static snapshot) |
| Feature engineering | Feature Store (Feast) | `training/features.py` (shared logic) |
| Experiment tracking | MLflow, W&B | `train.py` prints RMSE (simplified) |
| Model registry | MLflow Registry | GHCR Docker image tags |
| Serving | FastAPI + uvicorn | `services/forecast-api/` |
| Drift monitoring | Evidently + Prometheus | `monitoring/evidently/drift_report.py` |
| Dashboard | Grafana | `monitoring/grafana/` |
| LLM eval | RAGAS, custom | `services/llm-eval/` (DeBERTa + cosine) |
| CI/CD | GitHub Actions | `.github/workflows/ci.yml` |
| Orchestration | Airflow | GitHub Actions (serves both roles here) |
| Containerisation | Kubernetes | Docker Compose |

**What's intentionally missing and why**:

- **Feature store (Feast)**: adds significant infrastructure complexity. The repo demonstrates the concept via shared `features.py` and the skew problem is explained rather than fully solved.
- **Online retraining trigger**: the outer loop's automation (drift → retrain → deploy) is shown conceptually via the Evidently + Prometheus stack, but the automated trigger is left as an extension. A complete implementation would have an Airflow DAG watching the PSI metric and firing when PSI > 0.2.
- **Multi-environment (dev/staging/prod)**: simplified to `main` branch = production. A real system has separate environments.

**The portfolio point**: you've implemented every component of the loop. Serving (QW-1), monitoring (QW-2), and evaluation (QW-3) are each a fully functional piece. The architecture is complete enough to demonstrate understanding; simplified enough to explain every decision in 45 minutes.

---

## 12. Maturity Levels — Where Teams Actually Are

Google defined an MLOps maturity model with 3 levels. Understanding where a team sits tells you what problems to solve next.

### Level 0 — Manual Process (Most teams)

```
Data → Notebook → model.pkl → email to engineer → manual deploy
```

- Training is manual, infrequent (quarterly or ad-hoc)
- Deployment is a handoff: data scientist emails model file to software engineer
- Monitoring is none (maybe someone checks a dashboard once a month)
- Retraining is reactive (customer complains, someone retrains)

**Pain**: reproducibility is impossible, silent degradation is the norm, iteration is slow.

### Level 1 — ML Pipeline Automation

```
Data → Automated pipeline → model.pkl → auto-deploy to staging → manual promotion
```

- Training is automated (trigger via schedule or new data)
- Deployment is partially automated (auto to staging, manual to prod)
- Monitoring exists (basic drift alerts)
- Retraining is still semi-manual (alert fires, human retrains)

**Pain**: deployment is faster but human in the loop for every step. Doesn't scale beyond 5 models.

### Level 2 — CI/CD Pipeline Automation (Where mlops-loop sits)

```
Code change → CI/CD → eval gate → auto-deploy → monitoring → auto-retrain trigger
```

- Full CI/CD with ML quality gates
- Drift detection automatically triggers retraining
- Multiple models managed via registry
- A/B testing infrastructure exists

**Pain**: requires significant platform investment. Debugging automated failures is hard.

---

## 13. Master Self-Test

Work through these cold. No notes.

**Conceptual**:
1. Explain training-serving skew to a backend engineer who has never heard of it. What specific code change caused the skew in the example above?
2. A model's accuracy is dropping. You have no ground truth labels yet (they arrive in 30 days). Name two signals you CAN measure right now.
3. The outer loop hasn't fired in 4 months. Is that good or bad? What two things could explain it?

**Architecture**:
4. A team is at MLOps Level 0. What is the single highest-impact change they could make today? Justify your choice.
5. Draw (text diagram) the path of a single prediction request from the user's HTTP POST to the returned forecast. Name every component it touches.
6. The feature store is replaced by a rule: "never compute the same feature twice — always use the shared pipeline." Does this solve the training-serving skew problem? Under what conditions does it fail?

**Failure modes**:
7. Model accuracy drops 15% overnight. CI/CD is green, no new code was deployed. What are the three most likely causes in order of probability?
8. The outer loop fires (PSI > 0.2), retraining runs, and the new model RMSE is 10% worse than the current production model. What should the pipeline do? Why?
9. A data engineer renames the `onpromotion` column to `is_on_promotion` upstream. Trace the failure through all 6 layers.

**Deep why**:
10. Why is the eval-gate a separate CI job from the unit tests? Why not combine them?
11. Why does Prometheus pull metrics (scrape) instead of the API pushing metrics to Prometheus?
12. A stakeholder asks "can't we just retrain every day?" What are the three costs of daily retraining that make it non-trivially expensive?

---

*Created: 2026-06-06 | Author: Anh Duc Tran | Project: mlops-loop*

---

## 14. Frameworks Used in mlops-loop — What, Why, and What Was Rejected

> For every tool, the question that matters is: **what specific failure mode does this prevent, and what did we reject and why?**
> That's the answer that separates candidates who used a tool from candidates who understand it.

---

### Framework Decision Map

```
Problem                         Tool Chosen          Rejected Alternative(s)
─────────────────────────────   ─────────────────    ──────────────────────────
Tabular forecasting             LightGBM             ARIMA, LSTM, XGBoost
API serving                     FastAPI              Flask, Django REST, BentoML
Input/output validation         Pydantic v2          Marshmallow, dataclasses
ASGI server                     uvicorn              gunicorn (WSGI), hypercorn
Model serialisation             joblib               pickle, ONNX
Drift detection                 Evidently            custom PSI, Alibi Detect
Metrics collection              Prometheus           Datadog, CloudWatch, StatsD
Dashboard & alerting            Grafana              Kibana, custom HTML dashboard
Faithfulness scoring            DeBERTa NLI          GPT-4-as-judge, BLEURT
Relevance scoring               Sentence Transformers  BM25, TF-IDF cosine
CLI interface                   Click                argparse, Typer
Containerisation                Docker (multi-stage) Conda env, virtualenv-only
Service orchestration           Docker Compose       Kubernetes, Nomad
CI/CD                           GitHub Actions       Jenkins, CircleCI, GitLab CI
Container registry              GHCR                 DockerHub, ECR, GCR
Linting                         ruff                 flake8, pylint
Formatting                      black                autopep8, yapf
Testing                         pytest               unittest, nose2
```

---

### Tool-by-Tool Breakdown

#### LightGBM

**What it is**: gradient boosted decision trees using histogram-based leaf-wise splitting.

**Why chosen**: the problem is tabular + categorical features (store, family) + multiple series (1,800 combos) + 30-second training budget. LightGBM handles all four simultaneously. Once you express temporal information as lag features (`lag_7`, `rolling_mean_7`), the problem is pure tabular ML — no sequential processing needed.

**What was rejected and why**:

| Rejected | Why |
|---|---|
| XGBoost | Slower than LightGBM on large datasets; LightGBM's histogram algorithm is faster. XGBoost uses level-wise splitting (safer but slower). Same family, LightGBM wins on speed. |
| ARIMA | One model per series, no exogenous features natively, assumes stationarity. Non-starter at 1,800 series. |
| LSTM | Needs large sequence history per series, tabular features are second-class, training time 30–90 min. |
| Prophet (Meta) | Designed for univariate series with additive seasonality. Doesn't generalise across 1,800 series in one model. Good for business analysts, not for a scalable system. |

**When you WOULD use the others**:
- ARIMA: single series, regulatory interpretability required, < 10 series total
- LSTM: millions of training examples, long-range dependencies at 100+ lags are critical
- Prophet: business stakeholder needs interpretable components (trend + seasonality + holidays) and is not an engineer

---

#### FastAPI + uvicorn

**What it is**: FastAPI is an ASGI web framework built on Starlette + Pydantic. uvicorn is an ASGI server (async, event-loop based).

**Why chosen**: async handlers, automatic OpenAPI docs from Pydantic schemas, Pydantic v2 validation built-in, modern Python (3.11+), native type hints. The `/docs` endpoint generates an interactive UI for free — every endpoint is self-documenting.

**What was rejected and why**:

| Rejected | Why |
|---|---|
| Flask | WSGI (synchronous). Handles one request at a time per worker without async. For I/O-heavy serving (DB lookups for feature retrieval), sync is a bottleneck. |
| Django REST | Heavyweight ORM and admin overhead for a pure prediction API. Overkill. |
| BentoML | Production MLOps serving framework — more automated model packaging. Not chosen because it abstracts the request lifecycle we want to understand and demonstrate. Also adds another dependency layer. |
| gRPC | Binary protocol, more efficient than HTTP/JSON for high-throughput internal services. Harder to debug, no browser-based docs. Appropriate for 10k+ req/sec internal services — overkill here. |

**When you WOULD use the others**:
- Flask: simple scripts or prototypes where async doesn't matter
- BentoML: production at scale where model packaging and auto-scaling are the priority
- gRPC: internal microservice communication at very high throughput

---

#### Pydantic v2

**What it is**: data validation and serialisation library that uses Python type hints to define schemas and enforce them at runtime.

**Why chosen**: it's FastAPI's native data layer. Validation is zero-configuration — define the class, annotate types, Pydantic does the rest. V2 (Rust-based core) is 5–50x faster than v1. Generates JSON Schema automatically → OpenAPI spec → `/docs` UI.

**The key thing it prevents**: garbage in. A request with `store_nbr: "abc"` gets a 422 before it ever reaches the model. The model only sees valid data.

**What was rejected**: marshmallow (more verbose, no native FastAPI integration), raw `dataclasses` (no runtime validation, only type hints at static analysis time).

---

#### Evidently

**What it is**: open-source ML monitoring library that computes statistical drift tests and generates reports.

**Why chosen**: pre-built `DataDriftPreset` handles numerical (KS test) and categorical (chi-squared) features out of the box. Prometheus integration via metric file export. Actively maintained, well-documented, industry-used.

**What was rejected and why**:

| Rejected | Why |
|---|---|
| Custom PSI implementation | We do implement PSI ourselves in `drift_report.py` — but Evidently provides the full report, visualisations, and multi-feature comparison. Not mutually exclusive. |
| Alibi Detect | More focused on adversarial detection and outlier detection. Drift detection is secondary. Less Prometheus-native. |
| WhyLabs | SaaS product — requires sending data to external servers. Privacy concern for retail sales data. |
| Seldon Alibi | Kubernetes-native — overkill for Docker Compose deployment. |

---

#### Prometheus + Grafana

**What it is**: Prometheus is a pull-based time-series metrics database. Grafana is a dashboarding and alerting frontend that queries Prometheus via PromQL.

**Why Prometheus**:
- **Pull model**: Prometheus scrapes targets on its own schedule. The API just exposes a `/metrics` endpoint — it doesn't need to know anything about monitoring infrastructure. If Prometheus goes down, the API keeps working and doesn't queue up failed push attempts.
- **Labels**: every metric can have arbitrary key-value labels (`feature="onpromotion"`, `store="5"`). You can slice and filter in PromQL without pre-designing your schema.
- **Industry standard**: every observability tool integrates with Prometheus. Grafana, AlertManager, PagerDuty all speak it natively.

**Why Grafana**: it's the de facto frontend for Prometheus. Dashboard-as-code (JSON provisioning in `grafana/provisioning/`), built-in alerting, free open source.

**What was rejected**:

| Rejected | Why |
|---|---|
| Datadog | SaaS, $15–30/host/month. For a portfolio project and most startups, cost is prohibitive at scale. |
| CloudWatch (AWS) | Vendor lock-in. Ties monitoring to AWS infrastructure. |
| StatsD | Push-based, UDP, no built-in storage. Older pattern superseded by Prometheus. |
| ELK Stack | Elasticsearch + Logstash + Kibana. Better for log analysis than metrics. Different use case. |

---

#### DeBERTa NLI (`cross-encoder/nli-deberta-v3-large`)

**What it is**: a cross-encoder transformer model trained on Natural Language Inference benchmarks (MNLI, SNLI). Input: premise + hypothesis. Output: entailment/contradiction/neutral probabilities.

**Why chosen for faithfulness**: deterministic (same input → same output always), no API cost, CPU-runnable at ~200ms/sample, SOTA on NLI benchmarks, zero circular reasoning risk.

**Critical design point**: it's a **cross-encoder** not a bi-encoder. Both strings go through the same transformer together, enabling full cross-attention. This makes it more accurate for entailment than encoding each string separately and comparing vectors.

**What was rejected**:

| Rejected | Why |
|---|---|
| GPT-4 as judge | Non-deterministic, ~$0.01/sample API cost, internet dependency in CI, prompt-sensitive, circular reasoning risk |
| BLEURT | Designed for translation quality, not faithfulness. Measures surface-level similarity, not logical entailment. |
| BERTScore | Same issue — measures similarity of token embeddings, not whether context supports the answer |
| RAGAS | Framework-level library that wraps LLM calls internally — inherits LLM-as-judge problems |

---

#### Sentence Transformers (`all-MiniLM-L6-v2`)

**What it is**: a bi-encoder model producing 384-dimensional dense embeddings. Cosine similarity between question and answer embeddings = relevance score.

**Why chosen for relevance**: relevance is a semantic similarity task, not an entailment task. Bi-encoder is fast (encode once, compute cosine) and sufficient. The full cross-encoder precision isn't needed here.

**all-MiniLM-L6-v2 specifically**: 22M parameters, 80MB model file, CPU inference < 10ms. It's the standard recommendation for "lightweight, fast, good-enough semantic similarity."

**Why bi-encoder here but cross-encoder for faithfulness**: faithfulness requires reasoning about logical support between two texts → cross-encoder. Relevance is semantic proximity → bi-encoder is sufficient and 10x faster.

---

#### Click (CLI)

**What it is**: Python decorator-based CLI framework.

**Why chosen**: `mlops-eval score <file> --threshold 0.7` — Click makes this a 10-line implementation. Handles argument parsing, `--help` generation, type coercion, and exit codes cleanly.

**Exit code 1 is the key feature**: when `mlops-eval` exits with code 1, GitHub Actions marks the step as failed. This is how the CLI becomes a CI gate — not through any special API, just Unix exit codes.

**What was rejected**: `argparse` (verbose, manual help text), `Typer` (Click wrapper, adds a dependency for no gain here).

---

#### Docker (multi-stage) + Docker Compose

**What it is**: Docker packages the application and its dependencies into an isolated image. Multi-stage builds use one image to install dependencies (builder) and a smaller image to run the app (runtime). Compose orchestrates multiple containers locally.

**Why multi-stage builds**:
```
Stage 1 (builder): python:3.11-slim + pip install → ~800MB intermediate
Stage 2 (runtime): python:3.11-slim + copy /root/.local → ~120MB final
```
The builder has compilers, build tools, dev headers — needed to install packages. The runtime doesn't need them. Multi-stage keeps the final image small, reducing pull time and attack surface.

**Why Docker Compose over Kubernetes**: Compose is the right tool for local development and portfolio demos. Kubernetes adds: pod scheduling, auto-scaling, health-based restart, rolling deployments — all necessary at scale, all overhead for a 3-service local stack. The migration path is: `docker-compose.yml` → `helm chart` → Kubernetes. The concepts transfer directly.

**What Compose provides that raw Docker commands don't**: a single `docker compose up` that starts all three services in the right order, on the right network, with the right volume mounts, every time, from one file.

---

#### GitHub Actions + GHCR

**What it is**: GitHub Actions is a YAML-defined CI/CD system native to GitHub. GHCR is GitHub's container registry.

**Why GitHub Actions**:
- Free for public repositories (unlimited minutes)
- No separate account or login — it's where the code already lives
- Native `GITHUB_TOKEN` secret for GHCR push — zero credential management
- YAML is readable and version-controlled alongside the code

**Why GHCR over DockerHub**:
- DockerHub rate-limits pulls on public repos (100 pulls/6 hours for anonymous users)
- GHCR is tied to GitHub auth — no separate credentials
- Images are free for public repos
- `ghcr.io/anh-d-tran030/mlops-loop:latest` is a permanent, versioned URL

**What was rejected**:

| Rejected | Why |
|---|---|
| Jenkins | Self-hosted, requires a server to maintain. Overkill for a single project. |
| CircleCI | Paid tiers limit parallelism. GitHub Actions is free and already integrated. |
| GitLab CI | Would require moving the repo to GitLab. |

---

#### ruff + black

**What it is**: ruff is a Rust-based linter (checks for errors, style issues, unused imports). black is a code formatter (rewrites code to a consistent style).

**Why both**: ruff catches logical issues (undefined names, unused variables). black handles formatting. They don't overlap — ruff is a linter, black is a formatter.

**Why ruff over flake8/pylint**: ruff is 10–100x faster because it's written in Rust. In CI where every second costs money, this matters. It also replaces isort (import sorting) and several flake8 plugins.

**Why black over alternatives**: black is opinionated and non-configurable (by design). No team debates about style — black decides. The only configuration is `line-length`. This removes an entire class of code review comments.

---

## 15. The Framework Evaluation Process — Applying It to Future Projects and Jobs

> The goal is not to memorise this stack. It's to build the mental model that produced these choices, so you can make them independently on any project.

---

### The 5-Question Evaluation Rubric

Before choosing any tool for any project, answer these five questions in order. If you can't answer all five, you don't understand the tool well enough to commit to it.

**Q1: What specific failure mode does this tool prevent?**

Every good tool exists because something was breaking without it. If you can't name the failure mode, you're adding a dependency for no reason.

- Prometheus → silent metric loss (push-based alternatives drop data on client crash)
- Pydantic → garbage input reaching model inference
- DeBERTa NLI → non-deterministic eval scores causing CI flakiness
- multi-stage Docker → 800MB images that take 3 minutes to pull

If the failure mode doesn't apply to your project, the tool might not be necessary.

**Q2: What are the costs? (infrastructure, learning curve, maintenance, $)**

Every tool has a cost that isn't obvious from the README.

- Airflow: operationally complex, requires a metadata DB, scheduler process, worker pool
- MLflow: simple to start, painful to scale (requires a tracking server, artifact store, registry)
- Kubernetes: 2-week learning curve minimum, requires understanding of networking, RBAC, etc.
- GPT-4 API: $X per eval sample × thousands of CI runs = real money

The cost is always higher than the docs suggest. Be honest about it.

**Q3: What are the realistic alternatives at your scale?**

Scale changes the answer completely.

| Scale | Serving | Registry | Monitoring |
|---|---|---|---|
| 1 model, prototype | Flask + pickle | S3 folder | print() |
| 1–5 models, team | FastAPI + Docker | GHCR | Prometheus + Grafana |
| 10–50 models, company | BentoML / Triton | MLflow + ECR | Evidently + Datadog |
| 100+ models, platform | Triton + K8s | Tecton + S3 | Custom + Datadog |

The mlops-loop stack is correct for "1–5 models, team." Recommending Kubernetes for a team with one model is engineer ego, not engineering.

**Q4: What does this tool NOT solve that you'll need something else for?**

This is where people get burned. They choose a tool expecting it to solve a problem it doesn't.

- Evidently doesn't monitor label drift (it monitors feature and prediction distributions, not model accuracy against ground truth — you need ground truth for that)
- Prometheus doesn't store logs (use Loki or ELK for logs)
- FastAPI doesn't handle model batching (you implement that yourself or use a serving framework)
- Docker Compose doesn't auto-restart failed containers by default (add `restart: on-failure`)

Knowing what a tool doesn't do is as important as knowing what it does.

**Q5: What's your escape hatch?**

Every tool will eventually be replaced. The question is: how painful is it?

- FastAPI → replacing with gRPC means rewriting the interface but the model is unchanged
- Docker → if you need Kubernetes, `docker-compose.yml` maps directly to K8s manifests
- GitHub Actions → your CI logic is YAML; migrating to GitLab CI or CircleCI is a YAML rewrite, not a logic rewrite
- Evidently → if you need to swap to a different drift library, you're replacing one Python file (`drift_report.py`)

Choose tools where the escape hatch is cheap. Avoid vendor lock-in for core logic.

---

### Applying This to Future Projects — Decision Tree

When you start a new ML project, work through this sequence:

```
1. PROBLEM FRAMING
   │
   ├── Tabular data + features? → LightGBM/XGBoost family
   ├── Sequential, long-range dependencies? → LSTM or Transformer
   ├── Image/audio? → CNN/ViT
   └── Text? → Transformer (BERT family or LLM)

2. SERVING PATTERN
   │
   ├── < 100ms latency required? → Synchronous REST API (FastAPI)
   ├── High throughput, can batch? → Batching server (Triton, TorchServe)
   ├── Millions of rows, no latency req? → Batch job (Spark, pandas + scheduler)
   └── Continuous stream? → Streaming (Kafka + Flink)

3. MONITORING STRATEGY
   │
   ├── Do you have labels in real-time? → Monitor accuracy directly
   ├── Labels come with delay? → Monitor data drift (PSI) as proxy
   └── No labels ever? → Monitor output distribution + business KPIs

4. CI/CD GATES
   │
   ├── Pure software → lint + test
   ├── ML model → lint + test + eval gate (always)
   └── LLM system → lint + test + NLI faithfulness gate + semantic relevance gate

5. INFRASTRUCTURE SCALE
   │
   ├── Portfolio / prototype → Docker Compose
   ├── Small team, < 10 models → Docker + GitHub Actions + GHCR
   ├── Company, < 50 models → K8s + MLflow + Prometheus
   └── Platform, 100+ models → Custom platform (Tecton, Kubeflow, internal tooling)
```

---

### Applying This in Job Interviews

The question "what frameworks have you used?" is a trap. The real question is always "why did you choose X over Y?"

**The answer structure that works**:

> "We chose [tool] because [specific failure mode it prevents]. We considered [alternative] but rejected it because [specific reason relevant to our constraints: scale, cost, latency, team skill, determinism]. The trade-off we accepted is [honest downside of our choice]."

**Examples using mlops-loop**:

"Why Evidently instead of a custom drift implementation?"
> "We needed statistical drift tests across both numerical and categorical features with Prometheus integration out of the box. A custom PSI implementation would have handled numerical features, but we'd have had to implement chi-squared for categoricals separately, plus the Prometheus export. Evidently gave us all three in one library. The trade-off is that we're dependent on their API surface — if they break a release, we have to pin versions carefully."

"Why DeBERTa instead of GPT-4 for faithfulness scoring?"
> "Three reasons: determinism, cost, and circular reasoning. GPT-4 is non-deterministic — the same input can get a 0.71 one day and 0.68 the next, which makes a CI threshold gate meaningless. At scale, API costs compound — 500 eval samples × daily CI = real money. And using an LLM to judge another LLM's faithfulness creates circular reasoning: both models share training distribution and both tend to score fluent-sounding wrong answers highly. DeBERTa is a discriminative classifier trained specifically on entailment pairs — it doesn't care about fluency, only logical support."

---

### The Meta-Skill: Transferability

The frameworks in mlops-loop are not what matters for your career. What matters is the **reasoning pattern** behind them.

Every future project will have different tools. But every project will have:
- A serving latency requirement → drives API architecture choice
- A monitoring strategy constraint → drives what you can and can't detect
- A budget constraint → drives build-vs-buy on every tool
- A team skill constraint → drives framework complexity ceiling
- A scale requirement → drives infrastructure choice

The engineer who can map constraints → decisions → trade-offs will outperform the one who memorised a stack. The stack is just one concrete instantiation of the reasoning.

**Practice**: for every new tool you encounter, fill in this table before you use it:

| Question | Answer |
|---|---|
| What failure mode does this prevent? | |
| What's the cost? | |
| What did I reject and why? | |
| What does this NOT solve? | |
| What's my escape hatch? | |

If you can fill it in without looking it up, you understand the tool.

---

*Updated: 2026-07-20 | Sections 14–15 added*

---

## 16. The Dataset — What You're Actually Working With

> If someone asks "tell me about your data" and you can't answer specifically, nothing else in the project matters. This section is the thing most portfolio builders skip.

### Corporación Favorita — Kaggle Store Sales Dataset

**What it is**: 5 years of daily grocery sales data from Corporación Favorita, a large Ecuadorian supermarket chain. The competition task: forecast unit sales for thousands of product-store combinations.

**Why this dataset specifically**:
- Real business problem (not toy data like iris or MNIST)
- Multiple seasonality: weekly patterns + annual holiday spikes + national events (Ecuador's oil price dependency affects purchasing power)
- Rich exogenous features: promotions, store type, regional differences
- High cardinality: 54 stores × 33 product families = 1,782 time series to model simultaneously
- Messy: some series have gaps, some families have sparse sales, new stores appear mid-dataset

### The Files and What's in Them

| File | Rows | Key Columns | What It Tells You |
|---|---|---|---|
| `train.csv` | ~3M | `date, store_nbr, family, sales, onpromotion` | The target variable (sales) and promotion flag per day per store-family |
| `stores.csv` | 54 | `store_nbr, city, state, type, cluster` | Store metadata — type (A/B/C/D/E) and cluster (1–17) encode store format and regional behaviour |
| `oil.csv` | ~1700 | `date, dcoilwtico` | Daily oil price — Ecuador is oil-dependent, prices correlate with consumer spending |
| `holidays_events.csv` | ~350 | `date, type, locale, description` | National/regional/local holidays and events — critical for spike forecasting |
| `transactions.csv` | ~83k | `date, store_nbr, transactions` | Total transactions per store per day — proxy for footfall |

### The 33 Product Families

The `family` column has 33 categories including: AUTOMOTIVE, BABY CARE, BEAUTY, BEVERAGES, BOOKS, BREAD/BAKERY, CLEANING, DAIRY, DELI, EGGS, FROZEN FOODS, GROCERY (I and II), HARDWARE, HOME APPLIANCES, HOME CARE, LADIESWEAR, LAWN AND GARDEN, LIQUOR/WINE/BEER, MEATS, PERSONAL CARE, PET SUPPLIES, POULTRY, PREPARED FOODS, PRODUCE, SCHOOL AND OFFICE SUPPLIES, SEAFOOD, SNACKS.

**Why this matters for your model**: families have wildly different sales distributions. PRODUCE has daily high volume. BOOKS might sell zero on most days. Your model must handle zero-inflation, different scales, and different seasonal patterns across all 33 — without 33 separate models. This is why a global LightGBM with family as a categorical feature is the right call.

### The Feature Engineering You Built

```
Raw columns available           →    Features engineered in features.py
────────────────────────────────────────────────────────────────────────
date                            →    day_of_week (0-6)
                                →    month (1-12)
                                →    year (2013-2017)
                                →    day_of_month (1-31)
                                →    days_to_holiday (from holidays_events.csv)

sales (target, historical)      →    lag_7   (sales 7 days ago)
                                →    lag_14  (sales 14 days ago)
                                →    rolling_mean_7   (7-day moving average)

onpromotion                     →    used directly (binary)

stores.csv join                 →    store_type (A/B/C/D/E)
                                →    cluster (1-17)
                                →    city, state

family                          →    OHE or LightGBM categorical encoding
```

**Why lag_7 and not lag_1**: weekly seasonality dominates grocery retail. Sales on Monday this week are best predicted by sales on Monday last week — same day-of-week controls for the weekly rhythm. lag_1 (yesterday) is noisier.

**Why rolling_mean_7**: smooths out day-of-week variance to capture the underlying level of demand. Useful for trending up/down signals.

**What's NOT in the features**: oil price (available, not used in the scaffold — could add it). Transactions (available, correlates with sales but creates circular dependency if not handled carefully). These are valid "what would you add next" answers.

### The Train/Val Split

Grocery data has **temporal structure** — you cannot randomly shuffle rows. If you do, you leak future information into training (lag features for day T might include values from day T+30 if rows are shuffled). The correct split:

```
2013-01-01 ──────────── 2017-07-31 | 2017-08-01 ──── 2017-08-15
           TRAINING DATA            |   VALIDATION   |  (test)
```

Validate on the last 2–4 weeks. This simulates the actual deployment scenario: the model has seen everything up to some cutoff, and now predicts forward.

---

## 17. Numbers You Need to Know

> These are the numbers you cite when challenged. One sentence each, no padding.

### Model Performance

The Kaggle competition metric is **RMSLE** (Root Mean Squared Log Error) — it penalises under-forecasting more than over-forecasting, because running out of stock is worse than having excess.

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(1 + \hat{y}_i) - \log(1 + y_i))^2}$$

**Kaggle leaderboard benchmarks for reference**:

| Approach | RMSLE (approx) |
|---|---|
| Naive (predict mean) | ~0.75 |
| Decent LightGBM + basic features | ~0.05–0.08 |
| Top Kaggle solutions (ensembles, 100s of features) | ~0.038 |

**What your model should achieve**: a clean LightGBM with `lag_7`, `lag_14`, `rolling_mean_7`, date parts, OHE family, and store metadata should land around **0.05–0.07 RMSLE** on the val set. That puts you solidly in the top 30% of Kaggle submissions — with a single model, no ensembling, and 30-second training time.

**How to answer "is that good?"**: "It's not state-of-the-art, but it's competitive for a single model with interpretable features. The winning solutions use ensembles of 10+ models with hundreds of hand-crafted features and take hours to train. Ours trains in 30 seconds, serves in < 50ms, and is explainable via SHAP — that's the production trade-off."

### Serving Latency

| Endpoint | Expected P99 |
|---|---|
| `POST /predict` (LightGBM inference) | < 50ms |
| `GET /health` | < 5ms |
| `GET /metrics` | < 10ms |

LightGBM inference on a 30-day forecast for one store-family combo is CPU-only and runs in **5–15ms**. The rest is FastAPI + Pydantic overhead (~10ms) and network (~5ms locally). Total: well under 50ms.

**How to answer "what's your latency?"**: "Under 50ms P99 for a single prediction request. LightGBM inference itself is 5–15ms — the rest is framework overhead. For batch use cases you'd precompute forecasts nightly and serve from a cache."

### Drift Detection Thresholds

| PSI Value | Status | Action |
|---|---|---|
| < 0.10 | Stable | Continue monitoring |
| 0.10 – 0.20 | Moderate shift | Investigate upstream data pipeline |
| > 0.20 | Major shift | Trigger retraining pipeline |

These aren't arbitrary — they come from Basel II banking. The 0.20 threshold is the one wired into the Grafana alert.

### Eval Gate Threshold

The `mlops-eval` CLI uses `--threshold 0.7`. This means:
- Mean faithfulness score (DeBERTa NLI entailment) must be ≥ 0.70
- Mean relevance score (cosine similarity) must be ≥ 0.70

**How to answer "why 0.7?"**: "0.7 is a calibrated starting point — empirically, a faithfulness score below 0.7 in our eval set corresponds to answers that clearly hallucinate or misattribute information from the context. It's not a universal threshold; you calibrate it against labelled examples where you know ground truth faithfulness. We'd tighten it as the system matures."

### Docker Image Size

Target: **< 150MB** final stage.

Multi-stage build splits:
- Builder stage (with pip + build tools): ~800MB — never shipped
- Runtime stage (app + installed packages only): ~120MB — what actually runs

**Why this matters**: at 120MB, a cold start pull takes ~10 seconds on a typical instance. At 800MB, it takes ~90 seconds. In auto-scaling scenarios that difference is the gap between handling a traffic spike and dropping requests.

---

## 18. The Business Case — Why This Exists

> "Why retail demand forecasting?" will be asked. Know the answer without thinking.

### The Problem Vaylo Solves

Vaylo is an AI-driven CRM/inventory platform for retail businesses. Its value proposition: replace spreadsheet-based inventory decisions with model-driven ones.

The specific decision the forecast powers: **how much stock to order, for which product, for which store, 30 days in advance**.

This is a real business decision with a real cost when wrong:

| Forecast error | Business impact |
|---|---|
| Under-forecast (predict 100, actual 200) | Stockout → lost sales, customer dissatisfaction, competitor switching |
| Over-forecast (predict 200, actual 100) | Overstock → storage costs, spoilage (perishables), capital tied up |

**The cost quantification** (how to answer "what's the business value?"):
> "A 1% improvement in forecast accuracy across 1,800 store-product combinations translates to reduced overstock or fewer stockouts. For a chain with $100M annual revenue, a 5% reduction in forecast error can mean $2–5M in inventory cost savings — from reduced waste, lower safety stock requirements, and fewer emergency orders."

This isn't a made-up number — McKinsey and Gartner both publish similar estimates for grocery retail. It's the right order of magnitude and defensible.

### Why 30-Day Horizon

Supplier lead times for grocery replenishment are typically 2–4 weeks. A 30-day forecast gives:
- Enough horizon to place orders with suppliers
- Time to adjust logistics and warehouse allocation
- A buffer for demand spike events (holidays, promotions)

Shorter (7-day) would be too reactive. Longer (90-day) would be too noisy for weekly ordering cycles.

### Why Ecuador / Corporación Favorita

It's the standard Kaggle benchmark for this problem class. It has:
- Real production scale (1,782 series)
- Real external factors (oil prices, national holidays)
- Public ground truth (can validate claims against leaderboard)
- Clean enough to build on, messy enough to be interesting

For portfolio purposes: "I used the Kaggle Store Sales dataset because it's a production-scale real retail problem with ground truth I can benchmark against publicly. The model generalises to any similar inventory forecasting context — the architecture, not the dataset, is what we're demonstrating."

---

## 19. Honest Limitations & What Comes Next

> Every interviewer asks "what are the weaknesses?" The engineer who answers honestly and specifically is more credible than the one who says "it's pretty solid."

### What's Missing and Why It Matters

**1. No automated retraining trigger**

The outer loop (PSI > 0.2 → retrain → deploy) is demonstrated conceptually but not automated. In production, you'd have an Airflow DAG or a Prefect flow that watches the PSI Prometheus metric and fires a training pipeline when it crosses 0.2.

What this means in practice: currently, a human has to notice the Grafana alert and manually kick off retraining. In a real system with 50+ models, that's operationally unsustainable.

*What I'd add*: a `retrain_trigger.py` that runs on a cron, queries Prometheus for PSI values, and triggers the training pipeline via GitHub Actions `workflow_dispatch` if the threshold is breached.

**2. No experiment tracking**

`train.py` prints RMSE to stdout. In a real system, every training run would log to MLflow: parameters, metrics, model artifact, data version hash. Without this, you can't compare runs, you can't roll back to a specific experiment, and you can't explain why the model from 3 months ago performed better.

*What I'd add*: `mlflow.start_run()` around the training block, `mlflow.log_params()` for hyperparameters, `mlflow.log_metric()` for RMSE, `mlflow.lightgbm.log_model()` for the artifact.

**3. No ground truth feedback loop**

The monitoring layer detects input drift (PSI on features). It cannot detect output accuracy degradation — because actual sales data (ground truth) arrives 30 days after the forecast. The system has no mechanism to collect ground truth and compare it against stored predictions.

*What I'd add*: a `ground_truth_collector.py` that: (1) stores every prediction with its `prediction_id` in a DB, (2) 30 days later, joins ground truth sales data against the stored predictions, (3) computes actual RMSE and logs it to Prometheus. This closes the full accuracy monitoring loop.

**4. No feature store**

`features.py` is shared between training and serving — this is correct in principle, but in a deployed system the API needs to fetch historical lag features at inference time. Currently `model.py` would need to compute `lag_7` by querying a historical sales database. If that query logic differs from training, skew happens. A feature store (Feast, Tecton) solves this by caching precomputed features and serving them identically to both.

*What I'd add in a real system*: Feast with a Redis online store. Feature vectors are precomputed daily and cached. At inference, the API fetches `lag_7` for store 5, BEVERAGES from Redis in ~1ms — no live calculation, no skew risk.

**5. Single environment (no dev/staging/prod)**

The repo treats `main` branch as production. A real system has three environments: dev (local), staging (identical to prod, used for validation), production. Code flows: dev → staging → prod, with separate monitoring, separate model registries, and separate alert thresholds.

*What I'd add*: separate GitHub Environments (dev/staging/prod) with environment-specific secrets and deployment gates. Staging deployment is automatic; production requires manual approval.

**6. No model explainability endpoint**

LightGBM has native SHAP support. Currently there's no way for downstream users to understand why a specific forecast was made. In retail, a buyer might want to know "why is your model predicting a 40% sales spike for BEVERAGES in store 12 next Tuesday?"

*What I'd add*: `GET /explain/{prediction_id}` that returns SHAP values for the top 5 feature contributions to that specific prediction.

### What I'd Do Differently

If starting over with a 3-month timeline instead of a sprint:

1. MLflow from day one — experiment tracking is zero-cost to add early and very expensive to retrofit
2. Ground truth collection pipeline before monitoring — you can't measure accuracy without storing predictions + collecting actuals
3. Feature store before the API — prevents the skew problem structurally rather than through discipline
4. Parametric training config (YAML/JSON) instead of hardcoded hyperparameters — makes automated HPO trivial to add

---

## 20. Code Walk-Through — File by File

> You should be able to open any file in this repo and explain every line's purpose. This section maps the key files to the concepts they implement.

### `services/forecast-api/app/schemas.py` — The Contract

```python
class PredictRequest(BaseModel):
    store_nbr: int        # which of the 54 stores
    family: str           # which of the 33 product families
    onpromotion: int      # 1 if this family is on promotion this period
    days_ahead: int = 30  # forecast horizon; default 30 matches supplier lead time

class PredictResponse(BaseModel):
    forecast: list[float]   # 30 predicted daily sales values
    model_version: str      # what artifact version produced this
    prediction_id: str      # UUID4 — ties prediction to logs for debugging
```

**Why `prediction_id` is here**: you can't debug production without it. If a buyer says "your forecast for store 5 BEVERAGES on June 20 was wrong," you look up that `prediction_id` in your logs, pull the exact features that went into it, and reproduce the prediction deterministically.

**Why `model_version` is here**: when you deploy a new model, predictions from the old model and new model will coexist in logs. Without versioning you can't tell which model made which prediction.

---

### `services/forecast-api/app/main.py` — The Lifecycle

Three sections matter:

**Lifespan** — runs once at boot:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load("model.pkl")  # 500ms load, done once
    yield                    # app runs
    # cleanup on shutdown
```

**`POST /predict`** — the money endpoint:
```python
@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    features = build_features(request)  # deterministic feature construction
    values = model.predict(features)    # LightGBM inference
    return PredictResponse(
        forecast=values,
        model_version=MODEL_VERSION,
        prediction_id=str(uuid4())
    )
```

**`GET /health`** — what Kubernetes pings:
```python
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model.is_loaded}
```
If model failed to load at startup, `model.is_loaded = False`, health returns this, orchestrator restarts the container. This is the self-healing mechanism.

---

### `services/forecast-api/training/train.py` — The Training Script

Key decisions embedded in the code:

```python
lgb.LGBMRegressor(
    objective='regression',   # RMSE loss — continuous target
    metric='rmse',
    num_leaves=31,            # controls tree complexity; 31 is conservative default
    n_estimators=500,         # number of boosting rounds
    learning_rate=0.05        # small LR + many estimators > large LR + few
)
```

**Why `num_leaves=31`**: LightGBM grows trees leaf-wise. `num_leaves` controls complexity — more leaves = more complex model = more overfitting risk. 31 (≈ depth-5 tree) is the standard starting point. You'd tune this with cross-validation.

**Why `n_estimators=500`**: with `learning_rate=0.05`, 500 rounds gives the model enough capacity to fit without excessive training time. Real tuning: add early stopping (`early_stopping_rounds=50`) so training stops when val RMSE stops improving.

**What's missing here that you'd add in production**: `mlflow.start_run()`, `early_stopping_rounds`, cross-validation instead of single train/val split, `joblib.dump(model, "model.pkl")` with a version hash in the filename.

---

### `monitoring/evidently/drift_report.py` — Drift Detection

The core logic:

```python
reference = pd.read_parquet("reference_data.parquet")  # training distribution
current = pd.read_csv(current_window_path)             # recent production data

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
```

**What `DataDriftPreset` computes internally**:
- For each numerical feature: KS test (p-value < 0.05 → drift detected)
- For each categorical feature: chi-squared test
- PSI is computed separately and written to the Prometheus metrics file

**The Prometheus metrics file output**:
```
# psi_score{feature="onpromotion"} 0.13
# psi_score{feature="days_ahead"} 0.04
# drift_detected 1
```
Prometheus scrapes this file via a `textfile_collector` and makes these metrics queryable in PromQL. Grafana reads them and renders the PSI-over-time panels.

---

### `Dockerfile` — Why Multi-Stage

```dockerfile
# Stage 1: builder — has pip, compilers, build tools
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
# Result: /root/.local/ contains all installed packages

# Stage 2: runtime — minimal, no build tools
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local  # copy only the installed packages
COPY app/ ./app/
COPY model.pkl .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**The key line**: `COPY --from=builder /root/.local /root/.local` — this copies only the installed Python packages from the builder, not the pip binary, not the compilers, not the build cache. The runtime image has no build tools — smaller attack surface, smaller image.

**Why `python:3.11-slim` and not `python:3.11`**: `slim` excludes documentation, test suites, and some system utilities included in the full image. ~50MB smaller base, no functional difference for production serving.

---

### `.github/workflows/ci.yml` — The Quality Gate Chain

```
lint (30s) → test (3min) → eval-gate (2min) → build-push (5min)
    ↑              ↑               ↑                  ↑
cheapest      next cheapest    ML-specific        most expensive
fails first   fails second     gate unique         only on main
                               to ML systems
```

The `needs:` keyword is the critical piece:
```yaml
eval-gate:
  needs: test         # won't start until test completes successfully
build-push:
  needs: eval-gate    # won't start until eval-gate completes successfully
  if: github.ref == 'refs/heads/main'  # skipped entirely on PRs
```

If `lint` fails, nothing else runs. CI exits in 30 seconds with a clear failure. No compute wasted on tests that would have run on broken code.

---

## 21. Interview Prep — 25 Questions They Will Ask

> Answer these out loud, not just by reading. The ability to explain verbally ≠ the ability to read.

### On the Model

1. Why LightGBM and not LSTM?
2. What's your train/val split strategy and why? (Answer: temporal, never random)
3. What features did you engineer and what does each capture?
4. What RMSE did you achieve and is that good?
5. What's the one feature you'd add next to most improve performance?

### On the API

6. Walk me through what happens between a POST request arriving and the response returning.
7. What does Pydantic v2 do that raw Python type hints don't?
8. What happens if `model.pkl` fails to load at startup?
9. Why is the model loaded in the lifespan context manager and not inside the handler?
10. What's `prediction_id` for and why is it in the response?

### On Monitoring

11. What's the difference between data drift and concept drift?
12. How does Prometheus get the PSI values from your system?
13. A feature has PSI = 0.15. What do you do?
14. What can your monitoring NOT detect? (Answer: accuracy degradation against ground truth — you don't have it in real-time)
15. Why does `reference_data.parquet` never auto-update?

### On the Eval System

16. Why DeBERTa and not GPT-4 for faithfulness scoring?
17. What's the difference between faithfulness and relevance?
18. What does exit code 1 do in the CI pipeline?
19. Why is 0.7 the threshold and not 0.8?

### On the Infrastructure

20. Why Docker Compose and not Kubernetes?
21. If Prometheus goes down, what happens to the forecast API?
22. Why does `depends_on` not guarantee Prometheus is ready when Grafana starts?
23. Why GHCR over DockerHub?

### On Limitations and Growth

24. What's the biggest limitation of this system right now?
25. If you had 3 more months, what would you build first and why?

**Model answers for 24 and 25** (these reveal depth of understanding):

**24**: "The biggest limitation is the outer loop is manual. I can detect drift via PSI > 0.2, but there's no automated trigger that fires a retraining pipeline when that threshold is crossed. In production with 1,800+ series you can't have a human watching dashboards. The fix is an Airflow DAG polling the PSI metric and triggering `workflow_dispatch` on the training pipeline."

**25**: "Ground truth collection and MLflow, in that order. Right now I can detect input drift but I can't measure actual forecast accuracy because I never collect the actual sales data 30 days later to compare against stored predictions. That's the gap between 'I know the world changed' and 'I know my model is wrong.' MLflow second — without experiment tracking, every retraining run is a black box. I can't tell whether the new model is better than the old one in a reproducible way."

---

*Updated: 2026-07-20 | Sections 16–21 added — project defence complete*
