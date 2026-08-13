# MLOps Loop — Deep Understanding Guide

> Goal: every decision in this repo is answerable in a 45-minute interview without notes.
> This doc follows the Feynman principle: intuition first, formalism second, stress-test third.

---

## Table of Contents

1. [LightGBM vs ARIMA vs LSTM](#1-lightgbm-vs-arima-vs-lstm)
2. [FastAPI Model Serving — Request Lifecycle](#2-fastapi-model-serving--request-lifecycle)
3. [PSI — What It Measures & Why It's Used](#3-psi--what-it-measures--why-its-used)
4. [Evidently + Grafana — Drift Reporting Internals](#4-evidently--grafana--drift-reporting-internals)
5. [DeBERTa NLI vs LLM-as-Judge](#5-deberta-nli-vs-llm-as-judge)
6. [GitHub Actions — Job Order & Why](#6-github-actions--job-order--why)
7. [Docker Compose — Networking & Failure Behaviour](#7-docker-compose--networking--failure-behaviour)

---

## 1. LightGBM vs ARIMA vs LSTM

### THE INTUITIVE EXPLANATION

**Big picture**: You have 1,800+ store-product combinations, each with daily sales history plus features like promotions and store type. You need to forecast 30 days ahead. Which model do you pick, and why does the answer matter?

Think of it this way: **ARIMA is a pattern-memoriser for a single series**, **LSTM is a sequence-learner that struggles with extra context**, and **LightGBM is a feature-machine that handles everything tabular natively**. Retail demand is fundamentally a tabular problem dressed up in a time-series costume.

---

### WHY ARIMA FAILS HERE

ARIMA (AutoRegressive Integrated Moving Average) assumes:
1. **One series at a time** — you'd need 1,800 separate fitted models, one per store-family combo. That's 1,800 × training time + 1,800 × tuning cycles.
2. **Stationarity** — the series must have constant mean and variance over time. Real retail data has trend + multiple seasonality (weekly rhythm + annual holiday spikes). You must difference the series to make it stationary, but differencing destroys information about the level.
3. **No exogenous structure** — promotions, store type, cluster — ARIMA can't natively encode these as structured categorical inputs. ARIMAX exists but it's clunky, requires manual feature selection, and doesn't handle interactions.

**Where ARIMA wins**: Single well-behaved series, high stationarity, no external regressors, explainability is mandatory (e.g., economics papers, regulatory forecasting). That's not retail.

**The exact failure condition**: When `n_series > 50` and `n_features > 5`, ARIMA becomes operationally untenable and statistically inferior.

---

### WHY LSTM FAILS HERE

LSTM (Long Short-Term Memory) is a recurrent neural network designed for sequences. Its selling point: it can learn long-range dependencies (what happened 90 days ago might matter now).

The problems:

| Issue | Why It Hurts Here |
|---|---|
| Needs long sequences | Kaggle Store Sales has ~1,700 days. That sounds like a lot until you split train/val — you're left with sparse sequences per store-family. |
| Tabular features are second-class | LSTM processes sequences naturally. Injecting promotion flags, store type, cluster as side inputs requires custom architectures (e.g., concatenation at each timestep). LightGBM treats them as first-class features natively. |
| Training time | A reasonable LSTM for this dataset takes 30–90 min on GPU. LightGBM: 30 seconds on CPU. |
| Tuning surface | Hidden size, layers, dropout, sequence length, learning rate schedule — each is a hyperparameter. LightGBM has `num_leaves` and `n_estimators`. |
| Black box locally | Gradient-based attribution (SHAP) exists for LSTM but is expensive. LightGBM SHAP is O(n features) and comes built-in. |

**Where LSTM wins**: Long univariate sequences (EEG, audio), when temporal dependencies at 100+ lags are critical, when you have millions of training examples per series.

**The exact failure condition**: When your data is wide (many features) but not tall (limited history per series), and when training budget is under 5 minutes.

---

### WHY LIGHTGBM WINS

LightGBM is gradient-boosted decision trees with histogram-based splitting.

**Key insight**: Once you engineer lag features (`lag_7`, `lag_14`, `rolling_mean_7`), you've encoded the temporal structure as tabular columns. The model doesn't need to "see" the sequence — it sees: "7 days ago, sales were X. 14 days ago, Y. This week's rolling mean is Z." That's enough.

What LightGBM gets for free:
- **Categorical features natively** — store, family, cluster go in without OHE if you tell it the column type
- **One global model** — train once on all 1,800 series; the model generalises across store-family combos
- **Missing values** — handles them internally, no imputation required
- **SHAP out of the box** — explainability is trivial
- **30-second training** — iterate fast

**The exact conditions where LightGBM wins**:
1. Features + lag features together explain variance better than raw temporal structure
2. `n_series` is large (global model advantage)
3. Training time budget is under 5 minutes
4. Interpretability is required post-deployment

---

### MISTAKE AUTOPSY

**MISTAKE: "LSTM is better because it understands time"**

```
SYMPTOM:  Claiming LSTM handles time-series "natively" so it must be better
CAUSE:    Confusing "can process sequences" with "best for temporal prediction"
CURE:     Lag features ARE sequence information — just in tabular form. The question 
          is whether the model needs sequential processing or whether features suffice.
DETECTOR: Any claim of LSTM superiority without specifying what temporal pattern 
          lag features can't capture.
```

**MISTAKE: "Use one ARIMA per series"**

```
SYMPTOM:  Proposing ARIMA for a 1,800-series problem
CAUSE:    Learning ARIMA in a univariate forecasting context and not questioning the 
          scaling assumption
CURE:     Ask: "How many models do I need to fit, tune, and maintain?" If > 10, 
          reconsider.
DETECTOR: Any dataset with store × product combos.
```

---

### SELF-TEST

1. Name two specific features in this dataset that ARIMA cannot encode without becoming ARIMAX.
2. If you only had 30 days of history per series, which model fails hardest and why?
3. What information does `lag_7` encode, and why does it make LightGBM competitive with LSTM on temporal tasks?

---

## 2. FastAPI Model Serving — Request Lifecycle

### THE INTUITIVE EXPLANATION

**Big picture**: A POST request hits your server. Before your model ever runs, 5 things happen. Understanding each step tells you where bugs hide and where latency comes from.

Think of FastAPI as a **factory assembly line**: raw JSON enters one end, gets validated, shaped, processed, and exits as clean JSON. Each station on the line can reject the part if it's defective — no garbage ever reaches the model.

---

### THE REQUEST LIFECYCLE (step by step)

```
Client
  │
  ▼
[1] uvicorn (ASGI server)
    Receives raw TCP bytes, parses HTTP, hands off to Starlette
  │
  ▼
[2] Starlette Router
    Matches path /predict → finds your POST handler
    Checks HTTP method (405 if wrong)
  │
  ▼
[3] Pydantic v2 validation
    Parses JSON body → constructs PredictRequest
    Type coercion: "30" → 30 (int), fails on "abc"
    Constraint checking (if you add Field validators)
    Returns 422 Unprocessable Entity on validation failure
  │
  ▼
[4] Your async handler runs
    Calls model.predict(request.store_nbr, request.family, ...)
    Awaits anything that's awaitable (DB calls, external HTTP)
    Pure CPU work (model inference) is synchronous — that's fine
  │
  ▼
[5] Pydantic serializes response
    PredictResponse → JSON
    Returns HTTP 200 with Content-Type: application/json
  │
  ▼
Client receives response
```

---

### LIFESPAN CONTEXT MANAGER

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs ONCE when server boots
    model.load("model.pkl")  # expensive — do it here, not on every request
    yield
    # Shutdown: runs ONCE when server stops
    # cleanup, flush buffers, close connections

app = FastAPI(lifespan=lifespan)
```

**Why this matters**: If you load the model inside the request handler, it reloads on every single request. 1,000 requests/sec × 500ms model load = 500 seconds of loading per second. The lifespan pattern loads once, caches in memory.

---

### PYDANTIC V2 VALIDATION INTERNALS

```python
class PredictRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    days_ahead: int = 30  # default — optional field
```

What Pydantic actually does:
1. **Coerce types**: if `days_ahead` comes in as `"30"` (string), Pydantic converts it to `30` (int). This is called "lax" mode.
2. **Validate constraints**: if you add `Field(ge=1, le=365)`, any value outside that range → 422.
3. **Reject unknown fields**: by default Pydantic v2 ignores extra fields. Add `model_config = ConfigDict(extra='forbid')` to reject them.
4. **Generate OpenAPI schema automatically** — that's why `/docs` works out of the box.

**The 422 response tells the client exactly which field failed and why** — this is a contract, not just error handling.

---

### ASYNC HANDLERS

```python
@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    result = model.predict(...)  # CPU-bound, synchronous
    return PredictResponse(...)
```

`async def` means this handler runs on the event loop. The critical rule:

- **I/O bound work** (HTTP calls, DB queries): `await` them — lets other requests run while waiting
- **CPU bound work** (model inference): runs synchronously inline — acceptable for < 200ms inference
- **Long CPU work** (> 1s): use `asyncio.run_in_executor()` to push to a thread pool, otherwise you block the entire event loop and all requests queue behind yours

LightGBM inference is ~5-10ms. No need for executor. DeBERTa on CPU is ~200ms. Still fine inline.

---

### HEALTH & METRICS ENDPOINTS

```
GET /health  → {"status": "ok", "model_loaded": true}
GET /metrics → Prometheus text format
```

`/health` is what Kubernetes/Docker health checks ping. If it returns non-200, the container gets restarted. It must be fast (< 50ms) and must not do real work.

`/metrics` uses `prometheus_client` to expose counters and histograms. Prometheus scrapes this endpoint every 15 seconds (as configured in `prometheus.yml`). The endpoint is read-only — it never mutates state.

---

### MISTAKE AUTOPSY

**MISTAKE: Loading model inside the handler**

```
SYMPTOM:  High latency on first request, or consistent 500ms overhead
CAUSE:    joblib.load() inside predict() — runs on every call
CURE:     Move to lifespan startup
DETECTOR: Request P99 latency >> model inference time
```

**MISTAKE: Using `def` instead of `async def` for I/O-heavy handlers**

```
SYMPTOM:  All requests suddenly queue when one hits a slow DB query
CAUSE:    Sync function blocks the event loop — no concurrency
CURE:     `async def` + `await` for any I/O
DETECTOR: FastAPI logs show requests stacking up during external calls
```

---

### SELF-TEST

1. A request comes in with `{"store_nbr": "5", "family": "BEVERAGES", "onpromotion": 0}`. What does Pydantic do with `store_nbr`?
2. Where would you add a 500ms sleep to simulate slow inference without blocking other requests?
3. What's the difference between `/health` returning 200 vs 503 from the container orchestrator's perspective?

---

## 3. PSI — What It Measures & Why It's Used

### THE INTUITIVE EXPLANATION

**Big picture**: Your model was trained on last year's data. Now it's serving requests based on this year's data. If the distribution of input features has shifted dramatically, your model's predictions are based on a world that no longer exists. PSI quantifies how much the world has changed.

**Analogy**: PSI is like comparing two population histograms — the reference (training data) vs. the current (production data). If the bars line up, the distribution is stable. If they've diverged, something has shifted.

---

### THE FORMULA

$$\text{PSI} = \sum_{i=1}^{n} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$

Where:
- $A_i$ = fraction of **actual** (current) data in bin $i$
- $E_i$ = fraction of **expected** (reference/training) data in bin $i$
- $n$ = number of bins (typically 10 for continuous, one per category for categorical)

**Step by step**:
1. Take a feature (e.g., `days_ahead`).
2. Bin the reference data into 10 equal-frequency bins.
3. Apply the same bin boundaries to the current data.
4. Compute what fraction of each dataset falls into each bin.
5. Apply the formula above, sum across bins.

**Worked example**:

| Bin | Reference % | Current % | (A-E) | ln(A/E) | Contribution |
|-----|-------------|-----------|-------|---------|--------------|
| 1   | 10%         | 12%       | +0.02 | +0.182  | +0.00364     |
| 2   | 10%         | 8%        | -0.02 | -0.223  | +0.00446     |
| ... | ...         | ...       | ...   | ...     | ...          |

PSI = sum of all contributions. Notice: both positive and negative deviations add to PSI (the product of two negatives or two positives is always positive).

---

### THE THRESHOLDS AND WHERE THEY CAME FROM

| PSI Value | Interpretation | Action |
|---|---|---|
| < 0.1 | No significant shift | Monitor normally |
| 0.1 – 0.2 | Moderate shift | Investigate; consider retraining |
| > 0.2 | Major shift | Model likely degraded; retrain or rollback |

These thresholds come from **Basel II banking regulation** — financial institutions used PSI to monitor scorecard stability across borrower populations. They were empirically validated across thousands of models. The fact they're industry-standard means any ML team you interview with already knows them — using KL divergence with ad-hoc thresholds would require you to justify every number.

---

### WHY NOT KL DIVERGENCE?

KL divergence formula: $D_{KL}(P \| Q) = \sum_i P_i \ln\left(\frac{P_i}{Q_i}\right)$

**Problem 1: Asymmetry**
$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

"How much has current drifted from reference" gives a different number than "how much has reference drifted from current." In production monitoring, there's no principled reason to pick one direction. PSI is symmetric: PSI(P, Q) = PSI(Q, P). This is because PSI = $D_{KL}(A \| E) + D_{KL}(E \| A)$ — it's the sum of both directions.

**Problem 2: Infinite values**
If any bin has 0 probability in Q but non-zero in P: $\ln(P_i / 0) = \infty$. KL divergence explodes to infinity. You must add smoothing (e.g., add 0.001 to each bin), which is an arbitrary choice that changes your metric. PSI has the same issue but the industry-standard fix is to clip bins to a minimum of 0.0001 — a documented, reproducible choice.

**Problem 3: No standard thresholds**
KL divergence values are dataset-specific. A KL of 0.3 might mean nothing in one context and catastrophic drift in another. You'd need to calibrate thresholds per model, per feature. PSI's 0.1/0.2 apply universally because they were calibrated across massive population studies.

---

### MISTAKE AUTOPSY

**MISTAKE: Comparing raw counts instead of percentages**

```
SYMPTOM:  PSI explodes when current dataset is larger than reference
CAUSE:    Using counts A_i and E_i instead of fractions
CURE:     Always normalise to percentages before computing PSI
DETECTOR: PSI increases monotonically with dataset size
```

**MISTAKE: Using too few bins**

```
SYMPTOM:  PSI misses subtle shifts because detail is lost
CAUSE:    Using 5 bins instead of 10
CURE:     10 equal-frequency bins for continuous; native bins for categorical
DETECTOR: Visualising the histograms shows obvious shift that PSI didn't catch
```

---

### SELF-TEST

1. Feature X has PSI = 0.15. What action do you take, and what are two possible root causes of this shift?
2. Why does PSI = $D_{KL}(A \| E) + D_{KL}(E \| A)$? Walk through the algebra from the PSI formula.
3. If a new product family is added to production data that wasn't in training data, what happens to PSI for the `family` feature?

---

## 4. Evidently + Grafana — Drift Reporting Internals

### WHAT EVIDENTLY ACTUALLY COMPUTES

Evidently is not magic — it's a statistical testing library wrapped in a nice report API. When you call `DataDriftPreset`, here's what runs under the hood:

**For numerical features** (e.g., `days_ahead`, `lag_7`):
- Default test: **Kolmogorov-Smirnov test** — measures the maximum difference between two empirical CDFs
- Alternative: **Wasserstein distance** (earth mover's distance) — how much "mass" you'd need to move to transform one distribution into the other

**For categorical features** (e.g., `family`, `store_nbr`):
- Default test: **Chi-squared test** — compares observed vs expected frequencies per category
- High cardinality override: Jensen-Shannon divergence

**PSI** is computed separately in `drift_report.py` — Evidently provides it as an optional metric but defaults to the statistical tests above.

**What Evidently outputs**:
```json
{
  "feature": "onpromotion",
  "drift_detected": true,
  "drift_score": 0.23,  // PSI or test statistic depending on config
  "p_value": 0.003,     // if using statistical test
  "reference_distribution": {...},
  "current_distribution": {...}
}
```

---

### THE DRIFT REPORT PIPELINE

```
reference_data.parquet  ──┐
                           ├──► Evidently DataDriftPreset ──► HTML report
current_window.csv      ──┘                                ──► JSON metrics
                                                           ──► Prometheus metrics file
```

`reference_data.parquet` is a **snapshot of the training feature distribution** — it's the "what normal looks like" baseline. It never changes unless you retrain. This is critical: if you update the reference on every run, you're comparing current vs. yesterday, not current vs. training-time — you'll miss slow drift.

---

### WHAT THE GRAFANA PANELS REPRESENT

**Panel 1: PSI per feature over time** (time-series line chart)
- X-axis: time (each drift report run)
- Y-axis: PSI value
- Threshold lines at 0.1 and 0.2
- What to look for: gradual creep toward 0.2 = slow concept drift; sudden spike = data pipeline issue upstream

**Panel 2: Drift detected (boolean)** (stat panel, green/red)
- Simple boolean: any feature with PSI > 0.2
- This is the "pager alert" panel — if it turns red, something needs human attention

**Panel 3: Request volume & latency** (from `/metrics` endpoint)
- Prometheus counter: total prediction requests
- Histogram: request latency buckets (P50, P95, P99)
- Helps correlate drift with traffic patterns — sometimes drift is seasonal, not model failure

**Panel 4: Model output distribution** (histogram)
- Distribution of predicted sales values over time
- If output distribution shifts without input distribution shifting → potential label drift or model behaviour change

---

### THE DATA FLOW

```
forecast-api /metrics ──► Prometheus (scrape every 15s) ──► Grafana (query on dashboard load)
drift_report.py       ──► Prometheus textfile ──────────────────────────────────────────────┘
```

Prometheus stores metrics as time-series. Grafana queries Prometheus using **PromQL** (Prometheus Query Language). A typical panel query:

```promql
psi_score{feature="onpromotion"}
```

This returns the PSI time-series for the `onpromotion` feature. Grafana renders it as a line chart.

---

### SELF-TEST

1. `reference_data.parquet` is updated every week automatically. What problem does this cause?
2. KS test p-value is 0.003 for `lag_7`. Evidently marks it as "drift detected." PSI is 0.08. How do you reconcile these two signals?
3. Prometheus is scraping every 15 seconds but Grafana panels only show data every 5 minutes. Where is the resolution mismatch happening?

---

## 5. DeBERTa NLI vs LLM-as-Judge

### THE INTUITIVE EXPLANATION

**Big picture**: You've built a RAG system. Given a question and a context passage, it generates an answer. How do you verify the answer is actually supported by the context (faithfulness) and actually addresses the question (relevance)?

**Option A**: Ask GPT-4 "is this answer faithful to this context? Rate 1-10."
**Option B**: Run DeBERTa NLI — a model trained to classify whether one sentence entails, contradicts, or is neutral to another.

For production eval, Option B wins on every axis that matters in CI.

---

### WHAT NLI IS

Natural Language Inference is the task: given a **premise** and a **hypothesis**, classify their relationship:
- **Entailment**: premise logically implies hypothesis
- **Contradiction**: premise logically denies hypothesis  
- **Neutral**: premise doesn't establish hypothesis either way

For faithfulness scoring:
- Premise = context (the retrieved document chunk)
- Hypothesis = answer (what the LLM generated)
- Score = entailment probability (0 to 1)

If the answer says "The store opened in 1995" but the context says "The store opened in 1987" → contradiction score high → faithfulness low.

---

### WHY DEBERTA-LARGE FOR NLI

DeBERTa (Decoding-enhanced BERT with Disentangled Attention) achieves state-of-the-art on MNLI (Multi-Genre NLI). Specifically, `cross-encoder/nli-deberta-v3-large` is:
- Trained directly on NLI pairs — it's not a general language model, it's a specialist
- A **cross-encoder**: both premise and hypothesis are fed together through the same transformer, allowing full attention between them. This is more accurate than bi-encoders for NLI.
- ~400M parameters — substantial but CPU-runnable at ~200ms/sample

---

### THE COMPARISON TABLE

| Dimension | LLM-as-Judge (GPT-4) | DeBERTa NLI |
|---|---|---|
| Cost | ~$0.01 per eval sample | $0 (local inference) |
| Latency | 1–3 seconds (API round-trip) | ~200ms (CPU) |
| Determinism | Non-deterministic (temperature, API version changes) | Deterministic — same input = same output, always |
| Reproducibility | CI runs can disagree across days | CI runs are identical |
| Circular reasoning risk | High — one LLM judges another's fluency, not factuality | None — NLI is a discriminative classifier, not generative |
| Prompt sensitivity | Score changes with phrasing of the question | No prompt — it's a classification model |
| Infrastructure dependency | Requires internet + API key in CI | Self-contained, runs offline |

---

### THE CIRCULAR REASONING PROBLEM (why it's a real issue)

When an LLM judges another LLM's output, both models share the same training distribution. A confidently stated wrong answer that sounds fluent will often receive a high faithfulness score from GPT-4 — because GPT-4 was trained to produce fluent text and recognises fluency as quality. DeBERTa doesn't care about fluency — it asks one question: does the context logically support the answer?

This is why `cross-encoder/nli-deberta-v3-large` is the correct architecture for faithfulness, not GPT-4.

---

### THE RELEVANCE SCORER

`relevance.py` uses `sentence-transformers/all-MiniLM-L6-v2`:
- Encodes question and answer as dense vectors
- Cosine similarity between them = relevance score

This is a **bi-encoder** — question and answer are encoded independently. Fast, but less precise than a cross-encoder. Appropriate for relevance (semantic similarity) where cross-attention isn't critical.

---

### MISTAKE AUTOPSY

**MISTAKE: Using LLM-as-judge in CI gates**

```
SYMPTOM:  CI passes on Monday, fails on Tuesday with identical code
CAUSE:    LLM judge is non-deterministic; API response varies
CURE:     Replace with deterministic scorer (NLI or cosine similarity)
DETECTOR: Eval scores vary by > 0.05 between identical runs
```

**MISTAKE: Treating NLI score as "correctness"**

```
SYMPTOM:  System claims answer is faithful when it's factually wrong
CAUSE:    NLI only checks if context SUPPORTS the answer, not if context is correct
CURE:     Faithfulness ≠ correctness. Faithfulness = "did the model hallucinate?"
DETECTOR: Answer is faithful to context but context itself is outdated/wrong
```

---

### SELF-TEST

1. Your faithfulness score drops from 0.82 to 0.61 after a retriever update. What are two possible causes?
2. Why would a bi-encoder be worse than a cross-encoder for faithfulness but acceptable for relevance?
3. The CLI exits with code 1 in CI. What does the engineer need to look at first?

---

## 6. GitHub Actions — Job Order & Why

### THE 4 JOBS

```
lint ──► test ──► eval-gate ──► build-push
```

Each `needs:` keyword creates a dependency edge. Jobs don't run in parallel unless `needs:` is absent or they share the same dependency.

---

### WHY THIS ORDER

**lint** runs first because:
- It's the cheapest job (< 30 seconds)
- It catches the broadest class of errors: syntax issues, import errors, formatting violations
- Failing fast here saves the 5-minute test job from running on broken code
- `ruff check` + `black --check` — no test infra needed, just Python installed

**test** runs after lint because:
- Only valid, formatted code should be tested
- Runs `pytest services/forecast-api/tests/` — needs a model fixture but not the full stack
- Validates that the API logic is correct independently of ML quality

**eval-gate** runs after test because:
- Only code that passes unit tests should be checked for ML quality
- This is the ML-specific gate: faithfulness and relevance must be above 0.7
- It installs the LLM eval package (`pip install -e services/llm-eval/`) and scores a fixture dataset
- Exit code 1 from `mlops-eval` causes this job to fail, blocking build-push

**build-push** runs last and only on `main` because:
- Only code that's linted, tested, and eval-gated gets containerised
- Only the `main` branch gets pushed to GHCR — PRs build but don't push
- This is the deployment gate — no broken or low-quality models go to the registry

---

### THE FAIL-FAST PRINCIPLE

```
lint:    30s   — catches 60% of errors
test:    3min  — catches 30% of remaining errors
eval:    2min  — catches ML quality regressions
build:   5min  — only runs if everything above passes
```

Total worst-case time: ~10 minutes. But if lint fails, you save 10 minutes. If test fails after lint passes, you save 7 minutes. Each gate is ordered by (speed / error detection rate).

---

### WHAT HAPPENS ON A PULL REQUEST

```yaml
on:
  push:
    branches: [main]
  pull_request:  # triggers on any PR, any branch
```

On a PR: lint → test → eval-gate all run. `build-push` is skipped because of:

```yaml
if: github.ref == 'refs/heads/main'
```

This means PRs get full quality checking but don't produce container images. Only merges to `main` produce deployable artifacts.

---

### THE EVAL-GATE AS A CI CONCEPT

The eval-gate is MLOps-specific and not standard in software CI. Here's why it exists:

In traditional software, if tests pass, you ship. In ML systems, a code change can:
- Not break any unit tests
- But subtly change the model's inference path
- Which degrades real-world output quality

The eval-gate catches this. It runs against a fixed fixture dataset (`tests/fixtures/eval_set.jsonl`) — a curated set of question/context/answer triples where you know what good performance looks like. If your code change makes the evaluator score drop below 0.7, CI fails.

---

### MISTAKE AUTOPSY

**MISTAKE: Running build-push on every PR**

```
SYMPTOM:  Registry fills with images for unmerged branches; secrets exposed on forks
CAUSE:    Missing `if: github.ref == 'refs/heads/main'`
CURE:     Always gate image pushes on branch condition
DETECTOR: GHCR shows images tagged with branch names, not just `latest`
```

**MISTAKE: Running eval-gate before unit tests**

```
SYMPTOM:  Spending 2 minutes on eval inference only to fail on a broken import
CAUSE:    Wrong job ordering
CURE:     test must pass before eval runs — eval is more expensive
DETECTOR: Job run time analysis shows eval running on broken code
```

---

### SELF-TEST

1. A contributor submits a PR that passes lint and test but fails eval-gate with faithfulness 0.63. What does the contributor need to fix?
2. Why is `build-push` gated on `github.ref == 'refs/heads/main'` instead of a branch protection rule?
3. If `eval-gate` takes 2 minutes and `lint` takes 30 seconds, would you ever run them in parallel? What's the trade-off?

---

## 7. Docker Compose — Networking & Failure Behaviour

### THE INTUITIVE EXPLANATION

**Big picture**: Three containers (forecast-api, prometheus, grafana) need to talk to each other, but they're isolated processes. Docker Compose creates a private virtual network and assigns each service a DNS hostname equal to its service name.

**Analogy**: Compose networking is like a private office LAN. Each computer gets a name (forecast-api, prometheus, grafana). Any machine on the LAN can reach any other by name. The outside world (your laptop browser) reaches in only through the published port mappings.

---

### HOW THE NETWORK IS BUILT

When you run `docker compose up`, Docker automatically:
1. Creates a bridge network named `<project>_default`
2. Attaches all services to this network
3. Registers each service's name as a DNS entry in the internal resolver
4. Applies port mappings from `ports:` to expose services externally

```
┌─────────────────────────────────────────────────────┐
│  Docker bridge network: mlops-loop_default           │
│                                                       │
│  forecast-api:8000  ←─── prometheus:9090 (scraping) │
│                           ↑                           │
│                      grafana:3000 (querying)          │
└─────────────────────────────────────────────────────┘
        │                    │             │
    port 8000            port 9090     port 3000
        │                    │             │
   your browser         your browser  your browser
```

Internal communication (between containers):
- Prometheus scrapes `http://forecast-api:8000/metrics` — not `localhost:8000`
- Grafana queries `http://prometheus:9090` — not `localhost:9090`

External communication (browser → container):
- `http://localhost:8000/docs` → port-mapped to `forecast-api:8000`

This separation is critical: services use internal names; external clients use `localhost`.

---

### DEPENDS_ON — WHAT IT DOES AND DOESN'T DO

```yaml
grafana:
  depends_on: [prometheus]
```

`depends_on` controls **startup order**: Docker starts prometheus before grafana.

**What it does NOT do**: wait for prometheus to be *ready* (healthy, accepting connections). It just waits for the container to start, not for the service inside to be running.

For true health-aware startup:
```yaml
depends_on:
  prometheus:
    condition: service_healthy
```

This requires a `healthcheck:` block in the prometheus service definition. Without it, grafana might start before prometheus is accepting connections and fail to connect — then reconnect once prometheus is ready (most services handle this gracefully with retry logic).

---

### WHAT HAPPENS WHEN A SERVICE FAILS

**Scenario 1: forecast-api crashes**
- Prometheus continues running, scraping fails with connection refused
- Prometheus marks the target as DOWN (visible in Prometheus UI)
- Grafana shows gaps in the time-series panels
- Your browser gets 502 on `localhost:8000`
- `docker compose` does NOT automatically restart it — you need `restart: on-failure` or `restart: always` in the compose file

**Scenario 2: Prometheus crashes**
- forecast-api continues serving predictions normally (it doesn't depend on Prometheus)
- Grafana shows "No data" on all panels
- Metrics are lost for the duration of the outage (no buffer — Prometheus scrapes are not queued)

**Scenario 3: Grafana crashes**
- forecast-api and Prometheus keep running normally
- You lose the dashboard UI only

**Key insight**: This architecture has no single point of failure for the core prediction path. `forecast-api` → `prometheus` → `grafana` is a monitoring chain, not a serving chain. The forecast API serves predictions independently of whether monitoring is running.

---

### VOLUMES — WHAT THEY DO HERE

```yaml
forecast-api:
  volumes:
    - ./monitoring/evidently/reference_data.parquet:/app/reference_data.parquet
```

This bind mounts a file from your host into the container. The container at `/app/reference_data.parquet` sees the same file as your host at `./monitoring/evidently/reference_data.parquet`. Changes on the host are immediately visible inside the container — no rebuild needed.

For Prometheus and Grafana, volumes mount configuration files. This is the standard pattern: **config lives on host, container reads it at startup**.

---

### MISTAKE AUTOPSY

**MISTAKE: Using `localhost` inside container-to-container communication**

```
SYMPTOM:  prometheus.yml has `targets: ["localhost:8000"]` — scraping fails
CAUSE:    Inside the prometheus container, localhost = prometheus itself, not forecast-api
CURE:     Use service name: `targets: ["forecast-api:8000"]`
DETECTOR: Prometheus UI shows target DOWN despite forecast-api running
```

**MISTAKE: Assuming depends_on means "ready"**

```
SYMPTOM:  Grafana starts, immediately fails to connect to Prometheus, crashes on startup
CAUSE:    depends_on only waits for container start, not service readiness
CURE:     Add healthcheck + condition: service_healthy, or add retry logic in grafana config
DETECTOR: Grafana logs show "connection refused" on first startup, works after manual restart
```

---

### SELF-TEST

1. You add a fourth service, `alertmanager`, that should only start after Prometheus is healthy. Write the `depends_on` block.
2. Prometheus is configured to scrape every 15 seconds. `forecast-api` crashes for 2 minutes. How many data points are missing in Grafana?
3. A developer changes `reference_data.parquet` on the host while `forecast-api` is running. Does the container see the change immediately? Why?

---

## Master Checklist — Full System

Before any interview or exam on this repo, you should be able to:

```
Model Selection
□ Explain why LightGBM, not ARIMA or LSTM, for this specific dataset
□ State the exact conditions where each model would beat LightGBM
□ Explain how lag features replace sequential processing

API Serving
□ Walk through the full request lifecycle from TCP bytes to JSON response
□ Explain what lifespan context manager does and why it matters for latency
□ Distinguish CPU-bound vs I/O-bound async handling

Drift Detection
□ Derive PSI from first principles given bin fractions
□ Explain why PSI is symmetric and KL divergence is not
□ State the three threshold values and what action each triggers

Evidently + Grafana
□ Name the statistical test Evidently uses for numerical features by default
□ Explain why reference_data.parquet must never auto-update
□ Describe what each Grafana panel type represents

LLM Evaluation
□ Explain entailment in plain English
□ Give three concrete reasons DeBERTa beats LLM-as-judge for CI
□ Distinguish faithfulness from correctness

CI/CD
□ Explain the fail-fast rationale for job ordering
□ State exactly when build-push is skipped
□ Explain what the eval-gate actually runs and what it tests

Infrastructure
□ Explain why `localhost` breaks inter-container communication
□ Describe what happens to Grafana panels when Prometheus crashes
□ State the difference between depends_on and depends_on with service_healthy
```

---

*Created: 2026-06-06 | Project: mlops-loop | Author: Anh Duc Tran*
