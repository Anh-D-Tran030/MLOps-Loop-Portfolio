"""FastAPI application for the Forecast API (QW-1).

Endpoints:
    POST /predict  — returns a demand forecast
    GET  /health   — liveness check
    GET  /metrics  — Prometheus metrics
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.model import LGBMModel
from app.schemas import PredictRequest, PredictResponse

# ---------------------------------------------------------------------------
# Prometheus metrics — module-level globals so they are registered exactly once
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "forecast_requests_total",
    "Total number of POST /predict requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "forecast_request_duration_seconds",
    "Latency of POST /predict in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0],
)

PSI_SCORE = Gauge(
    "psi_score",
    "Population Stability Index per feature (updated by drift_report)",
    ["feature"],
)

MODEL_LOADED_GAUGE = Gauge(
    "model_loaded",
    "1 if the LightGBM model is loaded and ready, 0 otherwise",
)


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # type: ignore[override]
    model = LGBMModel()
    app.state.model = model  # type: ignore[attr-defined]
    MODEL_LOADED_GAUGE.set(1 if model.model_loaded else 0)
    yield
    # Shutdown: nothing to clean up for a plain in-process model


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="mlops-loop Forecast API",
    description="LightGBM-based retail demand forecasting — QW-1",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, req: Request) -> PredictResponse:
    """Return a multi-step demand forecast."""
    model: LGBMModel = req.app.state.model  # type: ignore[attr-defined]

    start = time.perf_counter()
    try:
        features = {
            "store_nbr": request.store_nbr,
            "family": request.family,
            "onpromotion": request.onpromotion,
        }
        forecast = model.predict(features, days_ahead=request.days_ahead)
        REQUEST_COUNT.labels(status="success").inc()
        return PredictResponse(
            forecast=forecast,
            model_version=model.model_version,
            prediction_id=str(uuid.uuid4()),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/health")
async def health(req: Request) -> dict:
    """Liveness check — always returns 200."""
    model: LGBMModel = req.app.state.model  # type: ignore[attr-defined]
    return {"status": "ok", "model_loaded": model.model_loaded}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus text-format metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
