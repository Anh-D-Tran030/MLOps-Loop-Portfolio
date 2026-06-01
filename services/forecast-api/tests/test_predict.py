"""pytest test suite for the Forecast API."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.conftest import VALID_PREDICT_PAYLOAD


def test_health_returns_ok(test_client: TestClient) -> None:
    """GET /health should return 200 with status=ok and model_loaded=true."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(test_client: TestClient) -> None:
    """POST /predict with a valid payload returns forecast of the correct length."""
    resp = test_client.post("/predict", json=VALID_PREDICT_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert "forecast" in body
    assert len(body["forecast"]) == VALID_PREDICT_PAYLOAD["days_ahead"]

    # prediction_id must be a valid uuid4
    try:
        parsed = uuid.UUID(body["prediction_id"], version=4)
        assert str(parsed) == body["prediction_id"]
    except (ValueError, KeyError) as exc:
        raise AssertionError(f"prediction_id is not a valid uuid4: {body!r}") from exc


def test_predict_missing_feature_does_not_crash(test_client: TestClient) -> None:
    """POST /predict without optional features should still return 200 with defaults."""
    minimal_payload = {
        "store_nbr": 2,
        "family": "BEVERAGES",
        "onpromotion": 1,
        # days_ahead deliberately omitted — should default to 30
    }
    resp = test_client.post("/predict", json=minimal_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["forecast"]) == 30  # default days_ahead


def test_predict_invalid_store_nbr_type(test_client: TestClient) -> None:
    """POST /predict with wrong type for store_nbr should return 422."""
    invalid_payload = {
        "store_nbr": "not-an-int",
        "family": "GROCERY I",
        "onpromotion": 0,
        "days_ahead": 7,
    }
    resp = test_client.post("/predict", json=invalid_payload)
    assert resp.status_code == 422


def test_metrics_endpoint(test_client: TestClient) -> None:
    """GET /metrics should return 200 and contain Prometheus metric names."""
    # Trigger a predict first so counters are populated
    test_client.post("/predict", json=VALID_PREDICT_PAYLOAD)

    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "forecast_requests_total" in body


def test_predict_days_ahead_default(test_client: TestClient) -> None:
    """POST /predict omitting days_ahead should produce a forecast of length 30."""
    payload = {
        "store_nbr": 1,
        "family": "PRODUCE",
        "onpromotion": 0,
        # days_ahead omitted — Pydantic default = 30
    }
    resp = test_client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["forecast"]) == 30
