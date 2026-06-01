"""pytest fixtures for the forecast-api test suite.

The minimal_model fixture trains a tiny LightGBM on 100 synthetic rows and
writes a temp model.pkl so tests run without touching the real training data
or the internet.
"""

from __future__ import annotations

import os
import sys
import tempfile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Ensure the service root (services/forecast-api) is importable
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.dirname(_TESTS_DIR)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

FAMILIES = ["GROCERY I", "BEVERAGES", "PRODUCE"]
RANDOM_SEED = 42


@pytest.fixture(scope="session")
def minimal_model(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a tiny LightGBM on 100 synthetic rows; return path to model.pkl."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 100

    families = rng.choice(FAMILIES, size=n)
    X_raw = pd.DataFrame(
        {
            "store_nbr": rng.integers(1, 5, size=n).astype(float),
            "onpromotion": rng.integers(0, 2, size=n).astype(float),
            "day_of_week": rng.integers(0, 7, size=n).astype(float),
            "month": rng.integers(1, 13, size=n).astype(float),
            "year": np.full(n, 2017, dtype=float),
            "days_to_holiday": rng.integers(0, 30, size=n).astype(float),
            "lag_7": rng.uniform(0, 100, size=n),
            "lag_14": rng.uniform(0, 100, size=n),
            "rolling_mean_7": rng.uniform(0, 100, size=n),
            "family": families,
        }
    )
    y = rng.uniform(0, 200, size=n)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    family_ohe = encoder.fit_transform(X_raw[["family"]])
    ohe_cols = encoder.get_feature_names_out(["family"])
    ohe_df = pd.DataFrame(family_ohe, columns=ohe_cols)
    X = pd.concat([X_raw.drop(columns=["family"]).reset_index(drop=True), ohe_df], axis=1)

    model = lgb.LGBMRegressor(
        n_estimators=10,
        num_leaves=4,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X, y)

    tmp_dir = tmp_path_factory.mktemp("model")
    model_path = str(tmp_dir / "model.pkl")
    joblib.dump({"model": model, "encoder": encoder}, model_path)
    return model_path


@pytest.fixture(scope="session")
def test_client(minimal_model: str) -> TestClient:
    """Return a FastAPI TestClient with the minimal fixture model loaded."""
    # Point the LGBMModel loader at the temp model.pkl before importing app
    os.environ["MODEL_PKL_PATH"] = minimal_model

    from app.model import LGBMModel
    from app.main import app

    # Override the model on app state directly after lifespan runs via TestClient
    with TestClient(app) as client:
        # Replace the model loaded by lifespan with the fixture model
        client.app.state.model = LGBMModel(model_path=minimal_model)
        yield client


VALID_PREDICT_PAYLOAD = {
    "store_nbr": 1,
    "family": "GROCERY I",
    "onpromotion": 0,
    "days_ahead": 7,
}
