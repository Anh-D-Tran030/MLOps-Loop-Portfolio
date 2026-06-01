"""LightGBM model loader and inference wrapper.

Loads model.pkl from disk on init. Handles missing file gracefully.
All feature engineering is imported from training.features to prevent
training/serving skew.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve the project root so `training.features` can be imported from
# the services/forecast-api directory regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.dirname(_THIS_DIR)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from training.features import (  # noqa: E402
    add_lag_features,
    add_rolling_features,
    encode_categoricals,
    extract_date_features,
)

MODEL_VERSION = "1.0.0"


class LGBMModel:
    """Wrapper around a persisted LightGBM regressor.

    Attributes:
        model_loaded: True if model.pkl was found and loaded successfully.
        model_version: Semantic version string embedded in this wrapper.
    """

    def __init__(self, model_path: str = "model.pkl") -> None:
        self.model_version = MODEL_VERSION
        self.model_loaded = False
        self._model: Optional[Any] = None
        self._encoder: Optional[Any] = None

        self._load(model_path)

    def _load(self, model_path: str) -> None:
        """Attempt to load (model, encoder) from disk. Never raises."""
        if not os.path.isfile(model_path):
            logger.warning(
                "model.pkl not found at '%s' — inference will be unavailable",
                model_path,
            )
            return

        try:
            import joblib

            payload = joblib.load(model_path)
            if isinstance(payload, dict):
                self._model = payload["model"]
                self._encoder = payload.get("encoder")
            else:
                # Legacy: plain model object
                self._model = payload
                self._encoder = None
            self.model_loaded = True
            logger.info("Model loaded from '%s'", model_path)
        except Exception as exc:
            logger.error("Failed to load model from '%s': %s", model_path, exc)

    def predict(self, features: dict, days_ahead: int = 30) -> list[float]:
        """Generate a forecast of length `days_ahead`.

        Builds a sequence of synthetic future rows from the request features,
        applies the identical feature engineering pipeline used during training,
        and returns point forecasts.

        Args:
            features:   Dict of input feature values (store_nbr, family,
                        onpromotion, plus optional date/lag hints).
            days_ahead: Number of forecast steps to return.

        Returns:
            List of `days_ahead` non-negative float predictions.
        """
        if not self.model_loaded or self._model is None:
            logger.warning("Model not loaded — returning zeros for %d steps", days_ahead)
            return [0.0] * days_ahead

        try:
            rows = self._build_feature_rows(features, days_ahead)
            raw = self._model.predict(rows)
            # Clip to non-negative sales
            return [max(0.0, float(v)) for v in raw]
        except Exception as exc:
            logger.error("Inference error: %s", exc, exc_info=True)
            return [0.0] * days_ahead

    def _build_feature_rows(self, features: dict, days_ahead: int) -> pd.DataFrame:
        """Construct a DataFrame of `days_ahead` rows ready for LightGBM."""
        # --- Defaults for missing keys ---
        store_nbr = self._get(features, "store_nbr", 1)
        onpromotion = self._get(features, "onpromotion", 0)
        family = self._get(features, "family", "GROCERY I")

        base_date_str = features.get("date")
        if base_date_str:
            base_date = pd.Timestamp(base_date_str)
        else:
            base_date = pd.Timestamp.today().normalize()

        dates = [base_date + pd.Timedelta(days=i) for i in range(days_ahead)]

        sales_hint = float(features.get("sales", 0.0))
        # Synthetic history for lag/rolling computation
        history_len = 14
        history = [sales_hint] * history_len

        rows = []
        for i, d in enumerate(dates):
            row = {
                "date": d,
                "store_nbr": store_nbr,
                "family": family,
                "onpromotion": onpromotion,
                # Append a placeholder sales value for lag computation
                "sales": history[-(14 - i)] if i < 14 else float(
                    np.mean(history[-7:]) if history else 0.0
                ),
            }
            rows.append(row)
            history.append(row["sales"])

        df = pd.DataFrame(rows)

        # Apply identical pipeline as training
        df = extract_date_features(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df, _ = encode_categoricals(df, fit=False, encoder=self._encoder)

        # Drop columns not used by the model
        drop_cols = [c for c in ["date", "sales"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # Align columns with model's expected features
        if hasattr(self._model, "feature_name_"):
            expected = self._model.feature_name_
            for col in expected:
                if col not in df.columns:
                    logger.warning("Missing feature '%s' — filling with 0", col)
                    df[col] = 0
            df = df[expected]

        return df

    @staticmethod
    def _get(features: dict, key: str, default: Any) -> Any:
        """Return features[key] if present and non-None, else default with warning."""
        val = features.get(key)
        if val is None:
            logger.warning("Feature '%s' missing — using default %r", key, default)
            return default
        return val
