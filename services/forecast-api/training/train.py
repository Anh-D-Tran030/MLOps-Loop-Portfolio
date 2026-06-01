"""LightGBM training script for retail demand forecasting.

Usage:
    python -m training.train
    # or
    python services/forecast-api/training/train.py

Expects:
    data/train.csv  — Kaggle Store Sales dataset at the repo root.

Outputs:
    model.pkl       — joblib-serialised dict {"model": lgbm, "encoder": ohe}
                      written to the current working directory.
"""

from __future__ import annotations

import logging
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Ensure the service root is on sys.path so features.py is importable.
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "n_estimators": 500,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(_SERVICE_DIR)),  # repo root (3 up from training/)
    "data",
    "train.csv",
)
OUTPUT_MODEL_PATH = "model.pkl"


def load_data(path: str) -> pd.DataFrame:
    """Load the Kaggle Store Sales CSV.

    Expected columns (minimum): date, store_nbr, family, onpromotion, sales
    """
    logger.info("Loading data from '%s'", path)
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "store_nbr", "family", "onpromotion", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Loaded %d rows", len(df))
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Apply the full feature engineering pipeline.

    Returns:
        (feature_df, fitted_encoder) — feature_df has no 'date' or 'sales' cols.
    """
    df = df.sort_values("date").reset_index(drop=True)

    df = extract_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df, encoder = encode_categoricals(df, fit=True)

    # Drop target and date
    drop_cols = [c for c in ["date", "sales"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df, encoder


def train(data_path: str = DEFAULT_DATA_PATH) -> None:
    """Full training pipeline: load → features → split → fit → evaluate → save."""
    # 1. Load
    df_raw = load_data(data_path)

    # 2. Extract target before feature engineering (sales column is consumed)
    target = df_raw["sales"].values

    # 3. Build features
    X, encoder = build_features(df_raw)
    y = target

    logger.info("Feature matrix shape: %s", X.shape)

    # 4. Train / validation split (chronological — not shuffled)
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(
        "Split: %d train rows / %d val rows", len(X_train), len(X_val)
    )

    # 5. Fit LightGBM
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # 6. Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val   RMSE: {val_rmse:.4f}")

    # 7. Persist
    payload = {"model": model, "encoder": encoder}
    joblib.dump(payload, OUTPUT_MODEL_PATH)
    logger.info("Model saved to '%s'", OUTPUT_MODEL_PATH)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM demand forecasting model")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to Kaggle Store Sales train.csv",
    )
    args = parser.parse_args()
    train(data_path=args.data_path)


if __name__ == "__main__":
    main()
