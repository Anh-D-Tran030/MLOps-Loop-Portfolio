"""Shared feature engineering functions.

Single source of truth for both training (train.py) and inference (model.py).
Any divergence between training and serving paths = training/serving skew.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Colombian public holidays (month, day) — a simplified static set for the
# Kaggle Store Sales dataset which covers Ecuador retail stores.
# Days-to-holiday is computed as the minimum days to the next holiday.
_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1),  # New Year
    (5, 1),  # Labour Day
    (7, 4),  # Independence Day (simplified)
    (8, 10),  # Independence Day (Quito)
    (10, 9),  # Independence of Guayaquil
    (11, 2),  # All Souls Day
    (11, 3),  # Independence of Cuenca
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
]


def _days_to_next_holiday(date: pd.Timestamp) -> int:
    """Return number of days from `date` to the nearest upcoming holiday."""
    min_days = 365
    year = date.year
    for month, day in _HOLIDAYS:
        for y in (year, year + 1):
            try:
                h = pd.Timestamp(year=y, month=month, day=day)
                delta = (h - date).days
                if delta >= 0 and delta < min_days:
                    min_days = delta
            except ValueError:
                continue
    return min_days


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based feature columns derived from the 'date' column.

    Adds:
        day_of_week   (0=Monday … 6=Sunday)
        month         (1-12)
        year          (e.g. 2023)
        days_to_holiday (int >= 0)

    The 'date' column must be parseable as a datetime.
    """
    df = df.copy()
    if "date" not in df.columns:
        logger.warning("'date' column not found — skipping date feature extraction")
        df["day_of_week"] = 0
        df["month"] = 1
        df["year"] = 2017
        df["days_to_holiday"] = 0
        return df

    dates = pd.to_datetime(df["date"])
    df["day_of_week"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    df["year"] = dates.dt.year
    df["days_to_holiday"] = dates.apply(_days_to_next_holiday)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag-7 and lag-14 features from the 'sales' column.

    For rows that lack history (NaN), fills with 0.
    The 'sales' column must exist; if not, both lag columns are set to 0.
    """
    df = df.copy()
    if "sales" not in df.columns:
        logger.warning("'sales' column not found — setting lag features to 0")
        df["lag_7"] = 0.0
        df["lag_14"] = 0.0
        return df

    df["lag_7"] = df["sales"].shift(7).fillna(0.0)
    df["lag_14"] = df["sales"].shift(14).fillna(0.0)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 7-day rolling mean of 'sales'.

    Fills NaN (insufficient history) with 0.
    """
    df = df.copy()
    if "sales" not in df.columns:
        logger.warning("'sales' column not found — setting rolling_mean_7 to 0")
        df["rolling_mean_7"] = 0.0
        return df

    df["rolling_mean_7"] = (
        df["sales"].rolling(window=7, min_periods=1).mean().fillna(0.0)
    )
    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = False,
    encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode the 'family' column.

    Args:
        df:      Input DataFrame (must contain 'family').
        fit:     If True, fit a new encoder on `df`. If False, use `encoder`.
        encoder: Pre-fitted OneHotEncoder to use when fit=False.

    Returns:
        (transformed_df, encoder) — transformed_df has OHE columns added
        and the original 'family' column dropped.

    Raises:
        ValueError: if fit=False and encoder is None.
    """
    df = df.copy()

    if "family" not in df.columns:
        logger.warning("'family' column not found — skipping OHE")
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return df, encoder

    family_values = df[["family"]]

    if fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(family_values)
    else:
        if encoder is None:
            raise ValueError("encoder must be provided when fit=False")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoded = encoder.transform(family_values)

    feature_names = encoder.get_feature_names_out(["family"])
    # LightGBM rejects JSON-special chars in feature names; Kaggle families
    # include "BREAD/BAKERY" and "LIQUOR,WINE,BEER". Sanitize once, here.
    feature_names = [
        "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        for name in feature_names
    ]
    ohe_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=["family"])
    df = pd.concat([df, ohe_df], axis=1)
    return df, encoder
