"""Produce a synthetic "current window" Parquet for drift checks.

CLI:
    python scripts/simulate_current_window.py --out /tmp/current.parquet
    python scripts/simulate_current_window.py --out /tmp/current.parquet --shift

Without --shift: draws a random sample of data/train.csv — PSI should stay low.
With    --shift: scales onpromotion frequency and family distribution — PSI crosses 0.2.

Output schema matches monitoring/evidently/reference_data.parquet:
    store_nbr  (int64)
    family     (object)
    onpromotion (int64)
    day_of_week (int64)
    month      (int64)
    year       (int64)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

RANDOM_SEED = 42
SAMPLE_SIZE = 5_000
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "train.csv",
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.join(os.path.dirname(_THIS_DIR), "services", "forecast-api")
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from training.features import extract_date_features


def build_window(apply_shift: bool) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = extract_date_features(df)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(df), size=min(SAMPLE_SIZE, len(df)), replace=False)
    sample = df.iloc[idx].copy().reset_index(drop=True)

    if apply_shift:
        # Inflate onpromotion frequency: flip ~60% of zeros to ones
        zero_mask = sample["onpromotion"] == 0
        flip_idx = rng.choice(
            zero_mask.sum(), size=int(zero_mask.sum() * 0.6), replace=False
        )
        sample.loc[sample.index[zero_mask][flip_idx], "onpromotion"] = 1

        # Skew family distribution toward a single dominant category
        dominant = "GROCERY I"
        swap_mask = sample["family"] != dominant
        swap_idx = rng.choice(
            swap_mask.sum(), size=int(swap_mask.sum() * 0.7), replace=False
        )
        sample.loc[sample.index[swap_mask][swap_idx], "family"] = dominant

    keep = ["store_nbr", "family", "onpromotion", "day_of_week", "month", "year"]
    return sample[keep].astype(
        {
            "store_nbr": "int64",
            "onpromotion": "int64",
            "day_of_week": "int64",
            "month": "int64",
            "year": "int64",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic current window for drift checks"
    )
    parser.add_argument("--out", required=True, help="Output Parquet path")
    parser.add_argument(
        "--shift",
        action="store_true",
        help="Apply distribution shift so PSI exceeds 0.2 (triggers retrain)",
    )
    args = parser.parse_args()

    window = build_window(apply_shift=args.shift)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    window.to_parquet(args.out, index=False)
    print(f"Written {len(window):,} rows to {args.out} (shift={args.shift})")


if __name__ == "__main__":
    main()
