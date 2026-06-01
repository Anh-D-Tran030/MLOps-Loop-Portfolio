"""
PSI drift detection script.

Usage:
    python drift_report.py --current <current_window.parquet>

Loads reference_data.parquet from the same directory, computes PSI per feature
using evidently.legacy (evidently>=0.7.x moved the classic API to evidently.legacy),
prints PSI scores as JSON to stdout, and writes Prometheus textfile metrics to
/tmp/drift_metrics.prom.

Evidently note: v0.7.x moved Report/DataDriftPreset to evidently.legacy.*
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(SCRIPT_DIR, "reference_data.parquet")
PROM_OUTPUT = "/tmp/drift_metrics.prom"


# ---------------------------------------------------------------------------
# PSI extraction helpers
# ---------------------------------------------------------------------------

def _psi_via_evidently(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute PSI per feature using evidently.legacy DataDriftPreset (stattest='psi').

    Works with evidently>=0.4 (legacy API). Extracts drift_score from
    metrics[1]['result']['drift_by_columns'][col]['drift_score'].
    """
    from evidently.legacy.report import Report  # type: ignore[import]
    from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore[import]

    # Evidently requires numeric or categorical columns; align dtypes.
    cols = [c for c in reference_df.columns if c in current_df.columns]
    ref = reference_df[cols].copy()
    cur = current_df[cols].copy()

    # Cast object columns to 'category' for stable PSI calculation.
    for col in cols:
        if ref[col].dtype == object:
            ref[col] = ref[col].astype("category")
            cur[col] = cur[col].astype("category")

    report = Report(metrics=[DataDriftPreset(stattest="psi")])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    # DataDriftTable is always metrics[1] in evidently 0.4–0.7 legacy path.
    for metric in result.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            drift_by_cols = metric["result"].get("drift_by_columns", {})
            psi_scores: dict[str, float] = {}
            for col_name, col_result in drift_by_cols.items():
                raw_score = col_result.get("drift_score", 0.0)
                # drift_score may be numpy scalar — coerce to Python float.
                psi_scores[col_name] = float(raw_score)
            return psi_scores

    raise RuntimeError(
        "Could not find DataDriftTable in evidently result. "
        "result keys: " + str([m.get("metric") for m in result.get("metrics", [])])
    )


def _psi_hand_rolled(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Hand-rolled PSI fallback — used when evidently import fails.

    PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)) over bins.
    Numeric features: 10 quantile bins from reference distribution.
    Categorical features: per-category frequencies.
    """
    import numpy as np

    psi_scores: dict[str, float] = {}
    cols = [c for c in reference_df.columns if c in current_df.columns]

    for col in cols:
        ref_col = reference_df[col].dropna()
        cur_col = current_df[col].dropna()

        if ref_col.dtype == object or str(ref_col.dtype) == "category":
            # Categorical: use category labels as bins.
            categories = ref_col.unique().tolist()
            ref_counts = ref_col.value_counts(normalize=True).reindex(
                categories, fill_value=0.0
            )
            cur_counts = cur_col.value_counts(normalize=True).reindex(
                categories, fill_value=0.0
            )
            ref_pct = ref_counts.values + epsilon
            cur_pct = cur_counts.values + epsilon
        else:
            # Numeric: quantile bins from reference.
            quantiles = np.quantile(ref_col, np.linspace(0, 1, n_bins + 1))
            quantiles = np.unique(quantiles)
            if len(quantiles) < 2:
                psi_scores[col] = 0.0
                continue
            ref_hist, _ = np.histogram(ref_col, bins=quantiles)
            cur_hist, _ = np.histogram(cur_col, bins=quantiles)
            ref_pct = (ref_hist / ref_hist.sum()) + epsilon
            cur_pct = (cur_hist / cur_hist.sum()) + epsilon

        psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        psi_scores[col] = round(psi_val, 6)

    return psi_scores


def compute_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, float]:
    """Dispatch to evidently or hand-rolled PSI depending on availability."""
    try:
        return _psi_via_evidently(reference_df, current_df)
    except ImportError:
        logger.warning(
            "evidently.legacy not available — falling back to hand-rolled PSI."
        )
        return _psi_hand_rolled(reference_df, current_df)


# ---------------------------------------------------------------------------
# Prometheus textfile writer
# ---------------------------------------------------------------------------

def write_prometheus_metrics(psi_scores: dict[str, float], output_path: str) -> None:
    """Write PSI scores as Prometheus textfile metrics."""
    lines = [
        "# HELP psi_score Population Stability Index per feature",
        "# TYPE psi_score gauge",
    ]
    for feature, score in sorted(psi_scores.items()):
        lines.append(f'psi_score{{feature="{feature}"}} {score:.6f}')

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Prometheus metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSI drift scores (reference vs current window)."
    )
    parser.add_argument(
        "--current",
        required=True,
        metavar="PATH",
        help="Path to current window data (.parquet or .csv).",
    )
    parser.add_argument(
        "--reference",
        default=REFERENCE_PATH,
        metavar="PATH",
        help="Path to reference data (.parquet). Defaults to reference_data.parquet.",
    )
    parser.add_argument(
        "--prom-output",
        default=PROM_OUTPUT,
        metavar="PATH",
        help="Path for Prometheus textfile output.",
    )
    return parser.parse_args(argv)


def load_dataframe(path: str) -> pd.DataFrame:
    """Load parquet or CSV into a DataFrame."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.reference):
        logger.error("Reference data not found: %s", args.reference)
        return 1

    if not os.path.exists(args.current):
        logger.error("Current data not found: %s", args.current)
        return 1

    reference_df = load_dataframe(args.reference)
    current_df = load_dataframe(args.current)

    logger.info(
        "Reference: %d rows | Current: %d rows", len(reference_df), len(current_df)
    )

    psi_scores = compute_psi(reference_df, current_df)

    # Print JSON to stdout (primary output for downstream consumers).
    print(json.dumps(psi_scores, indent=2))

    # Write Prometheus textfile metrics.
    write_prometheus_metrics(psi_scores, args.prom_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
