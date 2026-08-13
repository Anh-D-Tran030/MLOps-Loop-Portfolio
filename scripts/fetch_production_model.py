"""Download the Production model artifact from MLflow Registry.

CLI: python scripts/fetch_production_model.py --out services/forecast-api/model.pkl

Exits non-zero with a clear message if no Production version exists yet.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile


def fetch_production_model(out_path: str) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    client = MlflowClient()
    model_name = "mlops-loop-forecast"

    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        print(
            f"ERROR: No Production version found for model '{model_name}'.\n"
            "Run the Bootstrap Sequence first:\n"
            "  1. Trigger retrain-check.yml with force_retrain=true\n"
            "  2. Run promote.yml with the version number from the comparison issue",
            file=sys.stderr,
        )
        sys.exit(1)

    prod_version = versions[0]
    run_id = prod_version.run_id
    print(f"Found Production version {prod_version.version} (run_id={run_id})")

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/model.pkl",
            dst_path=tmp_dir,
        )
        shutil.copy(artifact_path, out_path)

    print(f"Production model.pkl written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the Production model from MLflow Registry"
    )
    parser.add_argument(
        "--out",
        default="services/forecast-api/model.pkl",
        help="Destination path for model.pkl",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fetch_production_model(args.out)


if __name__ == "__main__":
    main()
