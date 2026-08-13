"""Transition an MLflow model version to Production.

CLI: python scripts/promote_model.py --version <n>

Archives whatever version was previously Production before promoting.
"""

from __future__ import annotations

import argparse


def promote(version: int) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    client = MlflowClient()
    model_name = "mlops-loop-forecast"

    # Record the current Production version (if any)
    existing = client.get_latest_versions(model_name, stages=["Production"])
    old_version = existing[0].version if existing else None

    if old_version:
        print(f"Archiving current Production version: {old_version}")

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )

    new_prod = client.get_latest_versions(model_name, stages=["Production"])
    print(f"Promoted version {new_prod[0].version} to Production")
    if old_version:
        print(f"Previous Production version {old_version} is now Archived")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote an MLflow model version to Production"
    )
    parser.add_argument(
        "--version",
        required=True,
        type=int,
        help="MLflow model version number to promote",
    )
    args = parser.parse_args()
    promote(args.version)


if __name__ == "__main__":
    main()
