"""Open a GitHub Issue comparing a candidate Staging model vs. current Production.

CLI: python scripts/open_promotion_issue.py --max-psi <float>

Reads GITHUB_TOKEN from env. Uses the GitHub REST API (no new dependency).
"""

from __future__ import annotations

import argparse
import os
import sys

import requests


def get_model_info(client, model_name: str, stage: str) -> dict | None:
    """Return {version, run_id, val_rmse} for the latest model at the given stage."""
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    v = versions[0]
    run = client.get_run(v.run_id)
    val_rmse = run.data.metrics.get("val_rmse")
    return {"version": v.version, "run_id": v.run_id, "val_rmse": val_rmse}


def open_issue(max_psi: float) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    client = MlflowClient()
    model_name = "mlops-loop-forecast"

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)

    staging = get_model_info(client, model_name, "Staging")
    if not staging:
        print(
            "ERROR: No Staging version found — retrain must have failed",
            file=sys.stderr,
        )
        sys.exit(1)

    production = get_model_info(client, model_name, "Production")

    prod_block = (
        f"| **Current Production version** | {production['version']} |\n"
        f"| Production `val_rmse` | `{production['val_rmse']:.4f}` |"
        if production
        else "| **Current Production version** | *(none — first promotion)* |"
    )

    body = f"""## Candidate Model Ready for Review

| Field | Value |
|---|---|
| **Candidate version** | {staging['version']} |
| **MLflow run ID** | `{staging['run_id']}` |
| Candidate `val_rmse` | `{staging['val_rmse']:.4f}` |
| Max PSI (trigger) | `{max_psi:.4f}` |
{prod_block}

## How to Promote

Run the **Promote Model** workflow manually:

1. Go to **Actions → Promote Model → Run workflow**
2. Enter `model_version`: `{staging['version']}`
3. Click **Run workflow**

This transitions version `{staging['version']}` to `Production`, archives the previous
Production version, creates a release tag, and triggers a new image build on GHCR.

## Review Checklist

- [ ] `val_rmse` is acceptable (lower is better)
- [ ] Drift was due to real distribution shift, not a data pipeline issue
- [ ] MLflow run artifacts look correct at `localhost:5000`
"""

    title = f"Candidate model ready for review — PSI {max_psi:.4f}"

    repo = os.environ.get("GITHUB_REPOSITORY", "Anh-D-Tran030/MLOps-Loop-Portfolio")
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.post(url, headers=headers, json={"title": title, "body": body})
    resp.raise_for_status()
    issue = resp.json()
    print(f"Opened issue #{issue['number']}: {issue['html_url']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a GitHub Issue for a candidate model promotion"
    )
    parser.add_argument(
        "--max-psi",
        required=True,
        type=float,
        help="Max PSI score that triggered the retrain",
    )
    args = parser.parse_args()
    open_issue(args.max_psi)


if __name__ == "__main__":
    main()
